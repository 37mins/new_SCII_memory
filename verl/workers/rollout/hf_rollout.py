# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single
GPU model. Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model
to perform generation.
"""

import contextlib
import os

import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig

from verl import DataProto
from verl.utils.torch_functional import get_response_mask
from verl.utils.device import get_torch_device

from .base import BaseRollout

__all__ = ["HFRollout"]


class HFRollout(BaseRollout):
    def __init__(self, module: nn.Module, config):
        super().__init__()
        self.config = config
        self.module = module
        self.no_thinking = None
        if config.get("no_thinking_tokenizer_path"):
            from verl.utils.sc2_no_thinking import load_constraint
            self.no_thinking = load_constraint(config.no_thinking_tokenizer_path)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // self.config.get("micro_batch_size", batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        # make sampling args can be overriden by inputs
        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        is_validate = prompts.meta_info.get("validate", False)

        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = max(0, prompts.meta_info.get("top_k", self.config.get("top_k", 0)))  # to be compatible with vllm

        if not do_sample:
            # do_sample==False -> greedy decoding
            kwargs = {
                "do_sample": False,
                "num_beams": 1,
            }
        elif is_validate:
            # do validate and do sample -> use val_kwargs
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_k": max(0, self.config.val_kwargs.top_k),  # to be compatible with vllm
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "num_return_sequences": 1,  # if validate, already repeat in ray_trainer
            }
        else:
            # do_sample -> use rollout config
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                "num_return_sequences": self.config.n,
            }

        # make config according to generate mode
        generation_config = GenerationConfig(**kwargs)

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        prompt_length = idx.size(1)
        attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]
        override_eos_token_id = os.getenv("SPB1_ROLLOUT_EOS_TOKEN_ID")
        if override_eos_token_id:
            eos_token_id = int(override_eos_token_id)
            print(f"[SPB1_ROLLOUT_EOS_TOKEN_ID] overriding rollout eos_token_id={eos_token_id}", flush=True)
        if pad_token_id is None:
            pad_token_id = eos_token_id

        self.module.eval()
        param_ctx = contextlib.nullcontext()

        if isinstance(self.module, FSDP):
            # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        model_config = getattr(self.module, "config", None)
        model_type = getattr(model_config, "model_type", None)
        if model_type == "gla_swa" and not getattr(self.module, "_spb1_gla_mask_trim_patch", False):
            original_prepare_inputs_for_generation = self.module.prepare_inputs_for_generation

            def _spb1_prepare_inputs_for_generation(*args, **kwargs):
                model_inputs = original_prepare_inputs_for_generation(*args, **kwargs)
                cur_input_ids = model_inputs.get("input_ids")
                cur_attention_mask = model_inputs.get("attention_mask")
                if (
                    cur_input_ids is not None
                    and cur_attention_mask is not None
                    and cur_attention_mask.shape[-1] > cur_input_ids.shape[-1]
                ):
                    model_inputs["attention_mask"] = cur_attention_mask[:, : cur_input_ids.shape[-1]]
                return model_inputs

            self.module.prepare_inputs_for_generation = _spb1_prepare_inputs_for_generation
            self.module._spb1_gla_mask_trim_patch = True
        # SPB1's GLA-SWA generation code expects its recurrent cache to stay in
        # the model-specific dict form; recent transformers versions convert it
        # to legacy tuple cache during generate, which breaks gla_attention.
        use_cache = model_type != "gla_swa"
        generate_kwargs = {
            "input_ids": idx,
            "attention_mask": attention_mask,
            "do_sample": do_sample,
            "max_new_tokens": response_length,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "generation_config": generation_config,
            "output_scores": False,  # this is potentially very large
            "return_dict_in_generate": True,
            "use_cache": use_cache,
        }
        if self.config.get("capture_log_probs", False):
            generate_kwargs.update(output_logits=True, output_scores=True)
        if getattr(self, "no_thinking", None) is not None:
            from transformers import LogitsProcessorList
            from verl.utils.sc2_no_thinking import NoThinkingLogitsProcessor
            generate_kwargs["logits_processor"] = LogitsProcessorList([
                NoThinkingLogitsProcessor(self.no_thinking, prompt_length)
            ])
        # Qwen3 derives advancing decode positions from the growing attention mask.
        if model_type not in {"qwen3", "sse_swa_moba", "gla_swa"}:
            generate_kwargs["position_ids"] = position_ids

        with param_ctx, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = self.module.generate(**generate_kwargs)

        # TODO: filter out the seq with no answers like ds-chat
        seq = output.sequences
        generated_batch_size = seq.size(0)  # bs * num_return_sequences

        # huggingface generate will stop generating when all the batch reaches [EOS].
        # We have to pad to response_length
        sequence_length = prompt_length + self.config.response_length
        delta_length = sequence_length - seq.shape[1]

        if delta_length > 0:
            delta_tokens = torch.ones(size=(generated_batch_size, delta_length), device=seq.device, dtype=seq.dtype)
            delta_tokens = pad_token_id * delta_tokens
            seq = torch.cat((seq, delta_tokens), dim=1)
        assert seq.shape[1] == sequence_length

        # make necessary reputations if num_return_sequences > 1
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        if num_return_sequences > 1:
            position_ids = position_ids.repeat_interleave(num_return_sequences, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)

        prompt = seq[:, :prompt_length]  # (generated_batch_size, prompt_length)
        response = seq[:, prompt_length:]  # (generated_batch_size, response_length)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(generated_batch_size, 1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=generated_batch_size,
        )
        if self.config.get("capture_log_probs", False):
            from .hf_diagnostics import selected_log_probs
            unwarped, sampling = selected_log_probs(
                output, response, response_attention_mask, generation_config.temperature)
            batch["rollout_unwarped_log_probs"] = unwarped
            batch["rollout_sampling_log_probs"] = sampling

        # empty cache before compute old_log_prob
        get_torch_device().empty_cache()

        self.module.train()
        return DataProto(batch=batch)
