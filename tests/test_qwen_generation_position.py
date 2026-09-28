"""Qwen-only cached-generation regression; no model download or SC2 launch."""

import ast
import contextlib
import hashlib
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
ROLLOUT = ROOT / "verl/workers/rollout/hf_rollout.py"
OLD = 'if model_type not in {"sse_swa_moba", "gla_swa"}:'
NEW = 'if model_type not in {"qwen3", "sse_swa_moba", "gla_swa"}:'
COMMENT = '        # Qwen3 derives advancing decode positions from the growing attention mask.\n'


class QwenPositionTest(unittest.TestCase):
    def test_only_qwen_generate_argument_changes(self):
        source = ROLLOUT.read_text()
        # Strip the two explicit opt-in observation blocks, then retain the
        # original byte freeze. Separate sampling tests cover their behavior.
        for block in (
            '        if self.config.get("capture_log_probs", False):\n'
            '            generate_kwargs.update(output_logits=True, output_scores=True)\n',
            '        if self.config.get("capture_log_probs", False):\n'
            '            from .hf_diagnostics import selected_log_probs\n'
            '            unwarped, sampling = selected_log_probs(\n'
            '                output, response, response_attention_mask, generation_config.temperature)\n'
            '            batch["rollout_unwarped_log_probs"] = unwarped\n'
            '            batch["rollout_sampling_log_probs"] = sampling\n',
        ):
            self.assertEqual(source.count(block), 1)
            source = source.replace(block, "")
        self.assertEqual(source.count(NEW), 1)
        original = source.replace(NEW, OLD).replace(COMMENT, "")
        self.assertEqual(hashlib.sha256(original.encode()).hexdigest(),
                         "651a939ac2f71960eb40a6fe72b2f96f5a7974b855fa6370ab47b757255b8daf")
        # Actor changes have their own freeze and real-update regression tests.

    def test_other_model_position_argument_paths_are_unchanged(self):
        node = next(n for n in ast.walk(ast.parse(ROLLOUT.read_text()))
                    if isinstance(n, ast.If) and ast.unparse(n.test) ==
                    "model_type not in {'qwen3', 'sse_swa_moba', 'gla_swa'}")
        for model_type, supplied in (("qwen3", False), ("sse_swa_moba", False),
                                     ("gla_swa", False), ("llama", True)):
            scope = dict(model_type=model_type, generate_kwargs={}, position_ids="sentinel")
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROLLOUT), "exec"), scope)
            self.assertEqual("position_ids" in scope["generate_kwargs"], supplied)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
                         "requires pinned Linux Torch/Transformers")
    def test_real_hf_rollout_advances_positions_and_matches_full_forward(self):
        import torch
        from omegaconf import OmegaConf
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from verl import DataProto
        from verl.utils.model import compute_position_id_with_mask
        from verl.workers.rollout.hf_rollout import HFRollout

        torch.set_num_threads(1)
        torch.manual_seed(7)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, attention_dropout=0.0, bos_token_id=1, eos_token_id=63,
            pad_token_id=0, max_position_embeddings=128)
        config._attn_implementation = "eager"
        model = Qwen3ForCausalLM(config).eval()
        ids = torch.tensor([[0, 0, 3, 4, 5, 6], [0, 3, 4, 5, 6, 7]])
        mask = (ids != 0).long()
        positions = compute_position_id_with_mask(mask)
        prompts = DataProto.from_dict(tensors=dict(input_ids=ids, attention_mask=mask,
            position_ids=positions), meta_info=dict(eos_token_id=63, pad_token_id=0))
        rollout = HFRollout(model, OmegaConf.create(dict(do_sample=False, temperature=1.0,
                                                       response_length=8, n=1)))
        traces, logits = [], []

        def before(_module, _args, kwargs):
            traces.append(kwargs["position_ids"].detach().clone())

        def after(_module, _args, output):
            logits.append(output.logits[:, -1].detach().clone())

        hooks = [model.register_forward_pre_hook(before, with_kwargs=True),
                 model.register_forward_hook(after)]
        # CPU harness: no CUDA autocast or device-cache cleanup; model math is real.
        with patch("verl.workers.rollout.hf_rollout.torch.autocast", return_value=contextlib.nullcontext()), \
                patch("verl.workers.rollout.hf_rollout.get_torch_device"):
            output = rollout.generate_sequences(prompts)
        for hook in hooks:
            hook.remove()
        self.assertGreater(len(traces), 1)
        for step, trace in enumerate(traces[1:], 1):
            self.assertTrue(torch.equal(trace, positions[:, -1:] + step))
        self.assertTrue(torch.equal(output.batch["position_ids"][:, :6], positions))
        model.eval()
        full_ids = output.batch["input_ids"]
        full_mask = torch.cat([mask, torch.ones_like(output.batch["responses"])], dim=1)
        with torch.no_grad():
            full_logits = model(input_ids=full_ids, attention_mask=full_mask,
                position_ids=compute_position_id_with_mask(full_mask), use_cache=False).logits
        for step, cached_logits in enumerate(logits):
            torch.testing.assert_close(cached_logits, full_logits[:, 5 + step], atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
