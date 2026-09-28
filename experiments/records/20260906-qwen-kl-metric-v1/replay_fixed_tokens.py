"""Four-rank, bounded r7 numerical replay; no SC2 or production behavior edits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from transformers import GenerationConfig, LogitsProcessor

from verl import DataProto
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


class FixedTokens(LogitsProcessor):
    """Force saved tokens, leaving generate's raw logits intact for observation."""
    def __init__(self, response, prompt_length):
        self.response = response
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores):
        index = input_ids.shape[1] - self.prompt_length
        forced = torch.full_like(scores, -torch.inf)
        return forced.scatter(1, self.response[:, index:index + 1], 0.)


def selected(logits, labels, temperature):
    return (logits.float() / temperature).log_softmax(-1).gather(
        -1, labels.unsqueeze(-1)).squeeze(-1)


def gap(left, right, mask):
    diff = (left.float() - right.float())[mask]
    return dict(tokens=diff.numel(), finite=bool(diff.isfinite().all()),
                abs_mean=diff.abs().mean().item(), abs_max=diff.abs().max().item())


def rebased_tensors(tensors, starting_logp):
    """Original actor is the reference here; retain tokens/A, never edit r7 tensors."""
    return {**tensors, "old_log_probs": starting_logp.detach().cpu().clone(),
            "ref_log_prob": starting_logp.detach().cpu().clone()}


def fp32_head_logp(head, hidden, labels, temperature):
    """Shadow calculation only; neither hidden states nor the actual head output is changed."""
    with torch.autocast(hidden.device.type, enabled=False):
        bias = head.bias.float() if head.bias is not None else None
        logits = torch.nn.functional.linear(hidden.float(), head.weight.float(), bias)
        return selected(logits, labels, temperature)


def prediction_states(value, path, response_length, valid_length):
    """Align states predicting the response, excluding padding and post-EOS work."""
    value = value[0] if isinstance(value, tuple) else value
    if path == "hf_full_fp32_normalization":
        value = value[:, -response_length - 1:-1][:, :valid_length]
    else:
        value = value[:, -1:]
    return value.detach().float().cpu().flatten(2)


def layer_gaps(trace):
    rows = []
    for name, full in trace["full"].items():
        cached = torch.cat(trace["cached"][name], dim=1)
        if cached.shape != full.shape:
            raise ValueError(f"Layer trace alignment failed for {name}: {full.shape} vs {cached.shape}")
        delta = cached - full
        rows.append(dict(layer=name, shape=list(full.shape), finite=bool(delta.isfinite().all()),
            abs_max=float(delta.abs().max()), abs_mean=float(delta.abs().mean()),
            full_rms=float(full.square().mean().sqrt()),
            delta_rms=float(delta.square().mean().sqrt()),
            per_token_abs_max=delta.abs().amax(-1).flatten().tolist()))
    return rows


def load_config(run, source, param_dtype=None, fp32_normalization=False):
    overrides = json.loads((run / "effective_config.json").read_text())["overrides"]
    with initialize_config_dir(config_dir=str(source / "verl/trainer/config"), version_base=None):
        config = compose(config_name="ppo_trainer",
                         overrides=[f"{key}={value}" for key, value in overrides.items()])
    # ray_trainer normally inserts this before worker initialization; r7 has one step.
    with open_dict(config.actor_rollout_ref.actor.optim):
        config.actor_rollout_ref.actor.optim.total_training_steps = 1
    if param_dtype is not None:
        OmegaConf.update(config, "actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype",
                         param_dtype, force_add=True)
    if fp32_normalization:
        OmegaConf.update(config, "actor_rollout_ref.actor.fp32_normalization", True, force_add=True)
    return config


def main():
    from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, load_fsdp_optimizer
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--max-updates", type=int, choices=[0, 1, 2, 3, 4, 35], default=4)
    parser.add_argument("--param-dtype", choices=("bf16", "fp32"),
                        help="Zero-update A/B of the existing FSDP parameter precision setting")
    parser.add_argument("--fp32-normalization", action="store_true")
    parser.add_argument("--recompute-start", action="store_true",
                        help="Recompute old/ref at original weights before candidate updates; not on-policy RL")
    parser.add_argument("--hf-full-precision", action="store_true",
                        help="Zero-update HF-only FP32 arithmetic diagnostic; generation cache stays enabled")
    parser.add_argument("--observe-fp32-head", action="store_true",
                        help="Observe a shadow FP32 output projection; do not replace the model output")
    parser.add_argument("--trace-layers", action="store_true")
    parser.add_argument("--rowwise-rms-norm", action="store_true")
    parser.add_argument("--model-fp32", action="store_true",
                        help="Run the model forward in FP32 on every diagnostic path; keep generation cache")
    parser.add_argument("--disable-bf16-reduced-reduction", action="store_true",
                        help="Use PyTorch's existing full-precision BF16 GEMM accumulation option")
    parser.add_argument("--skip-paths", action="store_true",
                        help="Only reproduce the optimizer prefix; omit generation-path comparisons")
    args = parser.parse_args()
    if args.max_updates == 35 and not (args.param_dtype == "fp32" and args.fp32_normalization
            and args.recompute_start and args.skip_paths and not (args.model_fp32 or args.rowwise_rms_norm)):
        parser.error("Full replay is restricted to the user-approved FP32 parameter/temperature, BF16 model path")
    if args.model_fp32 and (args.param_dtype != "fp32" or not args.fp32_normalization):
        parser.error("FP32 model diagnostic requires FP32 parameters and normalization")
    if (args.hf_full_precision or args.observe_fp32_head or args.trace_layers or args.disable_bf16_reduced_reduction) and (args.max_updates != 0 or args.skip_paths):
        parser.error("HF full-precision comparison requires paths and zero updates")
    if (args.param_dtype is not None or args.fp32_normalization or args.rowwise_rms_norm) and args.max_updates != 0 and not args.recompute_start:
        parser.error("Candidate updates require --recompute-start; do not mix historical BF16 old/ref")
    rank = int(os.environ["RANK"])
    if int(os.environ["WORLD_SIZE"]) != 4:
        raise ValueError("r7 saved four rank-local batches; replay requires exactly four ranks")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(1)
    if args.disable_bf16_reduced_reduction:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    args.output.mkdir(parents=True, exist_ok=True)
    stream = (args.output / f"rank-{rank:02d}.jsonl").open("x")

    def emit(event, **values):
        stream.write(json.dumps(dict(event=event, rank=rank,
            time=datetime.now(timezone.utc).isoformat(), **values)) + "\n")
        stream.flush()

    config = load_config(args.run, args.source, args.param_dtype, args.fp32_normalization)
    worker_config = config.actor_rollout_ref
    if worker_config.model.path != "/mnt/data0601/yangjinyue/new_SCII/assets/Qwen3-4B-Instruct-2507-cdbee75":
        raise ValueError("This replay is scoped to the recorded original Instruct-2507 snapshot")
    with open_dict(worker_config.actor):
        worker_config.actor.ppo_diagnostics_dir = str(args.output / "ppo_diagnostics")
    emit("config", config=OmegaConf.to_container(worker_config, resolve=True),
         source=str(args.source), historical_run=str(args.run), max_updates=args.max_updates,
         skip_paths=args.skip_paths, param_dtype=args.param_dtype,
         fp32_normalization=args.fp32_normalization, recompute_start=args.recompute_start,
         hf_full_precision=args.hf_full_precision,
         observe_fp32_head=args.observe_fp32_head,
         trace_layers=args.trace_layers,
         rowwise_rms_norm=args.rowwise_rms_norm,
         model_fp32=args.model_fp32,
         allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
         allow_tf32=torch.backends.cuda.matmul.allow_tf32,
         float32_matmul_precision=torch.get_float32_matmul_precision())
    worker = ActorRolloutRefWorker(worker_config, role="actor_rollout")
    worker.init_model()
    if args.model_fp32:
        # Decorate the shared model forward, not just the rollout call. This covers
        # cached generate, full recompute and checkpointed train-mode forwards.
        worker.actor_module.forward = torch.autocast("cuda", enabled=False)(worker.actor_module.forward)
    if args.rowwise_rms_norm:
        from verl.models.transformers.qwen3_rowwise_rms_norm import install_qwen3_rowwise_rms_norm
        count = install_qwen3_rowwise_rms_norm(worker.actor_module)
        emit("rms_norm_installed", modules=count, source="flash_attn 2.6.3 rms_norm_fn; Qwen cast/multiply order preserved")
    actor, module = worker.actor, worker.actor_module_fsdp
    if worker._is_offload_param:
        load_fsdp_model_to_gpu(module)
    if worker._is_offload_optimizer:
        load_fsdp_optimizer(worker.actor_optimizer, device_id=torch.cuda.current_device())
    if actor.config.ppo_mini_batch_size != 2 or actor.config.ppo_epochs != 1:
        raise ValueError("Expected r7 local mini-batch=2 and one PPO epoch")
    path = args.run / "ppo_diagnostics" / f"update-0001-rank-{rank:02d}.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    emit("input", sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
         shape=list(saved["input_ids"].shape), optimizer_lr=worker.actor_optimizer.param_groups[0]["lr"])
    temperature = float(worker_config.rollout.temperature)
    response_length = saved["responses"].shape[1]
    prompt_length = saved["input_ids"].shape[1] - response_length
    # Same count/order of FSDP collective forwards on each rank.
    probes = [] if args.skip_paths else [7, [49, 0, 44, 62][rank]]
    current_path = ""
    dtypes = {}

    def observe_embedding(embedding, inputs, output):
        dtypes.setdefault(current_path, dict(weight=str(embedding.weight.dtype),
                                             activation=str(output.dtype)))

    head_values = {}
    layer_trace = {"full": {}, "cached": {}}

    def observe_layer(name, unused_module, unused_inputs, output):
        if current_path == "hf_full_fp32_normalization":
            layer_trace["full"][name] = prediction_states(output, current_path, response_length, length)
        elif current_path == "hf_cached_generate":
            values = layer_trace["cached"].setdefault(name, [])
            if len(values) < length:
                values.append(prediction_states(output, current_path, response_length, length))

    layer_handles = []
    if args.trace_layers:
        from functools import partial
        base = worker.actor_module.model
        targets = [("embedding", base.embed_tokens)]
        for index, layer in enumerate(base.layers):
            targets.extend([(f"layer{index}.input_norm", layer.input_layernorm),
                            (f"layer{index}.attention", layer.self_attn),
                            (f"layer{index}.post_attention_norm", layer.post_attention_layernorm),
                            (f"layer{index}.mlp", layer.mlp), (f"layer{index}.output", layer)])
            if index == 0:
                targets.extend((f"layer0.{name}", getattr(layer.self_attn, name))
                    for name in ("q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "o_proj"))
        targets.append(("final_norm", base.norm))
        layer_handles = [target.register_forward_hook(partial(observe_layer, name)) for name, target in targets]

    def observe_head(head, inputs, output):
        if current_path == "hf_full_fp32_normalization":
            head_values["full"] = fp32_head_logp(head,
                inputs[0][:, -response_length - 1:-1], labels, temperature)
        elif current_path == "hf_cached_generate":
            index = len(head_values["cached"])
            head_values["cached"].append(fp32_head_logp(head,
                inputs[0][:, -1:], labels[:, index:index + 1], temperature))
        # Returning None leaves the real forward output unchanged.

    head_handle = worker.actor_module.get_output_embeddings().register_forward_hook(observe_head) if args.observe_fp32_head else None
    handle = worker.actor_module.get_input_embeddings().register_forward_hook(observe_embedding)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
        module.eval()
        for sample in probes:
            head_values = {"cached": []}
            layer_trace = {"full": {}, "cached": {}}
            micro = {key: value[sample:sample + 1].cuda() for key, value in saved.items()}
            mask = micro["attention_mask"][:, -response_length:].bool()
            length = int(mask.sum())
            labels = micro["responses"]
            current_path = "actor_recompute"
            _, actor_logp = actor._forward_micro_batch(micro, temperature, calculate_entropy=False)
            current_path = "fsdp_full_fp32_normalization"
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = module(input_ids=micro["input_ids"], attention_mask=micro["attention_mask"],
                    position_ids=micro["position_ids"], use_cache=False).logits[:, -response_length - 1:-1]
            full_fsdp = selected(logits, labels, temperature)
            del logits
            with FSDP.summon_full_params(module, writeback=False, recurse=False), \
                 torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.hf_full_precision):
                current_path = "hf_full_fp32_normalization"
                logits = worker.actor_module(input_ids=micro["input_ids"],
                    attention_mask=micro["attention_mask"], position_ids=micro["position_ids"],
                    use_cache=False).logits[:, -response_length - 1:-1]
                full_hf = selected(logits, labels, temperature)
                dtypes[current_path]["logits"] = str(logits.dtype)
                del logits
                current_path = "hf_cached_generate"
                generation = module.generate(input_ids=micro["input_ids"][:, :prompt_length],
                    attention_mask=micro["attention_mask"][:, :prompt_length],
                    generation_config=GenerationConfig(do_sample=True, num_beams=1,
                        top_p=worker_config.rollout.top_p, top_k=worker_config.rollout.top_k,
                        temperature=temperature, num_return_sequences=1),
                    max_new_tokens=response_length, eos_token_id=worker.tokenizer.eos_token_id,
                    pad_token_id=worker.tokenizer.pad_token_id, use_cache=True,
                    logits_processor=[FixedTokens(labels, prompt_length)],
                    output_logits=True, output_scores=False, return_dict_in_generate=True)
                generated = generation.sequences[:, prompt_length:]
                dtypes[current_path]["logits"] = str(generation.logits[0].dtype)
                if not torch.equal(generated, labels[:, :length]):
                    raise RuntimeError("Fixed-token generate did not reproduce the saved response")
                cached = torch.zeros_like(actor_logp)
                for index, logits in enumerate(generation.logits):
                    cached[:, index] = selected(logits, labels[:, index], temperature)
                del generation
            values = dict(actor=actor_logp, fsdp_fp32_norm=full_fsdp, hf_full=full_hf,
                          hf_cached=cached, old=micro["old_log_probs"],
                          recorded_generate=micro["rollout_unwarped_log_probs"])
            comparisons = {name: gap(values[a], values[b], mask) for name, a, b in (
                ("actor_vs_old", "actor", "old"),
                ("normalization", "actor", "fsdp_fp32_norm"),
                ("parameter_precision_path", "fsdp_fp32_norm", "hf_full"),
                ("cached_vs_full", "hf_cached", "hf_full"),
                ("cached_vs_recorded_generate", "hf_cached", "recorded_generate"),
                ("cached_vs_actor", "hf_cached", "actor"))}
            if args.observe_fp32_head:
                head_cached = torch.zeros_like(cached)
                # Synced FSDP generate keeps forwarding after this rank's EOS.
                head_cached[:, :length] = torch.cat(head_values["cached"], dim=1)[:, :length]
                values.update(hf_full_fp32_head=head_values["full"], hf_cached_fp32_head=head_cached)
                comparisons["cached_vs_full_fp32_head"] = gap(head_cached, head_values["full"], mask)
                comparisons["fp32_head_effect_full"] = gap(head_values["full"], full_hf, mask)
                comparisons["fp32_head_effect_cached"] = gap(head_cached, cached, mask)
            emit("probability_paths", sample=sample, response=worker.tokenizer.decode(labels[mask]),
                 dtypes=dtypes, comparisons=comparisons,
                 log_probs={key: value[mask].float().cpu().tolist() for key, value in values.items()})
            if args.trace_layers:
                emit("layer_paths", sample=sample, layers=layer_gaps(layer_trace))
    handle.remove()
    if head_handle is not None:
        head_handle.remove()
    for handle in layer_handles:
        handle.remove()

    # Exact historical replay, or an explicit new candidate start with the same tokens/A.
    count = args.max_updates * actor.config.ppo_mini_batch_size
    if count:
        if args.max_updates == 35:
            if len(saved["responses"]) != 69:
                raise ValueError("Expected the complete historical 69-sample rank batch")
            count = len(saved["responses"])
        tensors = {key: value[:count] for key, value in saved.items()}
        prefix = DataProto.from_dict(tensors=tensors, meta_info={"temperature": temperature})
        module.eval()
        max_gap = 0.
        starting = []
        with torch.no_grad():
            for micro in prefix.batch.split(1):
                micro = micro.cuda()
                _, logp = actor._forward_micro_batch(micro, temperature, calculate_entropy=False)
                mask = micro["attention_mask"][:, -response_length:].bool()
                if not torch.isfinite(logp[mask]).all():
                    raise RuntimeError("Nonfinite starting probability; refusing parameter updates")
                max_gap = max(max_gap, gap(logp, micro["old_log_probs"], mask)["abs_max"])
                starting.append(logp.cpu())
        global_gap = torch.tensor(max_gap, device="cuda")
        dist.all_reduce(global_gap, op=dist.ReduceOp.MAX)
        emit("prefix_start", max_logprob_gap_all_ranks=global_gap.item(), tolerance=1e-5,
             comparison="historical_old", recompute_start=args.recompute_start)
        if args.recompute_start:
            tensors = rebased_tensors(tensors, torch.cat(starting))
            prefix = DataProto.from_dict(tensors=tensors, meta_info={"temperature": temperature})
            module.train()
            train_gap = 0.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
                for micro in prefix.batch.split(1):
                    micro = micro.cuda()
                    _, logp = actor._forward_micro_batch(micro, temperature, calculate_entropy=False)
                    mask = micro["attention_mask"][:, -response_length:].bool()
                    if not torch.isfinite(logp[mask]).all():
                        raise RuntimeError("Nonfinite train-mode starting probability")
                    train_gap = max(train_gap, gap(logp, micro["old_log_probs"], mask)["abs_max"])
            train_gap_all = torch.tensor(train_gap, device="cuda")
            dist.all_reduce(train_gap_all, op=dist.ReduceOp.MAX)
            emit("rebased_start", max_train_eval_gap_all_ranks=train_gap_all.item(),
                 tolerance=1e-5, reference="original actor evaluated by candidate path; not a separate ref worker")
            if train_gap_all.item() > 1e-5:
                raise RuntimeError("Train/eval candidate start differs; refusing updates")
        elif global_gap.item() > 1e-5:
            raise RuntimeError("Saved starting probabilities not reproduced; refusing parameter updates")
        # Match ActorRolloutRefWorker.update_actor's device boundary.
        prefix = prefix.to(torch.cuda.current_device())
        if args.max_updates == 35:
            from full_update_observers import install_update_observers
            install_update_observers(actor, prefix.batch, temperature, emit, args.output / f"post-each-rank-{rank:02d}.jsonl")
        metrics = actor.update_policy(prefix)
        emit("updates_complete", optimizer_updates=args.max_updates, metrics=metrics,
             optimizer_state_dtypes=sorted({str(v.dtype) for state in worker.actor_optimizer.state.values()
                 for v in state.values() if torch.is_tensor(v)}))
    emit("completed", checkpoint_saved=False,
         peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
         peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(),
         note="Numerical replay only; no SC2 or learning result")
    stream.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
