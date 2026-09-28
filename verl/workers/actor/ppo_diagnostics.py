"""Opt-in, rank-local PPO observations; never used to form a training loss."""

import json
from pathlib import Path

import torch

from verl.trainer.ppo.core_algos import kl_penalty


def record_micro_batch(path, actor, data, log_prob, phase, updates, sample_offset):
    mask = data["attention_mask"][:, -data["responses"].shape[-1]:].bool()
    new = log_prob.detach().double().cpu()
    old = data["old_log_probs"].detach().double().cpu()
    mask = mask.cpu()
    advantage = data["advantages"].detach().double().cpu()
    gap = new[mask] - old[mask]
    row = dict(phase=phase, optimizer_updates_before=updates,
               sample_offset=sample_offset, samples=len(new), tokens=int(mask.sum()),
               finite=bool(torch.isfinite(gap).all()), log_prob=new.tolist())
    if gap.numel() and row["finite"]:
        low = actor.config.clip_ratio_low
        high = actor.config.clip_ratio_high
        low = actor.config.clip_ratio if low is None else low
        high = actor.config.clip_ratio if high is None else high
        ratio = gap.clamp(-20, 20).exp()
        below, above = ratio < 1 - low, ratio > 1 + high
        a = advantage[mask]
        row.update(
            ppo_k1_mean=float(-gap.mean()),
            abs_logprob_gap_mean=float(gap.abs().mean()),
            abs_logprob_gap_max=float(gap.abs().max()),
            ppo_k3_capped_mean=float(kl_penalty(old[mask], new[mask], "k3").mean()),
            ratio_min=float(ratio.min()), ratio_max=float(ratio.max()),
            ratio_below_clip=float(below.double().mean()),
            ratio_above_clip=float(above.double().mean()),
            objective_clipfrac=float(((below & (a < 0)) | (above & (a > 0))).double().mean()),
            exponent_clamped_tokens=int((gap.abs() > 20).sum()),
        )
        worst = torch.nonzero(mask)[gap.abs().argmax()]
        i, j = map(int, worst)
        row["worst_token"] = dict(sample=sample_offset + i, response_index=j,
            token_id=int(data["responses"][i, j]), old=float(old[i, j]),
            new=float(new[i, j]), advantage=float(advantage[i, j]))
        if "ref_log_prob" in data:
            ref = data["ref_log_prob"].detach().double().cpu()[mask]
            row["reference_k3_capped_mean"] = float(kl_penalty(new[mask], ref, "k3").mean())
        for name in ("unwarped", "sampling"):
            key = f"rollout_{name}_log_probs"
            if key in data:
                rollout = data[key].detach().double().cpu()[mask]
                diff = old[mask] - rollout
                row[f"rollout_{name}_finite"] = bool(torch.isfinite(diff).all())
                if row[f"rollout_{name}_finite"]:
                    row[f"rollout_{name}_eval_abs_gap_mean"] = float(diff.abs().mean())
                    row[f"rollout_{name}_eval_abs_gap_max"] = float(diff.abs().max())
                    row[f"rollout_{name}_eval_signed_gap_mean"] = float(diff.mean())
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def record_full_batch(path, actor, batch, temperature, phase, updates):
    """Observe the same complete batch at fixed weights; restore CPU/CUDA RNG."""
    device = batch["input_ids"].device
    devices = [device.index] if device.type == "cuda" else []
    size = actor.config.ppo_micro_batch_size_per_gpu
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        for index, micro in enumerate(batch.split(size)):
            _, log_prob = actor._forward_micro_batch(micro, temperature=temperature,
                                                     calculate_entropy=False)
            record_micro_batch(path, actor, micro, log_prob, phase, updates, index * size)


def start_trace(actor, batch, temperature, multi_turn):
    # This diagnostic is deliberately scoped to the current fixed-size text actor.
    if multi_turn or actor.config.use_dynamic_bsz:
        raise ValueError("PPO diagnostics require fixed text micro-batches without loss_mask")
    actor._ppo_trace_step = getattr(actor, "_ppo_trace_step", 0) + 1
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    root = Path(actor.config.ppo_diagnostics_dir)
    root.mkdir(parents=True, exist_ok=True)
    stem = f"update-{actor._ppo_trace_step:04d}-rank-{rank:02d}"
    path = root / f"{stem}.jsonl"
    with path.open("x"):
        pass
    # Preserve the exact tokens, masks, old/ref probabilities and advantages.
    with (root / f"{stem}.pt").open("xb") as stream:
        torch.save({key: value.detach().cpu() for key, value in batch.items()}, stream)
    record_full_batch(path, actor, batch, temperature, "pre_update", 0)
    return path
