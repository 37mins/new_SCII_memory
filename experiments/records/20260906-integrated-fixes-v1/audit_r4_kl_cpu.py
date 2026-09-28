"""Read frozen r4 diagnostics on CPU; no model/checkpoint tensor loads or writes.

Run with CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1.
Only step-2 diagnostic tensors (~19 MB), JSONL and frozen source are read.
Gradients below are loss-to-logprob derivatives, NOT parameter gradients.
"""
import ast
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import types

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
sys.path.insert(0, "/root/new-scii-spb2-level5-control/python")
import torch

torch.set_num_threads(1)
SOURCE = Path("/root/new-scii-empty-bo-frame-20260907/code-ca7885d")
RUN = Path("/mnt/data0601/yangjinyue/new_SCII/experiments/20260906-integrated-fixes-v1/runs/20260906-integrated-fixes-v1-qwen-instruct2507-l6-8gpu-fp32-rl96-gate8-s0-r4-train8")


def frozen_functions(relative, names, namespace):
    tree = ast.parse((SOURCE / relative).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), relative, "exec"), namespace)


ns = {"torch": torch}
frozen_functions("verl/utils/torch_functional.py", ["masked_mean"], ns)
ns["verl_F"] = types.SimpleNamespace(masked_mean=ns["masked_mean"])
frozen_functions("verl/trainer/ppo/core_algos.py", ["agg_loss", "compute_policy_loss", "kl_penalty"], ns)
artifacts, ranks, records, hashes, phase_values, minis = [], [], [], collections.defaultdict(list), collections.defaultdict(list), []
all_old, all_ref, all_adv, all_mask, all_post = [], [], [], [], []
for rank in range(8):
    stem = RUN / "ppo_diagnostics" / f"update-0002-rank-{rank:02d}"
    paths = [stem.with_suffix(".pt"), stem.with_suffix(".jsonl")]
    for path in paths:
        content = path.read_bytes()
        artifacts.append({"path": str(path), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
    data = torch.load(paths[0], map_location="cpu", weights_only=True)
    rows = [json.loads(line) for line in paths[1].read_text().splitlines()]
    raw_mask = data["attention_mask"][:, -96:]
    assert bool(((raw_mask == 0) | (raw_mask == 1)).all())
    mask = raw_mask.bool()
    assert data["responses"].shape == (23, 96)
    assert set(mask.sum(1).gt(0).tolist()) == {True}
    assert torch.equal(mask, torch.arange(96)[None, :] < mask.sum(1)[:, None])
    assert bool((data["responses"].gather(1, (mask.sum(1)-1)[:, None]) == 151645).all())
    assert all(v.device.type == "cpu" for v in data.values())
    assert collections.Counter(r["phase"] for r in rows) == {"pre_update": 23, "update": 23, "post_update": 23}
    rank_record = {"rank": rank, "samples": 23, "tokens": int(mask.sum()), "phases": {}}
    for phase in ("pre_update", "update", "post_update"):
        phase_rows = [r for r in rows if r["phase"] == phase]
        assert [r["sample_offset"] for r in phase_rows] == list(range(23))
        if phase == "update":
            assert [r["optimizer_updates_before"] for r in phase_rows] == list(range(23))
        if phase == "post_update":
            assert {r["optimizer_updates_before"] for r in phase_rows} == {23}
        new = torch.tensor([r["log_prob"][0] for r in phase_rows], dtype=torch.float32)
        gap = (new - data["old_log_probs"])[mask].double()
        phase_values[phase].extend(phase_rows)
        rank_record["phases"][phase] = {"micro_k1": sum(r["ppo_k1_mean"] for r in phase_rows) / 23,
            "token_k1": float(-gap.mean()), "drop_gt20": int((gap < -20).sum()), "abs_max": float(gap.abs().max())}
        if phase == "post_update":
            all_post.append(new)
    ranks.append(rank_record)
    for j in range(23):
        active_ids = data["input_ids"][j][data["attention_mask"][j].bool()]
        digest = hashlib.sha256(active_ids.numpy().tobytes()).hexdigest()
        hashes[digest].append({"rank": rank, "offset": j, "tokens": int(mask[j].sum())})
        records.append({"rank": rank, "offset": j, "active_input_sha256": digest,
            "tokens": int(mask[j].sum()), "adv_min": float(data["advantages"][j][mask[j]].min()),
            "adv_max": float(data["advantages"][j][mask[j]].max())})
    all_old.append(data["old_log_probs"])
    all_ref.append(data["ref_log_prob"])
    all_adv.append(data["advantages"])
    all_mask.append(mask)

old, ref, adv, mask, post = map(torch.cat, (all_old, all_ref, all_adv, all_mask, all_post))
post = post.requires_grad_(True)
# Sum independent local response means, then apply the 8-rank mean. This gives
# the actual per-update normalization for each sample, not a 23-update sum loss.
pg_terms, ref_terms = [], []
for i in range(len(old)):
    m = mask[i:i+1].float()
    loss = ns["compute_policy_loss"](old[i:i+1], post[i:i+1], adv[i:i+1], m,
        cliprange=.2, cliprange_low=.2, cliprange_high=.28, clip_ratio_c=3., loss_agg_mode="token-mean")[0]
    kld = ns["kl_penalty"](post[i:i+1], ref[i:i+1], "low_var_kl")
    pg_terms.append(loss / 8)
    ref_terms.append(ns["agg_loss"](kld, m, "token-mean") * .01 / 8)
pg_grad = torch.autograd.grad(sum(pg_terms), post, retain_graph=True)[0]
ref_grad = torch.autograd.grad(sum(ref_terms), post)[0]
gap = (post.detach() - old)
tail = (gap < -20) & mask
lengths = mask.sum(1)
active_gap = gap[mask].double()
worst = torch.nonzero(mask)[active_gap.argmin()]
i, j = map(int, worst)
for update in range(23):
    sample_rows = [r for r in records if r["offset"] == update]
    rows = [r for r in phase_values["update"] if r["sample_offset"] == update]
    minis.append({"optimizer_update_1based": update+1, "samples": sample_rows,
        "micro_k1_before": sum(r["ppo_k1_mean"] for r in rows)/8,
        "max_gap_before": max(r["abs_logprob_gap_max"] for r in rows),
        "exponent_clamped_before": sum(r["exponent_clamped_tokens"] for r in rows)})
summary = {}
for phase, rows in phase_values.items():
    summary[phase] = {"rows": len(rows), "tokens": sum(r["tokens"] for r in rows),
        "micro_k1": sum(r["ppo_k1_mean"] for r in rows)/len(rows),
        "token_k1": sum(r["ppo_k1_mean"]*r["tokens"] for r in rows)/sum(r["tokens"] for r in rows),
        "max_abs_gap": max(r["abs_logprob_gap_max"] for r in rows),
        "clamped_token_count": sum(r["exponent_clamped_tokens"] for r in rows)}
out = {"at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "torch": torch.__version__,
    "source_commit": "ca7885d886aaaafb39fc105b1d5d46cddf4bcc5a", "run_id": RUN.name,
    "scope": "CPU diagnostic tensors only; no model forward, parameter gradients, optimizer update or checkpoint tensor loads",
    "artifacts": artifacts, "per_rank": ranks, "phases": summary,
    "samples": 184, "distinct_active_token_inputs": len(hashes),
    "mask_checks": "All 184 response masks binary, nonempty, contiguous; include final EOS 151645 and exclude right padding",
    "duplicate_inputs": [v for v in hashes.values() if len(v)>1],
    "response_lengths": {"min": int(lengths.min()), "max": int(lengths.max()), "sum": int(lengths.sum()),
        "histogram": dict(sorted(collections.Counter(lengths.tolist()).items()))},
    "post_gap_quantiles": dict(zip(["min", "p01", "p05", "p50", "p95", "p99", "max"],
        torch.quantile(active_gap, torch.tensor([0., .01, .05, .5, .95, .99, 1.], dtype=torch.float64)).tolist())),
    "post_drop_gt20": {"tokens": int(tail.sum()), "positive_adv": int((tail & (adv>0)).sum()),
        "negative_adv": int((tail & (adv<0)).sum()), "zero_pg_direct_gradient": int((tail & (pg_grad==0)).sum()),
        "zero_ref_direct_gradient": int((tail & (ref_grad==0)).sum())},
    "worst": {"rank": i//23, "offset": i%23, "token_index": j, "old": float(old[i,j]),
        "post": float(post.detach()[i,j]), "ref": float(ref[i,j]), "advantage": float(adv[i,j]),
        "gap": float(gap[i,j]), "pg_direct_gradient": float(pg_grad[i,j]), "ref_direct_gradient": float(ref_grad[i,j])},
    "mini_batches": minis,
    "not_saved": ["per-update parameter gradients before/after reduction and clipping", "per-update parameter deltas", "per-update Adam moments", "cached rollout logprob in this trace", "decision IDs in .pt"]}
print(json.dumps(out, allow_nan=False))
