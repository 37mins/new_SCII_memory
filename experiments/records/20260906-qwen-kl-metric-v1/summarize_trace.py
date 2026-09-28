"""Summarize saved rank-local diagnostics without loading a model or changing a run."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def aggregate(rows):
    valid = [r for r in rows if r["finite"] and r["tokens"]]
    tokens = sum(r["tokens"] for r in valid)
    result = dict(micro_batches=len(rows), finite_micro_batches=len(valid), tokens=tokens)
    if not valid:
        return result
    for key in ("ppo_k1_mean", "ppo_k3_capped_mean", "reference_k3_capped_mean",
                "abs_logprob_gap_mean", "ratio_below_clip", "ratio_above_clip", "objective_clipfrac"):
        if all(key in r for r in valid):
            result[key] = sum(r[key] for r in valid) / len(valid)
            result[key + "_token_weighted"] = sum(r[key]*r["tokens"] for r in valid) / tokens
    result["abs_logprob_gap_max"] = max(r["abs_logprob_gap_max"] for r in valid)
    for name in ("unwarped", "sampling"):
        prefix = f"rollout_{name}"
        captured = [r for r in valid if r.get(prefix + "_finite")]
        result[prefix + "_finite_micro_batches"] = len(captured)
        if captured:
            count = sum(r["tokens"] for r in captured)
            for suffix in ("_eval_abs_gap_mean", "_eval_signed_gap_mean"):
                result[prefix + suffix] = sum(r[prefix + suffix] * r["tokens"] for r in captured) / count
            result[prefix + "_eval_abs_gap_max"] = max(r[prefix + "_eval_abs_gap_max"] for r in captured)
    return result


def summarize(directory, expected_ranks=8):
    phases, updates, rank_counts = defaultdict(list), defaultdict(list), {}
    for path in sorted(directory.glob("update-0001-rank-*.jsonl")):
        counts = defaultdict(int)
        # A currently written final line is not a completed diagnostic record.
        lines = path.read_text().splitlines(keepends=True)
        for line in lines:
            if not line.endswith("\n"):
                continue
            row = json.loads(line)
            phases[row["phase"]].append(row)
            counts[row["phase"]] += row["samples"]
            if row["phase"] == "update":
                updates[row["optimizer_updates_before"]].append(row)
        rank_counts[path.stem] = dict(counts)
    complete = len(rank_counts) == expected_ranks and all(
        c.get("pre_update", 0) > 0 and c.get("pre_update") == c.get("update") == c.get("post_update")
        for c in rank_counts.values())
    return dict(complete_single_epoch_trace=complete, rank_sample_counts=rank_counts,
                phases={name: aggregate(rows) for name, rows in phases.items()},
                optimizer_updates=[dict(updates_before=index, **aggregate(rows))
                                   for index, rows in sorted(updates.items())])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--ranks", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory, args.ranks), indent=2))
