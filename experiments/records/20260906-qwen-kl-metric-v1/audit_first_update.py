"""Read existing r32 tensors/logs on CPU; no model load, backward or file writes."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines(keepends=True)
            if s.endswith("\n")]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(run, historical, model):
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    original = rows(historical / "prompt_response.jsonl")
    artifacts, samples, gradients, deltas = [], [], [], []
    batches, events = [], []
    worst = None
    for rank in range(4):
        tensor_path = run / "ppo_diagnostics" / f"update-0001-rank-{rank:02d}.pt"
        pre_path = tensor_path.with_suffix(".jsonl")
        post_path = run / f"post-each-rank-{rank:02d}.jsonl"
        event_path = run / f"rank-{rank:02d}.jsonl"
        artifacts.extend(dict(path=str(p), sha256=sha(p))
                         for p in (tensor_path, pre_path, post_path, event_path))
        saved = torch.load(tensor_path, map_location="cpu", weights_only=True)
        batches.append(saved)
        event_rows = rows(event_path)
        events.append(event_rows)
        pre = {x["sample_offset"]: x for x in rows(pre_path) if x["phase"] == "pre_update"}
        for item in rows(post_path):
            if item["optimizer_updates_before"] != 1:
                continue
            i = item["sample_offset"]
            mask = saved["attention_mask"][i, -saved["responses"].shape[1]:].bool()
            before = torch.tensor(pre[i]["log_prob"][0], dtype=torch.float64)
            after = torch.tensor(item["log_prob"][0], dtype=torch.float64)
            delta = after - before
            deltas.append(delta[mask])
            j = int(torch.where(mask, delta.abs(), -torch.inf).argmax())
            if worst is None or abs(float(delta[j])) > abs(worst["delta_logprob"]):
                worst = dict(rank=rank, sample=i, response_index=j,
                             pre_logprob=float(before[j]), post_logprob=float(after[j]),
                             delta_logprob=float(delta[j]))
        gradients.extend({"rank": rank, **x} for x in event_rows
                         if (x["event"] == "parameter_gradient_shards" and x["optimizer_updates_before"] == 0)
                         or (x["event"] == "optimizer_update" and x["optimizer_updates"] == 1))

    def decode(saved, i):
        length = saved["responses"].shape[1]
        active = saved["attention_mask"][i].bool()
        ids = saved["input_ids"][i][active]
        prompt = tokenizer.decode(saved["input_ids"][i, :-length][active[:-length]])
        response_ids = saved["responses"][i][active[-length:]]
        response = tokenizer.decode(response_ids)
        raw_response = tokenizer.decode(response_ids, skip_special_tokens=True)
        matches = [dict(line=n + 1, **r) for n, r in enumerate(original)
                   if r["llm_response"] == raw_response
                   and tokenizer.apply_chat_template([dict(role="user", content=r["prompt"])],
                       tokenize=False, add_generation_prompt=True) == prompt]
        return dict(prompt=prompt, response=response, response_token_ids=response_ids.tolist(),
                    active_full_input_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                    raw_log_matches=matches)

    saved = batches[worst["rank"]]
    i, j = worst["sample"], worst["response_index"]
    worst.update(decode(saved, i))
    worst.update(token_id=int(saved["responses"][i, j]),
                 token=tokenizer.decode([int(saved["responses"][i, j])]),
                 old_logprob=float(saved["old_log_probs"][i, j]),
                 advantage=float(saved["advantages"][i, j]),
                 pre_probability=math.exp(worst["pre_logprob"]),
                 post_probability=math.exp(worst["post_logprob"]))
    for rank, (saved, event_rows) in enumerate(zip(batches, events)):
        config = event_rows[0]["config"]
        accumulation = (config["actor"]["ppo_mini_batch_size"] * config["rollout"]["n"] // 4
                        // config["actor"]["ppo_micro_batch_size_per_gpu"])
        for item in event_rows:
            if item["event"] != "token_loss_gradient" or item["optimizer_updates_before"] != 0:
                continue
            i = item["sample_offset"]
            active = saved["attention_mask"][i, -saved["responses"].shape[1]:].bool()
            advantage = saved["advantages"][i][active].double()
            expected = -advantage / (accumulation * int(active.sum()))
            measured = torch.tensor(item["gradient"], dtype=torch.float64)
            decoded = decode(saved, i)
            # Full source rows are kept for the anomalous target; first-eight matches only need identity.
            decoded["raw_log_matches"] = [{k: r[k] for k in ("line", "time", "step_index", "env_idx", "policy_action_id")}
                                          for r in decoded["raw_log_matches"]]
            samples.append(dict(rank=rank, sample=i, tokens=int(active.sum()), **decoded,
                advantage=advantage.tolist(), token_loss_gradient=measured.tolist(),
                ppo_gradient_formula_max_error=float((measured - expected).abs().max()),
                matches_target=decoded["active_full_input_sha256"] == worst["active_full_input_sha256"]))
    delta = torch.cat(deltas)
    shard_norm = math.sqrt(sum(s["l2"] ** 2 for e in gradients if e["event"] == "parameter_gradient_shards" for s in e["shards"]))
    return dict(run_id=run.name, frozen_source=(run / "source_commit.txt").read_text().strip(),
        original_log=dict(path=str(historical / "prompt_response.jsonl"), sha256=sha(historical / "prompt_response.jsonl")),
        artifacts=artifacts, target=worst, first_update_samples=samples,
        first_update_tokens=sum(s["tokens"] for s in samples),
        first_eight_unique_full_inputs=len({s["active_full_input_sha256"] for s in samples}),
        parameter_gradient_events=gradients, norm_reconstructed_from_shards=shard_norm,
        full_batch=dict(responses=len(deltas), valid_tokens=len(delta),
            abs_quantiles=dict(zip(["p50", "p90", "p99", "max"], torch.quantile(delta.abs(), torch.tensor([.5, .9, .99, 1.], dtype=torch.float64)).tolist())),
            abs_mean=float(delta.abs().mean())),
        limitation="Existing logs have no parameter-gradient tensors, parameter deltas or Adam moments. Direct logprob gradient excludes entropy. This audit cannot certify the parameter/optimizer chain.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("historical", type=Path)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run, args.historical, args.model), ensure_ascii=False, indent=2))
