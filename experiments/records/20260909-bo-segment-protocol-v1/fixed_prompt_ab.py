"""Frozen-query prompt A/B through existing FSDP/HF generation, without SC2/PPO."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests"))
from test_r6_format_reward import strict_validator

BASELINE_SHA = "71fae36ccc112d4c9cd303ab084c1e0a0c7eebec"
INPUT_SHA = "5067499ab9efd79f88bb5c11a333780dea003417bb6d53a89fdc6a7ba8ecdebf"
MARKER = "The example below is generated from that same current list. Follow its shape exactly:"
CLARIFICATION = """For each action name, add its quantities across ALL lines in your answer: a line without `x N` counts as 1. This cumulative total must not exceed that action's count in the remaining BO list.
The example below demonstrates output format only; it is not an additional task or an extra quota. Do not append the remaining list to a copied example and count both separately.
Aim for approximately 15 list entries, but do not repeat actions just to fill that target. Once this near-term segment is complete, close </answer> and stop; do not restart the list. Repeated names and `x N` remain allowed within the cumulative quota.
The example below is generated from that same current list. Follow its format:"""


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def candidate(prompt):
    if prompt.count(MARKER) != 1:
        raise ValueError("Expected one frozen example marker")
    return prompt.replace(MARKER, CLARIFICATION)


def load_rows(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != INPUT_SHA:
        raise ValueError("Wrong source log")
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    rows = [dict(r, source_line=i) for i, r in enumerate(rows, 1) if r["policy_action_applied"]]
    assert len(rows) == 41
    assert len({(r["env_idx"], r["policy_action_id"]) for r in rows}) == 41
    return rows


def load_config(run):
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict
    overrides = json.loads((run / "effective_config.json").read_text())["overrides"]
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=[f"{k}={v}" for k, v in overrides.items()])
    # Common four-rank orchestration; neither arm calls a trainer or optimizer step.
    config.trainer.n_gpus_per_node = 4
    config.env.rollout.n = 4
    with open_dict(config.actor_rollout_ref.actor.optim):
        config.actor_rollout_ref.actor.optim.total_training_steps = 1
    return config


def score(prompt, answer, validate):
    valid, reason = validate(prompt, answer)
    allowed = {}
    for line in prompt.split("The remaining BO list in this phase:\n", 1)[1].split(MARKER, 1)[0].splitlines():
        match = re.fullmatch(r"- (.+?) x (\d+)", line)
        if match:
            allowed[match[1]] = int(match[2])
    selected = Counter()
    lines = []
    for line in answer.splitlines():
        if not line.startswith("- "):
            continue
        lines.append(line)
        match = re.fullmatch(r"- (.+?) x (\d+)", line)
        name, count = (match[1], int(match[2])) if match else (line[2:].strip(), 1)
        if name in allowed:
            selected[name] += count
    return dict(valid=valid, reason=reason, selected_counts=dict(selected),
                over_quota={k: [v, allowed[k]] for k, v in selected.items() if v > allowed[k]},
                bullet_count=len(lines), duplicate_lines=len(lines)-len(set(lines)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    rows = load_rows(args.source_run / "prompt_response.jsonl")
    manifest = json.loads((args.source_run / "manifest.json").read_text())
    assert manifest["source_commit"] == BASELINE_SHA
    config = load_config(args.source_run)
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path,
                                              local_files_only=True, trust_remote_code=False)
    os.environ["SC2_ORIGINAL_BO_SEGMENT"] = "1"
    validate = strict_validator()
    cases = []
    for i, row in enumerate(rows):
        a, b = row["prompt"], candidate(row["prompt"])
        assert b.replace(CLARIFICATION, MARKER) == a
        assert validate(a, row["llm_response"]) == validate(b, row["llm_response"])
        rendered = {arm: tokenizer.apply_chat_template([dict(role="user", content=p)],
                    tokenize=False, add_generation_prompt=True, **config.data.apply_chat_template_kwargs)
                    for arm, p in (("A", a), ("B", b))}
        lengths = {arm: len(tokenizer.encode(p, add_special_tokens=False)) for arm, p in rendered.items()}
        assert max(lengths.values()) <= config.data.max_prompt_length
        cases.append(dict(index=i, source_line=row["source_line"], env=row["env_idx"],
                          action_id=row["policy_action_id"], seed=10000+i,
                          prompts=dict(A=a, B=b), token_lengths=lengths))
    if args.prepare_only:
        print(json.dumps(dict(cases=len(cases), historical_valid=sum(r["semantic_action_valid"] for r in rows),
              source_sha=BASELINE_SHA, source_log_sha256=INPUT_SHA,
              prompt_ranges={arm:[min(c["token_lengths"][arm] for c in cases),max(c["token_lengths"][arm] for c in cases)] for arm in ("A","B")},
              generation_config=OmegaConf.to_container(config.actor_rollout_ref.rollout, resolve=True),
              candidate_example=cases[0]["prompts"]["B"]),ensure_ascii=False,indent=2))
        return

    import torch
    import torch.distributed as dist
    import numpy as np
    from types import SimpleNamespace
    from verl import DataProto
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
    if int(os.environ["WORLD_SIZE"]) != 4:
        raise ValueError("Approved four-rank experiment only")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(1)
    output = args.output
    stream = (output / f"rank-{rank}.jsonl").open("x")
    def emit(event, **data):
        stream.write(json.dumps(dict(event=event, rank=rank, time=now(), **data),ensure_ascii=False)+"\n")
        stream.flush()
    if rank == 0:
        (output / "cases.json").write_text(json.dumps(cases,ensure_ascii=False,indent=2))
        (output / "effective_config.json").write_text(json.dumps(OmegaConf.to_container(config, resolve=True),indent=2))
        (output / "manifest.json").write_text(json.dumps(dict(status="running",started_at=now(),
            source_sha=os.environ["SOURCE_SHA"],baseline_sha=BASELINE_SHA,
            source_run=str(args.source_run),source_log_sha256=INPUT_SHA,
            model_source=manifest["model_source"],model_manifest_sha256=manifest["model_manifest_sha256"],
            gpu_selection="4,5,6,7",world_size=4,real_queries=41,arms=["A","B"],
            sampling_seed_rule="10000 + original consumed-query index, reset independently before each arm",
            max_wall_seconds=1800,optimizer_updates=0,games=0),indent=2))
    worker = ActorRolloutRefWorker(config.actor_rollout_ref, role="actor_rollout")
    worker.init_model()
    assert worker.actor_model_config.model_type == "sse_swa_moba"
    assert worker.actor_model_config.swa_dropout == 0
    assert next(worker.actor_module_fsdp.parameters()).dtype == torch.float32
    assert len(worker.actor_optimizer.state) == 0
    emit("model_loaded", architecture=worker.actor_model_config.architectures, swa_dropout=0,
         param_dtype="fp32", autocast="bf16", cache=True, optimizer_updates=0)
    collector = TrajectoryCollector(config, worker.tokenizer)
    dummy = SimpleNamespace(non_tensor_batch=dict(raw_prompt=[None],data_source=["sc2"]))
    for arm in ("A", "B"):
        for start in range(0, 41, 4):
            index = start + rank
            is_padding = index >= 41
            case = cases[start if is_padding else index]
            prompt = case["prompts"][arm]
            sample = collector.preprocess_single_sample(0, dummy,
                       dict(text=[prompt], image=None, anchor=[prompt]))
            batch = DataProto.from_dict(tensors={k:sample[k].unsqueeze(0) for k in
                           ("input_ids","attention_mask","position_ids")},
                           meta_info=dict(validate=True,do_sample=True))
            torch.manual_seed(case["seed"])
            np.random.seed(case["seed"])
            emit("generation_start", arm=arm, index=case["index"], seed=case["seed"], padding=is_padding,
                 cuda_rng_sha256=hashlib.sha256(bytes(torch.cuda.get_rng_state().tolist())).hexdigest())
            result = worker.generate_sequences(batch)
            ids = result.batch["responses"][0].tolist()
            eos = worker.tokenizer.eos_token_id
            eos_seen = eos in ids
            count = ids.index(eos)+1 if eos_seen else len(ids)
            response = worker.tokenizer.decode(ids,skip_special_tokens=True)
            emit("generation", arm=arm, index=case["index"], env=case["env"], action_id=case["action_id"],
                 seed=case["seed"], padding=is_padding, raw_answer=response, token_ids=ids,
                 token_count=count, eos_seen=eos_seen, prompt_sha256=sha(prompt),
                 **score(rows[case["index"]]["prompt"],response,validate))
    assert len(worker.actor_optimizer.state) == 0
    emit("completed", optimizer_updates=0, games=0)
    dist.barrier()
    if rank == 0:
        all_rows = [json.loads(line) for n in range(4) for line in (output/f"rank-{n}.jsonl").read_text().splitlines()]
        answers = [r for r in all_rows if r["event"]=="generation" and not r["padding"]]
        assert len(answers)==82
        by_key={(r["arm"],r["index"]):r for r in answers}
        assert len(by_key)==82
        pairs=[(by_key["A",i],by_key["B",i]) for i in range(41)]
        summary=dict(status="completed",ended_at=now(),optimizer_updates=0,games=0,
                     real_answers=82,padding_answers=6,
                     corrected=sum(not a["valid"] and b["valid"] for a,b in pairs),
                     regressed=sum(a["valid"] and not b["valid"] for a,b in pairs),
                     arms={arm:dict(valid=sum(r["valid"] for r in answers if r["arm"]==arm),
                          over_quota=sum(bool(r["over_quota"]) for r in answers if r["arm"]==arm),
                          no_eos=sum(not r["eos_seen"] for r in answers if r["arm"]==arm)) for arm in ("A","B")})
        (output/"result.json").write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary),flush=True)
    dist.barrier()
    stream.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
