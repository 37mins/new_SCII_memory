"""Read saved L7 logs through step24; no model, optimizer, or environment calls."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

FIRST = "20260910-spb-fsdp-layer-wrap-v1-spb2-l7-g2-opening-quota-rl4-4gpu-s0-r1"
RESUME = "20260910-spb-fsdp-layer-wrap-v1-spb2-l7-g2-opening-quota-resume4to96-4gpu-s0-r1"


def read_jsonl(path):
    return [dict(json.loads(line), source_line=i) for i, line in enumerate(path.read_text().splitlines(), 1)]


def audit(first, resume):
    batches, examples, artifacts = [], [], {}
    answers = []
    for folder, run_id, start, stop, source in (
        (first, FIRST, 1, 4, "cfdd3b468cf9bbe3a1f85ce8dc9a5a9ea530ffbd"),
        (resume, RESUME, 5, 24, "90b3fdc2fb7fadc1230861306b20c27540d9b736"),
    ):
        manifest = json.loads((folder / "manifest.json").read_text())
        assert manifest["source_commit"] == source and not manifest["dirty"]
        raw = read_jsonl(folder / "prompt_response.jsonl")
        outcomes = read_jsonl(folder / "episode_outcomes.jsonl")
        metrics = {int(r["step"]):r for r in json.loads((folder / "metrics.json").read_text())}
        seeds = {r["reset_index"]:r["seeds"] for r in read_jsonl(folder / "training_seeds.jsonl")}
        artifacts[run_id] = {name:hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in
            ("manifest.json", "effective_config.json", "prompt_response.jsonl", "episode_outcomes.jsonl", "metrics.json", "training_seeds.jsonl")}
        for step in range(start, stop + 1):
            ends = {r["env_index"]:r for r in outcomes if r["reset_index"] == step - 1}
            assert len(ends) == 8
            low = max([r["step_index"] for r in outcomes if r["reset_index"] == step - 2],
                      default=min(r["step_index"] for r in raw) - 1) + 1
            lengths, rewards, actual = [], [], []
            for env in range(8):
                rows = [r for r in raw if r["env_idx"] == env and low <= r["step_index"] <= ends[env]["step_index"]]
                policy = [r for r in rows if r["policy_action_applied"]]
                assert len(policy) == ends[env]["policy_decision_count"]
                assert ends[env]["terminated"] and not ends[env]["truncated"]
                reward = sum(r["reward"] for r in rows)
                lengths.append(len(policy))
                rewards.append(reward)
                actual.extend(policy)
                if (step, env) in ((5, 0), (5, 4), (7, 0), (7, 6), (18, 0), (20, 0), (24, 0)):
                    examples.append(dict(global_step=step, run_id=run_id, env=env,
                        result=ends[env]["game_result"], environment_reward=reward,
                        decisions=len(policy), responses=[{k:r[k] for k in
                            ("source_line", "llm_response", "reward", "is_action_valid")} for r in policy],
                        second_prompt_sha256=hashlib.sha256(policy[1]["prompt"].encode()).hexdigest()))
            answers.extend(actual)
            m = metrics[step]
            assert abs(sum(rewards) / 8 - m["episode/reward/mean"]) < 0.001
            padded = math.ceil(len(actual) / 4) * 4
            batches.append(dict(global_step=step, game_seeds=seeds[step-1],
                wins=sum(r["game_result"] == "Victory" for r in ends.values()),
                decisions=len(actual), per_game_decisions=lengths, environment_rewards=rewards,
                invalid_decisions=sum(not r["is_action_valid"] for r in actual),
                copied_padding_count=padded-len(actual), inferred_optimizer_updates=math.ceil(padded / 8),
                logged_advantage_min=m["critic/advantages/min"], logged_advantage_max=m["critic/advantages/max"],
                logged_pg_loss=m["actor/pg_loss"], logged_grad_norm=m["actor/grad_norm"],
                logged_reference_kl=m["actor/kl_loss"], logged_entropy=m["actor/entropy_loss"]))
    supply_left = []
    for row in answers:
        obs = row["prompt"].split("Current observations are:\n", 1)[1].split("The remaining BO list", 1)[0]
        match = re.search(r"(\d+) remains unused", obs)
        assert match, "Observation lacks explicit remaining supply"
        supply_left.append(int(match[1]))
    return dict(scope="L7 global1–24 only; log statistics, not parameter-update reconstruction",
        games=192, wins=sum(r["wins"] for r in batches), policy_decisions=len(answers),
        zero_supply_queries=supply_left.count(0), minimum_observed_supply_left=min(supply_left),
        supply_left_counts=dict(Counter(supply_left)),
        execution_feedback_queries=sum("Execution feedback for the previous segment" in r["prompt"] for r in answers),
        inferred_optimizer_updates=sum(r["inferred_optimizer_updates"] for r in batches),
        logged_zero_advantage_batches=[r["global_step"] for r in batches if r["logged_advantage_min"] == r["logged_advantage_max"] == 0],
        batches=batches, examples=examples, source_artifact_sha256=artifacts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-run", type=Path, required=True)
    parser.add_argument("--resume-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.first_run, args.resume_run)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ("batches", "examples", "source_artifact_sha256", "supply_left_counts")}, ensure_ascii=False))
