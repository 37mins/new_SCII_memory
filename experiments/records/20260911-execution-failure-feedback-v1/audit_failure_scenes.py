"""Screen retained natural evaluations, not new games or causal reward estimates.

python audit_failure_scenes.py --output <unused JSON path>
Input locations and hashes come from learning-eval-analysis.json in this directory.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


MARKER = "Execution feedback for the previous segment (not owned units/buildings):\n"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(reason):
    if "Supply blocked" in reason:
        return "supply_blocked"
    if "Vespene" in reason:
        return "vespene"
    return "other"


def consumed(rows, outcome):
    return [(line, row) for line, row in enumerate(rows, 1)
            if row["env_idx"] == outcome["env_index"]
            and row["step_index"] <= outcome["step_index"]
            and row["policy_action_applied"]]


def audit():
    authority = Path(__file__).with_name("learning-eval-analysis.json")
    source = json.loads(authority.read_text())
    root = Path(source["snapshot"])
    games, verified = [], 0
    for arm, group in source["groups"].items():
        folder = root / arm
        for name, expected in group["input_hashes"].items():
            assert digest(folder / name) == expected, (arm, name)
            verified += 1
        manifest = json.loads((folder / "manifest.json").read_text())
        result = json.loads((folder / "result.json").read_text())
        load = json.loads((folder / "checkpoint_evaluation.json").read_text())
        assert manifest["source_commit"] == source["runtime_sha"] and not manifest["dirty"]
        assert result == json.loads((folder / ".done").read_text())
        assert result["process_exit_code"] == result["training_updates"] == load["optimizer_updates"] == 0
        assert load["loaded_global_step"] == (0 if arm == "original" else 4)
        rows = [json.loads(line) for line in (folder / "prompt_response.jsonl").read_text().splitlines()]
        outcomes = [json.loads(line) for line in (folder / "episode_outcomes.jsonl").read_text().splitlines()]
        assert len(outcomes) == len({o["env_index"] for o in outcomes}) == 8
        for outcome in sorted(outcomes, key=lambda o: o["env_index"]):
            env = outcome["env_index"]
            old = next(e for e in group["episodes"] if e["env"] == env)
            assert outcome["terminated"] and not outcome["truncated"]
            active = consumed(rows, outcome)
            assert len(active) == outcome["policy_decision_count"] == old["calls"]
            reward = sum(float(r["reward"]) for r in rows if r["env_idx"] == env
                         and r["step_index"] <= outcome["step_index"])
            assert abs(reward - old["reward"]) < 1e-9
            assert old["seed"] == load["game_seeds"][env]
            feedback = []
            for line, row in active:
                if MARKER not in row["prompt"]:
                    continue
                value = json.JSONDecoder().raw_decode(row["prompt"].split(MARKER, 1)[1])[0]
                reason = value["failed"]["reason"]
                feedback.append(dict(raw_line=line, game_time=re.search(
                    r"It is now (\d+:\d+)", row["prompt"]).group(1),
                    category=classify(reason), failed=value["failed"],
                    prompt_sha256=hashlib.sha256(row["prompt"].encode()).hexdigest(),
                    answer=row["llm_response"], valid=row["is_action_valid"]))
            line, second = active[1]
            assert line == old["second_answer"]["raw_line"]
            assert second["llm_response"] == old["second_answer"]["answer"]
            assert hashlib.sha256(second["prompt"].encode()).hexdigest() == old["second_answer"]["prompt_sha256"]
            marine = sum(int(n or 1) for n in re.findall(
                r"(?m)^- marine(?: x (\d+))?\s*$", second["llm_response"]))
            games.append(dict(arm=arm, env=env, seed=old["seed"], result=outcome["game_result"],
                reward=reward, calls=len(active), feedback=feedback,
                second_answer={**old["second_answer"], "marine_count": marine}))
    pairs = []
    for a in (g for g in games if g["arm"] == "A4"):
        b = next(g for g in games if g["arm"] == "B4" and g["seed"] == a["seed"])
        assert a["second_answer"]["prompt_sha256"] == b["second_answer"]["prompt_sha256"]
        pairs.append(dict(seed=a["seed"], prompt_sha256=a["second_answer"]["prompt_sha256"],
            A4_marine=a["second_answer"]["marine_count"], B4_marine=b["second_answer"]["marine_count"],
            A4_result=a["result"], B4_result=b["result"]))
    counts = Counter(f'{g["arm"]}/{g["result"]}/{f["category"]}' for g in games for f in g["feedback"])
    return dict(scope="CPU only; recorded feedback, not all possible blocking events",
        runtime_sha=source["runtime_sha"], authority_sha256=digest(authority),
        input_snapshot=str(root), verified_input_files=verified, games=len(games),
        defeats=sum(g["result"] == "Defeat" for g in games),
        feedback_counts=dict(sorted(counts.items())), paired_second_answers=pairs, episodes=games)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit()
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, ensure_ascii=False, indent=2))
