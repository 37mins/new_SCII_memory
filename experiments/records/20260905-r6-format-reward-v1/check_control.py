"""Read-only gate for the single approved r2 -> format-reward handoff."""

import argparse
import json
import math
from pathlib import Path

CONTROL_SHA = "76180970d2e139ef2da052df659546621a84539d"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_control(run):
    if not (run / ".done").exists():
        return {"status": "pending"}
    result = json.loads((run / "result.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    require(manifest["source_commit"] == CONTROL_SHA and not manifest["dirty"]
            and manifest["formal"], "wrong or dirty control source")
    require(result["status"] == "completed" and result["process_exit_code"] == 0
            and result["completed_steps"] == result["target_steps"] == 1
            and result["checkpoint_complete"] and result["metrics_finite"], "control did not complete")
    require(len(metrics) == 1 and metrics[0]["step"] == 1, "missing step-1 metrics")
    require(all(isinstance(v, (int, float)) and math.isfinite(v) for v in metrics[0].values()),
            "nonfinite or nonnumeric metric")
    actor = {key: metrics[0][key] for key in ("actor/grad_norm", "actor/ppo_kl", "actor/kl_loss")}
    outcomes = result["episode_outcomes"]["outcomes"]
    require(len(outcomes) == 8 and all(x["terminated"] and not x["truncated"] for x in outcomes),
            "expected eight natural games")
    require({x["env_index"] for x in outcomes} == set(range(8)), "duplicate/missing game outcome")
    active = [[] for _ in range(8)]
    ended = set()
    for line in (run / "prompt_response.jsonl").read_text().splitlines():
        row = json.loads(line)
        i = row["env_idx"]
        if i in ended:
            continue
        active[i].append(row)
        if row["done"]:
            ended.add(i)
    require(ended == set(range(8)), "prompt ledger is missing final outcomes")
    audits = []
    for line in (run / "console.log").read_text(errors="replace").splitlines():
        if "SC2 policy credit: " in line:
            audits.append(json.JSONDecoder().raw_decode(line.split("SC2 policy credit: ", 1)[1])[0])
    require(len(audits) == 1 and len(audits[0]) == 8, "missing or repeated policy audit")
    for i, rows in enumerate(active):
        consumed = [x for x in rows if x["policy_action_applied"]]
        skipped = [x for x in rows if not x["policy_action_applied"]]
        require(all(x["done"] for x in skipped), "unconsumed nonterminal reply")
        require([x["last_consumed_policy_action_id"] for x in consumed] == list(range(1, len(consumed)+1)),
                "consumption IDs are not unique/sequential")
        require(all(x["policy_action_id"] == x["last_consumed_policy_action_id"] for x in consumed),
                "sent/consumed action ID mismatch")
        audit = audits[0][i]
        require(audit["policy_decisions"] == len(consumed)
                and audit["excluded_terminal_rows"] == len(skipped)
                and audit["environment_interactions"] == len(rows), "collector/log row count mismatch")
        require(math.isclose(audit["environment_reward"], sum(x["reward"] for x in rows),
                             rel_tol=1e-6, abs_tol=1e-5), "environment reward was not conserved")
        outcome = next(x for x in outcomes if x["env_index"] == i)
        require(outcome["policy_decision_count"] == len(consumed), "outcome/collector count mismatch")
    return {"status": "passed", "natural_games": 8,
            "wins": sum(x["game_result"] == "Victory" for x in outcomes),
            "policy_decisions": sum(x["policy_decisions"] for x in audits[0]),
            "excluded_terminal_rows": sum(x["excluded_terminal_rows"] for x in audits[0]),
            "actor_metrics": actor,
            "scope": "completion and attribution gate, not learning quality or architecture proof"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    try:
        result = check_control(args.run)
    except (ValueError, KeyError, OSError) as error:
        print(json.dumps({"status": "blocked", "reason": str(error)}))
        raise SystemExit(2)
    print(json.dumps(result))
    raise SystemExit(0 if result["status"] == "passed" else 3)
