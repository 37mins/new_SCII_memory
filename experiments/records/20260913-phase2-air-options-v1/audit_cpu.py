"""Read an A4 multiseed snapshot; exercise synthetic answers, no model/game.

PYTHONPATH=.:tests python -B experiments/records/20260913-phase2-air-options-v1/audit_cpu.py SNAPSHOT OUTPUT
Output contains counts/hashes only, not full internal prompts or raw model answers.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

from test_phase2_air_options import phase_and_plan


def audit(root):
    manifest = json.loads((root / "manifest.json").read_text())
    result = json.loads((root / "result.json").read_text())
    assert manifest["source_commit"] == "e8d8de164ed5312c33f9905613176503a9d1832c" and not manifest["dirty"]
    assert result == json.loads((root / ".done").read_text()) and result["process_exit_code"] == 0
    source = root / "prompt_response.jsonl"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == "6223ae35495666b2200483760c5322a73c8ea919a7e04e568161bdb5ba5fc6fb"
    rows = list(map(json.loads, source.read_text().splitlines()))
    report = dict(source_sha=manifest["source_commit"], prompt_response_sha256=digest,
                  model_generations=0, games_started=0, ppo_updates=0, cases=[])
    for env in (0, 2, 4, 3):
        line, row = next((i, r) for i, r in enumerate(rows, 1)
                         if r["env_idx"] == env and r["step_index"] == 1)
        obs = row["prompt"].split("Current observations are:\n", 1)[1].split(
            "The remaining BO list in this phase:", 1)[0].strip()
        for label, answer in (("historical_second_answer", row["llm_response"]),
                              ("synthetic_refinery", "- refinery"),
                              ("synthetic_factory", "- factory"),
                              ("synthetic_starport", "- starport"),
                              ("synthetic_viking", "- vikingfighter")):
            variants = []
            for enabled in (False, True):
                phase, out = phase_and_plan(answer, enabled, obs)
                queue = [x["action"] for x in out["executor_queue"]]
                variants.append(dict(arm="B" if enabled else "A", valid=out["valid"],
                    reason=out["reason"], parsed_count=len(out["parsed"]),
                    executor_queue_count=len(queue), executor_queue_counts=dict(Counter(queue)),
                    executor_queue_sha256=hashlib.sha256(json.dumps(queue).encode()).hexdigest(),
                    generated_query_sha256=hashlib.sha256(out["query"].encode()).hexdigest(),
                    raw_strategy_air={k: phase["bo todo"].get(k, 0) for k in ("starport", "vikingfighter")}))
            report["cases"].append(dict(env=env, source_line=line, answer_kind=label,
                observation_sha256=hashlib.sha256(obs.encode()).hexdigest(), variants=variants))
    return report


if __name__ == "__main__":
    output = Path(sys.argv[2])
    report = audit(Path(sys.argv[1]))
    with output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    for case in report["cases"]:
        print(case["env"], case["answer_kind"],
              [(v["arm"], v["valid"], v["executor_queue_count"]) for v in case["variants"]])
