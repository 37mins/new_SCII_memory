"""Counterfactual BO conversion of existing answers, not new game trajectories."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from test_valid_bo_passthrough import replay


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(folder):
    rows = [json.loads(s) for s in (folder / "prompt_response.jsonl").read_text().splitlines()]
    result = json.loads((folder / "result.json").read_text())
    assert digest(folder / "console.log") == result["console_sha256"]
    lines = (folder / "console.log").read_text().splitlines()
    final_lines = [(i + 1, s) for i, s in enumerate(lines) if "BO ori:" in s and ", BO:" in s]
    assert len(rows) == len(final_lines)
    attack_ct = 0
    records = []
    for row, (line_number, line) in zip(rows, final_lines):
        prompt, raw = row["prompt"], row["llm_response"]
        logged_raw, logged_plan = line.split("BO ori:", 1)[1].split(", BO:", 1)
        assert raw.replace("\n", ",").strip().rstrip(",") == logged_raw.strip().rstrip(",")
        logged_plan = ast.literal_eval(logged_plan.strip())
        expected = ["Attacking Instruction: " + s if s.startswith("Launch OFFENSE") else s for s in logged_plan]
        phase, timing = re.search(r"Current Stage: (.*?) (\(.*\))", prompt).groups()
        strategy = dict(phase_summary=phase, timing=timing,
                        instruction=re.search(r"- Detailed Instruction: (.*)", prompt)[1],
                        attack_instruction=re.search(r"- Attacking Instruction: (.*)", prompt)[1])
        obs = prompt.split("Current observations are:\n", 1)[1].split("\nThe remaining BO list", 1)[0]
        todo_section = prompt.split("The remaining BO list in this phase:\n", 1)[1].split("Use only exact", 1)[0]
        todo = {}
        for bullet in todo_section.splitlines():
            match = re.fullmatch(r"- (.+?) x (\d+)", bullet)
            if match:
                todo[match[1]] = int(match[2])
        a = replay(raw, False, obs, todo, strategy, attack_ct)
        b = replay(raw, True, obs, todo, strategy, attack_ct)
        assert a["prompt"] == b["prompt"] == prompt, (folder.name, row["step_index"], "query mismatch")
        assert a["plan"] == expected, (folder.name, row["step_index"], a["plan"], expected)
        assert a["valid"] == b["valid"] == row["semantic_action_valid"]
        assert a["reason"] == b["reason"] == row["failure_reason"]
        assert a["exceed"] == b["exceed"]
        # Attack positioning/prefix conversion remain outside this ablation.
        builds = lambda plan: [s for s in plan if not s.startswith("Attacking Instruction:")]
        actions = [s.strip()[2:].strip() for s in raw.splitlines() if s.strip().startswith("- ")]
        raw_builds = [s for s in actions if not s.startswith("Launch OFFENSE")]
        if a["valid"]:
            assert builds(b["plan"]) == raw_builds
        else:
            assert a["plan"] == b["plan"]
        records.append(dict(step_index=row["step_index"], time=re.search(r"It is now (\d+:\d+)", obs)[1],
                            valid=a["valid"], reason=a["reason"], prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                            raw_answer=raw, old_plan=a["plan"], candidate_plan=b["plan"],
                            console_line=line_number, changed=a["plan"] != b["plan"]))
        attack_ct += any(s.startswith("Attacking Instruction:") for s in expected)
    valid = [r for r in records if r["valid"]]
    return dict(input_dir=str(folder), hashes={n: digest(folder / n) for n in (
        "prompt_response.jsonl", "console.log", "result.json", "effective_config.json")},
        summary=dict(answers=len(records), valid=len(valid), invalid=len(records)-len(valid),
                     baseline_reproduced=len(records), unchanged_queries=len(records),
                     valid_build_sequences_preserved=len(valid),
                     changed_valid=sum(r["changed"] for r in valid), invalid_unchanged=len(records)-len(valid),
                     old_valid_build_counts=dict(Counter(s for r in valid for s in r["old_plan"] if not s.startswith("Attacking Instruction:"))),
                     candidate_valid_build_counts=dict(Counter(s for r in valid for s in r["candidate_plan"] if not s.startswith("Attacking Instruction:")))),
        records=records)


if __name__ == "__main__":
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if dirty:
        raise SystemExit("Run the recorded A/B from a committed clean candidate")
    print(json.dumps(dict(source_sha=sha, mode="offline-counterfactual-no-game-no-model",
                         reports=[audit(Path(s)) for s in sys.argv[1:]]), ensure_ascii=False, indent=2))
