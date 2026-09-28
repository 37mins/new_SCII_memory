"""Read saved answers with the frozen validator; no model, game or source edits.

Usage: PYTHONPATH=.:tests python audit_budget192_answers.py prompt_response.jsonl
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from unittest.mock import patch

from test_original_bo_segment import frozen_source
from test_r6_format_reward import planner_scope, strict_validator
from test_r6_natural_termination import ROOT, method

SHA = "71fae36ccc112d4c9cd303ab084c1e0a0c7eebec"


def audit(path):
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    with frozen_source(SHA):
        validate = strict_validator()
        scope = planner_scope()
        counts = method(ROOT / "agent_system/environments/strategic_sc2_env_manager_v2.py",
                        "StrategicSC2EnvironmentManagerV2", "_strict_legacy_bo_candidate_counts",
                        Any=object, Counter=Counter, re=re).__func__
    cases = []
    with patch.dict(os.environ, SC2_ORIGINAL_BO_SEGMENT="1"):
        for line_number, row in enumerate(rows, 1):
            if not row["policy_action_applied"]:
                continue
            prompt, answer = row["prompt"], row["llm_response"]
            valid, reason = validate(prompt, answer)
            assert (valid, reason) == (row["semantic_action_valid"], row["failure_reason"])
            allowed = counts(prompt)
            example = prompt.split("<answer>\n", 1)[1].split("</answer>", 1)[0].strip()
            assert validate(prompt, example)[0]
            selected = Counter()
            for line in answer.splitlines():
                if not line.startswith("- "):
                    continue
                item = line[2:].strip()
                match = re.fullmatch(r"(.+?)\s*x\s*(\d+)", item)
                name, count = (match[1].strip(), int(match[2])) if match else (item, 1)
                if name in allowed:
                    selected[name] += count
            excess = {k: dict(selected=v, allowed=allowed[k])
                      for k, v in selected.items() if v > allowed[k]}
            if valid:
                parsed = scope["parse_build_order"](answer, choice_map=dict(enumerate(allowed)),
                                                    strict_names_only=True, original_segment=True)
                assert Counter(parsed) == selected
            trimmed_valid, trimmed_reason = validate(prompt, "\n".join(answer.splitlines()[:-1]))
            cases.append(dict(log_line=line_number, env=row["env_idx"], id=row["policy_action_id"],
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), candidates=dict(allowed),
                valid=valid, reason=reason, valid_example=example,
                starts_with_example=re.sub(r"</?answer>", "", answer).strip().startswith(example),
                complete_action_excess=excess, raw_answer=answer,
                remove_last_line_only=dict(valid=trimmed_valid, reason=trimmed_reason)))
    return dict(source_sha=SHA, log_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                scope="CPU frozen validation and valid-answer parsing only; no model/game/update",
                summary=dict(consumed=len(cases), valid=sum(c["valid"] for c in cases),
                    validator_log_matches=len(cases), valid_examples=len(cases),
                    valid_parser_matches=sum(c["valid"] for c in cases),
                    starts_with_example=sum(c["starts_with_example"] for c in cases),
                    invalid_with_complete_action_excess=sum(not c["valid"] and bool(c["complete_action_excess"]) for c in cases),
                    single_build_candidate=sum(len([k for k in c["candidates"] if not k.startswith("Launch OFFENSE")]) == 1 for c in cases)),
                cases=cases)


if __name__ == "__main__":
    print(json.dumps(audit(Path(sys.argv[1])), indent=2, ensure_ascii=False))
