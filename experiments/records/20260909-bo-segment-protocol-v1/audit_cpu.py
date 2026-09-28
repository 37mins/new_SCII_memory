"""Read-only protocol audit; AST-load real parsers/guard, never load a model or SC2.

Usage: PYTHONDONTWRITEBYTECODE=1 python audit_cpu.py SC2_online.zip prompt_response.jsonl
Prints JSON. Original get_bo is a parsing reference, NOT an implemented B pipeline.
"""
import ast
from collections import Counter
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from test_r6_format_reward import planner_scope, strict_validator
from test_r6_natural_termination import ARENA, method
from test_valid_bo_passthrough import CONTROLS

ARCHIVE_SHA = "e40fb7dad98f4c7d336031f3b2047ee3e15113ce14925830cfb6540aaa5888ab"
PLANNER_SHA = "a3668a9cee19ec2077158a5dd4a758955050b509467f671d5619c2379a91cf84"


def load_original(path):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ARCHIVE_SHA
    member = "SC2_online/SC2ArenaDyStrategy/agents/planner.py"
    with zipfile.ZipFile(path) as archive:
        source = archive.read(member)
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == "get_bo")
    scope = {"re": re}
    exec(compile(ast.Module(body=[node], type_ignores=[]), member, "exec"), scope)
    return scope["get_bo"], hashlib.sha256(source).hexdigest()


def input_context(row):
    prompt = row["prompt"]
    stage = prompt.split("- Current Stage: ", 1)[1].splitlines()[0]
    strategy = dict(phase_summary=stage, timing="",  # Only concatenated for prompt display.
                    instruction=prompt.split("- Detailed Instruction: ", 1)[1].splitlines()[0],
                    attack_instruction=prompt.split("- Attacking Instruction: ", 1)[1].splitlines()[0])
    obs = prompt.split("Current observations are:\n", 1)[1].split(
        "The remaining BO list in this phase:", 1)[0].rstrip()
    scope = {"Any": object, "Counter": Counter, "re": re}
    count = method(ROOT / "agent_system/environments/strategic_sc2_env_manager_v2.py",
                   "StrategicSC2EnvironmentManagerV2", "_strict_legacy_bo_candidate_counts", **scope)
    todo = {k: v for k, v in count.__func__(prompt).items() if not k.startswith("Launch OFFENSE")}
    return obs, todo, strategy


def opening_guard(plan, obs, todo, phase):
    guard = method(ARENA / "agents/commander.py", "SC2Commander", "ensure_opening_marine",
                   normalize_bo_name=lambda s: str(s).strip().lower(),
                   read_int_env=lambda key, default: int(os.getenv(key, default)))
    commander = SimpleNamespace(strategy_renew_ct=phase, waiting_bo=todo, current_bo=list(plan))
    guard(commander, obs)
    return commander.current_bo


def current_pipeline(answer, row, phase):
    scope = planner_scope()
    obs, todo, strategy = input_context(row)
    trace = {}
    parse = scope["parse_build_order"]

    def tracked_parse(text, **kwargs):
        trace["after_line_truncation"] = text
        trace["parsed_before_line_truncation"] = parse(answer, **kwargs)
        trace["parsed_after_line_truncation"] = parse(text, **kwargs)
        return trace["parsed_after_line_truncation"]

    scope["parse_build_order"] = tracked_parse
    planner = scope["SC2Planner"].__new__(scope["SC2Planner"])
    planner.ct = 0
    planner.correct_format = planner.exceed = False
    planner.last_consumed_policy_action_id = None
    sent = []
    planner._send = sent.append

    def receive():
        trace["valid"], trace["reason"] = strict_validator()(sent[-1]["obs"], answer)
        return dict(msg="ACTION", value=answer, policy_action_id=1,
                    plan_format_valid=trace["valid"])

    planner._recv = receive
    with patch.dict(os.environ, dict(CONTROLS, SC2_PRESERVE_VALID_BO="1",
                                    SC2_ENABLE_AIR_OPTIONS="1")):
        plan = planner.generate_plan(obs, todo, strategy, 0, attack_ct=0)
        trace["planner_queue"] = plan
        trace["executor_action_queue"] = opening_guard(plan, obs, todo, phase)
    trace["line_limit_exceeded"] = planner.exceed
    trace["executor_scope"] = "CPU action queue only; no engine command submitted"
    return trace


def main():
    archive, log = map(Path, sys.argv[1:])
    assert hashlib.sha256((ARENA / "agents/planner.py").read_bytes()).hexdigest() == PLANNER_SHA
    assert hashlib.sha256((ARENA / "agents/commander.py").read_bytes()).hexdigest() == "565dec300e7f9e8d3c937a9931e74953df75a41d29211485131cad60a1aeb1b1"
    assert hashlib.sha256((ROOT / "agent_system/environments/strategic_sc2_env_manager_v2.py").read_bytes()).hexdigest() == "e1cc55a8daa298656b545942fa0964c737ea0d9101dc77410b5258ccefca5ff2"
    original, original_sha = load_original(archive)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    raw, later = rows[37], rows[45]
    assert raw["env_idx"] == later["env_idx"] == 5
    assert raw["policy_action_applied"] and later["policy_action_applied"]
    assert original(raw["llm_response"]) == ["marine"] * 29
    controls = [
        ("actual_29_marine_remainder8", raw["llm_response"], raw, 1, "unaltered log line38"),
        ("actual_six_actions", later["llm_response"], later, 2, "unaltered log line46"),
        ("derived_seven_marine", "\n".join(["- marine"] * 7), later, 2, "first7 actions of line38; line46 context, remainder42"),
        ("derived_fifteen_marine", "\n".join(["- marine"] * 15), later, 2, "first15 actions of line38; line46 context, remainder42"),
        ("derived_count29", "- marine x 29", later, 2, "lossless count encoding of line38; line46 context, remainder42"),
        ("derived_mixed_18_actions", "- supplydepot x 2\n- barracks\n- marine x 15", later, 2, "synthetic count boundary, real line46 candidates"),
        ("derived_bad_name", "- barrack\n- marine x 2", later, 2, "synthetic exact-name negative control"),
    ]
    output = dict(status="boundary-conflict; B not implemented", baseline_sha="0d7ba09ddbdf94d85d67b553bf7978f402cf2052",
                  archive_sha256=ARCHIVE_SHA, archive_planner_sha256=original_sha,
                  log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                  cases=[])
    for name, answer, row, phase, origin in controls:
        debug = io.StringIO()
        with contextlib.redirect_stdout(debug):
            current = current_pipeline(answer, row, phase)
        parsed = original(answer)
        _, todo, _ = input_context(row)
        selected = Counter(parsed)
        violations = {k: dict(selected=n, allowed=todo.get(k, 0))
                      for k, n in selected.items() if n > todo.get(k, 0)}
        output["cases"].append(dict(name=name, origin=origin, raw_answer=answer,
                                    A=current, original_parser_result=parsed,
                                    frozen_candidate_violations=violations,
                                    B_executor_queue=None, debug=debug.getvalue()))
    cases = {case["name"]: case for case in output["cases"]}
    assert cases["actual_six_actions"]["A"]["valid"]
    for name, length in (("derived_seven_marine", 7), ("derived_fifteen_marine", 15),
                         ("derived_count29", 29), ("derived_mixed_18_actions", 18)):
        assert not cases[name]["A"]["valid"]
        assert len(cases[name]["original_parser_result"]) == length
        assert not cases[name]["frozen_candidate_violations"]
    assert cases["actual_29_marine_remainder8"]["frozen_candidate_violations"] == {
        "marine": {"selected": 29, "allowed": 8}}
    guard_input = ["supplydepot"] * 3 + ["barracks"] * 4
    debug = io.StringIO()
    with patch.dict(os.environ, SC2_PLAN_SEGMENT_MAX="6"), contextlib.redirect_stdout(debug):
        guarded = opening_guard(guard_input, "1 barracks", {"marine": 8}, 1)
    assert len(guarded) == 6 and guarded[-1] == "marine"
    output["guard_conflict"] = dict(origin="synthetic legal-name boundary; not a generated answer",
                                    before=guard_input, after=guarded,
                                    removed=2, rule_added=["marine"], debug=debug.getvalue())
    output["assertions_passed"] = True
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
