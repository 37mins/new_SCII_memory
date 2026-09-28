"""Replay logged verdicts and compare old/new Planner on identical CPU fixtures."""

import argparse
from collections import Counter
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from agent_system.environments.strategic_sc2_env_manager_v2 import StrategicSC2EnvironmentManagerV2


def load_planner(root, name):
    path = root / "agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/agents/planner.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SC2Planner


def make_planner(cls, answer, verdict):
    planner = cls.__new__(cls)
    planner.ct = 0
    planner.correct_format = False
    planner.exceed = False
    planner.last_consumed_policy_action_id = None
    sent = []
    planner._send = sent.append
    planner._recv = lambda: dict(msg="ACTION", value=answer, policy_action_id=1,
                                 plan_format_valid=verdict(sent[-1]["obs"], answer))
    planner._wait_terminal_ack = lambda data: None
    return planner, sent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-source", type=Path, required=True)
    parser.add_argument("--prompt-log", type=Path, required=True)
    args = parser.parse_args()
    old = load_planner(args.control_source, "control_planner")
    new = load_planner(ROOT, "candidate_planner")
    manager = StrategicSC2EnvironmentManagerV2.__new__(StrategicSC2EnvironmentManagerV2)
    manager._strict_legacy_plan_contract = True
    os.environ.update(SC2_PLANNER_PROTOCOL="legacy", SC2_PLAN_BULLET_MAX="6",
                      SC2_PLAN_SEGMENT_MAX="6", SC2_STEP_SHAPING_MODE="legacy")
    ended, active = set(), []
    for line in args.prompt_log.read_text().splitlines():
        row = json.loads(line)
        key = row["env_idx"]  # This gate has one reset/batch only.
        if key in ended:
            continue
        active.append(row)
        if row["done"]:
            ended.add(key)
    real = [row for row in active if row["policy_action_applied"]]
    verdict = lambda prompt, answer: manager._validate_current_bo_response(prompt, answer)[0]
    strategy = dict(phase_summary="paired fixture", timing="00:00-03:00",
                    instruction="Build the listed units", attack_instruction="No attacks in this phase")
    todo = dict(marine=30, supplydepot=6, siegetank=6, marauder=6)
    bonus_deltas = []
    with contextlib.redirect_stdout(io.StringIO()):
        for row in real:
            results = []
            for cls in (old, new):
                planner, sent = make_planner(cls, row["llm_response"], verdict)
                plan = planner.generate_plan("same observation", todo.copy(), strategy, 0)
                results.append((sent[0]["obs"], plan))
            assert results[0] == results[1], "Query or executable BO changed on identical inputs"
            if row["done"]:
                # Logged prompts may be truncated. Reuse the runtime's full-input
                # verdict, rather than pretending to revalidate the truncated log.
                valid = bool(row["semantic_action_evaluable"] and row["semantic_action_valid"]
                             and row["is_action_valid"])
                planner, sent = make_planner(new, row["llm_response"], lambda *args: valid)
                planner.get_action(row["prompt"])
                assert row["won"] == 1.0, "This saved control's real terminal actions were all wins"
                planner.game_end("terminal observation", "Victory")
                expected = 20.5 if valid else 20.0
                assert sent[-1]["reward"] == expected
                bonus_deltas.append(dict(env_idx=row["env_idx"], valid=valid,
                    old_reward=row["reward"], corrected_reward=expected,
                    delta=expected - row["reward"]))
    print(json.dumps(dict(
        prompt_log_sha256=hashlib.sha256(args.prompt_log.read_bytes()).hexdigest(),
        active_environment_rows=len(active), consumed_rows=len(real),
        excluded_terminal_rows=len(active)-len(real),
        fully_valid_rows=sum(row["is_action_valid"] for row in real),
        paired_identical_query_and_plan_cases=len(real),
        terminal_bonus_pairs=bonus_deltas,
        total_terminal_reward_delta=sum(row["delta"] for row in bonus_deltas),
        errors=dict(Counter(row["failure_reason"] for row in real if not row["is_action_valid"])),
        scope="CPU replay/fixtures, not new games or a PPO update",
    ), indent=2))


if __name__ == "__main__":
    main()
