"""Real methods with fake socket/Ray boundaries; no model or game launch."""

import ast
from collections import Counter
import contextlib
import io
import json
import os
import re
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent_system.sc2_episode import gymnasium_transition
from test_r6_natural_termination import ARENA, ROOT, method


def planner_scope():
    path = ARENA / "agents/planner.py"
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    scope = dict(Counter=Counter, re=re, os=os, json=json)
    exec(compile(tree, str(path), "exec"), scope)
    return scope


def strict_validator():
    path = ROOT / "agent_system/environments/strategic_sc2_env_manager_v2.py"
    scope = dict(Any=object, Tuple=tuple, Counter=Counter, re=re, os=os)
    counts = method(path, "StrategicSC2EnvironmentManagerV2", "_strict_legacy_bo_candidate_counts", **scope)
    validate = method(path, "StrategicSC2EnvironmentManagerV2", "_validate_current_bo_response", **scope)
    manager = SimpleNamespace(_strict_legacy_plan_contract=True,
                              _bo_candidate_context=lambda prompt: ({}, Counter()),
                              _strict_legacy_bo_candidate_counts=counts.__func__)
    return lambda prompt, answer: validate(manager, prompt, answer)


class FormatRewardTest(unittest.TestCase):
    def test_existing_validator_checks_whole_answer_before_truncation(self):
        prompt = "The remaining BO list in this phase:\n- marine x 8\n- barracks x 1\nExample:"
        validate = strict_validator()
        with patch.dict(os.environ, SC2_PLAN_BULLET_MAX="6", SC2_PLAN_SEGMENT_MAX="6"):
            self.assertTrue(validate(prompt, "- marine\n- barracks")[0])
            for answer in ("- tank", "- marine x 2", "- marine\n- tank", "- marine\n" * 7):
                self.assertFalse(validate(prompt, answer)[0], answer)

    def test_manager_transports_existing_validation_without_altering_text(self):
        step = method(ROOT / "agent_system/environments/strategic_sc2_env_manager_v2.py",
                      "StrategicSC2EnvironmentManagerV2", "step", List=list, os=os, json=json,
                      to_numpy=lambda x: x, infer_failure_reason=lambda *args: "")
        for evaluable, valid, tool_invalid, projected in (
            (True, True, False, True), (True, False, False, True),
            (False, True, False, True), (True, True, True, True),
            (True, True, False, False),
        ):
            sent = []
            def env_step(actions):
                sent.extend(actions)
                return ["next"], [1.0], [False], [False], [dict(terminated=False, truncated=False)]
            manager = SimpleNamespace(
                _forbid_thinking_output=False,
                _apply_verified_candidate_override=lambda x: x,
                _sanitize_final_actions=lambda x: (x, [tool_invalid]),
                _last_prompts=["query"], _semantic_action_evaluable=lambda x: evaluable,
                _validate_current_bo_response=lambda *args: (valid, ""),
                projection_f=lambda x: (x, [projected]), envs=SimpleNamespace(step=env_step),
                _terminal_by_env=[False], _episode_step_by_env=[0],
                config=SimpleNamespace(env=SimpleNamespace(max_steps=0)),
                _extract_failure_reason=lambda x: "", pre_text_obs=["query"],
                use_tool_call_memory=False, _no_progress_counts=[0], memory=None,
                _write_prompt_response_log=lambda **kw: None,
                _write_tool_call_log=lambda **kw: None, _step_index=0,
                is_train=True, build_text_obs=lambda x: x,
            )
            step(manager, ["- marine"])
            self.assertEqual(sent, [dict(text="- marine", plan_format_valid=(
                evaluable and valid and not tool_invalid and projected))])

    def test_socket_forwards_validation_and_identical_action_text(self):
        path = ROOT / "agent_system/environments/env_package/sc2_online_restored/envSC2.py"
        step = method(path, "SC2Env", "step", gymnasium_transition=gymnasium_transition)
        for valid in (True, False):
            sent = []
            data = dict(obs="next", done=False, reward=1.0)
            env = SimpleNamespace(
                reset_failed=False, game_end_flag=False, _next_policy_action_id=0,
                pre_action_terminal_drain_timeout=0,
                _receive_pending_from_a=lambda **kw: None,
                _send_to_a=sent.append, _receive_from_a=lambda: data,
                _mark_terminal_if_done=lambda data: None,
                _policy_transition=lambda *args: args,
            )
            step(env, dict(text="- marine", plan_format_valid=valid))
            self.assertEqual(sent, [dict(msg="ACTION", value="- marine", policy_action_id=1,
                                         plan_format_valid=valid)])

    def test_raw_invalid_answer_does_not_gain_reward_after_rule_repair(self):
        scope = planner_scope()
        cls = scope["SC2Planner"]
        strategy = dict(phase_summary="test", timing="00:00-03:00",
                        instruction="Build marines", attack_instruction="No attacks in this phase")
        validate = strict_validator()
        with patch.dict(os.environ, SC2_PLANNER_PROTOCOL="legacy", SC2_PLAN_BULLET_MAX="6",
                        SC2_PLAN_SEGMENT_MAX="6", SC2_STEP_SHAPING_MODE="legacy"):
            for answer in ("- marine", "- tank", "- marine\n- tank", "- marine\n" * 7):
                planner = cls.__new__(cls)
                planner.ct = 0
                planner.correct_format = False
                planner.exceed = False
                planner.last_consumed_policy_action_id = None
                sent = []
                planner._send = sent.append
                planner._recv = lambda: dict(msg="ACTION", value=answer, policy_action_id=1,
                    plan_format_valid=validate(sent[-1]["obs"], answer)[0])
                planner._wait_terminal_ack = lambda data: None
                with contextlib.redirect_stdout(io.StringIO()):
                    plan = planner.generate_plan("observation", {"marine": 8}, strategy, 0)
                    expected = validate(sent[0]["obs"], answer)[0]
                    planner.game_end("observation", "Victory")
                self.assertTrue(plan)  # Existing repair still produces executable BO.
                self.assertEqual(planner.correct_format, expected, answer)
                self.assertEqual(sent[-1]["reward"], 20.5 if expected else 20.0, answer)

    def test_next_consumed_answer_replaces_previous_validation(self):
        get_action = method(ARENA / "agents/planner.py", "SC2Planner", "get_action",
                            os=os, final_state_shaping_reward=lambda obs: 0)
        planner = SimpleNamespace(correct_format=False, exceed=False,
                                  last_consumed_policy_action_id=None, _send=lambda data: None)
        for valid in (True, False, True):
            planner._recv = lambda: dict(msg="ACTION", value="unchanged", policy_action_id=1,
                                         plan_format_valid=valid)
            self.assertEqual(get_action(planner, "same query"), "unchanged")
            self.assertEqual(planner.correct_format, valid)

    def test_unvalidated_direct_text_cannot_earn_a_format_bonus(self):
        get_action = method(ARENA / "agents/planner.py", "SC2Planner", "get_action",
                            os=os, final_state_shaping_reward=lambda obs: 0)
        planner = SimpleNamespace(correct_format=True, exceed=False,
                                  last_consumed_policy_action_id=1, _send=lambda data: None,
                                  _recv=lambda: dict(msg="ACTION", value="- marine", policy_action_id=2))
        self.assertEqual(get_action(planner, "query"), "- marine")
        self.assertFalse(planner.correct_format)


if __name__ == "__main__":
    unittest.main()
