"""Use real Planner/get_action/validator with the socket replaced; no SC2/model."""
from collections import Counter
import contextlib
import io
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_r6_format_reward import planner_scope, strict_validator
from test_r6_natural_termination import ARENA, method

CONTROLS = dict(SC2_PLANNER_PROTOCOL="legacy", SC2_PLAN_SEGMENT_MAX="6",
                SC2_PLAN_BULLET_MAX="6", SC2_MAX_FORCED_ATTACKS="0",
                SC2_ATTACK_OPTION_ACCOUNT_ISSUED="1",
                SC2_ATTACK_ORDER_POSITION="after_build", SC2_STEP_SHAPING_MODE="off")


def replay(answer, enabled, obs, todo, strategy, attack_ct=0, controls=None):
    scope = planner_scope()
    planner = scope["SC2Planner"].__new__(scope["SC2Planner"])
    planner.ct = 0
    planner.correct_format = False
    planner.exceed = False
    planner.last_consumed_policy_action_id = None
    sent = []
    planner._send = sent.append
    verdict = []

    def receive():
        valid = strict_validator()(sent[-1]["obs"], answer)
        verdict.append(valid)
        return dict(msg="ACTION", value=answer, policy_action_id=1,
                    plan_format_valid=valid[0])

    planner._recv = receive
    env = dict(CONTROLS, SC2_PRESERVE_VALID_BO=str(int(enabled)))
    env.update(controls or {})
    with patch.dict(os.environ, env), contextlib.redirect_stdout(io.StringIO()):
        plan = planner.generate_plan(obs, todo, strategy, 0, attack_ct=attack_ct)
    return dict(prompt=sent[0]["obs"], plan=plan, valid=verdict[0][0],
                reason=verdict[0][1], correct_format=planner.correct_format,
                exceed=planner.exceed)


STRATEGY = dict(phase_summary="offline test", timing="06:07 onwards",
                instruction="Produce the listed units.",
                attack_instruction="Launch game-ending attacks.")
OBS = "It is now 14:13 in the game. We have 1 factory, 1 refinery and 1 barracks."
TODO = dict(marine=35, mule=1, siegetank=3)


class ValidBoPassthroughTest(unittest.TestCase):
    def test_six_marines_are_not_replaced_with_tanks(self):
        answer = "\n".join(["- marine"] * 6)
        a = replay(answer, False, OBS, TODO, STRATEGY)
        b = replay(answer, True, OBS, TODO, STRATEGY)
        self.assertEqual(a["plan"], ["marine"] * 3 + ["siegetank"] * 2 + ["mule"])
        self.assertEqual(b["plan"], ["marine"] * 6)
        self.assertEqual(a["prompt"], b["prompt"])
        self.assertTrue(a["valid"] and b["valid"])

    def test_valid_short_answer_is_not_padded(self):
        self.assertEqual(replay("- marine", True, OBS, TODO, STRATEGY)["plan"], ["marine"])

    def test_valid_build_order_and_counts_preserved(self):
        answer = "<answer>\nHere are the Build Orders with attack instructions:\n- siegetank\n- marine\n- mule\n- marine\n</answer>"
        self.assertEqual(replay(answer, True, OBS, TODO, STRATEGY)["plan"],
                         ["siegetank", "marine", "mule", "marine"])

    def test_invalid_answers_keep_original_fallback(self):
        for answer in ("", "- tank", "- marine x 2", "- marine\n- tank",
                       "- marine\n" * 7, "- mule\n- mule", "explanation\n- marine"):
            with self.subTest(answer=answer):
                a = replay(answer, False, OBS, TODO, STRATEGY)
                b = replay(answer, True, OBS, TODO, STRATEGY)
                self.assertFalse(a["valid"])
                self.assertEqual(a, b)

    def test_attack_remains_after_build_actions(self):
        obs = OBS + " Supply army: 20"
        answer = "- Launch OFFENSE with 8 marine\n- marine\n- siegetank"
        b = replay(answer, True, obs, TODO, STRATEGY)
        self.assertTrue(b["valid"])
        self.assertEqual(b["plan"], ["marine", "siegetank",
                                    "Attacking Instruction: Launch OFFENSE with 8 marine"])

    def test_numbered_mode_is_unchanged(self):
        a = replay("- marine", False, OBS, TODO, STRATEGY, controls={"SC2_PLANNER_PROTOCOL": "numbered"})
        b = replay("- marine", True, OBS, TODO, STRATEGY, controls={"SC2_PLANNER_PROTOCOL": "numbered"})
        self.assertEqual(a, b)

    def test_commander_opening_guard_is_not_silently_removed(self):
        scope = planner_scope()
        guard = method(ARENA / "agents/commander.py", "SC2Commander", "ensure_opening_marine",
                       normalize_bo_name=lambda s: s.lower(), read_int_env=lambda key, default: 6, os=os)
        plan = replay("- siegetank", True, OBS, TODO, STRATEGY)["plan"]
        commander = SimpleNamespace(strategy_renew_ct=1, waiting_bo={"marine": 35}, current_bo=plan[:])
        with contextlib.redirect_stdout(io.StringIO()):
            guard(commander, OBS)
        self.assertEqual(plan, ["siegetank"])
        self.assertEqual(commander.current_bo, ["siegetank", "marine"])


if __name__ == "__main__":
    unittest.main()
