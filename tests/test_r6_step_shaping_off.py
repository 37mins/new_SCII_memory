import ast
import contextlib
import hashlib
import io
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_r6_natural_termination import ARENA, method


class StepShapingOffTest(unittest.TestCase):
    def rewards(self, mode, potentials, incoming=0.0, done=False):
        values = iter(potentials)
        get_action = method(
            ARENA / "agents/planner.py", "SC2Planner", "get_action",
            os=os, final_state_shaping_reward=lambda obs: next(values),
        )
        sent = []
        planner = SimpleNamespace(
            ct=25, correct_format=False, exceed=False,
            last_consumed_policy_action_id=None,
            _send=sent.append, _recv=lambda: {"msg": "ACTION", "value": "- marine", "policy_action_id": 1},
        )
        with patch.dict(os.environ, SC2_STEP_SHAPING_MODE=mode), contextlib.redirect_stdout(io.StringIO()):
            for _ in potentials:
                answer = get_action(planner, "same observation", reward=incoming, done=done)
                self.assertEqual(answer, "- marine")
        return sent

    def test_reproduces_legacy_positive_reward_without_progress(self):
        rows = self.rewards("legacy", [2.0] * 20)
        self.assertEqual([row["reward"] for row in rows], [0.5] * 20)
        self.assertEqual(sum(row["reward"] for row in rows), 10.0)

    def test_off_removes_only_the_state_bonus(self):
        for incoming in (0.0, -0.1, 0.5):
            rows = self.rewards("off", [2.0] * 20, incoming=incoming)
            self.assertEqual([row["reward"] for row in rows], [incoming] * 20)
            self.assertTrue(all(not row["done"] for row in rows))

    def test_legacy_formula_and_terminal_transport_are_unchanged(self):
        rows = self.rewards("legacy", [0.0, 1.0, 2.0])
        self.assertEqual([row["reward"] for row in rows], [0.0, 0.25, 0.5])
        self.assertEqual(self.rewards("legacy", [2.0], incoming=-50, done=True),
                         self.rewards("off", [2.0], incoming=-50, done=True))

    def test_game_end_reward_is_identical_for_both_arms(self):
        game_end = method(
            ARENA / "agents/planner.py", "SC2Planner", "game_end",
            final_state_shaping_reward=lambda obs: 1.25,
        )
        for result, expected in (("Victory", 20.5), ("Defeat", -48.25), ("Tie", -48.25)):
            payloads = []
            for mode in ("legacy", "off"):
                planner = SimpleNamespace(correct_format=True, exceed=False,
                                          last_consumed_policy_action_id=None,
                                          _send=payloads.append, _wait_terminal_ack=lambda data: None)
                with patch.dict(os.environ, SC2_STEP_SHAPING_MODE=mode), contextlib.redirect_stdout(io.StringIO()):
                    game_end(planner, "final observation", result)
            self.assertEqual(payloads[0], payloads[1])
            self.assertEqual(payloads[0]["reward"], expected)

    def test_all_other_planner_code_matches_validated_8ac1e19(self):
        source = (ARENA / "agents/planner.py").read_text()
        tree = ast.parse(source)
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SC2Planner")
        node = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "get_action")
        lines = source.splitlines(keepends=True)
        frozen_source = "".join(lines[:node.lineno - 1] + lines[node.end_lineno:])
        # This candidate adds transport metadata, not reward/prompt/parser behavior.
        frozen_source = frozen_source.replace("        self.last_consumed_policy_action_id = None\n", "")
        frozen_source = frozen_source.replace(
            '        terminal_data["last_consumed_policy_action_id"] = self.last_consumed_policy_action_id\n', "")
        # Only the separately authorized format-verdict boundary changes here.
        frozen_source = frozen_source.replace(
            '        print(f"Plan format reward: {0.5 if self.correct_format else 0.0:.1f}")\n', "")
        frozen_source = frozen_source.replace(
            "        self.exceed = exceed\n", "        self.exceed = exceed\n        self.correct_format = True\n")
        self.assertEqual(hashlib.sha256(frozen_source.encode()).hexdigest(),
                         "0cad37132557cb87b1ba72d8792ffcc087310f0ae79b96cf28a9347c0d914694")


if __name__ == "__main__":
    unittest.main()
