"""CPU: real strategist -> Planner/validator -> Commander queue, fake model only."""
import copy
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_planner_air_options import strategist, module as strategist_module
from test_original_bo_segment import pipeline
from test_r6_natural_termination import ROOT

RECORD = ROOT / "experiments/records/20260913-phase2-air-options-v1"
spec = importlib.util.spec_from_file_location("phase2_candidate", RECORD / "phase2_air_options.py")
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
OBS = (
    "It is now 04:09 in the game, and we are playing as Terran with 140 minerals "
    "and 0 vespene gas. In terms of our military forces, we have 11 SCV, 20 marine. "
    "In terms of the structures, we have 1 bases in the form of commandcenter, "
    "4 barracks, 4 supplydepot. As for the visible enemy presence, there are 1 SCV. "
    "As for the visible enemy structures, there are 2 starport, 2 starport with techlab."
)


def phase_and_plan(answer, enabled, obs=OBS):
    s = strategist(True)
    phase = s.generate_strategy(obs)
    if enabled:
        phase = candidate.extend_phase_two(s, obs, phase)
    with patch.dict(os.environ, SC2_CLARIFY_BO_QUOTA="1", SC2_EXECUTION_FAILURE_FEEDBACK="0"):
        result = pipeline(answer, True, obs, phase["bo todo"], phase, phase=2)
    return phase, result


class PhaseTwoAirOptionsTest(unittest.TestCase):
    def test_only_second_phase_instruction_and_options_change(self):
        s = strategist(True)
        original_template = copy.deepcopy(s.strategy)
        a = s.generate_strategy(OBS)
        saved = copy.deepcopy(a)
        b = candidate.extend_phase_two(s, OBS, a)
        self.assertEqual(a, saved)
        self.assertEqual(s.strategy, original_template)
        self.assertEqual({k for k in a if a[k] != b[k]}, {"instruction", "bo todo"})
        self.assertEqual(b["bo todo"], dict(a["bo todo"], starport=1, vikingfighter=8))
        for time in ("00:00", "10:11"):
            s = strategist(True)
            phase = s.generate_strategy(OBS.replace("04:09", time))
            self.assertIs(candidate.extend_phase_two(s, OBS, phase), phase)

    def test_missing_own_prerequisites_are_not_faked_by_enemy_starports(self):
        for answer, a_ok, b_ok in (("- refinery", True, True),
                                    ("- factory", False, True),
                                    ("- starport", False, False),
                                    ("- vikingfighter", False, False)):
            with self.subTest(answer=answer):
                a = phase_and_plan(answer, False)[1]
                b = phase_and_plan(answer, True)[1]
                self.assertEqual((a["valid"], b["valid"]), (a_ok, b_ok))
                if b_ok:
                    self.assertEqual([x["action"] for x in b["executor_queue"]], [answer[2:]])

    def test_existing_factory_then_starport_unlock_existing_filters(self):
        obs = OBS.replace("4 barracks,", "4 barracks, 1 refinery, 1 factory,")
        self.assertTrue(phase_and_plan("- starport", True, obs)[1]["valid"])
        self.assertFalse(phase_and_plan("- vikingfighter", True, obs)[1]["valid"])
        obs = obs.replace("1 factory,", "1 factory, 1 starport,")
        phase, out = phase_and_plan("- vikingfighter x 8", True, obs)
        self.assertNotIn("starport", phase["bo todo"])
        self.assertTrue(out["valid"])
        self.assertEqual([x["action"] for x in out["executor_queue"]], ["vikingfighter"] * 8)

    def test_valid_marine_plan_is_neither_forced_nor_reordered(self):
        answer = "- marine x 42\n- supplydepot x 8\n- barracks x 3\n- reactor barracks x 3\n- scv x 3\n- refinery"
        a, b = [phase_and_plan(answer, enabled)[1] for enabled in (False, True)]
        self.assertTrue(a["valid"] and b["valid"])
        self.assertEqual(a["executor_queue"], b["executor_queue"])
        self.assertEqual(len(b["executor_queue"]), 60)
        self.assertEqual(b["rule_added"], [])

    def test_factory_is_not_automatically_appended_to_valid_marine(self):
        out = phase_and_plan("- marine", True)[1]
        self.assertTrue(out["valid"])
        self.assertEqual([x["action"] for x in out["executor_queue"]], ["marine"])

    def test_whole_chain_in_one_answer_is_still_invalid(self):
        out = phase_and_plan("- refinery\n- factory\n- starport\n- vikingfighter x 8", True)[1]
        self.assertFalse(out["valid"])
        self.assertIn("starport", out["reason"])

    def test_late_starport_target_not_doubled_and_vikings_remain_stage_quota(self):
        s = strategist(True)
        candidate.extend_phase_two(s, OBS, s.generate_strategy(OBS))
        obs = OBS.replace("04:09", "10:11").replace("4 barracks,", "4 barracks, 1 factory, 1 starport,")
        late = s.generate_strategy(obs)
        original_late = strategist(True).generate_strategy(obs)
        self.assertEqual(late, original_late)
        self.assertNotIn("starport", late["bo todo"])
        # Early eight are additional optional production; no invented global
        # Viking budget or new carry-over/deduction mechanism is introduced.
        self.assertEqual(late["bo todo"]["vikingfighter"], 8)

    def test_inactive_hook_does_not_modify_default_template(self):
        self.assertNotIn("starport", strategist(True).strategy[1]["bo todo"])
        with patch.dict(os.environ, SC2_STRATEGY_PROFILE="anti_air", SC2_ENABLE_AIR_OPTIONS="1"):
            with self.assertRaises(ValueError):
                candidate.install()

    def test_explicit_hook_changes_only_isolated_strategy_instance(self):
        class IsolatedStrategist(strategist_module.SC2Strategist):
            pass

        module_name = "agents.commander"
        with patch.dict(sys.modules, {module_name: SimpleNamespace(SC2Strategist=IsolatedStrategist)}):
            with patch.dict(os.environ, SC2_STRATEGY_PROFILE="marine_pressure", SC2_ENABLE_AIR_OPTIONS="1"):
                candidate.install()
                obj = IsolatedStrategist()
                actual = obj.generate_strategy(OBS)
        expected, _ = phase_and_plan("- marine", True)
        self.assertEqual(actual, expected)
        self.assertNotIn("starport", strategist(True).generate_strategy(OBS)["bo todo"])


if __name__ == "__main__":
    unittest.main()
