"""Exercise the real strategist/Planner with a fake model socket, no SC2/GPU."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from test_r6_natural_termination import ARENA
from test_valid_bo_passthrough import replay


spec = importlib.util.spec_from_file_location("air_options_strategist", ARENA / "agents/strategist.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

OBS = ("It is now 10:11 in the game. We have 5 marine, 48 SCV. "
       "Our structures: 8 barracks, 5 barracks with techlab, 3 factory, "
       "2 refinery, 20 supplydepot. As for the visible enemy presence, "
       "there are 1 battlecruiser and 2 viking.")


def strategist(enabled, profile="marine_pressure"):
    with patch.dict(os.environ, SC2_ENABLE_AIR_OPTIONS=str(int(enabled)),
                    SC2_STRATEGY_PROFILE=profile):
        return module.SC2Strategist()


class PlannerAirOptionsTest(unittest.TestCase):
    def test_only_terminal_quota_is_extended(self):
        a, b = strategist(False).strategy, strategist(True).strategy
        self.assertEqual(a[:2], b[:2])
        self.assertEqual({k: v for k, v in a[-1].items() if k != "bo todo"},
                         {k: v for k, v in b[-1].items() if k != "bo todo"})
        self.assertEqual(b[-1]["bo todo"], dict(a[-1]["bo todo"], starport=1, vikingfighter=8))

    def test_default_off_and_other_profiles_are_unchanged(self):
        with patch.dict(os.environ, SC2_STRATEGY_PROFILE="marine_pressure"):
            os.environ.pop("SC2_ENABLE_AIR_OPTIONS", None)
            self.assertEqual(module.SC2Strategist().strategy, strategist(False).strategy)
        for profile in ("macro", "reference", "anti_air"):
            self.assertEqual(strategist(False, profile).strategy, strategist(True, profile).strategy)

    def test_existing_observed_starport_is_reconciled_without_using_enemy(self):
        no_own = strategist(True).generate_strategy(OBS + " Enemy has 2 starport.")
        self.assertEqual(no_own["bo todo"]["starport"], 1)
        own = strategist(True).generate_strategy(OBS.replace("3 factory", "3 factory, 1 starport"))
        self.assertNotIn("starport", own["bo todo"])
        self.assertEqual(own["bo todo"]["vikingfighter"], 8)

    def plan(self, raw, obs=OBS, enabled=True):
        s = strategist(enabled).generate_strategy(obs)
        return replay(raw, True, obs, s["bo todo"], s)

    def test_starport_can_be_selected_without_adding_production(self):
        self.assertFalse(self.plan("- starport", enabled=False)["valid"])
        b = self.plan("- starport")
        self.assertTrue(b["valid"])
        self.assertEqual(b["plan"], ["starport"])

    def test_viking_waits_for_existing_prerequisite_filter(self):
        self.assertFalse(self.plan("- vikingfighter")["valid"])
        obs = OBS.replace("3 factory", "3 factory, 1 starport")
        b = self.plan("- vikingfighter", obs)
        self.assertTrue(b["valid"])
        self.assertEqual(b["plan"], ["vikingfighter"])
        no_factory = OBS.replace("3 factory, ", "")
        self.assertFalse(self.plan("- starport", no_factory)["valid"])

    def test_valid_all_marine_is_not_overridden(self):
        b = self.plan("\n".join(["- marine"] * 6))
        self.assertTrue(b["valid"])
        self.assertEqual(b["plan"], ["marine"] * 6)

    def test_prompt_template_and_example_unchanged_for_same_observation(self):
        a, b = self.plan("- marine", enabled=False), self.plan("- marine")
        self.assertEqual(a["prompt"], b["prompt"].replace("- starport x 1\n", ""))

    def test_invalid_answer_can_still_trigger_rule_generated_air_actions(self):
        # Deliberately retained and user-approved confound, not model selection.
        b = self.plan("- siege tank")
        self.assertFalse(b["valid"])
        self.assertIn("starport", b["plan"])
        obs = OBS.replace("3 factory", "3 factory, 1 starport")
        b = self.plan("- siege tank", obs)
        self.assertFalse(b["valid"])
        self.assertIn("vikingfighter", b["plan"])


if __name__ == "__main__":
    unittest.main()
