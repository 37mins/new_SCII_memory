"""The stopped opening run's empty input, with real Planner/validator and no GPU."""
import json
import unittest

from test_r6_format_reward import planner_scope
from test_r6_natural_termination import ROOT
from test_planner_air_options import strategist
from test_valid_bo_passthrough import replay


EVIDENCE = ROOT / "experiments/records/20260908-g2-planner-handoff-v1/opening-empty-candidates-audit.json"
PROMPT = json.loads(EVIDENCE.read_text())["first_empty"]["prompt"]
OBS = PROMPT.split("Current observations are:\n", 1)[1].split(
    "The remaining BO list in this phase:", 1
)[0].rstrip("\n")
TODO = dict(siegetank=9, starport=1, vikingfighter=8)
STRATEGY = strategist(True).generate_strategy(OBS)


class FactoryPrerequisiteCandidateTest(unittest.TestCase):
    def setUp(self):
        self.scope = planner_scope()
        self.filter = self.scope["filter_bo_todo_for_prereqs"]

    def test_actual_empty_input_exposes_only_factory_without_mutating_todo(self):
        todo = dict(TODO)
        self.assertEqual(self.filter(todo, OBS), {"factory": 1})
        self.assertEqual(todo, TODO)

    def test_factory_answer_is_accepted_and_not_padded_with_units(self):
        result = replay("- factory", True, OBS, TODO, STRATEGY, attack_ct=2)
        self.assertTrue(result["valid"], result["reason"])
        self.assertEqual(result["plan"], ["factory"])
        self.assertIn("The remaining BO list in this phase:\n- factory x 1", result["prompt"])
        self.assertNotIn("in this phase:\nnone", result["prompt"])
        self.assertIn("Here are the Build Orders with attack instructions:\n- factory", result["prompt"])

    def test_each_hidden_tech_target_has_a_factory_candidate(self):
        for target in TODO:
            with self.subTest(target=target):
                self.assertEqual(self.filter({target: 1}, OBS), {"factory": 1})

    def test_existing_factory_quota_is_not_overwritten(self):
        self.assertEqual(self.filter(dict(TODO, factory=3), OBS), {"factory": 3})

    def test_observed_factory_preserves_previous_candidates(self):
        obs = OBS.replace("7 barracks,", "7 barracks, 1 factory,")
        self.assertEqual(self.filter(TODO, obs), {"siegetank": 9, "starport": 1})
        result = replay("- starport", True, obs, TODO, STRATEGY, attack_ct=2)
        self.assertTrue(result["valid"], result["reason"])
        self.assertEqual(result["plan"], ["starport"])

    def test_enemy_factory_does_not_satisfy_own_prerequisite(self):
        self.assertIn("1 factory", OBS.split("As for the visible enemy", 1)[1])
        self.assertEqual(self.scope["_own_structure_count_from_obs"](OBS, "factory"), 0)
        self.assertEqual(self.filter(TODO, OBS), {"factory": 1})

    def test_no_tech_goal_does_not_inject_factory(self):
        for todo in ({}, {"marine": 4}, {"supplydepot": 1, "barracks": 2}):
            with self.subTest(todo=todo):
                self.assertEqual(self.filter(todo, OBS), todo)

    def test_model_can_choose_marine_without_forced_factory_execution(self):
        result = replay("- marine", True, OBS, dict(TODO, marine=4), STRATEGY, attack_ct=2)
        self.assertTrue(result["valid"], result["reason"])
        self.assertIn("- factory x 1", result["prompt"])
        self.assertEqual(result["plan"], ["marine"])

    def test_existing_invalid_fallback_is_not_mislabelled_as_model_selection(self):
        result = replay("- tank", True, OBS, TODO, STRATEGY, attack_ct=2)
        self.assertFalse(result["valid"])
        self.assertFalse(result["correct_format"])
        self.assertEqual(result["plan"], ["factory"])


if __name__ == "__main__":
    unittest.main()
