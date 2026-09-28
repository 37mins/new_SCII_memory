import hashlib
import os
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ARENA = REPO / "agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy"
sys.path.insert(0, str(ARENA))

from agents.planner import SC2Planner, parse_build_order  # noqa: E402


class PlannerContractTest(unittest.TestCase):
    def setUp(self):
        os.environ.update(
            SC2_PLANNER_PROTOCOL="legacy",
            SC2_PLAN_SEGMENT_MAX="6",
            SC2_PLAN_BULLET_MAX="6",
            SC2_MAX_FORCED_ATTACKS="0",
        )

    def test_strict_parser_accepts_only_canonical_bullets(self):
        text = "<answer>\nHere are the Build Orders with attack instructions:\n- marine\n- barracks\n</answer>"
        choices = {"1": "marine", "2": "barracks"}
        self.assertEqual(
            parse_build_order(text, choice_map=choices, strict_names_only=True),
            ["marine", "barracks"],
        )
        self.assertEqual(parse_build_order("- barrack", choice_map=choices, strict_names_only=True), [])

    def test_generated_query_contains_action_table_and_one_shot(self):
        planner = SC2Planner.__new__(SC2Planner)
        planner.ct = 0
        planner.correct_format = False
        planner.exceed = False
        captured = {}

        def reply(prompt, **_):
            captured["prompt"] = prompt
            return "<answer>\nHere are the Build Orders with attack instructions:\n- barracks\n- marine\n</answer>"

        planner.get_action = reply
        strategy = {
            "phase_summary": "Opening",
            "timing": "00:00 - 03:00",
            "instruction": "Build an early army.",
            "attack_instruction": "No attacks in this phase.",
        }
        plan = planner.generate_plan("Own structures: none", {"barracks": 1, "marine": 2}, strategy, 0.0)

        self.assertIn("## Optional Operations/Units:", captured["prompt"])
        self.assertIn("<answer>\nHere are the Build Orders with attack instructions:", captured["prompt"])
        self.assertTrue(plan)
        self.assertTrue(set(plan) <= {"barracks", "marine"})
        self.assertLessEqual(len(plan), 6)


class FrozenInputTest(unittest.TestCase):
    def test_small_inputs_match_release(self):
        expected = {
            "data/train.parquet": "7c8f6b07eacef5e06963b475c3c36e90196da870fae095b12c2ea0f7faf10912",
            "data/test.parquet": "4dbe5e3ff68df3fbf76e91284b28b06c52bf7495193013a4bf4ed7ca417a0cd7",
        }
        for name, value in expected.items():
            self.assertEqual(hashlib.sha256((REPO / name).read_bytes()).hexdigest(), value)

    def test_instruct_launcher_reuses_release_training_method(self):
        launcher = (REPO / "scripts/train_qwen_instruct2507_level5.sh").read_text()
        config = (REPO / "configs/qwen3-4b-instruct-2507-level5.env").read_text()
        training_launcher = (REPO / "scripts/train_sc2.sh").read_text()
        sc2_environment = (
            REPO / "agent_system/environments/env_package/sc2_online_restored/envSC2.py"
        ).read_text()
        self.assertIn("train_sc2.sh", launcher)
        self.assertIn("Qwen/Qwen3-4B-Instruct-2507", config)
        self.assertIn("cdbee75f17c01a7cc42f958dc650907174af0554", config)
        self.assertTrue((REPO / "scripts/check_qwen_instruct2507.py").is_file())
        self.assertIn('hydra.run.dir="${OUTPUT_DIR}/hydra"', training_launcher)
        self.assertIn('SC2_RUNTIME_LOG_ROOT="${OUTPUT_DIR}/sc2-logs"', training_launcher)
        self.assertIn('os.getenv("SC2_RUNTIME_LOG_ROOT", "logs")', sc2_environment)


if __name__ == "__main__":
    unittest.main()
