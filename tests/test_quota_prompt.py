"""Prompt-only candidate: identical frozen transformation, queues and eval budget."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_fixed_prompt_ab import module as fixed
from test_original_bo_segment import pipeline, ROOT
from test_r6_format_reward import planner_scope


class QuotaPromptTest(unittest.TestCase):
    def test_transformation_matches_completed_fixed_query_candidate(self):
        query = pipeline("- marine x 15", True)["query"]
        self.assertEqual(planner_scope()["clarify_bo_quota"](query), fixed.candidate(query))

    def test_real_query_changes_but_same_answer_keeps_all_downstream_results(self):
        for answer in ("- marine x 20", "- marine\n" * 15, "- marine x 99", "- barrack", "- marine x"):
            with patch.dict(os.environ, SC2_CLARIFY_BO_QUOTA="0"):
                a = pipeline(answer, True)
                self.assertEqual(a, pipeline(answer, True, sha="385829c48099c4cbcf749d0ae83c4b7761117235"))
            with patch.dict(os.environ, SC2_CLARIFY_BO_QUOTA="1"):
                b = pipeline(answer, True)
            self.assertEqual(b.pop("query"), fixed.candidate(a.pop("query")))
            self.assertEqual(a, b)

    def test_six_action_protocol_is_untouched(self):
        with patch.dict(os.environ, SC2_CLARIFY_BO_QUOTA="1"):
            self.assertEqual(pipeline("- marine", False),
                pipeline("- marine", False, sha="385829c48099c4cbcf749d0ae83c4b7761117235"))

    def test_two_eval_configs_only_differ_by_prompt_flag(self):
        entry = ROOT / "experiments/records/20260909-bo-segment-protocol-v1/run_quota_eval.sh"
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        configs = []
        with tempfile.TemporaryDirectory() as tmp:
            for arm in ("A", "B"):
                folder = Path(tmp) / arm
                (folder / "model").mkdir(parents=True)
                (folder / "sc2").mkdir()
                env = {k: v for k, v in os.environ.items() if not k.startswith(
                    ("SC2_", "SPB_", "MODEL_", "CONFIG_", "MAX_", "BO_", "CUDA_VISIBLE_DEVICES"))}
                env.update(SOURCE_DIR=str(ROOT), SOURCE_SHA=sha, PYTHON_BIN=sys.executable,
                    DRY_RUN="1", VERIFY_ASSETS="0", MODEL_PATH=str(folder / "model"),
                    SC2PATH=str(folder / "sc2"), RUN_ID="quota-" + arm,
                    OUTPUT_DIR=str(folder / "run"), RAY_TMPDIR=str(folder / "ray"))
                result = subprocess.run(["bash", str(entry), arm], env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                cfg = json.loads((folder / "run/effective_config.json").read_text())["overrides"]
                manifest = json.loads((folder / "run/manifest.json").read_text())
                self.assertEqual(manifest["mode"], "evaluation")
                self.assertEqual(manifest["world_size"], 4)
                self.assertEqual(manifest["gpu_selection"], "4,5,6,7")
                self.assertEqual(cfg.pop("+ray_init.runtime_env.env_vars.SC2_CLARIFY_BO_QUOTA"),
                                 "'0'" if arm == "A" else "'1'")
                for key, value in {"trainer.val_only":"True", "trainer.resume_mode":"disable",
                    "trainer.save_freq":"-1", "data.max_response_length":"192",
                    "env.history_length":"0", "env.rollout.n":"4", "+env.val_rollout_n":"4",
                    "actor_rollout_ref.rollout.val_kwargs.n":"4",
                    "+ray_init.runtime_env.env_vars.SC2_ORIGINAL_BO_SEGMENT":"'1'",
                    "+ray_init.runtime_env.env_vars.SC2_FIXED_BO_PREFIX_LIMIT":"'-1'"}.items():
                    self.assertEqual(cfg[key], value, key)
                configs.append({k:v for k,v in cfg.items() if str(folder) not in v and k!="trainer.experiment_name"})
        self.assertEqual(configs[0], configs[1])


if __name__ == "__main__":
    unittest.main()
