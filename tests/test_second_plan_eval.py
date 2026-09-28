"""Historical prefix, actual parser and frozen four-card evaluation configuration."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_r6_format_reward import strict_validator
from test_valid_bo_passthrough import replay

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "experiments/records/20260911-execution-failure-feedback-v1"
sys.path[:0] = [str(RECORD), str(ROOT / "experiments/records/20260908-g2-planner-handoff-v1")]
try:
    spec = importlib.util.spec_from_file_location("second_plan_eval", RECORD / "second_plan_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
finally:
    del sys.path[:2]


class SecondPlanTest(unittest.TestCase):
    def test_marine_only_changes_one_count_and_preserves_online_answers(self):
        first, second = mod.FIXTURE["rows"]
        a = mod.select_answer(1, mod.ARMS[0], second["prompt"], "unused", marine_only=True)
        b = mod.select_answer(1, "marine_42_only", second["prompt"], "unused", marine_only=True)
        self.assertEqual(a, second["control_answer"])
        self.assertEqual(a.count("- marine x 8\n"), 1)
        self.assertEqual(b.replace("- marine x 42\n", "- marine x 8\n", 1), a)
        self.assertIn("- supplydepot x 2\n", b)
        self.assertNotIn("Here are", b)
        self.assertEqual(mod.select_answer(0, "marine_42_only", first["prompt"], "unused",
                                         marine_only=True), first["answer"])
        self.assertEqual(mod.select_answer(2, "marine_42_only", "new state", "online B4",
                                         marine_only=True), "online B4")

    def test_exact_prefix_and_recorded_answers(self):
        first, second = mod.FIXTURE["rows"]
        for i, row in enumerate((first, second)):
            self.assertEqual(hashlib.sha256(row["prompt"].encode()).hexdigest(), row["prompt_sha256"])
            for arm in mod.ARMS[:2]:
                answer = mod.select_answer(i, arm, row["prompt"], "unused generation")
                with patch.dict(os.environ, SC2_ORIGINAL_BO_SEGMENT="1"):
                    self.assertTrue(strict_validator()(row["prompt"], answer)[0])
        self.assertEqual(mod.select_answer(0, mod.ARMS[0], first["prompt"], ""),
                         mod.select_answer(0, mod.ARMS[1], first["prompt"], ""))
        self.assertEqual(mod.select_answer(1, mod.ARMS[0], second["prompt"], ""), second["control_answer"])
        self.assertEqual(mod.select_answer(1, mod.ARMS[1], second["prompt"], ""), second["replacement_answer"])
        with self.assertRaises(ValueError): mod.select_answer(1, mod.ARMS[0], "wrong observation", "")
        self.assertEqual(mod.select_answer(2, mod.ARMS[1], "new state", "online B4"), "online B4")

    def test_actual_parser_does_not_restore_six_action_cap(self):
        row = mod.FIXTURE["rows"][1]
        p = row["prompt"]
        obs = p.split("Current observations are:\n", 1)[1].split("The remaining BO list", 1)[0].strip()
        stage = re.search(r"Current Stage: (.*?) \((.*?)\)", p)
        strategy = dict(phase_summary=stage[1], timing=stage[2],
                        instruction=re.search(r"Detailed Instruction: (.*)", p)[1],
                        attack_instruction=re.search(r"Attacking Instruction: (.*)", p)[1])
        todo = {"Launch OFFENSE with 8 marine": 1, "marine": 42, "supplydepot": 8,
                "barracks": 3, "reactor barracks": 3, "scv": 2, "refinery": 1}
        marine_only = mod.select_answer(1, "marine_42_only", p, "unused", marine_only=True)
        for answer, marines, depots in ((row["control_answer"], 8, 2),
                                      (row["replacement_answer"], 42, 8), (marine_only, 42, 2)):
            result = replay(answer, True, obs, todo, strategy,
                            controls={"SC2_ORIGINAL_BO_SEGMENT": "1", "SC2_CLARIFY_BO_QUOTA": "1"})
            self.assertTrue(result["valid"], result["reason"])
            self.assertEqual(result["plan"], ["marine"]*marines + ["supplydepot"]*depots +
                ["barracks"]*3 + ["reactor barracks"]*3 + ["scv"]*2 + ["refinery"])

    def test_declared_seed_and_no_extra_game_batch(self):
        rng = mod.feedback_eval.PairedSeeds([mod.SEED]*8)
        self.assertEqual(rng.randint(65536, 2**32-1, 8).tolist(), [1098341402]*8)
        with self.assertRaises(ValueError): rng.randint(65536, 2**32-1, 8)
        self.assertEqual(mod.ARMS, ["control_B_answer", "replace_with_A_answer"]*4)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires pinned CPU Torch")
    def test_setup_hook_defers_trainer_without_cuda_probe(self):
        code = """
import sys, torch
def forbidden(*args, **kwargs):
    raise AssertionError('Premature CUDA probe')
torch._C._cuda_getDeviceCount = forbidden
torch.cuda._lazy_init = forbidden
import second_plan_eval
getattr(second_plan_eval, sys.argv[1])()
assert 'verl.trainer.ppo.ray_trainer' not in sys.modules
assert not torch.cuda.is_initialized()
print('Second-plan setup hook deferred trainer without CUDA probe')
"""
        paths = [str(RECORD), str(ROOT / "experiments/records/20260910-spb-fsdp-layer-wrap-v1"),
                 str(ROOT / "experiments/records/20260908-g2-planner-handoff-v1"),
                 str(ROOT / "experiments/records/20260906-integrated-fixes-v1"), str(ROOT),
                 os.environ.get("PYTHONPATH", "")]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", SC2PATH=os.environ.get("SC2PATH", str(ROOT)),
                   PYTHONPATH=os.pathsep.join(paths))
        for hook in ("install", "install_marine_only"):
            r = subprocess.run([sys.executable, "-c", code, hook], cwd=ROOT, env=env, text=True,
                               capture_output=True, timeout=30)
            self.assertEqual(r.returncode, 0, r.stdout+r.stderr)

    def test_effective_config_only_changes_diagnostic_hook(self):
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        configs = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"model").mkdir(); (root/"sc2").mkdir()
            for label, entry in (("baseline", "run_feedback_eval_node47.sh"),
                                 ("candidate", "run_second_plan_node43.sh"),
                                 ("marine_only", "run_marine_count_node43.sh")):
                out = root/label
                env = dict(os.environ, DRY_RUN="1", VERIFY_ASSETS="0", PYTHON_BIN=sys.executable,
                    SOURCE_SHA=sha, MODEL_PATH=str(root/"model"), SC2PATH=str(root/"sc2"),
                    FEEDBACK_STORAGE_ROOT=str(root/"storage"), FEEDBACK_LEARNING_MODEL="B4",
                    FEEDBACK_PYTHON_OVERLAY="/root/new-scii-spb-level5-96-20260908/python",
                    FEEDBACK_INCLUDE_ROOT="/root/new-scii-spb-level5-96-20260908/include",
                    FEEDBACK_SECOND_PLAN="0", FEEDBACK_SECOND_PLAN_VARIANT="recorded",
                    FEEDBACK_EVAL_GPUS="0,1,2,3", RUN_ID="second-plan-config-test",
                    OUTPUT_DIR=str(out), RAY_TMPDIR="/tmp/second-plan-test")
                command = ["bash", str(RECORD/entry)] + (["B"] if label=="baseline" else [])
                r = subprocess.run(command,cwd=ROOT,env=env,text=True,capture_output=True,timeout=30)
                self.assertEqual(r.returncode, 0, r.stdout+r.stderr)
                raw = json.loads((out/"effective_config.json").read_text())["overrides"]
                configs.append({k:v.replace(str(out),"<RUN>") for k,v in raw.items()})
                manifest = json.loads((out/"manifest.json").read_text())
                self.assertEqual((manifest["mode"],manifest["world_size"],manifest["gpu_selection"]),
                                 ("evaluation",4,"0,1,2,3"))
        a,b,c = configs
        self.assertEqual({k for k in b.keys()|c.keys() if b.get(k)!=c.get(k)},
                         {"+ray_init.runtime_env.worker_process_setup_hook"})
        self.assertEqual(c["+ray_init.runtime_env.worker_process_setup_hook"],
                         "second_plan_eval.install_marine_only")
        self.assertEqual({k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)},
                         {"+ray_init.runtime_env.worker_process_setup_hook"})
        self.assertEqual(b["+ray_init.runtime_env.worker_process_setup_hook"], "second_plan_eval.install")
        self.assertTrue(b["trainer.resume_from_path"].endswith("B-rl4-4gpu-s0-r2/checkpoints/global_step_4"))
        for key,value in {"trainer.val_only":"True", "trainer.save_freq":"-1", "env.rollout.n":"8",
            "data.max_response_length":"192", "trainer.del_local_ckpt_after_load":"False"}.items():
            self.assertEqual(b[key],value)
        entry = RECORD/"run_second_plan_node43.sh"
        subprocess.run(["bash", "-n", str(entry)], check=True)
        subprocess.run(["bash", "-n", str(RECORD/"run_marine_count_node43.sh")], check=True)
        self.assertIn("+ 2700", entry.read_text())
        self.assertIn("--kill-after=15s", entry.read_text())
        suffix="/session_2026-09-11_23-59-59_999999_2147483647/sockets/plasma_store"
        self.assertLessEqual(len(("/dev/shm/sp_0911_r1"+suffix).encode()),107)


if __name__ == "__main__": unittest.main()
