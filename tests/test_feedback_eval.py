"""CPU-only frozen-config, paired-seed and startup-boundary checks; no GPU/game."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "experiments/records/20260911-execution-failure-feedback-v1"
HANDOFF = ROOT / "experiments/records/20260908-g2-planner-handoff-v1"
OLD_RECORD = ROOT / "experiments/records/20260910-spb-fsdp-layer-wrap-v1"
INTEGRATED = ROOT / "experiments/records/20260906-integrated-fixes-v1"
SEEDS = [52620, 52620, 17089, 17089]


def load_hook():
    sys.path.insert(0, str(HANDOFF))
    try:
        spec = importlib.util.spec_from_file_location("feedback_eval", RECORD / "feedback_eval.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(HANDOFF))


class FeedbackEvaluationTest(unittest.TestCase):
    def test_four_game_effective_configs_differ_only_in_feedback(self):
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "model").mkdir()
            (folder / "sc2").mkdir()
            configs = []
            for arm, cards in (("A", "0,1,2,3"), ("B", "4,5,6,7")):
                output = folder / arm
                env = dict(os.environ, DRY_RUN="1", VERIFY_ASSETS="0", PYTHON_BIN=sys.executable,
                           SOURCE_SHA=sha, MODEL_PATH=str(folder / "model"), SC2PATH=str(folder / "sc2"),
                           FEEDBACK_STORAGE_ROOT=str(folder / "storage"),
                           OUTPUT_DIR=str(output), RUN_ID="feedback-config-test",
                           RAY_TMPDIR="/tmp/feedback-config-test")
                result = subprocess.run(["bash", str(RECORD / "run_feedback_eval_node47.sh"), arm],
                                        cwd=ROOT, env=env, text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                raw = json.loads((output / "effective_config.json").read_text())
                cfg = {key: value.replace(str(output), "<RUN>") for key, value in raw["overrides"].items()}
                configs.append(cfg)
                manifest = json.loads((output / "manifest.json").read_text())
                self.assertEqual(manifest["mode"], "evaluation")
                self.assertEqual(manifest["world_size"], 4)
                self.assertEqual(manifest["gpu_selection"], cards)
                self.assertEqual(manifest["bo_profile"], "canonical")
                for key, expected in {
                    "trainer.val_only": "True", "trainer.val_before_train": "True",
                    "trainer.total_epochs": "1", "trainer.save_freq": "-1",
                    "trainer.resume_mode": "resume_path", "trainer.del_local_ckpt_after_load": "False",
                    "trainer.n_gpus_per_node": "4", "trainer.nnodes": "1",
                    "env.rollout.n": "4", "+env.val_rollout_n": "4",
                    "actor_rollout_ref.rollout.val_kwargs.n": "4",
                    "actor_rollout_ref.rollout.val_kwargs.do_sample": "True",
                    "actor_rollout_ref.rollout.temperature": "0.4",
                    "actor_rollout_ref.rollout.val_kwargs.temperature": "0.4",
                    "actor_rollout_ref.rollout.val_kwargs.top_p": "0.9",
                    "actor_rollout_ref.rollout.val_kwargs.top_k": "0",
                    "data.max_prompt_length": "4096", "data.max_response_length": "192",
                    "actor_rollout_ref.rollout.name": "hf",
                    "+actor_rollout_ref.model.attn_implementation": "eager",
                    "+actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype": "fp32",
                    "+actor_rollout_ref.actor.fp32_normalization": "True",
                    "+actor_rollout_ref.ref.fsdp_config.model_dtype": "fp32",
                    "+actor_rollout_ref.ref.fsdp_config.mixed_precision.param_dtype": "fp32",
                    "+actor_rollout_ref.ref.fp32_normalization": "True",
                    "++actor_rollout_ref.rollout.fsdp_layer_wrap": "True",
                    "+actor_rollout_ref.model.override_config.swa_dropout": "0.0",
                    "actor_rollout_ref.actor.optim.lr": "1e-6", "algorithm.gamma": "0.95",
                    "env.history_length": "0", "+env.sc2_memory_mode": "off",
                    "+ray_init.runtime_env.worker_process_setup_hook": "feedback_eval.install",
                }.items():
                    self.assertEqual(cfg[key], expected, key)
                self.assertTrue(cfg["trainer.resume_from_path"].endswith("/global_step_24"))
                self.assertIn("spb2-l7-g2-opening-quota-resume4to96-4gpu-s0-r1", cfg["trainer.resume_from_path"])
                self.assertNotIn("+trainer.sc2_checkpoint_resume", cfg)
                for key, expected in {
                    "SC2_FIXED_BO_PREFIX_LIMIT": "-1", "SC2_ORIGINAL_BO_SEGMENT": "1",
                    "SC2_CLARIFY_BO_QUOTA": "1", "SC2_DIFFICULTY": "Harder",
                    "SC2_STEP_SHAPING_MODE": "off", "SC2_PRESERVE_VALID_BO": "1",
                    "SC2_ENABLE_AIR_OPTIONS": "1",
                }.items():
                    self.assertEqual(cfg["+ray_init.runtime_env.env_vars." + key].strip("'"), expected)
            a, b = configs
            changed = {key: (a.get(key), b.get(key)) for key in a.keys() | b.keys() if a.get(key) != b.get(key)}
            self.assertEqual(changed, {
                "+ray_init.runtime_env.env_vars.SC2_EXECUTION_FAILURE_FEEDBACK": ("'0'", "'1'")})

    def test_seed_selection_is_paired_and_single_use(self):
        mod = load_hook()
        self.assertEqual(mod.GAME_SEEDS, SEEDS)
        observed = []
        for _ in ("A", "B"):
            raw = SimpleNamespace(is_train=False, seed=1000, num_processes=4, reset_index=0)
            proof = mod.select_seeds(raw)
            self.assertEqual(proof["seeds"], SEEDS)
            self.assertEqual(proof["games"], 4)
            self.assertEqual(proof["extra_games"], 0)
            self.assertEqual(proof["independent_scenarios"], 2)
            observed.append(raw.validation_rng.randint(2**16, 2**32 - 1, 4).tolist())
            with self.assertRaises(ValueError):
                raw.validation_rng.randint(2**16, 2**32 - 1, 4)
        self.assertEqual(observed, [SEEDS, SEEDS])

    def test_seed_selection_rejects_training_resets_or_wrong_width(self):
        mod = load_hook()
        for key, bad in (("is_train", True), ("seed", 0), ("num_processes", 8), ("reset_index", 1)):
            with self.subTest(key=key):
                raw = SimpleNamespace(is_train=False, seed=1000, num_processes=4, reset_index=0)
                setattr(raw, key, bad)
                with self.assertRaises(ValueError):
                    mod.select_seeds(raw)
        with self.assertRaises(ValueError):
            mod.PairedSeeds().randint(2**16, 2**32 - 1, 8)

    def test_frozen_hf_spb_cache_and_bf16_autocast_are_retained(self):
        tree = ast.parse((ROOT / "verl/workers/rollout/hf_rollout.py").read_text())
        cache = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "use_cache" for target in node.targets))
        scope = {"model_type": "sse_swa_moba"}
        exec(compile(ast.Module(body=[cache], type_ignores=[]), "cache_config", "exec"), scope)
        self.assertIs(scope["use_cache"], True)
        generation = next(node for node in ast.walk(tree) if isinstance(node, ast.With)
                          and "self.module.generate" in ast.unparse(node))
        autocast = next(item.context_expr for item in generation.items
                        if isinstance(item.context_expr, ast.Call)
                        and ast.unparse(item.context_expr.func) == "torch.autocast")
        self.assertEqual({key.arg: ast.unparse(key.value) for key in autocast.keywords},
                         {"device_type": "'cuda'", "dtype": "torch.bfloat16"})

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires pinned CPU Torch")
    def test_setup_hook_defers_trainer_and_does_not_probe_cuda(self):
        code = """
import sys, torch
def forbidden(*args, **kwargs):
    raise AssertionError('Premature CUDA probe in feedback setup hook')
torch._C._cuda_getDeviceCount = forbidden
torch.cuda._lazy_init = forbidden
import feedback_eval
feedback_eval.install()
assert 'verl.trainer.ppo.ray_trainer' not in sys.modules
assert not torch.cuda.is_initialized()
print('Feedback setup hook deferred trainer import without CUDA probe')
"""
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", SC2PATH=os.environ.get("SC2PATH", str(ROOT)),
                   PYTHONPATH=os.pathsep.join((str(RECORD), str(OLD_RECORD), str(HANDOFF),
                                              str(INTEGRATED), str(ROOT), os.environ.get("PYTHONPATH", ""))))
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_both_shell_entrypoints_parse(self):
        for name in ("run_feedback_eval_node47.sh", "run_feedback_pair_node47.sh", "run_feedback_pair_node43.sh"):
            with self.subTest(name=name):
                result = subprocess.run(["bash", "-n", str(RECORD / name)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_node43_only_relocates_paths_and_shortens_budget(self):
        wrapper = (RECORD / "run_feedback_pair_node43.sh").read_text()
        settings = dict(re.findall(r"(FEEDBACK_\w+)=(\S+)", wrapper))
        self.assertEqual(settings, {
            "FEEDBACK_PYTHON_OVERLAY": "/root/new-scii-spb-level5-96-20260908/python",
            "FEEDBACK_INCLUDE_ROOT": "/root/new-scii-spb-level5-96-20260908/include",
            "FEEDBACK_REPLICATE": "${FEEDBACK_REPLICATE:-r2}", "FEEDBACK_PAIR_SECONDS": "3400"})
        self.assertIn("timeout --signal=INT --kill-after=15s 3405", wrapper)
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "model").mkdir()
            (folder / "sc2").mkdir()
            configs = []
            for node, overrides in ((47, {}), (43, {**settings, "FEEDBACK_REPLICATE": "r2"})):
                output = folder / str(node)
                env = dict(os.environ, DRY_RUN="1", VERIFY_ASSETS="0", PYTHON_BIN=sys.executable,
                           SOURCE_SHA=sha, MODEL_PATH=str(folder / "model"), SC2PATH=str(folder / "sc2"),
                           FEEDBACK_STORAGE_ROOT=str(folder / "storage"),
                           OUTPUT_DIR=str(output), RUN_ID="feedback-node-config-test",
                           RAY_TMPDIR="/tmp/feedback-node-test", **overrides)
                result = subprocess.run(["bash", str(RECORD / "run_feedback_eval_node47.sh"), "A"],
                                        cwd=ROOT, env=env, text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                cfg = json.loads((output / "effective_config.json").read_text())["overrides"]
                configs.append({k: v.replace(str(output), "<RUN>")
                    .replace("/root/new-scii-spb2-level5-control/python", "<PYTHON>")
                    .replace(settings["FEEDBACK_PYTHON_OVERLAY"], "<PYTHON>") for k, v in cfg.items()})
            self.assertEqual(configs[0], configs[1])

    def test_local_storage_changes_paths_not_experiment_configuration(self):
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp).resolve()
            configs = []
            for mode in ("shared", "local"):
                storage = folder / mode
                assets = storage / "yangjinyue/sc2-verl-agent-experiments/assets"
                model = assets / "models/Panyuqi/SpikingBrain-2.0-instruct"
                sc2 = assets / "sc2/StarCraftII"
                model.mkdir(parents=True)
                sc2.mkdir(parents=True)
                env = dict(os.environ, DRY_RUN="1", VERIFY_ASSETS="0", PYTHON_BIN=sys.executable,
                           SOURCE_SHA=sha, FEEDBACK_STORAGE_ROOT=str(storage), FEEDBACK_REPLICATE="r3",
                           RUN_ID="feedback-storage-test", RAY_TMPDIR="/tmp/feedback-storage-test")
                for name in ("MODEL_PATH", "SC2PATH", "OUTPUT_DIR"):
                    env.pop(name, None)
                result = subprocess.run(["bash", str(RECORD / "run_feedback_eval_node47.sh"), "A"],
                                        cwd=ROOT, env=env, text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                output = storage / "yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs/feedback-storage-test"
                raw = (output / "effective_config.json").read_text()
                self.assertNotIn("/mnt/data0601", raw)
                configs.append(raw.replace(str(storage), "<STORAGE>"))
                manifest = json.loads((output / "manifest.json").read_text())
                self.assertEqual(manifest["model_source"], str(model))
                self.assertTrue(manifest["resume_from_path"].startswith(str(storage)))
            self.assertEqual(configs[0], configs[1])
        preflight = ast.parse((RECORD / "checkpoint_preflight.py").read_text())
        root_assignment = next(node for node in preflight.body if isinstance(node, ast.Assign)
                               and any(isinstance(t, ast.Name) and t.id == "storage_root" for t in node.targets))
        self.assertIn("FEEDBACK_STORAGE_ROOT", ast.unparse(root_assignment))
        self.assertIn("${FEEDBACK_STORAGE_ROOT:-/mnt/data0601}",
                      (RECORD / "run_feedback_pair_node47.sh").read_text())


if __name__ == "__main__":
    unittest.main()
