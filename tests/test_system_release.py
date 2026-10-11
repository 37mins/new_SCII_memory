import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from prepare_spb_model import patch_training_modes, prepare
from run_record import finish, finish_evaluation
from verify_assets import RUSH_BO, RUSH_BO_SHA256


class LauncherTest(unittest.TestCase):
    def test_execution_feedback_is_off_by_default_and_only_changes_its_ray_flag(self):
        configs = []
        with tempfile.TemporaryDirectory() as tmp:
            for value in ("0", "1"):
                folder = Path(tmp) / value
                folder.mkdir()
                result = self.launch(folder, "train_spb_level5.sh",
                                     SC2_EXECUTION_FAILURE_FEEDBACK=value)
                self.assertEqual(result.returncode, 0, result.stdout)
                config = json.loads((folder / "run/effective_config.json").read_text())["overrides"]
                self.assertEqual(config.pop("+ray_init.runtime_env.env_vars.SC2_EXECUTION_FAILURE_FEEDBACK"),
                                 f"'{value}'")
                configs.append({k: v for k, v in config.items() if str(folder) not in v})
            self.assertEqual(configs[0], configs[1])
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ):
                os.environ.pop("SC2_EXECUTION_FAILURE_FEEDBACK", None)
                result = self.launch(tmp, "train_sc2.sh")
            self.assertEqual(result.returncode, 0, result.stdout)
            config = json.loads((Path(tmp) / "run/effective_config.json").read_text())["overrides"]
            self.assertEqual(config["+ray_init.runtime_env.env_vars.SC2_EXECUTION_FAILURE_FEEDBACK"], "'0'")

    def test_g2_handoff_is_the_only_nonpath_ab_override_and_reaches_ray(self):
        for script in ("train_qwen_instruct2507_level5.sh", "train_spb_level5.sh"):
            configs = []
            with self.subTest(script=script), tempfile.TemporaryDirectory() as tmp:
                for limit in ("0", "138"):
                    folder = Path(tmp) / limit
                    folder.mkdir()
                    result = self.launch(folder, script, SC2_FIXED_BO_PREFIX_LIMIT=limit)
                    self.assertEqual(result.returncode, 0, result.stdout)
                    config = json.loads((folder / "run/effective_config.json").read_text())["overrides"]
                    key = "+ray_init.runtime_env.env_vars.SC2_FIXED_BO_PREFIX_LIMIT"
                    self.assertEqual(config.pop(key), f"'{limit}'")
                    configs.append({k: v for k, v in config.items() if str(folder) not in v})
                self.assertEqual(configs[0], configs[1])

    def test_explicit_stage8_overrides_keep_96_step_goal(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.launch(tmp, "train_sc2.sh", TOTAL_EPOCHS="8", SAVE_FREQ="1",
                cli_overrides=("trainer.total_training_steps=96",
                    "+actor_rollout_ref.actor.fp32_normalization=True"))
            self.assertEqual(result.returncode, 0, result.stdout)
            root = Path(tmp) / "run"
            manifest = json.loads((root / "manifest.json").read_text())
            config = json.loads((root / "effective_config.json").read_text())["overrides"]
            self.assertEqual(manifest["target_steps"], 8)
            self.assertEqual(manifest["training_goal_global_steps"], 96)
            self.assertEqual(config["trainer.save_freq"], "1")
            self.assertEqual(config["+actor_rollout_ref.actor.fp32_normalization"], "True")

    def test_checkpoint_evaluation_records_actual_eight_card_controls(self):
        entry = (REPO / "experiments/records/20260906-integrated-fixes-v1/run_real_rl_node44.sh").read_text()
        evaluation_case = entry.split("initial-eval|step8-eval)", 1)[1].split("if [[", 1)[0]
        evaluation_overrides = shlex.split(re.search(r"overrides\+=\((.*?)\)", evaluation_case, re.S).group(1))
        with tempfile.TemporaryDirectory() as tmp:
            result = self.launch(tmp, "train_sc2.sh", EVAL_ONLY="1", cli_overrides=(
                *evaluation_overrides,
                "trainer.resume_mode=resume_path", "trainer.resume_from_path=/frozen/global_step_8"))
            self.assertEqual(result.returncode, 0, result.stdout)
            manifest = json.loads((Path(tmp) / "run/manifest.json").read_text())
            self.assertEqual(manifest["world_size"], 8)
            self.assertEqual(manifest["mode"], "evaluation")
            self.assertEqual(manifest["resume_from_path"], "/frozen/global_step_8")
            config = json.loads((Path(tmp) / "run/effective_config.json").read_text())["overrides"]
            self.assertEqual(config["env.rollout.n"], "8")
            self.assertEqual(config["+env.val_rollout_n"], "8")
            self.assertEqual(config["actor_rollout_ref.rollout.val_kwargs.n"], "8")
            self.assertEqual(int(config["data.train_batch_size"]) * int(config["env.rollout.n"]) % manifest["world_size"], 0)

    def test_integrated_defaults_select_instruct_and_disable_call_shaping(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.launch(tmp, "train_sc2.sh")
            self.assertEqual(result.returncode, 0, result.stdout)
            folder = Path(tmp) / "run"
            config = json.loads((folder / "effective_config.json").read_text())["overrides"]
            manifest = json.loads((folder / "manifest.json").read_text())
            self.assertEqual(manifest["model_id"], "Qwen/Qwen3-4B-Instruct-2507")
            self.assertEqual(config["+ray_init.runtime_env.env_vars.SC2_STEP_SHAPING_MODE"], "'off'")
            self.assertEqual(config["env.max_steps"], "0")
            self.assertEqual(config["data.max_response_length"], "96")
            self.assertEqual(config["+ray_init.runtime_env.env_vars.SC2_PLAN_SEGMENT_MAX"], "'6'")
            self.assertEqual(config["+ray_init.runtime_env.env_vars.SC2_FIXED_BO_PREFIX_LIMIT"], "'0'")
            self.assertEqual(config["+env.sc2_memory_mode"], "off")
            self.assertNotIn("+actor_rollout_ref.actor.ppo_diagnostics_dir", config)

    def test_long_loop_memory_mode_is_forwarded_to_environment_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.launch(tmp, "train_sc2.sh", SC2_MEMORY_MODE="long_loop")
            self.assertEqual(result.returncode, 0, result.stdout)
            config = json.loads((Path(tmp) / "run/effective_config.json").read_text())["overrides"]
            self.assertEqual(config["+env.sc2_memory_mode"], "long_loop")
            self.assertEqual(
                config["+env.sc2_long_loop_memory_path"],
                str(Path(tmp) / "run" / "long_loop_memory.json"),
            )

    def test_integrated_spb_default_disables_only_swa_dropout(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.launch(tmp, "train_spb_level5.sh")
            self.assertEqual(result.returncode, 0, result.stdout)
            config = json.loads((Path(tmp) / "run/effective_config.json").read_text())["overrides"]
            self.assertEqual(config["+actor_rollout_ref.model.override_config.swa_dropout"], "0.0")
            self.assertEqual(config["+ray_init.runtime_env.env_vars.SC2_STEP_SHAPING_MODE"], "'off'")
            self.assertNotIn("+actor_rollout_ref.model.override_config.use_cache", config)

    def test_spb_dropout_override_only_changes_swa_config(self):
        configs = []
        with tempfile.TemporaryDirectory() as tmp:
            for disabled in ("0", "1"):
                folder = Path(tmp) / disabled
                folder.mkdir()
                result = self.launch(folder, "train_spb_level5.sh", SPB_DISABLE_SWA_DROPOUT=disabled)
                self.assertEqual(result.returncode, 0, result.stdout)
                config = json.loads((folder / "run/effective_config.json").read_text())["overrides"]
                key = "+actor_rollout_ref.model.override_config.swa_dropout"
                if disabled == "1":
                    self.assertEqual(config.pop(key), "0.0")
                else:
                    self.assertNotIn(key, config)
                configs.append({k: v for k, v in config.items() if str(folder) not in v})
            self.assertEqual(configs[0], configs[1])

    def test_four_update_probe_saves_each_step_and_preserves_training_controls(self):
        configs = []
        with tempfile.TemporaryDirectory() as tmp:
            for steps in (1, 4):
                folder = Path(tmp) / str(steps)
                folder.mkdir()
                result = self.launch(folder, "train_qwen_instruct2507_level5.sh",
                                     N_GPUS_PER_NODE="8", CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
                                     TOTAL_EPOCHS=str(steps), SAVE_FREQ="1",
                                     SC2_DIFFICULTY="Harder", SC2_MAX_STEPS="0")
                self.assertEqual(result.returncode, 0, result.stdout)
                config = json.loads((folder / "run/effective_config.json").read_text())["overrides"]
                self.assertEqual(config["trainer.total_epochs"], str(steps))
                self.assertEqual(config["trainer.save_freq"], "1")
                self.assertEqual(config["trainer.resume_mode"], "disable")
                configs.append({k: v for k, v in config.items()
                                if str(folder) not in v and k != "trainer.total_epochs"})
            self.assertEqual(configs[0], configs[1])

    def test_eight_gpu_long_budget_preserves_global_batch_and_rollouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.launch(tmp, "train_qwen_instruct2507_level5.sh",
                                 N_GPUS_PER_NODE="8", CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
                                 TOTAL_EPOCHS="96", SAVE_FREQ="8", SC2_DIFFICULTY="Harder",
                                 SC2_MAX_STEPS="0")
            self.assertEqual(result.returncode, 0, result.stdout)
            config = json.loads((Path(tmp) / "run/effective_config.json").read_text())["overrides"]
            for key, value in {
                "trainer.n_gpus_per_node": "8", "trainer.total_epochs": "96",
                "trainer.save_freq": "8", "env.rollout.n": "8",
                "data.train_batch_size": "1", "actor_rollout_ref.actor.ppo_mini_batch_size": "8",
                "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": "1",
                "data.max_response_length": "96", "env.max_steps": "0",
                "+ray_init.runtime_env.env_vars.SC2_DIFFICULTY": "'Harder'",
                "actor_rollout_ref.actor.optim.lr": "1e-6",
            }.items():
                self.assertEqual(config[key], value, key)

    def test_four_gpu_override_preserves_rollout_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.launch(tmp, N_GPUS_PER_NODE="4", CUDA_VISIBLE_DEVICES="4,5,6,7")
            self.assertEqual(result.returncode, 0, result.stdout)
            root = Path(tmp) / "run"
            config = json.loads((root / "effective_config.json").read_text())["overrides"]
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(config["trainer.n_gpus_per_node"], "4")
            self.assertEqual(config["env.rollout.n"], "8")
            self.assertEqual(config["actor_rollout_ref.actor.ppo_mini_batch_size"], "8")
            self.assertEqual(manifest["world_size"], 4)
            self.assertEqual(manifest["gpu_selection"], "4,5,6,7")

    def test_two_gpu_kl_probe_preserves_global_training_controls(self):
        configs = []
        with tempfile.TemporaryDirectory() as tmp:
            for gpus, selection in ((4, "0,1,2,3"), (2, "1,2")):
                root = Path(tmp) / str(gpus)
                root.mkdir()
                result = self.launch(root, "train_qwen_instruct2507_level5.sh",
                                     N_GPUS_PER_NODE=str(gpus), CUDA_VISIBLE_DEVICES=selection,
                                     TOTAL_EPOCHS="1", SAVE_FREQ="1", SC2_DIFFICULTY="Harder",
                                     PPO_DIAGNOSTICS="1", ROLLOUT_LOGPROB_DIAGNOSTICS="1")
                self.assertEqual(result.returncode, 0, result.stdout)
                config = json.loads((root / "run/effective_config.json").read_text())["overrides"]
                manifest = json.loads((root / "run/manifest.json").read_text())
                self.assertEqual(manifest["world_size"], gpus)
                self.assertEqual(manifest["gpu_selection"], selection)
                self.assertEqual(config["env.rollout.n"], "8")
                self.assertEqual(config["actor_rollout_ref.actor.ppo_mini_batch_size"], "8")
                self.assertEqual(config["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"], "1")
                self.assertEqual(config["+actor_rollout_ref.rollout.capture_log_probs"], "True")
                configs.append({k: v for k, v in config.items()
                                if str(root) not in v and k != "trainer.n_gpus_per_node"})
            self.assertEqual(configs[0], configs[1])

    def test_step_shaping_is_the_only_nonpath_ab_override(self):
        configs = []
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ("legacy", "off"):
                root = Path(tmp) / mode
                root.mkdir()
                result = self.launch(root, SC2_STEP_SHAPING_MODE=mode)
                self.assertEqual(result.returncode, 0, result.stdout)
                config = json.loads((root / "run/effective_config.json").read_text())["overrides"]
                key = "+ray_init.runtime_env.env_vars.SC2_STEP_SHAPING_MODE"
                self.assertEqual(config[key], f"'{mode}'")
                configs.append({k: v for k, v in config.items() if str(root) not in v and k != key})
            self.assertEqual(configs[0], configs[1])

    def test_unknown_shaping_mode_is_rejected_before_run_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.launch(tmp, SC2_STEP_SHAPING_MODE="delta")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("SC2_STEP_SHAPING_MODE must be legacy or off", result.stdout)
            self.assertFalse((Path(tmp) / "run").exists())

    def test_natural_gate_requires_an_explicit_ray_directory(self):
        env = {k: v for k, v in os.environ.items() if k != "RAY_TMPDIR"}
        env["SOURCE_DIR"] = str(REPO)
        script = REPO / "experiments/records/20260903-r6-natural-termination-ab-v1/run_l5_uncapped_1step.sh"
        result = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("set RAY_TMPDIR to a short unique path", result.stderr)

    def launch(self, folder, script="train_qwen_level5.sh", cli_overrides=(), **overrides):
        root = Path(folder)
        (root / "model").mkdir(exist_ok=True)
        (root / "sc2").mkdir(exist_ok=True)
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("SC2_", "SPB_", "MODEL_", "CONFIG_", "MAX_", "BO_", "DRY_RUN", "VERIFY_ASSETS"))}
        env.update(
            PATH=str(Path(sys.executable).parent) + os.pathsep + env["PATH"],
            MODEL_PATH=str(root / "model"), SC2PATH=str(root / "sc2"),
            OUTPUT_DIR=str(root / "run"), RUN_ID="test-system", DRY_RUN="1", VERIFY_ASSETS="0",
            **overrides,
        )
        result = subprocess.run(["bash", str(REPO / "scripts" / script), *cli_overrides], env=env,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return result

    def test_qwen_spb_share_all_nonmodel_training_controls(self):
        configs = []
        with tempfile.TemporaryDirectory() as tmp:
            for index, script in enumerate(("train_qwen_instruct2507_level5.sh", "train_spb_level5.sh")):
                root = Path(tmp) / str(index)
                root.mkdir()
                result = self.launch(root, script)
                self.assertEqual(result.returncode, 0, result.stdout)
                config = json.loads((root / "run/effective_config.json").read_text())
                overrides = config["overrides"]
                self.assertEqual(overrides["+env.strict_legacy_plan_contract"], "True")
                self.assertEqual(overrides["+data.apply_chat_template_kwargs.enable_thinking"], "False")
                self.assertEqual(overrides["data.max_response_length"], "96")
                self.assertEqual(overrides["algorithm.gigpo.mode"], "mean_norm")
                swa_key = "+actor_rollout_ref.model.override_config.swa_dropout"
                if script == "train_spb_level5.sh":
                    self.assertEqual(overrides.pop(swa_key), "0.0")
                else:
                    self.assertNotIn(swa_key, overrides)
                # Normalize only generated filesystem locations and the model input.
                configs.append({k: v for k, v in overrides.items()
                                if str(root) not in v and k != "actor_rollout_ref.model.path"})
                manifest = json.loads((root / "run/manifest.json").read_text())
                self.assertEqual(manifest["bo_profile"], "canonical")
                self.assertFalse(manifest["formal"])
                self.assertFalse((root / "run/model").exists())
            self.assertEqual(configs[0], configs[1])

    def test_rush_is_explicit_and_does_not_select_seed2(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.launch(tmp, "train_spb_level5.sh", BO_PROFILE="rush")
            self.assertEqual(result.returncode, 0, result.stdout)
            data = json.loads((Path(tmp) / "run/effective_config.json").read_text())["overrides"]
            self.assertEqual(data["env.seed"], "0")
            self.assertEqual(data["+ray_init.runtime_env.env_vars.SC2_BO_ADDRESS"], f"'{RUSH_BO}'")
            self.assertEqual(data["+ray_init.runtime_env.env_vars.SC2_ENABLE_NEWEXEC_MICRO"], "'0'")

    def test_refuses_existing_run_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = self.launch(tmp)
            self.assertEqual(first.returncode, 0, first.stdout)
            p = Path(tmp) / "run/manifest.json"
            before = p.read_bytes()
            second = self.launch(tmp)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(before, p.read_bytes())

    def test_invalid_bo_profile_stops_before_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.launch(tmp, BO_PROFILE="unknown")
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((Path(tmp) / "run").exists())

    def test_evaluation_keeps_system_but_disables_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.launch(tmp, EVAL_ONLY="1")
            self.assertEqual(result.returncode, 0, result.stdout)
            root = Path(tmp) / "run"
            config = json.loads((root / "effective_config.json").read_text())["overrides"]
            self.assertEqual(config["trainer.val_only"], "True")
            self.assertEqual(config["trainer.val_before_train"], "True")
            self.assertEqual(config["trainer.save_freq"], "-1")
            self.assertEqual(config["trainer.n_gpus_per_node"], "1")
            self.assertEqual(config["env.rollout.n"], "1")
            self.assertEqual(config["actor_rollout_ref.rollout.val_kwargs.do_sample"], "False")
            self.assertEqual(config["+env.prompt_response_log_train_only"], "False")
            self.assertEqual(config["data.max_response_length"], "96")
            self.assertEqual(config["+ray_init.runtime_env.env_vars.SC2_ENABLE_NEWEXEC_MICRO"], "'0'")
            self.assertEqual(json.loads((root / "manifest.json").read_text())["mode"], "evaluation")


class ModelPreparationTest(unittest.TestCase):
    def source_text(self):
        return ("mode = 'fused_recurrent' if q_len <= 64 else self.mode\n" * 2 +
                "mode = 'fused_recurrent' if q_len // self.num_sparse_partition <= 64 else self.mode\n" * 2)

    def test_matches_historical_training_branch_only(self):
        after = patch_training_modes(self.source_text())
        self.assertEqual(after.count("self.mode if self.training else"), 4)
        self.assertEqual(after.count("'fused_recurrent'"), 4)
        with self.assertRaises(ValueError):
            patch_training_modes(after)

    def test_preparation_does_not_modify_source_or_copy_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "original"
            source.mkdir()
            contents = {"sse.py": self.source_text(), "sse_swa.py": self.source_text(),
                        "config.json": "{}", "pytorch_model-00001.bin": "weights"}
            manifest = root / "manifest.sha256"
            for name, value in contents.items():
                (source / name).write_text(value)
            manifest.write_text("\n".join(
                hashlib.sha256(value.encode()).hexdigest() + "  " + name
                for name, value in contents.items()))
            destination = root / "run/model"
            result = prepare(source, destination, manifest)
            self.assertEqual((source / "sse.py").read_text(), self.source_text())
            self.assertTrue((destination / "pytorch_model-00001.bin").is_symlink())
            self.assertEqual(len(result["patched_files"]), 2)
            with self.assertRaises(FileExistsError):
                prepare(source, destination, manifest)
            (source / "config.json").write_text("changed")
            with self.assertRaises(ValueError):
                prepare(source, root / "other/model", manifest)
            self.assertFalse((root / "other/model").exists())


class ResultAuditTest(unittest.TestCase):
    def test_step8_evaluation_requires_all_eight_natural_games(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(json.dumps({"episode_api": "gymnasium-v0.26"}))
            (root / "effective_config.json").write_text(json.dumps({"overrides": {
                "data.val_batch_size": "1", "+env.val_rollout_n": "8"}}))
            (root / "console.log").write_text("step:8 - val/success_rate:0.0\n")
            outcomes = [dict(reset_index=0, env_index=i, terminated=True, truncated=False,
                game_result="Defeat", termination_reason="game_result") for i in range(8)]
            path = root / "episode_outcomes.jsonl"
            path.write_text("".join(json.dumps(x) + "\n" for x in outcomes))
            self.assertEqual(finish_evaluation(root, 0)["status"], "completed")
            path.write_text("".join(json.dumps(x) + "\n" for x in outcomes[:-1]))
            self.assertEqual(finish_evaluation(root, 0)["status"], "failed")

    def test_eval_requires_result_and_replay_but_not_model_participation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "console.log").write_text("step:0 - val/sc2/test_score:20.0 - val/success_rate:1.0\n")
            self.assertEqual(finish_evaluation(root, 0)["status"], "failed")
            (root / "sc2-logs").mkdir()
            (root / "sc2-logs/replay.SC2Replay").write_bytes(b"replay")
            result = finish_evaluation(root, 0)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["response_record_count"], 0)
            self.assertEqual(result["training_updates"], 0)
            self.assertEqual(finish_evaluation(root, 1)["status"], "failed")
            (root / "prompt_response.jsonl").write_text(json.dumps({
                "done": True, "is_action_valid": True, "semantic_action_evaluable": False,
            }) + "\n")
            result = finish_evaluation(root, 0)
            self.assertEqual(result["response_record_count"], 1)
            self.assertEqual(result["evaluable_action_count"], 0)
            self.assertEqual(result["valid_evaluable_action_count"], 0)

    def fixture(self, root, gradient="2.5", world_size=8):
        (root / "manifest.json").write_text(json.dumps({"target_steps": 1, "world_size": world_size}))
        (root / "console.log").write_text(
            f"step:1 - actor/grad_norm:{gradient} - episode/success_rate:0.875\n")
        checkpoint = root / "checkpoints/global_step_1"
        (checkpoint / "actor").mkdir(parents=True)
        (checkpoint / "data.pt").write_bytes(b"state")
        for kind in ("model", "optim", "extra_state"):
            for rank in range(world_size):
                (checkpoint / "actor" / f"{kind}_world_size_{world_size}_rank_{rank}.pt").write_bytes(b"state")

    def test_four_gpu_checkpoint_is_audited_against_four_ranks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, world_size=4)
            self.assertEqual(finish(root, 0)["status"], "completed")
            (root / "checkpoints/global_step_1/actor/optim_world_size_4_rank_3.pt").unlink()
            self.assertEqual(finish(root, 0)["status"], "failed")

    def test_complete_and_incomplete_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            self.assertEqual(finish(root, 0)["status"], "completed")
            (root / "checkpoints/global_step_1/actor/model_world_size_8_rank_7.pt").unlink()
            self.assertEqual(finish(root, 0)["status"], "failed")

    def test_nonfinite_or_process_failure_is_not_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, "nan")
            self.assertFalse(finish(root, 0)["metrics_finite"])
            self.assertEqual(finish(root, 143)["status"], "failed")
            json.loads((root / "metrics.json").read_text())


class SourceFreezeTest(unittest.TestCase):
    def test_rush_bo_is_exact_historical_input(self):
        self.assertEqual(hashlib.sha256(RUSH_BO.read_bytes()).hexdigest(), RUSH_BO_SHA256)
        bo = json.loads(RUSH_BO.read_text())
        self.assertEqual(sum(x["action"] == "marine" for x in bo), 24)
        self.assertEqual(sum(x["action"] == "barracks" for x in bo), 3)
        self.assertFalse(any(x["action"] in ("refinery", "commandcenter") for x in bo))

    def test_frozen_system_source(self):
        for line in (REPO / "configs/system-v1.sha256").read_text().splitlines():
            value, name = line.split(maxsplit=1)
            self.assertEqual(hashlib.sha256((REPO / name).read_bytes()).hexdigest(), value, name)


if __name__ == "__main__":
    unittest.main()
