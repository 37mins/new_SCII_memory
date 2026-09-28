"""Exercise the short-RL entry on CPU; no GPU allocation or model loading."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / 'experiments/records/20260909-bo-segment-protocol-v1/run_quota_train4_node44.sh'


class QuotaRLLaunchTest(unittest.TestCase):
    def test_four_global_steps_preserve_evaluated_candidate(self):
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'model').mkdir()
            (folder/'sc2').mkdir()
            env = dict(os.environ, DRY_RUN='1', VERIFY_ASSETS='0', PYTHON_BIN=sys.executable,
                       SOURCE_SHA=sha, MODEL_PATH=str(folder/'model'), SC2PATH=str(folder/'sc2'),
                       RUN_ID='quota-rl-test', OUTPUT_DIR=str(folder/'run'))
            result = subprocess.run(['bash', str(ENTRY)], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            cfg = json.loads((folder/'run/effective_config.json').read_text())['overrides']
            manifest = json.loads((folder/'run/manifest.json').read_text())
            self.assertEqual((manifest['mode'], manifest['world_size'], manifest['target_steps'],
                              manifest['training_goal_global_steps']), ('training', 4, 4, 96))
            expected = {'trainer.total_epochs':'4', 'trainer.total_training_steps':'96',
                        'trainer.save_freq':'1', 'trainer.val_only':'False',
                        'trainer.val_before_train':'False', 'trainer.resume_mode':'disable',
                        'env.rollout.n':'8', 'env.history_length':'0', '+env.sc2_memory_mode':'off',
                        'data.max_response_length':'192', 'algorithm.adv_estimator':'gigpo',
                        'algorithm.gigpo.mode':'mean_norm',
                        'actor_rollout_ref.actor.ppo_mini_batch_size':'8',
                        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu':'1',
                        'actor_rollout_ref.actor.optim.lr':'1e-6',
                        '+actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype':'fp32',
                        '+actor_rollout_ref.actor.fp32_normalization':'True',
                        '+actor_rollout_ref.ref.fsdp_config.model_dtype':'fp32',
                        '+actor_rollout_ref.ref.fp32_normalization':'True',
                        '+actor_rollout_ref.model.override_config.swa_dropout':'0.0',
                        '+ray_init.runtime_env.env_vars.SC2_ORIGINAL_BO_SEGMENT':"'1'",
                        '+ray_init.runtime_env.env_vars.SC2_CLARIFY_BO_QUOTA':"'1'",
                        '+ray_init.runtime_env.env_vars.SC2_FIXED_BO_PREFIX_LIMIT':"'-1'",
                        '+ray_init.runtime_env.env_vars.SC2_PRESERVE_VALID_BO':"'1'",
                        '+ray_init.runtime_env.env_vars.SC2_ENABLE_AIR_OPTIONS':"'1'",
                        '+ray_init.runtime_env.env_vars.SC2_RESET_RETRIES':"'1'",
                        '+ray_init.runtime_env.env_vars.SC2_DIFFICULTY':"'Harder'"}
            for key, value in expected.items():
                self.assertEqual(cfg[key], value, key)
            for key in ('+ray_init.runtime_env.worker_process_setup_hook',
                        '+actor_rollout_ref.actor.ppo_diagnostics_dir',
                        '+actor_rollout_ref.model.override_config.use_cache'):
                self.assertNotIn(key, cfg)


if __name__ == '__main__':
    unittest.main()
