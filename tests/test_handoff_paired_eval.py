"""CPU-only checks of actual seed dispatch, one-time model RNG setup and CLI."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from test_r6_natural_termination import ROOT, method

RECORD = ROOT / 'experiments/records/20260908-g2-planner-handoff-v1'
sys.path.insert(0, str(RECORD))
from handoff_eval import GAME_SEEDS, select_seed, seeded_generation, install, install_sampling_on_initialized_worker


class HandoffEvaluationTest(unittest.TestCase):
    def test_setup_defers_sampling_install_until_initialized_worker_rpc(self):
        events = []
        class Trainer:
            def _validate(self):
                events.append('actual validation')
                return 'original result'
        environments = SimpleNamespace(make_envs=lambda cfg: ('train_env', 'val_env'))
        pair_seed = SimpleNamespace(install=Mock())
        modules = {'agent_system.environments': environments, 'r4_pair_seed': pair_seed,
                   'verl.trainer.ppo.ray_trainer': SimpleNamespace(RayPPOTrainer=Trainer)}
        with patch.dict(sys.modules, modules):
            install()
            self.assertEqual(events, [])
            self.assertEqual(environments.make_envs(object()), ('train_env', 'val_env'))
            self.assertEqual(events, [])
            trainer = Trainer()
            rpc = Mock(side_effect=lambda *args: events.append('initialized worker RPC'))
            trainer.actor_rollout_wg = SimpleNamespace(execute_all_sync=rpc)
            self.assertEqual(trainer._validate(), 'original result')
            rpc.assert_called_once_with('actor_rollout_execute_with_func_generator', install_sampling_on_initialized_worker)
            self.assertEqual(events, ['initialized worker RPC', 'actual validation'])

    def test_initialized_worker_patches_only_its_existing_rollout_class(self):
        class Rollout:
            def generate_sequences(self, prompts):
                return prompts
        original = Rollout.generate_sequences
        execute = method(
            ROOT/'verl/single_controller/base/worker.py', 'Worker', 'execute_with_func_generator',
            register=lambda **kwargs: lambda function: function,
            Dispatch=SimpleNamespace(DP_COMPUTE_PROTO_WITH_FUNC=object()))
        with patch.dict(sys.modules, {'verl.workers.rollout.hf_rollout': SimpleNamespace(HFRollout=Rollout)}):
            self.assertIsNone(execute(SimpleNamespace(), install_sampling_on_initialized_worker))
        self.assertIs(Rollout.generate_sequences.__wrapped__, original)

    def test_four_games_are_dispatched_once_by_actual_reset(self):
        raw = SimpleNamespace(is_train=False, seed=1000, num_processes=4, reset_index=0)
        calls = []
        raw.workers = [SimpleNamespace(reset=SimpleNamespace(remote=lambda seed: calls.append(seed) or ('obs', {}))) for _ in range(4)]
        self.assertEqual(select_seed(SimpleNamespace(envs=raw))['seeds'], GAME_SEEDS)
        reset = method(ROOT/'agent_system/environments/env_package/sc2_online_restored/envSC2.py',
                       'SC2MultiProcessEnv', 'reset', np=np, os=os, json=json,
                       ray=SimpleNamespace(get=lambda values: values))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, SC2_EVAL_SEEDS_LOG_PATH=tmp+'/seeds.jsonl'):
            reset(raw)
            self.assertEqual(calls, [2732, 2732, 43567, 43567])
            self.assertEqual(json.loads(Path(tmp+'/seeds.jsonl').read_text())['seeds'], calls)
            with self.assertRaises(ValueError):
                reset(raw)
        self.assertEqual(len(calls), 4)

    def test_seed_once_per_worker_not_once_per_model_answer(self):
        state = SimpleNamespace(tolist=lambda: [1, 2, 3])
        torch = SimpleNamespace(distributed=SimpleNamespace(get_rank=lambda: 2),
                                manual_seed=Mock(), get_rng_state=lambda: state,
                                cuda=SimpleNamespace(get_rng_state=lambda: state))
        original = Mock(return_value='unaltered output')
        generate = seeded_generation(original)
        rollout, prompts = SimpleNamespace(), object()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, torch=torch), patch.dict(os.environ, SC2_EVAL_SEEDS_LOG_PATH=tmp+'/seeds.jsonl'):
            self.assertEqual(generate(rollout, prompts), 'unaltered output')
            self.assertEqual(generate(rollout, prompts), 'unaltered output')
            self.assertEqual(json.loads(Path(tmp+'/sampling-rank2.json').read_text())['seed'], 2)
        torch.manual_seed.assert_called_once_with(2)
        self.assertEqual(original.call_count, 2)
        original.assert_called_with(rollout, prompts)

    def test_actual_arm_launchers_differ_only_in_handoff_and_gpu_paths(self):
        self.check_arm_configs(138)

    def test_may_opening_arm_launchers_differ_only_in_handoff_and_gpu_paths(self):
        self.check_arm_configs(-1)

    def check_arm_configs(self, candidate):
        configs = []
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        with tempfile.TemporaryDirectory() as tmp:
            for arm in ('A', 'B'):
                folder = Path(tmp)/arm
                (folder/'model').mkdir(parents=True)
                (folder/'sc2').mkdir()
                env = {k: v for k, v in os.environ.items() if not k.startswith(('SC2_', 'SPB_', 'MODEL_', 'CONFIG_', 'MAX_', 'BO_', 'CUDA_VISIBLE_DEVICES'))}
                env.update(SOURCE_SHA=sha, PYTHON_BIN=sys.executable, DRY_RUN='1', VERIFY_ASSETS='0',
                           SC2_HANDOFF_B_PREFIX=str(candidate),
                           MODEL_PATH=str(folder/'model'), SC2PATH=str(folder/'sc2'),
                           RUN_ID='handoff-test-'+arm, OUTPUT_DIR=str(folder/'run'), RAY_TMPDIR=str(folder/'ray'))
                result = subprocess.run(['bash', str(RECORD/'run_eval.sh'), arm], env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                cfg = json.loads((folder/'run/effective_config.json').read_text())['overrides']
                manifest = json.loads((folder/'run/manifest.json').read_text())
                self.assertEqual(manifest['world_size'], 4)
                self.assertEqual(manifest['mode'], 'evaluation')
                self.assertEqual(manifest['gpu_selection'], '0,1,2,3' if arm == 'A' else '4,5,6,7')
                for key, value in {'trainer.val_only':'True', 'trainer.resume_mode':'disable', 'trainer.save_freq':'-1',
                                   'env.rollout.n':'4', '+env.val_rollout_n':'4', 'actor_rollout_ref.rollout.val_kwargs.n':'4',
                                   'actor_rollout_ref.rollout.val_kwargs.temperature':'0.4',
                                   '+actor_rollout_ref.model.override_config.swa_dropout':'0.0'}.items():
                    self.assertEqual(cfg[key], value)
                key = '+ray_init.runtime_env.env_vars.SC2_FIXED_BO_PREFIX_LIMIT'
                self.assertEqual(cfg.pop(key), "'0'" if arm == 'A' else f"'{candidate}'")
                configs.append({k: v for k, v in cfg.items() if str(folder) not in v and k != 'trainer.experiment_name'})
            self.assertEqual(configs[0], configs[1])


if __name__ == '__main__':
    unittest.main()
