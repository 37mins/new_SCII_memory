"""CPU-only checks for the bounded eight-GPU long-run queue."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT/'experiments/records/20260916-combat-result-reward-v1'
spec = importlib.util.spec_from_file_location('combat_long_queue', RECORD/'queue_long_node197.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class LongQueueTest(unittest.TestCase):
    def test_requires_all_eight_idle_and_sufficient_host_memory(self):
        rows = '\n'.join(f'{i}, 0, 0' for i in range(8))
        self.assertTrue(queue.idle(rows, '', 700*queue.GIB))
        self.assertFalse(queue.idle(rows, '123', 700*queue.GIB))
        self.assertFalse(queue.idle(rows, '', 500*queue.GIB))
        self.assertFalse(queue.idle(rows.replace('7, 0, 0', '7, 30000, 70'), '', 700*queue.GIB))

    def test_absent_or_failed_gate_cannot_launch(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/'gate.json'
            self.assertFalse(queue.read_gate(path, 'sha', ['run']))
            value = dict(source_sha='sha', run_ids=['run'], evidence='review.md', decision='stop')
            path.write_text(json.dumps(value))
            with self.assertRaises(RuntimeError): queue.read_gate(path, 'sha', ['run'])
            value['decision'] = 'continue'
            path.write_text(json.dumps(value))
            self.assertTrue(queue.read_gate(path, 'sha', ['run']))
            with self.assertRaises(ValueError): queue.read_gate(path, 'other-sha', ['run'])

    def test_training_acceptance_requires_full_state_and_correct_step_count(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest = dict(source_commit='sha', dirty=False, world_size=8)
            (run/'manifest.json').write_text(json.dumps(manifest))
            result = dict(status='completed', process_exit_code=0, start_step=0,
                          last_global_step=8, completed_steps=8, checkpoint_complete=True,
                          metrics_finite=True, empty_policy_batches=[])
            for name in ('result.json', '.done'):
                (run/name).write_text(json.dumps(result))
            cp = run/'checkpoints/global_step_8'
            (cp/'actor').mkdir(parents=True)
            files = [cp/'data.pt', cp/'sc2_state.pt'] + [cp/'actor'/f'{kind}_world_size_8_rank_{rank}.pt'
                     for kind in ('model', 'optim', 'extra_state') for rank in range(8)]
            for f in files: f.write_bytes(b'fixture')
            self.assertEqual(queue.verify_result(run, 'train8', 'sha'), result)
            (cp/'sc2_state.pt').unlink()
            with self.assertRaises(RuntimeError): queue.verify_result(run, 'train8', 'sha')

    def test_eval_checks_actual_seed_log_not_just_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run/'manifest.json').write_text(json.dumps(dict(source_commit='sha', dirty=False, world_size=8)))
            result = dict(status='completed', process_exit_code=0, training_updates=0,
                          episode_outcomes={'natural_games':8})
            for name in ('result.json', '.done'): (run/name).write_text(json.dumps(result))
            proof = dict(loaded_global_step=0, val_only=True, optimizer_updates=0,
                         game_seeds=list(range(8)))
            (run/'checkpoint_evaluation.json').write_text(json.dumps(proof))
            log = run/'evaluation_seeds.jsonl'
            log.write_text(json.dumps({'seeds':list(range(8))})+'\n')
            self.assertEqual(queue.verify_result(run, 'initial-eval', 'sha'), result)
            log.write_text(json.dumps({'seeds':[0]*8})+'\n')
            with self.assertRaises(RuntimeError): queue.verify_result(run, 'initial-eval', 'sha')

    def test_stage_dry_configs_and_same_D_learning_configuration(self):
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        with tempfile.TemporaryDirectory(prefix='combat-long-test-') as directory:
            root = Path(directory)
            configs = {}
            for stage in queue.STAGES + ('D',):
                out = root/stage
                env = dict(os.environ, DRY_RUN='1', VERIFY_ASSETS='0', SOURCE_SHA=sha,
                           RUN_ID='dry-'+stage, OUTPUT_DIR=str(out), MODEL_PATH=str(root),
                           SC2PATH=str(root), PYTHON_BIN=sys.executable,
                           PARENT_CHECKPOINT=str(root/'train8/checkpoints/global_step_8'),
                           FEEDBACK_STORAGE_ROOT=str(root), FEEDBACK_PYTHON_OVERLAY=str(root),
                           FEEDBACK_INCLUDE_ROOT=str(root))
                script = RECORD/('run_arm.sh' if stage == 'D' else 'run_long_stage.sh')
                subprocess.run(['bash', str(script), stage], cwd=ROOT, env=env, check=True,
                               capture_output=True, text=True)
                cfg = json.loads((out/'effective_config.json').read_text())['overrides']
                configs[stage] = {k: v.replace(str(out), '<RUN>').replace('dry-'+stage, '<ID>')
                                  for k, v in cfg.items()}
                if stage != 'D':
                    self.assertEqual(cfg['trainer.n_gpus_per_node'], '8')
                    self.assertEqual(cfg['env.rollout.n'], '8')
                    self.assertEqual(cfg['trainer.total_training_steps'], '96')
                for key in ('algorithm.gamma', 'actor_rollout_ref.rollout.temperature',
                            '+ray_init.runtime_env.env_vars.SC2_COMBAT_REWARD_SCALE'):
                    self.assertEqual(cfg[key], {'algorithm.gamma':'1.0',
                        'actor_rollout_ref.rollout.temperature':'0.8',
                        '+ray_init.runtime_env.env_vars.SC2_COMBAT_REWARD_SCALE':"'0.001'"}[key])
            for stage in ('initial-eval', 'step8-eval'):
                self.assertEqual(configs[stage]['trainer.val_only'], 'True')
                self.assertEqual(configs[stage]['actor_rollout_ref.rollout.val_kwargs.n'], '8')
                self.assertEqual(configs[stage]['actor_rollout_ref.rollout.val_kwargs.temperature'], '0.8')
                self.assertNotIn('+trainer.sc2_checkpoint_resume', configs[stage])
            for stage in ('train8', 'train96'):
                self.assertEqual(configs[stage]['trainer.val_only'], 'False')
                self.assertEqual(configs[stage]['trainer.save_freq'], '8')
                self.assertEqual(configs[stage]['+trainer.sc2_checkpoint_resume'], 'True')
            self.assertEqual(configs['train8']['trainer.total_epochs'], '8')
            self.assertEqual(configs['train8']['trainer.resume_mode'], 'disable')
            self.assertEqual(configs['train96']['trainer.total_epochs'], '96')
            self.assertEqual(configs['train96']['trainer.resume_mode'], 'resume_path')
            # All learning/game overrides match D. Only declared orchestration keys may differ.
            allowed = {'trainer.n_gpus_per_node', 'trainer.total_epochs', 'trainer.save_freq',
                       'trainer.resume_mode', 'trainer.resume_from_path',
                       '+ray_init.runtime_env.worker_process_setup_hook',
                       '+trainer.sc2_checkpoint_resume', 'trainer.del_local_ckpt_after_load',
                       'trainer.sc2_signal_max_batches'}
            self.assertEqual(configs['train8']['trainer.sc2_signal_max_batches'], '0')
            self.assertEqual(configs['train96']['trainer.sc2_signal_max_batches'], '0')
            for key in configs['D'].keys() | configs['train8'].keys():
                if key in allowed or any(t in key for t in ('path', '_dir', 'PATH', 'DIR', 'LOG', 'name')):
                    continue
                self.assertEqual(configs['D'].get(key), configs['train8'].get(key), key)


if __name__ == '__main__':
    unittest.main()
