"""CPU completion/entry tests, with tiny fake artifacts and no SC2/model."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import run_record


class SignalRecordTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/'checkpoints').mkdir()

    def write(self, name, data, jsonl=False):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(map(json.dumps, data)) if jsonl else json.dumps(data))

    def fixture(self, accepted=(1, 3)):
        count, step = len(accepted), 4
        self.write('manifest.json', dict(mode='training', start_step=4, target_steps=8,
            world_size=4, sc2_signal_max_batches=4, episode_api='gymnasium-v0.26'))
        self.write('effective_config.json', {'overrides': {'data.train_batch_size': '1', 'env.rollout.n': '8'}})
        ledger = []
        for index in range(1, 5):
            keep = index in accepted
            ledger.append(dict(action='accept' if keep else 'skip_zero_signal', has_task_signal=keep,
                               pending_global_step=step+1, collection_index=index, reset_index=index-1))
            step += int(keep)
        self.write('checkpoints/signal_sampling.jsonl', ledger, True)
        self.write('checkpoints/signal_sampling_summary.json', dict(sampled_batches=4, sampled_games=32,
            start_global_step=4, last_global_step=4+count, completed_global_steps=count,
            skipped_batches=4-count))
        self.write('episode_outcomes.jsonl', [dict(reset_index=r, env_index=i,
            game_result='Victory', terminated=True, truncated=False, policy_decision_count=2)
            for r in range(4) for i in range(8)], True)
        (self.root/'console.log').write_text('\n'.join(
            f'step:{s} - actor/grad_norm:1.0 - episode/success_rate:1.0' for s in range(5, 5+count)))
        if count:
            base = f'checkpoints/global_step_{4+count}'
            self.write(base+'/data.pt', 1)
            for kind in ('model', 'optim', 'extra_state'):
                for rank in range(4):
                    self.write(f'{base}/actor/{kind}_world_size_4_rank_{rank}.pt', 1)

    def test_two_real_updates_counts_all_games_without_claiming_step8(self):
        self.fixture()
        result = run_record.finish(self.root, 0)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual((result['completed_steps'], result['last_global_step']), (2, 6))
        self.assertEqual(result['episode_outcomes']['natural_games'], 32)
        self.assertEqual(result['target_steps'], 8)
        self.assertFalse(result['update_target_reached'])
        self.assertFalse(result['learning_validated'])

    def test_no_updates_is_not_training_acceptance(self):
        self.fixture(())
        result = run_record.finish(self.root, 0)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['last_global_step'], 4)
        self.assertFalse(result['checkpoint_complete'])

    def test_missing_summary_cannot_pass_as_ordinary_training(self):
        self.fixture((1, 2, 3, 4))
        (self.root/'checkpoints/signal_sampling_summary.json').unlink()
        self.assertEqual(run_record.finish(self.root, 0)['status'], 'failed')

    def test_missing_skipped_games_rejected(self):
        self.fixture()
        path = self.root/'episode_outcomes.jsonl'
        path.write_text('\n'.join(path.read_text().splitlines()[:24]))
        self.assertEqual(run_record.finish(self.root, 0)['status'], 'failed')

    def test_wrong_counter_or_nonfinite_metric_rejected(self):
        self.fixture()
        path = self.root/'checkpoints/signal_sampling_summary.json'
        summary = json.loads(path.read_text()); summary['completed_global_steps'] = 4
        path.write_text(json.dumps(summary))
        self.assertEqual(run_record.finish(self.root, 0)['status'], 'failed')
        self.fixture()
        log = self.root/'console.log'; log.write_text(log.read_text().replace('grad_norm:1.0', 'grad_norm:nan'))
        self.assertEqual(run_record.finish(self.root, 0)['status'], 'failed')

    def test_control_four_updates(self):
        self.fixture((1, 2, 3, 4))
        result = run_record.finish(self.root, 0)
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(result['update_target_reached'])


class SignalEntryTest(unittest.TestCase):
    def test_pair_path_defaults_and_explicit_relocation(self):
        entry = ROOT/'experiments/records/20260914-zero-task-signal-sampling-v1/run_pair_node43.sh'
        assignments = '\n'.join(line for line in entry.read_text().splitlines()
            if line.startswith(('export FEEDBACK_STORAGE_ROOT=', 'export PYTHON_BIN=', 'export PYTHONPATH=')))
        script = 'source_dir=/frozen\n' + assignments + '\nprintf "%s\\n" "$FEEDBACK_STORAGE_ROOT" "$PYTHONPATH"'
        env = dict(os.environ)
        for key in ('FEEDBACK_STORAGE_ROOT', 'FEEDBACK_PYTHON_OVERLAY', 'PYTHON_BIN'):
            env.pop(key, None)
        default = subprocess.check_output(['bash', '-c', script], env=env, text=True).splitlines()
        self.assertEqual(default[0], '/root/new-scii-execution-feedback-local-20260911/shared')
        self.assertTrue(default[1].endswith(':/root/new-scii-spb-level5-96-20260908/python'))
        env.update(FEEDBACK_STORAGE_ROOT='/new/shared', FEEDBACK_PYTHON_OVERLAY='/new/python')
        moved = subprocess.check_output(['bash', '-c', script], env=env, text=True).splitlines()
        self.assertEqual(moved[0], '/new/shared')
        self.assertEqual(moved[1], default[1].rsplit(':', 1)[0] + ':/new/python')

    def test_replicate_names_preserve_r1_default_and_reject_path_injection(self):
        entry = ROOT/'experiments/records/20260914-zero-task-signal-sampling-v1/run_pair_node43.sh'
        # Evaluate only the ID-setting header, before auth, mkdir, GPU or game work.
        header = entry.read_text().split('record_dir=', 1)[0]
        for value, valid in [('r1', True), ('r2', True), ('r12', True), ('../r2', False), ('r0', False)]:
            run = subprocess.run(['bash', '-c', header + '\necho "$variant"'],
                env=dict(os.environ, SIGNAL_REPLICATE=value, SC2_SCENE_SAMPLING='0',
                         SC2_ZERO_SIGNAL_AUTHORIZED='1'), capture_output=True, text=True)
            self.assertEqual(run.returncode == 0, valid)
            if valid:
                self.assertEqual(run.stdout.strip(), 'pair-' + value)
        source = entry.read_text()
        self.assertIn('replicate=${SIGNAL_REPLICATE:-r1}', source)
        self.assertIn('-32games-4gpu-${replicate}', source)
        self.assertIn('${ray_date}_${replicate}', source)
        self.assertNotIn('-32games-4gpu-r1', source)

    def test_scene_retry_changes_only_dependency_paths(self):
        entry = ROOT/'experiments/records/20260914-zero-task-signal-sampling-v1/run_arm_node43.sh'
        sha = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        configs = []
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); (folder/'model').mkdir(); (folder/'sc2').mkdir()
            for label in ('original','relocated'):
                out = folder/label
                env = dict(os.environ,DRY_RUN='1',VERIFY_ASSETS='0',PYTHON_BIN=sys.executable,
                    SOURCE_SHA=sha,MODEL_PATH=str(folder/'model'),SC2PATH=str(folder/'sc2'),
                    RUN_ID='scene-path-test',OUTPUT_DIR=str(out),FEEDBACK_STORAGE_ROOT=str(folder),
                    RAY_TMPDIR='/tmp/scene-path-test',SC2_SCENE_SAMPLING='1')
                env.pop('FEEDBACK_PYTHON_OVERLAY',None); env.pop('FEEDBACK_INCLUDE_ROOT',None)
                if label=='relocated':
                    env.update(FEEDBACK_PYTHON_OVERLAY='/dev/shm/scene-retry/python',
                               FEEDBACK_INCLUDE_ROOT='/dev/shm/scene-retry/include')
                run = subprocess.run(['bash',str(entry),'B'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=30)
                self.assertEqual(run.returncode,0,run.stdout+run.stderr)
                cfg=json.loads((out/'effective_config.json').read_text())['overrides']
                configs.append({k:v.replace(str(out),'<RUN>') for k,v in cfg.items()})
        a,b=configs
        changed={k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
        self.assertEqual(changed,{'+ray_init.runtime_env.env_vars.PYTHONPATH'})
        self.assertEqual(b['+ray_init.runtime_env.env_vars.PYTHONPATH'],
            a['+ray_init.runtime_env.env_vars.PYTHONPATH'].replace(
                '/root/new-scii-spb-level5-96-20260908/python','/dev/shm/scene-retry/python'))
        # CPATH is inherited from the driver, not a Hydra override in the original entry.
        launcher=ROOT/'experiments/records/20260911-execution-failure-feedback-v1/run_feedback_train4_node43.sh'
        self.assertIn('export CPATH=${include_root}/python3.12:${include_root}',launcher.read_text())

    def test_scene_ab_changes_only_seed_hook_and_never_skips(self):
        entry = ROOT/'experiments/records/20260914-zero-task-signal-sampling-v1/run_arm_node43.sh'
        sha = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
        configs = []
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); (folder/'model').mkdir(); (folder/'sc2').mkdir()
            for arm in ('A','B'):
                out = folder/arm
                env = dict(os.environ, DRY_RUN='1', VERIFY_ASSETS='0', PYTHON_BIN=sys.executable,
                    SOURCE_SHA=sha, MODEL_PATH=str(folder/'model'), SC2PATH=str(folder/'sc2'),
                    RUN_ID='scene-sampling-test', OUTPUT_DIR=str(out), FEEDBACK_STORAGE_ROOT=str(folder),
                    RAY_TMPDIR='/tmp/scene-sampling-test', SC2_SCENE_SAMPLING='1')
                run = subprocess.run(['bash',str(entry),arm],cwd=ROOT,env=env,capture_output=True,text=True,timeout=30)
                self.assertEqual(run.returncode,0,run.stdout+run.stderr)
                cfg = json.loads((out/'effective_config.json').read_text())['overrides']
                configs.append({k:v.replace(str(out),'<RUN>') for k,v in cfg.items()})
                manifest = json.loads((out/'manifest.json').read_text())
                self.assertEqual((manifest['start_step'],manifest['target_steps'],manifest['world_size']),(4,8,4))
                self.assertEqual(cfg['trainer.sc2_skip_zero_signal'],'False')
                self.assertEqual(cfg['trainer.sc2_signal_max_batches'],'4')
                self.assertEqual(cfg['env.rollout.n'],'8')
                self.assertEqual(cfg['+ray_init.runtime_env.env_vars.SC2_EXECUTION_FAILURE_FEEDBACK'],"'0'")
                self.assertEqual(cfg['trainer.total_training_steps'],'96')
                self.assertEqual(cfg['trainer.save_freq'],'4')
                self.assertTrue(cfg['trainer.resume_from_path'].endswith('/global_step_4'))
        a,b = configs
        self.assertEqual({k:(a.get(k),b.get(k)) for k in a.keys()|b.keys() if a.get(k)!=b.get(k)},
            {'+ray_init.runtime_env.worker_process_setup_hook': ('continue_a4.install_signal_A','continue_a4.install_signal_B')})
        controller = entry.with_name('run_pair_node43.sh').read_text()
        self.assertIn('SC2_SCENE_SAMPLING_AUTHORIZED',controller)
        self.assertIn('deadline=$(( $(date +%s) + 10800 ))',controller)
        self.assertNotIn('run_feedback_eval',controller)

    def test_ab_changes_only_skip_flag_and_keeps_parent_and_game_budget(self):
        entry = ROOT/'experiments/records/20260914-zero-task-signal-sampling-v1/run_arm_node43.sh'
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); (folder/'model').mkdir(); (folder/'sc2').mkdir()
            configs = []
            for arm in ('A', 'B'):
                out = folder/arm
                env = dict(os.environ, DRY_RUN='1', VERIFY_ASSETS='0', PYTHON_BIN=sys.executable,
                    SOURCE_SHA=sha, MODEL_PATH=str(folder/'model'), SC2PATH=str(folder/'sc2'),
                    RUN_ID='signal-config-test', OUTPUT_DIR=str(out), FEEDBACK_STORAGE_ROOT=str(folder),
                    RAY_TMPDIR='/tmp/signal-config-test')
                run = subprocess.run(['bash', str(entry), arm], cwd=ROOT, env=env,
                    capture_output=True, text=True, timeout=30)
                self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
                cfg = json.loads((out/'effective_config.json').read_text())['overrides']
                configs.append({k:v.replace(str(out), '<RUN>') for k,v in cfg.items()})
                manifest = json.loads((out/'manifest.json').read_text())
                self.assertEqual(manifest['sc2_signal_max_batches'], 4)
                self.assertEqual(manifest['start_step'], 4)
                self.assertEqual(manifest['world_size'], 4)
                self.assertTrue(cfg['trainer.resume_from_path'].endswith('/global_step_4'))
                self.assertEqual(cfg['trainer.sc2_signal_max_batches'], '4')
                self.assertEqual(cfg['env.rollout.n'], '8')
                self.assertEqual(cfg['+ray_init.runtime_env.worker_process_setup_hook'], 'continue_a4.install_coverage')
            a, b = configs
            self.assertEqual({k:(a.get(k), b.get(k)) for k in a.keys()|b.keys() if a.get(k)!=b.get(k)},
                             {'trainer.sc2_skip_zero_signal': ('False', 'True')})


if __name__ == '__main__':
    unittest.main()
