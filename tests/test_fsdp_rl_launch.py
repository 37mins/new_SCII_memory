"""CPU effective-config comparison; no GPU or real model is used."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/records/20260909-bo-segment-protocol-v1/run_quota_train4_node44.sh'
CANDIDATE = ROOT / 'experiments/records/20260910-spb-fsdp-layer-wrap-v1/run_train1_node44.sh'
LONG = ROOT / 'experiments/records/20260910-spb-fsdp-layer-wrap-v1/run_train96_node43.sh'
LEVEL7 = LONG.with_name('run_train4_level7_node44.sh')


class FSDPRLLaunchTest(unittest.TestCase):
    def test_level7_resume_changes_only_checkpoint_controls_and_budget(self):
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'model').mkdir()
            (folder / 'sc2').mkdir()
            configs = []
            for label, entry in [('short', LEVEL7), ('resume', LEVEL7.with_name('run_resume96_level7_node44.sh'))]:
                output = folder / label
                env = dict(os.environ, DRY_RUN='1', VERIFY_ASSETS='0', PYTHON_BIN=sys.executable,
                           SOURCE_SHA=sha, MODEL_PATH=str(folder/'model'), SC2PATH=str(folder/'sc2'),
                           RUN_ID='resume-entry-test', OUTPUT_DIR=str(output), RAY_TMPDIR='/tmp/resume-entry-test')
                result = subprocess.run(['bash', str(entry)], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                cfg = json.loads((output/'effective_config.json').read_text())['overrides']
                configs.append({k:v.replace(str(output), '<RUN>') for k,v in cfg.items()})
            short, resume = configs
            changed = {k:(short.get(k), resume.get(k)) for k in short.keys() | resume.keys()
                       if short.get(k) != resume.get(k)}
            checkpoint = '/mnt/data0601/yangjinyue/new_SCII/experiments/20260910-spb-fsdp-layer-wrap-v1/runs/20260910-spb-fsdp-layer-wrap-v1-spb2-l7-g2-opening-quota-rl4-4gpu-s0-r1/checkpoints/global_step_4'
            self.assertEqual(changed, {
                'trainer.total_epochs': ('4', '96'), 'trainer.save_freq': ('4', '8'),
                'trainer.resume_mode': ('disable', 'resume_path'),
                'trainer.resume_from_path': (None, checkpoint),
                'trainer.del_local_ckpt_after_load': (None, 'False'),
                '+trainer.sc2_checkpoint_resume': (None, 'True'),
                '+trainer.sc2_import_legacy_final': (None, 'True'),
            })
            manifest = json.loads((folder/'resume/manifest.json').read_text())
            self.assertEqual((manifest['start_step'], manifest['target_steps'], manifest['world_size']), (4, 96, 4))
            self.assertTrue(manifest['sc2_checkpoint_resume'])

    def test_only_wrap_and_acceptance_duration_change(self):
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'model').mkdir()
            (folder / 'sc2').mkdir()
            configs = []
            for label, entry in [('base', BASE), ('candidate', CANDIDATE)]:
                output = folder / label
                env = dict(os.environ, DRY_RUN='1', VERIFY_ASSETS='0', PYTHON_BIN=sys.executable,
                           SOURCE_SHA=sha, MODEL_PATH=str(folder / 'model'), SC2PATH=str(folder / 'sc2'),
                           RUN_ID='fsdp-entry-test', OUTPUT_DIR=str(output), RAY_TMPDIR='/tmp/fsdp-entry-test')
                run = subprocess.run(['bash', str(entry)], env=env, capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                cfg = json.loads((output / 'effective_config.json').read_text())['overrides']
                configs.append({k: v.replace(str(output), '<RUN>') for k, v in cfg.items()})
            base, candidate = configs
            changed = {k: (base.get(k), candidate.get(k)) for k in base.keys() | candidate.keys()
                       if base.get(k) != candidate.get(k)}
            self.assertEqual(changed, {
                'trainer.total_epochs': ('4', '1'),
                '++actor_rollout_ref.rollout.fsdp_layer_wrap': (None, 'True'),
            })
            manifest = json.loads((folder / 'candidate/manifest.json').read_text())
            self.assertEqual((manifest['mode'], manifest['world_size'], manifest['target_steps'],
                              manifest['training_goal_global_steps']), ('training', 4, 1, 96))
            self.assertEqual(candidate['trainer.save_freq'], '1')
            self.assertEqual(candidate['trainer.resume_mode'], 'disable')
            self.assertEqual(candidate['env.rollout.n'], '8')

    def test_long_run_only_changes_budget_cards_save_and_overlay_path(self):
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'model').mkdir()
            (folder / 'sc2').mkdir()
            configs = []
            for label, entry in [('short', CANDIDATE), ('long', LONG)]:
                output = folder / label
                env = dict(os.environ, DRY_RUN='1', VERIFY_ASSETS='0', PYTHON_BIN=sys.executable,
                           SOURCE_SHA=sha, MODEL_PATH=str(folder / 'model'), SC2PATH=str(folder / 'sc2'),
                           RUN_ID='fsdp-entry-test', OUTPUT_DIR=str(output), RAY_TMPDIR='/tmp/fsdp-entry-test')
                run = subprocess.run(['bash', str(entry)], env=env, capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                cfg = json.loads((output / 'effective_config.json').read_text())['overrides']
                configs.append({k: v.replace(str(output), '<RUN>').replace(
                    '/root/new-scii-spb2-level5-control/python', '<OVERLAY>').replace(
                    '/root/new-scii-spb-level5-96-20260908/python', '<OVERLAY>') for k, v in cfg.items()})
            short, long = configs
            changed = {k: (short.get(k), long.get(k)) for k in short.keys() | long.keys()
                       if short.get(k) != long.get(k)}
            self.assertEqual(changed, {'trainer.total_epochs': ('1', '96'),
                                      'trainer.n_gpus_per_node': ('4', '8'),
                                      'trainer.save_freq': ('1', '4')})
            manifest = json.loads((folder / 'long/manifest.json').read_text())
            self.assertEqual((manifest['mode'], manifest['world_size'], manifest['target_steps']),
                             ('training', 8, 96))
            self.assertEqual(long['trainer.resume_mode'], 'disable')

    def test_level7_only_changes_difficulty_steps_and_save_from_four_gpu_gate(self):
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'model').mkdir()
            (folder / 'sc2').mkdir()
            configs = []
            for label, entry in [('level6', CANDIDATE), ('level7', LEVEL7)]:
                output = folder / label
                env = dict(os.environ, DRY_RUN='1', VERIFY_ASSETS='0', PYTHON_BIN=sys.executable,
                           SOURCE_SHA=sha, MODEL_PATH=str(folder / 'model'), SC2PATH=str(folder / 'sc2'),
                           RUN_ID='fsdp-entry-test', OUTPUT_DIR=str(output), RAY_TMPDIR='/tmp/fsdp-entry-test')
                run = subprocess.run(['bash', str(entry)], env=env, capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                cfg = json.loads((output / 'effective_config.json').read_text())['overrides']
                configs.append({k: v.replace(str(output), '<RUN>') for k, v in cfg.items()})
            level6, level7 = configs
            changed = {k: (level6.get(k), level7.get(k)) for k in level6.keys() | level7.keys()
                       if level6.get(k) != level7.get(k)}
            self.assertEqual(changed, {
                '+ray_init.runtime_env.env_vars.SC2_DIFFICULTY': ("'Harder'", "'VeryHard'"),
                '+ray_init.runtime_env.env_vars.SC2_EXPECTED_DIFFICULTY': ("'Harder'", "'VeryHard'"),
                'trainer.total_epochs': ('1', '4'),
                'trainer.save_freq': ('1', '4'),
            })
            manifest = json.loads((folder / 'level7/manifest.json').read_text())
            self.assertEqual((manifest['mode'], manifest['world_size'], manifest['target_steps']),
                             ('training', 4, 4))
            self.assertEqual(level7['env.rollout.n'], '8')
            self.assertEqual(level7['trainer.resume_mode'], 'disable')
            self.assertEqual(level7['trainer.total_training_steps'], '96')


class PredecessorAcceptanceTest(unittest.TestCase):
    def test_complete_predecessor_and_fail_closed_cases(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('check_rl1', LONG.with_name('check_rl1_acceptance.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {'source_commit': module.EXPECTED_SHA, 'model_manifest_sha256': module.MODEL_MANIFEST,
                        'dirty': False, 'formal': True, 'mode': 'training', 'world_size': 4, 'target_steps': 1}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            names = ['checkpoints/global_step_1/data.pt'] + [
                f'checkpoints/global_step_1/actor/{kind}_world_size_4_rank_{rank}.pt'
                for kind in ('model', 'optim', 'extra_state') for rank in range(4)]
            result = {'status': 'completed', 'process_exit_code': 0, 'completed_steps': 1,
                      'target_steps': 1, 'metrics_finite': True, 'checkpoint_complete': True,
                      'checkpoint_files': {n: {'bytes': 1} for n in names}}
            for name in names:
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b'x')
            for name in ['result.json', '.done']:
                (root / name).write_text(json.dumps(result))
            module.check(root)
            with self.subTest('missing completion'):
                (root / '.done').unlink()
                with self.assertRaises(FileNotFoundError):
                    module.check(root)
                (root / '.done').write_text(json.dumps(result))
            with self.subTest('failed or incomplete'):
                failed = dict(result, checkpoint_complete=False)
                for name in ['result.json', '.done']:
                    (root / name).write_text(json.dumps(failed))
                with self.assertRaises(ValueError):
                    module.check(root)
                for name in ['result.json', '.done']:
                    (root / name).write_text(json.dumps(result))
            with self.subTest('truncated checkpoint'):
                (root / names[-1]).write_bytes(b'')
                with self.assertRaises(ValueError):
                    module.check(root)
                (root / names[-1]).write_bytes(b'x')
            with self.subTest('wrong source'):
                (root / 'manifest.json').write_text(json.dumps(dict(manifest, source_commit='other')))
                with self.assertRaises(ValueError):
                    module.check(root)


if __name__ == '__main__':
    unittest.main()
