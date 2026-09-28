"""Dry-run the real training entry: no models, GPU, or SC2."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / 'experiments/records/20260911-execution-failure-feedback-v1/run_feedback_train4_node43.sh'
BASE = ROOT / 'experiments/records/20260910-spb-fsdp-layer-wrap-v1/run_train1_node44.sh'


class FeedbackRLLaunchTest(unittest.TestCase):
    def configs(self, include_base=False, serial=False):
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp).resolve()
            (folder/'model').mkdir(); (folder/'sc2').mkdir()
            configs, manifests = {}, {}
            entries = [('A', ENTRY, ['A']), ('B', ENTRY, ['B'])]
            if include_base:
                entries.insert(0, ('base', BASE, []))
            for label, entry, args in entries:
                output = folder/label
                env = dict(os.environ, DRY_RUN='1', VERIFY_ASSETS='0', PYTHON_BIN=sys.executable,
                    SOURCE_SHA=sha, MODEL_PATH=str(folder/'model'), SC2PATH=str(folder/'sc2'),
                    RUN_ID='feedback-rl-config-test', OUTPUT_DIR=str(output),
                    FEEDBACK_STORAGE_ROOT=str(folder), RAY_TMPDIR='/tmp/feedback-rl-test',
                    SC2_EXECUTION_FAILURE_FEEDBACK='0')
                if serial:
                    env['FEEDBACK_RL_GPUS'] = '0,1,2,3'
                run = subprocess.run(['bash', str(entry), *args], cwd=ROOT, env=env,
                                     capture_output=True, text=True, timeout=30)
                self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
                cfg = json.loads((output/'effective_config.json').read_text())['overrides']
                configs[label] = {k:v.replace(str(output), '<RUN>').replace(str(folder), '<STORAGE>').replace(
                    '/mnt/data0601', '<STORAGE>').replace('/root/new-scii-spb2-level5-control/python', '<PYTHON>').replace(
                    '/root/new-scii-spb-level5-96-20260908/python', '<PYTHON>') for k,v in cfg.items()}
                manifests[label] = json.loads((output/'manifest.json').read_text())
            return configs, manifests

    def test_serial_keeps_training_config(self):
        parallel, _ = self.configs()
        serial, manifests = self.configs(serial=True)
        self.assertEqual(parallel, serial)
        for manifest in manifests.values():
            self.assertEqual(manifest['world_size'], 4)

    def test_matches_previous_four_card_gate(self):
        configs, _ = self.configs(include_base=True)
        a, b = configs['base'], configs['A']
        diff = {k:(a.get(k),b.get(k)) for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
        self.assertEqual(diff, {'trainer.total_epochs':('1','4'), 'trainer.save_freq':('1','4')})

    def test_only_feedback_differs_and_no_checkpoint_is_resumed(self):
        configs, manifests = self.configs()
        a, b = configs['A'], configs['B']
        diff = {k:(a.get(k),b.get(k)) for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
        self.assertEqual(diff, {
            '+ray_init.runtime_env.env_vars.SC2_EXECUTION_FAILURE_FEEDBACK':("'0'","'1'")})
        for arm in ('A','B'):
            cfg, manifest = configs[arm], manifests[arm]
            self.assertEqual((manifest['mode'],manifest['world_size'],manifest['target_steps']), ('training',4,4))
            self.assertEqual(cfg['trainer.resume_mode'],'disable')
            self.assertNotIn('trainer.resume_from_path',cfg)
            self.assertFalse(any('setup_hook' in k for k in cfg))
            self.assertEqual(cfg['trainer.val_only'],'False')
            self.assertEqual(cfg['env.rollout.n'],'8')
            self.assertEqual(cfg['algorithm.gamma'],'0.95')
            self.assertEqual(cfg['trainer.total_training_steps'],'96')


if __name__ == '__main__':
    unittest.main()
