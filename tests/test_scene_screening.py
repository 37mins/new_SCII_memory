"""Approved A4 screening: seed-only change to the existing evaluation path."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_feedback_learning_eval import module, ROOT, RECORD

SCREEN = ROOT/'experiments/records/20260914-zero-task-signal-sampling-v1'


class SceneScreeningTest(unittest.TestCase):
    def test_predeclared_seeds(self):
        mod = module('learning_eval')
        expected = [int.from_bytes(hashlib.sha256(
            f'20260914-task-signal-screening-v1/train/{i}'.encode()).digest()[:4], 'big') for i in range(8)]
        self.assertEqual(mod.SCREEN_GAME_SEEDS, expected)
        self.assertEqual(len(set(expected)), 8)
        self.assertFalse(set(expected) & set(mod.GAME_SEEDS))
        self.assertTrue(all(65536 <= s < 2**32-1 for s in expected))

    def test_hook_reuses_A4_and_exactly_one_eight_game_batch(self):
        mod = module('learning_eval')
        boundary = module('continue_a4')
        with patch.dict(sys.modules, continue_a4=boundary):
            for index, seed in enumerate(mod.SCREEN_GAME_SEEDS):
                with patch.dict(os.environ, SC2_FEEDBACK_LEARNING_MODEL='A4',
                                SC2_EXECUTION_FAILURE_FEEDBACK='0', SC2_SCREEN_SCENE_INDEX=str(index)):
                    with patch.object(mod.feedback_eval, 'install') as install:
                        mod.install_screen_scene()
                        install.assert_called_once_with(expected_step=4, game_seeds=[seed]*8)
                raw=SimpleNamespace(is_train=False, seed=1000, num_processes=8, reset_index=0)
                mod.feedback_eval.select_seeds(raw, [seed]*8)
                self.assertEqual(raw.validation_rng.randint(65536, 2**32-1, 8).tolist(), [seed]*8)
                with self.assertRaises(ValueError): raw.validation_rng.randint(65536, 2**32-1, 8)

    def test_wrong_checkpoint_feedback_or_scene_rejected(self):
        mod = module('learning_eval'); boundary = module('continue_a4')
        with patch.dict(sys.modules, continue_a4=boundary):
            for label, feedback, index in [('B4','0','0'), ('A5scene','0','0'),
                                           ('A4','1','0'), ('A4','0','-1'), ('A4','0','8')]:
                with patch.dict(os.environ, SC2_FEEDBACK_LEARNING_MODEL=label,
                                SC2_EXECUTION_FAILURE_FEEDBACK=feedback, SC2_SCREEN_SCENE_INDEX=index):
                    with self.assertRaises(ValueError): mod.install_screen_scene()

    def test_only_hook_and_scene_id_change_effective_config(self):
        sha = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
        configs = []
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/'model').mkdir(); (p/'sc2').mkdir()
            for index in range(-1,8):
                out=p/str(index)
                env=dict(os.environ, DRY_RUN='1', VERIFY_ASSETS='0', PYTHON_BIN=sys.executable,
                    SOURCE_SHA=sha, MODEL_PATH=str(p/'model'), SC2PATH=str(p/'sc2'),
                    FEEDBACK_STORAGE_ROOT=str(p/'storage'), FEEDBACK_EVAL_GPUS='0,1,2,3',
                    FEEDBACK_LEARNING_MODEL='A4', FEEDBACK_SCENE_EVAL='1',
                    FEEDBACK_SCENE_SET='single' if index==-1 else 'screening',
                    SC2_SCREEN_SCENE_INDEX=str(index), RUN_ID='scene-screen-test',
                    OUTPUT_DIR=str(out), RAY_TMPDIR='/tmp/scene-screen-test')
                result=subprocess.run(['bash',str(RECORD/'run_feedback_eval_node47.sh'),'A'],
                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                cfg={k:v.replace(str(out),'<RUN>') for k,v in
                     json.loads((out/'effective_config.json').read_text())['overrides'].items()}
                self.assertEqual(cfg['trainer.val_only'],'True')
                self.assertEqual(cfg['trainer.save_freq'],'-1')
                self.assertTrue(cfg['trainer.resume_from_path'].endswith('A-rl4-4gpu-s0-r2/checkpoints/global_step_4'))
                self.assertEqual(cfg['env.rollout.n'],'8')
                self.assertEqual(cfg['+ray_init.runtime_env.env_vars.SC2_EXECUTION_FAILURE_FEEDBACK'],"'0'")
                configs.append(cfg)
        flag='+ray_init.runtime_env.env_vars.SC2_SCREEN_SCENE_INDEX'
        hook='+ray_init.runtime_env.worker_process_setup_hook'
        for index,cfg in enumerate(configs[1:]):
            self.assertEqual({k for k in configs[0].keys()|cfg.keys() if configs[0].get(k)!=cfg.get(k)}, {flag,hook})
            self.assertEqual(cfg[flag],f"'{index}'")
            self.assertEqual(cfg[hook],'learning_eval.install_screen_scene')

    def test_controller_budget_and_no_concurrent_load(self):
        script=SCREEN/'run_scene_screen_node43.sh'
        self.assertEqual(subprocess.run(['bash','-n',str(script)]).returncode,0)
        text=script.read_text()
        for required in ('SC2_SCENE_SCREEN_AUTHORIZED', 'pair-r1/done', '+ 5400',
                         'for index in 0 1 2 3 4 5 6 7', 'FEEDBACK_EVAL_GPUS=0,1,2,3',
                         'test ! -e "${OUTPUT_DIR}"', '--kill-after=15s',
                         "result['training_updates']==0", 'assert actual==[[seed]*8]'):
            self.assertIn(required,text)
        for forbidden in ('pkill','run_feedback_train','rm -rf','hf download'):
            self.assertNotIn(forbidden,text)
        suffix='/session_2026-09-14_23-59-59_999999_2147483647/sockets/plasma_store'
        for index in range(8):
            self.assertLessEqual(len(f'/dev/shm/ss_{index}_0914_r1{suffix}'.encode()),107)


if __name__=='__main__': unittest.main()
