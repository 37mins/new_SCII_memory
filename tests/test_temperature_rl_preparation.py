"""Prepared temperature-only continuation; no GPU, game or checkpoint load."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_feedback_learning_eval import module, ROOT, RECORD


class TemperatureRLPreparationTest(unittest.TestCase):
    def test_training_only_changes_temperature_and_eval_keeps_old_conditions(self):
        sha = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        configs = {}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/'model').mkdir(); (p/'sc2').mkdir()
            for label in ('train04','train08','A8','T08A8'):
                train=label.startswith('train'); out=p/label
                env=dict(os.environ,DRY_RUN='1',VERIFY_ASSETS='0',PYTHON_BIN=sys.executable,
                    SOURCE_SHA=sha,MODEL_PATH=str(p/'model'),SC2PATH=str(p/'sc2'),
                    FEEDBACK_STORAGE_ROOT=str(p/'storage'),OUTPUT_DIR=str(out),RUN_ID='temp-rl-test',
                    RAY_TMPDIR='/tmp/temp-rl-test',FEEDBACK_RL_GPUS='0,1,2,3',FEEDBACK_EVAL_GPUS='0,1,2,3',
                    FEEDBACK_CONTINUE_A4='1' if train else '0',FEEDBACK_RL_TEMPERATURE='0.8' if label=='train08' else '',
                    FEEDBACK_LEARNING_MODEL='' if train else label,FEEDBACK_TEMPERATURE_AB='',FEEDBACK_SECOND_PLAN='0')
                entry='run_feedback_train4_node43.sh' if train else 'run_feedback_eval_node47.sh'
                r=subprocess.run(['bash',str(RECORD/entry),'A' if train else 'B'],cwd=ROOT,
                                 env=env,text=True,capture_output=True,timeout=30)
                self.assertEqual(r.returncode,0,r.stdout+r.stderr)
                cfg=json.loads((out/'effective_config.json').read_text())['overrides']
                configs[label]={k:v.replace(str(out),'<RUN>') for k,v in cfg.items()}
                manifest=json.loads((out/'manifest.json').read_text())
                self.assertEqual(manifest['world_size'],4)
                if train:
                    self.assertEqual((manifest['start_step'],manifest['target_steps']),(4,8))
                    self.assertEqual(cfg['trainer.save_freq'],'4')
                    self.assertEqual(cfg['+ray_init.runtime_env.env_vars.SC2_EXECUTION_FAILURE_FEEDBACK'],"'0'")
                else:
                    self.assertEqual(cfg['trainer.val_only'],'True')
                    self.assertEqual(cfg['trainer.save_freq'],'-1')
                    self.assertEqual(cfg['actor_rollout_ref.rollout.val_kwargs.temperature'],'0.4')
                    self.assertEqual(cfg['+ray_init.runtime_env.env_vars.SC2_EXECUTION_FAILURE_FEEDBACK'],"'1'")
                self.assertEqual(cfg['actor_rollout_ref.actor.optim.lr'],'1e-6')
                self.assertEqual(cfg['data.max_response_length'],'192')
        def changed(a,b): return {k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
        self.assertEqual(changed(configs['train04'],configs['train08']),{'actor_rollout_ref.rollout.temperature'})
        self.assertEqual(configs['train08']['actor_rollout_ref.rollout.temperature'],'0.8')
        self.assertEqual(changed(configs['A8'],configs['T08A8']),
            {'trainer.resume_from_path','+ray_init.runtime_env.env_vars.SC2_FEEDBACK_LEARNING_MODEL'})

    def test_candidate_eval_uses_same_eight_seeds(self):
        mod=module('learning_eval')
        with patch.dict(os.environ,SC2_FEEDBACK_LEARNING_MODEL='T08A8',SC2_EXECUTION_FAILURE_FEEDBACK='1'):
            with patch.object(mod.feedback_eval,'install') as install:
                mod.install()
                install.assert_called_once_with(expected_step=8,game_seeds=mod.GAME_SEEDS)

    def test_controller_requires_separate_authority_and_keeps_budget(self):
        entry=RECORD/'run_a4_continue_node43.sh'
        self.assertEqual(subprocess.run(['bash','-n',str(entry)]).returncode,0)
        text=entry.read_text()
        for marker in ('SC2_TEMPERATURE_RL_AUTHORIZED','+ 7200','--kill-after=15s',
                       'export FEEDBACK_LEARNING_MODEL=T08A8','test ! -e "${OUTPUT_DIR}"'):
            self.assertIn(marker,text)
        for name in ('t8r_0912_r1','t8v_0912_r1'):
            suffix='/session_2026-09-12_23-59-59_999999_2147483647/sockets/plasma_store'
            self.assertLessEqual(len(('/dev/shm/'+name+suffix).encode()),107)
        self.assertNotIn('pkill',text)


if __name__=='__main__': unittest.main()
