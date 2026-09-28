"""Temperature A/B configuration checks, without a model or game."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import json
import unittest
from unittest.mock import patch

from test_feedback_learning_eval import module, ROOT, RECORD


class TemperatureEvaluationTest(unittest.TestCase):
    def test_controller_is_serial_bounded_and_authorized(self):
        entry = RECORD/'run_temperature_pair_node43.sh'
        self.assertEqual(subprocess.run(['bash','-n',str(entry)]).returncode,0)
        text = entry.read_text()
        for required in ('SC2_TEMPERATURE_AB_AUTHORIZED', '+ 2700',
                         'for temperature in 0.4 0.8', '--kill-after=15s',
                         "r['training_updates']==0", "actual==[[42613]*8]",
                         'test ! -e "${OUTPUT_DIR}"'):
            self.assertIn(required,text)
        self.assertNotIn('pkill',text)

    def test_hook_keeps_checkpoint_seed_and_feedback(self):
        mod = module('temperature_eval')
        self.assertEqual(mod.GAME_SEEDS, [42613]*8)
        for temperature in ('0.4','0.8'):
            with patch.dict(os.environ, SC2_FEEDBACK_TEMPERATURE_AB=temperature,
                            SC2_FEEDBACK_LEARNING_MODEL='A4', SC2_EXECUTION_FAILURE_FEEDBACK='1'):
                with patch.object(mod.feedback_eval, 'install') as install:
                    mod.install()
                    install.assert_called_once_with(expected_step=4, game_seeds=[42613]*8,
                                                    expected_temperature=float(temperature))
        with patch.dict(os.environ, SC2_FEEDBACK_TEMPERATURE_AB='0.8',
                        SC2_FEEDBACK_LEARNING_MODEL='B4', SC2_EXECUTION_FAILURE_FEEDBACK='1'):
            with self.assertRaises(ValueError): mod.install()

    def test_effective_config_is_temperature_only(self):
        sha = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        configs = {}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); (p/'model').mkdir(); (p/'sc2').mkdir()
            for temperature in ('', '0.4', '0.8'):
                out = p/(temperature or 'default')
                env = dict(os.environ, DRY_RUN='1', VERIFY_ASSETS='0', PYTHON_BIN=sys.executable,
                    SOURCE_SHA=sha, MODEL_PATH=str(p/'model'), SC2PATH=str(p/'sc2'),
                    FEEDBACK_STORAGE_ROOT=str(p/'storage'), FEEDBACK_LEARNING_MODEL='A4',
                    FEEDBACK_EVAL_GPUS='0,1,2,3', FEEDBACK_TEMPERATURE_AB=temperature,
                    FEEDBACK_SECOND_PLAN='0', RUN_ID='temperature-config-test',
                    OUTPUT_DIR=str(out), RAY_TMPDIR='/tmp/sc2-temperature-test')
                run = subprocess.run(['bash',str(RECORD/'run_feedback_eval_node47.sh'),'B'],
                    cwd=ROOT,env=env,text=True,capture_output=True,timeout=30)
                self.assertEqual(run.returncode,0,run.stdout+run.stderr)
                cfg = json.loads((out/'effective_config.json').read_text())['overrides']
                configs[temperature] = {k:v.replace(str(out),'<RUN>') for k,v in cfg.items()}
                self.assertEqual(cfg['trainer.val_only'],'True')
                self.assertEqual(cfg['trainer.save_freq'],'-1')
                self.assertTrue(cfg['trainer.resume_from_path'].endswith('A-rl4-4gpu-s0-r2/checkpoints/global_step_4'))
                self.assertEqual(cfg['env.rollout.n'],'8')
                self.assertEqual(cfg['data.max_response_length'],'192')
                self.assertEqual(cfg['actor_rollout_ref.rollout.val_kwargs.top_p'],'0.9')
                self.assertEqual(cfg['actor_rollout_ref.rollout.val_kwargs.top_k'],'0')
            def changed(a,b):
                return {k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
            flag = '+ray_init.runtime_env.env_vars.SC2_FEEDBACK_TEMPERATURE_AB'
            self.assertEqual(changed(configs[''],configs['0.4']),
                {flag,'+ray_init.runtime_env.worker_process_setup_hook'})
            self.assertEqual(changed(configs['0.4'],configs['0.8']), {flag,
                'actor_rollout_ref.rollout.temperature', 'actor_rollout_ref.rollout.val_kwargs.temperature'})


if __name__ == '__main__': unittest.main()
