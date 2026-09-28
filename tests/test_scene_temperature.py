"""CPU contract for SceneB8/feedback-off temperature-only evaluation."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_feedback_learning_eval import module, ROOT, RECORD

HERE = ROOT/'experiments/records/20260914-zero-task-signal-sampling-v1'


class SceneTemperatureTest(unittest.TestCase):
    def test_hook_restores_step8_and_exact_seed(self):
        mod = module('temperature_eval')
        for temperature in ('0.4', '0.8'):
            with patch.dict(os.environ, SC2_SCENE_TEMPERATURE=temperature, SC2_FEEDBACK_LEARNING_MODEL='SceneB8', SC2_EXECUTION_FAILURE_FEEDBACK='0'):
                with patch.object(mod.feedback_eval, 'install') as install:
                    mod.install_scene_b8()
                    install.assert_called_once_with(expected_step=8, game_seeds=[1436145665]*8, expected_temperature=float(temperature))
        for label, feedback, temperature in [('A4','0','0.4'),('SceneB8','1','0.8'),('SceneB8','0','1.0')]:
            with patch.dict(os.environ, SC2_SCENE_TEMPERATURE=temperature, SC2_FEEDBACK_LEARNING_MODEL=label, SC2_EXECUTION_FAILURE_FEEDBACK=feedback):
                with self.assertRaises(ValueError): mod.install_scene_b8()

    def test_effective_configs_only_temperature_differs(self):
        sha = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        with patch.dict(sys.modules, learning_eval=module('learning_eval'), temperature_eval=module('temperature_eval')):
            spec = importlib.util.spec_from_file_location('scene_posteval_check',HERE/'scene_posteval_check.py')
            common = importlib.util.module_from_spec(spec); spec.loader.exec_module(common)
            with patch.dict(sys.modules, scene_posteval_check=common):
                spec = importlib.util.spec_from_file_location('scene_temperature_check',HERE/'scene_temperature_check.py')
                check = importlib.util.module_from_spec(spec); spec.loader.exec_module(check)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/'model').mkdir(); (p/'sc2').mkdir(); paths=[]
            for temperature in ('0.4','0.8'):
                label='temp-'+temperature.replace('.',''); out=p/label; paths.append(out)
                env=dict(os.environ, DRY_RUN='1', VERIFY_ASSETS='0', PYTHON_BIN=sys.executable, SOURCE_SHA=sha,
                    MODEL_PATH=str(p/'model'), SC2PATH=str(p/'sc2'), FEEDBACK_STORAGE_ROOT=str(p/'storage'),
                    FEEDBACK_EVAL_GPUS='4,5,6,7', FEEDBACK_LEARNING_MODEL='SceneB8', FEEDBACK_SCENE_EVAL='0',
                    FEEDBACK_SCENE_POSTEVAL='1', FEEDBACK_SCENE_TEMPERATURE=temperature,
                    FEEDBACK_POST_CHECKPOINT=str(common.checkpoint('SceneB8')), RUN_ID=label,
                    OUTPUT_DIR=str(out), RAY_TMPDIR='/dev/shm/st'+label+'_0915_r1')
                result=subprocess.run(['bash',str(RECORD/'run_feedback_eval_node47.sh'),'A'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=30)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                cfg=json.loads((out/'effective_config.json').read_text())['overrides']
                for k,v in {'trainer.val_only':'True','trainer.save_freq':'-1','env.rollout.n':'8','env.history_length':'0',
                    '+env.sc2_memory_mode':'off','data.max_response_length':'192','+actor_rollout_ref.model.override_config.swa_dropout':'0.0',
                    '+ray_init.runtime_env.env_vars.SC2_EXECUTION_FAILURE_FEEDBACK':"'0'"}.items(): self.assertEqual(cfg[k],v,k)
            self.assertTrue(check.compare(paths)['manifests_match'])
            with self.assertRaises(AssertionError): common.compare(paths)
            cfg=json.loads((paths[1]/'effective_config.json').read_text())
            cfg['overrides']['data.max_response_length']='384'
            (paths[1]/'effective_config.json').write_text(json.dumps(cfg))
            with self.assertRaises(AssertionError): check.compare(paths)

    def test_four_card_serial_total_budget(self):
        path=HERE/'run_scene_temperature_node47.sh'; script=path.read_text()
        self.assertEqual(subprocess.run(['bash','-n',str(path)]).returncode,0)
        for text in ['+ 2700','FEEDBACK_EVAL_GPUS=4,5,6,7','arm 0.4\narm 0.8','deadline - $(date +%s) - 45','--kill-after=15s']:
            self.assertIn(text,script)
        for text in ['pkill','ray stop','run_feedback_train','rm -rf','hf download']:
            self.assertNotIn(text,script)
        suffix='/session_2026-09-15_23-59-59_999999_2147483647/sockets/plasma_store'
        self.assertLessEqual(len(('/dev/shm/st04_0915_r1'+suffix).encode()),107)


if __name__ == '__main__': unittest.main()
