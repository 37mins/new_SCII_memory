"""CPU-only checks for the paired temperature launcher and seed boundary."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT/'experiments/records/20260910-spb-fsdp-layer-wrap-v1'


class TemperatureEvaluationTest(unittest.TestCase):
    def test_effective_config_diff_is_only_temperature(self):
        sha = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'model').mkdir()
            (folder/'sc2').mkdir()
            configs = []
            for arm in ['A','B']:
                out = folder/arm
                env = dict(os.environ, DRY_RUN='1', VERIFY_ASSETS='0', PYTHON_BIN=sys.executable,
                           SOURCE_SHA=sha, MODEL_PATH=str(folder/'model'), SC2PATH=str(folder/'sc2'),
                           OUTPUT_DIR=str(out), RUN_ID='temperature-config-test',
                           RAY_TMPDIR='/tmp/temperature-config-test')
                r = subprocess.run(['bash',str(RECORD/'run_temperature_eval_node44.sh'),arm],
                                   env=env, text=True, capture_output=True)
                self.assertEqual(r.returncode,0,r.stdout+r.stderr)
                raw = json.loads((out/'effective_config.json').read_text())
                cfg = {k:v.replace(str(out),'<RUN>') for k,v in raw['overrides'].items()}
                configs.append(cfg)
                manifest = json.loads((out/'manifest.json').read_text())
                self.assertEqual(manifest['mode'],'evaluation')
                self.assertEqual(manifest['world_size'],4)
                for key,value in {
                    'trainer.val_only':'True','trainer.val_before_train':'True',
                    'trainer.total_epochs':'1','trainer.save_freq':'-1',
                    'trainer.resume_mode':'resume_path','trainer.del_local_ckpt_after_load':'False',
                    'actor_rollout_ref.rollout.val_kwargs.do_sample':'True',
                    'actor_rollout_ref.rollout.val_kwargs.top_p':'0.9',
                    'env.rollout.n':'8','+env.val_rollout_n':'8',
                    'actor_rollout_ref.rollout.val_kwargs.n':'8','data.max_response_length':'192',
                    '++actor_rollout_ref.rollout.fsdp_layer_wrap':'True',
                    '+actor_rollout_ref.model.override_config.swa_dropout':'0.0',
                    'algorithm.gamma':'0.95','env.history_length':'0',
                }.items():
                    self.assertEqual(cfg[key],value,key)
                self.assertTrue(cfg['trainer.resume_from_path'].endswith('/global_step_24'))
                self.assertNotIn('+trainer.sc2_checkpoint_resume',cfg)
                self.assertEqual(cfg['+ray_init.runtime_env.worker_process_setup_hook'],'temperature_eval.install')
                for key,value in {'SC2_FIXED_BO_PREFIX_LIMIT':'-1','SC2_ORIGINAL_BO_SEGMENT':'1',
                                  'SC2_CLARIFY_BO_QUOTA':'1','SC2_DIFFICULTY':'Harder'}.items():
                    self.assertEqual(cfg['+ray_init.runtime_env.env_vars.'+key].strip("'"),value)
            a,b = configs
            changed = {k:(a.get(k),b.get(k)) for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
            self.assertEqual(changed,{
                'actor_rollout_ref.rollout.temperature':('0.4','1.0'),
                'actor_rollout_ref.rollout.val_kwargs.temperature':('0.4','1.0')})

    def test_eight_seeds_once_and_fresh_environment_only(self):
        path = ROOT/'experiments/records/20260908-g2-planner-handoff-v1'
        sys.path.insert(0,str(path))
        try:
            spec = importlib.util.spec_from_file_location('temperature_eval',RECORD/'temperature_eval.py')
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        finally:
            sys.path.remove(str(path))
        raw = SimpleNamespace(is_train=False,seed=1000,num_processes=8,reset_index=0)
        record = mod.select_seeds(raw)
        self.assertEqual(record['independent_scenarios'],2)
        self.assertEqual(raw.validation_rng.randint(2**16,2**32-1,8).tolist(),
                         [52620]*4+[17089]*4)
        with self.assertRaises(ValueError):
            raw.validation_rng.randint(2**16,2**32-1,8)
        raw.is_train = True
        with self.assertRaises(ValueError):
            mod.select_seeds(raw)


if __name__ == '__main__':
    unittest.main()
