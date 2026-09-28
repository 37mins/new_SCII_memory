"""CPU config checks: gamma is the only A/B treatment, eight new steps from step4."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT/'experiments/records/20260910-spb-fsdp-layer-wrap-v1'


class GammaABTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires pinned CPU Torch')
    def test_setup_hook_does_not_probe_cuda_or_import_trainer(self):
        code = '''
import sys, torch
def forbidden(*args, **kwargs):
    raise AssertionError('Premature CUDA probe in worker setup hook')
torch._C._cuda_getDeviceCount = forbidden
torch.cuda._lazy_init = forbidden
import gamma_ab
gamma_ab.install()
assert 'verl.trainer.ppo.ray_trainer' not in sys.modules
assert not torch.cuda.is_initialized()
print('Setup hook makes no CUDA probe and leaves trainer import deferred')
'''
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='',
                   PYTHONPATH=str(RECORD)+os.pathsep+os.environ.get('PYTHONPATH',''))
        r = subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,text=True,capture_output=True,timeout=30)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)

    def test_only_gamma_changes_and_budget_is_eight_new_steps(self):
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
                           OUTPUT_DIR=str(out), RUN_ID='gamma-config-test', RAY_TMPDIR='/tmp/gamma-test')
                r = subprocess.run(['bash',str(RECORD/'run_gamma_ab_node44.sh'),arm],
                                   env=env, text=True, capture_output=True)
                self.assertEqual(r.returncode,0,r.stdout+r.stderr)
                raw = json.loads((out/'effective_config.json').read_text())
                cfg = {k:v.replace(str(out),'<RUN>') for k,v in raw['overrides'].items()}
                configs.append(cfg)
                manifest = json.loads((out/'manifest.json').read_text())
                for key,value in {'mode':'training','world_size':8,'start_step':4,
                                  'target_steps':12,'training_goal_global_steps':96,
                                  'sc2_checkpoint_resume':False}.items():
                    self.assertEqual(manifest[key],value,key)
                for key,value in {'trainer.total_epochs':'8','trainer.save_freq':'4',
                                  'trainer.total_training_steps':'96','env.rollout.n':'8',
                                  'env.seed':'0','data.max_response_length':'192',
                                  'actor_rollout_ref.rollout.temperature':'0.4',
                                  'actor_rollout_ref.actor.optim.lr':'1e-6',
                                  'trainer.del_local_ckpt_after_load':'False',
                                  '+ray_init.runtime_env.worker_process_setup_hook':'gamma_ab.install',
                                  '++actor_rollout_ref.rollout.fsdp_layer_wrap':'True'}.items():
                    self.assertEqual(cfg[key],value,key)
                self.assertTrue(cfg['trainer.resume_from_path'].endswith('/global_step_4'))
            a,b = configs
            self.assertEqual({k:(a.get(k),b.get(k)) for k in a.keys()|b.keys() if a.get(k)!=b.get(k)},
                             {'algorithm.gamma':('0.95','1.0')})

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires pinned CPU Torch')
    def test_pairing_is_fresh_games_not_fabricated_sc2_resume(self):
        import torch
        spec = importlib.util.spec_from_file_location('gamma_ab',RECORD/'gamma_ab.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        trainer = SimpleNamespace(global_steps=4,train_dataloader=[],envs=SimpleNamespace(envs=SimpleNamespace(reset_index=0)),
                                  config=SimpleNamespace(trainer=SimpleNamespace(resume_from_path='/parent/global_step_4')))
        proof = mod.pair_driver(trainer)
        draw = torch.rand(4)
        self.assertFalse(proof['sc2_state_restored'])
        self.assertEqual(proof['game_seed_per_new_step'],[2732,43567,42613,52416,45891,21243,30403,32103])
        mod.pair_driver(trainer)
        self.assertTrue(torch.equal(draw,torch.rand(4)))
        self.assertEqual(trainer.global_steps,4)
        trainer.train_dataloader = [object()]
        with self.assertRaisesRegex(ValueError,'discard a sample'):
            mod.pair_driver(trainer)
        trainer.global_steps = 0
        with self.assertRaises(ValueError):
            mod.pair_driver(trainer)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires pinned CPU Torch')
    def test_close_saved_last_yield_then_run_eight_real_batches(self):
        import torch
        from torch.utils.data import TensorDataset, RandomSampler
        from torchdata.stateful_dataloader import StatefulDataLoader
        spec = importlib.util.spec_from_file_location('gamma_ab',RECORD/'gamma_ab.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        data = TensorDataset(torch.arange(1))
        def loader():
            return StatefulDataLoader(data,batch_size=1,num_workers=0,
                sampler=RandomSampler(data,generator=torch.Generator().manual_seed(1)))
        original = loader()
        next(iter(original))
        restored = loader()
        restored.load_state_dict(original.state_dict())
        trainer = SimpleNamespace(global_steps=4,train_dataloader=restored,
            envs=SimpleNamespace(envs=SimpleNamespace(reset_index=0)),
            config=SimpleNamespace(trainer=SimpleNamespace(resume_from_path='/parent/global_step_4')))
        mod.pair_driver(trainer)
        self.assertEqual([len(list(restored)) for _ in range(8)],[1]*8)


if __name__ == '__main__':
    unittest.main()
