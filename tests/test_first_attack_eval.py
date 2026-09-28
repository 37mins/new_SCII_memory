"""First-attack-only hook, retained prefix, and opt-in four-card configuration."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT/'experiments/records/20260911-execution-failure-feedback-v1'
sys.path[:0] = [str(RECORD),str(ROOT/'experiments/records/20260908-g2-planner-handoff-v1')]
try:
    import first_attack_eval as mod
finally:
    del sys.path[:2]


class FirstAttackEvalTest(unittest.TestCase):
    def test_only_first_threshold_changes(self):
        original=dict(first_attack_threshold=20,subsequent_attack_threshold=10,
                      harassment_triggers=[],enable_early_scv_defense=False,early_scv_defense_threshold=5)
        for threshold in (20,40):
            p=SimpleNamespace(strategy_params=dict(original))
            proof=mod.set_threshold(p,threshold)
            self.assertEqual(proof['before'],original)
            self.assertEqual(p.strategy_params,{**original,'first_attack_threshold':threshold})
            self.assertEqual(proof['after'],p.strategy_params)
        with self.assertRaises(ValueError): mod.set_threshold(SimpleNamespace(strategy_params=original),30)

    def test_same_historical_answers_and_online_passthrough(self):
        for i,row in enumerate(mod.FIXTURE['rows']):
            self.assertEqual(mod.select_answer(i,row['prompt'],'unused generation'),row['answer'])
            with self.assertRaises(ValueError): mod.select_answer(i,'different state','unused')
        self.assertEqual(mod.select_answer(2,'new state','online A4'),'online A4')
        self.assertEqual(mod.GAME_SEEDS,[42613]*4)
        rng=mod.feedback_eval.PairedSeeds(mod.GAME_SEEDS)
        self.assertEqual(rng.randint(65536,2**32-1,4).tolist(),[42613]*4)
        with self.assertRaises(ValueError): rng.randint(65536,2**32-1,4)

    def test_dry_run_ab_only_changes_threshold_and_default_is_unchanged(self):
        sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        configs=[]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'model').mkdir();(root/'sc2').mkdir()
            for label in ('default','20','40'):
                out=root/label
                env=dict(os.environ,DRY_RUN='1',VERIFY_ASSETS='0',PYTHON_BIN=sys.executable,
                    SOURCE_SHA=sha,MODEL_PATH=str(root/'model'),SC2PATH=str(root/'sc2'),
                    FEEDBACK_STORAGE_ROOT=str(root/'storage'),FEEDBACK_LEARNING_MODEL='A4',
                    FEEDBACK_EVAL_GPUS='0,1,2,3',FEEDBACK_SECOND_PLAN='0',FEEDBACK_TEMPERATURE_AB='',
                    FEEDBACK_FIRST_ATTACK_THRESHOLD='' if label=='default' else label,
                    RUN_ID='first-attack-test',OUTPUT_DIR=str(out),RAY_TMPDIR='/dev/shm/fa_test')
                r=subprocess.run(['bash',str(RECORD/'run_feedback_eval_node47.sh'),'B'],cwd=ROOT,
                                 env=env,text=True,capture_output=True,timeout=30)
                self.assertEqual(r.returncode,0,r.stdout+r.stderr)
                raw=json.loads((out/'effective_config.json').read_text())['overrides']
                configs.append({k:v.replace(str(out),'<RUN>') for k,v in raw.items()})
                manifest=json.loads((out/'manifest.json').read_text())
                self.assertEqual((manifest['mode'],manifest['world_size'],manifest['gpu_selection']),
                                 ('evaluation',4,'0,1,2,3'))
        default,a,b=configs
        key='+ray_init.runtime_env.env_vars.SC2_FIRST_ATTACK_THRESHOLD_AB'
        self.assertEqual({k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)},{key})
        self.assertEqual([a[key],b[key]],["'20'","'40'"])
        self.assertEqual(default['+ray_init.runtime_env.worker_process_setup_hook'],'learning_eval.install')
        self.assertEqual(default['env.rollout.n'],'8')
        self.assertNotIn(key,default)
        for cfg in (a,b):
            for k,v in {'+ray_init.runtime_env.worker_process_setup_hook':'first_attack_eval.install',
                'trainer.val_only':'True','trainer.save_freq':'-1','env.rollout.n':'4','+env.val_rollout_n':'4',
                'actor_rollout_ref.rollout.val_kwargs.n':'4','data.max_response_length':'192',
                'actor_rollout_ref.rollout.temperature':'0.4','actor_rollout_ref.rollout.val_kwargs.temperature':'0.4',
                'trainer.del_local_ckpt_after_load':'False'}.items():
                self.assertEqual(cfg[k],v,(k,cfg[k]))
            self.assertTrue(cfg['trainer.resume_from_path'].endswith('A-rl4-4gpu-s0-r2/checkpoints/global_step_4'))
        subprocess.run(['bash','-n',str(RECORD/'run_first_attack_pair_node43.sh')],check=True)
        script=(RECORD/'run_first_attack_pair_node43.sh').read_text()
        self.assertIn('+ 2700',script)
        self.assertIn('--kill-after=15s',script)
        self.assertIn('SC2_FIRST_ATTACK_AB_AUTHORIZED',script)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'requires pinned CPU Torch')
    def test_setup_hook_does_not_probe_cuda_or_load_trainer(self):
        code="""
import sys, torch
def forbidden(*args, **kwargs): raise AssertionError('Premature CUDA probe')
torch._C._cuda_getDeviceCount=forbidden
torch.cuda._lazy_init=forbidden
import first_attack_eval
first_attack_eval.install()
assert 'verl.trainer.ppo.ray_trainer' not in sys.modules
assert not torch.cuda.is_initialized()
"""
        paths=[str(RECORD),str(ROOT/'experiments/records/20260910-spb-fsdp-layer-wrap-v1'),
               str(ROOT/'experiments/records/20260908-g2-planner-handoff-v1'),
               str(ROOT/'experiments/records/20260906-integrated-fixes-v1'),str(ROOT),os.environ.get('PYTHONPATH','')]
        for threshold in ('20','40'):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',SC2PATH=os.environ.get('SC2PATH',str(ROOT)),
                PYTHONPATH=os.pathsep.join(paths),SC2_FIRST_ATTACK_THRESHOLD_AB=threshold,
                SC2_FEEDBACK_LEARNING_MODEL='A4',SC2_EXECUTION_FAILURE_FEEDBACK='1')
            r=subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,text=True,capture_output=True,timeout=30)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)


if __name__=='__main__': unittest.main()
