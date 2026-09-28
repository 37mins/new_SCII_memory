"""A4 continuation/evaluation configuration; CPU only, no models or games."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_feedback_learning_eval import module, RECORD, ROOT


class A4ContinuationTest(unittest.TestCase):
    def test_training_and_evaluation_keep_existing_configs(self):
        sha = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        configs, manifests = {}, {}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp); (path/'model').mkdir(); (path/'sc2').mkdir()
            for label in ('train4','continue','scene','coverage','A4','A8'):
                train=label in ('train4','continue','scene','coverage'); out=path/label
                env=dict(os.environ,DRY_RUN='1',VERIFY_ASSETS='0',PYTHON_BIN=sys.executable,
                    SOURCE_SHA=sha,MODEL_PATH=str(path/'model'),SC2PATH=str(path/'sc2'),
                    FEEDBACK_STORAGE_ROOT=str(path/'storage'),OUTPUT_DIR=str(out),
                    RUN_ID='a4-test',RAY_TMPDIR='/tmp/a4-test',FEEDBACK_RL_GPUS='0,1,2,3',
                    FEEDBACK_EVAL_GPUS='0,1,2,3',FEEDBACK_CONTINUE_A4='1' if label in ('continue','scene','coverage') else '0',
                    FEEDBACK_A4_SCENE1='1' if label=='scene' else '0',
                    FEEDBACK_A4_COVERAGE='1' if label=='coverage' else '0')
                if not train: env['FEEDBACK_LEARNING_MODEL']=label
                entry='run_feedback_train4_node43.sh' if train else 'run_feedback_eval_node47.sh'
                run=subprocess.run(['bash',str(RECORD/entry),'A' if train else 'B'],
                    cwd=ROOT,env=env,capture_output=True,text=True,timeout=30)
                self.assertEqual(run.returncode,0,run.stdout+run.stderr)
                configs[label]={k:v.replace(str(out),'<RUN>') for k,v in json.loads((out/'effective_config.json').read_text())['overrides'].items()}
                manifests[label]=json.loads((out/'manifest.json').read_text())
        before, after=configs['train4'],configs['continue']
        path_key='+ray_init.runtime_env.env_vars.PYTHONPATH'
        self.assertEqual(after[path_key],before[path_key].replace(str(ROOT)+':',
            str(ROOT)+':'+str(RECORD)+':'+str(RECORD.parent/'20260910-spb-fsdp-layer-wrap-v1')+':',1))
        diff={k:(before.get(k),after.get(k)) for k in before.keys()|after.keys() if before.get(k)!=after.get(k)}
        self.assertEqual(set(diff),{'trainer.resume_mode','trainer.resume_from_path',
            'trainer.del_local_ckpt_after_load','+ray_init.runtime_env.worker_process_setup_hook',path_key})
        self.assertEqual(after['+ray_init.runtime_env.worker_process_setup_hook'],'continue_a4.install')
        self.assertEqual(after['trainer.del_local_ckpt_after_load'],'False')
        self.assertEqual(after['trainer.resume_mode'],'resume_path')
        self.assertIn('g2-A-rl4-4gpu-s0-r2/checkpoints/global_step_4',after['trainer.resume_from_path'])
        self.assertEqual((manifests['continue']['start_step'],manifests['continue']['target_steps']),(4,8))
        self.assertEqual(after['trainer.val_before_train'],'False')
        self.assertEqual(after['trainer.test_freq'],'-1')
        self.assertEqual(after['trainer.save_freq'],'4')
        scene = configs['scene']
        self.assertEqual({k for k in after.keys()|scene.keys() if after.get(k)!=scene.get(k)},
            {'trainer.total_epochs','trainer.save_freq','+ray_init.runtime_env.worker_process_setup_hook'})
        self.assertEqual(scene['trainer.total_epochs'],'1')
        self.assertEqual(scene['trainer.save_freq'],'1')
        self.assertEqual(scene['trainer.total_training_steps'],'96')  # frozen scheduler, not run budget
        self.assertEqual(scene['+ray_init.runtime_env.worker_process_setup_hook'],'continue_a4.install_scene')
        self.assertEqual((manifests['scene']['start_step'],manifests['scene']['target_steps']),(4,5))
        coverage=configs['coverage']
        self.assertEqual({k for k in after.keys()|coverage.keys() if after.get(k)!=coverage.get(k)},
            {'+ray_init.runtime_env.worker_process_setup_hook'})
        self.assertEqual(coverage['+ray_init.runtime_env.worker_process_setup_hook'],'continue_a4.install_coverage')
        self.assertEqual((manifests['coverage']['start_step'],manifests['coverage']['target_steps']),(4,8))
        self.assertEqual(coverage["+ray_init.runtime_env.env_vars.SC2_EXECUTION_FAILURE_FEEDBACK"],"'0'")
        before,after=configs['A4'],configs['A8']
        self.assertEqual({k for k in before.keys()|after.keys() if before.get(k)!=after.get(k)},
            {'trainer.resume_from_path','+ray_init.runtime_env.env_vars.SC2_FEEDBACK_LEARNING_MODEL'})
        self.assertTrue(after['trainer.resume_from_path'].endswith('/global_step_8'))
        for label,m in manifests.items():
            self.assertEqual((m['world_size'],m['gpu_selection']),(4,'0,1,2,3'))

    def test_step8_evaluation_reuses_the_same_seeds(self):
        mod=module('learning_eval')
        with patch.dict(os.environ,SC2_FEEDBACK_LEARNING_MODEL='A8',SC2_EXECUTION_FAILURE_FEEDBACK='1'):
            with patch.object(mod.feedback_eval,'install') as install:
                mod.install()
                install.assert_called_once_with(expected_step=8,game_seeds=mod.GAME_SEEDS)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'requires pinned CPU Torch')
    def test_exact_four_new_batches_and_declared_game_boundary(self):
        import torch
        import numpy as np
        from torch.utils.data import TensorDataset,RandomSampler
        from torchdata.stateful_dataloader import StatefulDataLoader
        sys.path.insert(0,str(RECORD.parent/'20260910-spb-fsdp-layer-wrap-v1'))
        self.addCleanup(sys.path.pop,0)
        mod=module('continue_a4')
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp)/mod.PARENT_NAME; parent.mkdir()
            (parent/'manifest.json').write_text(json.dumps(dict(source_commit=mod.PARENT_SHA,dirty=False,world_size=4)))
            (parent/'.done').write_text(json.dumps(dict(status='completed',completed_steps=4,checkpoint_complete=True)))
            (parent/'training_seeds.jsonl').write_text('\n'.join(json.dumps(dict(seeds=[s]*8)) for s in mod.PAST_SEEDS))
            data=TensorDataset(torch.arange(1))
            def loader():
                return StatefulDataLoader(data,batch_size=1,num_workers=0,
                    sampler=RandomSampler(data,generator=torch.Generator().manual_seed(1)))
            original=loader(); next(iter(original)); restored=loader(); restored.load_state_dict(original.state_dict())
            cfg=SimpleNamespace(resume_from_path=str(parent/'checkpoints/global_step_4'),val_only=False,total_epochs=4,get=lambda k,d:d)
            raw=SimpleNamespace(reset_index=0,is_train=True,seed=0,env_num=1,group_n=8,training_rng=np.random.RandomState(0))
            trainer=SimpleNamespace(global_steps=4,config=SimpleNamespace(trainer=cfg),train_dataloader=restored,envs=SimpleNamespace(envs=raw))
            proof=mod.restore_boundary(trainer)
            self.assertEqual([len(list(restored)) for _ in range(4)],[1]*4)
            self.assertEqual(raw.training_rng.randint(0,2**16-1,size=4).tolist(),mod.NEXT_SEEDS)
            self.assertFalse(proof['sc2_state_restored'])
            self.assertEqual((trainer.global_steps,raw.reset_index),(4,0))
            restored.load_state_dict(original.state_dict())
            cfg.total_epochs = 1
            proof = mod.restore_boundary(trainer, scene=True)
            self.assertEqual(len(list(restored)),1)
            self.assertEqual(proof['new_run_global_steps'],[5])
            self.assertEqual(proof['game_seed_per_new_step'],[1098341402])
            self.assertEqual(np.repeat(raw.training_rng.randint(0,2**16-1,size=1),8).tolist(),[1098341402]*8)
            with self.assertRaises(ValueError): raw.training_rng.randint(0,2**16-1,size=1)
            cfg.total_epochs = 4
            restored.load_state_dict(original.state_dict())
            proof=mod.restore_boundary(trainer,coverage=True)
            self.assertEqual([len(list(restored)) for _ in range(4)],[1]*4)
            self.assertEqual([raw.training_rng.randint(0,2**16-1,1).item() for _ in range(4)],mod.COVERAGE_SEEDS)
            self.assertEqual(proof['new_run_global_steps'],[5,6,7,8])
            self.assertEqual(proof['seed_selection'],'explicit coverage table')
            self.assertFalse(proof['sc2_state_restored'])
            with self.assertRaises(ValueError): raw.training_rng.randint(0,2**16-1,1)
            cfg.get = lambda k, d: {'sc2_signal_max_batches': 4, 'sc2_skip_zero_signal': False}.get(k, d)
            for arm, expected in mod.SIGNAL_SCENE_SEEDS.items():
                restored.load_state_dict(original.state_dict())
                proof = mod.restore_boundary(trainer, coverage=True, coverage_seeds=expected)
                self.assertEqual([len(list(restored)) for _ in range(4)], [1]*4)
                self.assertEqual([raw.training_rng.randint(0,2**16-1,1).item() for _ in range(4)], expected)
                self.assertEqual(proof['game_seed_per_collection'], expected)
                self.assertEqual(proof['planned_max_new_global_steps'], [5,6,7,8])
                self.assertIn('skipping is disabled', proof['note'])
                with self.assertRaises(ValueError): raw.training_rng.randint(0,2**16-1,1)
            trainer.train_dataloader=[object()]
            with self.assertRaisesRegex(ValueError,'discard a sample'): mod.restore_boundary(trainer)
            trainer.global_steps=0
            with self.assertRaises(ValueError): mod.restore_boundary(trainer)

    def test_scene_selector_and_controller_are_bounded(self):
        mod=module('continue_a4')
        seeds=mod.SingleTrainingSeed(mod.SCENE_SEED)
        self.assertEqual(seeds.randint(0,2**16-1,1).tolist(),[1098341402])
        with self.assertRaises(ValueError): seeds.randint(0,2**16-1,1)
        with self.assertRaises(ValueError): mod.SingleTrainingSeed(1).randint(0,2**16-1,8)
        script=RECORD/'run_a4_scene1_node43.sh'; text=script.read_text()
        self.assertEqual(subprocess.run(['bash','-n',str(script)]).returncode,0)
        for required in ('SC2_A4_SCENE1_AUTHORIZED','+ 2700','--kill-after=15s',
                         "result['last_global_step'] == result['target_steps'] == 5",'SC2_COMMAND_TRACE_DIR'):
            self.assertIn(required,text)
        self.assertNotIn('run_feedback_eval',text)
        self.assertNotIn('pkill',text)

    def test_controller_keeps_authorization_budget_and_serial_gate(self):
        script=RECORD/'run_a4_continue_node43.sh'; text=script.read_text()
        self.assertEqual(subprocess.run(['bash','-n',str(script)]).returncode,0)
        for value in ('SC2_A4_CONTINUE_AUTHORIZED','+ 7200','--kill-after=15s',
            "result['last_global_step'] == result['target_steps'] == 8",'export FEEDBACK_LEARNING_MODEL=A8'):
            self.assertIn(value,text)
        self.assertNotIn('for label in original',text)
        self.assertNotIn('pkill',text)
        # train_sc2.sh puts trainer.default_local_dir under RUN/checkpoints.
        self.assertIn("p/'checkpoints/a4_continuation.json'",text)
        suffix='/session_2026-09-11_23-59-59_999999_2147483647/sockets/plasma_store'
        for name in ('a4t_0911_r1','a8e_0911_r1'):
            self.assertLessEqual(len(('/dev/shm/'+name+suffix).encode()),107)

    def test_coverage_table_and_serial_controller(self):
        mod=module('continue_a4'); seeds=mod.CoverageTrainingSeeds()
        self.assertEqual([seeds.randint(0,2**16-1,1).item() for _ in range(4)],
            [1098341402,42613,17089,3236598124])
        with self.assertRaises(ValueError): seeds.randint(0,2**16-1,1)
        with self.assertRaises(ValueError): mod.CoverageTrainingSeeds().randint(0,2**16-1,8)
        with self.assertRaises(ValueError): mod.restore_boundary(None,scene=True,coverage=True)
        script=RECORD/'run_a4_coverage_node43.sh'; text=script.read_text()
        self.assertEqual(subprocess.run(['bash','-n',str(script)]).returncode,0)
        for required in ('SC2_A4_COVERAGE_AUTHORIZED','+ 10800','for arm in A B',
                         'run_feedback_train4_node43.sh" A','FEEDBACK_RL_GPUS=0,1,2,3',
                         "p/'checkpoints/a4_continuation.json'",'--kill-after=15s',
                         '/mnt/data0601/yangjinyue/new_SCII/experiments/',
                         "r['episode_outcomes']['natural_games'] == 32"):
            self.assertIn(required,text)
        self.assertNotIn('run_feedback_eval',text)
        self.assertNotIn('pkill',text)

    def test_signal_tables_use_predeclared_order_and_mixed_scenes(self):
        mod = module('continue_a4')
        self.assertEqual(mod.SIGNAL_SCENE_SEEDS['A'], module('learning_eval').SCREEN_GAME_SEEDS[:4])
        self.assertEqual(mod.SIGNAL_SCENE_SEEDS['B'], [1098341402,3446496864]*2)
        for arm in ('A','B'):
            with patch.object(mod, 'install') as install:
                getattr(mod, 'install_signal_'+arm)()
                install.assert_called_once_with(coverage=True, coverage_seeds=mod.SIGNAL_SCENE_SEEDS[arm])
        with self.assertRaises(ValueError): mod.restore_boundary(None, coverage=True, coverage_seeds=[1])

    @unittest.skipUnless(importlib.util.find_spec('torch'),'requires pinned CPU Torch')
    def test_setup_hook_defers_trainer_and_cuda(self):
        paths=[ROOT,RECORD,RECORD.parent/'20260910-spb-fsdp-layer-wrap-v1']
        code='''import sys, torch
torch.cuda.is_available=lambda: (_ for _ in ()).throw(AssertionError("early CUDA probe"))
import continue_a4
continue_a4.install()
assert "verl.trainer.ppo.ray_trainer" not in sys.modules
assert not torch.cuda.is_initialized()
'''
        env=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,paths))+os.pathsep+os.environ.get('PYTHONPATH',''),CUDA_VISIBLE_DEVICES='')
        run=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env=env,capture_output=True,text=True,timeout=25)
        self.assertEqual(run.returncode,0,run.stdout+run.stderr)


if __name__=='__main__': unittest.main()
