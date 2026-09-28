"""CPU contract: common eight-seed evaluation, no new training mechanism."""
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

from test_feedback_learning_eval import module, ROOT, RECORD

HERE = ROOT/'experiments/records/20260914-zero-task-signal-sampling-v1'


class ScenePostEvaluationTest(unittest.TestCase):
    def test_existing_panel_is_disjoint_from_new_training(self):
        mod = module('learning_eval')
        self.assertEqual(mod.SCENE_POST_GAME_SEEDS, [2259791793,1436145665,363527354,462000319,3434611143,1968356657,4276540916,2065928601])
        self.assertEqual(len(set(mod.SCENE_POST_GAME_SEEDS)), 8)
        self.assertFalse(set(mod.SCENE_POST_GAME_SEEDS) & {3974528993,1183530000,3446496864,2996859384,1098341402})

    def test_reuse_existing_loader_and_exact_seed_batch(self):
        mod = module('learning_eval')
        for label, step in [('A4',4),('SceneA8',8),('SceneB8',8)]:
            with patch.dict(os.environ, SC2_FEEDBACK_LEARNING_MODEL=label, SC2_EXECUTION_FAILURE_FEEDBACK='0'):
                with patch.object(mod.feedback_eval,'install') as install:
                    mod.install_scene_posteval()
                    install.assert_called_once_with(expected_step=step, game_seeds=mod.SCENE_POST_GAME_SEEDS)
        raw=SimpleNamespace(is_train=False,seed=1000,num_processes=8,reset_index=0)
        mod.feedback_eval.select_seeds(raw,mod.SCENE_POST_GAME_SEEDS)
        self.assertEqual(raw.validation_rng.randint(65536,2**32-1,8).tolist(),mod.SCENE_POST_GAME_SEEDS)
        with self.assertRaises(ValueError): raw.validation_rng.randint(65536,2**32-1,8)
        for label,feedback in [('B4','0'),('SceneA8','1')]:
            with patch.dict(os.environ,SC2_FEEDBACK_LEARNING_MODEL=label,SC2_EXECUTION_FAILURE_FEEDBACK=feedback):
                with self.assertRaises(ValueError): mod.install_scene_posteval()

    def test_only_checkpoint_label_hook_differ_from_old_evaluation(self):
        sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        configs={}; paths=[]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/'model').mkdir(); (p/'sc2').mkdir()
            for label in ['old','A4','SceneA8','SceneB8']:
                out=p/label; gpu='4,5,6,7' if label=='SceneA8' else '0,1,2,3'
                env=dict(os.environ,DRY_RUN='1',VERIFY_ASSETS='0',PYTHON_BIN=sys.executable,SOURCE_SHA=sha,
                    MODEL_PATH=str(p/'model'),SC2PATH=str(p/'sc2'),FEEDBACK_STORAGE_ROOT=str(p/'storage'),
                    FEEDBACK_EVAL_GPUS=gpu,FEEDBACK_LEARNING_MODEL='A4' if label=='old' else label,
                    FEEDBACK_SCENE_POSTEVAL='0' if label=='old' else '1',FEEDBACK_SCENE_EVAL='1' if label=='old' else '0',
                    FEEDBACK_SCENE_SET='historical8',FEEDBACK_POST_CHECKPOINT=str(p/label/('global_step_4' if label=='A4' else 'global_step_8')),
                    RUN_ID=label,OUTPUT_DIR=str(out),RAY_TMPDIR='/dev/shm/pe_'+label+'_0915_r1')
                result=subprocess.run(['bash',str(RECORD/'run_feedback_eval_node47.sh'),'A'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=30)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                cfg=json.loads((out/'effective_config.json').read_text())['overrides']
                ray=cfg['+ray_init._temp_dir']
                configs[label]={k:v.replace(str(out),'<RUN>').replace(ray,'<RAY>').replace(label,'<LABEL>') if k in ['trainer.experiment_name'] else v.replace(str(out),'<RUN>').replace(ray,'<RAY>') for k,v in cfg.items()}
                for k,v in {'trainer.val_only':'True','trainer.save_freq':'-1','env.rollout.n':'8','env.history_length':'0','+env.sc2_memory_mode':'off','data.max_response_length':'192','actor_rollout_ref.rollout.val_kwargs.temperature':'0.4','+actor_rollout_ref.model.override_config.swa_dropout':'0.0','+ray_init.runtime_env.env_vars.SC2_EXECUTION_FAILURE_FEEDBACK':"'0'"}.items(): self.assertEqual(cfg[k],v,k)
                self.assertEqual(json.loads((out/'manifest.json').read_text())['gpu_selection'],gpu)
                if label!='old': paths.append(out)
            allowed={'trainer.resume_from_path','+ray_init.runtime_env.env_vars.SC2_FEEDBACK_LEARNING_MODEL','+ray_init.runtime_env.worker_process_setup_hook'}
            for label in ['A4','SceneA8','SceneB8']:
                diff={k for k in configs['old'].keys()|configs[label].keys() if configs['old'].get(k)!=configs[label].get(k)}
                self.assertLessEqual(diff,allowed,diff)
            spec=importlib.util.spec_from_file_location('scene_check',HERE/'scene_posteval_check.py'); check=importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules,learning_eval=module('learning_eval')): spec.loader.exec_module(check)
            self.assertTrue(check.compare(paths)['configs_match_except_checkpoint_and_label'])
            cfg=json.loads((paths[1]/'effective_config.json').read_text());cfg['overrides']['actor_rollout_ref.rollout.val_kwargs.temperature']='0.8'
            (paths[1]/'effective_config.json').write_text(json.dumps(cfg))
            with self.assertRaises(AssertionError): check.compare(paths)

    def test_parallel_budget_and_no_automatic_retry(self):
        p=HERE/'run_scene_posteval_node47.sh'; script=p.read_text()
        self.assertEqual(subprocess.run(['bash','-n',str(p)]).returncode,0)
        for text in ['+ 3600','arm A4 0,1,2,3','arm SceneB8 0,1,2,3','arm SceneA8 4,5,6,7','wait "${left}"','wait "${right}"','800*1024**2','--kill-after=15s']:
            self.assertIn(text,script)
        for text in ['pkill','ray stop','run_feedback_train','rm -rf','hf download']:
            self.assertNotIn(text,script)
        suffix='/session_2026-09-15_23-59-59_999999_2147483647/sockets/plasma_store'
        for label in ['A4','SceneA8','SceneB8']:
            self.assertLessEqual(len(('/dev/shm/pe_'+label+'_0915_r1'+suffix).encode()),107)


if __name__=='__main__': unittest.main()
