"""CPU-only independent-evaluation configuration and reward-ledger checks."""
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'experiments/records/20260911-execution-failure-feedback-v1'
HANDOFF = ROOT / 'experiments/records/20260908-g2-planner-handoff-v1'


def module(name):
    sys.path[:0] = [str(RECORD), str(HANDOFF)]
    try:
        spec = importlib.util.spec_from_file_location(name, RECORD / (name + '.py'))
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result
    finally:
        del sys.path[:2]


class LearningEvaluationTest(unittest.TestCase):
    def test_multiseed_keeps_scene_config_except_seed_hook(self):
        sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        configs={}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/'model').mkdir(); (p/'sc2').mkdir()
            for label in ('A4','A5scene'):
                for scene_set in ('single','historical8'):
                    out=p/(label+'-'+scene_set)
                    env=dict(os.environ,DRY_RUN='1',VERIFY_ASSETS='0',PYTHON_BIN=sys.executable,
                        SOURCE_SHA=sha,MODEL_PATH=str(p/'model'),SC2PATH=str(p/'sc2'),
                        FEEDBACK_STORAGE_ROOT=str(p/'storage'),OUTPUT_DIR=str(out),RUN_ID='multiseed-test',
                        RAY_TMPDIR='/tmp/multiseed-test',FEEDBACK_EVAL_GPUS='0,1,2,3',
                        FEEDBACK_LEARNING_MODEL=label,FEEDBACK_SCENE_EVAL='1',FEEDBACK_SCENE_SET=scene_set)
                    result=subprocess.run(['bash',str(RECORD/'run_feedback_eval_node47.sh'),'A'],
                        env=env,cwd=ROOT,capture_output=True,text=True,timeout=30)
                    self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                    configs[label,scene_set]={k:v.replace(str(out),'<RUN>') for k,v in
                        json.loads((out/'effective_config.json').read_text())['overrides'].items()}
                a,b=(configs[label,s] for s in ('single','historical8'))
                self.assertEqual({k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)},
                    {'+ray_init.runtime_env.worker_process_setup_hook'})
                self.assertEqual(b['+ray_init.runtime_env.worker_process_setup_hook'],'learning_eval.install_multiseed')
                self.assertEqual(b['+ray_init.runtime_env.env_vars.SC2_EXECUTION_FAILURE_FEEDBACK'],"'0'")
                self.assertEqual(b['trainer.val_only'],'True')
        a,b=(configs[label,'historical8'] for label in ('A4','A5scene'))
        self.assertEqual({k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)},
            {'trainer.resume_from_path','+ray_init.runtime_env.env_vars.SC2_FEEDBACK_LEARNING_MODEL'})

    def test_multiseed_uses_original_panel_with_feedback_off(self):
        mod=module('learning_eval'); boundary=module('continue_a4')
        with patch.dict(sys.modules,continue_a4=boundary):
            for label,step in (('A4',4),('A5scene',5)):
                with patch.dict(os.environ,SC2_FEEDBACK_LEARNING_MODEL=label,SC2_EXECUTION_FAILURE_FEEDBACK='0'):
                    with patch.object(mod.feedback_eval,'install') as install:
                        mod.install_multiseed()
                        install.assert_called_once_with(expected_step=step,game_seeds=mod.GAME_SEEDS)
            with patch.dict(os.environ,SC2_FEEDBACK_LEARNING_MODEL='A5scene',SC2_EXECUTION_FAILURE_FEEDBACK='1'):
                with self.assertRaises(ValueError): mod.install_multiseed()
        script=(RECORD/'run_a4_scene_eval_node43.sh').read_text()
        for text in ('SC2_A4_MULTI_EVAL_AUTHORIZED','scenario_name=multiseed','ray_prefix=me',
                     "assert seeds==[expected_seeds]"):
            self.assertIn(text,script)
        suffix='/session_2026-09-13_23-59-59_999999_2147483647/sockets/plasma_store'
        for label in ('A4','A5scene'):
            self.assertLessEqual(len(('/dev/shm/me_'+label+'_0913_r1'+suffix).encode()),107)

    def test_scene_pair_keeps_everything_except_checkpoint_and_label(self):
        sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        configs={}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/'model').mkdir(); (p/'sc2').mkdir()
            for label in ('A4','A5scene'):
                out=p/label
                env=dict(os.environ,DRY_RUN='1',VERIFY_ASSETS='0',PYTHON_BIN=sys.executable,
                    SOURCE_SHA=sha,MODEL_PATH=str(p/'model'),SC2PATH=str(p/'sc2'),
                    FEEDBACK_STORAGE_ROOT=str(p/'storage'),OUTPUT_DIR=str(out),RUN_ID='scene-eval-test',
                    RAY_TMPDIR='/tmp/scene-eval-test',FEEDBACK_EVAL_GPUS='0,1,2,3',
                    FEEDBACK_LEARNING_MODEL=label,FEEDBACK_SCENE_EVAL='1')
                result=subprocess.run(['bash',str(RECORD/'run_feedback_eval_node47.sh'),'A'],
                    env=env,cwd=ROOT,capture_output=True,text=True,timeout=30)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                configs[label]={k:v.replace(str(out),'<RUN>') for k,v in
                    json.loads((out/'effective_config.json').read_text())['overrides'].items()}
                cfg=configs[label]
                for key,value in {'trainer.val_only':'True','trainer.save_freq':'-1',
                    'env.rollout.n':'8','+env.val_rollout_n':'8','actor_rollout_ref.rollout.val_kwargs.n':'8',
                    '+ray_init.runtime_env.env_vars.SC2_EXECUTION_FAILURE_FEEDBACK':"'0'",
                    '+ray_init.runtime_env.worker_process_setup_hook':'learning_eval.install_scene',
                    'data.max_response_length':'192','env.history_length':'0','+env.sc2_memory_mode':'off',
                    'actor_rollout_ref.rollout.val_kwargs.temperature':'0.4',
                    'actor_rollout_ref.rollout.val_kwargs.top_p':'0.9','actor_rollout_ref.rollout.val_kwargs.top_k':'0',
                    '+actor_rollout_ref.model.override_config.swa_dropout':'0.0'}.items():
                    self.assertEqual(cfg[key],value,key)
                self.assertTrue(cfg['trainer.resume_from_path'].endswith('/global_step_'+('4' if label=='A4' else '5')))
                self.assertEqual(json.loads((out/'manifest.json').read_text())['mode'],'evaluation')
        a,b=configs.values()
        self.assertEqual({k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)},
            {'trainer.resume_from_path','+ray_init.runtime_env.env_vars.SC2_FEEDBACK_LEARNING_MODEL'})

    def test_scene_hook_and_budget(self):
        mod=module('learning_eval'); boundary=module('continue_a4')
        with patch.dict(sys.modules,continue_a4=boundary):
            for label,step in (('A4',4),('A5scene',5)):
                with patch.dict(os.environ,SC2_FEEDBACK_LEARNING_MODEL=label,SC2_EXECUTION_FAILURE_FEEDBACK='0'):
                    with patch.object(mod.feedback_eval,'install') as install:
                        mod.install_scene()
                        install.assert_called_once_with(expected_step=step,game_seeds=[1098341402]*8)
            with patch.dict(os.environ,SC2_FEEDBACK_LEARNING_MODEL='A5scene',SC2_EXECUTION_FAILURE_FEEDBACK='1'):
                with self.assertRaises(ValueError): mod.install_scene()
        entry=RECORD/'run_a4_scene_eval_node43.sh'; script=entry.read_text()
        self.assertEqual(subprocess.run(['bash','-n',str(entry)]).returncode,0)
        for text in ('SC2_A4_SCENE_EVAL_AUTHORIZED','+ 2700','for label in A4 A5scene',
                     "result['training_updates']==0",'--kill-after=15s','SC2_COMMAND_TRACE_DIR'):
            self.assertIn(text,script)
        self.assertNotIn('run_feedback_train',script)
        self.assertNotIn('pkill',script)
        suffix='/session_2026-09-13_23-59-59_999999_2147483647/sockets/plasma_store'
        for label in ('A4','A5scene'):
            self.assertLessEqual(len(('/dev/shm/se_'+label+'_0913_r1'+suffix).encode()),107)

    def test_three_models_share_evaluation_config(self):
        sha = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        configs = {}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path/'model').mkdir(); (path/'sc2').mkdir()
            for label in ('original','A4','B4'):
                out = path/label
                env = dict(os.environ, DRY_RUN='1', VERIFY_ASSETS='0', PYTHON_BIN=sys.executable,
                    SOURCE_SHA=sha, MODEL_PATH=str(path/'model'), SC2PATH=str(path/'sc2'),
                    FEEDBACK_STORAGE_ROOT=str(path/'storage'), FEEDBACK_LEARNING_MODEL=label,
                    FEEDBACK_EVAL_GPUS='0,1,2,3', RUN_ID='learning-config-test',
                    OUTPUT_DIR=str(out), RAY_TMPDIR='/tmp/sc2-learning-test')
                run = subprocess.run(['bash',str(RECORD/'run_feedback_eval_node47.sh'),'B'],
                    cwd=ROOT,env=env,text=True,capture_output=True,timeout=30)
                self.assertEqual(run.returncode,0,run.stdout+run.stderr)
                raw = json.loads((out/'effective_config.json').read_text())['overrides']
                cfg = {k:v.replace(str(out),'<RUN>') for k,v in raw.items()}
                configs[label] = cfg
                manifest = json.loads((out/'manifest.json').read_text())
                self.assertEqual((manifest['mode'],manifest['world_size'],manifest['gpu_selection']),
                                 ('evaluation',4,'0,1,2,3'))
                for k,v in {
                    'trainer.val_only':'True','trainer.save_freq':'-1','trainer.total_epochs':'1',
                    'env.rollout.n':'8','+env.val_rollout_n':'8','actor_rollout_ref.rollout.val_kwargs.n':'8',
                    'actor_rollout_ref.rollout.val_kwargs.do_sample':'True',
                    'actor_rollout_ref.rollout.val_kwargs.temperature':'0.4',
                    'actor_rollout_ref.rollout.val_kwargs.top_p':'0.9',
                    'actor_rollout_ref.rollout.val_kwargs.top_k':'0',
                    'data.max_response_length':'192','env.history_length':'0','+env.sc2_memory_mode':'off',
                    '+ray_init.runtime_env.env_vars.SC2_EXECUTION_FAILURE_FEEDBACK':"'1'",
                    '+ray_init.runtime_env.env_vars.SC2_STEP_SHAPING_MODE':"'off'",
                    '++ray_init.runtime_env.env_vars.SC2_ROLLOUT_N':"'8'",
                    '+ray_init.runtime_env.worker_process_setup_hook':'learning_eval.install',
                    '+actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype':'fp32',
                    '+actor_rollout_ref.actor.fp32_normalization':'True',
                    '++actor_rollout_ref.rollout.fsdp_layer_wrap':'True',
                    '+actor_rollout_ref.model.override_config.swa_dropout':'0.0',
                }.items(): self.assertEqual(cfg[k],v,k)
                self.assertNotIn('+trainer.sc2_checkpoint_resume',cfg)
                if label=='original':
                    self.assertEqual(cfg['trainer.resume_mode'],'disable')
                    self.assertNotIn('trainer.resume_from_path',cfg)
                else:
                    self.assertTrue(cfg['trainer.resume_from_path'].endswith(
                        f'{label[0]}-rl4-4gpu-s0-r2/checkpoints/global_step_4'))
        allowed = {'trainer.resume_mode','trainer.resume_from_path',
                   '+ray_init.runtime_env.env_vars.SC2_FEEDBACK_LEARNING_MODEL'}
        for label in ('A4','B4'):
            a,b=configs['original'],configs[label]
            self.assertEqual({k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)},allowed)
        a,b=configs['A4'],configs['B4']
        self.assertEqual({k for k in a if a[k]!=b[k]},allowed-{'trainer.resume_mode'})

    def test_seeds_are_explicit_paired_and_not_resampled(self):
        mod=module('learning_eval')
        expected=[42613,52620,17089,4018390961,3236598124,3297512220,773351930,1098341402]
        self.assertEqual(mod.GAME_SEEDS,expected)
        for _ in ('original','A4','B4'):
            raw=SimpleNamespace(is_train=False,seed=1000,num_processes=8,reset_index=0)
            record=mod.feedback_eval.select_seeds(raw,mod.GAME_SEEDS)
            self.assertEqual(record['independent_scenarios'],8)
            self.assertEqual(raw.validation_rng.randint(65536,2**32-1,8).tolist(),expected)
            with self.assertRaises(ValueError): raw.validation_rng.randint(65536,2**32-1,8)

    def test_hook_checks_model_label_and_common_feedback(self):
        mod=module('learning_eval')
        for label,step in (('original',0),('A4',4),('B4',4)):
            with patch.dict(os.environ,SC2_FEEDBACK_LEARNING_MODEL=label,SC2_EXECUTION_FAILURE_FEEDBACK='1'):
                with patch.object(mod.feedback_eval,'install') as install:
                    mod.install()
                    install.assert_called_once_with(expected_step=step,game_seeds=mod.GAME_SEEDS)
        with patch.dict(os.environ,SC2_FEEDBACK_LEARNING_MODEL='A4',SC2_EXECUTION_FAILURE_FEEDBACK='0'):
            with self.assertRaises(ValueError): mod.install()

    def test_missing_or_failed_checkpoint_cannot_be_promoted(self):
        mod=module('learning_eval')
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError): mod.checkpoint_preflight(Path(tmp))

    def test_reward_audit_reattributes_terminal_without_fabricating_decision(self):
        mod=module('audit_rl4_rewards')
        raw=[dict(env_idx=0,step_index=0,reward=0.5,done=False,policy_action_applied=True),
             dict(env_idx=0,step_index=1,reward=20.,done=True,policy_action_applied=False)]
        end=dict(env_index=0,step_index=1,policy_decision_count=1,terminated=True,truncated=False)
        policy,reward=mod.credited_episode(raw,end,0)
        self.assertEqual(len(policy),1)
        self.assertEqual((reward,policy[0]['rewards']),(20.5,20.5))
        self.assertEqual(raw[0]['reward'],0.5)

    def test_controller_is_authorized_serial_and_bounded(self):
        entry=RECORD/'run_learning_eval_node43.sh'
        self.assertEqual(subprocess.run(['bash','-n',str(entry)]).returncode,0)
        text=entry.read_text()
        self.assertIn('SC2_FEEDBACK_LEARNING_EVAL_AUTHORIZED',text)
        self.assertIn('for label in original A4 B4',text)
        self.assertIn('+ 7140',text)
        self.assertIn('--kill-after=15s',text)
        self.assertNotIn('pkill',text)
        queue=RECORD/'queue_learning_eval_node43.sh'
        self.assertEqual(subprocess.run(['bash','-n',str(queue)]).returncode,0)
        self.assertIn('EXIT_CODE=0',queue.read_text())
        self.assertIn('--kill-after=15s 7125',queue.read_text())

    def test_ray_socket_paths_fit_linux_unix_socket_limit(self):
        text=(RECORD/'run_learning_eval_node43.sh').read_text()
        template=re.search(r'export RAY_TMPDIR=(\S+)',text).group(1)
        suffix='/session_2026-09-11_23-59-59_999999_2147483647/sockets/plasma_store'
        old='/dev/shm/sc2_feedback_learning_original_0911_r1'+suffix
        self.assertGreater(len(old.encode()),107)
        for label in ('original','A4','B4'):
            path=template.replace('${label}',label)+suffix
            self.assertLessEqual(len(path.encode()),107,path)


if __name__=='__main__': unittest.main()
