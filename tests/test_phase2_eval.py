"""CPU-only evaluation wiring, paired seeds and frozen A/B overrides."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_phase2_air_options import RECORD, ROOT, OBS, candidate, strategist_module

spec = importlib.util.spec_from_file_location('phase2_eval', RECORD/'phase2_eval.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
ENTRY = ROOT/'experiments/records/20260911-execution-failure-feedback-v1/run_feedback_eval_node47.sh'


class PhaseTwoEvaluationTest(unittest.TestCase):
    def test_actual_commander_alias_is_patched_and_both_arms_record(self):
        proofs = {}
        for arm in ('A', 'B'):
            class Isolated(strategist_module.SC2Strategist):
                pass
            feedback = SimpleNamespace(install=Mock())
            with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {
                    'agents.commander': SimpleNamespace(SC2Strategist=Isolated),
                    'feedback_eval': feedback, 'phase2_air_options': candidate}):
                with patch.dict(os.environ, SC2_PHASE2_AIR_AB=arm, SC2_FEEDBACK_LEARNING_MODEL='A4',
                        SC2_EXECUTION_FAILURE_FEEDBACK='0', SC2_STRATEGY_PROFILE='marine_pressure',
                        SC2_ENABLE_AIR_OPTIONS='1', SC2_EVAL_SEEDS_LOG_PATH=tmp+'/seeds.jsonl'):
                    evaluation.install()
                    feedback.install.assert_called_once_with(expected_step=4,game_seeds=evaluation.GAME_SEEDS)
                    obj=Isolated()
                    for time in ('00:00','04:09','10:11'):
                        obj.generate_strategy(OBS.replace('04:09',time))
                    proofs[arm]=[json.loads(x) for x in next(Path(tmp).glob('*-phase2-options.jsonl')).read_text().splitlines()]
        self.assertEqual([p['phase_index'] for p in proofs['A']],[0,1,2])
        for index in range(3):
            a,b=proofs['A'][index],proofs['B'][index]
            self.assertEqual(a['persistent_template_sha256'],b['persistent_template_sha256'])
            self.assertEqual(a['attack_instruction_sha256'],b['attack_instruction_sha256'])
            if index==1:
                self.assertNotIn('starport',a['bo_todo'])
                self.assertEqual(b['bo_todo']['vikingfighter'],8)
                self.assertNotEqual(a['instruction_sha256'],b['instruction_sha256'])
            else:
                self.assertEqual(a['bo_todo'],b['bo_todo'])
                self.assertEqual(a['instruction_sha256'],b['instruction_sha256'])

    def test_overrides_differ_only_in_candidate_selector(self):
        sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        configs={}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/'model').mkdir(); (p/'sc2').mkdir()
            for arm in ('A','B'):
                out=p/arm
                env=dict(os.environ,DRY_RUN='1',VERIFY_ASSETS='0',PYTHON_BIN=sys.executable,
                    SOURCE_SHA=sha,MODEL_PATH=str(p/'model'),SC2PATH=str(p/'sc2'),
                    FEEDBACK_STORAGE_ROOT=str(p/'storage'),OUTPUT_DIR=str(out),RUN_ID='phase2-test',
                    RAY_TMPDIR='/tmp/phase2-test',FEEDBACK_EVAL_GPUS='0,1,2,3',
                    FEEDBACK_LEARNING_MODEL='A4',FEEDBACK_PHASE2_AIR=arm)
                run=subprocess.run(['bash',str(ENTRY),'A'],env=env,cwd=ROOT,capture_output=True,text=True,timeout=30)
                self.assertEqual(run.returncode,0,run.stdout+run.stderr)
                cfg={k:v.replace(str(out),'<RUN>') for k,v in json.loads((out/'effective_config.json').read_text())['overrides'].items()}
                configs[arm]=cfg
                for key,value in {'trainer.val_only':'True','trainer.save_freq':'-1',
                    'env.rollout.n':'4','+env.val_rollout_n':'4','actor_rollout_ref.rollout.val_kwargs.n':'4',
                    '+ray_init.runtime_env.env_vars.SC2_EXECUTION_FAILURE_FEEDBACK':"'0'",
                    '+ray_init.runtime_env.worker_process_setup_hook':'phase2_eval.install',
                    'data.max_response_length':'192','env.history_length':'0','+env.sc2_memory_mode':'off',
                    'actor_rollout_ref.rollout.val_kwargs.temperature':'0.4',
                    'actor_rollout_ref.rollout.val_kwargs.top_p':'0.9',
                    '+actor_rollout_ref.model.override_config.swa_dropout':'0.0',
                    '+actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype':'fp32',
                    '+actor_rollout_ref.actor.fp32_normalization':'True',
                    '++actor_rollout_ref.rollout.fsdp_layer_wrap':'True'}.items():
                    self.assertEqual(cfg[key],value,key)
                self.assertTrue(cfg['trainer.resume_from_path'].endswith('-A-rl4-4gpu-s0-r2/checkpoints/global_step_4'))
                self.assertEqual(json.loads((out/'manifest.json').read_text())['mode'],'evaluation')
        a,b=configs.values()
        self.assertEqual({k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)},
                         {'+ray_init.runtime_env.env_vars.SC2_PHASE2_AIR_AB'})

    def test_controller_preserves_budget_and_no_retry(self):
        entry=RECORD/'run_pair_node43.sh'
        self.assertEqual(subprocess.run(['bash','-n',str(entry)]).returncode,0)
        text=entry.read_text()
        for expected in ('SC2_PHASE2_AIR_AUTHORIZED','for arm in A B','+ 2700',
                         '--kill-after=15s',"result['training_updates']==0"):
            self.assertIn(expected,text)
        self.assertNotIn('pkill',text)
        self.assertNotIn('run_feedback_train',text)
        self.assertEqual(evaluation.GAME_SEEDS,[42613,17089,3236598124,4018390961])


if __name__=='__main__': unittest.main()
