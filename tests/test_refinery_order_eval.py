"""CPU boundary/config/real-parser checks; no model or SC2 startup."""
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_r6_format_reward import strict_validator
from test_valid_bo_passthrough import replay

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'experiments/records/20260910-spb-fsdp-layer-wrap-v1'
spec = importlib.util.spec_from_file_location('refinery_order_eval', RECORD/'refinery_order_eval.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class RefineryOrderTest(unittest.TestCase):
    def test_exact_historical_inputs_and_one_move(self):
        rows = json.loads((RECORD/'refinery-order-inputs.json').read_text())
        self.assertEqual([hashlib.sha256(r['prompt'].encode()).hexdigest() for r in rows], mod.PROMPT_HASHES)
        self.assertEqual([r['answer'] for r in rows], [mod.OPENING, mod.ORIGINAL])
        for arm in ['A', 'B']:
            self.assertEqual(mod.select_answer(0, arm, rows[0]['prompt'], 'unused'),mod.OPENING)
        a=mod.select_answer(1,'A',rows[1]['prompt'],'unused')
        b=mod.select_answer(1,'B',rows[1]['prompt'],'unused')
        self.assertEqual(Counter(a.splitlines()),Counter(b.splitlines()))
        self.assertEqual(a.replace('- refinery x 1\n',''), b.replace('- refinery x 1\n',''))
        self.assertLess(b.index('- refinery'), b.index('- reactor'))
        with patch.dict(os.environ,SC2_ORIGINAL_BO_SEGMENT='1'):
            self.assertTrue(strict_validator()(rows[1]['prompt'],a)[0])
            self.assertTrue(strict_validator()(rows[1]['prompt'],b)[0])
        with self.assertRaises(ValueError):
            mod.select_answer(1,'B','changed state','unused')
        self.assertEqual(mod.select_answer(2,'B','new observation','online answer'),'online answer')

    def test_real_parser_preserves_seven_action_order(self):
        p=json.loads((RECORD/'refinery-order-inputs.json').read_text())[1]['prompt']
        obs=p.split('Current observations are:\n',1)[1].split('The remaining BO list',1)[0].strip()
        stage=re.search(r'Current Stage: (.*?) \((.*?)\)',p)
        strategy=dict(phase_summary=stage[1],timing=stage[2],
            instruction=re.search(r'Detailed Instruction: (.*)',p)[1],
            attack_instruction=re.search(r'Attacking Instruction: (.*)',p)[1])
        todo={'Launch OFFENSE with 8 marine':1,'marine':42,'supplydepot':8,
              'barracks':3,'reactor barracks':3,'scv':3,'refinery':1}
        control={'SC2_ORIGINAL_BO_SEGMENT':'1','SC2_CLARIFY_BO_QUOTA':'1'}
        queues=[]
        for answer in [mod.ORIGINAL,mod.move_refinery(mod.ORIGINAL)]:
            result=replay(answer,True,obs,todo,strategy,controls=control)
            self.assertTrue(result['valid'],result['reason'])
            queues.append(result['plan'])
        self.assertEqual(queues[0],['marine','supplydepot','barracks','reactor barracks','reactor barracks','scv','refinery'])
        self.assertEqual(queues[1],['marine','supplydepot','barracks','refinery','reactor barracks','reactor barracks','scv'])

    def test_single_seed_batch(self):
        rng=mod.FourSeeds()
        self.assertEqual(rng.randint(2**16,2**32-1,4).tolist(),[17089]*4)
        with self.assertRaises(ValueError):
            rng.randint(2**16,2**32-1,4)

    def test_frozen_b_config_except_eval_batch_and_hook(self):
        sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'model').mkdir(); (root/'sc2').mkdir()
            configs=[]
            for name,args in [('temperature',['B']),('refinery_order',[])]:
                out=root/name
                env=dict(os.environ,DRY_RUN='1',VERIFY_ASSETS='0',PYTHON_BIN=sys.executable,
                         SOURCE_SHA=sha,MODEL_PATH=str(root/'model'),SC2PATH=str(root/'sc2'),
                         OUTPUT_DIR=str(out),RUN_ID='order-config-test',RAY_TMPDIR='/tmp/order-config-test')
                r=subprocess.run(['bash',str(RECORD/f'run_{name}_eval_node44.sh' if name=='temperature' else RECORD/'run_refinery_order_node44.sh'),*args],env=env,capture_output=True,text=True)
                self.assertEqual(r.returncode,0,r.stdout+r.stderr)
                configs.append({k:v.replace(str(out),'<RUN>') for k,v in json.loads((out/'effective_config.json').read_text())['overrides'].items()})
            a,b=configs
            diff={k:(a.get(k),b.get(k)) for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
            self.assertEqual(diff,{
                '+ray_init.runtime_env.worker_process_setup_hook':('temperature_eval.install','refinery_order_eval.install'),
                '++ray_init.runtime_env.env_vars.SC2_ROLLOUT_N':("'8'","'4'"),
                'env.rollout.n':('8','4'),'+env.val_rollout_n':('8','4'),
                'actor_rollout_ref.rollout.val_kwargs.n':('8','4')})


if __name__=='__main__':
    unittest.main()
