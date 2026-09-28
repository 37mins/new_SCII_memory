"""Recorded inputs, actual validator/parser, and evaluation-only configuration."""
import asyncio
import copy
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
RECORD = ROOT / 'experiments/records/20260911-execution-failure-feedback-v1'
spec = importlib.util.spec_from_file_location('supply_order_eval', RECORD/'supply_order_eval.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class SupplyOrderTest(unittest.TestCase):
    def test_online_third_answer_keeps_recorded_input_but_never_replaces_generated_text(self):
        for i, row in enumerate(mod.FIXTURE['rows']):
            for answer in ('<answer>\n- supplydepot x 1\n</answer>', 'invalid original answer'):
                actual = mod.select_answer(i, 'online', row['prompt'], answer, prefix_answers=2)
                self.assertEqual(actual, row['answer'] if i < 2 else answer)
        with self.assertRaises(ValueError):
            mod.select_answer(2, 'online', 'different failure input', 'answer', prefix_answers=2)
        self.assertEqual(mod.select_answer(3, 'online', 'new observation', 'next answer', 2), 'next answer')

    def test_online_third_answer_has_no_human_or_historical_origin(self):
        payloads = [dict(text='raw', plan_format_valid=True)] * 4
        marked = mod.trace_payloads(payloads, [0, 1, 2, 3], [False]*4, prefix_answers=2)
        self.assertEqual([r['trace_origin'] for r in marked],
                         ['historical_replay']*2 + ['online_model_transport']*2)
        self.assertEqual([r['text'] for r in marked], ['raw']*4)

    def test_trace_labels_leave_validated_payloads_unchanged(self):
        payloads = [dict(text=f'answer-{i}', plan_format_valid=i % 2 == 0) for i in range(8)]
        before = copy.deepcopy(payloads)
        calls = [0, 1, 2, 2, 3, 4, 0, 5]
        marked = mod.trace_payloads(payloads, calls, [False]*7+[True])
        self.assertEqual(payloads, before)
        self.assertEqual([row['trace_origin'] for row in marked], [
            'historical_replay', 'historical_replay', 'historical_replay',
            'human_order_intervention', 'online_model_transport', 'online_model_transport',
            'historical_replay', 'terminal_unused'])
        self.assertEqual([{k: v for k, v in row.items() if k != 'trace_origin'} for row in marked], before)

    def test_recording_entry_preserves_effective_game_and_model_configuration(self):
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        configs = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'model').mkdir()
            (root/'sc2').mkdir()
            for script, name in (('run_supply_order_node44.sh', 'baseline'),
                                 ('run_command_trace_node43.sh', 'trace'),
                                 ('run_depot_issuer_node43.sh', 'issuer'),
                                 ('run_depot_association_node43.sh', 'association'),
                                 ('run_online_correction_node43.sh', 'online')):
                output = root/name
                env = dict(os.environ, DRY_RUN='1', VERIFY_ASSETS='0', PYTHON_BIN=sys.executable,
                           SOURCE_SHA=sha, MODEL_PATH=str(root/'model'), SC2PATH=str(root/'sc2'),
                           OUTPUT_DIR=str(output), RUN_ID='supply-config-test')
                env.pop('SC2_COMMAND_TRACE_DIR', None)
                env.pop('SC2_COMMAND_TRACE_VALIDATION', None)
                result = subprocess.run(['bash', str(RECORD/script)], cwd=ROOT, env=env,
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                saved = json.loads((output/'effective_config.json').read_text())
                cfg = saved['overrides']
                if name == 'online':
                    self.assertEqual(cfg['+ray_init.runtime_env.worker_process_setup_hook'],
                                     'online_correction_eval.install')
                    cfg['+ray_init.runtime_env.worker_process_setup_hook'] = 'supply_order_eval.install'
                if name != 'baseline':
                    self.assertEqual(cfg.pop('+ray_init.runtime_env.env_vars.SC2_COMMAND_TRACE_DIR'), str(output))
                # Only run/output identity, known dependency location and trace metadata may differ.
                text = json.dumps(cfg, sort_keys=True).replace(str(output), '<OUTPUT>')
                text = text.replace('20260911-execution-failure-feedback-v1-spb2-original-l6-g2-commandtrace-ab8-8gpu-s52620-r1', '<RUN>')
                text = text.replace('20260911-execution-failure-feedback-v1-spb2-original-l6-g2-commandtrace-ab8-8gpu-s52620-r2', '<RUN>')
                text = text.replace('20260911-execution-failure-feedback-v1-spb2-original-l6-g2-commandtrace-ab8-8gpu-s52620-r3', '<RUN>')
                text = text.replace('20260911-execution-failure-feedback-v1-spb2-original-l6-g2-online-from3-8gpu-s52620-r1', '<RUN>')
                text = text.replace('supply-config-test', '<RUN>')
                text = text.replace('/root/new-scii-execution-feedback-local-20260911/shared', '/mnt/data0601')
                text = text.replace('/root/new-scii-spb-level5-96-20260908/python', '/root/new-scii-spb2-level5-control/python')
                text = text.replace('/root/new-scii-spb-level5-96-20260908/include', '/usr/include')
                text = text.replace('/dev/shm/sc2_supply_order_0911_r1', '<TEMP>')
                text = text.replace('/dev/shm/ct_0912_r1', '<TEMP>')
                text = text.replace('/dev/shm/ct_0912_r2', '<TEMP>')
                text = text.replace('/dev/shm/ct_0913_r3', '<TEMP>')
                text = text.replace('/dev/shm/oc_0913_r1', '<TEMP>')
                configs.append(json.loads(text))
            differences = {k: (configs[0].get(k), configs[1].get(k))
                           for k in configs[0].keys() | configs[1].keys()
                           if configs[0].get(k) != configs[1].get(k)}
            self.assertEqual(differences, {})
            self.assertEqual(configs[1], configs[2])
            self.assertEqual(configs[2], configs[3])
            self.assertEqual(configs[3], configs[4])
        subprocess.run(['bash', '-n', str(RECORD/'run_command_trace_node43.sh')], check=True)
        subprocess.run(['bash', '-n', str(RECORD/'run_depot_issuer_node43.sh')], check=True)
        subprocess.run(['bash', '-n', str(RECORD/'run_depot_association_node43.sh')], check=True)
        subprocess.run(['bash', '-n', str(RECORD/'run_online_correction_node43.sh')], check=True)

    def test_historical_boundary_and_single_move(self):
        for i, row in enumerate(mod.FIXTURE['rows']):
            self.assertEqual(hashlib.sha256(row['prompt'].encode()).hexdigest(), row['prompt_sha256'])
            for arm in ('A', 'B'):
                answer = mod.select_answer(i, arm, row['prompt'], 'discarded generation')
                with patch.dict(os.environ, SC2_ORIGINAL_BO_SEGMENT='1'):
                    self.assertTrue(strict_validator()(row['prompt'], answer)[0])
                self.assertEqual(Counter(answer.splitlines()), Counter(row['answer'].splitlines()))
                if i < 2 or arm == 'A':
                    self.assertEqual(answer, row['answer'])
        row = mod.FIXTURE['rows'][2]
        a, b = row['answer'], mod.move_depot(row['answer'])
        self.assertEqual(a.replace('- supplydepot x 8\n', ''), b.replace('- supplydepot x 8\n', ''))
        self.assertEqual(b.splitlines()[1], '- supplydepot x 8')
        self.assertIn('Supply blocked (Available: 0', row['prompt'])
        with self.assertRaises(ValueError):
            mod.select_answer(2, 'B', 'different observation', '')
        self.assertEqual(mod.select_answer(3, 'B', 'new state', 'online'), 'online')

    def test_real_parser_preserves_all_quantities_and_order(self):
        row = mod.FIXTURE['rows'][2]
        p = row['prompt']
        obs = p.split('Current observations are:\n', 1)[1].split('The remaining BO list', 1)[0].strip()
        stage = re.search(r'Current Stage: (.*?) \((.*?)\)', p)
        strategy = dict(phase_summary=stage[1], timing=stage[2],
                        instruction=re.search(r'Detailed Instruction: (.*)', p)[1],
                        attack_instruction=re.search(r'Attacking Instruction: (.*)', p)[1])
        todo = {'Launch OFFENSE with 8 marine': 1, 'marine': 29, 'supplydepot': 8,
                'barracks': 3, 'reactor barracks': 3, 'scv': 2, 'refinery': 1}
        queues = []
        for answer in (row['answer'], mod.move_depot(row['answer'])):
            result = replay(answer, True, obs, todo, strategy,
                            controls={'SC2_ORIGINAL_BO_SEGMENT': '1', 'SC2_CLARIFY_BO_QUOTA': '1'})
            self.assertTrue(result['valid'], result['reason'])
            queues.append(result['plan'])
        expected = ['marine']*29 + ['supplydepot']*8 + ['barracks']*3 + ['reactor barracks']*3 + ['scv']*2 + ['refinery']
        self.assertEqual(queues[0], expected)
        self.assertEqual(queues[1], ['supplydepot']*8 + expected[:29] + expected[37:])
        self.assertEqual(Counter(queues[0]), Counter(queues[1]))
        self.assertEqual(len(queues[0]), 46)

    def test_single_seed_batch(self):
        rng = mod.EightSeeds()
        self.assertEqual(rng.randint(2**16, 2**32-1, 8).tolist(), [52620]*8)
        with self.assertRaises(ValueError):
            rng.randint(2**16, 2**32-1, 8)
        self.assertEqual(mod.ARMS, ['A', 'B']*4)

    def test_observer_preserves_original_event(self):
        events = []
        async def original(player, unit):
            events.append((player, unit))
            return 'unchanged'
        unit = type('Unit', (), dict(tag=10, type_id=type('ID', (), {'name': 'SUPPLYDEPOT'}),
                                     build_progress=1.0, position=(1, 2)))()
        with patch.object(mod, 'write_game_event') as write:
            result = asyncio.run(mod.observe_callback(original, 'complete')('player', unit))
        self.assertEqual(result, 'unchanged')
        self.assertEqual(events, [('player', unit)])
        write.assert_called_once()

    def test_original_model_evaluation_only_config(self):
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'model').mkdir(); (root/'sc2').mkdir()
            output = root/'run'
            env = dict(os.environ, DRY_RUN='1', VERIFY_ASSETS='0', PYTHON_BIN=sys.executable,
                       SOURCE_SHA=sha, MODEL_PATH=str(root/'model'), SC2PATH=str(root/'sc2'),
                       OUTPUT_DIR=str(output), RUN_ID='supply-config-test', RAY_TMPDIR='/tmp/supply-config-test')
            result = subprocess.run(['bash', str(RECORD/'run_supply_order_node44.sh')],
                                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            cfg = json.loads((output/'effective_config.json').read_text())['overrides']
            manifest = json.loads((output/'manifest.json').read_text())
            self.assertEqual(manifest['mode'], 'evaluation')
            self.assertEqual(manifest['world_size'], 8)
            for key, value in {
                'trainer.val_only': 'True', 'trainer.resume_mode': 'disable', 'trainer.save_freq': '-1',
                'trainer.total_epochs': '1', 'env.rollout.n': '8', '+env.val_rollout_n': '8',
                'actor_rollout_ref.rollout.val_kwargs.n': '8',
                'actor_rollout_ref.rollout.val_kwargs.temperature': '0.4',
                'actor_rollout_ref.rollout.val_kwargs.do_sample': 'True',
                '+ray_init.runtime_env.worker_process_setup_hook': 'supply_order_eval.install',
                '+ray_init.runtime_env.env_vars.SC2_EXECUTION_FAILURE_FEEDBACK': "'1'",
                '+ray_init.runtime_env.env_vars.SC2_FIXED_BO_PREFIX_LIMIT': "'-1'",
                '+ray_init.runtime_env.env_vars.SC2_ORIGINAL_BO_SEGMENT': "'1'",
                'data.max_response_length': '192', 'env.history_length': '0',
                'algorithm.gamma': '0.95', 'actor_rollout_ref.actor.optim.lr': '1e-6',
                '+actor_rollout_ref.actor.fp32_normalization': 'True',
                '+actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype': 'fp32',
                '+actor_rollout_ref.model.override_config.swa_dropout': '0.0',
            }.items():
                self.assertEqual(cfg[key], value, key)
            self.assertNotIn('trainer.resume_from_path', cfg)
        subprocess.run(['bash', '-n', str(RECORD/'run_supply_order_node44.sh')], check=True)


if __name__ == '__main__':
    unittest.main()
