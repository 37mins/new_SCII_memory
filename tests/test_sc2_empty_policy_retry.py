"""CPU-only regression of the actual collector, driver boundary and run audit."""
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from agent_system.sc2_episode import policy_transitions
from test_r6_natural_termination import ROOT, method
import test_sc2_evaluation_without_policy as evaluation
import test_system_release as release


class Config(dict):
    __getattr__ = dict.__getitem__


class EmptyPolicyRetryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log = Path(self.tmp.name) / 'empty_policy_batches.jsonl'
        self.config = SimpleNamespace(
            trainer=Config(sc2_empty_policy_max_consecutive=3, sc2_empty_policy_max_total=4,
                           sc2_empty_policy_log_path=str(self.log), val_only=False),
            env=SimpleNamespace(env_name='SC2', rollout=SimpleNamespace(n=1)),
            algorithm=SimpleNamespace(filter_groups=SimpleNamespace(enable=False)))
        self.collect = method(ROOT/'verl/trainer/ppo/ray_trainer.py', 'RayPPOTrainer',
                              '_collect_training_batch', deepcopy=deepcopy, json=json)
        self.loop = method(ROOT/'agent_system/multi_turn_rollout/rollout_loop.py',
                           'TrajectoryCollector', 'multi_turn_loop', DataProto=evaluation.Batch,
                           EnvironmentManagerBase=object, policy_transitions=policy_transitions)

    def empty(self, reset_index=0):
        return evaluation.Batch(meta_info={'sc2_empty_policy_batch': {
            'reset_index': reset_index,
            'episodes': [dict(env_index=i, policy_decisions=0, reward=20,
                              success={'success_rate': 1}) for i in range(8)]}})

    def trainer(self, outputs):
        queue = iter(outputs)
        seen = []
        def rollout(**kwargs):
            seen.append(deepcopy(kwargs['gen_batch']))
            kwargs['gen_batch'].non_tensor_batch.pop('env_kwargs', None)
            value = next(queue)
            if isinstance(value, Exception):
                raise value
            return value
        trainer = SimpleNamespace(config=self.config, sc2_empty_policy_batches=0, global_steps=1,
                                  actor_rollout_wg=object(), envs=object(),
                                  traj_collector=SimpleNamespace(multi_turn_loop=rollout))
        return trainer, seen

    def entries(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_empty_then_real_does_not_advance_global_step_or_mutate_input(self):
        real = evaluation.Batch(batch={'responses': [1]}, meta_info={})
        trainer, seen = self.trainer([self.empty(), real])
        original = evaluation.Batch(non_tensor_batch={'env_kwargs': [{'map': 'Flat128'}]})
        self.assertIs(self.collect(trainer, original), real)
        self.assertEqual(trainer.global_steps, 1)
        self.assertEqual(len(seen), 2)
        self.assertEqual([b.non_tensor_batch for b in seen], [original.non_tensor_batch]*2)
        self.assertEqual(trainer.sc2_empty_policy_batches, 1)
        self.assertEqual(self.entries()[0]['optimizer_updates'], 0)
        self.assertEqual(self.entries()[0]['action'], 'skip')
        # Actual fit invokes this boundary before returns, balancing or actor update.
        tree = ast.parse((ROOT/'verl/trainer/ppo/ray_trainer.py').read_text())
        fit = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'fit')
        calls = {n.func.attr: n.lineno for n in ast.walk(fit)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertLess(calls['_collect_training_batch'], calls['update_actor'])

    def test_third_consecutive_empty_stops_without_fourth_collection(self):
        trainer, seen = self.trainer([self.empty(i) for i in range(4)])
        with self.assertRaisesRegex(RuntimeError, 'budget exhausted'):
            self.collect(trainer, evaluation.Batch())
        self.assertEqual(len(seen), 3)
        self.assertEqual(trainer.global_steps, 1)
        self.assertEqual([r['action'] for r in self.entries()], ['skip', 'skip', 'stop'])

    def test_total_limit_persists_across_successful_global_batches(self):
        real = evaluation.Batch(meta_info={})
        trainer, seen = self.trainer([self.empty(), real, self.empty(2), real,
                                      self.empty(4), real, self.empty(6), real])
        for _ in range(3):
            self.assertIs(self.collect(trainer, evaluation.Batch()), real)
            trainer.global_steps += 1
        with self.assertRaisesRegex(RuntimeError, 'budget exhausted'):
            self.collect(trainer, evaluation.Batch())
        self.assertEqual(len(seen), 7)
        self.assertEqual(self.entries()[-1]['consecutive_empty_batches'], 1)
        self.assertEqual(self.entries()[-1]['total_empty_batches'], 4)

    def test_other_errors_and_disabled_mode_never_retry(self):
        for limit in (0, 3):
            self.config.trainer['sc2_empty_policy_max_consecutive'] = limit
            trainer, seen = self.trainer([RuntimeError('Unconsumed nonterminal policy response')])
            with self.assertRaisesRegex(RuntimeError, 'Unconsumed nonterminal'):
                self.collect(trainer, evaluation.Batch())
            self.assertEqual(len(seen), 1)
            self.assertFalse(self.log.exists())

    def test_invalid_limits_and_dynamic_sampling_are_rejected_before_collection(self):
        self.config.algorithm.filter_groups.enable = True
        trainer, seen = self.trainer([])
        with self.assertRaises(ValueError):
            self.collect(trainer, evaluation.Batch())
        self.assertFalse(seen)

    def test_original_reset_advances_seed_stream_without_reseeding(self):
        from unittest.mock import patch
        reset = method(ROOT/'agent_system/environments/env_package/sc2_online_restored/envSC2.py',
                       'SC2MultiProcessEnv', 'reset', np=np, os=os, json=json,
                       ray=SimpleNamespace(get=lambda values: values))
        seeds = []
        worker = SimpleNamespace(reset=SimpleNamespace(
            remote=lambda seed: (seeds.append(seed), {})))
        raw = SimpleNamespace(is_train=True, env_num=1, group_n=8, seed=0, reset_index=0,
                              training_rng=np.random.RandomState(0), workers=[worker]*8)
        with patch.dict(os.environ, {'SC2_TRAIN_SEEDS_LOG_PATH': str(Path(self.tmp.name)/'seeds.jsonl')}):
            for _ in range(3):
                reset(raw)
        self.assertEqual(seeds, [2732]*8 + [43567]*8 + [42613]*8)
        self.assertEqual(raw.reset_index, 3)

    def loop_output(self, natural=True, applied=False, enabled=True):
        self.config.trainer['sc2_empty_policy_max_consecutive'] = 3 if enabled else 0
        trajectories = [[{**evaluation.transition(20, applied, True, uid=str(i)),
                           'sc2_natural_terminal': natural}] for i in range(8)]
        original = deepcopy(trajectories)
        gather = method(ROOT/'agent_system/multi_turn_rollout/rollout_loop.py',
                        'TrajectoryCollector', 'gather_rollout_data', List=list, Dict=dict,
                        np=np, DataProto=SimpleNamespace(from_single_dict=lambda data: data),
                        collate_fn=lambda rows: rows, policy_transitions=policy_transitions)
        collector = SimpleNamespace(config=self.config)
        collector.vanilla_multi_turn_loop = lambda **kw: (
            trajectories, np.array([20]*8), np.array([1]*8),
            {'success_rate': np.array([1]*8)}, np.array([str(i) for i in range(8)]), np.zeros(8))
        collector.gather_rollout_data = lambda **kw: gather(collector, **kw)
        output = self.loop(collector, evaluation.Batch(), None,
                           SimpleNamespace(_episode_reset_index=0), is_train=True)
        self.assertEqual(trajectories, original)
        return output

    def test_eight_natural_zero_decision_wins_return_evidence_not_ppo_samples(self):
        output = self.loop_output()
        self.assertIsNone(output.batch)
        self.assertEqual(len(output.meta_info['sc2_empty_policy_batch']['episodes']), 8)

    def test_empty_truncated_or_missing_evidence_still_fails(self):
        with self.assertRaisesRegex(RuntimeError, 'natural terminal evidence'):
            self.loop_output(natural=False)

    def test_default_empty_guard_and_real_sample_credit_are_unchanged(self):
        with self.assertRaisesRegex(RuntimeError, 'No consumed policy actions'):
            self.loop_output(enabled=False)
        self.assertEqual(self.loop_output(applied=True, enabled=False),
                         self.loop_output(applied=True, enabled=True))

    def test_finish_counts_skipped_games_without_forging_an_extra_training_step(self):
        root = Path(self.tmp.name)
        release.ResultAuditTest().fixture(root)
        (root/'manifest.json').write_text(json.dumps(
            {'target_steps': 1, 'world_size': 8, 'episode_api': 'gymnasium-v0.26'}))
        (root/'effective_config.json').write_text(json.dumps({'overrides': {
            'data.train_batch_size': '1', 'env.rollout.n': '8'}}))
        outcomes = [dict(reset_index=r, env_index=i, terminated=True, truncated=False,
                         game_result='Victory', policy_decision_count=r)
                    for r in range(2) for i in range(8)]
        (root/'episode_outcomes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in outcomes))
        trainer, _ = self.trainer([self.empty(), evaluation.Batch(meta_info={})])
        self.collect(trainer, evaluation.Batch())
        result = release.finish(root, 0)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['completed_steps'], 1)
        self.assertEqual(result['episode_outcomes']['trajectories'], 16)
        # Skipping a consumed game or a budget-stop must not pass the finish audit.
        rows = self.entries()
        rows[0]['reset_index'] = 1
        self.log.write_text(json.dumps(rows[0])+'\n')
        self.assertEqual(release.finish(root, 0)['status'], 'failed')
        rows[0].update(reset_index=0, action='stop')
        self.log.write_text(json.dumps(rows[0])+'\n')
        self.assertEqual(release.finish(root, 0)['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
