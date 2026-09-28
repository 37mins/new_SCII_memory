"""CPU tests of real GiGPO scoring and the driver sampling boundary; no SC2/RPC."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from contextlib import ExitStack

import numpy as np
from omegaconf import OmegaConf
import torch

from agent_system.reward_manager.episode import EpisodeRewardManager
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.sc2_task_signal import inspect_task_signal


def config(path):
    return OmegaConf.create(dict(
        trainer=dict(default_local_dir=str(path), sc2_signal_max_batches=4,
                     sc2_signal_max_attempts=3, sc2_skip_zero_signal=True,
                     sc2_empty_policy_max_consecutive=0),
        env=dict(env_name='SC2', rollout=dict(n=8)),
        algorithm=dict(adv_estimator='gigpo', gamma=.95,
                       gigpo=dict(mode='mean_norm', step_advantage_w=1.,
                                  enable_similarity=False, similarity_thresh=.95)),
        actor_rollout_ref=dict(actor=dict(use_invalid_action_penalty=True,
                                         invalid_action_penalty_coef=.1))))


def group(returns, calls=None):
    calls = calls or [2]*8
    reward, trajectories, anchors, totals, lengths = [], [], [], [], []
    for i, (total, n) in enumerate(zip(returns, calls)):
        for call in range(n):
            reward.append(total if call == n-1 else 0.)
            trajectories.append(f'env{i}')
            anchors.append(f'call{call}')
            totals.append(total)
            lengths.append(n)
    n = len(reward)
    tensors = dict(prompts=torch.ones(n, 2, dtype=torch.long),
                   responses=torch.ones(n, 3, dtype=torch.long),
                   input_ids=torch.ones(n, 5, dtype=torch.long),
                   attention_mask=torch.ones(n, 5, dtype=torch.long),
                   position_ids=torch.arange(5).expand(n, -1).clone())
    return DataProto.from_dict(tensors=tensors, non_tensors={
        'uid': np.array(['group0']*n, dtype=object),
        'traj_uid': np.array(trajectories, dtype=object),
        'anchor_obs': np.array(anchors, dtype=object),
        'rewards': np.array(reward, dtype=object),
        'active_masks': np.ones(n, dtype=bool),
        'episode_rewards': np.array(totals, dtype=object),
        'episode_lengths': np.array(lengths, dtype=object),
        'is_action_valid': np.array([[True]]*n, dtype=object),
        'data_source': np.array(['sc2']*n, dtype=object),
    })


class TaskSignalTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.cfg = config(self.path)
        self.reward = EpisodeRewardManager(SimpleNamespace(decode=lambda *a, **kw: 'text'), 0)

    def trainer(self, outputs):
        trainer = RayPPOTrainer.__new__(RayPPOTrainer)
        trainer.config, trainer.reward_fn = self.cfg, self.reward
        trainer.global_steps = 5
        trainer.sc2_signal_start_step = trainer._last_saved_global_step = 4
        trainer.sc2_signal_batches = trainer.sc2_signal_skipped = 0
        trainer.total_training_steps = 96
        trainer.envs = SimpleNamespace(_episode_reset_index=-1)
        # Any accidental model/optimizer RPC from collection fails immediately.
        trainer.actor_rollout_wg = object()
        queue, seen = iter(outputs), []
        def collect(**kw):
            seen.append(deepcopy(kw['gen_batch']))
            kw['gen_batch']['nested'].append('mutated')
            trainer.envs._episode_reset_index += 1
            value = next(queue)
            if isinstance(value, Exception):
                raise value
            return value
        trainer.traj_collector = SimpleNamespace(multi_turn_loop=collect)
        return trainer, seen

    def entries(self):
        return [json.loads(line) for line in (self.path/'signal_sampling.jsonl').read_text().splitlines()]

    def test_identical_wins_and_losses_are_exact_zero(self):
        for total in (20.5, -50., -49.375):
            result = inspect_task_signal(group([total]*8), self.cfg, self.reward)
            self.assertFalse(result['has_task_signal'])
            self.assertEqual(result['max_abs_task_advantage'], 0)

    def test_mixed_outcomes_keeps_all_real_responses_including_loss(self):
        batch = group([20.5]*7+[-50.])
        before = DataProto.from_dict(tensors={k: v.clone() for k, v in batch.batch.items()},
                                     non_tensors=deepcopy(batch.non_tensor_batch))
        result = inspect_task_signal(batch, self.cfg, self.reward)
        self.assertTrue(result['has_task_signal'])
        self.assertEqual(result['real_responses'], 16)
        self.assertEqual(result['episode_returns']['env7'], -50.)
        self.assertFalse(batch.batch.is_locked)
        for k in batch.batch.keys():
            self.assertTrue(torch.equal(batch.batch[k], before.batch[k]))
        for k in batch.non_tensor_batch:
            self.assertTrue(np.array_equal(batch.non_tensor_batch[k], before.non_tensor_batch[k]))

    def test_all_fail_but_nonzero_reward_difference_is_not_discarded(self):
        self.assertTrue(inspect_task_signal(group([-50.]*7+[-49.99]), self.cfg, self.reward)['has_task_signal'])

    def test_equal_score_rounding_residual_is_not_task_signal(self):
        result = inspect_task_signal(group([-49.382]*8), self.cfg, self.reward)
        self.assertGreater(result['max_abs_task_advantage'], 0)
        self.assertFalse(result['has_task_signal'])
        self.assertEqual(result['signal_kind'], 'reduction_residual_only')
        self.assertFalse(result['episode_score_varies'])
        self.assertEqual(result['step_groups_with_score_variation'], 0)

    def test_one_float32_ulp_reward_difference_is_kept_without_epsilon(self):
        total = np.float32(-49.382)
        nearby = np.nextafter(total, np.float32(0))
        result = inspect_task_signal(group([total]*7+[nearby]), self.cfg, self.reward)
        self.assertTrue(result['has_task_signal'])
        self.assertTrue(result['episode_score_varies'])

    def test_same_episode_reward_is_not_the_gate_when_discounted_returns_differ(self):
        result = inspect_task_signal(group([20.5]*8, calls=[2]*7+[3]), self.cfg, self.reward)
        self.assertTrue(result['has_task_signal'])
        self.assertEqual(result['signal_kind'], 'step_score_difference_only')
        self.assertFalse(result['episode_return_varies'])

    def test_differences_between_distinct_step_groups_do_not_create_signal(self):
        batch = group([20.5]*8, calls=[2]*7+[3])
        batch.non_tensor_batch['anchor_obs'] = np.array([f'unique{i}' for i in range(len(batch))], dtype=object)
        result = inspect_task_signal(batch, self.cfg, self.reward)
        self.assertFalse(result['has_task_signal'])
        self.assertEqual(result['step_groups_with_score_variation'], 0)

    def test_disabled_step_weight_does_not_admit_step_only_variation(self):
        self.cfg.algorithm.gigpo.step_advantage_w = 0.
        result = inspect_task_signal(group([20.5]*8, calls=[2]*7+[3]), self.cfg, self.reward)
        self.assertFalse(result['has_task_signal'])

    def test_invalid_action_penalty_is_preserved(self):
        batch = group([20.5]*8)
        batch.non_tensor_batch['is_action_valid'][0] = [False]
        self.assertTrue(inspect_task_signal(batch, self.cfg, self.reward)['has_task_signal'])

    def test_later_invalid_penalty_is_checked_in_step_group_not_episode_score(self):
        batch = group([20.5]*8)
        batch.non_tensor_batch['is_action_valid'][1] = [False]
        result = inspect_task_signal(batch, self.cfg, self.reward)
        self.assertTrue(result['has_task_signal'])
        self.assertFalse(result['episode_score_varies'])
        self.assertEqual(result['signal_kind'], 'step_score_difference_only')

    def test_no_standardization_formula_change(self):
        self.cfg.algorithm.gigpo.mode = 'mean_std_norm'
        self.assertFalse(inspect_task_signal(group([-49.382]*8), self.cfg, self.reward)['has_task_signal'])
        self.assertTrue(inspect_task_signal(group([20.5]*7+[-50.]), self.cfg, self.reward)['has_task_signal'])

    def test_masked_padding_does_not_create_signal(self):
        batch = group([20.5]*8)
        batch.batch['attention_mask'][:, -1] = 0
        self.assertFalse(inspect_task_signal(batch, self.cfg, self.reward)['has_task_signal'])

    def test_empty_nonfinite_and_multiple_groups_are_not_skipped_silently(self):
        with self.assertRaisesRegex(RuntimeError, 'No real'):
            inspect_task_signal(DataProto(), self.cfg, self.reward)
        with self.assertRaises(FloatingPointError):
            inspect_task_signal(group([float('nan')]*8), self.cfg, self.reward)
        batch = group([20.5]*8)
        batch.non_tensor_batch['uid'][0] = 'other'
        with self.assertRaises(ValueError):
            inspect_task_signal(batch, self.cfg, self.reward)

    def test_inspection_preserves_numpy_rng_including_reward_debug(self):
        original = np.random.get_state()
        self.reward.num_examine = 1
        inspect_task_signal(group([20.5]*8), self.cfg, self.reward)
        current = np.random.get_state()
        self.assertTrue(np.array_equal(original[1], current[1]))
        self.assertEqual(original[2:], current[2:])

    def test_zero_then_signal_reuses_no_samples_and_keeps_pending_step(self):
        real = group([20.5]*7+[-50.])
        trainer, seen = self.trainer([group([20.5]*8), real])
        prompt = {'nested': []}
        self.assertIs(trainer._collect_training_batch(prompt), real)
        self.assertEqual(prompt, {'nested': []})
        self.assertEqual(seen, [prompt, prompt])
        self.assertEqual(trainer.global_steps, 5)
        self.assertEqual([r['action'] for r in self.entries()], ['skip_zero_signal', 'accept'])
        self.assertEqual(self.entries()[-1]['sampled_games'], 16)

    def test_residual_then_new_scene_keeps_entire_group_without_old_sample_reuse(self):
        real = group([20.5]*7+[-50.])
        real.non_tensor_batch['uid'][:] = 'next_scene'
        trainer, _ = self.trainer([group([-49.382]*8), real])
        result = trainer._collect_training_batch({'nested': []})
        self.assertIs(result, real)
        self.assertEqual(set(result.non_tensor_batch['uid']), {'next_scene'})
        self.assertEqual(self.entries()[0]['signal_kind'], 'reduction_residual_only')
        self.assertEqual([r['action'] for r in self.entries()], ['skip_zero_signal', 'accept'])
        self.assertEqual(trainer.global_steps, 5)

    def test_attempt_cap_never_falls_back_to_zero_group(self):
        trainer, seen = self.trainer([group([20.5]*8) for _ in range(4)])
        self.assertIsNone(trainer._collect_training_batch({'nested': []}))
        self.assertEqual(len(seen), 3)
        self.assertEqual(trainer.global_steps, 5)
        self.assertEqual(trainer.sc2_signal_skipped, 3)

    def test_total_cap_spans_collection_windows_and_prevents_extra_game(self):
        trainer, seen = self.trainer([group([20.5]*8) for _ in range(5)])
        for _ in range(3):
            self.assertIsNone(trainer._collect_training_batch({'nested': []}))
        self.assertEqual(len(seen), 4)
        self.assertTrue(self.entries()[-1]['total_limit_reached'])

    def test_control_accepts_zero_but_shares_game_budget_and_scoring(self):
        self.cfg.trainer.sc2_skip_zero_signal = False
        zero = group([20.5]*8)
        trainer, seen = self.trainer([zero]*5)
        for _ in range(4):
            self.assertIs(trainer._collect_training_batch({'nested': []}), zero)
        self.assertIsNone(trainer._collect_training_batch({'nested': []}))
        self.assertEqual(len(seen), 4)
        self.assertEqual(trainer.sc2_signal_skipped, 0)
        self.assertTrue(all(not r['has_task_signal'] and r['action'] == 'accept' for r in self.entries()))

    def test_runtime_error_is_not_retried(self):
        trainer, seen = self.trainer([RuntimeError('SC2 timeout'), group([20.5]*8)])
        with self.assertRaisesRegex(RuntimeError, 'SC2 timeout'):
            trainer._collect_training_batch({'nested': []})
        self.assertEqual(len(seen), 1)

    def test_default_off_never_runs_new_inspection(self):
        self.cfg.trainer.sc2_signal_max_batches = 0
        self.cfg.trainer.sc2_skip_zero_signal = False
        zero = group([20.5]*8)
        trainer, seen = self.trainer([zero])
        with patch('verl.trainer.ppo.sc2_task_signal.inspect_task_signal', side_effect=AssertionError('gate enabled')):
            self.assertIs(trainer._collect_training_batch({'nested': []}), zero)
        self.assertEqual(len(seen), 1)
        self.assertFalse((self.path/'signal_sampling.jsonl').exists())

    def test_real_config_validation_and_no_unbounded_or_nested_sampling(self):
        base = OmegaConf.load(Path(__file__).resolve().parents[1]/'verl/trainer/config/ppo_trainer.yaml')
        self.cfg = OmegaConf.merge(base, self.cfg, dict(data=dict(train_batch_size=1),
            trainer=dict(n_gpus_per_node=4, nnodes=1),
            actor_rollout_ref=dict(actor=dict(ppo_micro_batch_size_per_gpu=1),
                                  rollout=dict(log_prob_micro_batch_size_per_gpu=1))))
        trainer, _ = self.trainer([])
        trainer.use_critic = trainer.use_reference_policy = False
        trainer._validate_config()
        for field, value in [('trainer.sc2_signal_max_batches', 0),
                             ('trainer.sc2_signal_max_attempts', 0),
                             ('algorithm.filter_groups.enable', True),
                             ('trainer.sc2_empty_policy_max_consecutive', 3),
                             ('algorithm.use_kl_in_reward', True)]:
            saved = OmegaConf.select(self.cfg, field)
            OmegaConf.update(self.cfg, field, value)
            with self.assertRaisesRegex(ValueError, 'Zero-signal sampling requires'):
                trainer._validate_config()
            OmegaConf.update(self.cfg, field, saved)

    def test_finish_saves_last_real_step_not_pending_and_zero_updates_no_checkpoint(self):
        for steps in (0, 2):
            trainer, _ = self.trainer([])
            trainer.global_steps = 5+steps
            saved = []
            trainer._save_checkpoint = lambda: saved.append(trainer.global_steps)
            summary_path = self.path/'signal_sampling_summary.json'
            if summary_path.exists():
                summary_path.unlink()
            trainer._finish_signal_sampling('rollout_budget_exhausted')
            summary = json.loads(summary_path.read_text())
            self.assertEqual(saved, [6] if steps else [])
            self.assertEqual(summary['completed_global_steps'], steps)
            self.assertEqual(summary['last_global_step'], 4+steps)
            self.assertFalse(summary['update_target_reached'])
            self.assertEqual(trainer.global_steps, 5+steps)

    def run_fit(self, totals, skip):
        """Real driver fit/reward/advantage; fake only SC2, model RPC and disk IO."""
        self.cfg = OmegaConf.merge(self.cfg, dict(
            trainer=dict(project_name='cpu', experiment_name='cpu', logger=[], total_epochs=4,
                         balance_batch=False, critic_warmup=0, save_freq=-1,
                         sc2_skip_zero_signal=skip),
            algorithm=dict(lam=1., use_kl_in_reward=False, use_pf_ppo=False,
                           pf_ppo=dict(reweight_method='pow', weight_pow=2.)),
            reward_model=dict(launch_reward_fn_async=False),
            actor_rollout_ref=dict(actor=dict(loss_agg_mode='token-mean'),
                                  rollout=dict(n=1, multi_turn=dict(enable=False)))))
        trainer, _ = self.trainer([])
        queue = iter(totals)
        def collect(**kwargs):
            trainer.envs._episode_reset_index += 1
            return group(next(queue))
        trainer.traj_collector = SimpleNamespace(multi_turn_loop=collect)
        trainer._load_checkpoint = lambda: setattr(trainer, 'global_steps', 4)
        trainer.val_reward_fn = None
        trainer.use_critic = trainer.use_rm = trainer.use_reference_policy = False
        trainer.resource_pool_manager = SimpleNamespace(get_n_gpus=lambda: 4)
        trainer.train_dataloader = [dict(input_ids=torch.ones(1, 2, dtype=torch.long),
                                        attention_mask=torch.ones(1, 2, dtype=torch.long),
                                        position_ids=torch.zeros(1, 2, dtype=torch.long),
                                        raw_prompt_ids=np.array(['prompt'], dtype=object),
                                        data_source=np.array(['sc2'], dtype=object))]
        def old(batch):
            z = torch.zeros_like(batch.batch['responses'], dtype=torch.float32)
            return DataProto.from_dict(tensors={'old_log_probs': z, 'entropys': z.clone()})
        actor = SimpleNamespace(compute_log_prob=Mock(side_effect=old),
            update_actor=Mock(return_value=DataProto(meta_info={'metrics': {'actor/pg_loss': [0.]}})))
        trainer.actor_rollout_wg = actor
        saved = []
        def save():
            saved.append(trainer.global_steps)
            trainer._last_saved_global_step = trainer.global_steps
        trainer._save_checkpoint = save
        with ExitStack() as stack:
            tracker = stack.enter_context(patch('verl.utils.tracking.Tracking'))
            stack.enter_context(patch('verl.trainer.ppo.ray_trainer.adjust_batch', side_effect=lambda cfg, b: b))
            for name in ('compute_data_metrics', 'compute_timing_metrics', 'compute_throughout_metrics'):
                stack.enter_context(patch('verl.trainer.ppo.ray_trainer.'+name, return_value={}))
            trainer.fit()
        summary = json.loads((self.path/'signal_sampling_summary.json').read_text())
        return actor, saved, summary, tracker.return_value.log.call_args_list

    def test_real_fit_all_zero_never_calls_old_or_optimizer_or_logs_fake_steps(self):
        actor, saved, summary, logs = self.run_fit([[20.5]*8]*4, True)
        actor.compute_log_prob.assert_not_called()
        actor.update_actor.assert_not_called()
        self.assertEqual(saved, [])
        self.assertEqual(logs, [])
        self.assertEqual(summary['sampled_games'], 32)
        self.assertEqual(summary['last_global_step'], 4)

    def test_real_fit_residual_only_budget_never_updates(self):
        actor, saved, summary, logs = self.run_fit([[-49.382]*8]*4, True)
        actor.compute_log_prob.assert_not_called()
        actor.update_actor.assert_not_called()
        self.assertEqual(saved, [])
        self.assertEqual(logs, [])
        self.assertEqual(summary['sampled_games'], 32)
        self.assertEqual(summary['skipped_batches'], 4)

    def test_real_fit_residual_mixed_residual_mixed_has_two_real_updates(self):
        residual, mixed = [-49.382]*8, [20.5]*7+[-50.]
        actor, saved, summary, logs = self.run_fit([residual, mixed, residual, mixed], True)
        self.assertEqual(actor.update_actor.call_count, 2)
        self.assertEqual(saved, [6])
        self.assertEqual([item.kwargs['step'] for item in logs], [5, 6])
        self.assertEqual(summary['sampled_games'], 32)
        self.assertEqual(summary['skipped_batches'], 2)

    def test_real_fit_skips_two_groups_but_continues_two_actual_updates(self):
        mixed = [20.5]*7+[-50.]
        actor, saved, summary, logs = self.run_fit([[20.5]*8, mixed, [20.5]*8, mixed], True)
        self.assertEqual(actor.update_actor.call_count, 2)
        self.assertEqual(actor.compute_log_prob.call_count, 2)
        self.assertEqual(saved, [6])
        self.assertEqual([item.kwargs['step'] for item in logs], [5, 6])
        self.assertEqual(summary['sampled_games'], 32)
        self.assertEqual(summary['skipped_batches'], 2)

    def test_real_fit_control_updates_four_times_on_same_game_budget(self):
        actor, saved, summary, logs = self.run_fit([[20.5]*8]*4, False)
        self.assertEqual(actor.update_actor.call_count, 4)
        self.assertEqual(saved, [8])
        self.assertEqual(summary['sampled_games'], 32)
        self.assertEqual(summary['skipped_batches'], 0)


if __name__ == '__main__':
    unittest.main()
