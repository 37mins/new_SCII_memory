"""CPU tests of the actual SC2 reset and trainer save/load boundaries."""
import json
import contextlib
import io
import os
from pathlib import Path
import random
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from agent_system.sc2_resume import environment_state, restore_environment, legacy_environment_state
from test_r6_natural_termination import ROOT, method
from test_sc2_empty_policy_retry import Config
from run_record import finish


def manager(is_train=True):
    worker = SimpleNamespace(reset=SimpleNamespace(remote=lambda seed: (seed, {})))
    raw = SimpleNamespace(is_train=is_train, env_num=1, group_n=8, num_processes=8,
                          seed=0, reset_index=0, training_rng=np.random.RandomState(0),
                          validation_rng=np.random.RandomState(0), workers=[worker] * 8)
    return SimpleNamespace(envs=raw, memory=None, _terminal_by_env=[],
                           _episode_reset_index=-1, _step_index=0)


class SeedResumeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.reset = method(ROOT / 'agent_system/environments/env_package/sc2_online_restored/envSC2.py',
                            'SC2MultiProcessEnv', 'reset', np=np, os=os, json=json, random=random,
                            ray=SimpleNamespace(get=lambda results: results))

    def advance(self, m):
        with patch.dict(os.environ, {'SC2_TRAIN_SEEDS_LOG_PATH': '', 'SC2_EVAL_SEEDS_LOG_PATH': ''}), contextlib.redirect_stdout(io.StringIO()):
            seeds, _ = self.reset(m.envs)
        m._episode_reset_index += 1
        return seeds

    def legacy(self):
        m = manager()
        rows = [dict(base_seed=0, reset_index=i, seeds=self.advance(m)) for i in range(4)]
        result = dict(status='completed', process_exit_code=0, target_steps=4,
                      completed_steps=4, checkpoint_complete=True, empty_policy_batches=[])
        for name in ('result.json', '.done'):
            (self.root / name).write_text(json.dumps(result))
        (self.root / 'training_seeds.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
        outcomes = [dict(reset_index=i, env_index=e, terminated=True, truncated=False)
                    for i in range(4) for e in range(8)]
        (self.root / 'episode_outcomes.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in outcomes))
        (self.root / 'prompt_response.jsonl').write_text(json.dumps({'step_index': 30})+'\n')
        return m

    def test_actual_reset_restored_remaining_92_batches_equal_continuous(self):
        continuous = manager()
        all96 = [self.advance(continuous) for _ in range(96)]
        prefix = self.legacy()
        restored = manager()
        state = legacy_environment_state(restored, self.root, 4)
        restore_environment(restored, state)
        self.assertEqual(restored._step_index, 31)
        self.assertEqual(restored._episode_reset_index, 3)
        self.assertEqual(restored.envs.reset_index, 4)
        self.assertEqual([self.advance(restored) for _ in range(92)], all96[4:])
        self.assertEqual(all96[4], [45891] * 8)
        # Direct checkpoint state has the same sequence, including the validation stream.
        restored = manager()
        restore_environment(restored, environment_state(prefix))
        self.assertEqual(self.advance(restored), all96[4])
        val = manager(False)
        self.advance(val)
        val_restored = manager(False)
        restore_environment(val_restored, environment_state(val))
        self.assertEqual(self.advance(val_restored), self.advance(val))

    def test_mismatched_or_missing_legacy_evidence_is_not_guessed(self):
        self.legacy()
        path = self.root / 'training_seeds.jsonl'
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        rows[2]['seeds'][0] += 1
        path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
        with self.assertRaisesRegex(ValueError, 'seed mismatch'):
            legacy_environment_state(manager(), self.root, 4)
        self.legacy()
        (self.root / '.done').unlink()
        with self.assertRaises(FileNotFoundError):
            legacy_environment_state(manager(), self.root, 4)
        self.legacy()
        with self.assertRaisesRegex(ValueError, 'completed, unskipped'):
            legacy_environment_state(manager(), self.root, 3)

    def test_original_game_seeds_resume_same_92_batches(self):
        with patch.dict(os.environ, SC2_ORIGINAL_GAME_SEEDS='1'):
            continuous = manager()
            all96 = [self.advance(continuous) for _ in range(96)]
            prefix = manager()
            for _ in range(4): self.advance(prefix)
            restored = manager()
            restore_environment(restored, environment_state(prefix))
            self.assertEqual([self.advance(restored) for _ in range(92)], all96[4:])
            self.assertEqual(all96[4], [60,71,12,45,55,40,78,81])

    def test_configuration_counters_and_live_games_are_checked(self):
        original = manager()
        state = environment_state(original)
        for name, value in [('group_n', 4), ('seed', 1), ('is_train', False)]:
            changed = manager()
            setattr(changed.envs, name, value)
            with self.assertRaisesRegex(ValueError, 'configuration mismatch'):
                restore_environment(changed, state)
        state['reset_index'] = 1
        with self.assertRaisesRegex(ValueError, 'counters disagree'):
            restore_environment(original, state)
        original._terminal_by_env = [True, False]
        with self.assertRaisesRegex(ValueError, 'live SC2 game'):
            environment_state(original)

    def test_resumed_run_audit_expects_only_new_steps_and_new_games(self):
        root = self.root
        (root / 'manifest.json').write_text(json.dumps(dict(start_step=4, target_steps=8,
            world_size=4, sc2_checkpoint_resume=True, episode_api='gymnasium-v0.26')))
        (root / 'effective_config.json').write_text(json.dumps({'overrides': {
            'data.train_batch_size': '1', 'env.rollout.n': '8'}}))
        log = ''.join(f'step:{s} - actor/grad_norm:2 - episode/success_rate:0.5\n' for s in range(5, 9))
        (root / 'console.log').write_text(log)
        outcomes = [dict(reset_index=r, env_index=e, terminated=True, truncated=False,
                         game_result='Victory', policy_decision_count=1)
                    for r in range(4, 8) for e in range(8)]
        (root / 'episode_outcomes.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in outcomes))
        cp = root / 'checkpoints/global_step_8'
        (cp / 'actor').mkdir(parents=True)
        for name in ['data.pt', 'sc2_state.pt'] + [f'actor/{kind}_world_size_4_rank_{r}.pt'
                for kind in ('model', 'optim', 'extra_state') for r in range(4)]:
            (cp / name).write_bytes(b'test')
        result = finish(root, 0)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual((result['completed_steps'], result['start_step'], result['last_global_step']), (4, 4, 8))
        (root / 'console.log').write_text(log.replace('step:5', 'step:1'))
        self.assertEqual(finish(root, 0)['status'], 'failed')


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'requires the pinned CPU Torch runtime')
class TrainerResumeTest(unittest.TestCase):
    def test_actual_trainer_save_load_keeps_adam_scheduler_rng_and_environment(self):
        with patch.dict(os.environ, SC2_ORIGINAL_GAME_SEEDS='0'):
            self._check_trainer_resume([45891]*8)

    def test_original_seed_resume_preview_matches_actual_next_batch(self):
        with patch.dict(os.environ, SC2_ORIGINAL_GAME_SEEDS='1'):
            self._check_trainer_resume([60,71,12,45,55,40,78,81])

    def _check_trainer_resume(self, expected_next_seeds):
        from torchdata.stateful_dataloader import StatefulDataLoader
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            saved_actor = {}
            model = torch.nn.Linear(2, 1)
            optim = torch.optim.AdamW(model.parameters(), lr=1e-6)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda _: 1.0)
            for _ in range(4):
                optim.zero_grad()
                model(torch.ones(1, 2)).sum().backward()
                optim.step()
                scheduler.step()
            def save_actor(path, remote, step, **kwargs):
                Path(path).mkdir(parents=True)
                torch.save((model.state_dict(), optim.state_dict(), scheduler.state_dict()), Path(path)/'test.pt')
                saved_actor['step'] = step
            def load_actor(path, **kwargs):
                weights, state, sched = torch.load(Path(path)/'test.pt', weights_only=False)
                model.load_state_dict(weights)
                optim.load_state_dict(state)
                scheduler.load_state_dict(sched)
            actor = SimpleNamespace(save_checkpoint=save_actor, load_checkpoint=load_actor)
            config = SimpleNamespace(trainer=Config(default_local_dir=str(root), default_hdfs_dir=None,
                resume_mode='resume_path', resume_from_path=str(root/'global_step_4'),
                del_local_ckpt_after_load=False, sc2_checkpoint_resume=True, val_only=False))
            trainer = SimpleNamespace(config=config, global_steps=4, actor_rollout_wg=actor,
                use_critic=False, sc2_empty_policy_batches=2, envs=manager(), val_envs=manager(False),
                train_dataloader=StatefulDataLoader([0], batch_size=1, num_workers=0))
            seed_test = SeedResumeTest()
            seed_test.setUp()
            try:
                trainer.envs = seed_test.legacy()
            finally:
                seed_test.doCleanups()
            list(trainer.train_dataloader)
            save = method(ROOT/'verl/trainer/ppo/ray_trainer.py', 'RayPPOTrainer', '_save_checkpoint',
                          os=os, torch=torch, np=np, random=random)
            load = method(ROOT/'verl/trainer/ppo/ray_trainer.py', 'RayPPOTrainer', '_load_checkpoint',
                          os=os, torch=torch, np=np, random=random, json=json,
                          find_latest_ckpt_path=lambda _: None)
            before = {k:v.clone() for k,v in model.state_dict().items()}
            save(trainer)
            expected_rng = (random.random(), np.random.rand(), torch.rand(1))
            with torch.no_grad():
                model.weight.add_(10)
            optim.state.clear()
            scheduler.last_epoch = 0
            trainer.envs = manager()
            trainer.global_steps = 0
            trainer.sc2_empty_policy_batches = 0
            load(trainer)
            for k, value in model.state_dict().items():
                torch.testing.assert_close(value, before[k], rtol=0, atol=0)
            self.assertEqual({int(v['step']) for v in optim.state.values()}, {4})
            self.assertEqual(scheduler.last_epoch, 4)
            self.assertEqual((trainer.global_steps, trainer.envs.envs.reset_index,
                              trainer.envs._episode_reset_index, trainer.sc2_empty_policy_batches), (4, 4, 3, 2))
            self.assertEqual((random.random(), np.random.rand()), expected_rng[:2])
            torch.testing.assert_close(torch.rand(1), expected_rng[2], rtol=0, atol=0)
            evidence = json.loads((root/'resume_state.json').read_text())
            self.assertEqual(evidence['next_seeds'], expected_next_seeds)
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
