"""CPU-only observer equivalence; no model or SC2 initialization."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'experiments/records/20260906-integrated-fixes-v1/replay_r4_first_update.py'
spec = importlib.util.spec_from_file_location('r4_replay', PATH)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
analysis_path = PATH.with_name('analyze_r4_first_update.py')
analysis_spec = importlib.util.spec_from_file_location('r4_analysis', analysis_path)
analysis = importlib.util.module_from_spec(analysis_spec)
analysis_spec.loader.exec_module(analysis)


class ReplayTest(unittest.TestCase):
    def test_observation_preserves_update_and_adam_history(self):
        def run(observe, root):
            p = torch.nn.Parameter(torch.linspace(-1., 1., 31))
            opt = torch.optim.AdamW([p], lr=1e-6, betas=(.9, .999), eps=1e-8, weight_decay=.01)
            for _ in range(22):
                p.square().mean().backward()
                opt.step()
                opt.zero_grad()

            def forward(*args, **kwargs):
                return p.square(), -torch.nn.functional.softplus(p * 2.)

            def step():
                norm = torch.nn.utils.clip_grad_norm_([p], 1.)
                opt.step()
                return norm

            actor = SimpleNamespace(actor_optimizer=opt, _optimizer_step=step, _forward_micro_batch=forward)
            finish = replay.observe_one_step(actor, root, 0) if observe else None
            entropy, logp = actor._forward_micro_batch()
            (-logp.mean() - .001 * entropy.mean()).backward()
            norm = actor._optimizer_step()
            opt.zero_grad()
            evidence = finish() if finish else None
            return p.detach(), opt.state[p], norm, evidence

        with tempfile.TemporaryDirectory() as temp:
            a, sa, na, _ = run(False, Path(temp))
            b, sb, nb, evidence = run(True, Path(temp))
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(na, nb)
        for key in sa:
            self.assertTrue(torch.equal(sa[key], sb[key]))
        self.assertEqual((evidence['before']['step'], evidence['after']['step']), (22, 23))
        self.assertEqual(evidence['observed_optimizer_steps'], 1)
        self.assertTrue(torch.equal(evidence['parameter_delta'], b - evidence['before']['parameter']))
        self.assertTrue(torch.allclose(evidence['loss_to_logprob'], torch.full((31,), -1 / 31)))
        group = evidence['group']
        m = evidence['before']['exp_avg'].clone().lerp_(evidence['gradient_after_global_clip'], 1 - group['betas'][0])
        v = evidence['before']['exp_avg_sq'] * group['betas'][1]
        v.addcmul_(evidence['gradient_after_global_clip'], evidence['gradient_after_global_clip'], value=1 - group['betas'][1])
        self.assertTrue(torch.equal(m, evidence['after']['exp_avg']))
        self.assertTrue(torch.equal(v, evidence['after']['exp_avg_sq']))
        audit = analysis.audit_adam(evidence)
        for key in ('parameter', 'parameter_delta', 'exp_avg', 'exp_avg_sq'):
            self.assertEqual(audit[key]['abs_max'], 0.)

    def test_block_selection_is_deterministic_and_bounded(self):
        for size in (1, 31, 502808512):
            index = replay.block_indices(size)
            self.assertTrue(torch.equal(index, replay.block_indices(size)))
            self.assertEqual(int(index[0]), 0)
            self.assertEqual(int(index[-1]), size - 1)
            self.assertLessEqual(len(index), 65536)
            self.assertEqual(len(index), len(torch.unique(index)))

    def test_summary_excludes_padding_and_separates_weighting(self):
        mask = torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.bool)
        old = torch.zeros(2, 3)
        new = torch.tensor([[-4., 100., 100.], [-1., -1., -1.]])
        row = replay.probability_summary(new, old, mask)
        self.assertEqual(row['tokens'], 4)
        self.assertEqual(row['k1_token_mean'], 1.75)
        self.assertEqual(row['k1_response_mean'], 2.5)
        self.assertEqual(row['abs_max'], 4.)

    def test_restore_comparison_covers_last_response_not_only_first_mini(self):
        mask = torch.tensor([[1, 0]] * 23, dtype=torch.bool)
        old = torch.zeros(23, 2)
        restored = old.clone()
        restored[:, 1] = float('nan')
        row = replay.probability_summary(restored, old, mask)
        self.assertTrue(row['finite'])
        self.assertEqual(row['abs_max'], 0.)
        restored[-1, 0] = .125
        self.assertEqual(replay.probability_summary(restored, old, mask)['abs_max'], .125)
        restored[-1, 0] = float('nan')
        self.assertFalse(replay.probability_summary(restored, old, mask)['finite'])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
