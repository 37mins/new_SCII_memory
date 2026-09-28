"""Scalar audit checks only; these do not change or test a GPU training path."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/records/20260916-combat-result-reward-v1'))
from audit_temperature_credit import score_rows


def episode(env, rewards, anchors, valid=None):
    valid = valid or [True]*len(rewards)
    return dict(env=env, reward=sum(rewards), decisions=len(rewards), answers=[
        dict(credited_reward=r, prompt_sha256=a, valid=v)
        for r, a, v in zip(rewards, anchors, valid)])


class TemperatureCreditAuditTest(unittest.TestCase):
    def test_identical_trajectories_have_zero_joint_signal(self):
        rows, _ = score_rows([episode(i, [0, -50], ['start', 'second']) for i in range(8)], .95)
        self.assertTrue(all(abs(r['joint_before_copy']) < 1e-12 for r in rows))

    def test_more_calls_can_make_a_worse_defeat_positive(self):
        episodes = [episode(0, [0, -50], ['start', 'second']),
                    episode(1, [0, 0, 0, -51], ['start', 'second', 'third', 'last'])]
        discounted, _ = score_rows(episodes, .95)
        undiscounted, _ = score_rows(episodes, 1.)
        selected = next(r for r in discounted if r['env']==1 and r['call']==2)
        control = next(r for r in undiscounted if r['env']==1 and r['call']==2)
        self.assertLess(selected['episode_component_before_copy'], 0)
        self.assertGreater(selected['joint_before_copy'], 0)
        self.assertLess(control['joint_before_copy'], 0)

    def test_invalid_penalty_and_singleton_anchor(self):
        episodes = [episode(0, [0, -50], ['start', 'a']),
                    episode(1, [0, -50], ['start', 'b'], [True, False])]
        rows, _ = score_rows(episodes, .95)
        invalid = rows[-1]
        self.assertAlmostEqual(invalid['step_score'], -50.1)
        self.assertEqual(invalid['step_component_before_copy'], 0)
        self.assertEqual(invalid['joint_before_copy'], 0)


if __name__ == '__main__':
    unittest.main()
