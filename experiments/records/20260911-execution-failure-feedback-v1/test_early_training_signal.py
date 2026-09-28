import unittest

from audit_early_training_signal import first_phase_answer, scalar_advantages, PHASE


class EarlySignalTest(unittest.TestCase):
    def test_stage_not_call_number(self):
        episode = {'answers': [{'source_line': i} for i in (1, 4, 9)]}
        raw = {1: {'prompt': 'Opening'}, 4: {'prompt': 'Opening'},
               9: {'prompt': PHASE + '(03:30-06:00)'}}
        self.assertEqual(first_phase_answer(episode, raw), 2)

    def test_equal_returns_and_shared_state_have_zero_signal(self):
        rows = [dict(env=i, valid=True, episode_reward=20.5, step_r=20.5,
                     prompt_sha256='shared') for i in range(2)]
        self.assertEqual(scalar_advantages(rows), [0, 0])

    def test_copy_changes_step_mean_but_not_episode_baseline(self):
        rows = [dict(env=i, valid=True, episode_reward=r, step_r=r,
                     prompt_sha256='shared') for i, r in enumerate((0, 6))]
        # Episode mean stays 3; copied env0's episode weight is divided by 2.
        self.assertEqual(scalar_advantages(rows, [0]), [-3.5, 7, -3.5])

    def test_invalid_row_cannot_use_all_valid_shortcut(self):
        with self.assertRaises(AssertionError):
            scalar_advantages([dict(valid=False)])


if __name__ == '__main__':
    unittest.main()
