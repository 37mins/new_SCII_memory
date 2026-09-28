import unittest

from audit_b4_training_signal import depot_choice, episode_component_bounds


class FeedbackSignalTest(unittest.TestCase):
    def test_depot_only_is_a_valid_ordering_choice(self):
        result = depot_choice('<answer>\n- supplydepot x 1\n- barracks x 1\n</answer>')
        self.assertTrue(result['depot_is_first_bullet'])
        self.assertEqual(result['order'], 'not_comparable')

    def test_marine_then_depot_is_not_depot_first(self):
        self.assertFalse(depot_choice('- marine x 2\n- supplydepot')['depot_is_first_bullet'])

    def test_equal_returns_and_valid_rows(self):
        episodes = [dict(reward=20.5, decisions=2, answers=[dict(valid=True)]*2)]*8
        self.assertEqual(episode_component_bounds(episodes, 0, 3), [0, 0])

    def test_other_invalid_representative_does_not_imply_task_success(self):
        episodes = [dict(reward=20.5, decisions=2, answers=[dict(valid=True)]*2) for _ in range(8)]
        episodes[1]['answers'][0] = dict(valid=False)
        low, high = episode_component_bounds(episodes, 0, 0)
        self.assertEqual(low, 0)
        self.assertAlmostEqual(high, .1/8/2)


if __name__ == '__main__':
    unittest.main()
