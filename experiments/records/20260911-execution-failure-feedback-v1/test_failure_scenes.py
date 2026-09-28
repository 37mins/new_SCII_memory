"""Boundary tests for the offline screen, not new SC2/RL acceptance."""
import unittest

from audit_failure_scenes import classify, consumed


class FailureSceneTest(unittest.TestCase):
    def test_only_consumed_decisions_before_real_terminal(self):
        rows = [dict(env_idx=e, step_index=s, policy_action_applied=a)
                for e, s, a in [(0, 0, True), (1, 0, True), (0, 1, False),
                                (0, 2, True), (0, 3, True)]]
        result = consumed(rows, dict(env_index=0, step_index=2))
        self.assertEqual([line for line, _ in result], [1, 4])

    def test_failure_categories_are_literal_not_inferred_from_outcome(self):
        self.assertEqual(classify("timeout: Supply blocked"), "supply_blocked")
        self.assertEqual(classify("Insufficient Vespene"), "vespene")
        self.assertEqual(classify("Tech Lab required"), "other")


if __name__ == "__main__":
    unittest.main()
