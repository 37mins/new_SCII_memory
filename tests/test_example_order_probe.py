"""The candidate only changes a displayed example, not todo or feedback."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/records/20260911-execution-failure-feedback-v1"))
from example_order_probe import candidate, MARKER, EXAMPLE_A, EXAMPLE_B
from fixed_feedback_compare import score


class ExampleOrderTest(unittest.TestCase):
    def setUp(self):
        self.prefix = "The remaining BO list in this phase:\n- marine x 30\n- supplydepot x 8\n- barracks x 3\nUse only exact canonical names\n"
        self.suffix = "\nExecution feedback: Supply blocked; Construct a SupplyDepot immediately.\n"
        self.prompt = self.prefix + MARKER + EXAMPLE_A + self.suffix

    def test_only_two_example_lines_swap(self):
        b = candidate(self.prompt)
        self.assertEqual(b, self.prefix + MARKER + EXAMPLE_B + self.suffix)
        self.assertEqual(sorted(b.splitlines()), sorted(self.prompt.splitlines()))
        self.assertEqual(b.replace(MARKER+EXAMPLE_B, MARKER+EXAMPLE_A), self.prompt)

    def test_validator_and_counts_unchanged(self):
        with patch.dict(os.environ, SC2_ORIGINAL_BO_SEGMENT="1"):
            for answer, depot_first in ((EXAMPLE_A, False), (EXAMPLE_B, True)):
                a, b = score(self.prompt, answer), score(candidate(self.prompt), answer)
                self.assertEqual(a, b)
                self.assertTrue(a["valid"])
                self.assertEqual(a["depot_first"], depot_first)
                self.assertEqual(sorted(a["parsed_queue"]), ["barracks"]*2+["marine"]*2+["supplydepot"]*2)

    def test_reject_unexpected_or_duplicate_example(self):
        for prompt in (self.prefix, self.prompt+self.prompt, self.prompt.replace("marine x 2", "marine x 3")):
            with self.assertRaises(ValueError):
                candidate(prompt)


if __name__ == "__main__":
    unittest.main()
