"""CPU score contract: valid ordering is not game execution success."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/records/20260911-execution-failure-feedback-v1"))
from fixed_feedback_compare import score


class FixedFeedbackTest(unittest.TestCase):
    def test_order_not_just_format(self):
        prompt = "The remaining BO list in this phase:\n- marine x 29\n- supplydepot x 8\n- scv x 2\nUse only exact canonical names"
        with patch.dict(os.environ, SC2_ORIGINAL_BO_SEGMENT="1"):
            a = score(prompt, "<answer>\n- marine x 29\n- supplydepot x 8\n</answer>")
            self.assertTrue(a["valid"])
            self.assertFalse(a["depot_before_supply_consumer"])
            b = score(prompt, "<answer>\n- supplydepot\n- marine x 29\n</answer>")
            self.assertTrue(b["depot_first"])
            self.assertTrue(b["depot_before_supply_consumer"])
            c = score(prompt, "<answer>\n- supplydepot x 9\n- marine\n</answer>")
            self.assertFalse(c["valid"])
            self.assertFalse(c["depot_before_supply_consumer"])
            self.assertFalse(score(prompt, "<answer>\n- scv\n- supplydepot\n</answer>")["depot_before_supply_consumer"])


if __name__ == "__main__":
    unittest.main()
