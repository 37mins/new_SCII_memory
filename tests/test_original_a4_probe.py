"""Reference comparison keeps incomplete-but-accepted answers separate."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/records/20260911-execution-failure-feedback-v1"))
from original_a4_probe import compare


class OriginalA4ProbeTest(unittest.TestCase):
    def test_paired_seeds_and_complete_metric(self):
        rows=[dict(seed=s, valid=True, depot_first=True, depot_before_supply_consumer=True, eos_seen=True,
                   raw_answer="<answer>\n- supplydepot\n</answer>", token_count=12, reason="")
              for s in range(12000,12008)]
        old={r['seed']:dict(r) for r in rows}
        old[12000].update(eos_seen=False, token_count=192, raw_answer="<answer>\n- supplydepot")
        result=compare(rows, dict(run_id="existing_A4", answers=old))
        self.assertEqual(result['newly_generated_answers'],8)
        self.assertEqual(result['reused_answers'],8)
        self.assertTrue(result['pairs'][0]['A4']['valid'])
        self.assertFalse(result['pairs'][0]['A4']['complete_correction'])
        self.assertTrue(result['pairs'][0]['original']['complete_correction'])
        rows[1]['depot_first']=False
        self.assertTrue(compare(rows, dict(run_id="existing_A4", answers=old))['pairs'][1]['original']['complete_correction'])
        with self.assertRaises(AssertionError):
            compare(rows[:-1], dict(run_id="existing_A4",answers=old))


if __name__ == '__main__':
    unittest.main()
