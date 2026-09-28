import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'experiments/records/20260911-execution-failure-feedback-v1/audit_online_correction.py'
spec = importlib.util.spec_from_file_location('online_correction_audit', PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class OnlineCorrectionAuditTest(unittest.TestCase):
    def test_end_with_incomplete_building_is_not_reported_as_command_failure(self):
        self.assertEqual(audit.end_evidence([{'tag': 10}], [], [{'tag': 10, 'progress': .825}], True),
                         'unfinished_at_natural_game_end')

    def test_missing_building_or_nonterminal_has_insufficient_completion_evidence(self):
        for units, terminal in [([], True), ([{'tag': 11, 'progress': .8}], True),
                                ([{'tag': 10, 'progress': .8}], False),
                                ([{'tag': 10, 'progress': 1.}], True)]:
            self.assertEqual(audit.end_evidence([{'tag': 10}], [], units, terminal),
                             'start_only_completion_not_established')

    def test_no_start_is_distinct_from_confirmed_completion(self):
        self.assertEqual(audit.end_evidence([], [], [], True), 'no_attributed_depot_start')
        self.assertEqual(audit.end_evidence([{'tag': 10}], [{'tag': 10}], [], True),
                         'completed_link_present')

    def test_depot_only_answer_is_a_depot_choice_not_completed_execution(self):
        self.assertEqual(audit.first_bullet('<answer>\n- supplydepot x 1\n</answer>'), '- supplydepot x 1')


if __name__ == '__main__':
    unittest.main()
