"""The offline audit must not confuse geometric candidates with proven origins."""
import copy
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'experiments/records/20260911-execution-failure-feedback-v1/audit_depot_coordinates.py'
spec = importlib.util.spec_from_file_location('depot_coordinate_audit', PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture(positions=((28.5, 22.5),), codes=(1,), unique=True):
    commands = [dict(event='local_command', event_id=i + 1,
                     ability='TERRANBUILD_SUPPLYDEPOT', target_position=list(p),
                     issuer_tag=i + 10, origin={'policy_action_id': i + 1})
                for i, p in enumerate(positions)]
    return commands + [
        dict(event='wire_request', event_id=100, action_count=len(positions), groups=[
            dict(action_index=i, command_ids=[i + 1], local_mapping_unique=unique)
            for i in range(len(positions))]),
        dict(event='wire_result', event_id=101, request_event_id=100, result_codes=list(codes),
             error_type=None, results_aligned=len(codes) == len(positions)),
        dict(event='depot_started', event_id=102, game_loop=324, tag=200, position=[29, 23],
             candidate_command_ids=[], command_id=None),
    ]


class CoordinateAuditTest(unittest.TestCase):
    def test_real_half_integer_example_is_hypothesis_not_exact_link(self):
        rows = fixture()
        original = copy.deepcopy(rows)
        result = audit.audit_rows(rows)[0]['hypotheses']
        self.assertIsNone(result['exact']['unique_acked_candidate_id'])
        self.assertIsNone(result['sdk_floor']['unique_acked_candidate_id'])
        self.assertEqual(result['half_up_hypothesis']['unique_acked_candidate_id'], 1)
        self.assertEqual(result['ceil_hypothesis']['unique_acked_candidate_id'], 1)
        self.assertEqual(rows, original)

    def test_duplicate_normalized_positions_remain_ambiguous(self):
        result = audit.audit_rows(fixture(((28.5, 22.5), (28.5, 22.5)), (1, 1)))[0]
        candidate = result['hypotheses']['half_up_hypothesis']
        self.assertEqual(candidate['candidate_ids'], [1, 2])
        self.assertIsNone(candidate['unique_acked_candidate_id'])

    def test_rejected_missing_and_ambiguous_ack_are_not_success(self):
        for codes, unique in [((2,), True), ((), True), ((1,), False)]:
            with self.subTest(codes=codes, unique=unique):
                result = audit.audit_rows(fixture(codes=codes, unique=unique))[0]
                self.assertIsNone(result['hypotheses']['half_up_hypothesis']['unique_acked_candidate_id'])

    def test_future_ack_cannot_link_an_earlier_start(self):
        rows = fixture()
        rows[-2], rows[-1] = rows[-1], rows[-2]
        rows[-1]['event_id'] = 103
        result = audit.audit_rows(rows)[0]
        self.assertIsNone(result['hypotheses']['half_up_hypothesis']['unique_acked_candidate_id'])

    def test_non_half_input_distinguishes_unverified_hypotheses(self):
        # No such command was observed in r1; this is not an engine test.
        self.assertEqual(audit.RULES['half_up_hypothesis'](28.25), 28)
        self.assertEqual(audit.RULES['ceil_hypothesis'](28.25), 29)


if __name__ == '__main__':
    unittest.main()
