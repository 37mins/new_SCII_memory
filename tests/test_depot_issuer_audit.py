"""Offline order evidence must not silently become a command assignment."""
import copy
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'experiments/records/20260911-execution-failure-feedback-v1/audit_depot_issuer.py'
spec = importlib.util.spec_from_file_location('depot_issuer_audit', PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture(codes=(1,), unique=True, observed=True):
    ids = list(range(1, len(codes) + 1))
    order = dict(ability_id=319, progress=0.0, target_kind='target_world_space_pos',
                 target_position=[29, 23], target_tag=None)
    return [dict(event='local_command', event_id=i, ability='TERRANBUILD_SUPPLYDEPOT',
                 issuer_tag=10, origin={'selection_source': 'validated_answer'}) for i in ids] + [
        dict(event='wire_request', event_id=20, action_count=len(ids), groups=[
            dict(action_index=j, command_ids=[i], local_mapping_unique=unique) for j, i in enumerate(ids)]),
        dict(event='wire_result', event_id=21, request_event_id=20, result_codes=list(codes),
             error_type=None, results_aligned=True),
        dict(event='depot_started', event_id=22, game_loop=324, tag=200, position=[29, 23],
             command_id=None, issuer_observations=[dict(issuer_tag=10, command_ids=ids,
             observed=observed, position=[28, 23] if observed else None,
             orders=[order] if observed else None)]),
    ]


class IssuerAuditTest(unittest.TestCase):
    def test_unique_issuer_preserves_all_historical_commands_and_input(self):
        rows = fixture((1, 1)); before = copy.deepcopy(rows)
        report = audit.audit_rows(rows)
        self.assertEqual(report['historical_command_counts'], {2: 1})
        self.assertEqual(report['recorded_start_links'], 0)
        self.assertEqual(rows, before)

    def test_failed_or_ambiguous_ack_is_not_accepted(self):
        for codes, unique in [((2,), True), ((1,), False)]:
            report = audit.audit_rows(fixture(codes, unique))
            self.assertEqual(report['observed_issuer_counts'], {1: 1})
            self.assertEqual(report['accepted_commands'], 0)
            self.assertEqual(report['recorded_start_links'], 0)

    def test_absent_worker_is_not_a_builder(self):
        self.assertEqual(audit.audit_rows(fixture(observed=False))['observed_issuer_counts'], {0: 1})

    def test_target_presence_and_ability_are_required(self):
        start = dict(position=[0, 0], tag=200)
        order = dict(ability_id=319, target_kind=None, target_position=[0, 0], target_tag=200)
        self.assertFalse(audit.order_targets_start(order, start))
        order['target_kind'] = 'target_unit_tag'
        self.assertTrue(audit.order_targets_start(order, start))
        order['ability_id'] = 295
        self.assertFalse(audit.order_targets_start(order, start))


def completed_then_reused():
    rows = fixture()
    rows[0]['game_loop'] = 264
    done = dict(event='depot_completed', event_id=23, game_loop=804, tag=200,
                position=[29, 23], progress=1.0, duplicate=False)
    command = dict(rows[0], event_id=24, game_loop=980)
    request = dict(event='wire_request', event_id=25, action_count=1,
                   groups=[dict(action_index=0, command_ids=[24], local_mapping_unique=True)])
    ack = dict(event='wire_result', event_id=26, request_event_id=25, result_codes=[1],
               error_type=None, results_aligned=True)
    start = copy.deepcopy(rows[-1])
    start.update(event_id=27, game_loop=1040, tag=201, position=[25, 27])
    start['issuer_observations'][0]['command_ids'] = [1, 24]
    start['issuer_observations'][0]['orders'][0]['target_position'] = [25, 27]
    return rows + [done, command, request, ack, start]


class CompletedHistoryTest(unittest.TestCase):
    def test_candidate_full_history_is_checked_independently_of_recorded_link(self):
        rows = completed_then_reused()
        rows[3]['command_id'] = 1
        rows[-1]['command_id'] = 24
        before = copy.deepcopy(rows)
        report = audit.audit_rows(rows, full_issuer_history=True)
        findings = audit.audit_completed_history(rows, report['starts'])
        self.assertEqual([f['inferred_command_id'] for f in findings], [1, 24])
        self.assertEqual(report['recorded_start_links'], 2)
        self.assertEqual(rows, before)
        rows[-1]['command_id'] = 1
        report = audit.audit_rows(rows, full_issuer_history=True)
        self.assertEqual(audit.audit_completed_history(rows, report['starts'])[-1]['inferred_command_id'], 24)

    def findings(self, rows):
        return audit.audit_completed_history(rows, audit.audit_rows(rows)['starts'])

    def test_completed_old_command_is_excluded_without_changing_original_links(self):
        rows = completed_then_reused(); before = copy.deepcopy(rows)
        first, second = self.findings(rows)
        self.assertEqual(first['inferred_command_id'], 1)
        self.assertEqual(second['inferred_command_id'], 24)
        self.assertEqual(second['retired'][0]['completed_event_id'], 23)
        self.assertIsNone(second['recorded_command_id'])
        self.assertEqual(rows, before)

    def test_future_or_incomplete_completion_does_not_retire_history(self):
        for variant in ['future', 'incomplete', 'wrong_position']:
            rows = completed_then_reused()
            if variant == 'future':
                rows.append(rows.pop(4))
            elif variant == 'incomplete':
                rows[4]['progress'] = 0.5
            else:
                rows[4]['position'] = [99, 99]
            second = self.findings(rows)[1]
            self.assertEqual(second['status'], 'no_unique_pending_command')
            self.assertEqual(second['remaining_command_ids'], [1, 24])
            self.assertIsNone(second['inferred_command_id'])

    def test_two_unresolved_commands_do_not_choose_latest_or_fifo(self):
        finding = self.findings(fixture((1, 1)))[0]
        self.assertEqual(finding['remaining_command_ids'], [1, 2])
        self.assertIsNone(finding['inferred_command_id'])

    def test_two_issuers_targeting_same_building_stay_unknown(self):
        rows = fixture((1, 1))
        rows[1]['issuer_tag'] = 11
        first = rows[-1]['issuer_observations'][0]
        second = copy.deepcopy(first)
        first['command_ids'] = [1]
        second.update(issuer_tag=11, command_ids=[2])
        rows[-1]['issuer_observations'].append(second)
        self.assertEqual(self.findings(rows)[0]['status'], 'no_unique_observed_issuer')

    def test_missing_failed_or_future_ack_does_not_become_success(self):
        for variant in ['failed', 'unknown', 'future']:
            rows = fixture((2,) if variant == 'failed' else (1,), unique=variant != 'unknown')
            if variant == 'future':
                rows[-2], rows[-1] = rows[-1], rows[-2]
            self.assertEqual(self.findings(rows)[0]['status'], 'ack_not_confirmed')

    def test_command_and_building_cannot_be_claimed_twice(self):
        for same_tag in [True, False]:
            rows = fixture()
            another = copy.deepcopy(rows[-1]); another['event_id'] = 23
            if not same_tag:
                another['tag'] = 201
            rows.append(another)
            second = self.findings(rows)[1]
            self.assertIsNone(second['inferred_command_id'])
            self.assertEqual(second['status'], 'duplicate_start' if same_tag else
                             'command_already_associated_with_another_start')


if __name__ == '__main__':
    unittest.main()
