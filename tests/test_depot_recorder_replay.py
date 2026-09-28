"""Run the candidate recorder itself on lifecycle/ambiguity counterexamples."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

from agent_system.sc2_command_trace import CommandTrace
from test_depot_issuer_audit import fixture, completed_then_reused

PATH = Path(__file__).resolve().parents[1] / 'experiments/records/20260911-execution-failure-feedback-v1/replay_depot_recorder.py'
spec = importlib.util.spec_from_file_location('depot_recorder_replay', PATH)
replayer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replayer)


class RecorderReplayTest(unittest.TestCase):
    def replay_starts(self, rows):
        before = copy.deepcopy(rows)
        with tempfile.TemporaryDirectory() as directory:
            replayed = replayer.replay(rows, CommandTrace, directory)
        self.assertEqual(rows, before)
        return [r for r in replayed if r['event'] in ('depot_started', 'duplicate_construction_callback')]

    def test_completed_history_reuses_worker_without_reusing_command(self):
        first, second = self.replay_starts(completed_then_reused())
        self.assertEqual([first['command_id'], second['command_id']], [1, 24])
        self.assertEqual(second['candidate_command_ids'], [24])
        self.assertEqual(second['issuer_observations'][0]['command_ids'], [1, 24])

    def test_future_incomplete_or_wrong_completion_keeps_ambiguity(self):
        for variant in ['future', 'incomplete', 'wrong_position', 'wrong_tag']:
            rows = completed_then_reused()
            if variant == 'future':
                rows.append(rows.pop(4))
            elif variant == 'incomplete':
                rows[4]['progress'] = .5
            elif variant == 'wrong_position':
                rows[4]['position'] = [99, 99]
            else:
                rows[4]['tag'] = 999
            with self.subTest(variant=variant):
                second = self.replay_starts(rows)[1]
                self.assertIsNone(second['command_id'])
                self.assertEqual(second['candidate_command_ids'], [1, 24])

    def test_missing_failed_or_future_ack_never_links(self):
        for variant in ['failed', 'unknown', 'future']:
            rows = fixture((2,) if variant == 'failed' else (1,), unique=variant != 'unknown')
            if variant == 'future':
                rows[-2], rows[-1] = rows[-1], rows[-2]
            with self.subTest(variant=variant):
                self.assertIsNone(self.replay_starts(rows)[0]['command_id'])

    def test_two_pending_commands_or_two_issuers_never_choose_a_winner(self):
        rows = fixture((1, 1))
        self.assertIsNone(self.replay_starts(rows)[0]['command_id'])
        rows[1]['issuer_tag'] = 11
        first = rows[-1]['issuer_observations'][0]
        second = copy.deepcopy(first)
        first['command_ids'] = [1]
        second.update(issuer_tag=11, command_ids=[2])
        rows[-1]['issuer_observations'].append(second)
        self.assertIsNone(self.replay_starts(rows)[0]['command_id'])

    def test_duplicate_start_or_same_command_second_building_cannot_double_link(self):
        for same_tag in [True, False]:
            rows = fixture()
            another = copy.deepcopy(rows[-1]); another['event_id'] = 23
            if not same_tag:
                another['tag'] = 201
            rows.append(another)
            second = self.replay_starts(rows)[1]
            if same_tag:
                self.assertEqual(second['event'], 'duplicate_construction_callback')
            else:
                self.assertIsNone(second['command_id'])

    def test_absent_worker_does_not_link(self):
        self.assertIsNone(self.replay_starts(fixture(observed=False))[0]['command_id'])


if __name__ == '__main__':
    unittest.main()
