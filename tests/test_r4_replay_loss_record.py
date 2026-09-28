"""Checks on the archived replay evidence, not a gameplay/fix acceptance test."""
from collections import Counter
import json
from pathlib import Path
import unittest


RECORD = Path(__file__).resolve().parents[1] / 'experiments/records/20260906-integrated-fixes-v1'


class ReplayLossRecordTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((RECORD / 'r4-preplanner-replay-losses.json').read_text())

    def test_frozen_input_identity(self):
        manifest = json.loads((RECORD / 'r4-python-rng-preplanner.json').read_text())
        for item, expected in zip(self.data['inputs'], manifest['reports']):
            self.assertEqual(item['run_id'], expected['run_id'])
            sha = next(v for k, v in expected['artifact_hashes'].items() if k.endswith('.SC2Replay'))
            self.assertEqual(item['replay_sha'], sha)
            self.assertEqual(item['base_build'], 75689)
        first, second = self.data['inputs']
        self.assertNotEqual(first['replay_sha'], second['replay_sha'])
        for stream in ('tracker', 'message'):
            self.assertEqual(first['stream_sha'][stream], second['stream_sha'][stream])

    def test_birth_death_identity_and_killers(self):
        for unit_type, total in [('Marine', 74), ('SCV', 41)]:
            births = {tuple(b['tag']) for b in self.data['births']
                      if b['owner'] == 1 and b['type'] == unit_type}
            deaths = [d for d in self.data['deaths'] if d['owner'] == 1 and d['type'] == unit_type]
            self.assertEqual(len(births), total)
            self.assertEqual(len(deaths), total)
            self.assertEqual({tuple(d['tag']) for d in deaths}, births)
            self.assertTrue(all(d['killer_player'] == 2 for d in deaths))
            self.assertEqual(dict(Counter(d['killer_type'] for d in deaths)),
                             self.data['killers_by_victim_type'][unit_type])
        self.assertEqual(self.data['killers_by_victim_type']['SCV'],
                         {'Marine': 1, 'VikingAssault': 34, 'Marauder': 6})

    def test_late_workers_and_reinforcements(self):
        workers = [d for d in self.data['deaths']
                   if d['owner'] == 1 and d['type'] == 'SCV' and d['second'] > 500]
        self.assertEqual(len(workers), 40)
        self.assertAlmostEqual(workers[-1]['second'] - workers[0]['second'], 26.026, places=3)
        marines = self.data['late_marine_lifetimes']
        self.assertEqual(len(marines), 21)
        for item in marines:
            self.assertAlmostEqual(item['life_seconds'],
                                   (item['death_loop'] - item['born_loop']) / 22.4)
            self.assertGreater(item['life_seconds'], 0)
            self.assertLess(item['life_seconds'], 13)

    def test_three_techlab_attempts_same_factory(self):
        for frame, bo in [('2521', 123), ('2527', 129), ('2528', 130)]:
            evidence = self.data['factory_trace'][frame]
            self.assertEqual(evidence[f'llm_decision_final_action_{bo}_{frame}'],
                             [{'action': 'BUILD_TECHLAB_FACTORY', 'units': [266]}])
            self.assertIn('[266]Factory', evidence['obs']['Own structures'])
        starts = self.data['own_constructed_starts']
        self.assertEqual(starts['Factory'], 3)
        self.assertEqual(starts.get('FactoryTechLab', 0), 0)
        self.assertEqual(self.data['own_born'].get('SiegeTank', 0), 0)

    def test_factory_depot_order_precedes_factory_initialization(self):
        evidence = json.loads((RECORD / 'r4-factory-techlab-evidence.json').read_text())
        for run in evidence['reports']:
            self.assertLess(run['factory_command']['loop'], run['depot_command']['loop'])
            self.assertLess(run['depot_command']['loop'], run['factory_events'][0]['_gameloop'])
            self.assertNotIn('Factory', run['pre_init_observation']['Own structures'])
            self.assertIn('State: constructing [(26, 30)]', run['pre_init_observation']['Own units'])
            self.assertEqual(run['addon_center'], [29, 30])
            self.assertEqual(run['depot_center'], [29, 31])
            self.assertEqual(run['overlap_size'], [2, 1])

    def test_all_attempts_select_same_real_factory(self):
        evidence = json.loads((RECORD / 'r4-factory-techlab-evidence.json').read_text())
        for run in evidence['reports']:
            self.assertEqual([x['bo'] for x in run['attempts']], [123, 129, 130])
            for attempt in run['attempts']:
                self.assertEqual(attempt['replay_selection']['m_delta']['m_addUnitTags'],
                                 [(347 << 18) | 1])

    def test_cpu_probe_is_not_engine_evidence(self):
        evidence = json.loads((RECORD / 'r4-factory-techlab-cpu.json').read_text())
        self.assertEqual(evidence['exit_code'], 0)
        self.assertEqual(evidence['burnysc2'], '7.1.3')
        self.assertIn('not an SC2 engine placement response', evidence['limitation'])
        before, after = evidence['results']
        self.assertTrue(before['has_pending_factory_order'])
        self.assertTrue(after['has_pending_factory_order'])
        self.assertEqual(before['returned'], [28.5, 30.5])
        self.assertIsNone(after['returned'])


if __name__ == '__main__':
    unittest.main()
