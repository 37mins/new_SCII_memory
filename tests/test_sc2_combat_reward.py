"""CPU counter, attribution and actual Planner transport tests; no GPU/SC2."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent_system.sc2_combat_reward import CombatResultReward, combat_values
from test_r6_natural_termination import ARENA, method


def score(killed=0, lost=0):
    fields = {f'{kind}_{resource}_{category}': 0 for kind in ('killed', 'lost')
              for resource in ('minerals', 'vespene')
              for category in ('army', 'economy', 'technology')}
    fields.update(killed_minerals_army=killed, lost_minerals_army=lost)
    return SimpleNamespace(**fields)


class CombatRewardTest(unittest.TestCase):
    def test_engine_categories_include_workers_buildings_and_gas_symmetrically(self):
        state = score(50, 25)
        state.killed_vespene_technology = 100
        state.lost_minerals_economy = 75
        self.assertEqual(combat_values(state), (150, 100))

    def test_fixed_prefix_no_policy_and_duplicate_observation_get_no_bonus(self):
        ledger = CombatResultReward(.001)
        self.assertEqual(ledger.consume(1000, 100, False), 0)
        self.assertEqual(ledger.consume(1100, 150, True), .05)
        self.assertEqual(ledger.consume(1100, 150, True), 0)
        self.assertEqual(CombatResultReward(.001).consume(2000, 1000, False), 0)

    def test_delta_telescopes_and_call_splitting_does_not_create_reward(self):
        one, many = CombatResultReward(.001), CombatResultReward(.001)
        one.consume(0, 0, False); many.consume(0, 0, False)
        whole = one.consume(2000, 1800, True)
        pieces = [many.consume(1000, 400, True), many.consume(1500, 1700, True),
                  many.consume(2000, 1800, True)]
        self.assertAlmostEqual(whole, sum(pieces))
        self.assertEqual(many.consume(2000, 1800, True), 0)

    def test_self_loss_is_negative_and_passage_of_time_has_no_reward(self):
        ledger = CombatResultReward(.001)
        ledger.consume(0, 0, False)
        self.assertEqual(ledger.consume(0, 200, True), -.2)
        self.assertEqual(ledger.consume(0, 200, True), 0)

    def test_identical_trajectories_stay_equal_and_off_is_zero(self):
        results = []
        for scale in (.001, .001, 0):
            ledger = CombatResultReward(scale)
            ledger.consume(0, 0, False)
            results.append(ledger.consume(2475, 6450, True))
        self.assertEqual(results, [-3.975, -3.975, 0])

    def test_invalid_or_reset_counters_fail_instead_of_manufacturing_reward(self):
        for value in (float('nan'), float('inf'), -1):
            with self.assertRaises(ValueError):
                CombatResultReward(value)
        ledger = CombatResultReward(.001)
        ledger.consume(100, 100, False)
        with self.assertRaises(ValueError):
            ledger.consume(0, 100, True)

    def test_real_get_action_and_terminal_attach_once_without_changing_query(self):
        get_action = method(ARENA/'agents/planner.py', 'SC2Planner', 'get_action',
                            os=os, final_state_shaping_reward=lambda obs: 0)
        game_end = method(ARENA/'agents/planner.py', 'SC2Planner', 'game_end',
                          final_state_shaping_reward=lambda obs: .1)
        state = score()
        payloads = []
        planner = SimpleNamespace(combat_result_reward=CombatResultReward(.001),
            combat_score=lambda: (state, 1), last_consumed_policy_action_id=None,
            correct_format=False, exceed=False, _send=payloads.append,
            _recv=lambda: dict(msg='ACTION', value='- marine x 42', policy_action_id=1,
                               plan_format_valid=True), _wait_terminal_ack=lambda data: None)
        with patch.dict(os.environ, SC2_STEP_SHAPING_MODE='off'), contextlib.redirect_stdout(io.StringIO()):
            get_action(planner, 'unchanged query')
            state.killed_minerals_army = 200
            get_action(planner, 'unchanged query')
            state.lost_minerals_army = 50
            game_end(planner, 'terminal observation', 'Defeat')
        for actual, expected in zip(payloads, (0, .2, -49.45)):
            self.assertAlmostEqual(actual['reward'], expected)
        self.assertEqual([p['last_consumed_policy_action_id'] for p in payloads], [None, 1, 1])
        self.assertEqual(payloads[0]['obs'], payloads[1]['obs'])
        self.assertTrue(payloads[-1]['done'])

    def test_only_authorized_transport_lines_changed_in_planner(self):
        path = 'agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/agents/planner.py'
        root = ARENA.parents[4]
        old = subprocess.check_output(['git', 'show', 'e8cc6c9:' + path], cwd=root, text=True)
        current = (ARENA/'agents/planner.py').read_text()
        current = current.replace('        from agent_system.sc2_combat_reward import consume_combat_reward\n', '')
        for boundary in ('terminal', 'planner'):
            current = current.replace(f'        reward += consume_combat_reward(self, "{boundary}")\n', '')
        self.assertEqual(current, old)


if __name__ == '__main__':
    unittest.main()
