"""Small CPU checks of the log adapter; production reward/GiGPO is reused."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
    'experiments/records/20260914-zero-task-signal-sampling-v1'))
from audit_scene_signal import reconstruct, decompose, effective_config
from test_sc2_task_signal import config


def fixture(totals, calls=None):
    calls = calls or [2]*8
    rows, ends = [], []
    for env, (total, n) in enumerate(zip(totals, calls)):
        for call in range(n):
            rows.append(dict(env_idx=env, step_index=call,
                reward=total if call == n-1 else 0., done=call == n-1,
                policy_action_applied=True, is_action_valid=True,
                memory_context='', prompt=f'input{call}', source_line=len(rows)+1))
        ends.append(dict(env_index=env, step_index=n-1, policy_decision_count=n,
            terminated=True, truncated=False, game_result='Victory' if total>0 else 'Defeat'))
    return rows, ends


class SceneSignalAuditTest(unittest.TestCase):
    def test_hydra_off_string_and_last_override(self):
        cfg = effective_config({'+env.sc2_memory_mode': 'off',
            '+env.prompt_response_log_max_prompt_chars': '20000',
            '++env.prompt_response_log_max_prompt_chars': '0'})
        self.assertEqual(cfg.env.sc2_memory_mode, 'off')
        self.assertEqual(cfg.env.prompt_response_log_max_prompt_chars, 0)

    def parts(self, rows, ends, width=1):
        batch, _, _ = reconstruct(rows, ends, width)
        return decompose(batch, config('/unused-cpu-audit'))

    def test_equal_returns_equal_lengths_zero_and_width_invariant(self):
        for total in (20.5, -50.):
            rows, ends = fixture([total]*8)
            a, b = self.parts(rows, ends), self.parts(rows, ends, 7)
            self.assertEqual(a['joint'], [0.]*16)
            self.assertEqual(a['joint'], b['joint'])

    def test_mixed_outcomes_produce_both_signs(self):
        rows, ends = fixture([20.5]*7+[-50.])
        a = self.parts(rows, ends)
        self.assertLess(min(a['joint']), 0)
        self.assertGreater(max(a['joint']), 0)

    def test_equal_wins_different_lengths_have_only_step_component(self):
        rows, ends = fixture([20.5]*8, [2]*7+[3])
        a = self.parts(rows, ends)
        self.assertEqual(a['episode'], [0.]*17)
        self.assertGreater(max(map(abs, a['step'])), 0)

    def test_terminal_poll_credited_once_and_inactive_rows_excluded(self):
        rows, ends = fixture([20.5]*8)
        rows[1].update(reward=.5, done=False)
        rows.extend([dict(rows[1], step_index=2, reward=20., done=True,
                          policy_action_applied=False),
                     dict(rows[1], step_index=3, reward=999., done=True)])
        ends[0]['step_index'] = 2
        batch, episodes, selected = reconstruct(rows, ends)
        self.assertEqual(episodes[0]['reward'], 20.5)
        self.assertEqual(episodes[0]['terminal_polls'], 1)
        self.assertEqual(selected[1]['rewards'], 20.5)
        self.assertEqual(len(batch), 16)


if __name__ == '__main__':
    unittest.main()
