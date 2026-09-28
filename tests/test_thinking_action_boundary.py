"""Real manager boundary: one malformed sample must not kill eight games."""
import json
import os
from types import SimpleNamespace
import unittest

from test_r6_natural_termination import ROOT, method
from test_original_bo_segment import pipeline


class ThinkingBoundaryTest(unittest.TestCase):
    def boundary(self, responses, forbid=True):
        sent, audit = [], {}
        n = len(responses)
        step = method(ROOT/'agent_system/environments/strategic_sc2_env_manager_v2.py',
                      'StrategicSC2EnvironmentManagerV2', 'step', List=list, os=os, json=json,
                      to_numpy=lambda x: x, infer_failure_reason=lambda *args: args[-1])
        def env_step(actions):
            sent.extend(actions)
            return ['next']*n, [0.0]*n, [False]*n, [False]*n, [dict(terminated=False, truncated=False) for _ in responses]
        manager = SimpleNamespace(
            _forbid_thinking_output=forbid, _apply_verified_candidate_override=lambda x: x,
            _sanitize_final_actions=lambda x: (x, [False]*n),
            _last_prompts=['query']*n, _semantic_action_evaluable=lambda x: True,
            _validate_current_bo_response=lambda *args: (True, ''),
            projection_f=lambda x: (x, [bool(s.strip()) for s in x]), envs=SimpleNamespace(step=env_step),
            _terminal_by_env=[False]*n, _episode_step_by_env=[0]*n,
            config=SimpleNamespace(env=SimpleNamespace(max_steps=0)),
            _extract_failure_reason=lambda x: '', pre_text_obs=['query']*n,
            use_tool_call_memory=False, _no_progress_counts=[0]*n, memory=None,
            _write_prompt_response_log=lambda **kw: audit.update(kw),
            _write_tool_call_log=lambda **kw: None, _step_index=0,
            is_train=True, build_text_obs=lambda x: x)
        result = step(manager, responses)
        return sent, audit, result

    def test_one_offending_sample_rejected_without_mutating_input_or_other_games(self):
        for bad in ('<think>unfinished', '</think>- marine',
                    '<think>reason</think><answer>- marine</answer>'):
            original = ['- marine']*8
            original[0] = bad
            sent, audit, result = self.boundary(original)
            self.assertEqual(original[0], bad)
            self.assertEqual(sent[0], dict(text='', plan_format_valid=False))
            self.assertEqual(sent[1:], [dict(text='- marine', plan_format_valid=True)]*7)
            self.assertEqual(audit['text_actions'], original)
            self.assertFalse(audit['action_valids'][0])
            self.assertIn('thinking output forbidden', audit['failure_reasons'][0])
            self.assertEqual(result[2], [False]*8)

    def test_disabled_guard_is_unchanged(self):
        text = '<think>legacy</think>- marine'
        sent, audit, _ = self.boundary([text], forbid=False)
        self.assertEqual(sent[0]['text'], text)
        self.assertEqual(audit['text_actions'], [text])

    def test_regular_invalid_blank_and_valid_text_are_not_rewritten(self):
        sent, _, _ = self.boundary(['', '- marine', 'invalid'])
        self.assertEqual([a['text'] for a in sent], ['', '- marine', 'invalid'])

    def test_rejected_answer_uses_existing_invalid_answer_executor_path(self):
        sent, _, _ = self.boundary(['<think>- barracks</think>- marine'])
        # This is the existing blank-invalid repair path, NOT a new no-op or
        # expert plan. Normal raw sampled tokens remain in rollout/PPO upstream.
        self.assertEqual(pipeline(sent[0]['text'], False), pipeline('', False))


if __name__ == '__main__':
    unittest.main()
