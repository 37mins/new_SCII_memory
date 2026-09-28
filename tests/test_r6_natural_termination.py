import ast
import contextlib
import io
import itertools
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_system.sc2_episode import (
    apply_external_truncation,
    gymnasium_transition,
    interaction_limit_reached,
    interaction_steps,
    outcome_fields,
    truncation_fields,
)


ROOT = Path(__file__).resolve().parents[1]
ARENA = ROOT / "agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy"


def method(path, class_name, name, **scope):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    node = next(
        node for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


class R6NaturalTerminationTest(unittest.TestCase):
    def test_zero_disables_the_optional_interaction_limit(self):
        self.assertEqual(list(interaction_steps(3)), [0, 1, 2])
        self.assertEqual(list(itertools.islice(interaction_steps(0), 4)), [0, 1, 2, 3])
        self.assertFalse(interaction_limit_reached(0, 200))
        self.assertFalse(interaction_limit_reached(20, 19))
        self.assertTrue(interaction_limit_reached(20, 20))

    def test_only_real_results_have_outcome_rewards(self):
        self.assertEqual(outcome_fields("Victory")["outcome_reward"], 20)
        self.assertEqual(outcome_fields("Defeat")["outcome_reward"], -50)
        self.assertEqual(outcome_fields("Tie")["outcome_reward"], -50)
        cutoff = truncation_fields()
        self.assertEqual(cutoff["outcome_reward"], 0)
        self.assertIsNone(cutoff["game_result"])
        self.assertFalse(cutoff["terminated"])
        self.assertTrue(cutoff["truncated"])

    def test_real_result_wins_at_interaction_boundary(self):
        victory = outcome_fields("Victory")
        terminated, truncated, info = apply_external_truncation(True, False, victory, True)
        self.assertEqual((terminated, truncated, info), (True, False, victory))
        terminated, truncated, info = apply_external_truncation(
            False, False, {"terminated": False, "truncated": False}, True
        )
        self.assertEqual((terminated, truncated), (False, True))
        self.assertEqual(info["termination_reason"], "interaction_limit")

    def test_planner_never_turns_call_count_into_done(self):
        get_action = method(
            ARENA / "agents/planner.py",
            "SC2Planner",
            "get_action",
            os=os,
            DEFEAT_PENALTY=-50,
            final_state_shaping_reward=lambda obs: 0,
        )
        for call_count in (20, 200):
            sent = []
            planner = SimpleNamespace(
                ct=call_count,
                correct_format=False,
                exceed=False,
                last_consumed_policy_action_id=None,
                _send=sent.append,
                _recv=lambda: {"msg": "ACTION", "value": "- scv", "policy_action_id": 1},
            )
            self.assertEqual(get_action(planner, "obs", reward=0.25), "- scv")
            self.assertFalse(sent[0]["done"])
            self.assertEqual(sent[0]["reward"], 0.25)

    def test_game_end_reports_only_real_sc2_results(self):
        game_end = method(
            ARENA / "agents/planner.py",
            "SC2Planner",
            "game_end",
            DEFEAT_PENALTY=-50,
            final_state_shaping_reward=lambda obs: 1.25,
        )
        cases = (("Victory", 20.5), ("Defeat", -48.25), ("Tie", -48.25))
        for result, expected_reward in cases:
            sent = []
            planner = SimpleNamespace(
                correct_format=True,
                exceed=False,
                last_consumed_policy_action_id=None,
                _send=sent.append,
                _wait_terminal_ack=lambda data: None,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                game_end(planner, "obs", result)
            self.assertEqual(sent[0]["reward"], expected_reward)
            self.assertTrue(sent[0]["terminated"])
            self.assertFalse(sent[0]["truncated"])

    def test_sc2_step_uses_gymnasium_five_tuple(self):
        data = {
            "obs": "terminal obs",
            "reward": 20.0,
            "done": True,
            "success": True,
            "attack_ct": 1,
            **outcome_fields("Victory"),
        }
        obs, reward, terminated, truncated, info = gymnasium_transition(data, True)
        self.assertEqual((obs, reward, terminated, truncated), ("terminal obs", 20.0, True, False))
        self.assertEqual(info["game_result"], "Victory")
        with self.assertRaisesRegex(RuntimeError, "without terminated/truncated attribution"):
            gymnasium_transition({**data, "terminated": False}, True)

    def test_manager_records_cutoff_once_and_refuses_truncated_training(self):
        step = method(
            ROOT / "agent_system/environments/strategic_sc2_env_manager_v2.py",
            "StrategicSC2EnvironmentManagerV2",
            "step",
            List=list,
            os=os,
            json=json,
            to_numpy=lambda value: value,
            infer_failure_reason=lambda *args: "",
        )
        for max_steps in (2, 0):
            for training in (False, True):
                info = {
                    "terminated": False,
                    "truncated": False,
                    "game_result": None,
                    "termination_reason": None,
                    "outcome_reward": 0.0,
                }
                truncated = []
                envs = SimpleNamespace(
                    step=lambda actions: (["next"], [0.25], [False], [False], [info]),
                    truncate=lambda indices: truncated.extend(indices),
                )
                manager = SimpleNamespace(
                    _forbid_thinking_output=False,
                    _apply_verified_candidate_override=lambda value: value,
                    _sanitize_final_actions=lambda value: (value, [False]),
                    _last_prompts=["query"],
                    _semantic_action_evaluable=lambda value: True,
                    _validate_current_bo_response=lambda *args: (True, ""),
                    projection_f=lambda value: (value, [True]),
                    envs=envs,
                    _terminal_by_env=[False],
                    _episode_step_by_env=[1],
                    _episode_reset_index=0,
                    _step_index=1,
                    config=SimpleNamespace(env=SimpleNamespace(max_steps=max_steps)),
                    _extract_failure_reason=lambda value: "",
                    pre_text_obs=["query"],
                    use_tool_call_memory=False,
                    _no_progress_counts=[0],
                    memory=None,
                    _write_prompt_response_log=lambda **kwargs: None,
                    _write_tool_call_log=lambda **kwargs: None,
                    is_train=training,
                    build_text_obs=lambda value: value,
                )
                with tempfile.TemporaryDirectory() as tmp, patch.dict(
                    os.environ, SC2_EPISODE_OUTCOME_LOG=tmp + "/outcomes.jsonl"
                ):
                    outcome_path = Path(tmp + "/outcomes.jsonl")
                    if training and max_steps > 0:
                        with self.assertRaisesRegex(RuntimeError, "refusing a PPO update"):
                            step(manager, ["- scv"])
                    else:
                        step(manager, ["- scv"])
                    rows = outcome_path.read_text().splitlines() if outcome_path.exists() else []
                self.assertEqual(truncated, [0] if max_steps > 0 else [])
                self.assertEqual(len(rows), 1 if max_steps > 0 else 0)
                if rows:
                    self.assertEqual(json.loads(rows[0])["outcome_reward"], 0.0)

    def test_run_summary_counts_only_natural_results(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        from run_record import episode_outcomes

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {"reset_index": 0, "env_index": 0, **outcome_fields("Defeat")},
                {"reset_index": 0, "env_index": 1, **outcome_fields("Victory")},
                {
                    "reset_index": 0,
                    "env_index": 2,
                    **truncation_fields(),
                },
            ]
            (root / "episode_outcomes.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            summary = episode_outcomes(root)
            self.assertEqual((summary["natural_games"], summary["truncated"]), (2, 1))
            self.assertEqual(summary["natural_win_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
