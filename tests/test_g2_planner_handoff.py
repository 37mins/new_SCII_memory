"""Real queue/Commander/Planner methods; only SC2 and model IO are doubled."""
import ast
from collections import Counter, defaultdict
import contextlib
import copy
import hashlib
import io
import json
import os
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import test_empty_bo_frame as frame_tests
from test_r6_format_reward import planner_scope, strict_validator
from test_r6_natural_termination import ARENA
from test_valid_bo_passthrough import CONTROLS

G2 = ARENA / "knowledge/BO/replay_observations_Flat128_d4_TvT_Macro_g2_Result.Victory_twobase.json"
G2_SHA = "304b1f4ff12fe30c5e6d5f7b8eaef8ed6823ca65afed3a3706cae10e3639c494"
ENV = dict(CONTROLS, SC2_FIXED_BO_PREFIX_LIMIT="138", SC2_PRESERVE_VALID_BO="1",
           SC2_STRATEGY_PROFILE="marine_pressure", SC2_ENABLE_AIR_OPTIONS="1")
# Synthetic interface fixture, not a recovered historical observation or model run.
OBS = ("It is now 06:53 in the game. In terms of our military forces, we have "
       "15 marine, 40 SCV. In terms of the structures, we have 2 bases in the form "
       "of orbitalcommand, 8 barracks, 5 barracks with techlab, 3 factory, "
       "1 factory with techlab, 2 refinery, 11 supplydepot. "
       "For the visible enemy presence, there are no enemy units currently.")


def initial_player(limit=None, bo_path=G2):
    """Execute the actual complete constructor, without creating game/model clients."""
    class ScriptPlayer:
        def __init__(self, config):
            self.config = config
            self.model_name = "cpu-fixture"
            self.generation_config = {}
            self.llm_client = None
            self.log_path = None

        def _load_ability_data(self):
            return ""

        def _build_ability_requirements(self):
            return {}

    reader = {}
    path = ARENA / "tools/bo_reader.py"
    exec(compile(path.read_text(), str(path), "exec"), reader)
    path = ARENA / "players/llm_player.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LLMPlayer")
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]
    scope = dict(ScriptPlayer=ScriptPlayer, os=os, json=json, hashlib=hashlib,
                 load_bo_data=reader["load_bo_data"], AdjustAgent=SimpleNamespace,
                 MicroAgent=SimpleNamespace, WorldModelAgent=SimpleNamespace,
                 SC2Commander=SimpleNamespace)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
    env = {k: v for k, v in os.environ.items() if k != "SC2_FIXED_BO_PREFIX_LIMIT"}
    if limit is not None:
        env["SC2_FIXED_BO_PREFIX_LIMIT"] = str(limit)
    with patch.dict(os.environ, env, clear=True):
        return scope["LLMPlayer"](SimpleNamespace(port=0, enable_plan_verifier=False), str(bo_path))


def interface_evidence(answer, observation=OBS):
    """Run original Strategist -> Commander -> Planner/validator, replacing socket IO."""
    planner_cls = planner_scope()["SC2Planner"]
    planner = planner_cls.__new__(planner_cls)
    planner.ct, planner.correct_format, planner.exceed = 0, False, False
    planner.last_consumed_policy_action_id = None
    sent, verdicts = [], []
    planner._send = sent.append

    def receive():
        verdict = strict_validator()(sent[-1]["obs"], answer)
        verdicts.append(verdict)
        return dict(msg="ACTION", value=answer, policy_action_id=1, plan_format_valid=verdict[0])

    planner._recv = receive
    scope = dict(copy=copy, json=json, os=os, re=re, defaultdict=defaultdict)
    path = ARENA / "agents/strategist.py"
    exec(compile(path.read_text(), str(path), "exec"), scope)
    path = ARENA / "agents/commander.py"
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    scope.update(SC2Planner=lambda *args: planner, __file__=str(path))
    exec(compile(tree, str(path), "exec"), scope)
    with patch.dict(os.environ, ENV), contextlib.redirect_stdout(io.StringIO()):
        commander = scope["SC2Commander"]()
        queue = commander.renew_bo(observation)
        raw_phase = copy.deepcopy(commander.strategist.strategy[commander.strategy["_phase_index"]])
    return dict(evidence_type="synthetic CPU observation and hand-written answer; no model/game",
                observation=observation, raw_phase=raw_phase, selected_strategy=commander.strategy,
                waiting_before_answer=commander.bo_todo_strategy, query=sent[0]["obs"],
                answer=answer, strict_valid=verdicts[0][0], validation_reason=verdicts[0][1],
                executor_queue=queue, waiting_after_answer=commander.waiting_bo,
                selected_counts=dict(commander.bo_done))


class InitialQueueTest(unittest.TestCase):
    def test_default_and_explicit_zero_preserve_all_237_entries(self):
        original = json.loads(G2.read_text())
        for limit in (None, 0):
            with self.subTest(limit=limit):
                player = initial_player(limit)
                self.assertEqual(len(player.bo_data), 237)
                self.assertEqual([{k: v for k, v in row.items() if k != "seconds"}
                                  for row in player.bo_data], original)

    def test_candidate_preserves_exact_prefix_and_records_withheld_tail(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            player = initial_player(138)
        self.assertEqual(player.bo_data, initial_player(0).bo_data[:138])
        self.assertEqual(player.current_bo_index, 0)
        self.assertEqual(hashlib.sha256(G2.read_bytes()).hexdigest(), G2_SHA)
        evidence = json.loads(output.getvalue().split("Fixed BO handoff: ", 1)[1])
        self.assertEqual(evidence, dict(bo_sha256=G2_SHA, retained_prefix_count=138,
                                       withheld_indices_half_open=[138, 237]))

    def test_unsupported_limits_and_different_bo_are_rejected(self):
        for limit in (-2, 137, 139, "bad"):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                initial_player(limit)
        with self.assertRaisesRegex(ValueError, "frozen canonical g2"):
            initial_player(138, ARENA / "knowledge/data_BO2Action_1.json")

    def test_planner_from_start_withholds_all_fixed_actions_without_marking_done(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            player = initial_player(-1)
        self.assertEqual(player.bo_data, [])
        self.assertEqual(player.current_bo_index, 0)
        self.assertEqual(player.notified_bo_indices, set())
        evidence = json.loads(output.getvalue().split("Fixed BO handoff: ", 1)[1])
        self.assertEqual(evidence, dict(bo_sha256=G2_SHA, retained_prefix_count=0,
                                       withheld_indices_half_open=[0, 237]))

    def test_opening_uses_existing_phase_zero_and_answer_changes_queue(self):
        observation = ("It is now 00:00 in the game. In terms of our military forces, "
                       "we have 12 SCV. In terms of the structures, we have 1 base "
                       "in the form of commandcenter. For the visible enemy presence, "
                       "there are no enemy units currently.")
        a = interface_evidence("- supplydepot", observation)
        b = interface_evidence("- supplydepot\n- supplydepot", observation)
        self.assertEqual(a["selected_strategy"]["_phase_index"], 0)
        self.assertEqual(a["raw_phase"]["bo todo"],
                         dict(SCV=2, supplydepot=4, barracks=4, marine=24))
        self.assertEqual(a["waiting_before_answer"], dict(supplydepot=4, barracks=4, marine=24))
        self.assertEqual(a["query"], b["query"])
        self.assertTrue(a["strict_valid"] and b["strict_valid"])
        self.assertEqual([row["action"] for row in a["executor_queue"]], ["supplydepot"])
        self.assertEqual([row["action"] for row in b["executor_queue"]], ["supplydepot"] * 2)
        self.assertEqual(a["waiting_after_answer"]["supplydepot"], 3)
        self.assertEqual(b["waiting_after_answer"]["supplydepot"], 2)

    def test_stage_quota_source_is_existing_strategist_not_generated_tail(self):
        evidence = interface_evidence("- marine")
        todo = dict(evidence["raw_phase"]["bo todo"])
        self.assertEqual((todo.pop("starport"), todo.pop("vikingfighter")), (1, 8))
        tail = initial_player(0).bo_data[138:]
        known_actions = {key.lower(): value for key, value in todo.items()}
        counts = Counter(row["action"].lower() for row in tail
                         if row["action"].lower() in known_actions)
        self.assertEqual(dict(counts), known_actions)
        self.assertEqual(sum(counts.values()), 96)
        self.assertEqual(evidence["selected_strategy"]["_phase_index"], 2)
        self.assertTrue(evidence["selected_strategy"]["_terminal_phase"])

    def test_same_input_different_valid_answers_change_executor_queue(self):
        a = interface_evidence("\n".join(["- marine"] * 6))
        b = interface_evidence("- marine\n- starport")
        self.assertEqual(a["query"], b["query"])
        self.assertEqual(a["waiting_before_answer"], b["waiting_before_answer"])
        self.assertTrue(a["strict_valid"] and b["strict_valid"])
        self.assertEqual([row["action"] for row in a["executor_queue"]], ["marine"] * 6)
        self.assertEqual([row["action"] for row in b["executor_queue"]], ["marine", "starport"])
        self.assertEqual(a["selected_counts"], {"marine": 6})
        self.assertEqual(b["selected_counts"], {"marine": 1, "starport": 1})
        self.assertEqual(a["waiting_after_answer"]["marine"], 44)
        self.assertEqual(b["waiting_after_answer"]["marine"], 49)
        self.assertNotIn("starport", b["waiting_after_answer"])


class HandoffCallbackTest(unittest.IsolatedAsyncioTestCase):
    setUp = frame_tests.EmptyBoFrameTest.setUp

    async def test_planner_from_start_calls_before_first_bo_and_preserves_later_plan(self):
        self.player.bo_data = initial_player(-1).bo_data
        plan = [{"action": "marine"}, {"action": "marine"}]
        self.plans = iter([plan])
        await self.player.on_step(0)
        self.assertEqual(self.events, [(0, "maintenance"), (0, "planner"), (0, "submit")])
        self.assertIs(self.player.bo_data, plan)
        await self.player.on_step(1)
        self.player.commander.renew_bo.assert_called_once()
        self.assertEqual(self.player.run_actions.await_count, 2)

    def at_last_prefix_bo(self, limit=138):
        self.player.bo_data = initial_player(limit).bo_data
        self.player.current_bo_index = 137
        self.player.time = 413
        self.player.time_formatted = "06:53"

    async def test_handoff_only_after_last_prefix_bo_and_later_queues_are_not_cut(self):
        self.at_last_prefix_bo()
        # Deliberately longer than the limit to expose accidental re-slicing.
        later_queue = [{"action": "marine"} for _ in range(140)]
        self.plans = iter([later_queue, [{"action": "marine"}]])
        await self.player.on_step(6123)
        self.player.commander.renew_bo.assert_not_called()
        self.assertEqual(self.player.current_bo_index, 138)
        await self.player.on_step(6124)
        self.player.commander.renew_bo.assert_called_once()
        self.assertIs(self.player.bo_data, later_queue)
        self.assertEqual(len(self.player.bo_data), 140)
        for frame in range(6125, 6264):
            await self.player.on_step(frame)
        self.player.commander.renew_bo.assert_called_once()
        self.assertEqual(self.player.current_bo_index, 140)
        await self.player.on_step(6264)
        self.assertEqual(self.player.commander.renew_bo.call_count, 2)
        self.assertEqual(self.player.run_actions.await_count, 142)

    async def test_no_handoff_while_current_command_is_waiting(self):
        self.at_last_prefix_bo()
        self.player.validate_action_and_get_suggestion.return_value = "not ready"
        await self.player.on_step(6123)
        self.assertEqual(self.player.current_bo_index, 137)
        self.player.commander.renew_bo.assert_not_called()
        self.player.run_actions.assert_not_awaited()
        self.player.validate_action_and_get_suggestion.return_value = ""
        await self.player.on_step(6146)
        self.assertEqual(self.player.current_bo_index, 138)
        self.player.commander.renew_bo.assert_not_called()
        await self.player.on_step(6147)
        self.player.commander.renew_bo.assert_called_once()

    async def test_existing_timeout_skip_boundary_is_unchanged(self):
        self.at_last_prefix_bo()
        self.player.current_bo_wait_count = 10
        self.player.last_bo_checked_index = 137
        self.player.last_bo_check_time = 0
        self.player.chat_send = AsyncMock()
        await self.player.on_step(6123)
        self.assertEqual(self.player.current_bo_index, 138)
        self.player.run_actions.assert_not_awaited()
        self.player.commander.renew_bo.assert_not_called()
        self.player.chat_send.assert_awaited_once()
        await self.player.on_step(6124)
        self.player.commander.renew_bo.assert_called_once()

    async def test_disabled_candidate_still_executes_fixed_bo_138(self):
        self.at_last_prefix_bo(0)
        await self.player.on_step(6123)
        await self.player.on_step(6124)
        self.player.commander.renew_bo.assert_not_called()
        self.assertEqual(self.player.current_bo_index, 139)
        self.assertEqual(self.player.run_actions.await_count, 2)


if __name__ == "__main__":
    if sys.argv[1:] == ["--evidence"]:
        print(json.dumps([interface_evidence("\n".join(["- marine"] * 6)),
                          interface_evidence("- marine\n- starport")], indent=2))
    else:
        unittest.main()
