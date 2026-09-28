"""Execution-failure replanning with real methods and CPU-only socket/SC2 doubles."""
from collections import defaultdict
import contextlib
import copy
import io
import json
import os
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import test_empty_bo_frame as frame_test
from test_r6_format_reward import planner_scope, strict_validator
from test_r6_natural_termination import ARENA, method
from test_valid_bo_passthrough import CONTROLS, STRATEGY

ABILITY, ACTION = frame_test.ABILITY, frame_test.ACTION


def commander(todo, selected=(), next_plan=()):
    obj = SimpleNamespace(
        waiting_bo=dict(todo), current_bo=list(selected), current_bo_debits=[],
        strategy=dict(STRATEGY, _phase_index=1), strategy_renew_ct=2,
        has_attack=False, attack_ct=0, bo_done=defaultdict(int),
        bo_todo_strategy=dict(todo), strategy_start_time=210, strategy_last_time=150,
        planner=SimpleNamespace(generate_plan=Mock(return_value=list(next_plan)),
                                last_consumed_policy_action_id=41),
        should_renew_strategy_by_time=Mock(return_value=True),
        renew_strategy=Mock(), reward_planner_speed=Mock(return_value=100),
        ensure_opening_marine=Mock(), get_intent=lambda action: "test intent",
    )
    obj.ct_waiting_bo = lambda: len(obj.waiting_bo)
    scope = dict(os=os, copy=copy, defaultdict=defaultdict,
                 normalize_bo_name=lambda s: str(s).strip().lower())
    for name in ("renew_waiting_bo", "release_unsubmitted_bo", "renew_bo"):
        setattr(obj, name, MethodType(method(
            ARENA / "agents/commander.py", "SC2Commander", name, **scope), obj))
    return obj


def feedback():
    return dict(
        failed=dict(action="reactor barracks", reason="timeout: insufficient gas",
                    target_command_submitted=False, pre_move_command_attempted=False),
        unsubmitted_suffix=["refinery", "marine"],
        restored_selectable_quota={"reactor barracks": 1, "marine": 1},
        source_policy_action_id=41, phase_index=1, processed_prefix_count=1,
    )


class QuotaReleaseTest(unittest.TestCase):
    def test_records_only_actual_debits_and_preserves_normal_deduction(self):
        c = commander({"marine": 2}, ["Marine", "marine", "marine", "refinery"])
        c.renew_waiting_bo()
        self.assertEqual(c.waiting_bo, {})
        self.assertEqual(c.current_bo_debits, ["marine", "marine", None, None])

    def test_release_skips_processed_prefix_and_is_idempotent(self):
        c = commander({"marine": 2, "reactor barracks": 1},
                      ["marine", "reactor barracks", "refinery", "marine"])
        c.renew_waiting_bo()
        self.assertEqual(c.release_unsubmitted_bo(1), {"reactor barracks": 1, "marine": 1})
        self.assertEqual(c.waiting_bo, {"reactor barracks": 1, "marine": 1})
        self.assertEqual(c.current_bo_debits, ["marine", None, None, None])
        self.assertEqual(c.release_unsubmitted_bo(1), {})
        self.assertEqual(c.waiting_bo, {"reactor barracks": 1, "marine": 1})

    def test_release_does_not_restore_rule_added_or_over_quota_actions(self):
        c = commander({"marine": 1}, ["marine", "marine", "refinery"])
        c.renew_waiting_bo()
        self.assertEqual(c.release_unsubmitted_bo(1), {})
        self.assertEqual(c.waiting_bo, {})


class CommanderFeedbackTest(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def test_feedback_does_not_advance_empty_or_expired_phase(self):
        for todo in ({}, {"marine": 2}):
            with self.subTest(todo=todo):
                c = commander(todo)
                f = feedback()
                self.assertEqual(c.renew_bo("It is now 11:43 in the game.", execution_feedback=f), [])
                c.renew_strategy.assert_not_called()
                c.reward_planner_speed.assert_not_called()
                self.assertEqual(c.strategy_renew_ct, 2)
                call = c.planner.generate_plan.call_args
                self.assertEqual(call.kwargs["execution_feedback"], f)
                self.assertEqual(call.args[3:6], (0, False, 0))

    def test_feedback_keeps_existing_non_phase_attack_reward(self):
        c = commander({"marine": 2}, next_plan=["marine"])
        c.has_attack = True
        c.attack_ct = 3
        c.renew_bo("It is now 11:43 in the game.", execution_feedback=feedback())
        self.assertEqual(c.planner.generate_plan.call_args.args[3:6], (0.5, False, 4))
        self.assertFalse(c.has_attack)
        self.assertEqual(c.attack_ct, 4)
        c.renew_strategy.assert_not_called()

    def test_restored_quota_is_optional_and_only_new_selection_is_deducted(self):
        c = commander({"reactor barracks": 1, "marine": 2}, next_plan=["marine"])
        queue = c.renew_bo("It is now 11:43 in the game.", execution_feedback=feedback())
        self.assertEqual([item["action"] for item in queue], ["marine"])
        self.assertEqual(c.waiting_bo, {"reactor barracks": 1, "marine": 1})
        self.assertEqual(c.current_bo_debits, ["marine"])


def plan_with_feedback(answer, obs, todo, extra):
    scope = planner_scope()
    scope["json"] = json
    p = scope["SC2Planner"].__new__(scope["SC2Planner"])
    p.ct = 0
    p.correct_format = p.exceed = False
    p.last_consumed_policy_action_id = None
    sent = []
    p._send = sent.append
    validate = strict_validator()
    p._recv = lambda: dict(msg="ACTION", value=answer, policy_action_id=42,
                          plan_format_valid=validate(sent[-1]["obs"], answer)[0])
    controls = dict(CONTROLS, SC2_ORIGINAL_BO_SEGMENT="1", SC2_PRESERVE_VALID_BO="1",
                    SC2_CLARIFY_BO_QUOTA="1", SC2_EXECUTION_FAILURE_FEEDBACK="1")
    with patch.dict(os.environ, controls), contextlib.redirect_stdout(io.StringIO()):
        result = p.generate_plan(obs, todo, STRATEGY, 0, execution_feedback=extra)
    return result, sent[0]["obs"], p.correct_format, p.last_consumed_policy_action_id


class PlannerFeedbackTest(unittest.TestCase):
    def test_feedback_does_not_invent_observed_buildings_or_change_allowed_names(self):
        obs = "It is now 05:00 in the game. We have 1 barracks."
        f = feedback()
        f["failed"]["reason"] = "Not completed: 3 factory, 4 refinery, 2 starport."
        a = plan_with_feedback("- factory", obs, {"siegetank": 3}, None)
        b = plan_with_feedback("- factory", obs, {"siegetank": 3}, f)
        self.assertEqual(a[0], b[0])
        self.assertTrue(a[2] and b[2])
        self.assertTrue(b[1].startswith(a[1]))
        self.assertIn("Not completed: 3 factory, 4 refinery, 2 starport.", b[1])
        invalid = plan_with_feedback("- siegetank", obs, {"siegetank": 3}, f)
        self.assertFalse(invalid[2])

    def test_feedback_preserves_count_protocol_and_consumes_only_actual_answer(self):
        obs = "It is now 04:00 in the game. We have 4 barracks and 1 refinery."
        a = plan_with_feedback("- marine x 21", obs, {"marine": 42}, None)
        b = plan_with_feedback("- marine x 21", obs, {"marine": 42}, feedback())
        self.assertEqual(a[0], ["marine"] * 21)
        self.assertEqual(a[0], b[0])
        self.assertTrue(b[2])
        self.assertEqual(b[3], 42)
        self.assertTrue(b[1].startswith(a[1]))

    def test_different_new_answers_change_plan_without_forcing_failed_action(self):
        obs = "It is now 04:00 in the game. We have 4 barracks and 1 refinery."
        todo = {"reactor barracks": 1, "marine": 4}
        a = plan_with_feedback("- marine x 2", obs, todo, feedback())
        b = plan_with_feedback("- reactor barracks", obs, todo, feedback())
        self.assertEqual(a[0], ["marine", "marine"])
        self.assertEqual(b[0], ["reactor barracks"])
        self.assertTrue(a[2] and b[2])


class PlayerFeedbackTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        frame_test.EmptyBoFrameTest.setUp(self)
        p = self.player
        p.execution_feedback_enabled = True
        p.planner_queue_active = True
        p.pending_execution_feedback = None
        p.last_bo_block_reason = None
        p.current_bo_pre_move_attempted = False
        p._fail_current_bo = MethodType(method(
            ARENA / "players/llm_player.py", "LLMPlayer", "_fail_current_bo", json=json), p)
        p.chat_send = AsyncMock()
        p.bo_data = [{"action": name} for name in ("marine", "reactor barracks", "refinery", "marine")]
        p.current_bo_index = 1
        c = commander({"marine": 2, "reactor barracks": 1}, [item["action"] for item in p.bo_data])
        c.renew_waiting_bo()
        p.commander = c
        c.renew_bo = Mock(return_value=[{"action": "marine"}])

    async def test_mapping_failure_replans_next_frame_not_in_same_frame(self):
        p = self.player
        p.micro_agent.bo_to_ability.return_value = []
        await p.on_step(6123)
        p.commander.renew_bo.assert_not_called()
        p.run_actions.assert_not_awaited()
        f = copy.deepcopy(p.pending_execution_feedback)
        self.assertEqual(f["failed"]["action"], "reactor barracks")
        self.assertFalse(f["failed"]["target_command_submitted"])
        self.assertFalse(f["failed"]["pre_move_command_attempted"])
        self.assertEqual(f["unsubmitted_suffix"], ["refinery", "marine"])
        self.assertEqual(f["restored_selectable_quota"], {"reactor barracks": 1, "marine": 1})
        self.assertEqual(f["source_policy_action_id"], 41)
        self.assertEqual(f["phase_index"], 1)
        self.assertEqual(f["processed_prefix_count"], 1)
        self.assertEqual(f["queue_source"], "parsed_answer_plus_existing_rules")
        p.micro_agent.bo_to_ability.return_value = [ABILITY]
        await p.on_step(6124)
        p.commander.renew_bo.assert_called_once_with("observation-frame-6124", execution_feedback=f)
        p.run_actions.assert_awaited_once_with([ACTION])
        self.assertIsNone(p.pending_execution_feedback)
        self.assertEqual([item["action"] for item in p.bo_data], ["marine"])

    async def test_timeout_preserves_pre_move_command_but_not_stale_local_state(self):
        p = self.player
        p.time = 400
        p.current_bo_parsed_state = dict(type="OPERATION", ability_name=ABILITY)
        p.current_bo_wait_count = 10
        p.last_bo_check_time = 100
        p.last_bo_checked_index = 1
        p.last_bo_block_reason = "Insufficient Vespene Gas (0/50)"
        p.pending_pre_move = dict(bo_index=1, ability=ABILITY, unit_id=7)
        p.current_bo_pre_move_attempted = True
        await p.on_step(122)
        p.commander.renew_bo.assert_not_called()
        p.run_actions.assert_not_awaited()  # No cancel, STOP, or replacement MOVE.
        f = p.pending_execution_feedback
        self.assertTrue(f["failed"]["pre_move_command_attempted"])
        self.assertFalse(f["failed"]["target_command_submitted"])
        self.assertIn("timeout", f["failed"]["reason"].lower())
        self.assertIsNone(p.pending_pre_move)
        self.assertIsNone(p.current_bo_parsed_state)
        self.assertFalse(p.current_bo_pre_move_attempted)

    def test_lost_pre_move_cache_does_not_erase_prior_command_attempt(self):
        p = self.player
        p.pending_pre_move = None
        p.current_bo_pre_move_attempted = True
        p._fail_current_bo("timeout: worker missing")
        self.assertTrue(p.pending_execution_feedback["failed"]["pre_move_command_attempted"])
        p.run_actions.assert_not_awaited()

    async def test_temporary_block_does_not_replan_before_existing_timeout(self):
        p = self.player
        p.time = 200
        p.validate_action_and_get_suggestion.return_value = "Insufficient Minerals (0/50)"
        await p.on_step(100)
        self.assertIsNone(p.pending_execution_feedback)
        self.assertEqual(p.current_bo_index, 1)
        p.commander.renew_bo.assert_not_called()
        p.run_actions.assert_not_awaited()
        self.assertEqual(p.commander.waiting_bo, {})

    async def test_disabled_or_fixed_queue_mapping_failure_retains_skip_behavior(self):
        for enabled, planner_queue in ((False, True), (True, False)):
            with self.subTest(enabled=enabled, planner_queue=planner_queue):
                p = self.player
                p.execution_feedback_enabled = enabled
                p.planner_queue_active = planner_queue
                p.current_bo_index = 1
                p.micro_agent.bo_to_ability.return_value = []
                await p.on_step(100)
                self.assertEqual(p.current_bo_index, 2)
                self.assertIsNone(p.pending_execution_feedback)
                self.assertEqual(p.commander.waiting_bo, {})
                p.commander.renew_bo.assert_not_called()

    async def test_formal_submission_stays_on_existing_finish_path(self):
        await self.player.on_step(100)
        self.player.run_actions.assert_awaited_once_with([ACTION])
        self.assertEqual(self.player.current_bo_index, 2)
        self.assertIsNone(self.player.pending_execution_feedback)
        self.assertEqual(self.player.commander.waiting_bo, {})

    async def test_empty_selectable_quota_records_failure_and_keeps_old_skip(self):
        p = self.player
        p.bo_data = [{"action": "marine"}, {"action": "refinery"}]
        p.commander = commander({"marine": 1}, ["marine", "refinery"])
        p.commander.renew_waiting_bo()
        p.commander.renew_bo = Mock(return_value=[])
        p.micro_agent.bo_to_ability.return_value = []
        await p.on_step(100)
        events = [call.args[1] for call in p.logging.call_args_list
                  if call.args[0] == "execution_failure_feedback"]
        self.assertEqual(len(events), 1)
        self.assertFalse(events[0]["replan_scheduled"])
        self.assertEqual(events[0]["skip_reason"], "no_selectable_quota")
        self.assertEqual(events[0]["restored_selectable_quota"], {})
        self.assertEqual(p.commander.waiting_bo, {})
        self.assertEqual(p.current_bo_index, 2)
        self.assertIsNone(p.pending_execution_feedback)
        p.commander.renew_bo.assert_not_called()
        p.run_actions.assert_not_awaited()

    async def test_mule_skip_does_not_release_quota_or_request_feedback(self):
        p = self.player
        p.current_bo_parsed_state = dict(type="OPERATION", ability_name="CALLDOWNMULE_CALLDOWNMULE")
        await p.on_step(100)
        self.assertEqual(p.current_bo_index, 2)
        self.assertEqual(p.commander.waiting_bo, {})
        self.assertIsNone(p.pending_execution_feedback)
        p._finish_current_bo.assert_called_once_with()
        p.commander.renew_bo.assert_not_called()
        p.run_actions.assert_not_awaited()

    async def test_attack_accounting_does_not_release_quota_or_request_feedback(self):
        p = self.player
        p.current_bo_parsed_state = dict(type="ATTACK", attack_cmds=["Launch OFFENSE with 8 marine"])
        await p.on_step(100)
        self.assertEqual(p.current_bo_index, 2)
        self.assertEqual(p.commander.waiting_bo, {})
        self.assertIsNone(p.pending_execution_feedback)
        p._finish_current_bo.assert_called_once_with()
        p.commander.renew_bo.assert_not_called()
        p.run_actions.assert_not_awaited()

    async def test_feedback_empty_plan_does_not_requery_in_same_frame(self):
        p = self.player
        p._fail_current_bo("mapping_failed")
        f = copy.deepcopy(p.pending_execution_feedback)
        p.commander.renew_bo.return_value = []
        await p.on_step(100)
        p.commander.renew_bo.assert_called_once_with("observation-frame-100", execution_feedback=f)
        self.assertEqual(p.bo_data, [])
        self.assertIsNone(p.pending_execution_feedback)
        p.run_actions.assert_not_awaited()
        # The next frame may use the existing ordinary empty-queue path once.
        await p.on_step(101)
        self.assertEqual(p.commander.renew_bo.call_count, 2)
        self.assertEqual(p.commander.renew_bo.call_args.args, ("observation-frame-101",))
        self.assertEqual(p.commander.renew_bo.call_args.kwargs, {})
        p.run_actions.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
