"""Actual player callbacks with CPU-only Commander/SC2 boundary doubles."""
import contextlib
import io
from collections import Counter
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock

from test_r6_natural_termination import ARENA, method


ABILITY = "BARRACKSTRAIN_MARINE"
ACTION = {"action": ABILITY, "units": [7]}


class Units(list):
    def __add__(self, other):
        return Units(super().__add__(other))

    def filter(self, predicate):
        return Units(unit for unit in self if predicate(unit))


class EmptyBoFrameTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)
        self.plans = iter(())
        self.refreshes = Counter()
        self.events = []
        self.player = SimpleNamespace(
            strategy_params={}, bo_data=[], current_bo_index=0,
            current_bo_parsed_state=None, pending_pre_move=None,
            execution_feedback_enabled=False, planner_queue_active=False,
            pending_execution_feedback=None, last_bo_block_reason=None,
            current_bo_pre_move_attempted=False,
            time=0, time_formatted="00:00", ability_descriptions_text="abilities",
            units=Units([SimpleNamespace(tag=11)]),
            structures=Units([SimpleNamespace(tag=7)]), townhalls=[True],
            supply_used=12, supply_cap=15, sbr=SimpleNamespace(update=Mock()),
            log_current_iteration=Mock(), logging=Mock(), update_tag_to_health=Mock(),
            _looks_like_attack_bo=Mock(return_value=False),
            get_available_abilities=AsyncMock(return_value=[[], [ABILITY]]),
            validate_action_and_get_suggestion=AsyncMock(return_value=""),
            units_to_text_wo_contructing=AsyncMock(return_value="[7]Barracks"),
            units_to_text=AsyncMock(return_value="units"),
            structures_to_text=AsyncMock(return_value="structures"),
            obs_to_text_dict=AsyncMock(return_value={"Map information": "map"}),
            get_action_target_type=Mock(return_value="None"),
            _check_bo_delay=Mock(),
            micro_agent=SimpleNamespace(
                _local_fallback_enabled=Mock(return_value=False),
                bo_to_ability=AsyncMock(return_value=[ABILITY]),
                select_executor=AsyncMock(return_value=[7]),
                get_final_action=AsyncMock(return_value=[ACTION]),
            ),
        )

        async def maintain(iteration, **_):
            self.frame = iteration
            self.events.append((iteration, "maintenance"))
            return []

        def renew(obs):
            self.refreshes[self.frame] += 1
            # Bound the old synchronous loop without sleeping or hanging the test.
            if self.refreshes[self.frame] > 1:
                raise RuntimeError("repeated Planner refresh in the same frame")
            self.events.append((self.frame, "planner"))
            return next(self.plans, [])

        async def submit(actions):
            self.events.append((self.frame, "submit"))

        self.player.run_maintenance_tasks = AsyncMock(side_effect=maintain)
        self.player.get_obs = Mock(side_effect=lambda: f"observation-frame-{self.frame}")
        self.player.commander = SimpleNamespace(renew_bo=Mock(side_effect=renew))
        self.player.run_actions = AsyncMock(side_effect=submit)
        self.player.run = MethodType(method(
            ARENA / "players/llm_player.py", "LLMPlayer", "run",
            AbilityId={ABILITY: ABILITY},
        ), self.player)
        self.player.on_step = MethodType(method(
            ARENA / "players/base_player.py", "BasePlayer", "on_step",
        ), self.player)
        self.player._finish_current_bo = Mock(wraps=MethodType(method(
            ARENA / "players/llm_player.py", "LLMPlayer", "_finish_current_bo",
        ), self.player))
        self.player._fail_current_bo = MethodType(method(
            ARENA / "players/llm_player.py", "LLMPlayer", "_fail_current_bo",
        ), self.player)

    async def test_empty_plan_returns_through_on_step_without_inventing_actions(self):
        await self.player.on_step(6120)
        self.player.run_maintenance_tasks.assert_awaited_once_with(6120)
        self.player.commander.renew_bo.assert_called_once_with("observation-frame-6120")
        self.player.update_tag_to_health.assert_called_once_with()
        self.player.run_actions.assert_not_awaited()
        self.player._finish_current_bo.assert_not_called()
        self.assertEqual((self.player.bo_data, self.player.current_bo_index), ([], 0))

    async def test_consecutive_empty_plans_return_once_per_frame(self):
        for frame in (6123, 6124, 6125):
            await self.player.on_step(frame)
        self.assertEqual(self.refreshes, {6123: 1, 6124: 1, 6125: 1})
        self.assertEqual(self.player.run_maintenance_tasks.await_count, 3)
        self.assertEqual([call.args[0] for call in self.player.commander.renew_bo.call_args_list],
                         [f"observation-frame-{frame}" for frame in (6123, 6124, 6125)])
        self.player.run_actions.assert_not_awaited()

    async def test_empty_then_marine_submits_on_the_next_frame(self):
        self.plans = iter([[], [{"action": "marine"}]])
        await self.player.on_step(6123)
        self.player.run_actions.assert_not_awaited()
        await self.player.on_step(6124)
        self.player.run_actions.assert_awaited_once_with([ACTION])
        self.player._finish_current_bo.assert_called_once_with()
        self.assertEqual(self.player.current_bo_index, 1)
        self.assertEqual(self.events, [(6123, "maintenance"), (6123, "planner"),
                                     (6124, "maintenance"), (6124, "planner"),
                                     (6124, "submit")])

    async def test_existing_nonempty_queue_executes_without_refresh(self):
        self.player.bo_data = [{"action": "marine"}]
        await self.player.on_step(6123)
        self.player.commander.renew_bo.assert_not_called()
        self.player.run_actions.assert_awaited_once_with([ACTION])
        self.player._finish_current_bo.assert_called_once_with()

    async def test_exhausted_queue_refreshes_and_executes_in_same_frame(self):
        self.player.bo_data = [{"action": "marine"}]
        self.player.current_bo_index = 1
        self.plans = iter([[{"action": "marine"}]])
        await self.player.on_step(6123)
        self.player.commander.renew_bo.assert_called_once_with("observation-frame-6123")
        self.player.run_actions.assert_awaited_once_with([ACTION])
        self.assertEqual(self.player.current_bo_index, 1)

    async def test_planner_exception_propagates_without_retry(self):
        self.player.commander.renew_bo.side_effect = ValueError("planner failed")
        with self.assertRaisesRegex(ValueError, "planner failed"):
            await self.player.on_step(6123)
        self.player.commander.renew_bo.assert_called_once()
        self.player.run_actions.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
