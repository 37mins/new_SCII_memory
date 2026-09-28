"""Real player callbacks on recorded occupancy; no SC2 engine, model or GPU."""
import json
import re
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
from sc2.units import Units

import test_empty_bo_frame as frame_test
from test_pending_addon_placement import worker_order
from test_r6_natural_termination import ARENA, method


PLAYER = ARENA / "players/llm_player.py"
MARINE = AbilityId.BARRACKSTRAIN_MARINE
RECORD = ARENA.parents[4] / "experiments/records/20260906-integrated-fixes-v1/r4-reactor-conflict.json"


class BarracksSlotsTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        frame_test.EmptyBoFrameTest.setUp(self)
        p = self.player
        p.bo_data = [{"action": "marine"}]
        p.time, p.time_formatted = 400, "06:40"
        p.reactor_tags = set()
        p.miner_units = []
        p.start_location = Point2((24, 22))
        p.unit_to_text = AsyncMock(side_effect=lambda u: f"[{u.tag}]{u.name}")
        p.run = MethodType(method(PLAYER, "LLMPlayer", "run", AbilityId=AbilityId), p)
        p.units_to_text_wo_contructing = MethodType(method(
            ARENA / "players/base_player.py", "BasePlayer", "units_to_text_wo_contructing",
            Units=Units), p)
        micro = p.micro_agent
        micro._local_fallback_enabled.return_value = True
        micro.think = []
        micro._fallback_select_executor = MethodType(method(
            ARENA / "agents/micro_agent.py", "MicroAgent", "_fallback_select_executor", re=re), micro)
        micro.select_executor = AsyncMock(wraps=MethodType(method(
            ARENA / "agents/micro_agent.py", "MicroAgent", "select_executor", json=json), micro))
        micro.get_final_action = AsyncMock(wraps=MethodType(method(
            ARENA / "agents/micro_agent.py", "MicroAgent", "get_final_action"), micro))

    def unit(self, tag, count=0, addon=0, position=(24, 26)):
        return SimpleNamespace(tag=tag, name="Barracks", position=Point2(position),
            is_mine=True, is_constructing_scv=False, add_on_tag=addon,
            orders=[worker_order(MARINE) for _ in range(count)])

    def configure(self, units, allowed=None, ability=MARINE):
        p = self.player
        p.units = Units([SimpleNamespace(tag=99991)], p)  # A live worker permits on_step.
        p.structures = Units(units, p)
        if allowed is None:
            allowed = {u.tag for u in units}
        p.get_available_abilities.return_value = [[]] + [[ability] if u.tag in allowed else [] for u in units]
        p.micro_agent.bo_to_ability.return_value = [ability.name]

    def submitted(self):
        return self.player.run_actions.call_args.args[0]

    async def test_recorded_busy_barracks_are_not_selected_over_free_slots(self):
        snapshots = json.loads(RECORD.read_text())["snapshots"]
        for frame, expected in ((2095, 170), (2109, 17), (2333, 387), (2334, 387)):
            with self.subTest(frame=frame):
                s = next(s for s in snapshots if s["frame"] == frame)
                p = self.player
                p.current_bo_index, p.current_bo_parsed_state = 0, None
                p.last_bo_checked_index = -1
                p.reactor_tags = {b["addon_id"] for b in s["barracks"] if b["addon_type"] == "BarracksReactor"}
                self.configure([self.unit(b["id"], b["marine_orders"], b["addon_id"], b["position"])
                    for b in s["barracks"]], allowed=s["marine_capable_ids"])
                await p.on_step(frame)
                self.assertEqual(self.submitted(), [{"action": MARINE.name, "units": [expected]}])

    async def test_all_busy_waits_without_finishing_then_uses_new_observation(self):
        p = self.player
        self.configure([self.unit(849, 1)])
        await p.on_step(2001)
        p.run_actions.assert_not_awaited()
        p._finish_current_bo.assert_not_called()
        self.assertEqual(p.current_bo_index, 0)
        self.configure([self.unit(849, 0)])
        await p.on_step(2023)
        self.assertEqual(self.submitted(), [{"action": MARINE.name, "units": [849]}])
        self.assertEqual(p.current_bo_index, 1)
        p.commander.renew_bo.assert_not_called()

    async def test_reactor_uses_second_slot_but_not_waiting_queue(self):
        p = self.player
        p.reactor_tags = {122}
        p.bo_data *= 2
        self.configure([self.unit(387, 1, 122)])
        await p.on_step(2001)
        self.assertEqual(self.submitted()[0]["units"], [387])
        self.configure([self.unit(387, 2, 122)])
        await p.on_step(2002)
        self.assertEqual(p.run_actions.await_count, 1)
        self.assertEqual(p.current_bo_index, 1)

    async def test_sequential_orders_spread_and_never_cancel_existing_orders(self):
        p = self.player
        p.bo_data *= 3
        a, b = self.unit(849), self.unit(170, position=(32, 28))
        self.configure([a, b])
        await p.on_step(2001)
        self.assertEqual(self.submitted()[0]["units"], [849])
        a.orders.append(worker_order(MARINE))  # Synthetic next-frame engine observation.
        await p.on_step(2002)
        self.assertEqual(self.submitted()[0]["units"], [170])
        b.orders.append(worker_order(MARINE))
        await p.on_step(2003)
        self.assertEqual(p.run_actions.await_count, 2)
        self.assertEqual((len(a.orders), len(b.orders), p.current_bo_index), (1, 1, 2))

    async def test_engine_ability_filter_is_not_bypassed(self):
        self.configure([self.unit(849, 1), self.unit(170)], allowed={849})
        await self.player.on_step(2001)
        self.player.run_actions.assert_not_awaited()

    async def test_resource_rejection_is_not_bypassed(self):
        self.configure([self.unit(849)])
        self.player.validate_action_and_get_suggestion.return_value = "**Action Failed**: not enough resources"
        await self.player.on_step(2001)
        self.player.run_actions.assert_not_awaited()

    async def test_addon_and_other_training_commands_keep_original_selection(self):
        for ability in (AbilityId.BUILD_REACTOR_BARRACKS, AbilityId.FACTORYTRAIN_SIEGETANK,
                        AbilityId.COMMANDCENTERTRAIN_SCV, AbilityId.BARRACKSTRAIN_MARAUDER):
            with self.subTest(ability=ability):
                p = self.player
                p.current_bo_index, p.current_bo_parsed_state, p.last_bo_checked_index = 0, None, -1
                self.configure([self.unit(849, 3), self.unit(170, position=(32, 28))], ability=ability)
                await p.on_step(2001)
                self.assertEqual(self.submitted(), [{"action": ability.name, "units": [849]}])

    async def test_nonlocal_executor_candidates_are_unchanged(self):
        p = self.player
        p.micro_agent._local_fallback_enabled.return_value = False
        p.micro_agent.select_executor = AsyncMock(return_value=[849])
        self.configure([self.unit(849, 3), self.unit(170, position=(32, 28))])
        await p.on_step(2001)
        self.assertEqual(p.micro_agent.select_executor.call_args.kwargs["available_units_text"],
                         "[849]Barracks\n[170]Barracks")
        self.assertEqual(self.submitted()[0]["units"], [849])


if __name__ == "__main__":
    unittest.main()
