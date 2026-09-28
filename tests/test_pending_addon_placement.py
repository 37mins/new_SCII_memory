"""CPU regression for pending production-building orders; SC2 placement is stubbed."""
import ast
import random
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from s2clientprotocol import data_pb2, raw_pb2
from sc2.game_data import AbilityData
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import UnitOrder


ROOT = Path(__file__).resolve().parents[1]
PLAYER = ROOT / "agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/players/base_player.py"
FACTORY = Point2((26.5, 30.5))
DEPOT = Point2((28.5, 30.5))
ADDON = FACTORY.offset((2.5, -0.5))


def worker_order(ability=AbilityId.TERRANBUILD_FACTORY, target=FACTORY):
    """Use the installed library's real protobuf-to-order conversion."""
    proto = raw_pb2.UnitOrder(ability_id=ability.value)
    if isinstance(target, Point2):
        proto.target_world_space_pos.x, proto.target_world_space_pos.y = target
    else:
        proto.target_unit_tag = target
    data = AbilityData(None, data_pb2.AbilityData(ability_id=ability.value))
    bot = SimpleNamespace(game_data=SimpleNamespace(abilities={ability.value: data}))
    return UnitOrder.from_proto(proto, bot)


class PendingAddonPlacementTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # Execute the production method without loading the LLM/game entrypoints.
        tree = ast.parse(PLAYER.read_text())
        player = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BasePlayer")
        node = next(n for n in player.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "find_placement")
        scope = dict(AbilityId=AbilityId, UnitTypeId=UnitTypeId, Point2=Point2, random=random)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(PLAYER), "exec"), scope)
        cls.find_placement = staticmethod(scope["find_placement"])

    def bot(self, orders=(), structures=()):
        return SimpleNamespace(
            structures=SimpleNamespace(of_type=lambda _: structures),
            workers=[SimpleNamespace(orders=list(orders))],
            can_place_single=AsyncMock(return_value=True),
            client=SimpleNamespace(_query_building_placement_fast=AsyncMock(
                side_effect=lambda ability, points: [True] * len(points))),
            _is_blocking_mining_path=lambda _: False,
        )

    async def place(self, bot, **kwargs):
        return await self.find_placement(
            bot, AbilityId.TERRANBUILD_SUPPLYDEPOT, DEPOT, **{"max_distance": 0, **kwargs})

    async def test_recorded_factory_order_blocks_depot_before_factory_appears(self):
        order = worker_order()
        self.assertIsInstance(order.target, Point2)
        self.assertEqual(order.ability.exact_id, AbilityId.TERRANBUILD_FACTORY)
        self.assertIsNone(await self.place(self.bot([order])))

    async def test_all_three_production_buildings_including_queued_orders(self):
        for ability in (AbilityId.TERRANBUILD_BARRACKS, AbilityId.TERRANBUILD_FACTORY,
                        AbilityId.TERRANBUILD_STARPORT):
            with self.subTest(ability=ability):
                orders = [worker_order(AbilityId.MOVE_MOVE), worker_order(ability)]
                self.assertIsNone(await self.place(self.bot(orders)))

    async def test_spiral_search_filters_conflict_and_still_finds_alternative(self):
        bot = self.bot([worker_order()])
        result = await self.place(bot, max_distance=12, random_alternative=False)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.distance_to(ADDON), 4.5)
        for call in bot.client._query_building_placement_fast.call_args_list:
            self.assertTrue(all(p.distance_to(ADDON) >= 4.5 for p in call.args[1]))

    async def test_no_stale_reservation_after_order_disappears(self):
        bot = self.bot([worker_order()])
        self.assertIsNone(await self.place(bot))
        bot.workers[0].orders.clear()
        self.assertEqual(await self.place(bot), DEPOT)

    async def test_other_abilities_and_unit_tag_targets_do_not_reserve(self):
        orders = [worker_order(AbilityId.TERRANBUILD_SUPPLYDEPOT),
                  worker_order(AbilityId.MOVE_MOVE), worker_order(target=90963969)]
        self.assertIsInstance(orders[-1].target, int)
        self.assertEqual(await self.place(self.bot(orders)), DEPOT)

    async def test_existing_factory_protection_is_unchanged(self):
        for orders in ([], [worker_order()]):
            self.assertIsNone(await self.place(self.bot(orders, [SimpleNamespace(position=FACTORY)])))

    async def test_explicit_opt_out_is_unchanged(self):
        self.assertEqual(await self.place(
            self.bot([worker_order()]), avoid_production_addons=False), DEPOT)

    async def test_no_pending_production_keeps_choice_and_random_state(self):
        bot = self.bot()
        state = random.getstate()
        self.assertEqual(await self.place(bot), DEPOT)
        self.assertEqual(random.getstate(), state)
        bot.client._query_building_placement_fast.assert_not_called()

    async def test_engine_rejection_is_not_bypassed(self):
        bot = self.bot([worker_order()])
        bot.can_place_single.return_value = False
        bot.client._query_building_placement_fast.side_effect = lambda ability, points: [False] * len(points)
        self.assertIsNone(await self.place(bot, max_distance=12))


if __name__ == "__main__":
    unittest.main()
