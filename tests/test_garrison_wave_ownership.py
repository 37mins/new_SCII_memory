"""Run the actual garrison method on CPU; only SC2 observations/orders are doubles."""
import contextlib
import copy
import io
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.units import Units

from test_r6_natural_termination import ARENA, method


PLAYER = ARENA / "players/script_player.py"


class GarrisonWaveOwnershipTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.manage = staticmethod(method(
            PLAYER, "ScriptPlayer", "manage_garrison",
            UnitTypeId=UnitTypeId, Point2=Point2, math=math))

    def setUp(self):
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)
        self.reset_player()

    def reset_player(self):
        p = self.player = SimpleNamespace(
            total_attack_groups={}, active_scout_unit_tag=None,
            game_info=SimpleNamespace(map_center=Point2((76, 74))),
            GARRISON_PERPENDICULAR_DISTANCE=15,
            GARRISON_EXTENSION_DISTANCE=10, GARRISON_CHECK_RADIUS=4,
            GARRISON_DEFENSE_ZONE_RADIUS=30,
            _clamp_position_to_map_bounds=lambda point: point,
            _distance_units_to_pos=lambda units, point: [
                unit.position.distance_to(point) for unit in units],
        )
        p.units = Units([], p)
        p.structures = Units([SimpleNamespace(
            tag=60555265, type_id=UnitTypeId.ORBITALCOMMAND,
            position=Point2((25.5, 45.5)), is_ready=True)], p)
        p.get_enemy_units_near_structures = Mock(return_value=Units([], p))

    def unit(self, tag=17301515, kind=UnitTypeId.VIKINGFIGHTER, position=(29.21, 73.63), idle=True):
        point = Point2(position)
        unit = SimpleNamespace(tag=tag, type_id=kind, position=point,
                               can_attack=True, is_idle=idle,
                               distance_to=point.distance_to, move=Mock())
        self.player.units.append(unit)
        return unit

    def group(self, wave, *units, state="GATHERING"):
        self.player.total_attack_groups[wave] = dict(
            unit_tags={unit.tag for unit in units}, state=state,
            rally_point=Point2((29.21, 73.63)), min_units_to_launch=5)

    async def test_idle_viking_at_recorded_rally_is_not_pulled_back(self):
        viking = self.unit()
        marine = self.unit(2, UnitTypeId.MARINE, position=(26, 46))
        self.group(11, viking)
        groups = copy.deepcopy(self.player.total_attack_groups)
        await self.manage(self.player)
        viking.move.assert_not_called()
        marine.move.assert_called_once()
        self.assertEqual(self.player.total_attack_groups, groups)

    async def test_attacking_wave_members_are_also_protected_when_idle(self):
        unit = self.unit()
        self.group(11, unit, state="ATTACKING")
        await self.manage(self.player)
        unit.move.assert_not_called()

    async def test_all_waves_and_unit_types_are_excluded(self):
        viking = self.unit()
        marine = self.unit(2, UnitTypeId.MARINE)
        tank = self.unit(3, UnitTypeId.SIEGETANK)
        self.group(11, viking, marine)
        self.group(12, tank, state="ATTACKING")
        self.player.total_attack_groups[12]["unit_tags"].add(99999)  # No live unit with this tag.
        await self.manage(self.player)
        for unit in (viking, marine, tank):
            unit.move.assert_not_called()

    async def test_repeated_checks_do_not_create_new_orders_for_members(self):
        unit = self.unit()
        self.group(11, unit)
        for _ in range(3):
            await self.manage(self.player)
        unit.move.assert_not_called()

    async def test_no_wave_keeps_original_garrison_destination(self):
        unit = self.unit()
        await self.manage(self.player)
        expected = Point2((25.5, 45.5)).towards(Point2((76, 74)), 15)
        unit.move.assert_called_once_with(expected)

    async def test_release_by_member_or_wave_removal_restores_garrison(self):
        for delete_wave in (False, True):
            with self.subTest(delete_wave=delete_wave):
                self.reset_player()
                unit = self.unit()
                self.group(11, unit)
                await self.manage(self.player)
                unit.move.assert_not_called()
                if delete_wave:
                    self.player.total_attack_groups.clear()
                else:
                    self.player.total_attack_groups[11]["unit_tags"].remove(unit.tag)
                await self.manage(self.player)
                unit.move.assert_called_once()

    async def test_threat_and_existing_worker_idle_radius_filters_remain(self):
        available = self.unit(1, position=(26, 46))
        worker = self.unit(2, UnitTypeId.SCV)
        busy = self.unit(3, idle=False)
        far = self.unit(4, position=(100, 100))
        self.player.get_enemy_units_near_structures.return_value = Units([object()], self.player)
        await self.manage(self.player)
        available.move.assert_not_called()
        self.player.get_enemy_units_near_structures.return_value = Units([], self.player)
        await self.manage(self.player)
        available.move.assert_called_once()
        for unit in (worker, busy, far):
            unit.move.assert_not_called()

    async def test_scout_handling_is_not_changed_by_this_fix(self):
        unit = self.unit()
        self.player.active_scout_unit_tag = unit.tag
        await self.manage(self.player)
        unit.move.assert_called_once()  # Intentionally retains old garrison/scout behavior.


if __name__ == "__main__":
    unittest.main()
