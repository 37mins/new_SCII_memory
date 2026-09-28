"""CPU-only regression for the SC2 observation ability-query boundary."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock


ROOT = Path(__file__).resolve().parents[1]
PLAYER = ROOT / "agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/players/base_player.py"


class EmptyAbilitiesTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # Execute the real method without loading SC2, a model or GPU dependencies.
        player = next(node for node in ast.parse(PLAYER.read_text()).body
                      if isinstance(node, ast.ClassDef) and node.name == "BasePlayer")
        method = next(node for node in player.body
                      if isinstance(node, ast.AsyncFunctionDef) and node.name == "abilities_to_text")
        scope = {"Units": list, "TerranAbility": {"TRAIN_MARINE": {"enabled": True}}}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(PLAYER), "exec"), scope)
        cls.abilities_to_text = staticmethod(scope["abilities_to_text"])

    def bot(self, abilities=()):
        async def query(units, ignore_resource_requirements):
            # Match the observed python-sc2 boundary, not an empty-list fallback.
            assert units
            return [list(abilities) for _ in units]

        return SimpleNamespace(get_available_abilities=AsyncMock(side_effect=query),
                               miner_units=[], _id_to_abilities={}, tag_to_id=str)

    async def test_no_units_does_not_query_sc2(self):
        bot = self.bot()
        self.assertEqual(await self.abilities_to_text(bot, []), "[Empty]")
        bot.get_available_abilities.assert_not_awaited()
        self.assertEqual(bot._id_to_abilities, {})

    async def test_only_unfinished_structures_does_not_query_sc2(self):
        bot = self.bot()
        structure = SimpleNamespace(build_progress=0.5, name="CommandCenter", tag=1)
        self.assertEqual(await self.abilities_to_text(bot, [structure]), "[Empty]")
        bot.get_available_abilities.assert_not_awaited()

    async def test_mixed_units_preserve_ready_query_and_text(self):
        ability = SimpleNamespace(name="TRAIN_MARINE")
        bot = self.bot([ability])
        ready = SimpleNamespace(build_progress=1.0, name="Barracks", tag=2)
        unfinished = SimpleNamespace(build_progress=0.9, name="Barracks", tag=3)
        self.assertEqual(await self.abilities_to_text(bot, [unfinished, ready]),
                         "Barracks[2]: TRAIN_MARINE")
        bot.get_available_abilities.assert_awaited_once_with(
            [ready], ignore_resource_requirements=True)
        self.assertEqual(bot._id_to_abilities, {"2": ["TRAIN_MARINE"]})

    async def test_ready_unit_without_abilities_keeps_empty_convention(self):
        bot = self.bot()
        ready = SimpleNamespace(build_progress=1.0, name="Barracks", tag=2)
        self.assertEqual(await self.abilities_to_text(bot, [ready]), "[Empty]")
        bot.get_available_abilities.assert_awaited_once()
        self.assertEqual(bot._id_to_abilities, {"2": []})

    async def test_nonempty_query_errors_are_not_swallowed(self):
        bot = self.bot()
        bot.get_available_abilities.side_effect = RuntimeError("SC2 query failed")
        ready = SimpleNamespace(build_progress=1.0, name="Barracks", tag=2)
        with self.assertRaisesRegex(RuntimeError, "SC2 query failed"):
            await self.abilities_to_text(bot, [ready])


if __name__ == "__main__":
    unittest.main()
