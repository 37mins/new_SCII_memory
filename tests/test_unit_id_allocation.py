"""Exercise actual allocator and integer consumers without loading SC2 or a model."""
import asyncio
import contextlib
import io
import json
import random
import re
import signal
from types import SimpleNamespace
import unittest

from test_r6_natural_termination import ARENA, method


class UnitIdAllocationTest(unittest.TestCase):
    def setUp(self):
        self.allocate = method(ARENA / "players/base_player.py", "BasePlayer", "tag_to_id")
        self.reverse = method(ARENA / "players/base_player.py", "BasePlayer", "id_to_tag")
        self.state = SimpleNamespace(_tag_to_id={}, _id_to_tag={})

        def deadline(*_):
            raise TimeoutError("unit ID allocation did not terminate")

        self.previous_handler = signal.signal(signal.SIGALRM, deadline)
        signal.alarm(5)

    def tearDown(self):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self.previous_handler)

    def test_first_thousand_match_legacy_including_collisions(self):
        tags = random.Random(0).sample(range(2**32, 2**33), 1000)
        expected = {}
        for tag in tags:
            old_id = tag % 1000
            while old_id in expected:
                old_id = (old_id + 1) % 1000
            expected[old_id] = tag
            self.assertEqual(self.allocate(self.state, tag), old_id)
        self.assertEqual(self.state._id_to_tag, expected)

    def test_full_collision_table_extends_without_recycling(self):
        for i in range(10000):
            tag = (i + 1) * 1000
            self.assertEqual(self.allocate(self.state, tag), i)
            self.assertEqual(self.reverse(self.state, i), tag)
        self.assertEqual(len(self.state._id_to_tag), 10000)
        for i in (0, 1, 998, 999, 1000, 1001, 9999):
            self.assertEqual(self.allocate(self.state, (i + 1) * 1000), i)
        self.assertEqual(len(self.state._id_to_tag), 10000)

    def test_repeated_tag_does_not_consume_a_slot_at_boundary(self):
        for tag in range(999):
            self.allocate(self.state, tag)
        self.assertEqual(self.allocate(self.state, 0), 0)
        self.assertEqual(len(self.state._id_to_tag), 999)
        self.assertEqual(self.allocate(self.state, 999), 999)
        self.assertEqual(self.allocate(self.state, 0), 0)
        self.assertEqual(self.allocate(self.state, 1000), 1000)
        self.assertEqual(len(self.state._id_to_tag), 1001)

    def test_new_game_mapping_starts_fresh(self):
        for tag in range(1001):
            self.allocate(self.state, tag)
        fresh = SimpleNamespace(_tag_to_id={}, _id_to_tag={})
        self.assertEqual(self.allocate(fresh, 1000), 0)
        self.assertEqual(self.reverse(self.state, 1000), 1000)

    def test_extended_id_survives_actual_executor_consumers(self):
        select = method(ARENA / "agents/micro_agent.py", "MicroAgent",
                        "_fallback_select_executor", re=re)
        build = method(ARENA / "agents/micro_agent.py", "MicroAgent",
                       "get_final_action", json=json)
        for tag in range(10000):
            self.allocate(self.state, tag)
        tag = 4412669957
        short_id = self.allocate(self.state, tag)
        self.assertEqual(short_id, 10000)
        with contextlib.redirect_stdout(io.StringIO()):
            selected = select(None, f"[{short_id}]Barracks\nHealth: 1000/1000")
        action = asyncio.run(build(None, "BARRACKSTRAIN_MARINE", selected,
                                   "", "", "", "None"))
        self.assertEqual(action, [{"action": "BARRACKSTRAIN_MARINE", "units": [10000]}])
        self.assertEqual(self.reverse(self.state, action[0]["units"][0]), tag)


if __name__ == "__main__":
    unittest.main()
