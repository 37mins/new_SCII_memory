"""Exercise the actual child entry without importing SC2 or model dependencies."""
import os
import random
import traceback
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_r6_natural_termination import ROOT, method


class ChildSeedTest(unittest.TestCase):
    def setUp(self):
        self.state = random.getstate()
        self.addCleanup(random.setstate, self.state)

    def run_child(self, flag, seed):
        seen = []

        def construct(**kwargs):
            seen.append((kwargs, [random.random() for _ in range(5)]))
            return SimpleNamespace(run=lambda **_: seen.append(random.random()))

        start, ready = Mock(), Mock()
        entry = method(
            ROOT / "agent_system/environments/env_package/sc2_online_restored/envSC2.py",
            "SC2Env", "_start_env_a_process", os=os, random=random,
            time=SimpleNamespace(sleep=Mock()), traceback=traceback,
        )
        with patch.dict(os.environ, {}, clear=True):
            if flag is not None:
                os.environ["SC2_SEED_PYTHON_RANDOM"] = flag
            entry(SimpleNamespace(env_a_class=construct, host="127.0.0.1"),
                  seed, 12345, start, ready)
        start.wait.assert_called_once_with()
        ready.set.assert_called_once_with()
        self.assertEqual(seen[0][0]["seed"], seed)
        return seen

    def test_enabled_repeats_despite_different_parent_state(self):
        random.seed(19)
        first = self.run_child("1", 2621640384)
        random.seed(81)
        self.assertEqual(first, self.run_child("1", 2621640384))
        self.assertNotEqual(first, self.run_child("1", 2621640385))

    def test_default_and_disabled_leave_random_stream_unchanged(self):
        for flag in (None, "0"):
            random.seed(19)
            expected = [random.random() for _ in range(6)]
            random.seed(19)
            seen = self.run_child(flag, 2621640384)
            self.assertEqual(seen[0][1] + [seen[1]], expected)


if __name__ == "__main__":
    unittest.main()
