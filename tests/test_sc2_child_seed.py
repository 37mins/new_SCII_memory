"""Exercise the actual child entry without importing SC2 or model dependencies."""
import os
import random
import sys
import traceback
import unittest
import multiprocessing as mp
import tempfile
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
            sys=sys, time=SimpleNamespace(sleep=Mock()), traceback=traceback,
        )
        old_stdout, old_stderr = sys.stdout, sys.stderr
        with tempfile.TemporaryDirectory() as log_root, \
             patch.dict(os.environ, {"SC2_RUNTIME_LOG_ROOT": log_root}, clear=True), \
             patch.object(os, "dup2"):
            if flag is not None:
                os.environ["SC2_SEED_PYTHON_RANDOM"] = flag
            try:
                entry(SimpleNamespace(env_a_class=construct, host="127.0.0.1"),
                      seed, 12345, start, ready)
            finally:
                redirected_stdout = sys.stdout
                sys.stdout, sys.stderr = old_stdout, old_stderr
                redirected_stdout.close()
        start.wait.assert_called_once_with()
        ready.set.assert_called_once_with()
        self.assertEqual(seen[0][0]["seed"], seed)
        return seen

    def test_child_stdout_and_stderr_are_written_to_worker_log(self):
        with tempfile.TemporaryDirectory() as log_root:
            entry = method(
                ROOT / "agent_system/environments/env_package/sc2_online_restored/envSC2.py",
                "SC2Env", "_start_env_a_process", os=os, random=random,
                sys=sys, time=SimpleNamespace(sleep=lambda _: None), traceback=traceback,
            )

            def run_child():
                def construct(**kwargs):
                    return SimpleNamespace(
                        run=lambda **_: (
                            print("child stdout marker"),
                            print("child stderr marker", file=sys.stderr),
                        )
                    )

                entry(
                    SimpleNamespace(env_a_class=construct, host="127.0.0.1"),
                    7, 12345, Mock(), Mock(),
                )

            with patch.dict(os.environ, {"SC2_RUNTIME_LOG_ROOT": log_root}):
                process = mp.get_context("fork").Process(target=run_child)
                process.start()
                process.join(timeout=10)

            self.assertFalse(process.is_alive())
            self.assertEqual(process.exitcode, 0)
            output_path = (
                os.path.join(log_root, "worker-output", f"sc2-env-12345-{process.pid}.log")
            )
            with open(output_path, encoding="utf-8") as output_file:
                output = output_file.read()
            self.assertIn("child stdout marker", output)
            self.assertIn("child stderr marker", output)

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
