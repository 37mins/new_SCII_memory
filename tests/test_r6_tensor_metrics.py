"""Exercise the real tensor/metrics boundary with pinned CPU dependencies."""

import importlib.util
from pathlib import Path
import runpy
import unittest


class PolicyTensorMetricsTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"),
                         "requires pinned Linux Torch/NumPy runtime")
    def test_filtered_episode_lengths_reach_real_metrics(self):
        runpy.run_path(str(Path(__file__).resolve().parents[1] /
            "experiments/records/20260905-r6-terminal-policy-credit-v1/verify_tensor_path.py"))
