import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


path = Path(__file__).resolve().parents[1] / "experiments/records/20260906-integrated-fixes-v1/r4_pair_seed.py"
spec = importlib.util.spec_from_file_location("r4_pair_seed", path)
hook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)


class PairedSeedTests(unittest.TestCase):
    def test_only_validation_rng_advances_to_fourth_seed(self):
        raw = SimpleNamespace(is_train=False, seed=1000, num_processes=1,
                              reset_index=0, validation_rng=np.random.RandomState(1000),
                              training_rng=np.random.RandomState(0))
        record = hook.select_seed(SimpleNamespace(envs=raw))
        self.assertEqual(record["game_seed"], 2621640384)
        self.assertEqual(raw.validation_rng.randint(2**16, 2**32 - 1, size=1).tolist(), [2621640384])
        self.assertEqual(raw.training_rng.randint(0, 2**16 - 1), np.random.RandomState(0).randint(0, 2**16 - 1))
        self.assertEqual(raw.reset_index, 0)

    def test_reject_training_or_multiple_games(self):
        for training, count, reset in [(True, 1, 0), (False, 8, 0), (False, 1, 1)]:
            with self.subTest(training=training, count=count, reset=reset):
                raw = SimpleNamespace(is_train=training, seed=1000, num_processes=count, reset_index=reset)
                with self.assertRaises(ValueError):
                    hook.select_seed(SimpleNamespace(envs=raw))
