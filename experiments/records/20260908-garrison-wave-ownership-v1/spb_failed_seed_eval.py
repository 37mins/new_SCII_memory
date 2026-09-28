"""Replay the failed training game's seed in standalone evaluation only.

Reuse the existing Ray evaluation hook; no frozen runtime file is edited.
"""

import numpy as np


class FailedSeedBatch:
    def __init__(self):
        self.used = False

    def randint(self, low, high, size):
        if self.used or (low, high, size) != (2**16, 2**32 - 1, 8):
            raise ValueError("This control allows exactly one eight-game evaluation batch")
        self.used = True
        return np.full(8, 43567, dtype=np.int64)


def select_seed(val_envs):
    raw = val_envs.envs
    if raw.is_train or raw.seed != 1000 or raw.num_processes != 8 or raw.reset_index != 0:
        raise ValueError("Expected eight fresh validation workers and base seed 1000")
    historical = np.random.RandomState(0).randint(0, 2**16 - 1, size=2).tolist()
    if historical != [2732, 43567]:
        raise ValueError(f"Historical training seed sequence changed: {historical}")
    raw.validation_rng = FailedSeedBatch()
    return {
        "base_seed": 1000,
        "selection": "explicit historical training game seeds, not validation RNG draws",
        "historical_base_seed": 0,
        "historical_reset_index": 1,
        "seeds": [43567] * 8,
        "extra_games": 0,
        "scope": "evaluation orchestration only; model sampling RNG is not restored",
    }


def install():
    import r4_pair_seed

    r4_pair_seed.select_seed = select_seed
    r4_pair_seed.install()
