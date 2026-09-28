"""Select the historical fourth evaluation seed without extra games or core edits."""

import functools
import json
import os
from pathlib import Path


def select_seed(val_envs):
    import numpy as np

    raw = val_envs.envs
    if raw.is_train or raw.seed != 1000 or raw.num_processes != 1 or raw.reset_index != 0:
        raise ValueError("This paired evaluation requires one fresh validation worker, seed=1000")
    raw.validation_rng.randint(2**16, 2**32 - 1, size=3)
    probe = np.random.RandomState()
    probe.set_state(raw.validation_rng.get_state())
    actual = int(probe.randint(2**16, 2**32 - 1))
    if actual != 2621640384:
        raise ValueError(f"Unexpected paired game seed: {actual}")
    return {"base_seed": 1000, "skipped_seed_draws": 3, "game_seed": actual,
            "extra_games": 0, "scope": "evaluation orchestration only"}


def install():
    # Ray 2.48.0 worker_process_setup_hook runs inside TaskRunner too; patching
    # only the driver would not reach the process which constructs the envs.
    import agent_system.environments as environments

    original = environments.make_envs

    @functools.wraps(original)
    def make_envs(config):
        if not config.trainer.val_only or config.trainer.resume_mode != "disable":
            raise ValueError("Paired seed hook is evaluation-only, original model only")
        envs, val_envs = original(config)
        record = select_seed(val_envs)
        output = Path(os.environ["SC2_EVAL_SEEDS_LOG_PATH"]).parent
        with (output / "paired_seed_selection.json").open("x") as stream:
            json.dump(record, stream, indent=2)
        print(f"Paired evaluation seed selection: {record}", flush=True)
        return envs, val_envs

    environments.make_envs = make_envs
