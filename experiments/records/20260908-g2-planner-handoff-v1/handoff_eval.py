"""Four-game paired evaluation orchestration; no training or gameplay patch."""
import functools
import hashlib
import json
import os
from pathlib import Path

import numpy as np

GAME_SEEDS = [2732, 2732, 43567, 43567]


class PairedSeedBatch:
    def __init__(self):
        self.used = False

    def randint(self, low, high, size):
        if self.used or (low, high, size) != (2**16, 2**32 - 1, 4):
            raise ValueError("Handoff A/B permits exactly one four-game evaluation batch")
        self.used = True
        return np.array(GAME_SEEDS, dtype=np.int64)


def select_seed(val_envs):
    raw = val_envs.envs
    if raw.is_train or raw.seed != 1000 or raw.num_processes != 4 or raw.reset_index != 0:
        raise ValueError("Expected four fresh validation workers and base seed 1000")
    raw.validation_rng = PairedSeedBatch()
    return dict(base_seed=1000, seeds=GAME_SEEDS, extra_games=0,
                selection="explicit historical training seeds, paired by env_index",
                sampling_seed_rule="once before first HF generation: 0 + distributed rank",
                scope="evaluation orchestration only; not restored historical model RNG")


def seeded_generation(original):
    @functools.wraps(original)
    def generate(self, prompts):
        import torch

        if not getattr(self, "_handoff_sampling_seeded", False):
            rank = torch.distributed.get_rank()
            torch.manual_seed(rank)
            record = dict(rank=rank, seed=rank, pid=os.getpid(),
                          cpu_rng_sha256=hashlib.sha256(bytes(torch.get_rng_state().tolist())).hexdigest(),
                          cuda_rng_sha256=hashlib.sha256(bytes(torch.cuda.get_rng_state().tolist())).hexdigest())
            path = Path(os.environ["SC2_EVAL_SEEDS_LOG_PATH"]).parent / f"sampling-rank{rank}.json"
            with path.open("x") as stream:
                json.dump(record, stream, indent=2)
            print("Handoff paired sampling: " + json.dumps(record), flush=True)
            self._handoff_sampling_seeded = True
        return original(self, prompts)
    return generate


def install_sampling_on_initialized_worker(_worker):
    # Invoked by the existing Worker RPC only after init_model has completed.
    from verl.workers.rollout.hf_rollout import HFRollout

    HFRollout.generate_sequences = seeded_generation(HFRollout.generate_sequences)


def install():
    import r4_pair_seed
    import agent_system.environments as environments

    r4_pair_seed.select_seed = select_seed
    r4_pair_seed.install()
    original_make_envs = environments.make_envs

    @functools.wraps(original_make_envs)
    def make_envs(config):
        result = original_make_envs(config)
        # Only TaskRunner constructs these environments. Do not import rollout
        # or CUDA-detecting trainer modules in every Ray worker's setup hook.
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        original_validate = RayPPOTrainer._validate

        @functools.wraps(original_validate)
        def validate(trainer):
            trainer.actor_rollout_wg.execute_all_sync(
                "actor_rollout_execute_with_func_generator", install_sampling_on_initialized_worker)
            return original_validate(trainer)

        RayPPOTrainer._validate = validate
        return result

    environments.make_envs = make_envs
