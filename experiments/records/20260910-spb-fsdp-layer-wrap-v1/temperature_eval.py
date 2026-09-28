"""Paired temperature evaluation of a frozen checkpoint; no PPO or gameplay patch."""
import functools
import json
import os
from pathlib import Path

import numpy as np

from handoff_eval import install_sampling_on_initialized_worker

GAME_SEEDS = [52620] * 4 + [17089] * 4


class PairedSeeds:
    def __init__(self):
        self.used = False

    def randint(self, low, high, size):
        if self.used or (low, high, size) != (2**16, 2**32 - 1, 8):
            raise ValueError("Temperature A/B permits exactly one eight-game batch")
        self.used = True
        return np.array(GAME_SEEDS, dtype=np.int64)


def select_seeds(raw):
    if raw.is_train or raw.seed != 1000 or raw.num_processes != 8 or raw.reset_index != 0:
        raise ValueError("Expected eight fresh validation workers, base seed1000")
    raw.validation_rng = PairedSeeds()
    return {"seeds": GAME_SEEDS, "selection": "L6/L7 existing step18/20 seeds, four trials each",
            "sampling_seed_rule": "same initial Torch seed=rank in both arms; not reset per call",
            "independent_scenarios": 2, "games": 8, "extra_games": 0}


def install():
    import agent_system.environments as environments

    original_make = environments.make_envs

    @functools.wraps(original_make)
    def make_envs(config):
        if not config.trainer.val_only or config.trainer.resume_mode != "resume_path":
            raise ValueError("Temperature comparison is checkpoint evaluation only")
        checkpoint = Path(config.trainer.resume_from_path)
        if checkpoint.name != "global_step_24" or config.trainer.get("sc2_checkpoint_resume", False):
            raise ValueError("Use step24 weights with fresh paired evaluation environments")
        envs, val_envs = original_make(config)
        record = select_seeds(val_envs.envs)
        output = Path(os.environ["SC2_EVAL_SEEDS_LOG_PATH"]).parent
        with (output / "paired_seed_selection.json").open("x") as stream:
            json.dump(record, stream, indent=2)

        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        original_validate = RayPPOTrainer._validate

        @functools.wraps(original_validate)
        def validate(trainer):
            if trainer.global_steps != 24:
                raise ValueError("Checkpoint load did not return at step24")
            # Reuse the tested generation-RNG pairing RPC after model/checkpoint load.
            trainer.actor_rollout_wg.execute_all_sync(
                "actor_rollout_execute_with_func_generator", install_sampling_on_initialized_worker)
            proof = {"checkpoint": str(checkpoint), "loaded_global_step": trainer.global_steps,
                     "checkpoint_load_returned": True, "val_only": True,
                     "temperature": float(config.actor_rollout_ref.rollout.val_kwargs.temperature),
                     "top_p": float(config.actor_rollout_ref.rollout.val_kwargs.top_p),
                     "do_sample": bool(config.actor_rollout_ref.rollout.val_kwargs.do_sample),
                     "games": 8, "game_seeds": GAME_SEEDS,
                     "optimizer_loaded_by_existing_loader_but_not_updated": True}
            if not proof["do_sample"] or proof["temperature"] not in (0.4, 1.0):
                raise ValueError("Expected stochastic temperature arm")
            with (output / "checkpoint_evaluation.json").open("x") as stream:
                json.dump(proof, stream, indent=2)
            print("Temperature checkpoint evaluation: " + json.dumps(proof), flush=True)
            return original_validate(trainer)

        RayPPOTrainer._validate = validate
        return envs, val_envs

    environments.make_envs = make_envs
