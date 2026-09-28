"""Four paired games per feedback arm; no answer replacement or PPO."""
import functools
import json
import os
from pathlib import Path

import numpy as np

from handoff_eval import install_sampling_on_initialized_worker

GAME_SEEDS = [52620, 52620, 17089, 17089]


class PairedSeeds:
    def __init__(self, seeds=GAME_SEEDS):
        self.used = False
        self.seeds = seeds

    def randint(self, low, high, size):
        if self.used or (low, high, size) != (2**16, 2**32 - 1, len(self.seeds)):
            raise ValueError("Feedback evaluation permits exactly one declared game batch")
        self.used = True
        return np.array(self.seeds, dtype=np.int64)


def select_seeds(raw, seeds=GAME_SEEDS):
    if raw.is_train or raw.seed != 1000 or raw.num_processes != len(seeds) or raw.reset_index != 0:
        raise ValueError("Expected fresh validation workers matching the declared seed list")
    raw.validation_rng = PairedSeeds(seeds)
    return dict(seeds=seeds, games=len(seeds), extra_games=0, independent_scenarios=len(set(seeds)),
                sampling_seed_rule="same initial Torch seed=rank in both arms; not reset per call")


def install(expected_step=24, game_seeds=GAME_SEEDS,
            answer_description="online SpB; no replay or replacement", expected_temperature=0.4):
    import agent_system.environments as environments
    # Reuse only its read-only boundary capture and per-game artifact isolation.
    # Do NOT install the refinery experiment's answer intervention.
    from refinery_order_eval import install_game_observers
    install_game_observers()
    original_make = environments.make_envs

    @functools.wraps(original_make)
    def make_envs(config):
        resume_mode = "resume_path" if expected_step else "disable"
        if not config.trainer.val_only or config.trainer.resume_mode != resume_mode:
            raise ValueError("Feedback evaluation requires the declared model and val_only")
        checkpoint = Path(config.trainer.resume_from_path) if expected_step else None
        if ((checkpoint is not None and checkpoint.name != f"global_step_{expected_step}")
                or config.trainer.get("sc2_checkpoint_resume", False)):
            raise ValueError("Use declared weights and fresh paired evaluation environments")
        envs, val_envs = original_make(config)
        record = select_seeds(val_envs.envs, game_seeds)
        output = Path(os.environ["SC2_EVAL_SEEDS_LOG_PATH"]).parent
        with (output / "paired_seed_selection.json").open("x") as stream:
            json.dump(record, stream, indent=2)
        original_step = val_envs.step

        def capture_mapping(actions):
            result = original_step(actions)
            with (output / "game_worker_mapping.jsonl").open("a") as stream:
                for index, info in enumerate(result[-1]):
                    stream.write(json.dumps(dict(env=index, game_pid=info["order_diagnostic_game_pid"])) + "\n")
            return result

        val_envs.step = capture_mapping
        # Delayed until TaskRunner; never probe CUDA in the Ray worker setup hook.
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        original_validate = RayPPOTrainer._validate

        @functools.wraps(original_validate)
        def validate(trainer):
            if trainer.global_steps != expected_step:
                raise ValueError("Model load did not return at the declared global step")
            trainer.actor_rollout_wg.execute_all_sync(
                "actor_rollout_execute_with_func_generator", install_sampling_on_initialized_worker)
            proof = dict(checkpoint=str(checkpoint) if checkpoint else None, loaded_global_step=trainer.global_steps,
                         checkpoint_load_returned=bool(checkpoint), val_only=True, optimizer_updates=0,
                         feedback=os.environ["SC2_EXECUTION_FAILURE_FEEDBACK"],
                         temperature=float(config.actor_rollout_ref.rollout.val_kwargs.temperature),
                         do_sample=bool(config.actor_rollout_ref.rollout.val_kwargs.do_sample),
                         games=len(game_seeds), game_seeds=game_seeds,
                         answers=answer_description,
                         optimizer_loaded_by_existing_loader_but_not_updated=bool(checkpoint))
            if not proof["do_sample"] or proof["temperature"] != expected_temperature:
                raise ValueError(f"Expected declared stochastic T{expected_temperature} generation")
            with (output / "checkpoint_evaluation.json").open("x") as stream:
                json.dump(proof, stream, indent=2)
            print("Feedback checkpoint evaluation: " + json.dumps(proof), flush=True)
            return original_validate(trainer)

        RayPPOTrainer._validate = validate
        return envs, val_envs

    environments.make_envs = make_envs
