"""Experiment-only pairing after a legacy model/optimizer load, not SC2 resume."""
import functools
import json
import random
from pathlib import Path

import numpy as np


def pair_driver(trainer):
    import torch

    if trainer.global_steps != 4 or trainer.envs.envs.reset_index != 0:
        raise ValueError('Expected step4 model/optimizer with fresh game environments')
    # This snapshot was saved after its last yield, before StopIteration.
    # Close only that exhausted tail so eight epochs mean eight new batches.
    try:
        next(iter(trainer.train_dataloader))
    except StopIteration:
        pass
    else:
        raise ValueError('Expected an exhausted saved epoch; refusing to discard a sample')
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    preview = np.random.RandomState(0)
    return {
        'loaded_global_step': 4,
        'checkpoint': trainer.config.trainer.resume_from_path,
        'model_optimizer_load_returned': True,
        'sc2_state_restored': False,
        'exhausted_saved_epoch_closed_without_new_sample': True,
        'driver_seed': 0,
        'game_reset_index': 0,
        'game_seed_per_new_step': preview.randint(0, 2**16 - 1, size=8).tolist(),
        'note': 'New paired experiment; not an uninterrupted historical continuation. '
                'Per-rank RNG is restored by the existing checkpoint loader.',
    }


def install():
    # Ray executes setup hooks before assigning this worker's visible GPUs.
    # Importing the trainer here probes CUDA through verl.utils.device.
    import agent_system.environments as environments

    original_make = environments.make_envs

    @functools.wraps(original_make)
    def make_envs(config):
        # Called by TaskRunner.run, not by the GPU worker startup hook.
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        original = RayPPOTrainer._load_checkpoint

        @functools.wraps(original)
        def load_checkpoint(trainer):
            original(trainer)
            proof = pair_driver(trainer)
            path = Path(trainer.config.trainer.default_local_dir) / 'gamma_pairing.json'
            with path.open('x') as stream:
                json.dump(proof, stream, indent=2)
            print('Gamma A/B checkpoint pairing: ' + json.dumps(proof), flush=True)

        RayPPOTrainer._load_checkpoint = load_checkpoint
        return original_make(config)

    environments.make_envs = make_envs
