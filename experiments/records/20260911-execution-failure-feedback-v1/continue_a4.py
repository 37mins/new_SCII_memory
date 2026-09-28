"""A4 model/optimizer continuation with an explicit fresh-game seed boundary.

Only the experiment entry installs this hook. No answers, losses or game rules
are changed. The legacy checkpoint lacks driver/SC2 state, so this is not a
bitwise continuation of its games. Per-rank state uses the original loader.
"""
import functools
import json
from pathlib import Path

import numpy as np

PARENT_NAME = '20260911-execution-failure-feedback-v1-spb2-original-l6-g2-A-rl4-4gpu-s0-r2'
PARENT_SHA = '361cb7ae7fd86fe8811d6fcb6946d0241bf93e11'
PAST_SEEDS = [2732, 43567, 42613, 52416]
NEXT_SEEDS = [45891, 21243, 30403, 32103]
SCENE_SEED = 1098341402
COVERAGE_SEEDS = [SCENE_SEED, 42613, 17089, 3236598124]
# September15 approved sampling-only comparison; first four predeclared screening
# scenes versus one old/one new mixed-outcome scene. No outcome filtering in RL.
SIGNAL_SCENE_SEEDS = {
    'A': [3974528993, 1183530000, 3446496864, 2996859384],
    'B': [1098341402, 3446496864, 1098341402, 3446496864],
}


class SingleTrainingSeed:
    """One explicitly selected scene, not a draw from the legacy 16-bit RNG."""
    def __init__(self, seed):
        self.seed = seed
        self.used = False

    def randint(self, low, high, size):
        if self.used or (low, high, size) != (0, 2**16-1, 1):
            raise ValueError('Scene coverage permits exactly one eight-game batch')
        self.used = True
        return np.array([self.seed], dtype=np.int64)


class CoverageTrainingSeeds:
    """Four declared scenes, not samples from the legacy 16-bit distribution."""
    def __init__(self, seeds=None):
        self.seeds = iter(COVERAGE_SEEDS if seeds is None else seeds)

    def randint(self, low, high, size):
        if (low, high, size) != (0, 2**16-1, 1):
            raise ValueError('Coverage training requires one seed for each eight-game group')
        try:
            seed = next(self.seeds)
        except StopIteration:
            raise ValueError('Coverage training permits exactly four game batches') from None
        return np.array([seed], dtype=np.int64)


def restore_boundary(trainer, *, scene=False, coverage=False, coverage_seeds=None):
    if scene and coverage:
        raise ValueError('Select either the old one-scene run or the four-scene comparison')
    selected_seeds = COVERAGE_SEEDS if coverage_seeds is None else coverage_seeds
    if coverage_seeds is not None and (not coverage or len(coverage_seeds) != 4):
        raise ValueError('Explicit coverage requires four game seeds')
    from gamma_ab import pair_driver
    cfg = trainer.config.trainer
    parent = Path(cfg.resume_from_path).parent.parent
    manifest = json.loads((parent / 'manifest.json').read_text())
    result = json.loads((parent / '.done').read_text())
    seeds = [json.loads(line)['seeds'] for line in (parent / 'training_seeds.jsonl').read_text().splitlines()]
    raw = trainer.envs.envs
    if (parent.name != PARENT_NAME or manifest['source_commit'] != PARENT_SHA or manifest['dirty']
            or manifest['world_size'] != 4 or result['status'] != 'completed'
            or result['completed_steps'] != 4 or not result['checkpoint_complete']
            or seeds != [[seed]*8 for seed in PAST_SEEDS]
            or cfg.val_only or cfg.total_epochs != (1 if scene else 4) or cfg.get('sc2_checkpoint_resume', False)
            or not raw.is_train or raw.seed != 0 or raw.env_num != 1 or raw.group_n != 8):
        raise ValueError('Expected the frozen complete A-r2 and four new training batches')
    proof = pair_driver(trainer)  # existing exhausted-tail check and explicit driver seed0
    if scene:
        raw.training_rng = SingleTrainingSeed(SCENE_SEED)
    elif coverage:
        raw.training_rng = CoverageTrainingSeeds(selected_seeds)
    else:
        rng = np.random.RandomState(0)
        assert rng.randint(0, 2**16-1, size=4).tolist() == PAST_SEEDS
        raw.training_rng.set_state(rng.get_state())
        assert rng.randint(0, 2**16-1, size=4).tolist() == NEXT_SEEDS
    proof.update(game_seed_per_new_step=[SCENE_SEED] if scene else selected_seeds if coverage else NEXT_SEEDS,
                 game_seed_sequence_offset=None if scene or coverage else 4,
                 seed_selection='explicit coverage table' if coverage else 'explicit diagnostic scene' if scene else 'legacy RNG offset4',
                 parent_source_commit=PARENT_SHA, parent_run=parent.name,
                 new_run_global_steps=[5] if scene else [5, 6, 7, 8], game_reset_index=0,
                 note='Original model/optimizer/per-rank RNG restored; fresh games, '
                      'declared driver seed0. Not bitwise SC2 resume; no answer intervention.')
    if cfg.get('sc2_signal_max_batches', 0):
        proof['game_seed_per_collection'] = proof.pop('game_seed_per_new_step')
        proof['planned_max_new_global_steps'] = proof.pop('new_run_global_steps')
        proof['note'] += (' Zero-task groups may consume a seed without a global step.'
                          if cfg.get('sc2_skip_zero_signal', False)
                          else ' Zero-task skipping is disabled; all four groups are retained.')
    return proof


def install(*, scene=False, coverage=False, coverage_seeds=None):
    # Delay trainer/CUDA-related imports until TaskRunner, as in gamma_ab.
    import agent_system.environments as environments
    original_make = environments.make_envs

    @functools.wraps(original_make)
    def make_envs(config):
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        original_load = RayPPOTrainer._load_checkpoint

        @functools.wraps(original_load)
        def load_checkpoint(trainer):
            original_load(trainer)
            proof = restore_boundary(trainer, scene=scene, coverage=coverage, coverage_seeds=coverage_seeds)
            output = Path(config.trainer.default_local_dir) / 'a4_continuation.json'
            with output.open('x') as stream:
                json.dump(proof, stream, indent=2)
            print('A4 continuation load returned: ' + json.dumps(proof), flush=True)

        RayPPOTrainer._load_checkpoint = load_checkpoint
        return original_make(config)

    environments.make_envs = make_envs


def install_scene():
    install(scene=True)


def install_coverage():
    install(coverage=True)


def install_signal_A():
    install(coverage=True, coverage_seeds=SIGNAL_SCENE_SEEDS['A'])


def install_signal_B():
    install(coverage=True, coverage_seeds=SIGNAL_SCENE_SEEDS['B'])
