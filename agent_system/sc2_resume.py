"""SC2 checkpoint-boundary state, separate from per-rank FSDP RNG state.

Only closed-game, memory-off resumes are supported. Legacy import verifies a
completed run's entire seed ledger; it does not guess a mid-run RNG position.
"""
import json
from pathlib import Path

import numpy as np


def environment_state(manager):
    if manager.memory is not None:
        raise ValueError('SC2 checkpoint resume currently requires memory=off')
    if manager._terminal_by_env and not all(manager._terminal_by_env):
        raise ValueError('Cannot checkpoint a live SC2 game')
    raw = manager.envs
    return {
        'seed': raw.seed, 'env_num': raw.env_num, 'group_n': raw.group_n,
        'is_train': raw.is_train, 'reset_index': raw.reset_index,
        'training_rng': raw.training_rng.get_state(),
        'validation_rng': raw.validation_rng.get_state(),
        'episode_reset_index': manager._episode_reset_index,
        'step_index': manager._step_index,
    }


def restore_environment(manager, state):
    raw = manager.envs
    if manager.memory is not None:
        raise ValueError('SC2 checkpoint resume currently requires memory=off')
    for name in ('seed', 'env_num', 'group_n', 'is_train'):
        if getattr(raw, name) != state[name]:
            raise ValueError(f'SC2 resume configuration mismatch: {name}')
    if state['reset_index'] != state['episode_reset_index'] + 1:
        raise ValueError('SC2 seed and outcome counters disagree')
    raw.training_rng.set_state(state['training_rng'])
    raw.validation_rng.set_state(state['validation_rng'])
    raw.reset_index = state['reset_index']
    manager._episode_reset_index = state['episode_reset_index']
    manager._step_index = state['step_index']


def legacy_environment_state(manager, run, global_step):
    """Reconstruct only a completed legacy prefix with no skipped policy batches."""
    run = Path(run)
    result = json.loads((run / 'result.json').read_text())
    if (result != json.loads((run / '.done').read_text())
            or result['status'] != 'completed' or result['process_exit_code'] != 0
            or result['target_steps'] != global_step
            or result['completed_steps'] != global_step
            or not result['checkpoint_complete'] or result.get('empty_policy_batches')):
        raise ValueError('Legacy import requires the completed, unskipped final checkpoint')
    rows = [json.loads(line) for line in (run / 'training_seeds.jsonl').read_text().splitlines()]
    state = environment_state(manager)
    rng = np.random.RandomState(state['seed'])
    if len(rows) != global_step:
        raise ValueError('Legacy seed ledger does not match completed global steps')
    for index, row in enumerate(rows):
        expected = np.repeat(rng.randint(0, 2**16 - 1, size=state['env_num']), state['group_n']).tolist()
        if (row['reset_index'] != index or row['base_seed'] != state['seed']
                or row['seeds'] != expected):
            raise ValueError(f'Legacy seed mismatch at reset {index}')
    outcomes = [json.loads(line) for line in (run / 'episode_outcomes.jsonl').read_text().splitlines()]
    identities = [(row['reset_index'], row['env_index']) for row in outcomes]
    expected_ids = {(r, e) for r in range(global_step)
                    for e in range(state['env_num'] * state['group_n'])}
    if (len(identities) != len(expected_ids) or set(identities) != expected_ids
            or not all(row['terminated'] and not row['truncated'] for row in outcomes)):
        raise ValueError('Legacy outcome ledger does not close every game')
    with (run / 'prompt_response.jsonl').open() as stream:
        last_step = max(json.loads(line)['step_index'] for line in stream)
    state.update(training_rng=rng.get_state(), reset_index=len(rows),
                 episode_reset_index=len(rows) - 1, step_index=last_step + 1)
    return state
