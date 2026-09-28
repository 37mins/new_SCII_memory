#!/usr/bin/env python3
"""Read the completed predecessor record before allocating long-training GPUs."""
import json
import argparse
from pathlib import Path
import sys

EXPECTED_SHA = '7e5b1f9c58558ea2585c3b6fa58692851a0a1a9c'
MODEL_MANIFEST = '40d6f1417568d5fda1107cdafd66c7441a679b03fb148851c8f226ecc9cf884c'


def check(run: Path, expected_sha=EXPECTED_SHA, steps=1):
    manifest = json.loads((run / 'manifest.json').read_text())
    result = json.loads((run / 'result.json').read_text())
    done = json.loads((run / '.done').read_text())
    if done != result:
        raise ValueError('Predecessor result and completion record disagree')
    expected = {'source_commit': expected_sha, 'model_manifest_sha256': MODEL_MANIFEST,
                'dirty': False, 'formal': True, 'mode': 'training', 'world_size': 4,
                'target_steps': steps}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f'Unexpected predecessor {key}: {manifest.get(key)!r}')
    expected = {'status': 'completed', 'process_exit_code': 0, 'completed_steps': steps,
                'target_steps': steps, 'metrics_finite': True, 'checkpoint_complete': True}
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f'Predecessor not accepted: {key}={result.get(key)!r}')
    expected_files = [f'checkpoints/global_step_{steps}/data.pt'] + [
        f'checkpoints/global_step_{steps}/actor/{kind}_world_size_4_rank_{rank}.pt'
        for kind in ('model', 'optim', 'extra_state') for rank in range(4)]
    for name in expected_files:
        recorded = result['checkpoint_files'][name]['bytes']
        if recorded <= 0 or (run / name).stat().st_size != recorded:
            raise ValueError(f'Incomplete predecessor checkpoint: {name}')
    print(f'Predecessor completed {steps} global steps, finite metrics, full checkpoint; gate passed.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--expected-sha', default=EXPECTED_SHA)
    parser.add_argument('--steps', type=int, default=1)
    args = parser.parse_args()
    check(args.run, args.expected_sha, args.steps)
