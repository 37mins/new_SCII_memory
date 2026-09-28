"""Small-file preflight and acceptance for the approved three-checkpoint evaluation."""
import json
import os
from pathlib import Path
import sys

from learning_eval import SCENE_POST_GAME_SEEDS

SHARED = Path('/mnt/data0601/yangjinyue/new_SCII')
RUNS = SHARED/'experiments/20260914-zero-task-signal-sampling-v1/runs'
PARENTS = {
    'A4': (SHARED/'staging/20260915-scene-B-r2/shared/yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs/20260911-execution-failure-feedback-v1-spb2-original-l6-g2-A-rl4-4gpu-s0-r2', 4, '361cb7ae7fd86fe8811d6fcb6946d0241bf93e11'),
    'SceneA8': (RUNS/'20260914-zero-task-signal-sampling-v1-spb2-A4-l6-scene-A-rl4-4gpu-r1', 8, '75a6b5dc3621e9d670fbfb440ae917bfacd5e360'),
    'SceneB8': (RUNS/'20260914-zero-task-signal-sampling-v1-spb2-A4-l6-scene-B-rl4-4gpu-r2', 8, '8c2f0f8b103dab054a612ce95a041e4dde9c7c8c'),
}


def checkpoint(label):
    parent, step, _ = PARENTS[label]
    return parent/f'checkpoints/global_step_{step}'


def preflight(labels=tuple(PARENTS)):
    records = {}
    for label in labels:
        parent, step, sha = PARENTS[label]
        result = json.loads((parent/'result.json').read_text())
        manifest = json.loads((parent/'manifest.json').read_text())
        assert result == json.loads((parent/'.done').read_text())
        assert result['status'] == 'completed' and result['process_exit_code'] == 0
        assert result['checkpoint_complete'] and result['last_global_step'] == step
        assert manifest['source_commit'] == sha and not manifest['dirty'] and manifest['world_size'] == 4
        assert manifest['model_manifest_sha256'] == '40d6f1417568d5fda1107cdafd66c7441a679b03fb148851c8f226ecc9cf884c'
        assert all((parent/name).stat().st_size == meta['bytes'] for name, meta in result['checkpoint_files'].items())
        records[label] = dict(checkpoint=str(checkpoint(label)), source=sha, step=step, actual_load_still_required=True)
    return records


def compare(paths, allowed=None):
    configs, manifests = [], []
    for path in map(Path, paths):
        config = json.loads((path/'effective_config.json').read_text())['overrides']
        ray = config['+ray_init._temp_dir'] if '+ray_init._temp_dir' in config else config.get('ray_init._temp_dir')
        normalized = {}
        for key, value in config.items():
            value = value.replace(str(path), '<RUN>').replace(path.name, '<RUN_ID>')
            if ray:
                value = value.replace(ray, '<RAY>')
            normalized[key] = value
        configs.append(normalized)
        manifests.append(json.loads((path/'manifest.json').read_text()))
    if allowed is None:
        allowed = {'trainer.resume_from_path', '+ray_init.runtime_env.env_vars.SC2_FEEDBACK_LEARNING_MODEL'}
    for config, manifest in zip(configs[1:], manifests[1:]):
        differences = {k for k in config.keys() | configs[0].keys() if config.get(k) != configs[0].get(k)}
        assert differences <= allowed, differences
        for key in ('source_commit', 'source_files_sha256', 'versions', 'python', 'model_manifest_sha256', 'bo_sha256', 'world_size'):
            assert manifest[key] == manifests[0][key], key
    return dict(configs_match_except_checkpoint_and_label=allowed == {'trainer.resume_from_path', '+ray_init.runtime_env.env_vars.SC2_FEEDBACK_LEARNING_MODEL'}, allowed_differences=sorted(allowed), manifests_match=True)


def finish(label, path, game_seeds=SCENE_POST_GAME_SEEDS, gpu_selection=None):
    path = Path(path)
    result = json.loads((path/'result.json').read_text())
    proof = json.loads((path/'checkpoint_evaluation.json').read_text())
    manifest = json.loads((path/'manifest.json').read_text())
    assert result == json.loads((path/'.done').read_text())
    assert result['status'] == 'completed' and result['process_exit_code'] == 0
    assert result['training_updates'] == 0 and result['episode_outcomes']['natural_games'] == 8
    assert not result['episode_outcomes']['truncated']
    assert manifest['source_commit'] == os.environ['SOURCE_SHA'] and not manifest['dirty']
    expected_gpus = gpu_selection if gpu_selection is not None else ('4,5,6,7' if label == 'SceneA8' else '0,1,2,3')
    assert manifest['world_size'] == 4 and manifest['gpu_selection'] == expected_gpus
    assert proof['val_only'] and proof['optimizer_updates'] == 0 and proof['loaded_global_step'] == PARENTS[label][1]
    assert proof['feedback'] == '0' and proof['checkpoint'] == str(checkpoint(label))
    assert proof['game_seeds'] == game_seeds
    assert [json.loads(s)['seeds'] for s in (path/'evaluation_seeds.jsonl').read_text().splitlines()] == [game_seeds]
    states = []
    for rank in range(4):
        sampling = json.loads((path/f'sampling-rank{rank}.json').read_text())
        assert sampling['seed'] == sampling['rank'] == rank
        states.append({k: sampling[k] for k in ('cpu_rng_sha256', 'cuda_rng_sha256')})
    return dict(label=label, status='completed', games=8, wins=result['episode_outcomes']['wins'],
                training_updates=0, sampling_states=states, learning_validated=False)


if __name__ == '__main__':
    mode, *args = sys.argv[1:]
    if mode == 'checkpoint':
        print(checkpoint(args[0]))
    else:
        value = preflight() if mode == 'preflight' else compare(args) if mode == 'compare' else finish(*args)
        print(json.dumps(value, indent=2))
