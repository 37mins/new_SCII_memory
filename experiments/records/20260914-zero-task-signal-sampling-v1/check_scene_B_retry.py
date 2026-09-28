"""Check this path-only retry against the saved r1 B configuration/result."""
import json
from pathlib import Path
import sys

STAGE = Path('/mnt/data0601/yangjinyue/new_SCII/staging/20260915-scene-B-r2')
ROOT = Path('/dev/shm/sc2-scene-B-retry-20260915-r2')
RUNS = Path('/mnt/data0601/yangjinyue/new_SCII/experiments/20260914-zero-task-signal-sampling-v1/runs')
BEFORE = RUNS/'20260914-zero-task-signal-sampling-v1-spb2-A4-l6-scene-B-rl4-4gpu-r1'
SHA = '8c2f0f8b103dab054a612ce95a041e4dde9c7c8c'


def compare(output):
    before = json.loads((BEFORE/'effective_config.json').read_text())['overrides']
    after = json.loads((output/'effective_config.json').read_text())['overrides']
    old_manifest = json.loads((BEFORE/'manifest.json').read_text())
    new_manifest = json.loads((output/'manifest.json').read_text())
    replacements = [
        (str(BEFORE), '<RUN>'), (str(output), '<RUN>'),
        (BEFORE.name, '<RUN_ID>'), (output.name, '<RUN_ID>'),
        ('/root/new-scii-zero-signal-20260914/code-75a6b5d', '<CODE>'),
        (str(ROOT/'code-8c2f0f8'), '<CODE>'),
        ('/root/new-scii-execution-feedback-local-20260911/shared', '<STORAGE>'),
        (str(STAGE/'shared'), '<STORAGE>'),
        ('/root/new-scii-spb-level5-96-20260908/python', '<PYTHON>'),
        (str(ROOT/'deps/python'), '<PYTHON>'),
        ('/dev/shm/ss_B_0915_r1', '<RAY>'), ('/dev/shm/sc2_scene_B_0915_r2', '<RAY>'),
    ]
    def normalized(config):
        result = {}
        for key, value in config.items():
            for source, target in replacements:
                value = value.replace(source, target)
            result[key] = value
        return result
    a,b = normalized(before), normalized(after)
    differences = {k:[a.get(k), b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
    if differences:
        raise ValueError(json.dumps(differences, indent=2))
    assert new_manifest['source_commit']==SHA and not new_manifest['dirty']
    for key in ['source_files_sha256','versions','python','bo_sha256','model_manifest_sha256',
                'start_step','target_steps','world_size','gpu_selection']:
        assert old_manifest[key]==new_manifest[key],key
    return dict(effective_config_equal_after_path_mapping=True, source_files_equal=True,
                versions_equal=True, source_sha=SHA, compared_to=str(BEFORE), checked=str(output))


def finish(output):
    result = json.loads((output/'result.json').read_text())
    assert result==json.loads((output/'.done').read_text())
    assert result['status']=='completed' and result['process_exit_code']==0
    assert result['start_step']==4 and result['completed_steps']==4 and result['last_global_step']==8
    assert result['checkpoint_complete'] and result['metrics_finite'] and not result['empty_policy_batches']
    assert result['episode_outcomes']['natural_games']==32
    assert [json.loads(s)['seeds'] for s in (output/'training_seeds.jsonl').read_text().splitlines()]==[
        [s]*8 for s in [1098341402,3446496864,1098341402,3446496864]]
    proof=json.loads((output/'checkpoints/a4_continuation.json').read_text())
    assert proof['model_optimizer_load_returned'] and proof['game_seed_per_collection']==[
        1098341402,3446496864,1098341402,3446496864]
    assert result['signal_sampling']['skipped_batches']==0
    for name, entry in result['checkpoint_files'].items():
        assert (output/name).stat().st_size==entry['bytes'],name
    return dict(status='completed', additional_global_steps=4, checkpoint_complete=True,
                wins=result['episode_outcomes']['wins'], games=32, learning_validated=False)


if __name__=='__main__':
    mode,output=sys.argv[1:]
    if mode not in ('compare','finish'): raise ValueError(mode)
    print(json.dumps((compare if mode=='compare' else finish)(Path(output)),indent=2))
