"""Reuse checkpoint/seed/closure checks for the current SceneB8 temperature pair."""
import json
from pathlib import Path
import sys

import scene_posteval_check as common
from temperature_eval import SCENE_B8_SEEDS

TEMPERATURE_KEYS = {
    'actor_rollout_ref.rollout.temperature',
    'actor_rollout_ref.rollout.val_kwargs.temperature',
    '+ray_init.runtime_env.env_vars.SC2_SCENE_TEMPERATURE',
}


def compare(paths):
    assert len(paths) == 2
    result = common.compare(paths, allowed=TEMPERATURE_KEYS)
    for path, temperature in zip(map(Path, paths), ('0.4', '0.8')):
        cfg = json.loads((path/'effective_config.json').read_text())['overrides']
        assert cfg['trainer.resume_from_path'] == str(common.checkpoint('SceneB8'))
        assert cfg['+ray_init.runtime_env.worker_process_setup_hook'] == 'temperature_eval.install_scene_b8'
        for key in TEMPERATURE_KEYS:
            assert cfg[key].strip("'") == temperature, key
    return result


def finish(temperature, path):
    result = common.finish('SceneB8', path, game_seeds=SCENE_B8_SEEDS, gpu_selection='4,5,6,7')
    proof = json.loads((Path(path)/'checkpoint_evaluation.json').read_text())
    assert proof['checkpoint_load_returned']
    assert proof['temperature'] == float(temperature) and proof['do_sample']
    result['temperature'] = float(temperature)
    return result


if __name__ == '__main__':
    mode, *args = sys.argv[1:]
    if mode == 'preflight':
        result = common.preflight(('SceneB8',))
    elif mode == 'compare':
        result = compare(args)
    elif mode == 'finish':
        result = finish(*args)
    else:
        raise ValueError(mode)
    print(json.dumps(result, indent=2))
