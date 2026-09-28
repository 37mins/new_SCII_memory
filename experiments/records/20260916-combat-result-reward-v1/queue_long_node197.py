"""One node197 suite: wait for D/GPUs, eval, train8, eval, reviewed gate, train96."""
import datetime as dt
import fcntl
import json
from pathlib import Path
import shutil
import subprocess as sp
import sys
import time

GIB = 1024**3
D_SHA = '2ad7fb0a8cd67b8d349e2caa0cd4561599596c4c'
D_RUN = '20260916-combat-result-reward-v1-spb2-A4-l6-D-temp08-gamma1-rl4-4gpu-r1'
SUITE = '20260916-combat-result-reward-v1-spb2-original-l6-D-rl96-gate8-8gpu-s0-r1'
STAGES = ('initial-eval', 'train8', 'step8-eval', 'train96')
LABEL = 'sc2.combat.long.root'


def command(args, check=True):
    return sp.run(args, text=True, capture_output=True, check=check, timeout=30)


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def idle(rows, processes, available_bytes):
    rows = [r.split(',') for r in rows.strip().splitlines()]
    return (len(rows) == 8 and not processes.strip() and available_bytes >= 600*GIB
            and {int(r[0]) for r in rows} == set(range(8))
            and all(int(r[1]) <= 16 and int(r[2]) == 0 for r in rows))


def read_gate(path, expected_sha, expected_runs):
    if not path.exists():
        return False
    gate = json.loads(path.read_text())
    if (gate['source_sha'] != expected_sha or gate['run_ids'] != expected_runs
            or not gate.get('evidence')):
        raise ValueError('Gate evidence does not match the frozen experiment')
    if gate['decision'] != 'continue':
        raise RuntimeError('Gate declined; no alternative configuration or retry')
    return True


def verify_result(run, stage, sha):
    result = json.loads((run/'result.json').read_text())
    manifest = json.loads((run/'manifest.json').read_text())
    if (result != json.loads((run/'.done').read_text()) or result['status'] != 'completed'
            or result['process_exit_code'] != 0 or manifest['source_commit'] != sha
            or manifest['dirty'] or manifest['world_size'] != 8):
        raise RuntimeError(f'Incomplete or mismatched stage: {stage}')
    if stage.endswith('eval'):
        proof = json.loads((run/'checkpoint_evaluation.json').read_text())
        expected_step = 0 if stage == 'initial-eval' else 8
        if (result['training_updates'] != 0 or result['episode_outcomes']['natural_games'] != 8
                or proof['loaded_global_step'] != expected_step or not proof['val_only']
                or proof['optimizer_updates'] != 0 or len(set(proof['game_seeds'])) != 8):
            raise RuntimeError('Evaluation load/game-count check failed')
        actual_seeds = [json.loads(s)['seeds'] for s in (run/'evaluation_seeds.jsonl').read_text().splitlines()]
        if actual_seeds != [proof['game_seeds']]:
            raise RuntimeError('Actual evaluation seeds do not match declared pair')
    else:
        start, target = (0, 8) if stage == 'train8' else (8, 96)
        if (result['start_step'] != start or result['last_global_step'] != target
                or result['completed_steps'] != target-start or not result['checkpoint_complete']
                or not result['metrics_finite'] or result['empty_policy_batches']):
            raise RuntimeError('Training/checkpoint acceptance failed')
        cp = run/'checkpoints'/f'global_step_{target}'
        files = [cp/'data.pt', cp/'sc2_state.pt'] + [cp/'actor'/f'{kind}_world_size_8_rank_{rank}.pt'
                 for kind in ('model', 'optim', 'extra_state') for rank in range(8)]
        if not all(f.is_file() and f.stat().st_size > 0 for f in files):
            raise RuntimeError('Missing complete checkpoint state')
    return result


def stop_owned(name, root):
    info = command(['docker', 'inspect', name], check=False)
    if info.returncode:
        return
    obj = json.loads(info.stdout)[0]
    if obj['Config'].get('Labels', {}).get(LABEL) != str(root):
        raise RuntimeError('Refusing to stop an unowned container')
    command(['docker', 'stop', '--time', '20', name])


def run(root):
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/'state.json').exists():
            raise RuntimeError('This queue was already used; no automatic retry')
        config = json.loads((root/'deployment.json').read_text())
        source = Path(config['source'])
        assert command(['git', '-C', str(source), 'rev-parse', 'HEAD']).stdout.strip() == config['sha']
        assert not command(['git', '-C', str(source), 'status', '--porcelain']).stdout.strip()
        state = dict(phase='waiting-D', source_sha=config['sha'], deadline=None)
        container = None
        try:
            for stage in STAGES:
                while True:
                    state.update(checked_at=dt.datetime.now(dt.timezone.utc).isoformat(), stage=stage)
                    write_json(root/'state.json', state)
                    deadline = state['deadline'] or config['queue_deadline_unix']
                    if time.time() >= deadline:
                        raise TimeoutError('Authorized time window exhausted')
                    if not read_gate(root/'D-gate.json', D_SHA, [D_RUN]):
                        time.sleep(15)
                        continue
                    if stage == 'train96' and not read_gate(root/'step8-gate.json', config['sha'],
                            [SUITE+'-initial-eval', SUITE+'-train8', SUITE+'-step8-eval']):
                        state['phase'] = 'waiting-step8-review'
                        time.sleep(15)
                        continue
                    state['phase'] = 'waiting-eight-GPUs'
                    rows = command(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits']).stdout
                    processes = command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).stdout
                    memory = dict(s.split(':', 1) for s in Path('/proc/meminfo').read_text().splitlines())
                    if (not idle(rows, processes, int(memory['MemAvailable'].split()[0])*1024)
                            or command(['pgrep', '-x', 'raylet'], False).returncode == 0):
                        time.sleep(15)
                        continue
                    if shutil.disk_usage(root).free < 1536*GIB:
                        raise RuntimeError('Insufficient disk reserve; no cleanup permitted')
                    break
                if state['deadline'] is None:
                    state['deadline'] = time.time() + 48*3600
                    state['started_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
                run_id = SUITE+'-'+stage
                output = root/'runs'/run_id
                if output.exists():
                    raise RuntimeError(f'Existing run must not be overwritten: {output}')
                container = 'sc2-combat-long197-'+stage+'-r1'
                parent = root/'runs'/(SUITE+'-train8')/'checkpoints/global_step_8'
                args = ['docker', 'run', '--rm', '--name', container, '--label', f'{LABEL}={root}',
                        '--gpus', 'all', '--shm-size', '128g', '--entrypoint', '/bin/bash',
                        '-v', f'{root}:{root}', '-v', f'{source}:{source}:ro',
                        '-v', f"{config['venv']}:/runtime:ro", '-v', f"{config['model']}:/assets/spb:ro",
                        '-v', f"{config['sc2']}:/assets/StarCraftII:ro", '-w', str(source)]
                env = dict(SOURCE_SHA=config['sha'], RUN_ID=run_id, OUTPUT_DIR=str(output),
                           SC2_COMBAT_LONG_AUTHORIZED='1', PARENT_CHECKPOINT=str(parent),
                           PYTHONDONTWRITEBYTECODE='1', CONTAINER_IMAGE_ID=config['image'])
                for key, value in env.items():
                    args += ['-e', f'{key}={value}']
                args += [config['image'], str(source/'experiments/records/20260916-combat-result-reward-v1/run_long_stage.sh'), stage]
                write_json(root/'launches'/f'{stage}.json', dict(command=args, deadline=state['deadline']))
                state.update(phase='launching-unverified', run_id=run_id, container=container)
                write_json(root/'state.json', state)
                with (root/'logs'/f'{stage}.log').open('x') as log:
                    process = sp.Popen(args, stdout=log, stderr=sp.STDOUT)
                    next_storage_check = 0
                    while process.poll() is None:
                        if time.time() >= state['deadline'] - 30:
                            stop_owned(container, root)
                            process.wait(timeout=30)
                            raise TimeoutError('48-hour suite limit; no restart')
                        if time.time() >= next_storage_check:
                            used = int(command(['du', '-s', '-B1', str(root)]).stdout.split()[0])
                            if used >= 1536*GIB:
                                stop_owned(container, root)
                                process.wait(timeout=30)
                                raise RuntimeError('1.5TiB suite storage ceiling; no deletion or retry')
                            next_storage_check = time.time() + 120
                        time.sleep(5)
                    if process.returncode:
                        raise RuntimeError(f'{stage} exited {process.returncode}; no retry')
                verify_result(output, stage, config['sha'])
                if stage == 'step8-eval':
                    initial = root/'runs'/(SUITE+'-initial-eval')
                    a = json.loads((initial/'checkpoint_evaluation.json').read_text())
                    b = json.loads((output/'checkpoint_evaluation.json').read_text())
                    for key in ('game_seeds', 'temperature', 'do_sample', 'feedback', 'games'):
                        if a[key] != b[key]:
                            raise RuntimeError(f'Unpaired evaluation field: {key}')
                state.update(phase='stage-completed', exit_code=0)
                write_json(root/'state.json', state)
            state['phase'] = 'completed'
            write_json(root/'state.json', state)
        except Exception as error:
            state.update(phase='failed', error=str(error))
            write_json(root/'state.json', state)
            if container:
                stop_owned(container, root)
            raise


if __name__ == '__main__':
    run(Path(sys.argv[1]).resolve())
