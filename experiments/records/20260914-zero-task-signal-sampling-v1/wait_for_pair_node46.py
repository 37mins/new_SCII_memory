"""Wait within the existing budget, then launch the frozen four-GPU pair once."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/dev/shm/sc2-v2-pair-node46-20260915-r1')
RUNTIME_SHA = '2e2cf8b4bda86edb46d92f84026ee3a924ab7f11'
ENTRY = 'experiments/records/20260914-zero-task-signal-sampling-v1/run_pair_node46.sh'
POLL_SECONDS = 10
MIN_MEMORY_KIB = 600 * 1024**2


def probe():
    def output(args):
        return subprocess.check_output(args, text=True, stderr=subprocess.PIPE, timeout=5).strip()

    pids = output(['nvidia-smi', '-i', '0,1,2,3', '--query-compute-apps=pid',
                   '--format=csv,noheader'])
    rows = output(['nvidia-smi', '-i', '0,1,2,3',
                   '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'])
    gpus = [list(map(int, row.split(','))) for row in rows.splitlines()]
    if [row[0] for row in gpus] != [0, 1, 2, 3]:
        raise RuntimeError('GPU inventory does not match the four authorized devices')
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(memory['MemAvailable'].split()[0])
    ray = subprocess.run(['pgrep', '-x', 'raylet'], capture_output=True, text=True, timeout=5)
    if ray.returncode not in (0, 1):
        raise RuntimeError('Could not verify existing Ray ownership')
    return dict(ready=not pids and not ray.stdout.strip() and available >= MIN_MEMORY_KIB
                and all(mem <= 16 and util == 0 for _, mem, util in gpus),
                gpu_pids=pids.splitlines(), gpus=gpus, raylet_pids=ray.stdout.splitlines(),
                memory_available_kib=available)


def verify_runtime(root):
    code = root/'code'
    sha = subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True, timeout=5).strip()
    dirty = subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain'], text=True, timeout=5)
    if sha != RUNTIME_SHA or dirty:
        raise RuntimeError('Frozen training source is changed or dirty')


def launch(root, control):
    env = dict(os.environ, SOURCE_SHA=RUNTIME_SHA, SC2_ZERO_SIGNAL_AUTHORIZED='1')
    with (control/'launcher.log').open('x') as log:
        return subprocess.run(['bash', str(root/'code'/ENTRY)], env=env,
                              stdout=log, stderr=subprocess.STDOUT).returncode


def watch(root=ROOT):
    # Lock coordinates our watchers only; it does not reserve GPUs against other users.
    with (root/'wait-for-pair.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        control = root/'watch-r1'
        control.mkdir()
        with (control/'events.jsonl').open('x') as events:
            def emit(event, **fields):
                row = dict(time_unix=time.time(), event=event, **fields)
                events.write(json.dumps(row) + '\n')
                events.flush()
                print(json.dumps(row), flush=True)

            def finish(event, code, **fields):
                emit(event, **fields)
                with (control/'done.json').open('x') as out:
                    json.dump(dict(event=event, exit_code=code, time_unix=time.time(), **fields), out)
                return code

            try:
                deadline = int((root/'deadline_unix.txt').read_text())
                verify_runtime(root)
                emit('waiting', deadline_unix=deadline, runtime_sha=RUNTIME_SHA,
                     watcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     gpus=[0, 1, 2, 3], poll_seconds=POLL_SECONDS)
                previous = None
                while time.time() < deadline - 15:
                    if (root/'pair-r2').exists() or (root/'auto-launch-claim.json').exists():
                        return finish('already_attempted_or_started', 0)
                    try:
                        state = probe()
                    except (subprocess.SubprocessError, OSError, ValueError, RuntimeError) as exc:
                        state = dict(ready=False, probe_error=type(exc).__name__)
                    signature = (state['ready'], tuple(state.get('gpu_pids', [])),
                                 tuple(state.get('raylet_pids', [])), state.get('probe_error'))
                    if signature != previous:
                        emit('resource_check', **state)
                        previous = signature
                    if state['ready']:
                        verify_runtime(root)
                        if time.time() >= deadline - 15:
                            break
                        # Durable one-attempt marker is written before invoking the entry.
                        with (root/'auto-launch-claim.json').open('x') as out:
                            json.dump(dict(time_unix=time.time(), runtime_sha=RUNTIME_SHA,
                                           deadline_unix=deadline), out)
                        emit('launch_requested', **state)
                        rc = launch(root, control)
                        return finish('launcher_exited', rc, launcher_exit_code=rc)
                    time.sleep(min(POLL_SECONDS, max(0, deadline - 15 - time.time())))
                return finish('deadline_expired_without_launch', 124)
            except Exception as exc:
                finish('watcher_failed', 1, error_type=type(exc).__name__, error=str(exc))
                raise


if __name__ == '__main__':
    raise SystemExit(watch())
