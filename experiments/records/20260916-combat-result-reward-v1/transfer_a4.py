"""One authorized A4 copy via configured SSH aliases; no GPU or retries.

Run locally with an unused record directory. Existing source/target results are
never overwritten. SSH retains normal host-key checks. The 2-hour transfer cap
is separate from (and does not extend) the approved 3-hour training budget.
"""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

SOURCE = ('/root/new-scii-execution-feedback-local-20260911/shared/yangjinyue/'
          'new_SCII/experiments/20260911-execution-failure-feedback-v1/runs/'
          '20260911-execution-failure-feedback-v1-spb2-original-l6-g2-A-rl4-4gpu-s0-r2/'
          'checkpoints/global_step_4')
TARGET = '/data/yangjinyue/new-scii-combat-reward-20260916/staging/a4-r1'
OPTIONS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8']


def main(directory):
    directory.mkdir(exist_ok=False)
    def event(kind, **fields):
        with (directory / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                         event=kind, **fields)) + '\n')
    def ssh(host, command, seconds):
        return subprocess.check_output(['ssh', *OPTIONS, host, command], timeout=seconds, text=True)
    event('started', source=SOURCE, target=TARGET, gpu_used=False)
    status = 'failed'
    try:
        ssh('sc2-a100-197', f'test -d {TARGET} && test ! -e {TARGET}/global_step_4', 20)
        expected = ssh('sc2-a100-43', f'cd {SOURCE} && sha256sum data.pt actor/*', 1200)
        (directory / 'source.sha256').write_text(expected)
        event('source_hashed', files=len(expected.splitlines()),
              manifest_sha256=hashlib.sha256(expected.encode()).hexdigest())
        with (directory / 'copy.log').open('w') as log:
            subprocess.run(['scp', '-3', '-O', '-B', '-p', '-r', *OPTIONS,
                            f'sc2-a100-43:{SOURCE}', f'sc2-a100-197:{TARGET}/'],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=7200)
        event('copy_returned')
        actual = ssh('sc2-a100-197', f'cd {TARGET}/global_step_4 && sha256sum data.pt actor/*', 1200)
        (directory / 'destination.sha256').write_text(actual)
        if expected != actual:
            raise RuntimeError('A4 source/destination SHA256 manifests differ')
        status = 'completed'
        event('verified', files=len(actual.splitlines()))
    except Exception as exc:
        event('error', type=type(exc).__name__, message=str(exc))
        raise
    finally:
        (directory / 'result.json').write_text(json.dumps(dict(status=status, source=SOURCE,
            target=TARGET + '/global_step_4', gpu_used=False, automatic_retry=False), indent=2) + '\n')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
