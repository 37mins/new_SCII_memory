"""One-off cleanup of two explicitly retired SC2 smoke checkpoints on node 42."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from datetime import datetime, timezone

ROOT = Path('/dev/shm/new_scii_level5_attn_ab_assets/runs')
TARGETS = {
    '20260820-level5-attention-backend-ab-v1-spb-eager-smoke2-s0-r1':
        'spb2_spb2_level5_attention_eager_smoke2_v1_level5_oldmicro_sc2-a100-42_seed0_step2_resp96_level5_spb2_instruct_noskill_compact6_v7_newa100_20260815',
    '20260820-level5-attention-backend-ab-v1-spb-canonical-eager-level6-smoke2-s0-r4':
        'spb2_spb2_generic_rl_level6_canonical_plan_v2_eager_smoke2_level6_oldmicro_sc2-a100-42_seed0_step2_resp96_level6_spb2_instruct_noskill_compact6_v7_newa100_20260815',
}


def now():
    return datetime.now(timezone.utc).isoformat()


def active_references():
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            fields = [('cmdline', (proc / 'cmdline').read_bytes().decode(errors='replace'))]
            fields.append(('cwd', os.readlink(proc / 'cwd')))
            fields.append(('maps', (proc / 'maps').read_text(errors='replace')))
            for fd in (proc / 'fd').iterdir():
                try:
                    fields.append(('fd', os.readlink(fd)))
                except FileNotFoundError:
                    pass
            for source, value in fields:
                for run in TARGETS:
                    if run in value:
                        found.append({'pid': int(proc.name), 'source': source, 'run': run})
        except (FileNotFoundError, ProcessLookupError):
            pass
    return found


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def snapshot(path):
    s = path.lstat()
    assert stat.S_ISREG(s.st_mode) and s.st_nlink == 1, path
    assert path.resolve() == path, path
    return {'path': str(path), 'size': s.st_size, 'blocks': s.st_blocks,
            'inode': s.st_ino, 'device': s.st_dev, 'mtime_ns': s.st_mtime_ns}


mode, record_name = sys.argv[1:]
record = Path(record_name)
assert mode in ('audit', 'apply')
if mode == 'audit':
    assert not record.exists()
    refs = active_references()
    assert not refs, refs
    data = {'node_alias': 'sc2-a100-42', 'audited_at': now(), 'active_references': refs,
            'reason': 'Retired two-step smoke checkpoints; original models, baseline/final checkpoints and all non-model/non-optimizer artifacts retained.',
            'backup_verified': False, 'runs': [], 'files': []}
    for run, checkpoint_name in TARGETS.items():
        run_root = ROOT / run
        exit_path = run_root / 'driver.exit'
        assert exit_path.read_text().strip() == '0', run
        actor = run_root / 'results/checkpoints' / checkpoint_name / 'global_step_2/actor'
        data['runs'].append({'run_id': run, 'driver_exit': 0, 'actor': str(actor)})
        for kind in ('model', 'optim'):
            for rank in range(8):
                path = actor / f'{kind}_world_size_8_rank_{rank}.pt'
                item = snapshot(path)
                item['sha256'] = digest(path)
                assert snapshot(path) == {k: v for k, v in item.items() if k != 'sha256'}
                data['files'].append(item)
                print(json.dumps({'hashed': str(path), 'bytes': item['size']}), flush=True)
    data['bytes'] = sum(f['size'] for f in data['files'])
    data['allocated_bytes'] = sum(f['blocks'] * 512 for f in data['files'])
    data['active_references_after_hash'] = active_references()
    assert not data['active_references_after_hash']
    with record.open('x') as stream:
        json.dump(data, stream, indent=2)
    print(json.dumps({'manifest': str(record), 'files': len(data['files']), 'bytes': data['bytes']}))
else:
    data = json.loads(record.read_text())
    receipt = record.with_suffix('.deleted.jsonl')
    assert not receipt.exists()
    assert not active_references()
    expected = {str(ROOT / run / 'results/checkpoints' / ckpt / 'global_step_2/actor' /
                    f'{kind}_world_size_8_rank_{rank}.pt')
                for run, ckpt in TARGETS.items() for kind in ('model', 'optim') for rank in range(8)}
    assert {f['path'] for f in data['files']} == expected and len(data['files']) == 32
    for item in data['files']:
        assert snapshot(Path(item['path'])) == {k: v for k, v in item.items() if k != 'sha256'}
    # Only individually enumerated regular shard files are unlinked; no recursive deletion.
    with receipt.open('x') as stream:
        for item in data['files']:
            Path(item['path']).unlink()
            stream.write(json.dumps({'deleted_at': now(), 'path': item['path'], 'bytes': item['size']}) + '\n')
            stream.flush()
    print(json.dumps({'deleted_files': len(data['files']), 'bytes': data['bytes'], 'receipt': str(receipt)}))
