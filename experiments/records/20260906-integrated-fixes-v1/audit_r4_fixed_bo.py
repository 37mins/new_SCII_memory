"""CPU-only extraction of the already recorded r4 env3 fixed-BO trajectories."""
import ast
import hashlib
import json
from pathlib import Path
import re

WORKERS = {
    'initial': ('/dev/shm/sc2_real_r4_initial-eval/session_2026-09-07_01-23-19_820340_3649359/logs/worker-dd21f0bfe5e661dfea91514bc8e4cb89fc2e3936f9c6fa41134fb329-01000000-3652008.out', '5ea045084a7f4db9d487dc729428a40d7fa9435caaf4aafc9acbd5a30da6896c'),
    'step8': ('/dev/shm/sc2_real_r4_step8-eval/session_2026-09-07_08-07-31_923930_3951460/logs/worker-93ce7ecc30f4db4c401ea4a236a00e2f8e5cc5502935af78e69fc415-01000000-3954027.out', '00037381033007ed48cbe9c198a42a67558e0139c0b4f95666c61d98afceda55'),
}
FIELDS = ('time_seconds', 'minerals', 'vespene', 'cumulative_minerals_collected',
          'unit_mineral_value', 'structure_mineral_value', 'supply_workers',
          'supply_army', 'supply_left', 'n_structures', 'n_visible_enemy_units')
reports = {}
all_snapshots = {}
for stage, (name, expected_hash) in WORKERS.items():
    data = Path(name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected_hash
    lines = re.sub(r'\x1b\[[0-9;]*m', '', data.decode()).splitlines()
    stop = next(i for i, x in enumerate(lines) if 'Generate new plan... [0]' in x)
    bos, snapshots, events = [], {}, []
    iteration = None
    for i, line in enumerate(lines[:stop]):
        info = re.search(r'\[INFO\]: \((\d+)\) ([^:]+): (.*)', line)
        if info:
            iteration = int(info[1])
            if info[2] in FIELDS:
                snapshots.setdefault(iteration, {'line': i + 1, 'iteration': iteration})[info[2]] = float(info[3])
        start = re.search(r'\[(\d+:\d+)\] >>> Start Processing BO #(\d+): (.*?) <<<', line)
        if start:
            bos.append(dict(index=int(start[2]), action=start[3], start=start[1], start_line=i+1, commands=[], waits=[]))
        finish = re.search(r'\[(\d+:\d+)\] >>> BO #(\d+) Completed/Skipped', line)
        if finish:
            assert bos[-1]['index'] == int(finish[2])
            bos[-1].update(end=finish[1], end_line=i+1)
        action = re.search(r'\((\d+)\) llm_decision_final_action_(\d+)_(\d+): (.*)', line)
        if action:
            assert int(action[2]) == bos[-1]['index']
            bos[-1]['commands'].append(dict(line=i+1, iteration=int(action[1]), action=ast.literal_eval(action[4])))
        if '[Wait]' in line or '[Skip]' in line:
            bos[-1]['waits'].append(dict(line=i+1, last_logged_iteration=iteration, text=line.strip()))
        if ('[Auto-Attack]' in line or 'scouting:' in line or '正在派遣' in line or
            'destroyed' in line.lower() or 'scout_lost' in line.lower() or '[Pre-Move]' in line):
            events.append(dict(line=i+1, last_logged_iteration=iteration, text=line))
    assert [x['index'] for x in bos] == list(range(237))
    assert all('end' in x for x in bos)
    all_snapshots[stage] = snapshots
    reports[stage] = dict(worker=name, worker_sha256=expected_hash, first_planner_line=stop+1,
        early_bo=bos[:20], bo_timing=[{k:b[k] for k in ('index','action','start','end')} for b in bos],
        timeouts=[b for b in bos if any('[Skip]' in w['text'] for w in b['waits'])],
        early_snapshots=[s for s in snapshots.values() if s['iteration'] <= 600],
        later_snapshots=[s for s in snapshots.values() if s['iteration'] > 600 and s['iteration'] % 100 == 0],
        events=events)

a,b=(all_snapshots[k] for k in WORKERS)
shared=sorted(set(a) & set(b))
first_diffs={}
for field in FIELDS:
    for iteration in shared:
        if a[iteration].get(field) != b[iteration].get(field):
            first_diffs[field]=dict(initial=a[iteration], step8=b[iteration])
            break
print(json.dumps(dict(scope='Existing logs only; no model, GPU or server writes',
    seed=2621640384, source_sha='ca7885d886aaaafb39fc105b1d5d46cddf4bcc5a',
    first_snapshot_differences=first_diffs, reports=reports), ensure_ascii=False, separators=(',', ':')))
