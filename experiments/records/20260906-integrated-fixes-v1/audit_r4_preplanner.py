"""Read both completed first-Planner prefixes; no environment/model imports."""
import hashlib
import json
from pathlib import Path
import re
from collections import Counter

ROOT = Path('/mnt/data0601/yangjinyue/new_SCII/experiments/20260906-integrated-fixes-v1/runs')
reports, traces, boundaries, configs, timelines = [], [], [], [], []
for repeat in (1, 2):
    run = ROOT / f'20260906-integrated-fixes-v1-r4-python-rng-preplanner-s2621640384-repeat{repeat}-r1'
    assert (run / '.done').read_text().strip() == 'EXIT_CODE=0'
    result = json.loads((run / 'result.json').read_text())
    reached = result['status'] == 'reached_first_planner'
    if reached:
        trace_path = run / 'boundary_trace.json'
        boundary = json.loads((run / 'first_planner.json').read_text())
        assert boundary['bo_index'] == boundary['bo_length'] == 237
        assert boundary['request']['last_consumed_policy_action_id'] is None
        terminal = None
    else:
        assert result['status'] == 'ended_before_first_planner', result
        trace_path, = run.glob('sc2-logs/*/*/*/*/trace.json')
        boundary = None
        terminal = json.loads((run / 'terminal.json').read_text())['raw_terminal']
        assert terminal['last_consumed_policy_action_id'] is None
    trace = json.loads(trace_path.read_text())
    log = re.sub(r'\x1b\[[0-9;]*m', '', (run / 'driver.log').read_text())
    log = log.split('PREFIX_BOUNDARY:')[0]  # exclude deliberate shutdown
    assert log.count('Generate new plan...') == int(reached)
    starts = re.findall(r'\[(\d+:\d+)\] >>> Start Processing BO #(\d+): (.*?) <<<', log)
    ends = re.findall(r'\[(\d+:\d+)\] >>> BO #(\d+) Completed/Skipped\.', log)
    assert [int(s[1]) for s in starts] == list(range(len(starts)))
    assert [int(s[1]) for s in ends] == list(range(len(ends)))
    if reached:
        assert len(starts) == len(ends) == 237
    attacks = []
    frame = None
    for line in log.splitlines():
        match = re.search(r'\[INFO\]: \((\d+)\)', line)
        if match:
            frame = int(match[1])
        if '[Auto-Attack]' in line:
            attacks.append(dict(last_logged_iteration=frame, text=line[line.index('[Auto-Attack]'):]))
    commands = {f: {k:v for k,v in row.items() if k.startswith('llm_decision_final_action_')}
                for f,row in trace.items() if any(k.startswith('llm_decision_final_action_') for k in row)}
    skips = {f: {k:v for k,v in row.items() if k.startswith('bo_skipped_')}
             for f,row in trace.items() if any(k.startswith('bo_skipped_') for k in row)}
    snapshots = {f:{k:v for k,v in row.items() if isinstance(v, (int,float)) or k=='time_formatted'}
                 for f,row in trace.items() if 'minerals' in row}
    short_run = ROOT / f'20260906-integrated-fixes-v1-r4-python-rng-prefix120-s2621640384-repeat{repeat}-r1'
    short_path, = short_run.glob('sc2-logs/*/*/*/*/trace.json')
    short_trace = json.loads(short_path.read_text())
    overlap = [f for f in short_trace if int(f) <= 670]
    config = json.loads((run / 'effective_config.json').read_text())
    assert config['env']['SC2_SEED_PYTHON_RANDOM'] == '1'
    assert config['env']['CUDA_VISIBLE_DEVICES'] == ''
    assert config['env']['SC2_ENV_SEED'] == '2621640384'
    cleanup = [json.loads(s) for s in (run/'sc2-logs/cleanup.jsonl').read_text().splitlines()]
    assert cleanup and all(not s['alive_child_pids'] and not s['environment_alive'] for s in cleanup)
    artifacts = ['driver.log', str(trace_path.relative_to(run)), 'result.json',
                 'effective_config.json', 'sc2-logs/cleanup.jsonl']
    artifacts += ['first_planner.json'] if reached else ['terminal.json', str(trace_path.with_name('replay.SC2Replay').relative_to(run))]
    reports.append(dict(run_id=run.name, run=str(run), source_sha=(run/'source_sha.txt').read_text().strip(),
        started_at=(run/'started_at.txt').read_text().strip(), ended_at=(run/'ended_at.txt').read_text().strip(),
        exit_code=0, status=result['status'], trace_frames=len(trace), snapshot_frames=len(snapshots),
        observation_frames=sum('obs' in v for v in trace.values()), command_count=len(commands),
        prefix120_overlap_frames=len(overlap),
        prefix120_overlap_equal=all(trace.get(f) == short_trace[f] for f in overlap),
        bo_start_count=len(starts), bo_end_or_skip_count=len(ends), skipped=skips, attack_events=attacks,
        last_bo_start=starts[-1], last_bo_end=ends[-1] if ends else None,
        boundary={k:v for k,v in boundary.items() if k != 'request'} if boundary else None,
        first_query=boundary['request']['obs'] if boundary else None,
        terminal=terminal, last_snapshot=list(snapshots.values())[-1],
        cleanup=cleanup, artifact_hashes={k:hashlib.sha256((run/k).read_bytes()).hexdigest() for k in artifacts}))
    traces.append(trace); boundaries.append(boundary); configs.append(config)
    timelines.append(dict(starts=starts, ends=ends, attacks=attacks, commands=commands))

a,b=traces
frames=sorted(set(a)|set(b), key=int)
diffs=[f for f in frames if a.get(f) != b.get(f)]
changed_fields = Counter()
for f in diffs:
    for k in set(a.get(f, {})) | set(b.get(f, {})):
        av, bv = a.get(f, {}).get(k), b.get(f, {}).get(k)
        if av == bv:
            continue
        if k == 'obs' and isinstance(av, dict) and isinstance(bv, dict):
            changed_fields.update('obs.'+s for s in set(av)|set(bv) if av.get(s) != bv.get(s))
        else:
            changed_fields[k] += 1
first_diff = None
if diffs:
    f = diffs[0]
    changes = {}
    for k in set(a[f]) | set(b[f]):
        av, bv = a[f].get(k), b[f].get(k)
        if av != bv:
            changes[k] = ({s: [av.get(s),bv.get(s)] for s in set(av)|set(bv) if av.get(s)!=bv.get(s)}
                          if k == 'obs' else [av,bv])
    first_diff = dict(frame=f, changes=changes)
# Only replace displayed unit labels, never coordinates, health, counts or command IDs.
normalized = [re.sub(r'\[\d+\](?=[A-Za-z])', '[UNIT_ID]', json.dumps(t, sort_keys=True)) for t in traces]
snapshot_sets = [{f:{k:v for k,v in row.items() if isinstance(v, (int,float)) or k=='time_formatted'}
                  for f,row in t.items() if 'minerals' in row} for t in traces]
config_diffs={k:[configs[0]['env'].get(k),configs[1]['env'].get(k)]
              for k in set(configs[0]['env'])|set(configs[1]['env'])
              if configs[0]['env'].get(k) != configs[1]['env'].get(k)}
assert set(config_diffs) <= {'SC2_RUNTIME_LOG_ROOT','SC2_EPISODE_OUTCOME_LOG'}, config_diffs
print(json.dumps(dict(scope='Same-seed full fixed-BO repeatability only; no model answers or RL',
    reports=reports, full_trace_equal=a == b, compared_trace_frames=len(frames),
    differing_trace_frames=len(diffs), first_trace_difference=first_diff,
    raw_difference_fields=dict(changed_fields),
    trace_equal_ignoring_display_unit_labels=normalized[0] == normalized[1],
    all_resource_snapshots_equal=snapshot_sets[0] == snapshot_sets[1],
    both_reached_first_planner=all(b is not None for b in boundaries),
    boundary_equal=boundaries[0] == boundaries[1] if all(b is not None for b in boundaries) else None,
    timelines_equal=timelines[0] == timelines[1], config_differences=config_diffs),
    ensure_ascii=False,separators=(',',':')))
