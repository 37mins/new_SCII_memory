"""Read the two completed CPU prefixes. Never starts an environment or model."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path('/mnt/data0601/yangjinyue/new_SCII/experiments/20260906-integrated-fixes-v1/runs')
reports, traces, configs, bo_events = [], [], [], []
for repeat in (1, 2):
    run = ROOT / f'20260906-integrated-fixes-v1-r4-python-rng-prefix120-s2621640384-repeat{repeat}-r1'
    assert (run / '.done').read_text().strip() == 'EXIT_CODE=0'
    trace_paths = list(run.glob('sc2-logs/*/*/*/*/trace.json'))
    assert len(trace_paths) == 1, trace_paths
    trace_path = trace_paths[0]
    trace = json.loads(trace_path.read_text())
    traces.append(trace)
    log = re.sub(r'\x1b\[[0-9;]*m', '', (run / 'driver.log').read_text())
    assert 'Generate new plan...' not in log
    assert 'game_time_limit=120.0' in log
    assert trace['671']['time_cost'] == 119  # cap checked before the next callback
    terminal = json.loads((run / 'terminal.json').read_text())
    assert terminal['planner_answers_sent'] == 0
    assert terminal['raw_terminal']['last_consumed_policy_action_id'] is None
    config = json.loads((run / 'effective_config.json').read_text())
    assert config['env']['CUDA_VISIBLE_DEVICES'] == ''
    assert config['env']['SC2_SEED_PYTHON_RANDOM'] == '1'
    assert config['env']['SC2_ENV_SEED'] == '2621640384'
    configs.append(config)
    events = re.findall(r'\[(\d+:\d+)\] >>> (Start Processing BO #\d+: .*? <<<|BO #\d+ Completed/Skipped <<<)', log)
    bo_events.append(events)
    command_frames = {frame: {k: v for k, v in row.items() if k.startswith('llm_decision_final_action_')}
                      for frame, row in trace.items() if any(k.startswith('llm_decision_final_action_') for k in row)}
    artifacts = [run / 'driver.log', trace_path, trace_path.with_name('replay.SC2Replay'),
                 run / 'effective_config.json', run / 'terminal.json']
    reports.append(dict(run_id=run.name, run=str(run), exit_code=0,
        source_sha=(run / 'source_sha.txt').read_text().strip(),
        started_at=(run / 'started_at.txt').read_text().strip(),
        ended_at=(run / 'ended_at.txt').read_text().strip(),
        artifacts={str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts},
        trace_frames=len(trace), snapshot_frames=sum('minerals' in v for v in trace.values()),
        observation_frames=sum('obs' in v for v in trace.values()),
        command_frames=command_frames, bo_events=events, planner_calls=0,
        depot_observation_135=trace['135']['obs']['Own structures'],
        snapshot_130=trace['130'], snapshot_670=trace['670']))

a, b = traces
all_frames = sorted(set(a) | set(b), key=int)
differences = [dict(frame=frame, first=a.get(frame), second=b.get(frame))
               for frame in all_frames if a.get(frame) != b.get(frame)]
allowed_path_keys = {'SC2_RUNTIME_LOG_ROOT', 'SC2_EPISODE_OUTCOME_LOG'}
config_differences = {k: [configs[0]['env'].get(k), configs[1]['env'].get(k)]
                      for k in set(configs[0]['env']) | set(configs[1]['env'])
                      if configs[0]['env'].get(k) != configs[1]['env'].get(k)}
assert set(config_differences) <= allowed_path_keys, config_differences
print(json.dumps(dict(source_sha=reports[0]['source_sha'], reports=reports,
    full_trace_equal=a == b, compared_trace_frames=len(all_frames),
    first_trace_difference=differences[:1], differing_trace_frames=len(differences),
    bo_events_equal=bo_events[0] == bo_events[1], config_differences=config_differences,
    scope='Two seeded repeats, not an on/off A/B; no claims about whole-game determinism, KL or learning'),
    ensure_ascii=False, separators=(',', ':')))
