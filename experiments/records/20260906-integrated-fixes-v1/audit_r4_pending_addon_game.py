"""Read-only observation acceptance at the first-Planner boundary; no replay fabricated."""
import hashlib
import json
from pathlib import Path
import re
import sys

run = Path(sys.argv[1])
assert (run / '.done').read_text().strip() == 'EXIT_CODE=0'
assert (run / 'source_sha.txt').read_text().strip() == '836460a8533b6f0429f9e9d401ac9715adc55f5b'
config = json.loads((run / 'effective_config.json').read_text())
for key, value in dict(SC2_ENV_SEED='2621640384', SC2_DIFFICULTY='Harder',
                       SC2_AI_BUILD='Air', SC2_SEED_PYTHON_RANDOM='1',
                       SC2_RNG_PREFIX_REPEATS='1', CUDA_VISIBLE_DEVICES='').items():
    assert config['env'][key] == value, key
result = json.loads((run / 'result.json').read_text())
assert result['status'] == 'reached_first_planner' and result['planner_answers_sent'] == 0
boundary = json.loads((run / 'first_planner.json').read_text())
assert boundary['bo_index'] == boundary['bo_length'] == 237
assert boundary['request']['last_consumed_policy_action_id'] is None
trace = json.loads((run / 'boundary_trace.json').read_text())
commands, skipped, samples, first_started, first_done, first_tanks = [], [], [], {}, {}, {}
for frame, row in sorted(trace.items(), key=lambda item: int(item[0])):
    for key, value in row.items():
        if key.startswith('llm_decision_final_action_') and any(
                a['action'] in ('BUILD_TECHLAB_FACTORY', 'FACTORYTRAIN_SIEGETANK') for a in value):
            commands.append(dict(iteration=int(frame), key=key, actions=value))
        if key.startswith('bo_skipped_'):
            skipped.append(dict(iteration=int(frame), key=key, value=value))
    obs = row.get('obs', {})
    labs, tanks = [], []
    for field, name_pattern, found in (
            ('Own structures', 'FactoryTechLab', labs),
            ('Own units', 'SiegeTank(?:Sieged)?', tanks)):
        text = obs.get(field, '')
        # Anchored unit headers, not a production queue or available-ability list.
        for match in re.finditer(rf'^\[(\d+)\]({name_pattern})(?:\(building (\d+)%\))?$', text, re.M):
            block = text[match.start():].split('\n[', 1)[0]
            item = dict(iteration=int(frame), second=int(frame)*4/22.4,
                display_id=int(match[1]), type=match[2], complete=match[3] is None,
                observed_progress_percent=100 if match[3] is None else int(match[3]),
                raw='\n'.join(block.splitlines()[:5]))
            found.append(item)
            ledger = first_tanks if found is tanks else first_done if item['complete'] else first_started
            ledger.setdefault(item['display_id'], item)
    if labs or tanks:
        samples.append(dict(iteration=int(frame), second=int(frame)*4/22.4,
            labs_building=[x['display_id'] for x in labs if not x['complete']],
            labs_complete=[x['display_id'] for x in labs if x['complete']],
            tanks_on_map=[x['display_id'] for x in tanks if x['complete']]))
peak = {field: max(samples, key=lambda s: len(s[field]))
        for field in ('labs_building', 'labs_complete', 'tanks_on_map')}
cleanup = [json.loads(line) for line in (run / 'sc2-logs/cleanup.jsonl').read_text().splitlines()]
assert cleanup and all(not c['alive_child_pids'] and not c['environment_alive'] for c in cleanup)
artifacts = ['.done', 'source_sha.txt', 'effective_config.json', 'result.json',
    'started_at.txt', 'ended_at.txt', 'driver.log', 'boundary_trace.json',
    'first_planner.json', 'sc2-logs/cleanup.jsonl']
report = dict(run_id=run.name, source_sha=(run / 'source_sha.txt').read_text().strip(),
    started_at=(run / 'started_at.txt').read_text().strip(),
    ended_at=(run / 'ended_at.txt').read_text().strip(), exit_code=0,
    status=result['status'], planner_answers_sent=0,
    boundary={k: v for k, v in boundary.items() if k != 'request'},
    commands=commands, skipped=skipped, first_observed_under_construction=list(first_started.values()),
    first_observed_completed=list(first_done.values()), first_observed_tanks=list(first_tanks.values()),
    distinct_tanks_observed=len(first_tanks),
    display_id_basis='Frozen BasePlayer.tag_to_id retains the tag-to-ID mapping for the entire game; IDs are not reassigned.',
    peak_simultaneous_observations=peak,
    observation_production_gate=all(len(s[field]) > 0 for field, s in peak.items()),
    replay_available=bool(list(run.rglob('*.SC2Replay'))),
    limitation='First-Planner diagnostic stop, not win/loss. Times are first recorded observations, not exact birth/done event times. Tank count is a simultaneous on-map lower bound, not cumulative births.',
    completion_evidence='Frozen BasePlayer.unit_to_text emits a bare unit header only at build_progress == 1.0; otherwise it adds (building N%).',
    cleanup=cleanup,
    artifact_hashes={name: hashlib.sha256((run / name).read_bytes()).hexdigest() for name in artifacts})
print(json.dumps(report, ensure_ascii=False, indent=2))
