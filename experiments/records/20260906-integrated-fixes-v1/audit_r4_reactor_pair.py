"""Read completed paired evaluation artifacts; never load a model or run SC2."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

from mpyq import MPQArchive
from s2protocol.versions import build, latest


def replay_summary(path):
    archive = MPQArchive(str(path))
    header = latest().decode_replay_header(archive.header['user_data_header']['content'])
    protocol = build(header['m_version']['m_baseBuild'])
    units, born, started, completed, dead = {}, Counter(), Counter(), Counter(), Counter()
    upgrades, last_stats = [], None
    killers = defaultdict(Counter)
    for e in protocol.decode_replay_tracker_events(archive.read_file('replay.tracker.events')):
        kind = e['_event'].split('.')[-1]
        tag = (e.get('m_unitTagIndex'), e.get('m_unitTagRecycle'))
        if kind in ('SUnitBornEvent', 'SUnitInitEvent'):
            units[tag] = [e['m_unitTypeName'].decode(), e['m_controlPlayerId']]
            if units[tag][1] == 1:
                (born if kind == 'SUnitBornEvent' else started)[units[tag][0]] += 1
        elif kind == 'SUnitDoneEvent':
            if units[tag][1] == 1:
                completed[units[tag][0]] += 1
        elif kind == 'SUnitTypeChangeEvent':
            units[tag][0] = e['m_unitTypeName'].decode()
        elif kind == 'SUnitDiedEvent':
            if units[tag][1] == 1:
                dead[units[tag][0]] += 1
                killer = units.get((e.get('m_killerUnitTagIndex'), e.get('m_killerUnitTagRecycle')))
                killers[units[tag][0]][killer[0] if killer else 'unrecorded'] += 1
        elif kind == 'SUpgradeEvent' and e['m_playerId'] == 1:
            upgrades.append(e['m_upgradeTypeName'].decode())
        elif kind == 'SPlayerStatsEvent' and e['m_playerId'] == 1:
            last_stats = {'game_seconds': e['_gameloop'] / 22.4, **e['m_stats']}
    return dict(build=header['m_version']['m_baseBuild'],
                game_seconds=header['m_elapsedGameLoops'] / 22.4,
                own_born_including_initial=born, own_construction_started=started,
                own_construction_completed=completed, own_deaths=dead,
                killers_by_victim_type=killers, upgrades=upgrades, final_stats=last_stats)


def audit(run):
    result = json.loads((run / 'result.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    seeds = [json.loads(s) for s in (run / 'evaluation_seeds.jsonl').read_text().splitlines()]
    if seeds != [dict(base_seed=1000, reset_index=0, seeds=[2621640384])]:
        raise ValueError(f'Unexpected game seeds: {seeds}')
    if manifest['mode'] != 'evaluation' or result['training_updates'] != 0:
        raise ValueError('Not the approved evaluation-only run')
    rows = [json.loads(s) for s in (run / 'prompt_response.jsonl').read_text().splitlines()]
    applied = [r for r in rows if r.get('policy_action_applied') is True]
    evaluable = [r for r in applied if r['semantic_action_evaluable']]
    skipped, command_counts = [], Counter()
    traces = list((run / 'sc2-logs').rglob('trace.json'))
    for trace_path in traces:
        for frame, record in json.loads(trace_path.read_text()).items():
            for key, value in record.items():
                if key.startswith('bo_skipped_'):
                    skipped.append(dict(frame=frame, key=key, record=value))
                if key.startswith('llm_decision_final_action_'):
                    command_counts.update(a['action'] for a in value)
    replays = list((run / 'sc2-logs').rglob('*.SC2Replay'))
    if len(replays) != 1:
        raise ValueError(f'Expected one full replay, got {len(replays)}')
    artifacts = [run / n for n in ('manifest.json', 'effective_config.json', 'result.json',
                                  '.done', 'evaluation_seeds.jsonl', 'prompt_response.jsonl',
                                  'episode_outcomes.jsonl', 'console.log')]
    artifacts += traces + replays
    return dict(run_id=run.name, manifest=manifest, result=result,
                seeds=seeds, consumed_model_answers=len(applied),
                cumulative_environment_reward=sum(r['reward'] for r in rows),
                valid_evaluable_answers=sum(bool(r['is_action_valid']) for r in evaluable),
                evaluable_answers=len(evaluable),
                raw_response_frequency=Counter(r['llm_response'] for r in applied),
                first_answers=[{k:r[k] for k in ('step_index','llm_response','projected_action',
                    'is_action_valid','failure_reason','policy_action_id')} for r in applied[:3]],
                skipped=skipped, executor_action_records=command_counts,
                replay=replay_summary(replays[0]),
                artifact_sha256={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in artifacts},
                caveats='One seed, one game per arm; replay owner 1 is our first player. Born SCVs include the initial 12. Executor records are commands, not production. Reward sums all environment interactions including the terminal poll; no-policy games must not be excluded.')


if __name__ == '__main__':
    print(json.dumps([audit(Path(p)) for p in sys.argv[1:]], ensure_ascii=False, indent=2))
