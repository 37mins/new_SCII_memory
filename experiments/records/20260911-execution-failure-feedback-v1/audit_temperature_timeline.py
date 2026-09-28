"""CPU-only replay audit of the retained temperature pair; never loads a model.

Usage: existing-decoder/bin/python -B audit_temperature_timeline.py SNAPSHOT OUTPUT
Reuses the existing result and death-ledger audits. Iteration*4 is the replay
game-loop boundary; snapshots contain only events strictly before that boundary.
Command intents are not births. A death is evidence of combat, not first damage.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import runpy
import sys

from mpyq import MPQArchive
from sc2reader.engine.plugins import SelectionTracker
from s2protocol.versions import build, latest

FPS = 22.4


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def counts(units):
    return dict(Counter(u['type'] for u in units if not u['type'].startswith('Beacon')))


def snapshots(events, loops):
    units, result, index = {}, {}, 0
    for loop in sorted(set(loops)):
        while index < len(events) and events[index]['_gameloop'] < loop:
            e = events[index]
            kind = e['_event'].split('.')[-1]
            tag = (e.get('m_unitTagIndex'), e.get('m_unitTagRecycle'))
            if kind in ('SUnitBornEvent', 'SUnitInitEvent'):
                units[tag] = dict(type=e['m_unitTypeName'].decode(), owner=e['m_controlPlayerId'],
                                  complete=kind == 'SUnitBornEvent', alive=True)
            elif kind == 'SUnitDoneEvent':
                units[tag]['complete'] = True
            elif kind == 'SUnitDiedEvent':
                units[tag]['alive'] = False
            elif kind == 'SUnitTypeChangeEvent':
                units[tag]['type'] = e['m_unitTypeName'].decode()
            elif kind == 'SUnitOwnerChangeEvent':
                units[tag]['owner'] = e['m_controlPlayerId']
            index += 1
        result[str(loop)] = dict(second=loop/FPS, players={str(player): {
            'complete_alive': counts(u for u in units.values() if u['owner'] == player and u['alive'] and u['complete']),
            'incomplete_alive': counts(u for u in units.values() if u['owner'] == player and u['alive'] and not u['complete']),
        } for player in (1, 2)})
    return result


def first_wave(tracker, commands, attacks, proto, second_loop):
    """Reuse audit_wave_orders.py's selection decoding, only for wave 1."""
    launch = next(a for a in attacks if '已创建总攻波次 1，' in a['text'])
    rally = tuple(map(int, re.search(r'集结点: \((\d+), (\d+)\)', launch['text']).groups()))
    plugin, selection, last, candidates = SelectionTracker(), {i:[] for i in range(11)}, None, []
    for event in commands:
        if event['_gameloop'] > launch['loop']:
            break
        if event['_userid']['m_userId'] != 0:
            continue
        kind = event['_event'].split('.')[-1]
        if kind == 'SSelectionDeltaEvent':
            group, delta = event['m_controlGroupId'], event['m_delta']
            mode, data = next(iter(delta['m_removeMask'].items()))
            assert mode in ('None','ZeroIndices')
            previous, error = plugin._deselect(selection[group], mode, data)
            assert not error
            selection[group] = plugin._select(previous, delta['m_addUnitTags'])
        elif kind in ('SCmdUpdateTargetPointEvent','SCmdUpdateTargetUnitEvent'):
            last['m_data'] = {'TargetPoint' if kind=='SCmdUpdateTargetPointEvent' else 'TargetUnit':event['m_target']}
        elif kind in ('SCmdEvent','SCommandManagerStateEvent'):
            if kind == 'SCmdEvent':
                last = deepcopy(event)
            else:
                assert event['m_state']==1 and last is not None
            ability = last['m_abil']
            if event['_gameloop'] == launch['loop'] and ability and [ability['m_abilLink'],ability['m_abilCmdIndex']]==[45,0] and 'TargetPoint' in last['m_data']:
                point = last['m_data']['TargetPoint']
                if tuple(math.floor(point[k]/4096) for k in ('x','y')) == rally:
                    candidates.append(selection[10][:])
    assert candidates and all(set(c)<=set(candidates[0]) for c in candidates)
    tags, units, deaths = set(candidates[0]), {}, []
    for e in tracker:
        kind = e['_event'].split('.')[-1]
        tag = proto.unit_tag(e.get('m_unitTagIndex',0), e.get('m_unitTagRecycle',0))
        if kind in ('SUnitBornEvent','SUnitInitEvent'):
            units[tag] = dict(tag=tag, type=e['m_unitTypeName'].decode(), loop=e['_gameloop'], owner=e['m_controlPlayerId'])
        elif kind == 'SUnitTypeChangeEvent':
            units[tag]['type'] = e['m_unitTypeName'].decode()
        elif kind == 'SUnitDiedEvent' and tag in tags:
            killer_tag=proto.unit_tag(e.get('m_killerUnitTagIndex') or 0,e.get('m_killerUnitTagRecycle') or 0)
            deaths.append(dict(tag=tag, second=e['_gameloop']/FPS,x=e['m_x'],y=e['m_y'],killer=units.get(killer_tag,{}).get('type')))
    assert len(tags)==20 and all(units[t]['owner']==1 and units[t]['type']=='Marine' for t in tags)
    assert len(deaths)==20
    return dict(launch=launch, rally=rally, selected=sorted(tags),
                born_before_second=sum(units[t]['loop']<second_loop for t in tags),
                deaths=deaths, killers=dict(Counter(d['killer'] for d in deaths)))


def main(root, output):
    here = Path(__file__).parent
    summary = runpy.run_path(str(here/'audit_temperature_eval.py'))['audit'](root)
    decode = runpy.run_path(str(here.parent/'20260908-garrison-wave-ownership-v1/decode_failed_seed_replays.py'))['decode']
    games, common_trackers, common_commands, prefix_commands = [], [], [], []
    for arm, report in summary['arms'].items():
        run = Path(report['run'])
        result = json.loads((run/'result.json').read_text())
        for g in report['games']:
            log = run/g['executor_log']
            replay = log.parent/'replay.SC2Replay'
            assert hashlib.sha256(replay.read_bytes()).hexdigest() == result['replays'][str(replay.relative_to(run))]
            archive = MPQArchive(str(replay))
            header = latest().decode_replay_header(archive.header['user_data_header']['content'])
            assert header['m_version']['m_baseBuild'] == 75689
            proto = build(75689)
            tracker = list(proto.decode_replay_tracker_events(archive.read_file('replay.tracker.events')))
            commands = list(proto.decode_replay_game_events(archive.read_file('replay.game.events')))
            ledger = decode(replay)
            assert proto.decode_replay_details(archive.read_file('replay.details'))['m_playerList'][0]['m_result'] == 2
            second_loop = g['choices'][1]['queue']['iteration']*4
            first_common = g['choices'][0]['valid'] and g['choices'][0]['counts'] == {'supplydepot': 4, 'barracks': 4, 'marine': 24}
            attacks, scout, maps, intents, stats = [], [], [], [], []
            for line_no, line in enumerate(log.read_text().splitlines(), 1):
                match = re.search(r'\[INFO\]: \((\d+)\) (.*)', line)
                if not match:
                    continue
                iteration, text = int(match[1]), match[2]
                item = dict(loop=iteration*4, second=iteration*4/FPS, line=line_no, text=text)
                if any(s in text for s in ('已创建总攻波次', '集结完毕', '已全灭', 'ATTACK: Launch attack')):
                    attacks.append(item)
                if text.startswith('scouting:') and iteration*4 <= second_loop:
                    scout.append(item)
            trace = json.loads((log.parent/'trace.json').read_text())
            for iteration, record in trace.items():
                loop = int(iteration)*4
                for key, value in record.items():
                    if key.startswith('llm_decision_bo_map_'):
                        maps.append(dict(loop=loop, value=value, key=key))
                    if key.startswith('llm_decision_final_action_'):
                        intents.append(dict(loop=loop, value=value, key=key))
                if 'time_seconds' in record:
                    assert abs(record['time_seconds']-loop/FPS) < 1.01
                    stats.append(dict(loop=loop, **{k:record[k] for k in ('minerals','vespene','supply_army','supply_workers','supply_left')}))
            births = [dict(loop=e['_gameloop'], second=e['_gameloop']/FPS, unit=e['m_unitTypeName'].decode(), owner=e['m_controlPlayerId'],
                           kind=e['_event'].split('.')[-1], x=e['m_x'], y=e['m_y']) for e in tracker
                      if e['_event'].endswith(('SUnitBornEvent','SUnitInitEvent')) and e['m_controlPlayerId'] in (1,2)]
            own_marine_births = [e for e in births if e['owner'] == 1 and e['unit'] == 'Marine']
            combat_deaths = [d for d in ledger['deaths'] if d['owner'] in (1,2) and d['killer_owner'] == 3-d['owner']]
            pre = [d for d in ledger['deaths'] if d['second'] < second_loop/FPS]
            snapshot_loops = [second_loop, 1404*4, 1517*4, 300*FPS,360*FPS,420*FPS,600*FPS]
            snapshot_loops += [a['loop'] for a in attacks[:8]]
            windows = []
            for lo, hi in [(0,second_loop/FPS),(second_loop/FPS,300),(300,360),(360,420),(420,600)]:
                windows.append(dict(seconds=[lo,hi], marine_born=sum(lo <= b['second'] < hi for b in own_marine_births),
                    own_deaths=dict(Counter(d['birth_type'] for d in ledger['deaths'] if d['owner']==1 and lo<=d['second']<hi)),
                    enemy_killed=dict(Counter(d['birth_type'] for d in combat_deaths if d['owner']==2 and lo<=d['second']<hi))))
            prefix = [e for e in tracker if e['_gameloop'] < second_loop]
            command_prefix = [e for e in commands if e['_gameloop'] < second_loop]
            # Retain raw hashes; remove only the wall-clock sync field for a
            # separate comparison, never remove actual commands or game loops.
            normalized = [{k:v for k,v in e.items() if not(k=='m_syncTime' and e['_event'].endswith(
                ('SSetSyncPlayingTimeEvent','SSetSyncLoadingTimeEvent')))} for e in command_prefix]
            if first_common:
                assert second_loop == 1396*4
                common_trackers.append(prefix)
                common_commands.append(normalized)
                prefix_commands.append(command_prefix)
            games.append(dict(arm=arm, env=g['env'], first_plan_common=first_common, seconds=ledger['seconds'],
                second_plan_second=second_loop/FPS, choices=[{k:c[k] for k in ('prompt_game_time','valid','counts','prompt_sha256')} for c in g['choices']],
                raw_hashes={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (replay,log,log.parent/'trace.json')},
                prefix_tracker_hash=digest(prefix), prefix_command_hash=digest(command_prefix),
                prefix_command_without_wallclock_hash=digest(normalized), prefix_tracker_count=len(prefix),
                prefix_command_count=len(command_prefix), snapshots=snapshots(tracker,snapshot_loops),
                before_second_deaths=pre, first_cross_player_deaths=combat_deaths[:4],
                first_marine_death=next((d for d in combat_deaths if d['owner']==1 and d['birth_type']=='Marine'),None),
                before_second_scout_orders=scout,
                before_second_births=[b for b in births if b['loop']<second_loop and not b['unit'].startswith('Beacon')],
                own_total_born={k:v for k,v in ledger['own_born'].items() if not k.startswith('Beacon')},
                own_total_deaths=ledger['own_deaths'],
                marine_deaths=[d for d in ledger['deaths'] if d['owner']==1 and d['birth_type']=='Marine'],
                worker_deaths=[d for d in ledger['deaths'] if d['owner']==1 and d['birth_type']=='SCV'],
                windows=windows, attacks=attacks, marine_births=own_marine_births,
                post_second_born_marine_before_attack=sum(second_loop <= b['loop'] < next(a['loop'] for a in attacks if 'ATTACK: Launch attack' in a['text']) for b in own_marine_births),
                intents_after_second=[i for i in intents if second_loop<=i['loop']<300*FPS],
                trace_resources=[min(stats,key=lambda x:abs(x['loop']-loop)) for loop in (second_loop,300*FPS,360*FPS,420*FPS)],
                first_wave=first_wave(tracker,commands,attacks,proto,second_loop),
                outcome=g['outcome'], reward=g['reward']))
    assert len(games)==16 and len(common_trackers)==15
    comparisons = dict(matched_games=15, tracker_prefix_equal=all(t==common_trackers[0] for t in common_trackers),
        command_prefix_exact_equal=all(c==prefix_commands[0] for c in prefix_commands),
        command_prefix_equal_except_sync_wallclock=all(c==common_commands[0] for c in common_commands),
        first_wave_same_unit_tags=len({tuple(g['first_wave']['selected']) for g in games if g['first_plan_common']})==1)
    output.write_text(json.dumps(dict(source_sha=summary['source_sha'], snapshot=str(root), cpu_only=True,
        comparisons=comparisons, games=games,
        limitations='All games share one seed. Death times are not first damage times. Logs prove commands/intent; replay births prove completion. Timing cannot prove irrecoverability.'),
        ensure_ascii=False, indent=2)+'\n')
    print(comparisons)
    for g in games:
        print(g['arm'],g['env'],'first wave killers',g['first_wave']['killers'],
              'first wave born before second',g['first_wave']['born_before_second'])


if __name__ == '__main__':
    main(Path(sys.argv[1]),Path(sys.argv[2]))
