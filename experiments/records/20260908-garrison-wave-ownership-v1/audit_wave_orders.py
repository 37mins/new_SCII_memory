"""Offline replay/log comparison; no model, SC2, or runtime modifications.

Reuses the selection/command decoding validated in the preceding Viking audit.
Does not assume a Viking count, a win, or successful gathering in either arm.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import sys

from mpyq import MPQArchive
from sc2reader.engine.plugins import SelectionTracker
from s2protocol.versions import build, latest


def point(command):
    return tuple(command['target']['TargetPoint'][k] / 4096 for k in ('x', 'y'))


def audit(run):
    result = json.loads((run / 'result.json').read_text())
    replay = next((run / 'sc2-logs').rglob('*.SC2Replay'))
    replay_sha = hashlib.sha256(replay.read_bytes()).hexdigest()
    assert replay_sha == next(iter(result['replays'].values()))
    assert hashlib.sha256((run / 'console.log').read_bytes()).hexdigest() == result['console_sha256']
    archive = MPQArchive(str(replay))
    header = latest().decode_replay_header(archive.header['user_data_header']['content'])
    protocol = build(header['m_version']['m_baseBuild'])
    assert header['m_version']['m_baseBuild'] == 75689
    end_loop = header['m_elapsedGameLoops']
    units, bases = {}, set()
    for event in protocol.decode_replay_tracker_events(archive.read_file('replay.tracker.events')):
        kind, loop = event['_event'].split('.')[-1], event['_gameloop']
        tag = protocol.unit_tag(event.get('m_unitTagIndex', 0), event.get('m_unitTagRecycle', 0))
        if kind in ('SUnitBornEvent', 'SUnitInitEvent'):
            name = event['m_unitTypeName'].decode()
            units[tag] = dict(tag=tag, type=name, owner=event['m_controlPlayerId'],
                              born_loop=loop, born_xy=[event['m_x'], event['m_y']],
                              death_loop=None)
            if name == 'CommandCenter' and event['m_controlPlayerId'] == 1:
                # Tracker construction coordinates are tile centres rounded down.
                bases.add((event['m_x'] + .5, event['m_y'] + .5))
        elif kind == 'SUnitDiedEvent':
            killer = protocol.unit_tag(event.get('m_killerUnitTagIndex') or 0, event.get('m_killerUnitTagRecycle') or 0)
            units[tag].update(death_loop=loop, death_xy=[event['m_x'], event['m_y']],
                              killer_type=units.get(killer, {}).get('type'))

    selection = {i: [] for i in range(11)}
    plugin, last, commands, sequences = SelectionTracker(), None, [], set()
    for event in protocol.decode_replay_game_events(archive.read_file('replay.game.events')):
        if event['_userid']['m_userId'] != 0:
            continue
        kind = event['_event'].split('.')[-1]
        if kind == 'SSelectionDeltaEvent':
            group, delta = event['m_controlGroupId'], event['m_delta']
            mode, data = next(iter(delta['m_removeMask'].items()))
            assert mode in ('None', 'ZeroIndices')
            previous, error = plugin._deselect(selection[group], mode, data)
            assert not error
            selection[group] = plugin._select(previous, delta['m_addUnitTags'])
        elif kind in ('SCmdUpdateTargetPointEvent', 'SCmdUpdateTargetUnitEvent'):
            assert last is not None
            key = 'TargetPoint' if kind == 'SCmdUpdateTargetPointEvent' else 'TargetUnit'
            last['m_data'] = {key: event['m_target']}
        elif kind in ('SCmdEvent', 'SCommandManagerStateEvent'):
            if kind == 'SCmdEvent':
                last = deepcopy(event)
            else:
                assert event['m_state'] == 1 and last is not None
            sequence = event['m_sequence']
            assert sequence not in sequences
            sequences.add(sequence)
            ability = last['m_abil']
            commands.append(dict(loop=event['_gameloop'], sequence=sequence,
                selected=selection[10][:], target=deepcopy(last['m_data']),
                ability=[ability['m_abilLink'], ability['m_abilCmdIndex']] if ability else None))

    lines = [re.sub(r'\x1b\[[0-9;]*m', '', s) for s in (run / 'console.log').read_text().splitlines()]
    waves, lifecycle = {}, []
    for number, line in enumerate(lines, 1):
        frame = re.search(r'\]: \((\d+)\)', line)
        if not frame:
            continue
        loop = int(frame[1]) * 4
        created = re.search(r'已创建总攻波次 (\d+)，状态: GATHERING，集结点: \((\d+), (\d+)\)', line)
        if created:
            wave = int(created[1]); xy = [int(created[2]), int(created[3])]
            selected = [c for c in commands if c['loop'] == loop and c['ability'] == [45, 0]
                        and 'TargetPoint' in c['target'] and [math.floor(v) for v in point(c)] == xy]
            # Launch orders the whole group before same-frame maintenance may
            # repeat that exact target for an idle subset. Reject other ambiguity.
            assert selected, (wave, loop)
            assert all(point(c) == point(selected[0]) and
                       set(c['selected']) <= set(selected[0]['selected']) for c in selected), (wave, selected)
            waves[wave] = dict(wave=wave, start_loop=loop, end_loop=end_loop + 1,
                              selected=selected[0]['selected'], rally=point(selected[0]),
                              reached_attacking=False, console_line=number)
        match = re.search(r'总攻波次 (\d+)', line)
        if match and int(match[1]) in waves:
            wave = waves[int(match[1])]
            if '已全灭' in line or '取消本波' in line:
                wave['end_loop'] = loop
            if '集结完毕' in line:
                wave['reached_attacking'] = True
            if any(x in line for x in ('已创建', '已全灭', '取消本波', '集结完毕')):
                lifecycle.append(dict(console_line=number, loop=loop, text=line))

    # The same Flat128 map: centre (76,74), previously verified from replay rally rays.
    garrison_points = []
    for x, y in bases:
        dx, dy = 76 - x, 74 - y
        length = math.hypot(dx, dy); nx, ny = dx / length, dy / length
        for ox, oy in ((15*nx,15*ny), (5*nx,5*ny),
                       (7.5*nx-15*ny,7.5*ny+15*nx), (7.5*nx+15*ny,7.5*ny-15*nx)):
            garrison_points.append((x+ox, y+oy))
    moves, conflicts, unknown_periodic_moves = [], [], []
    for command in commands:
        if command['ability'] != [42, 0] or (command['loop'] // 4 + 1) % 67:
            continue
        if 'TargetPoint' not in command['target']:
            continue
        combat = [t for t in command['selected'] if t in units and units[t]['owner'] == 1
                  and units[t]['type'] in ('VikingFighter','Marine','Marauder','SiegeTank','Hellion')]
        if not combat:
            continue
        if not any(math.dist(point(command), p) < .001 for p in garrison_points):
            unknown_periodic_moves.append(command)
            continue
        for tag in combat:
            active = [w['wave'] for w in waves.values() if tag in w['selected']
                      and w['start_loop'] <= command['loop'] < w['end_loop']]
            row = dict(tag=tag, type=units[tag]['type'], loop=command['loop'],
                       sequence=command['sequence'], xy=point(command), active_waves=active)
            moves.append(row)
            if active:
                conflicts.append(row)
    vikings = []
    for tag, unit in units.items():
        if unit['owner'] != 1 or unit['type'] != 'VikingFighter':
            continue
        vk = dict(unit)
        vk['lifetime_seconds'] = ((unit['death_loop'] or end_loop) - unit['born_loop']) / 22.4
        vk['commands'] = [c for c in commands if tag in c['selected']]
        vk['waves'] = [w['wave'] for w in waves.values() if tag in w['selected']]
        vikings.append(vk)
    peak = max((sum(u['born_loop'] <= v['born_loop'] < (u['death_loop'] or end_loop+1)
                    for u in vikings) for v in vikings), default=0)
    trace = json.loads(next((run / 'sc2-logs').rglob('trace.json')).read_text())
    trace_train = sorted(int(k)*4 for k,row in trace.items() for name,value in row.items()
                         if 'final_action' in name and 'STARPORTTRAIN_VIKINGFIGHTER' in str(value))
    replay_train = [c['loop'] for c in commands if c['ability'] == [161,4]]
    assert trace_train == replay_train
    return dict(run=run.name, replay_sha256=replay_sha, console_sha256=result['console_sha256'],
                game_seconds=end_loop/22.4, commands=commands,
                waves=list(waves.values()), lifecycle=lifecycle,
                garrison_moves=moves, active_wave_garrison_conflicts=conflicts,
                unmatched_periodic_combat_moves=unknown_periodic_moves,
                vikings=vikings, viking_peak_alive=peak, viking_production_commands=len(trace_train),
                viking_deaths=sum(u['death_loop'] is not None for u in vikings),
                viking_killers=dict(Counter(u.get('killer_type') for u in vikings if u['death_loop'] is not None)),
                caveat='Submitted commands, not continuous positions. Garrison attribution combines 67-frame cadence and exact destination geometry. Active-wave membership ends at logged deletion; ambiguous wave-command matches fail. This decoder only covers the frozen old-micro GATHERING waves, not new-micro DIRECT/merge modes.')


if __name__ == '__main__':
    print(json.dumps([audit(Path(p)) for p in sys.argv[1:]], ensure_ascii=False, indent=2))
