"""Offline only: link tracker lifetimes, replay selection/commands and frozen logs."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys
import math
import importlib.metadata
from mpyq import MPQArchive
from s2protocol.versions import build, latest
from sc2reader.engine.plugins import SelectionTracker

if len(sys.argv) != 3:
    raise SystemExit('usage: audit_vikings.py COMPLETED_RUN OUTPUT_JSON')
run = Path(sys.argv[1])
output = Path(sys.argv[2])
replay = next((run / 'sc2-logs').rglob('*.SC2Replay'))
result = json.loads((run / 'result.json').read_text())
assert hashlib.sha256(replay.read_bytes()).hexdigest() == next(iter(result['replays'].values()))
assert hashlib.sha256((run / 'console.log').read_bytes()).hexdigest() == result['console_sha256']
archive = MPQArchive(str(replay))
header = latest().decode_replay_header(archive.header['user_data_header']['content'])
protocol = build(header['m_version']['m_baseBuild'])
assert header['m_version']['m_baseBuild'] == 75689
tracker = list(protocol.decode_replay_tracker_events(archive.read_file('replay.tracker.events')))
game = list(protocol.decode_replay_game_events(archive.read_file('replay.game.events')))

def stamp(loop):
    seconds = loop / 22.4
    return f'{int(seconds)//60:02d}:{seconds%60:04.1f}'

units, by_index, vikings, stars, positions = {}, {}, {}, [], {}
for event in tracker:
    kind = event['_event'].split('.')[-1]
    loop = event['_gameloop']
    tag = protocol.unit_tag(event.get('m_unitTagIndex', 0), event.get('m_unitTagRecycle', 0))
    if kind in ('SUnitBornEvent', 'SUnitInitEvent'):
        unit = dict(tag=tag, type=event['m_unitTypeName'].decode(), owner=event['m_controlPlayerId'],
                    born_loop=loop, born_time=stamp(loop), born_xy=[event['m_x'], event['m_y']])
        units[tag] = unit
        by_index[event['m_unitTagIndex']] = tag
        positions[tag] = [loop, *unit['born_xy']]
        if unit['owner'] == 1 and unit['type'] == 'VikingFighter':
            vikings[tag] = dict(unit, ordinal=len(vikings)+1, positions=[], commands=[], scout_logs=[])
        if unit['owner'] == 1 and unit['type'] == 'Starport':
            stars.append(unit)
    elif kind == 'SUnitTypeChangeEvent':
        units[tag]['type'] = event['m_unitTypeName'].decode()
    elif kind == 'SUnitPositionsEvent':
        index = event['m_firstUnitIndex']
        data = event['m_items']
        assert len(data) % 3 == 0
        for i in range(0, len(data), 3):
            index += data[i]
            if index in by_index:
                current = by_index[index]
                positions[current] = [loop, data[i+1], data[i+2]]
                if current in vikings:
                    vikings[current]['positions'].append([loop, data[i+1], data[i+2]])
    elif kind == 'SUnitDiedEvent':
        units[tag]['death_loop'] = loop
        if tag in vikings:
            killer_tag = protocol.unit_tag(event.get('m_killerUnitTagIndex',0), event.get('m_killerUnitTagRecycle',0))
            vikings[tag].update(death_loop=loop, death_time=stamp(loop),
                                lifetime_seconds=round((loop-vikings[tag]['born_loop'])/22.4,2),
                                death_xy=[event['m_x'],event['m_y']],
                                killer_tag=killer_tag, killer_type=units.get(killer_tag,{}).get('type'),
                                killer_last_position=positions.get(killer_tag))
assert len(vikings) == 12 and all('death_loop' in u for u in vikings.values())

selection = {i: [] for i in range(11)}
plugin = SelectionTracker()
last_command = None
all_commands = []
sequences = set()
for index, event in enumerate(game):
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
    elif kind == 'SCmdUpdateTargetPointEvent':
        assert last_command is not None
        last_command['m_data'] = {'TargetPoint':event['m_target']}
    elif kind == 'SCmdUpdateTargetUnitEvent':
        assert last_command is not None
        last_command['m_data'] = {'TargetUnit':event['m_target']}
    elif kind in ('SCmdEvent', 'SCommandManagerStateEvent'):
        if kind == 'SCmdEvent':
            last_command = deepcopy(event)
        else:
            assert event['m_state'] == 1 and last_command is not None
        sequence = event['m_sequence']
        assert sequence not in sequences
        sequences.add(sequence)
        ability = last_command['m_abil']
        command = dict(loop=event['_gameloop'], time=stamp(event['_gameloop']), game_event_index=index,
                       sequence=sequence, event_kind=kind, selected=selection[10][:],
                       ability=[ability['m_abilLink'],ability['m_abilCmdIndex']] if ability else None,
                       target=deepcopy(last_command['m_data']), flags=last_command['m_cmdFlags'])
        all_commands.append(command)
        for selected in selection[10]:
            if selected in vikings:
                assert vikings[selected]['born_loop'] <= command['loop'] <= vikings[selected]['death_loop']
                assert command['ability'] in ([40,0],[42,0],[45,0]),command
                vikings[selected]['commands'].append(command)

lines = (run/'console.log').read_text().splitlines()
wave_logs, scouts = [], []
for i, line in enumerate(lines):
    clean = re.sub(r'\x1b\[[0-9;]*m','',line)
    frame_match = re.search(r'\[INFO\]: \((\d+)\)',clean)
    if 'launch_total_attack: 已创建总攻波次' in clean:
        wave_logs.append(dict(console_line=i+1, iteration=int(frame_match[1]), text=clean))
    scout = re.search(r'正在派遣 VIKINGFIGHTER \(Tag: (\d+)\) 侦察 \((\d+), (\d+)\)',clean)
    if scout:
        tag = int(scout[1]) & 0xffffffff
        assert tag in vikings
        next_line = re.sub(r'\x1b\[[0-9;]*m','',lines[i+1])
        frame = int(re.search(r'\[INFO\]: \((\d+)\)',next_line)[1])
        record = dict(console_line=i+1, api_tag=int(scout[1]), iteration=frame, target=[int(scout[2]),int(scout[3])])
        vikings[tag]['scout_logs'].append(record)
        scouts.append(record)

report = dict(replay_sha256=hashlib.sha256(replay.read_bytes()).hexdigest(),
              console_sha256=result['console_sha256'], source_sha='3adaaf70ac879a7f4d7bdecbfb8be60b75bf4fa6',
              tracker_events=len(tracker), game_events=len(game), command_sequences=len(sequences),
              selection_errors=0, starports=stars, wave_logs=wave_logs, vikings=list(vikings.values()))

# Independently check replay command expansion against all logged production commands.
trace_path = next((run / 'sc2-logs').rglob('trace.json'))
trace = json.loads(trace_path.read_text())
trace_train = sorted(int(k)*4 for k,row in trace.items() for name,value in row.items()
                     if 'final_action' in name and 'STARPORTTRAIN_VIKINGFIGHTER' in str(value))
star_commands = [c for c in all_commands if any(s['tag'] in c['selected'] for s in stars)]
replay_train = [c['loop'] for c in star_commands if c['ability'] == [161,4]]
assert trace_train == replay_train and len(trace_train) == 14
report['starport_commands'] = star_commands
report['production_crosscheck'] = dict(trace_loops=trace_train, replay_loops=replay_train)

# Trace IDs are not replay tags. Match first appearances only if exactly one
# unassigned own Viking is alive; refuse an ambiguous association.
trace_ids = {}
for key,row in sorted(trace.items(), key=lambda item:int(item[0])):
    if 'obs' not in row:
        continue
    loop = int(key)*4
    blocks = re.findall(r'^\[(\d+)\]VikingFighter\n(.*?)(?=^\[|\Z)',
                        row['obs']['Own units'], re.M | re.S)
    for text_id, block in blocks:
        if text_id not in trace_ids:
            candidates = [tag for tag,u in vikings.items()
                          if tag not in trace_ids.values() and u['born_loop'] <= loop < u['death_loop']]
            assert len(candidates) == 1, (key,text_id,candidates)
            trace_ids[text_id] = candidates[0]
        unit = vikings[trace_ids[text_id]]
        assert unit['born_loop'] <= loop < unit['death_loop']
        unit.setdefault('trace_observations', []).append(
            dict(iteration=int(key), loop=loop, time=stamp(loop), trace_id=int(text_id), text=block.strip()))
report['trace_id_mapping'] = trace_ids
report['trace_sha256'] = hashlib.sha256(trace_path.read_bytes()).hexdigest()

# Exact iteration links the automatic-wave log to the selected unit tags.
for wave in wave_logs:
    wave_id = int(re.search(r'波次 (\d+)，状态: GATHERING',wave['text'])[1])
    if wave_id in (11,12,13,16):
        commands = [c for c in all_commands if c['loop'] == wave['iteration']*4 and c['ability']==[45,0]]
        assert len(commands) == 1
        wave['selected_types'] = [units[t]['type'] for t in commands[0]['selected']]
        for tag in commands[0]['selected']:
            if tag in vikings:
                assert 'automatic_wave' not in vikings[tag]
                vikings[tag]['automatic_wave'] = wave_id
                vikings[tag]['wave_join_loop'] = commands[0]['loop']
report['wave_lifecycle_logs'] = [
    re.sub(r'\x1b\[[0-9;]*m','',line) for line in lines
    if re.search(r'波次 (11|12|13|16)\D',line)]
report['defense_logs'] = [
    re.sub(r'\x1b\[[0-9;]*m','',line) for line in lines
    if 'Automatic Defense:' in line and re.search(r'\((4720|4956)\)',line)]
report['automatic_trigger_context'] = [
    [re.sub(r'\x1b\[[0-9;]*m','',s) for s in lines[i-2:i+1]]
    for i,line in enumerate(lines)
    if re.search(r'已创建总攻波次 (11|12|13|16)，',line)]

# Reconstruct the common map centre from two recorded production-rally rays.
# Do not assume the textual 128x128 playable size specifies absolute coordinates.
rallies = [c for c in star_commands if c['ability']==[102,0]]
assert len(rallies) == 2
a,b = (24.5,22.5),(49.5,21.5)
point = lambda c: tuple(c['target']['TargetPoint'][k]/4096 for k in ('x','y'))
pa,pb = point(rallies[0]),point(rallies[1])
u,v = (pa[0]-a[0],pa[1]-a[1]),(pb[0]-b[0],pb[1]-b[1])
cross = lambda x,y: x[0]*y[1]-x[1]*y[0]
t = cross((b[0]-a[0],b[1]-a[1]),v)/cross(u,v)
derived_centre = (a[0]+t*u[0],a[1]+t*u[1])
assert all(abs(x-y)<0.001 for x,y in zip(derived_centre,(76,74)))
garrison_points=[]
for centre in (a,(25.5,45.5),b):
    dx,dy = 76-centre[0],74-centre[1]
    length=math.hypot(dx,dy)
    nx,ny=dx/length,dy/length
    offsets=[(15*nx,15*ny),(5*nx,5*ny),
             (7.5*nx-15*ny,7.5*ny+15*nx),(7.5*nx+15*ny,7.5*ny-15*nx)]
    for i,(dx,dy) in enumerate(offsets):
        garrison_points.append(dict(base=centre, point=i+1, xy=(centre[0]+dx,centre[1]+dy)))
report['garrison_geometry'] = dict(centre_from_two_rallies=derived_centre, points=garrison_points)
for unit in vikings.values():
    unit['garrison_commands']=[]
    for command in unit['commands']:
        if command['ability']==[42,0] and (command['loop']//4+1)%67==0:
            xy = point(command)
            matched=[p for p in garrison_points if math.dist(xy,p['xy'])<0.001]
            assert len(matched)==1
            unit['garrison_commands'].append(dict(sequence=command['sequence'], time=command['time'],
                base=matched[0]['base'], point=matched[0]['point'], xy=xy,
                after_wave_join=command['loop']>=unit.get('wave_join_loop',float('inf'))))
        if 'TargetUnit' in command['target']:
            tag=command['target']['TargetUnit']['m_tag']
            command['target_tracker_type']=units[tag]['type']
    for scout in unit['scout_logs']:
        matches=[c for c in unit['commands'] if c['loop']==scout['iteration']*4 and c['ability']==[42,0]]
        assert len(matches)==1
        assert [int(n) for n in point(matches[0])]==scout['target']
assert [u['ordinal'] for u in vikings.values() if any(c['after_wave_join'] for c in u['garrison_commands'])] == [1,2,3]
assert [u['ordinal'] for u in vikings.values() if u['scout_logs']] == [8,9]
report['decoder_versions']={name:importlib.metadata.version(name) for name in ('s2protocol','mpyq','sc2reader')}
report['limitations'] = [
    'Tracker death events identify the last killer, not every attacker or damage share.',
    'Sparse positions and trace snapshots do not provide continuous damage/path history.',
    'Commands are submitted actions; trace observations independently confirm sampled engine states.',
    'Viking 12 has no live trace snapshot; its lifetime/death and production rally are recorded.',
    'No intervention or win-rate causal estimate; runtime unchanged.'
]
output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(vikings=len(vikings), observations=sum(len(u.get('trace_observations',[])) for u in vikings.values()),
      production_matches=len(trace_train), scout_ordinals=[8,9], wave_garrison_conflict_ordinals=[1,2,3]),ensure_ascii=False))
