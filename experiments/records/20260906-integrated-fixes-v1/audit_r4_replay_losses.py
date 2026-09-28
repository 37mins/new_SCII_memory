"""Offline s2protocol-75689 replay ledger; inputs are the two verified run copies."""
import hashlib
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path
import re
import sys

from mpyq import MPQArchive
from s2protocol.versions import build, latest

FPS = 22.4  # Matches frozen python-sc2 game time; not metadata's 16-loop Duration.
COMBAT = {'Marine', 'Marauder', 'SiegeTank', 'SiegeTankSieged', 'VikingFighter',
          'VikingAssault', 'Battlecruiser', 'Medivac', 'Banshee', 'Raven', 'Reaper',
          'Hellion', 'HellionTank', 'Thor', 'Liberator', 'Cyclone'}
ROOT = Path(__file__).resolve().parents[3]
manifest = json.loads((ROOT / 'experiments/records/20260906-integrated-fixes-v1/r4-python-rng-preplanner.json').read_text())
data = Path(sys.argv[1])
decoded, inputs = [], []
for repeat in (1, 2):
    folder = data / f'repeat{repeat}'
    expected = manifest['reports'][repeat-1]
    for name in ('replay.SC2Replay', 'driver.log', 'trace.json'):
        sha = next(v for k,v in expected['artifact_hashes'].items() if k.endswith(name))
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == sha
    archive = MPQArchive(str(folder/'replay.SC2Replay'))
    header = latest().decode_replay_header(archive.header['user_data_header']['content'])
    protocol = build(header['m_version']['m_baseBuild'])
    assert header['m_version']['m_baseBuild'] == 75689
    streams = {name: archive.read_file('replay.'+name+'.events') for name in ('tracker','game','message')}
    tracker = list(protocol.decode_replay_tracker_events(streams['tracker']))
    messages = list(protocol.decode_replay_message_events(streams['message']))
    games = list(protocol.decode_replay_game_events(streams['game']))
    decoded.append((tracker, messages, games))
    inputs.append(dict(run_id=expected['run_id'], replay_sha=next(v for k,v in expected['artifact_hashes'].items() if k.endswith('.SC2Replay')),
        stream_sha={k:hashlib.sha256(v).hexdigest() for k,v in streams.items()},
        event_counts=dict(Counter(e['_event'].split('.')[-1] for e in tracker)),
        base_build=header['m_version']['m_baseBuild'], duration_loops=header['m_elapsedGameLoops']))

assert inputs[0]['stream_sha']['tracker'] == inputs[1]['stream_sha']['tracker']
assert inputs[0]['stream_sha']['message'] == inputs[1]['stream_sha']['message']
tracker, messages, games = decoded[0]  # Identical tracker bytes: not two independent samples.
units, births, deaths, upgrades, stats, checkpoints = {}, [], [], [], [], []
sample_seconds = iter([180, 300, 359, 373, 400, 420, 443, 473, 495, 510, 540, 560, 583, 686])
next_sample = next(sample_seconds, None)

for e in tracker:
    sec = e['_gameloop']/FPS
    while next_sample is not None and sec > next_sample:
        checkpoints.append(dict(second=next_sample,
            own=dict(Counter(u['type'] for u in units.values() if u['owner']==1 and u['alive'] and u['complete'])),
            enemy=dict(Counter(u['type'] for u in units.values() if u['owner']==2 and u['alive'] and u['complete']))))
        next_sample = next(sample_seconds, None)
    kind = e['_event'].split('.')[-1]
    tag = (e.get('m_unitTagIndex'),e.get('m_unitTagRecycle'))
    if kind in ('SUnitBornEvent','SUnitInitEvent'):
        units[tag] = dict(type=e['m_unitTypeName'].decode(), owner=e['m_controlPlayerId'],
            alive=True, complete=kind=='SUnitBornEvent', born_loop=e['_gameloop'])
        if e['m_controlPlayerId'] in (1,2):
            births.append(dict(loop=e['_gameloop'], second=round(sec,3), tag=list(tag),
                type=units[tag]['type'], owner=units[tag]['owner'], x=e['m_x'], y=e['m_y'],
                event=kind))
    elif kind == 'SUnitDoneEvent':
        units[tag]['complete'] = True
    elif kind == 'SUnitTypeChangeEvent':
        units[tag]['type'] = e['m_unitTypeName'].decode()
    elif kind == 'SUnitDiedEvent':
        assert tag in units, ('untracked death',e)
        u = units[tag]
        killer_tag=(e.get('m_killerUnitTagIndex'),e.get('m_killerUnitTagRecycle'))
        killer = units.get(killer_tag,{})
        if u['owner'] in (1,2):
            deaths.append(dict(loop=e['_gameloop'], second=round(sec,3), tag=list(tag),
                owner=u['owner'], type=u['type'], complete=u['complete'], x=e['m_x'], y=e['m_y'],
                killer_player=e.get('m_killerPlayerId'), killer_type=killer.get('type'),
                killer_tag=list(killer_tag), born_loop=u['born_loop']))
        u['alive']=False
    elif kind == 'SUpgradeEvent':
        upgrades.append(dict(second=round(sec,3), player=e['m_playerId'], upgrade=e['m_upgradeTypeName'].decode()))
    elif kind == 'SPlayerStatsEvent':
        stats.append(dict(second=round(sec,3), player=e['m_playerId'], **e['m_stats']))

chats = [dict(loop=e['_gameloop'],second=round(e['_gameloop']/FPS,3),text=e['m_string'].decode())
         for e in messages if e['_event'].endswith('SChatMessage')]
windows=[]
for low, high in [(0,359),(359,400),(400,443),(443,473),(473,510),(510,540),(540,560),(560,600),(600,687)]:
    own=[d for d in deaths if d['owner']==1 and low<=d['second']<high and d['type'] in COMBAT|{'SCV'}]
    enemy=[d for d in deaths if d['owner']==2 and low<=d['second']<high]
    windows.append(dict(from_second=low,to_second=high,own_losses=dict(Counter(d['type'] for d in own)),
        killers=dict(Counter(str(d['killer_type']) for d in own)),
        enemy_losses=dict(Counter(d['type'] for d in enemy)),
        own_loss_positions=[[d['second'],d['type'],d['x'],d['y'],d['killer_type']] for d in own]))

log=(data/'repeat1/driver.log').read_text()
log=re.sub(r'\x1b\[[0-9;]*m','',log)
key_lines=[dict(line=i+1,text=l) for i,l in enumerate(log.splitlines()) if any(s in l for s in (
    '[Auto-Attack]','已创建总攻波次','集结完毕','集结后仅剩','已全灭','ATTACK:',
    'Automatic Defense:', 'bo_skipped_', 'Start Processing BO #165', 'Start Processing BO #172',
    'BUILD_TECHLAB_FACTORY', 'Tech Lab required'))]

# These two files only contain append/clear selection deltas. Keep explicit
# SCmdEvent records, not a claim to reconstruct compressed/reused commands.
tag_types = {protocol.unit_tag(*b['tag']): (b['type'], b['owner']) for b in births}
selected, worker_commands, factory_events = [], [], []
for e in games:
    kind = e['_event'].split('.')[-1]
    if kind == 'SSelectionDeltaEvent':
        assert e['m_controlGroupId'] == 10 and e['_userid']['m_userId'] == 0
        mask = e['m_delta']['m_removeMask']
        assert mask in ({'None': None}, {'ZeroIndices': []}), mask
        if mask == {'ZeroIndices': []}:
            selected = []
        selected.extend(e['m_delta']['m_addUnitTags'])
    if 10080 <= e['_gameloop'] <= 10120:
        factory_events.append(e)
    if (kind == 'SCmdEvent' and 515 < e['_gameloop']/FPS < 562
            and any(tag_types.get(t) == ('SCV',1) for t in selected)):
        worker_commands.append(dict(second=round(e['_gameloop']/FPS,3),
            selected_replay_tags=selected.copy(), event=e))

trace=json.loads((data/'repeat1/trace.json').read_text())
factory_trace={k:trace[k] for k in ('2521','2527','2528')}
loss_counts={kind:dict(Counter(d['killer_type'] for d in deaths
    if d['owner']==1 and d['type']==kind)) for kind in ('Marine','SCV')}
late_marines=[dict(tag=d['tag'],born_loop=d['born_loop'],death_loop=d['loop'],
    life_seconds=(d['loop']-d['born_loop'])/FPS) for d in deaths
    if d['owner']==1 and d['type']=='Marine' and d['born_loop']/FPS>=500]
print(json.dumps(dict(inputs=inputs, decoder=dict(s2protocol=version('s2protocol'),mpyq=version('mpyq'),
    protocol_sha=hashlib.sha256(Path(protocol.__file__).read_bytes()).hexdigest(),loops_per_second=FPS),
    identical_tracker_and_message_streams=True,
    own_born=dict(Counter(b['type'] for b in births if b['owner']==1 and b['event']=='SUnitBornEvent')),
    own_constructed_starts=dict(Counter(b['type'] for b in births if b['owner']==1 and b['event']=='SUnitInitEvent')),
    own_deaths=dict(Counter(d['type'] for d in deaths if d['owner']==1)),
    own_killers=dict(Counter(str(d['killer_type']) for d in deaths if d['owner']==1 and d['type'] in COMBAT|{'SCV'})),
    killers_by_victim_type=loss_counts, late_marine_lifetimes=late_marines,
    explicit_late_worker_commands=worker_commands, factory_game_events=factory_events,
    factory_trace=factory_trace,
    checkpoints=checkpoints,windows=windows,upgrades=upgrades,chats=chats,log_events=key_lines,
    births=births,deaths=deaths,final_stats=[next(s for s in reversed(stats) if s['player']==i) for i in (1,2)]),
    ensure_ascii=False,separators=(',',':')))
