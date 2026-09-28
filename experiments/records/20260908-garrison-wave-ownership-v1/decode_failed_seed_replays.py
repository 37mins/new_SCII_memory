"""Read two saved replays with the existing s2protocol tracker ledger method.

Usage: decoder/bin/python decode_failed_seed_replays.py SNAPSHOT_ROOT OUTPUT_JSON
One replay survived per batch because existing same-second paths collide. Match
the retained replay to its unique worker duration; do not claim 16 replay traces.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from mpyq import MPQArchive
from s2protocol.versions import build, latest

FPS = 22.4


def decode(path):
    archive = MPQArchive(str(path))
    header = latest().decode_replay_header(archive.header['user_data_header']['content'])
    assert header['m_version']['m_baseBuild'] == 75689
    protocol = build(75689)
    events = list(protocol.decode_replay_tracker_events(archive.read_file('replay.tracker.events')))
    units, deaths, upgrades, snapshots, stats = {}, [], [], [], []
    requested = iter([300,420,540,592,600,660,720,780,840,960,1080,1200,1320])
    sample = next(requested,None)
    for e in events:
        second=e['_gameloop']/FPS
        while sample is not None and sample<second:
            snapshots.append(dict(second=sample, players={str(player):dict(Counter(u['type'] for u in units.values()
                if u['owner']==player and u['alive'] and u['complete'])) for player in (1,2)}))
            sample=next(requested,None)
        kind=e['_event'].split('.')[-1]
        tag=(e.get('m_unitTagIndex'),e.get('m_unitTagRecycle'))
        if kind in ('SUnitBornEvent','SUnitInitEvent'):
            units[tag]=dict(type=e['m_unitTypeName'].decode(), birth_type=e['m_unitTypeName'].decode(),
                owner=e['m_controlPlayerId'], alive=True, complete=kind=='SUnitBornEvent',
                second=second, event=kind, x=e['m_x'],y=e['m_y'])
        elif kind=='SUnitDoneEvent':
            units[tag]['complete']=True
        elif kind=='SUnitTypeChangeEvent':
            units[tag]['type']=e['m_unitTypeName'].decode()
        elif kind=='SUnitDiedEvent':
            u=units[tag]
            killer=units.get((e.get('m_killerUnitTagIndex'),e.get('m_killerUnitTagRecycle')), {})
            deaths.append(dict(second=round(second,3), owner=u['owner'], type=u['type'], birth_type=u['birth_type'],
                born_second=u['second'], x=e['m_x'],y=e['m_y'],killer=killer.get('type'),killer_owner=killer.get('owner')))
            u['alive']=False
        elif kind=='SUpgradeEvent':
            upgrades.append(dict(second=second,player=e['m_playerId'],name=e['m_upgradeTypeName'].decode()))
        elif kind=='SPlayerStatsEvent':
            stats.append(dict(second=round(second,3),player=e['m_playerId'],stats=e['m_stats']))
    windows=[]
    for lo,hi in [(0,592),(592,720),(720,840),(840,960),(960,1200),(1200,1500)]:
        own=[d for d in deaths if d['owner']==1 and lo<=d['second']<hi]
        windows.append(dict(seconds=[lo,hi],losses=dict(Counter(d['birth_type'] for d in own)),
            worker_killers=dict(Counter(str(d['killer']) for d in own if d['birth_type']=='SCV')),
            army_killers=dict(Counter(str(d['killer']) for d in own if d['birth_type'] in ('Marine','Marauder','SiegeTank','VikingFighter')))))
    prefix=[e for e in events if e['_gameloop']<=3316*4]
    return dict(replay=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        seconds=header['m_elapsedGameLoops']/FPS, tracker_prefix_sha256=hashlib.sha256(json.dumps(prefix,sort_keys=True,default=str).encode()).hexdigest(),
        own_born=dict(Counter(u['birth_type'] for u in units.values() if u['owner']==1 and u['event']=='SUnitBornEvent')),
        own_deaths=dict(Counter(d['birth_type'] for d in deaths if d['owner']==1)),
        snapshots=snapshots,windows=windows,upgrades=upgrades,deaths=deaths,
        own_stats=[s for s in stats if s['player']==1])


if __name__=='__main__':
    root=Path(sys.argv[1])
    audit=json.loads((root/'analysis.json').read_text())
    result={arm:decode(root/arm/'replay.SC2Replay') for arm in ('original-final','after-step1')}
    for arm,r in result.items():
        matching=[i for i,e in audit['reports'][arm]['episodes'].items() if e['worker_evidence']['game_seconds']==int(r['seconds'])]
        assert len(matching)==1,(arm,matching)
        r['matched_env_by_unique_duration']=int(matching[0])
        print(arm,'env',matching[0],'duration',r['seconds'],'born',r['own_born'],'deaths',r['own_deaths'])
        print('snapshots',[s for s in r['snapshots'] if s['second'] in (592,720,840)])
        print('windows',r['windows'])
        print('own_upgrades',[u for u in r['upgrades'] if u['player']==1])
    expected=json.loads((root/'original-final'/'result.json').read_text())['replays']
    assert result['original-final']['sha256'] in expected.values()
    result['prefix_trackers_equal']=result['original-final']['tracker_prefix_sha256']==result['after-step1']['tracker_prefix_sha256']
    print('prefix_trackers_equal',result['prefix_trackers_equal'])
    Path(sys.argv[2]).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
