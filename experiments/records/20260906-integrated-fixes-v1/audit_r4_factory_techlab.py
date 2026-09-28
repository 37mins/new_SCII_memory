"""Read-only replay/trace evidence for the three failed Factory add-on attempts."""
import hashlib
import json
from pathlib import Path
import sys

from mpyq import MPQArchive
from s2protocol.versions import build


record = Path(__file__).resolve().parent
manifest = json.loads((record / 'r4-python-rng-preplanner.json').read_text())
protocol = build(75689)
reports = []
for repeat in (1, 2):
    folder = Path(sys.argv[1]) / f'repeat{repeat}'
    expected = manifest['reports'][repeat-1]
    hashes = {}
    for name in ('replay.SC2Replay', 'trace.json', 'driver.log'):
        hashes[name] = hashlib.sha256((folder/name).read_bytes()).hexdigest()
        assert hashes[name] == next(v for k,v in expected['artifact_hashes'].items() if k.endswith(name))
    archive = MPQArchive(str(folder/'replay.SC2Replay'))
    tracker = list(protocol.decode_replay_tracker_events(archive.read_file('replay.tracker.events')))
    game = list(protocol.decode_replay_game_events(archive.read_file('replay.game.events')))
    trace = json.loads((folder/'trace.json').read_text())

    def unit_events(index, recycle):
        return [{k:(v.decode() if isinstance(v, bytes) else v) for k,v in e.items()}
                for e in tracker if e.get('m_unitTagIndex') == index and e.get('m_unitTagRecycle') == recycle]

    def point_command(loop):
        e = next(e for e in game if e['_gameloop'] == loop
                 and e['_event'].endswith('SCmdEvent') and 'TargetPoint' in e['m_data'])
        p = e['m_data']['TargetPoint']
        return dict(loop=loop, second=loop/22.4, point=[p['x']/4096,p['y']/4096], event=e)

    factory, depot = point_command(8300), point_command(8336)
    factory_events, depot_events = unit_events(347,1), unit_events(351,1)
    assert factory_events[0]['m_unitTypeName'] == 'Factory'
    assert depot_events[0]['m_unitTypeName'] == 'SupplyDepot'
    assert factory['loop'] < depot['loop'] < factory_events[0]['_gameloop']
    for history in (factory_events,depot_events):
        assert next(e['_gameloop'] for e in history if e['_event'].endswith('SUnitDoneEvent')) < 10084
        assert all(e['_gameloop'] > 10112 for e in history if e['_event'].endswith('SUnitDiedEvent'))
    assert not any(e.get('m_unitTypeName') == 'FactoryFlying' for e in factory_events)
    assert not any(e.get('m_unitTypeName') == b'FactoryTechLab' for e in tracker)
    assert 'Factory' not in trace['2084']['obs']['Own structures']
    assert 'State: constructing [(26, 30)]' in trace['2084']['obs']['Own units']

    # Exact 3x3 Factory command center; the tracker records the 2x2 Depot
    # at its integer center. Offset/footprints are verified against burnysc2 7.1.3.
    addon = [factory['point'][0]+2.5, factory['point'][1]-.5]
    depot_center = [depot_events[0]['m_x'], depot_events[0]['m_y']]
    overlap = [max(0, min(addon[i]+1,depot_center[i]+1)-max(addon[i]-1,depot_center[i]-1))
               for i in (0,1)]
    attempts=[]
    for frame,bo in [(2521,123),(2527,129),(2528,130)]:
        item=trace[str(frame)]
        action=item[f'llm_decision_final_action_{bo}_{frame}']
        assert action == [{'action':'BUILD_TECHLAB_FACTORY','units':[266]}]
        loop=frame*4
        selection=next(e for e in game if e['_gameloop']==loop
                       and e['_event'].endswith('SSelectionDeltaEvent')
                       and e['m_delta']['m_addUnitTags'])
        assert selection['m_delta']['m_addUnitTags']==[protocol.unit_tag(347,1)]
        attempts.append(dict(bo=bo,iteration=frame,loop=loop,second=loop/22.4,
            action=action,replay_selection=selection,round_state=item['obs']['Round state']))

    reports.append(dict(run_id=expected['run_id'],hashes=hashes,
        factory_command=factory,depot_command=depot,
        factory_events=factory_events,depot_events=depot_events,
        addon_center=addon,depot_center=depot_center,overlap_size=overlap,
        pre_init_observation=trace['2084']['obs'],attempts=attempts))

print(json.dumps(dict(frozen_sha='5b24db566007b514686f61242ad999b5d296725b',
    scope='Offline geometry and command/initialization timeline; no original ActionResult available.',
    reports=reports),ensure_ascii=False,indent=2))
