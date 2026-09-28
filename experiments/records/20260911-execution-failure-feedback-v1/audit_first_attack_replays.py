"""CPU-only eight-replay audit of first-attack20/40; never changes runtime.

Usage: decoder/bin/python -B audit_first_attack_replays.py SNAPSHOT OUTPUT.json [SUMMARY.json]
Selection decoding follows the existing audit_wave_orders.py; death/snapshot
decoding reuses existing audits. Replay last-hit attribution is not total damage.
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
from audit_temperature_timeline import snapshots, digest

SHA = 'e3c0695adf0a2e36c97afba838ec3b508f000e34'
FPS = 22.4


def commands_from(events):
    selection = {i: [] for i in range(11)}
    plugin, last, commands, sequences = SelectionTracker(), None, [], set()
    for e in events:
        if e['_userid']['m_userId'] != 0:
            continue
        kind = e['_event'].split('.')[-1]
        if kind == 'SSelectionDeltaEvent':
            group, delta = e['m_controlGroupId'], e['m_delta']
            mode, data = next(iter(delta['m_removeMask'].items()))
            assert mode in ('None', 'ZeroIndices')
            previous, error = plugin._deselect(selection[group], mode, data)
            assert not error
            selection[group] = plugin._select(previous, delta['m_addUnitTags'])
        elif kind in ('SCmdUpdateTargetPointEvent', 'SCmdUpdateTargetUnitEvent'):
            assert last is not None
            key = 'TargetPoint' if kind == 'SCmdUpdateTargetPointEvent' else 'TargetUnit'
            last['m_data'] = {key: e['m_target']}
        elif kind in ('SCmdEvent', 'SCommandManagerStateEvent'):
            if kind == 'SCmdEvent':
                last = deepcopy(e)
            else:
                assert e['m_state'] == 1 and last is not None
            assert e['m_sequence'] not in sequences
            sequences.add(e['m_sequence'])
            ability = last['m_abil']
            commands.append(dict(loop=e['_gameloop'], selected=selection[10][:],
                target=deepcopy(last['m_data']), ability=[ability['m_abilLink'], ability['m_abilCmdIndex']] if ability else None))
    return commands


def audit(root, output, summary_path=None):
    decode = runpy.run_path(str(Path(__file__).parent.parent/'20260908-garrison-wave-ownership-v1/decode_failed_seed_replays.py'))['decode']
    games = []
    for threshold in (20, 40):
        run = root/f'20260911-execution-failure-feedback-v1-spb2-A4-l6-firstattack{threshold}-eval4-s42613-r1'
        manifest, result, proof = [json.loads((run/n).read_text()) for n in ('manifest.json', 'result.json', 'checkpoint_evaluation.json')]
        assert manifest['source_commit'] == SHA and not manifest['dirty']
        assert result == json.loads((run/'.done').read_text()) and result['process_exit_code'] == 0
        assert result['training_updates'] == 0 and proof['loaded_global_step'] == 4 and proof['game_seeds'] == [42613]*4
        assert hashlib.sha256((run/'console.log').read_bytes()).hexdigest() == result['console_sha256']
        mapping = {r['env']: r['game_pid'] for r in map(json.loads, (run/'game_worker_mapping.jsonl').read_text().splitlines())}
        responses = list(map(json.loads, (run/'prompt_response.jsonl').read_text().splitlines()))
        for env, pid in sorted(mapping.items()):
            log = next((run/'sc2-logs'/f'game-{pid}').rglob('*.log'))
            replay = log.parent/'replay.SC2Replay'
            assert hashlib.sha256(replay.read_bytes()).hexdigest() == result['replays'][str(replay.relative_to(run))]
            archive = MPQArchive(str(replay))
            header = latest().decode_replay_header(archive.header['user_data_header']['content'])
            assert header['m_version']['m_baseBuild'] == 75689
            proto = build(75689)
            events = list(proto.decode_replay_tracker_events(archive.read_file('replay.tracker.events')))
            raw_commands = list(proto.decode_replay_game_events(archive.read_file('replay.game.events')))
            commands = commands_from(raw_commands)
            units, deaths, complete_alive, peaks = {}, [], {}, []
            for e in events:
                kind, loop = e['_event'].split('.')[-1], e['_gameloop']
                tag = proto.unit_tag(e.get('m_unitTagIndex', 0), e.get('m_unitTagRecycle', 0))
                if kind in ('SUnitBornEvent', 'SUnitInitEvent'):
                    units[tag] = dict(tag=tag, type=e['m_unitTypeName'].decode(), birth_type=e['m_unitTypeName'].decode(),
                        owner=e['m_controlPlayerId'], born_loop=loop, done_loop=loop if kind=='SUnitBornEvent' else None,
                        death_loop=None, born_xy=[e['m_x'], e['m_y']])
                    if kind == 'SUnitBornEvent': complete_alive[tag] = units[tag]
                elif kind == 'SUnitDoneEvent':
                    units[tag]['done_loop'] = loop
                    complete_alive[tag] = units[tag]
                elif kind == 'SUnitTypeChangeEvent':
                    units[tag]['type'] = e['m_unitTypeName'].decode()
                elif kind == 'SUnitDiedEvent':
                    killer = proto.unit_tag(e.get('m_killerUnitTagIndex') or 0, e.get('m_killerUnitTagRecycle') or 0)
                    deaths.append(dict(loop=loop, second=loop/FPS, tag=tag, owner=units[tag]['owner'],
                        type=units[tag]['type'], xy=[e['m_x'], e['m_y']], killer_tag=killer,
                        killer=units.get(killer, {}).get('type'), killer_owner=units.get(killer, {}).get('owner')))
                    units[tag]['death_loop'] = loop
                    complete_alive.pop(tag, None)
                own = Counter(u['type'] for u in complete_alive.values() if u['owner']==1 and u['type']=='Marine')
                if not peaks or own['Marine'] > peaks[-1]['count']:
                    peaks.append(dict(loop=loop, second=loop/FPS, count=own['Marine']))
            # All completed own nonworker combat births are Marine in these eight replays.
            born_names = {u['birth_type'] for u in units.values() if u['owner']==1 and u['done_loop']==u['born_loop']}
            assert not (born_names-{'Marine', 'SCV', 'CommandCenter'}-{n for n in born_names if n.startswith('Beacon')})
            attacks, scouting = [], []
            for no, line in enumerate(log.read_text().splitlines(), 1):
                m = re.search(r'\[INFO\]: \((\d+)\) (.*)', line)
                if not m: continue
                item = dict(loop=int(m[1])*4, second=int(m[1])*4/FPS, line=no, text=m[2])
                if any(s in m[2] for s in ('已创建总攻波次', '集结完毕', '已全灭', 'ATTACK: Launch attack')): attacks.append(item)
                if m[2].startswith('scouting:'): scouting.append(item)
            launch = next((a for a in attacks if '已创建总攻波次 1，' in a['text']), None)
            wave = None
            if launch:
                rally = tuple(map(int, re.search(r'集结点: \((\d+), (\d+)\)', launch['text']).groups()))
                selected = [c for c in commands if c['loop']==launch['loop'] and c['ability']==[45,0]
                    and 'TargetPoint' in c['target'] and tuple(math.floor(c['target']['TargetPoint'][k]/4096) for k in ('x','y'))==rally]
                assert selected and all(set(c['selected'])<=set(selected[0]['selected']) for c in selected)
                tags = set(selected[0]['selected'])
                assert len(tags)==threshold and all(units[t]['owner']==1 and units[t]['birth_type']=='Marine' for t in tags)
                dead = [d for d in deaths if d['tag'] in tags]
                assert len(dead)==len(tags), 'Logged deletion is not enough: require actual tracker deaths for every tag'
                attacks_started = next(a for a in attacks if '总攻波次 1: 集结完毕' in a['text'])
                hits = [d for d in deaths if d['killer_tag'] in tags]
                selected_commands = [c for c in commands if set(c['selected'])&tags and c['loop']>=launch['loop']]
                wave = dict(launch=launch, attack=attacks_started, tags=sorted(tags), rally=rally,
                    deaths=dead, kills=hits, killer_counts=dict(Counter(d['killer'] for d in dead)),
                    kill_counts=dict(Counter(d['type'] for d in hits)), commands=selected_commands,
                    last_death_second=max(d['second'] for d in dead),
                    move_commands_after_launch=[c for c in selected_commands if c['ability']==[42,0]])
            ledger = decode(replay)
            trace = json.loads((log.parent/'trace.json').read_text())
            resources = [dict(iteration=int(k),second=int(k)*4/FPS, **{n:v for n,v in v.items() if n in
                ('minerals','vespene','supply_army','supply_workers','supply_left')}) for k,v in trace.items() if 'minerals' in v]
            plans = []
            for r in responses:
                if r['env_idx']!=env or not r.get('semantic_action_evaluable'): continue
                plans.append(dict(step_index=r['step_index'], prompt=r['prompt'], answer=r['llm_response'],
                    valid=r['semantic_action_valid'], reason=r['failure_reason'], reward=r['reward'],
                    source='historical_replay' if r['step_index']<2 else 'online_model'))
            prefix = [e for e in events if e['_gameloop']<1396*4]
            cp = [{k:v for k,v in e.items() if not(k=='m_syncTime' and e['_event'].endswith(
                ('SSetSyncPlayingTimeEvent','SSetSyncLoadingTimeEvent')))} for e in raw_commands if e['_gameloop']<1396*4]
            snap_loops = [int(s*FPS) for s in (328,349,420,450,480,510,540,570,600)]
            if wave: snap_loops += [wave['launch']['loop'], wave['attack']['loop']]
            games.append(dict(threshold=threshold, env=env, run=str(run), log=str(log.relative_to(run)),
                replay=str(replay.relative_to(run)), hashes={str(f.relative_to(run)):hashlib.sha256(f.read_bytes()).hexdigest()
                    for f in (log,replay,log.parent/'trace.json')}, seconds=ledger['seconds'],
                prefix_tracker_sha=digest(prefix), prefix_commands_sha=digest(cp), peak_marine=peaks[-1],
                units=list(units.values()), deaths=deaths, upgrades=ledger['upgrades'], first_wave=wave,
                attacks=attacks, scouting=scouting, snapshots=snapshots(events,snap_loops),
                resources=resources, plans=plans,
                trace_bo=[dict(loop=int(k)*4,key=n,value=v) for k,row in trace.items() for n,v in row.items() if
                          n.startswith(('llm_decision_bo','llm_decision_final_action'))]))
    assert len(games)==8
    assert len({g['prefix_tracker_sha'] for g in games})==1
    assert len({g['prefix_commands_sha'] for g in games})==1
    report=dict(source_sha=SHA,snapshot=str(root),cpu_only=True,all_eight_prefixes_equal=True,games=games,
        limitations='One game seed. Replayed first two answers are not online choices. Tracker killer is last-hit, not total damage; no per-frame HP/cloak/detection evidence.')
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if summary_path is not None:
        short = dict(source_sha=SHA, snapshot=str(root), full_audit=str(output),
            full_audit_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            all_eight_prefixes_equal=True, cpu_only=True, games=[], limitations=report['limitations'])
        for g in games:
            w = g['first_wave']
            own = [u for u in g['units'] if u['owner']==1]
            plans=[]
            for p in g['plans'][2:]:
                feedback=p['prompt'].split('Execution feedback for the previous segment (not owned units/buildings):\n')
                f=json.loads(feedback[1].splitlines()[0]) if len(feedback)>1 else None
                plans.append(dict(game_time=re.search(r'It is now ([^ ]+)',p['prompt'])[1], answer=p['answer'],
                    valid=p['valid'], feedback_failed=f['failed'] if f else None))
            short['games'].append(dict(threshold=g['threshold'],env=g['env'],seconds=g['seconds'],
                hashes=g['hashes'],peak_marine=g['peak_marine'],
                total_marine_born=sum(u['birth_type']=='Marine' for u in own),
                marine_deaths=sum(d['owner']==1 and d['type']=='Marine' for d in g['deaths']),
                own_noncosmetic_upgrades=[u for u in g['upgrades'] if u['player']==1 and not u['name'].startswith('Spray')],
                new_depots=[dict(start=u['born_loop']/FPS,complete=u['done_loop']/FPS if u['done_loop'] is not None else None)
                    for u in own if u['birth_type']=='SupplyDepot' and u['born_loop']>300*FPS],
                trace_samples=[min(g['resources'],key=lambda r:abs(r['second']-t)) for t in (328,360,450,480,510)],
                first_wave=None if not w else dict(unit_count=len(w['tags']),units='Marine only',
                    gathering_second=w['launch']['second'],attack_second=w['attack']['second'],
                    last_death_second=w['last_death_second'],all_tags_dead=True,
                    killers_last_hit=w['killer_counts'],kills_last_hit=w['kill_counts'],
                    killed_commandcenters_all_incomplete=all(u['done_loop'] is None for u in g['units']
                        if u['tag'] in {d['tag'] for d in w['kills'] if d['type']=='CommandCenter'}),
                    move_commands_after_launch=len(w['move_commands_after_launch']),
                    enemy_at_attack=g['snapshots'][str(w['attack']['loop'])]['players']['2'],
                    deaths_xy_bounds=[[min(d['xy'][i] for d in w['deaths']),max(d['xy'][i] for d in w['deaths'])] for i in (0,1)]),
                subsequent_plans=plans))
        summary_path.write_text(json.dumps(short,ensure_ascii=False,indent=2)+'\n')
    for g in games:
        w=g['first_wave']
        print(g['threshold'],g['env'],'peak',g['peak_marine'], 'wave',None if w is None else {
            'tags':len(w['tags']),'killers':w['killer_counts'],'kills':w['kill_counts'],
            'last_death_second':w['last_death_second'],'move_commands':len(w['move_commands_after_launch'])})


if __name__=='__main__':
    audit(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]) if len(sys.argv)>3 else None)
