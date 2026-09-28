"""Read-only r3 construction audit; reuses the existing command decoder."""
import hashlib
import json
from pathlib import Path
import re
import sys

from mpyq import MPQArchive
from s2protocol.versions import build

from audit_wave_orders import audit


def inspect(run):
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['source_commit'] == 'e1e25ce1d14362b209e34ec73871ebaeb63538a6'
    assert manifest['formal'] and not manifest['dirty'] and manifest['mode'] == 'evaluation'
    decoded = audit(run)  # Also verifies replay and console against result hashes.
    archive = MPQArchive(str(next((run / 'sc2-logs').rglob('*.SC2Replay'))))
    protocol = build(75689)
    events = list(protocol.decode_replay_tracker_events(archive.read_file('replay.tracker.events')))
    units, histories = {}, {}
    for event in events:
        if 'm_unitTagIndex' not in event:
            continue
        tag = protocol.unit_tag(event['m_unitTagIndex'], event['m_unitTagRecycle'])
        histories.setdefault(tag, []).append(event)
        if event['_event'].endswith(('SUnitBornEvent', 'SUnitInitEvent')):
            units[tag] = event
    stars = [tag for tag, event in units.items()
             if event['m_controlPlayerId'] == 1 and event['m_unitTypeName'] == b'Starport']
    assert len(stars) == 1, stars
    star = stars[0]
    start = units[star]
    commands = decoded['commands']
    # Match the recorded Starport command by actual target and raw ability.
    builds = [c for c in commands if c['ability'] == [129, 11]
              and 'TargetPoint' in c['target']
              and [c['target']['TargetPoint'][k] / 4096 for k in ('x', 'y')]
              == [start['m_x'] + .5, start['m_y'] + .5]
              and c['loop'] <= start['_gameloop']]
    assert len(builds) == 1 and len(builds[0]['selected']) == 1, builds
    command = builds[0]
    worker = command['selected'][0]
    assert units[worker]['m_unitTypeName'] == b'SCV'
    death = next(e for e in histories[worker] if e['_event'].endswith('SUnitDiedEvent'))
    worker_killer = protocol.unit_tag(death['m_killerUnitTagIndex'], death['m_killerUnitTagRecycle'])
    star_death = next(e for e in histories[star] if e['_event'].endswith('SUnitDiedEvent'))
    star_killer = protocol.unit_tag(star_death['m_killerUnitTagIndex'], star_death['m_killerUnitTagRecycle'])
    worker_commands = [c for c in commands if worker in c['selected']]
    star_target_commands = [c for c in commands if c['target'].get('TargetUnit', {}).get('m_tag') == star]
    lines = [re.sub(r'\x1b\[[0-9;]*m', '', s) for s in (run / 'console.log').read_bytes().decode().split('\n')]
    evidence_lines = [{'line': i, 'text': line} for i, line in enumerate(lines, 1)
                      if any(s in line for s in ('4345036803', '[803]SCV',
                          'Start Processing BO #4: starport', 'Start Processing BO #4: vikingfighter',
                          'Facility under construction.', 'BO #4 skipped after 11 attempts'))]
    live_scv = sum(e['m_controlPlayerId'] == 1 and e['m_unitTypeName'] == b'SCV'
                   and e['_gameloop'] <= death['_gameloop']
                   and not any(h['_event'].endswith('SUnitDiedEvent') and h['_gameloop'] <= death['_gameloop']
                               for h in histories[tag]) for tag, e in units.items())
    return dict(run='20260908-garrison-wave-ownership-v1-spb2-l6-g2-s2807211443-planner-r3',
        runtime_sha='e1e25ce1d14362b209e34ec73871ebaeb63538a6',
        replay_sha256=decoded['replay_sha256'], console_sha256=decoded['console_sha256'],
        trace_sha256=hashlib.sha256(next((run / 'sc2-logs').rglob('trace.json')).read_bytes()).hexdigest(),
        starport_tag=star, worker_tag=worker, worker_live_log_tag=4345036803,
        tag_note='Live log tag minus 2**32 matches replay SCV tag; command selection and target also match.',
        command=command, worker_commands=worker_commands,
        worker_commands_after_build=[c for c in worker_commands if c['sequence'] > command['sequence']],
        commands_targeting_starport=star_target_commands,
        starport_history=histories[star], worker_history=histories[worker],
        worker_killer_type=units[worker_killer]['m_unitTypeName'],
        starport_killer_type=units[star_killer]['m_unitTypeName'],
        seconds_from_start_to_worker_death=(death['_gameloop'] - start['_gameloop']) / 22.4,
        seconds_from_worker_death_to_starport_death=(star_death['_gameloop'] - death['_gameloop']) / 22.4,
        other_scvs_alive_at_worker_death=live_scv, console_evidence=evidence_lines,
        limitations='No continuous per-building build_progress/health or SCV position/order observations are saved. Tracker death identifies the killing unit, not every attacker. No counterfactual evidence that replacement SCVs would survive or the game would be won.')


if __name__ == '__main__':
    print(json.dumps(inspect(Path(sys.argv[1])), ensure_ascii=False, indent=2,
                     default=lambda value: value.decode() if isinstance(value, bytes) else str(value)))
