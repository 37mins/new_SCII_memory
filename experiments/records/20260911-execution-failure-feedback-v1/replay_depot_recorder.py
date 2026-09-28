"""CPU replay of recorded commands, ACKs and construction callbacks; no SC2 process."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

from sc2.ids.unit_typeid import UnitTypeId
from sc2.unit import Unit
from s2clientprotocol import raw_pb2, sc2api_pb2

from agent_system.sc2_command_trace import CommandTrace

ROOT = Path(__file__).resolve().parents[3]
FROZEN = '3f06211ca3179e5b9fb3348d20594fd7ed3c0d60'
FIELDS = ('game_loop', 'time_seconds', 'supply_cap', 'supply_used', 'minerals',
          'vespene', 'latest_policy_action_id')
ATTRIBUTION = {'command_id', 'candidate_command_ids', 'match_kind', 'origin'}


def observed_unit(tag, kind, position, progress=1, orders=(), game_loop=0):
    proto = raw_pb2.Unit(tag=tag, unit_type=kind.value, build_progress=progress,
                        alliance=raw_pb2.Self)
    proto.pos.x, proto.pos.y = position
    for observed in orders:
        order = proto.orders.add(ability_id=observed['ability_id'], progress=observed['progress'])
        if observed['target_kind'] == 'target_world_space_pos':
            order.target_world_space_pos.x, order.target_world_space_pos.y = observed['target_position']
        elif observed['target_kind'] == 'target_unit_tag':
            order.target_unit_tag = observed['target_tag']
    return Unit(proto, SimpleNamespace(state=SimpleNamespace(game_loop=game_loop)))


def replay(rows, cls, directory):
    state, requests, output = {}, {}, []
    trace = cls(directory, lambda: dict(state))
    try:
        for row in rows:
            state.clear()
            state.update({k: row[k] for k in FIELDS if k in row})
            trace.sequence = row['event_id'] - 1
            trace.pending.clear()
            event = row['event']
            if event in ('local_command', 'unscoped_wire_command'):
                # These recorded commands are inputs, not new SDK requests or new model choices.
                trace.commands.append(dict(copy.deepcopy(row), wire_attempted=False, accepted=None,
                                           building_tag=None, completed=False))
            elif event == 'wire_request':
                requests[row['event_id']] = row
                for group in row['groups']:
                    for command in trace.commands:
                        if command['event_id'] in group['command_ids']:
                            command['wire_attempted'] = True
            elif event == 'wire_result':
                trace.wire_result(requests[row['request_event_id']],
                    sc2api_pb2.Response(action=sc2api_pb2.ResponseAction(result=row['result_codes'])),
                    error=row['error_type'])
            elif event == 'depot_started':
                workers = [observed_unit(s['issuer_tag'], UnitTypeId.SCV, s['position'], orders=s['orders'],
                                         game_loop=row['game_loop'])
                           for s in row['issuer_observations'] if s['observed']]
                before = [w._proto.SerializeToString() for w in workers]
                trace.construction_started(observed_unit(row['tag'], UnitTypeId.SUPPLYDEPOT,
                                                         row['position'], .01, game_loop=row['game_loop']), workers)
                assert before == [w._proto.SerializeToString() for w in workers]
            elif event == 'depot_completed':
                trace.construction_completed(observed_unit(row['tag'], UnitTypeId.SUPPLYDEPOT,
                                                           row['position'], row['progress'], game_loop=row['game_loop']))
            output.append(copy.deepcopy(trace.pending[-1] if trace.pending else row))
    finally:
        trace.stream.close()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence_directory', type=Path)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(Path(__file__).with_name('commandtrace-r2-lifecycle.json').read_text())
    assert hashlib.sha256(args.audit.read_bytes()).hexdigest() == record['audit_sha256']
    audit = json.loads(args.audit.read_text())
    for name, digest in audit['input_sha256'].items():
        assert hashlib.sha256((args.evidence_directory / name).read_bytes()).hexdigest() == digest, name
    scope = {}
    source = subprocess.check_output(['git', 'show', f'{FROZEN}:agent_system/sc2_command_trace.py'], cwd=ROOT)
    exec(compile(source, 'frozen-recorder', 'exec'), scope)
    totals, games = Counter(), []
    with tempfile.TemporaryDirectory(prefix='depot-replay-') as directory:
        for game in audit['games']:
            rows = list(map(json.loads, (args.evidence_directory / game['file']).read_text().splitlines()))
            old = replay(rows, scope['CommandTrace'], directory)
            new = replay(rows, CommandTrace, directory)
            assert old == rows, f"Frozen recorder does not reproduce {game['file']}"
            expected = {s['event_id']: s for s in game['completed_history_audit']}
            changes = []
            for before, after in zip(old, new):
                assert before['event'] == after['event']
                if before['event'] not in ('depot_started', 'depot_completed'):
                    assert before == after
                    continue
                assert {k:v for k,v in before.items() if k not in ATTRIBUTION} == {
                        k:v for k,v in after.items() if k not in ATTRIBUTION}
                if before['event'] == 'depot_started':
                    assert after['command_id'] == expected[before['event_id']]['inferred_command_id']
                totals[before['event'] + '_baseline_linked'] += before['command_id'] is not None
                totals[before['event'] + '_candidate_linked'] += after['command_id'] is not None
                changes.append(dict(event_id=before['event_id'], event=before['event'],
                                    tag=before['tag'], old_command_id=before['command_id'],
                                    new_command_id=after['command_id'], origin=after.get('origin')))
            games.append(dict(env=game['env'], file=game['file'], changes=changes))
    report = dict(summary=dict(totals, frozen_sha=FROZEN, games=8, new_gpu_runs=0,
                              non_attribution_fields_equal=True, worker_protobuf_unchanged=True),
                  candidate_source_sha256=hashlib.sha256((ROOT/'agent_system/sc2_command_trace.py').read_bytes()).hexdigest(),
                  audit_sha256=record['audit_sha256'], input_sha256=audit['input_sha256'], games=games)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
