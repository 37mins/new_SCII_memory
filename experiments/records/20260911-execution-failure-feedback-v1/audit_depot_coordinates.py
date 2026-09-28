"""CPU-only comparison of position hypotheses; never assigns runtime provenance."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


RULES = {
    'exact': lambda x: x,
    'sdk_floor': math.floor,  # burnysc2 7.1.3 Point2.rounded, NOT a placement rule.
    'python_round': round,
    'half_up_hypothesis': lambda x: math.floor(x + .5),
    'ceil_hypothesis': math.ceil,
}


def audit_rows(rows):
    commands, requests = {}, {}
    used = {name: set() for name in RULES}
    starts = []
    for row in rows:
        event = row['event']
        if event in ('local_command', 'unscoped_wire_command'):
            commands[row['event_id']] = dict(row, sent=False, accepted=None)
        elif event == 'wire_request':
            requests[row['event_id']] = row
            for group in row['groups']:
                for command_id in group['command_ids']:
                    commands[command_id]['sent'] = True
        elif event == 'wire_result':
            request = requests[row['request_event_id']]
            codes = row['result_codes']
            aligned = row['error_type'] is None and len(codes) == request['action_count']
            assert aligned == row['results_aligned']
            for group in request['groups']:
                accepted = (codes[group['action_index']] == 1
                            if aligned and group['local_mapping_unique'] else None)
                for command_id in group['command_ids']:
                    commands[command_id]['accepted'] = accepted
        elif event == 'depot_started':
            eligible = [c for c in commands.values()
                        if c['ability'] == 'TERRANBUILD_SUPPLYDEPOT' and c['sent']
                        and c['accepted'] is not False and c['target_position'] is not None]
            compared = {}
            for name, transform in RULES.items():
                candidates = [c for c in eligible if c['event_id'] not in used[name]
                              and [transform(x) for x in c['target_position']] == row['position']]
                unique = (candidates[0] if len(candidates) == 1
                          and candidates[0]['accepted'] is True else None)
                if unique:
                    used[name].add(unique['event_id'])
                compared[name] = dict(
                    candidate_ids=[c['event_id'] for c in candidates],
                    unique_acked_candidate_id=unique['event_id'] if unique else None)
            # Exact replay must reproduce the frozen recorder, not improve its result silently.
            assert compared['exact']['candidate_ids'] == row['candidate_command_ids']
            assert compared['exact']['unique_acked_candidate_id'] == row['command_id']
            starts.append(dict(
                event_id=row['event_id'], game_loop=row['game_loop'], tag=row['tag'],
                position=row['position'], hypotheses=compared,
                eligible_commands=[dict(event_id=c['event_id'], position=c['target_position'],
                                        issuer_tag=c['issuer_tag'], origin=c['origin']) for c in eligible]))
    return starts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence_directory', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    record = json.loads(Path(__file__).with_name('commandtrace-r1-results.json').read_text())
    manifest_path = args.evidence_directory / 'manifest.json'
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == record['artifact_sha256']['manifest.json']
    assert json.loads(manifest_path.read_text())['source_commit'] == record['runtime_sha']
    reports, inputs, targets = [], {}, []
    paths = sorted(args.evidence_directory.glob('game-*-command-trace.jsonl'))
    assert len(paths) == 8
    for path in paths:
        inputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert rows[0]['run_id'] == record['run_id']
        targets.extend(tuple(r['target_position']) for r in rows
                       if r['event'] == 'local_command' and r['ability'] == 'TERRANBUILD_SUPPLYDEPOT')
        for start in audit_rows(rows):
            reports.append(dict(file=path.name, **start))
    counts = {name: dict(unique_acked_candidates=sum(
        r['hypotheses'][name]['unique_acked_candidate_id'] is not None for r in reports),
        ambiguous=sum(len(r['hypotheses'][name]['candidate_ids']) > 1 for r in reports)) for name in RULES}
    summary = dict(
        runtime_sha=record['runtime_sha'], run_id=record['run_id'], games=len(paths),
        independent_scenarios=1, starts=len(reports), submitted_depots=len(targets),
        distinct_target_positions=len(set(targets)),
        coordinate_fraction_counts=dict(Counter(str(x % 1) for p in targets for x in p)),
        comparisons=counts, adopted_runtime_rule=None, gpu_runs=0,
        limitation='Half-integer inputs cannot distinguish ceil from half-up. Candidate uniqueness is not causal proof.',
    )
    with args.output.open('x') as stream:
        json.dump(dict(summary=summary, input_sha256=inputs, starts=reports), stream, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
