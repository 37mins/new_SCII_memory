"""Read-only r2 SCV-order audit; never writes runtime command/building links."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


RUNTIME = '3f06211ca3179e5b9fb3348d20594fd7ed3c0d60'
RUN_ID = '20260911-execution-failure-feedback-v1-spb2-original-l6-g2-commandtrace-ab8-8gpu-s52620-r2'


def order_targets_start(order, start):
    if order['ability_id'] != 319:
        return False
    if order['target_kind'] == 'target_world_space_pos':
        return order['target_position'] == start['position']
    if order['target_kind'] == 'target_unit_tag':
        return order['target_tag'] == start['tag']
    return False


def audit_rows(rows, full_issuer_history=False):
    commands, requests, starts, completions = {}, {}, [], []
    linked = set()
    for row in rows:
        event = row['event']
        if event in ('local_command', 'unscoped_wire_command'):
            commands[row['event_id']] = dict(row, sent=False, accepted=None)
        elif event == 'wire_request':
            requests[row['event_id']] = row
            for group in row['groups']:
                for cid in group['command_ids']:
                    commands[cid]['sent'] = True
        elif event == 'wire_result':
            request = requests[row['request_event_id']]
            codes = row['result_codes']
            aligned = row['error_type'] is None and len(codes) == request['action_count']
            assert aligned == row['results_aligned']
            for group in request['groups']:
                accepted = (codes[group['action_index']] == 1
                            if aligned and group['local_mapping_unique'] else None)
                for cid in group['command_ids']:
                    commands[cid]['accepted'] = accepted
        elif event == 'depot_started':
            expected = {}
            for cid, command in commands.items():
                if (command['ability'] == 'TERRANBUILD_SUPPLYDEPOT' and command['sent']
                        and (full_issuer_history or cid not in linked)):
                    expected.setdefault(command['issuer_tag'], []).append(cid)
            snapshots = row['issuer_observations']
            assert {s['issuer_tag']: s['command_ids'] for s in snapshots} == expected
            assert len(snapshots) == len(expected)
            matches = []
            for snapshot in snapshots:
                if not snapshot['observed']:
                    assert snapshot['orders'] is None and snapshot['position'] is None
                    continue
                orders = [o for o in snapshot['orders'] if order_targets_start(o, row)]
                if orders:
                    candidates = [dict(commands[cid]) for cid in snapshot['command_ids']]
                    matches.append(dict(issuer_tag=snapshot['issuer_tag'], orders=orders,
                                        historical_commands=candidates))
            starts.append(dict(event_id=row['event_id'], game_loop=row['game_loop'],
                               tag=row['tag'], position=row['position'],
                               recorded_command_id=row['command_id'], matching_issuers=matches))
            if row['command_id'] is not None:
                linked.add(row['command_id'])
        elif event == 'depot_completed':
            assert any(s['tag'] == row['tag'] for s in starts)
            completions.append(row)
    counts = Counter(r['event'] for r in rows)
    return dict(
        event_counts=dict(counts), command_count=len(commands),
        accepted_commands=sum(c['accepted'] is True for c in commands.values()),
        command_origins=dict(Counter(str((c['origin'].get('transport_origin'),
                                         c['origin']['selection_source'])) for c in commands.values())),
        observed_issuer_counts=dict(Counter(len(s['matching_issuers']) for s in starts)),
        historical_command_counts=dict(Counter(len(s['matching_issuers'][0]['historical_commands'])
                                               for s in starts if len(s['matching_issuers']) == 1)),
        completion_unique_tags=len({c['tag'] for c in completions}),
        recorded_start_links=len(linked), starts=starts,
    )


def audit_completed_history(rows, starts):
    """Conditional offline evidence chains, retiring only previously completed chains."""
    by_event = {s['event_id']: s for s in starts}
    linked, completed, seen_tags, findings = {}, {}, set(), []
    for row in rows:
        if row['event'] == 'depot_completed':
            if row['progress'] == 1 and not row['duplicate']:
                completed.setdefault(row['tag'], row)
            continue
        if row['event'] != 'depot_started':
            continue
        start = by_event[row['event_id']]
        finding = dict(event_id=start['event_id'], game_loop=start['game_loop'],
                       tag=start['tag'], retired=[], remaining_command_ids=[],
                       inferred_command_id=None, recorded_command_id=start['recorded_command_id'])
        findings.append(finding)
        if start['tag'] in seen_tags:
            finding['status'] = 'duplicate_start'
            continue
        seen_tags.add(start['tag'])
        if len(start['matching_issuers']) != 1:
            finding['status'] = 'no_unique_observed_issuer'
            continue
        issuer = start['matching_issuers'][0]
        finding['issuer_tag'] = issuer['issuer_tag']
        pending = []
        for command in issuer['historical_commands']:
            cid = command['event_id']
            previous = linked.get(cid)
            done = completed.get(previous['tag']) if previous else None
            if done and done['position'] == previous['position']:
                finding['retired'].append(dict(command_id=cid, start_event_id=previous['event_id'],
                                               tag=previous['tag'], completed_event_id=done['event_id'],
                                               completed_game_loop=done['game_loop']))
            else:
                pending.append(command)
        finding['remaining_command_ids'] = [c['event_id'] for c in pending]
        if len(pending) != 1:
            finding['status'] = 'no_unique_pending_command'
        elif pending[0]['accepted'] is not True:
            finding['status'] = 'ack_not_confirmed'
        elif pending[0]['event_id'] in linked:
            finding['status'] = 'command_already_associated_with_another_start'
        else:
            command = pending[0]
            linked[command['event_id']] = start
            finding.update(status='unique_in_recorded_history', inferred_command_id=command['event_id'],
                           command_game_loop=command.get('game_loop'), origin=command['origin'])
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence_directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--completed-history', action='store_true',
                        help='Also explain multiple command IDs using earlier completed evidence chains')
    parser.add_argument('--association-candidate', action='store_true',
                        help='Audit the frozen r3 live association against independent history inference')
    args = parser.parse_args()
    root = args.evidence_directory
    runtime = '95f6e2185acb37d792970010429e8828e0c40164' if args.association_candidate else RUNTIME
    run_id = RUN_ID[:-2] + 'r3' if args.association_candidate else RUN_ID
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['source_commit'] == runtime and manifest['dirty'] is False
    result = json.loads((root / 'result.json').read_text())
    assert result['process_exit_code'] == 0 and result['training_updates'] == 0
    mapping = {r['game_pid']: r for r in map(json.loads, (root / 'game_worker_mapping.jsonl').read_text().splitlines())}
    paths = sorted(root.glob('game-*-command-trace.jsonl'))
    assert len(paths) == len(mapping) == 8
    games = []
    for path in paths:
        rows = list(map(json.loads, path.read_text().splitlines()))
        assert rows[0]['run_id'] == run_id
        report = audit_rows(rows, full_issuer_history=args.association_candidate)
        if args.completed_history or args.association_candidate:
            report['completed_history_audit'] = audit_completed_history(rows, report['starts'])
        if args.association_candidate:
            findings = report['completed_history_audit']
            by_tag = {f['tag']: f for f in findings if f['inferred_command_id'] is not None}
            by_id = {r['event_id']: r for r in rows}
            for f in findings:
                assert f['recorded_command_id'] == f['inferred_command_id'], f
                if f['inferred_command_id'] is not None:
                    assert by_id[f['event_id']]['origin'] == f['origin']
            completed = [r for r in rows if r['event'] == 'depot_completed']
            for r in completed:
                inferred = by_tag.get(r['tag'])
                assert r['command_id'] == (inferred['inferred_command_id'] if inferred else None), r
                if inferred:
                    # Completion retains the original command ID, not a copied origin field.
                    assert by_id[r['command_id']]['origin'] == inferred['origin']
                    assert by_id[inferred['event_id']]['position'] == r['position']
                    assert inferred['event_id'] < r['event_id']
            report['independent_association_equal'] = True
            report['recorded_completion_links'] = len({r['tag'] for r in completed
                                                       if r['command_id'] is not None and not r['duplicate']})
        games.append(dict(file=path.name, **mapping[rows[0]['game_pid']], **report))
    summary = dict(
        run_id=run_id, runtime_sha=runtime, games=8, independent_scenarios=1,
        starts=sum(g['event_counts']['depot_started'] for g in games),
        completed=sum(g['completion_unique_tags'] for g in games),
        unique_observed_issuer=sum(g['observed_issuer_counts'].get(1, 0) for g in games),
        unique_historical_command=sum(g['historical_command_counts'].get(1, 0) for g in games),
        multiple_historical_commands=sum(sum(n for k, n in g['historical_command_counts'].items() if k > 1) for g in games),
        recorded_start_links=sum(g['recorded_start_links'] for g in games),
        commands=sum(g['command_count'] for g in games),
        accepted_commands=sum(g['accepted_commands'] for g in games),
        adopted_runtime_rule=None, limitation='SCV order-to-building evidence is not unique command attribution.',
    )
    if args.completed_history or args.association_candidate:
        findings = [s for g in games for s in g['completed_history_audit']]
        summary['completed_history_status_counts'] = dict(Counter(s['status'] for s in findings))
        summary['starts_with_retired_history'] = sum(bool(s['retired']) for s in findings)
        summary['retired_command_occurrences'] = sum(len(s['retired']) for s in findings)
        summary['limitation'] = ('Conditional inference within the recorded command stream, not engine-supplied '
                                 'command IDs or an adopted runtime rule. No future completion, FIFO or rounding.')
    if args.association_candidate:
        summary['independent_association_equal'] = all(g['independent_association_equal'] for g in games)
        summary['recorded_completion_links'] = sum(g['recorded_completion_links'] for g in games)
        summary['adopted_runtime_rule'] = 'recorder_only_unique_order_and_completed_history'
        summary['limitation'] = ('One recorded scenario; observational association, not engine command IDs. '
                                 'No Marine birth attribution, reward or learning acceptance.')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.iterdir() if p.is_file() and p.suffix in ('.json', '.jsonl')}
    with args.output.open('x') as stream:
        json.dump(dict(summary=summary, input_sha256=hashes, games=games), stream, indent=2)
        stream.write('\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
