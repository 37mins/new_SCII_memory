"""Audit the authorized online-from-third evaluation; reads completed artifacts only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from audit_depot_issuer import audit_rows, audit_completed_history

RUNTIME = 'e23eea3219b9bbf00345e7d78f9cd9fd85fa4452'
RUN_ID = '20260911-execution-failure-feedback-v1-spb2-original-l6-g2-online-from3-8gpu-s52620-r1'


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def analyze(root):
    manifest = json.loads((root/'manifest.json').read_text())
    assert manifest['source_commit'] == RUNTIME and not manifest['dirty']
    result = json.loads((root/'result.json').read_text())
    assert result['process_exit_code'] == 0 and result['training_updates'] == 0
    proof = json.loads((root/'supply_evaluation.json').read_text())
    assert proof['val_only'] and proof['loaded_global_step'] == proof['optimizer_updates'] == 0
    assert proof['online_from_call'] == 3 and proof['game_seeds'] == [52620]*8
    seed_rows = rows(root/'evaluation_seeds.jsonl')
    assert len(seed_rows) == 1 and seed_rows[0]['seeds'] == [52620]*8
    assert seed_rows[0]['reset_index'] == 0
    assert json.loads((root/'intervention_coverage.json').read_text())['all_reached']
    interventions = rows(root/'interventions.jsonl')
    assert Counter(r['call_index'] for r in interventions) == {0: 8, 1: 8, 2: 8}
    third = {r['env']: r for r in interventions if r['call_index'] == 2}
    assert all(r['raw_generated'] == r['forwarded'] and r['source'] == 'online_unmodified'
               for r in third.values())
    fixture = json.loads(Path(__file__).with_name('supply-order-inputs.json').read_text())
    for r in interventions:
        assert r['prompt_sha256'] == fixture['rows'][r['call_index']]['prompt_sha256']
        if r['call_index'] < 2:
            assert r['forwarded'] == fixture['rows'][r['call_index']]['answer']
    mapping = {r['game_pid']: r['env'] for r in rows(root/'game_worker_mapping.jsonl')}
    prompts = rows(root/'prompt_response.jsonl')
    paths = sorted(root.glob('game-*-command-trace.jsonl'))
    assert len(paths) == len(mapping) == 8
    games = []
    for path in paths:
        events = rows(path)
        assert events[0]['run_id'] == RUN_ID
        pid = events[0]['game_pid']; env = mapping[pid]
        independent = audit_rows(events, full_issuer_history=True)
        findings = audit_completed_history(events, independent['starts'])
        assert all(f['inferred_command_id'] == f['recorded_command_id'] for f in findings)
        inferred = {f['tag']: f['inferred_command_id'] for f in findings}
        for event in events:
            if event['event'] == 'depot_completed':
                assert event['command_id'] == inferred.get(event['tag'])
        answers = [r for r in events if r['event'] == 'received_answer']
        assert [r['transport_origin'] for r in answers[:2]] == ['historical_replay']*2
        assert all(r['transport_origin'] == 'online_model_transport' for r in answers[2:])
        assert answers[2]['answer'] == third[env]['forwarded']
        commands = {r['event_id']: r for r in events if r['event'] == 'local_command'
                    and r['origin']['transport_origin'] == 'online_model_transport'
                    and r['origin']['selection_source'] == 'validated_answer'}
        starts = [r for r in events if r['event'] == 'depot_started' and r['command_id'] in commands]
        completed = [r for r in events if r['event'] == 'depot_completed'
                     and not r['duplicate'] and r['command_id'] in commands]
        per_answer = []
        for answer in answers[2:]:
            aid = answer['policy_action_id']
            own = {cid for cid, c in commands.items() if c['origin']['policy_action_id'] == aid}
            per_answer.append(dict(answer=answer,
                queues=[r for r in events if r['event'] == 'executor_queue' and r['latest_policy_action_id'] == aid],
                commands=[commands[cid] for cid in sorted(own)],
                depot_starts=[r for r in starts if r['command_id'] in own],
                depot_completions=[r for r in completed if r['command_id'] in own]))
        reward = 0; terminal = None
        for r in prompts:
            if r['env_idx'] != env:
                continue
            reward += r['reward']
            if r['done']:
                terminal = r
                break
        assert terminal is not None
        final = rows(root/f'game-{pid}-boundaries.jsonl')[-1]
        births = [r for r in events if r['event'] == 'marine_created' and r['event_id'] > answers[2]['event_id']]
        after_completion = [r for r in births if any(c['event_id'] < r['event_id'] for c in completed)]
        games.append(dict(env=env, game_pid=pid, cumulative_reward=reward,
            terminal_log=terminal, final_boundary=final, third_answer=third[env]['raw_generated'],
            online_calls=len(answers)-2, valid_online=sum(r['valid'] for r in answers[2:]),
            depot_starts=len(starts), depot_completions=len(completed),
            marine_births_after_third=len(births), marine_births_after_online_depot_completion=len(after_completion),
            marine_command_attribution='unknown; birth timing is not unique command ownership',
            online_details=per_answer, marine_birth_events=births,
            independent_association_equal=True, recorded_commands=independent['command_count'],
            accepted_commands=independent['accepted_commands']))
    return dict(runtime_sha=RUNTIME, run_id=RUN_ID, games=sorted(games,key=lambda r:r['env']),
        independent_scenarios=1, optimizer_updates=0,
        input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir())
                      if p.is_file() and (p.suffix in ('.json', '.jsonl') or p.name == '.done')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.snapshot)
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    for g in report['games']:
        print(json.dumps({k:g[k] for k in ('env','cumulative_reward','online_calls','valid_online',
              'depot_starts','depot_completions','marine_births_after_third',
              'marine_births_after_online_depot_completion')},ensure_ascii=False))
