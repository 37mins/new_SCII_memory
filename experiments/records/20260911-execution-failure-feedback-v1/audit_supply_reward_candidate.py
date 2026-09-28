"""Offline design review only; never imported by the SC2 environment or trainer."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


SUPPLY_SHA = '4d39d06658364689518d5d6f44a2af4bb5f73b49'


def event_verdict(witness, paid_episodes):
    """Proposed evidence contract, NOT an extractor of these facts from raw logs.

    None means unknown, not failed execution. This function emits no reward and
    does not mutate any historical record or PPO sample.
    """
    if witness['episode_id'] in paid_episodes:
        return 'no_second_event'
    for field in ('failure_origin', 'depot_origin', 'resumed_production_origin'):
        if witness[field] in ('rule', 'human', 'fixed_bo'):
            return 'exclude_non_model_action'
    facts = ('supply_failure', 'command_links_verified',
             'depot_started_after_failure', 'same_depot_completed',
             'capacity_increased', 'new_marine_from_post_completion_order')
    if any(witness[field] is False for field in facts):
        return 'not_resolved'
    if any(witness[field] is None for field in facts) or any(
        witness[field] != 'model' for field in
        ('failure_origin', 'depot_origin', 'resumed_production_origin')
    ):
        return 'insufficient_evidence'
    return 'eligible_event'


def ideal_episode_increment(bonuses, trajectory_lengths):
    """Analytic toy: all valid, no padding, fixed lengths, mean_norm episode term.

    Not recovered B4 advantages; the exact-anchor step term is a separate term.
    """
    mean = sum(bonuses) / len(bonuses)
    return [(b - mean) / n for b, n in zip(bonuses, trajectory_lengths)]


def read_jsonl(path):
    return [dict(json.loads(line), source_line=i)
            for i, line in enumerate(path.read_text().splitlines(), 1)]


def seconds(value):
    minute, second = map(int, value.split(':'))
    return 60 * minute + second


def analyze(snapshot):
    manifest = json.loads((snapshot / 'manifest.json').read_text())
    result = json.loads((snapshot / 'result.json').read_text())
    assert manifest['source_commit'] == SUPPLY_SHA and not manifest['dirty']
    assert result['status'] == 'completed' and result['process_exit_code'] == 0
    assert result['training_updates'] == 0
    assert result['console_sha256'] == 'e502a7b7ff3c6c81a6b6c0a865649644cc869b83bbbcc1b35b034beb5c5715ca'
    mapping = {r['env']: r for r in read_jsonl(snapshot / 'game_worker_mapping.jsonl')}
    interventions = read_jsonl(snapshot / 'interventions.jsonl')
    assert len(interventions) == 24 and set(mapping) == set(range(8))
    assert all(r['source'] == 'historical_prefix_or_order_intervention_not_model_choice'
               for r in interventions)
    games, changed_ids = [], []
    for env, m in sorted(mapping.items()):
        path = snapshot / f"game-{m['game_pid']}-supply-events.jsonl"
        rows = read_jsonl(path)
        starts = {r['tag']: r for r in rows
                  if r['event'] == 'on_building_construction_started'}
        target = next(r for r in interventions if r['env'] == env and r['call_index'] == 2)
        queues = [r for r in rows if r['event'] == 'executor_queue']
        assert [r['policy_action_id'] for r in queues[:3]] == [1, 2, 3]
        assert queues[2]['time'] == '06:13'
        if m['arm'] == 'B':
            assert len(queues) == 3  # This arm never returned to an online fourth answer.
        after = [r for r in rows if seconds(r['time']) > seconds('06:13')]
        depots = [r for r in after if r['event'] == 'on_building_construction_complete'
                  and r['unit'] in ('SUPPLYDEPOT', 'SUPPLYDEPOTLOWERED')]
        marines = [r for r in after if r['event'] == 'on_unit_created' and r['unit'] == 'MARINE']
        for r in rows:
            if r['event'] != 'on_building_construction_complete' or r['tag'] not in starts:
                continue
            before = starts[r['tag']]
            if before['policy_action_id'] != r['policy_action_id']:
                changed_ids.append(dict(env=env, file=path.name, tag=r['tag'], unit=r['unit'],
                    start_line=before['source_line'], complete_line=r['source_line'],
                    start_latest_id=before['policy_action_id'], complete_latest_id=r['policy_action_id']))
        games.append(dict(env=env, arm=m['arm'], game_pid=m['game_pid'],
            intervention_line=target['source_line'], intervention_source=target['source'],
            depot_completions_after_0613=len(depots), marine_creations_after_0613=len(marines),
            depot_events=depots, marine_events=marines,
            # Outcome counters establish a difference, not command ownership.
            reward_eligibility='excluded_human_intervention' if m['arm'] == 'B'
                               else 'no_marine_creation_after_intervention',
            missing_command_origin_links=True))
    assert [g['depot_completions_after_0613'] for g in games if g['arm'] == 'B'] == [8]*4
    assert [g['marine_creations_after_0613'] for g in games if g['arm'] == 'B'] == [2]*4
    assert sorted(g['depot_completions_after_0613'] for g in games if g['arm'] == 'A') == [0, 0, 0, 1]
    assert all(g['marine_creations_after_0613'] == 0 for g in games if g['arm'] == 'A')
    b4_path = Path(__file__).with_name('b4-training-signal.json')
    b4 = json.loads(b4_path.read_text())
    feedback = [r for r in b4['feedback_events'] if 'Supply blocked' in r['failed']['reason']]
    assert len(feedback) == 4
    return dict(status='design_review_only_no_runtime_reward',
        source_sha=SUPPLY_SHA, snapshot=str(snapshot),
        artifact_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(snapshot.iterdir()) if p.suffix in ('.json', '.jsonl')},
        b4_audit_sha256=hashlib.sha256(b4_path.read_bytes()).hexdigest(),
        games=games, callback_latest_id_changed_between_start_and_completion=changed_ids,
        historical_evidence_counts=dict(Counter(g['reward_eligibility'] for g in games)),
        b4_supply_feedback=[dict(source_line=r['source_line'], answer=r['answer'],
            valid=r['valid'], depot_first=r['depot_is_first_bullet'],
            verdict='insufficient_evidence_for_completed_model_owned_resolution') for r in feedback],
        limitations=['Completion callback policy_action_id is latest ID, not a command origin.',
            'Human interventions cannot be rewarded as model corrections.',
            'B4 prompt logs prove choices, not completion of a uniquely attributable event.',
            'One event per episode bounds repeat farming but does not remove incentive to create one blockage.',
            'No candidate coefficient, production integration or GPU test has been approved.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.snapshot)
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(historical=report['historical_evidence_counts'],
        callback_id_changes=len(report['callback_latest_id_changed_between_start_and_completion']),
        b4_unresolved_evidence=len(report['b4_supply_feedback'])), ensure_ascii=False))
