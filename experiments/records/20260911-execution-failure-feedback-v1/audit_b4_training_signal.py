"""Existing B4 logs only: feedback coverage and scalar GiGPO signal limits."""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import re
import subprocess

from audit_rl4_rewards import ROOT, SHA, audit, jsonl, bullets
from audit_a4_training_signal import supply_order


def depot_choice(answer):
    """Include Depot-only answers; absence of Marine is not failed correction."""
    lines = bullets(answer)
    first = bool(lines and re.fullmatch(r'- supplydepot(?: x \d+)?', lines[0]))
    return dict(**supply_order(answer), depot_is_first_bullet=first)


def episode_component_bounds(episodes, env, padding):
    # Production uses the first encountered (post-balance) row of a trajectory.
    # Its score may include an invalid penalty. The original permutation is absent.
    choices = [sorted({e['reward'] - .1*(not a['valid']) for a in e['answers']})
               for e in episodes]
    centered = [scores[env] - sum(scores)/len(scores)
                for scores in itertools.product(*choices)]
    length = episodes[env]['decisions']
    values = [v/n for v in centered for n in range(length, length+min(padding, length)+1)]
    return [min(values), max(values)]


def analyze(folder):
    base = audit(folder)
    result = json.loads((folder/'result.json').read_text())
    assert result['status'] == 'completed' and result['process_exit_code'] == 0
    assert result['checkpoint_complete'] and result['completed_steps'] == 4
    console = (folder/'console.log').read_text()
    assert "'SC2_EXECUTION_FAILURE_FEEDBACK': '1'" in console.splitlines()[0]
    assert "'SC2_STEP_SHAPING_MODE': 'off'" in console.splitlines()[0]
    assert "'enable_similarity': False" in console
    overrides = json.loads((folder/'effective_config.json').read_text())['overrides']
    for key, value in {'algorithm.gamma':'0.95', 'algorithm.gigpo.mode':'mean_norm',
        'algorithm.gigpo.step_advantage_w':'1.0', 'env.rollout.n':'8',
        'actor_rollout_ref.actor.invalid_action_penalty_coef':'0.1',
        'env.history_length':'0', '+env.sc2_memory_mode':'off'}.items():
        assert overrides[key] == value, key
    prior = json.loads((Path(__file__).with_name('a4-training-signal.json')).read_text())
    assert prior['source_sha'] == SHA
    source_hashes = {}
    names = list(prior['source_file_hashes']) + ['agent_system/environments/strategic_sc2_env_manager_v2.py']
    for name in names:
        frozen = subprocess.check_output(['git', 'show', f'{SHA}:{name}'], cwd=ROOT)
        assert frozen == (ROOT/name).read_bytes(), name
        source_hashes[name] = hashlib.sha256(frozen).hexdigest()
    raw = {r['source_line']:r for r in jsonl(folder/'prompt_response.jsonl')}
    events, batch_reports, shared_feedback, zero_supply = [], [], [], []
    for b in base['batches']:
        groups = defaultdict(list)
        for e in b['episodes']:
            for i, a in enumerate(e['answers']):
                row = raw[a['source_line']]
                assert row['policy_action_applied'] and not row['memory_context']
                obs = row['prompt'].split('Current observations are:\n', 1)[1].split('The remaining BO list', 1)[0]
                supply = int(re.search(r'(\d+) remains unused', obs)[1])
                a.update(env=e['env'], call=i+1, supply_left=supply,
                    game_time=re.search(r'It is now (.*?) in the game', obs)[1],
                    discounted_environment_return=sum(t['credited_reward']*.95**j
                        for j, t in enumerate(e['answers'][i:])), **depot_choice(a['answer']))
                groups[a['prompt_sha256']].append(a)
                if supply == 0:
                    zero_supply.append(dict(global_step=b['global_step'], source_line=a['source_line'],
                        env=e['env'], call=i+1, feedback=bool(a['feedback']),
                        answer=a['answer'], valid=a['valid'], depot_is_first_bullet=a['depot_is_first_bullet']))
        padding = (-b['decisions']) % 4
        b_events = []
        for e in b['episodes']:
            bounds = episode_component_bounds(b['episodes'], e['env'], padding)
            for a in e['answers']:
                if not a['feedback']:
                    continue
                failed = a['feedback']['failed']
                assert failed['target_command_submitted'] is False
                group = groups[a['prompt_sha256']]
                event = dict(global_step=b['global_step'], seed=e['seed'], env=e['env'],
                    source_line=a['source_line'], call=a['call'], game_time=a['game_time'],
                    prompt_sha256=a['prompt_sha256'], original_anchor_group_size=len(group),
                    supply_left=a['supply_left'], answer=a['answer'], valid=a['valid'],
                    depot_is_first_bullet=a['depot_is_first_bullet'], order=a['order'],
                    failed=failed, source_policy_action_id=a['feedback']['source_policy_action_id'],
                    game_result=e['result'], episode_reward=e['reward'],
                    credited_reward=a['credited_reward'],
                    discounted_environment_return=a['discounted_environment_return'],
                    episode_component_conservative_bounds=bounds)
                # Copying a singleton repeats exactly its score; it cannot create
                # a relative step signal. FP32 rounding residue is not a ranking.
                if len(group) == 1:
                    event['mathematical_step_component'] = 0
                events.append(event)
                b_events.append(event)
        for digest, group in groups.items():
            if len(group) > 1 and any(a['feedback'] for a in group):
                assert all(a['feedback'] for a in group)
                shared_feedback.append(dict(global_step=b['global_step'], prompt_sha256=digest,
                    rows=[dict(env=a['env'], source_line=a['source_line'], valid=a['valid'],
                        answer=a['answer'], discounted_environment_return=a['discounted_environment_return'])
                        for a in group]))
        batch_reports.append(dict(**{k:v for k,v in b.items() if k!='episodes'}, padding_rows=padding,
            singleton_feedback_rows=sum(e['original_anchor_group_size']==1 for e in b_events),
            zero_supply_queries=sum(a['supply_left']==0 for e in b['episodes'] for a in e['answers']),
            episodes=[dict(env=e['env'], seed=e['seed'], decisions=e['decisions'], reward=e['reward'],
                result=e['result'], invalid_source_lines=[a['source_line'] for a in e['answers'] if not a['valid']],
                first_discounted_environment_return=e['answers'][0]['discounted_environment_return']) for e in b['episodes']]))
    reasons = Counter((e['failed']['action'], re.sub(r'\([^)]*\)', '(...)',
        e['failed']['reason'].splitlines()[0])) for e in events)
    supply_events = [e for e in events if 'Supply blocked' in e['failed']['reason']]
    return dict(source_sha=SHA, source_snapshot=str(folder), source_file_hashes=source_hashes,
        artifact_hashes={**base['artifacts'], 'result.json':hashlib.sha256((folder/'result.json').read_bytes()).hexdigest()},
        summary=dict(decisions=sum(b['decisions'] for b in batch_reports), games=32,
            valid_answers=sum(b['valid_answers'] for b in batch_reports), feedback_answers=len(events),
            valid_feedback_answers=sum(e['valid'] for e in events), zero_supply_queries=len(zero_supply),
            supply_feedback_answers=len(supply_events), supply_feedback_depot_first=sum(e['valid'] and e['depot_is_first_bullet'] for e in supply_events),
            singleton_feedback_answers=sum(e['original_anchor_group_size']==1 for e in events),
            failure_types=[dict(action=k[0], reason=k[1], count=v) for k,v in sorted(reasons.items())]),
        batches=batch_reports, feedback_events=events, zero_supply_answers=zero_supply,
        shared_feedback_groups=shared_feedback,
        limits=['Singleton step component follows frozen exact-anchor grouping; not the full episode+step advantage or parameter gradient.',
                'Episode-component bounds enumerate possible penalized representatives and conservatively allow any legal trajectory padding count; not recovered PPO tensors.',
                'No copy indices, balancing order or full advantage tensors were saved; scalar double-precision calculations are not bitwise FP32 reconstruction.',
                'Depot-first includes Depot-only replies. It establishes a choice, not construction completion or learned causal improvement.',
                'Cross-step examples use different game seeds and evolving weights, not an order-only A/B.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.snapshot)
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report['summary'], ensure_ascii=False))
