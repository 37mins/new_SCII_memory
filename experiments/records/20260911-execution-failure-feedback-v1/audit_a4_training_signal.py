"""Read A4's existing four training batches; no model, game, or GPU calls."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess

from audit_rl4_rewards import ROOT, SHA, audit, jsonl


def supply_order(answer):
    """Describe exact raw bullets, not validator acceptance or executed units."""
    actions = []
    for line in answer.splitlines():
        match = re.fullmatch(r"- (supplydepot|marine|scv)(?: x (\d+))?", line.strip())
        if match:
            actions.append((match[1], int(match[2] or 1)))
    names = [name for name, _ in actions]
    if 'supplydepot' not in names or not {'marine', 'scv'}.intersection(names):
        return dict(order='not_comparable', demand_before_depot=None)
    depot = names.index('supplydepot')
    demand = sum(count for name, count in actions[:depot] if name in ('marine', 'scv'))
    return dict(order='depot_before_consumers' if demand == 0 else 'consumer_before_depot',
                demand_before_depot=demand)


def analyze(folder):
    report = audit(folder)  # Verifies source, real decisions, natural ends and total rewards.
    result = json.loads((folder/'result.json').read_text())
    assert result['status'] == 'completed' and result['process_exit_code'] == 0
    assert result['checkpoint_complete'] and result['completed_steps'] == 4
    config = json.loads((folder/'effective_config.json').read_text())['overrides']
    for key, value in {'algorithm.gamma':'0.95', 'algorithm.gigpo.mode':'mean_norm',
        'algorithm.gigpo.step_advantage_w':'1.0', 'algorithm.use_kl_in_reward':'False',
        'actor_rollout_ref.actor.invalid_action_penalty_coef':'0.1',
        'env.history_length':'0', '+env.sc2_memory_mode':'off'}.items():
        assert config[key] == value, key
    source_hashes = {}
    sources = ['agent_system/sc2_episode.py', 'gigpo/core_gigpo.py',
        'verl/trainer/ppo/ray_trainer.py', 'verl/workers/actor/dp_actor.py',
        'agent_system/multi_turn_rollout/rollout_loop.py',
        'agent_system/multi_turn_rollout/utils.py', 'agent_system/reward_manager/episode.py']
    sources += ['agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/agents/'
                + name for name in ('commander.py', 'planner.py')]
    for name in sources:
        frozen = subprocess.check_output(['git', 'show', f'{SHA}:{name}'], cwd=ROOT)
        assert frozen == (ROOT/name).read_bytes(), name
        source_hashes[name] = hashlib.sha256(frozen).hexdigest()
    raw = {r['source_line']:r for r in jsonl(folder/'prompt_response.jsonl')}
    supply_counts, order_counts = Counter(), Counter()
    paired_order_groups = []
    for batch in report['batches']:
        groups = defaultdict(list)
        for episode in batch['episodes']:
            answers = episode['answers']
            for i, answer in enumerate(answers):
                row = raw[answer['source_line']]
                assert row['policy_action_applied'] and row['semantic_action_evaluable']
                obs = row['prompt'].split('Current observations are:\n', 1)[1].split('The remaining BO list', 1)[0]
                supply = int(re.search(r'(\d+) remains unused', obs)[1])
                answer.update(call=i+1, game_time=re.search(r'It is now (.*?) in the game', obs)[1],
                    supply_left=supply, observation=obs.strip(), **supply_order(answer['answer']))
                # This is the pre-penalty scalar discounted environment return.
                # Do not present it as the actual saved per-token PPO advantage.
                answer['discounted_environment_return'] = sum(
                    a['credited_reward'] * .95**j for j, a in enumerate(answers[i:]))
                supply_counts[supply] += 1
                if answer['valid']:
                    order_counts[answer['order']] += 1
                    if answer['order'] != 'not_comparable':
                        groups[answer['prompt_sha256']].append(answer)
            episode['first_discounted_environment_return'] = answers[0]['discounted_environment_return']
        for digest, group in groups.items():
            if len({a['order'] for a in group}) > 1:
                paired_order_groups.append(dict(global_step=batch['global_step'],
                    prompt_sha256=digest, source_lines=[a['source_line'] for a in group]))
        batch['copied_padding_rows'] = (-batch['decisions']) % 4
        batch['per_game_decisions'] = [len(e['answers']) for e in batch['episodes']]
        batch['valid_same_input_comparable_groups'] = sum(len(g) > 1 for g in groups.values())
        # Verify the zero-signal conditions rather than infer them from win rate.
        if batch['global_step'] in (3, 4):
            assert batch['decisions'] == 16 and batch['valid_answers'] == 16
            assert len({e['reward'] for e in batch['episodes']}) == 1
            assert all(len(e['answers']) == 2 for e in batch['episodes'])
            for i in (0, 1):
                assert len({e['answers'][i]['prompt_sha256'] for e in batch['episodes']}) == 1
                assert len({e['answers'][i]['credited_reward'] for e in batch['episodes']}) == 1
            assert batch['logged_advantage'] == [0.0, 0.0] and batch['logged_pg_loss'] == 0
    all_answers = [a for b in report['batches'] for e in b['episodes'] for a in e['answers']]
    report.update(source_file_hashes=source_hashes,
        result_sha256=hashlib.sha256((folder/'result.json').read_bytes()).hexdigest(),
        total_decisions=len(all_answers), zero_supply_queries=supply_counts[0],
        supply_left_counts=dict(sorted(supply_counts.items())), minimum_supply_left=min(supply_counts),
        valid_raw_order_counts=dict(order_counts), opposite_order_same_input_groups=paired_order_groups,
        feedback_queries=sum(a['feedback'] is not None for a in all_answers),
        valid_demand_exceeds_observed_supply_lines=[a['source_line'] for a in all_answers
            if a['valid'] and a['order'] == 'consumer_before_depot'
            and a['demand_before_depot'] > a['supply_left']],
        limits=['No executed-unit or supply-block duration reconstruction; positive supply at query does not rule out later blocking.',
                'No original PPO advantage tensors, copy indices or balancing order; no exact reconstruction for steps 1/2.',
                'Discounted environment returns are scalar log calculations, before invalid penalties and centering.',
                'Logged zero advantages/PG loss do not imply zero entropy/ref-KL gradient, optimizer momentum or parameter update.',
                'Four game seeds, each repeated eight times; not 32 independent scenarios or a controlled order A/B.'])
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.snapshot)
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in
        ('batches', 'source_file_hashes', 'artifacts', 'limits')}, ensure_ascii=False))
    for batch in report['batches']:
        print(json.dumps({k:v for k,v in batch.items() if k != 'episodes'}))
