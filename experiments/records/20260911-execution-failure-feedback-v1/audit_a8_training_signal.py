"""CPU-only attribution of A4→A8 logged rewards; never loads model tensors."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import re
import subprocess

from audit_rl4_rewards import ROOT, credited_episode, jsonl, bullets

SHA = '4ad9a605cc6a91419fd2a2ef5d6e8a2af72aac32'


def mean(values):
    return sum(values) / len(values)


def equal_length_decomposition(episodes, metric):
    """Recorded steps 7/8: three decisions, identical prompt at each position.

    Reconstruct scalar advantages, not PPO token tensors/parameter gradients.
    No copied padding in these 24-row batches. The episode representative is
    chosen after balancing in production; enumerate whether each invalid final
    row could have supplied that representative, using logged extrema to check.
    """
    assert all(len(e['answers']) == 3 for e in episodes)
    assert all(len({e['answers'][j]['prompt_sha256'] for e in episodes}) == 1
               for j in range(3))
    assert all(all(a['credited_reward'] == 0 for a in e['answers'][:2]) for e in episodes)
    returns = [e['reward'] for e in episodes]
    formats = [.5 * e['answers'][-1]['valid'] for e in episodes]
    states = [r + 50 - f for r, f in zip(returns, formats)]
    invalid_envs = [i for i, e in enumerate(episodes) if not e['answers'][-1]['valid']]
    candidates = []
    for bits in itertools.product([0, 1], repeat=len(invalid_envs)):
        chosen = dict(zip(invalid_envs, bits))
        episode_scores = [r - .1 * chosen.get(i, 0) for i, r in enumerate(returns)]
        values = []
        for j in range(3):
            step_scores = [r * .95 ** (2-j) - .1 * (not e['answers'][j]['valid'])
                           for r, e in zip(returns, episodes)]
            values.extend((r - mean(episode_scores))/3 + s - mean(step_scores)
                          for r, s in zip(episode_scores, step_scores))
        if (abs(min(values) - metric['critic/advantages/min']) < .001
                and abs(max(values) - metric['critic/advantages/max']) < .001):
            candidates.append((bits, values))
    assert len(candidates) == 1 and not any(candidates[0][0]), candidates
    values = candidates[0][1]
    components = {'terminal_format': [], 'terminal_state': [], 'invalid_penalty': []}
    for j in range(3):
        invalid = [not e['answers'][j]['valid'] for e in episodes]
        for i in range(8):
            scale = 1/3 + .95 ** (2-j)
            components['terminal_format'].append((formats[i] - mean(formats)) * scale)
            components['terminal_state'].append((states[i] - mean(states)) * scale)
            components['invalid_penalty'].append(-.1 * (invalid[i] - mean(invalid)))
    assert all(abs(sum(c[i] for c in components.values()) - v) < 1e-10
               for i, v in enumerate(values))
    magnitudes = {k: sum(abs(v) for v in c) for k, c in components.items()}
    return dict(reconstructed_range=[min(values), max(values)],
        component_absolute_sums=magnitudes,
        legality_component_fraction=(magnitudes['terminal_format'] + magnitudes['invalid_penalty'])
                                    / sum(magnitudes.values()),
        metric_definition='Sum |scalar component| per response, NOT token weighting, gradient share or learning gain',
        representative_inference='Only unpenalized episode representatives match both logged extrema to .001; balancing permutation not saved',
        per_episode=[dict(env=i, terminal_format=formats[i], terminal_state_residual=states[i],
                         advantage_by_call=[values[j*8+i] for j in range(3)]) for i in range(8)])


def audit(folder):
    manifest = json.loads((folder/'manifest.json').read_text())
    assert manifest['source_commit'] == SHA and not manifest['dirty']
    assert manifest['start_step'] == 4 and manifest['world_size'] == 4
    source_hashes = {}
    files = list(manifest['source_files_sha256']) + [
        'gigpo/core_gigpo.py', 'verl/trainer/ppo/ray_trainer.py',
        'agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/agents/commander.py',
        'agent_system/multi_turn_rollout/rollout_loop.py',
        'agent_system/multi_turn_rollout/utils.py', 'agent_system/reward_manager/episode.py']
    for name in files:
        frozen = subprocess.check_output(['git', 'show', f'{SHA}:{name}'], cwd=ROOT)
        assert frozen == (ROOT/name).read_bytes(), name
        source_hashes[name] = hashlib.sha256(frozen).hexdigest()
        if name in manifest['source_files_sha256']:
            assert source_hashes[name] == manifest['source_files_sha256'][name]
    config = json.loads((folder/'effective_config.json').read_text())['overrides']
    for key, value in {'algorithm.gamma':'0.95', 'algorithm.gigpo.mode':'mean_norm',
        'algorithm.gigpo.step_advantage_w':'1.0', 'algorithm.use_kl_in_reward':'False',
        'actor_rollout_ref.actor.invalid_action_penalty_coef':'0.1',
        'env.history_length':'0', '+env.sc2_memory_mode':'off'}.items():
        assert config[key] == value, key
    raw, outcomes = jsonl(folder/'prompt_response.jsonl'), jsonl(folder/'episode_outcomes.jsonl')
    metrics = {int(m['step']):m for m in json.loads((folder/'metrics.json').read_text())}
    seeds = {r['reset_index']:r['seeds'] for r in jsonl(folder/'training_seeds.jsonl')}
    batches = []
    for reset, step in enumerate(range(5, 9)):
        ends = {r['env_index']:r for r in outcomes if r['reset_index'] == reset}
        low = max([r['step_index'] for r in outcomes if r['reset_index'] == reset-1], default=-1)+1
        assert len(ends) == 8
        episodes = []
        for env in range(8):
            policy, total = credited_episode(raw, ends[env], low)
            answers = []
            for r in policy:
                assert not r['memory_context'] and r['is_action_valid'] == r['semantic_action_valid']
                answers.append(dict(source_line=r['source_line'],
                    prompt_sha256=hashlib.sha256(r['prompt'].encode()).hexdigest(),
                    game_time=re.search(r'It is now (.*?) in the game', r['prompt']).group(1),
                    observation=r['prompt'].split('Current observations are:\n',1)[1].split('The remaining BO list',1)[0],
                    answer=r['llm_response'], valid=r['is_action_valid'],
                    failure_reason=r['failure_reason'], credited_reward=r['rewards']))
            episodes.append(dict(env=env, seed=seeds[reset][env], result=ends[env]['game_result'],
                                 reward=total, answers=answers))
        assert abs(mean([e['reward'] for e in episodes])-metrics[step]['episode/reward/mean']) < .001
        answers = [a for e in episodes for a in e['answers']]
        batch = dict(global_step=step, seed=seeds[reset][0],
            wins=sum(e['result']=='Victory' for e in episodes), decisions=len(answers),
            valid=sum(a['valid'] for a in answers), copied_rows=(-len(answers)) % 4,
            prompt_group_sizes=sorted(Counter(a['prompt_sha256'] for a in answers).values()),
            trajectory_bullet_variants=len({tuple(bullets(a['answer']) for a in e['answers']) for e in episodes}),
            first_two_unique_answers=[len({e['answers'][j]['answer'] for e in episodes}) for j in range(2)],
            logged_advantage=[metrics[step]['critic/advantages/min'], metrics[step]['critic/advantages/max']],
            episodes=episodes)
        if step in (7,8):
            batch['decomposition'] = equal_length_decomposition(episodes, metrics[step])
        batches.append(batch)
    return dict(source_sha=SHA, source_file_hashes=source_hashes, snapshot=str(folder),
        artifact_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                         for f in folder.iterdir() if f.is_file()}, batches=batches,
        limits=['No model/checkpoint/GPU loaded; no parameter-gradient attribution',
                'Step5 exact padded advantage not reconstructed: copied IDs and balancing order absent',
                'Step7/8 scalar reconstruction matches rounded extrema; original PPO tensor files absent',
                'Terminal state residual uses game_end formula; does not establish causal execution quality'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.snapshot)
    with args.output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    for b in result['batches']:
        print(json.dumps({k:v for k,v in b.items() if k not in ('episodes','decomposition')}))
        if 'decomposition' in b:
            print('decomposition', b['decomposition']['reconstructed_range'],
                  b['decomposition']['component_absolute_sums'])
