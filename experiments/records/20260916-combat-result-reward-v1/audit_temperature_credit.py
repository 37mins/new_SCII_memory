"""Read-only scalar GiGPO reconstruction; no model, game or training changes.

Reuse the existing policy-credit audit. Float64 equations are not recovered PPO
tensors; copying and post-balance representatives are bounded, not guessed.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments/records/20260911-execution-failure-feedback-v1'))
from audit_rl4_rewards import credited_episode, jsonl, metric_rows
from audit_b4_training_signal import episode_component_bounds

SHA = 'cf3ff4c26c9cb3f4f744b6cf4eeae909ccf92565'


def score_rows(episodes, gamma):
    rows = []
    for e in episodes:
        for i, a in enumerate(e['answers']):
            value = sum(t['credited_reward'] * gamma**j for j, t in enumerate(e['answers'][i:]))
            rows.append(dict(a, env=e['env'], call=i+1, episode_reward=e['reward'],
                             step_return=value, step_score=value-.1*(not a['valid'])))
    groups = defaultdict(list)
    for r in rows:
        groups[r['prompt_sha256']].append(r['step_score'])
    scores = {e['env']: e['reward']-.1*(not e['answers'][0]['valid']) for e in episodes}
    mean = sum(scores.values())/len(scores)
    lengths = {e['env']: len(e['answers']) for e in episodes}
    for r in rows:
        group = groups[r['prompt_sha256']]
        r['episode_component_before_copy'] = (scores[r['env']]-mean)/lengths[r['env']]
        r['step_component_before_copy'] = r['step_score']-sum(group)/len(group)
        r['joint_before_copy'] = r['episode_component_before_copy']+r['step_component_before_copy']
        r['anchor_size_before_copy'] = len(group)
    return rows, groups


def analyze(snapshot):
    manifest = json.loads((snapshot/'manifest.json').read_text())
    assert manifest['source_commit'] == SHA and not manifest['dirty']
    assert manifest['start_step'] == 4 and manifest['world_size'] == 4
    cfg = json.loads((snapshot/'effective_config.json').read_text())['overrides']
    for key, value in {'algorithm.gamma':'0.95', 'algorithm.gigpo.mode':'mean_norm',
        'algorithm.gigpo.step_advantage_w':'1.0', 'algorithm.use_kl_in_reward':'False',
        'actor_rollout_ref.actor.invalid_action_penalty_coef':'0.1',
        'env.history_length':'0', '+env.sc2_memory_mode':'off',
        'actor_rollout_ref.rollout.temperature':'0.8'}.items():
        assert cfg[key] == value, key
    hashes = {}
    for name in ('gigpo/core_gigpo.py', 'verl/trainer/ppo/ray_trainer.py',
                 'verl/trainer/ppo/sc2_task_signal.py', 'agent_system/sc2_episode.py',
                 'agent_system/multi_turn_rollout/utils.py',
                 'agent_system/multi_turn_rollout/rollout_loop.py',
                 'agent_system/environments/strategic_sc2_env_manager_v2.py',
                 'agent_system/reward_manager/episode.py', 'verl/trainer/main_ppo.py'):
        frozen = subprocess.check_output(['git','show',f'{SHA}:{name}'], cwd=ROOT)
        assert frozen == (ROOT/name).read_bytes(), name
        hashes[name] = hashlib.sha256(frozen).hexdigest()
    raw = jsonl(snapshot/'prompt_response.jsonl')
    assert not any(r['memory_context'] for r in raw)
    outcomes = jsonl(snapshot/'episode_outcomes.jsonl')
    seeds = {r['reset_index']:r['seeds'] for r in jsonl(snapshot/'training_seeds.jsonl')}
    signals = {r['pending_global_step']:r for r in jsonl(snapshot/'signal_sampling.jsonl')}
    metrics = metric_rows(snapshot/'console.log')
    console = (snapshot/'console.log').read_text()
    assert "'enable_similarity': False" in console
    # This flag is fixed in the frozen constructor, not printed in Hydra config.
    assert 'reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)' in (ROOT/'verl/trainer/main_ppo.py').read_text()
    batches = []
    for step, metric in sorted(metrics.items()):
        reset = step-manifest['start_step']-1
        ends = {r['env_index']:r for r in outcomes if r['reset_index']==reset}
        assert len(ends)==8
        low = max([r['step_index'] for r in outcomes if r['reset_index']==reset-1],
                  default=min(r['step_index'] for r in raw)-1)+1
        episodes = []
        for env in range(8):
            policy, total = credited_episode(raw, ends[env], low)
            answers = [dict(source_line=r['source_line'],
                            prompt_sha256=hashlib.sha256(r['prompt'].encode()).hexdigest(),
                            answer=r['llm_response'], valid=r['is_action_valid'],
                            credited_reward=r['rewards']) for r in policy]
            episodes.append(dict(env=env, reward=total, answers=answers,
                                 decisions=len(answers), result=ends[env]['game_result']))
        assert abs(sum(e['reward'] for e in episodes)/8-metric['episode/reward/mean']) < .001
        rows, groups = score_rows(episodes, .95)
        no_discount, _ = score_rows(episodes, 1.)
        padding = (-len(rows))%4
        maximum = max(abs(r['joint_before_copy']) for r in rows)
        assert abs(maximum-signals[step]['max_abs_task_advantage']) < 1e-4
        assert len(rows)==signals[step]['real_responses']
        for r, counterfactual in zip(rows, no_discount):
            group = sorted(groups[r['prompt_sha256']])
            means = [sum(group)/len(group)]
            for k in range(1, min(padding,len(group))+1):
                means.extend(((sum(group)+sum(group[:k]))/(len(group)+k),
                              (sum(group)+sum(group[-k:]))/(len(group)+k)))
            ep = episode_component_bounds(episodes, r['env'], padding)
            r['joint_conservative_bounds_after_copy'] = [ep[0]+r['step_score']-max(means),
                                                        ep[1]+r['step_score']-min(means)]
            r['gamma1_scalar_counterfactual_not_experiment'] = counterfactual['joint_before_copy']
        batches.append(dict(global_step=step, seed=seeds[reset][0],
            wins=sum(e['result']=='Victory' for e in episodes), decisions=len(rows),
            valid=sum(r['valid'] for r in rows), copied_rows=padding,
            logged_pre_copy_max_abs=signals[step]['max_abs_task_advantage'],
            reconstructed_pre_copy_max_abs=maximum,
            logged_post_copy_range=[metric['critic/advantages/min'],metric['critic/advantages/max']],
            episodes=[{k:v for k,v in e.items() if k!='answers'} for e in episodes], rows=rows))
    return dict(runtime_sha=SHA, snapshot=str(snapshot), source_hashes=hashes,
        artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.iterdir() if p.is_file()},
        batches=batches, limits=[
            'Only completed batches in this non-atomic read-only snapshot are analyzed.',
            'Frozen memory-off prompt identity supplies anchor groups; no full-token tensors recovered.',
            'Pre-copy maxima agree within 1e-4; scalar arithmetic is float64, not bitwise FP32 replay.',
            'Post-copy bounds include unknown duplicate IDs and penalized representatives after balancing.',
            'Gamma1 is arithmetic attribution on identical logs, not a new experiment or approved change.',
            'Positive response advantage is not proof that each token probability rose or that gameplay improved.'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    result = analyze(args.snapshot)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    for b in result['batches']:
        print('step', b['global_step'], 'wins', b['wins'], 'pre-copy max', b['reconstructed_pre_copy_max_abs'])
        for r in b['rows']:
            if r['call']==2:
                print(r['env'], r['valid'], r['episode_reward'], r['step_return'], r['joint_before_copy'],
                      r['joint_conservative_bounds_after_copy'], r['gamma1_scalar_counterfactual_not_experiment'])
