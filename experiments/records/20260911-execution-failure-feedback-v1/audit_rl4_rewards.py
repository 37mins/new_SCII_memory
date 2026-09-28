"""Audit existing r2 log snapshots; no Torch/model/GPU, no reward mutation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from agent_system.sc2_episode import policy_transitions
from run_record import metric_rows

SHA = '361cb7ae7fd86fe8811d6fcb6946d0241bf93e11'
FEEDBACK_MARKER = 'Execution feedback for the previous segment (not owned units/buildings):\n'


def jsonl(path):
    return [dict(json.loads(line), source_line=i) for i, line in
            enumerate(path.read_text().splitlines(), 1)]


def bullets(answer):
    return tuple(line.strip().lower() for line in answer.splitlines() if line.strip().startswith('-'))


def credited_episode(raw, end, low):
    rows = [r for r in raw if r['env_idx'] == end['env_index']
            and low <= r['step_index'] <= end['step_index']]
    policy = policy_transitions([dict(r, active_masks=True, rewards=r['reward'],
                                      environment_done=r['done']) for r in rows])
    assert len(policy) == end['policy_decision_count']
    assert end['terminated'] and not end['truncated']
    # Training log contains rewards once per active transition, including terminal
    # attribution; unlike val-only logs, do not discard an unconsumed terminal poll.
    total = sum(r['reward'] for r in rows)
    assert abs(total - sum(r['rewards'] for r in policy)) < 1e-5
    return policy, total


def audit(folder):
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['source_commit'] == SHA and not manifest['dirty']
    assert manifest['mode'] == 'training' and manifest['world_size'] == 4
    policy_file = 'agent_system/sc2_episode.py'
    assert hashlib.sha256((ROOT/policy_file).read_bytes()).hexdigest() == manifest['source_files_sha256'][policy_file]
    raw = jsonl(folder / 'prompt_response.jsonl')
    outcomes = jsonl(folder / 'episode_outcomes.jsonl')
    seeds = {r['reset_index']:r['seeds'] for r in jsonl(folder / 'training_seeds.jsonl')}
    metrics = metric_rows(folder / 'console.log')
    batches = []
    for step, m in sorted(metrics.items()):
        reset = step - 1
        ends = {r['env_index']:r for r in outcomes if r['reset_index'] == reset}
        assert len(ends) == 8
        low = max([r['step_index'] for r in outcomes if r['reset_index'] == reset - 1],
                  default=min(r['step_index'] for r in raw)-1) + 1
        episodes = []
        for env in range(8):
            policy, reward = credited_episode(raw, ends[env], low)
            answers = []
            for row in policy:
                feedback = None
                if FEEDBACK_MARKER in row['prompt']:
                    feedback, _ = json.JSONDecoder().raw_decode(row['prompt'].split(FEEDBACK_MARKER, 1)[1].lstrip())
                answers.append(dict(source_line=row['source_line'],
                    prompt_sha256=hashlib.sha256(row['prompt'].encode()).hexdigest(),
                    answer=row['llm_response'], projected_action=row['projected_action'],
                    valid=row['is_action_valid'], credited_reward=row['rewards'], feedback=feedback))
            episodes.append(dict(env=env, seed=seeds[reset][env], result=ends[env]['game_result'],
                reward=reward, decisions=len(policy), answers=answers))
        rewards = [e['reward'] for e in episodes]
        assert abs(sum(rewards)/8 - m['episode/reward/mean']) < 0.001
        # Text signatures are not semantic/execution equivalence judgments.
        variants = Counter(tuple(bullets(a['answer']) for a in e['answers']) for e in episodes)
        batches.append(dict(global_step=step, wins=sum(e['result']=='Victory' for e in episodes),
            mean_reward=sum(rewards)/8, distinct_logged_returns=len(set(rewards)),
            reward_spread=max(rewards)-min(rewards), trajectory_bullet_variants=len(variants),
            feedback_answers=sum(a['feedback'] is not None for e in episodes for a in e['answers']),
            valid_answers=sum(a['valid'] for e in episodes for a in e['answers']),
            decisions=sum(e['decisions'] for e in episodes),
            logged_advantage=[m['critic/advantages/min'],m['critic/advantages/max']],
            logged_pg_loss=m['actor/pg_loss'], logged_grad_norm=m['actor/grad_norm'],
            logged_ppo_k1=m['actor/ppo_kl'], logged_reference_kl=m['actor/kl_loss'], episodes=episodes))
    files = ('manifest.json','effective_config.json','prompt_response.jsonl',
             'episode_outcomes.jsonl','training_seeds.jsonl','console.log')
    return dict(source_sha=SHA, source_snapshot=str(folder),
        artifacts={n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in files},
        completed_global_steps=len(batches), batches=batches,
        pending_games_excluded=sum(r['reset_index']+1 not in metrics for r in outcomes),
        scope='Completed training batches only; logged reward/text audit, not exact PPO tensors or execution quality')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshots', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = {arm:audit(args.snapshots/arm) for arm in ('A','B')}
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    for arm, report in result.items():
        print(arm, json.dumps([{k:v for k,v in b.items() if k!='episodes'} for b in report['batches']]))
