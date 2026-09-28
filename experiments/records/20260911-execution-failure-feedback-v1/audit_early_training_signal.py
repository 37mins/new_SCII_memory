"""Audit the first reinforcement-phase answer in retained A/B RL4 logs.

No model/GPU/game, no runtime changes. Scalar calculations are not recovered PPO
tensors or parameter-gradient evidence. Use --output with a previously unused path.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess

from audit_rl4_rewards import ROOT, SHA, audit, jsonl
from audit_b4_training_signal import episode_component_bounds

PHASE = "- Current Stage: Marine Reinforcement Flood "


def first_phase_answer(episode, raw):
    return next((i for i, answer in enumerate(episode['answers'])
                 if PHASE in raw[answer['source_line']]['prompt']), None)


def scalar_advantages(rows, copies=()):
    """All-valid, mean_norm, unchanged full episode returns; float64 equations.

    step_r was discounted before copy/balance. The output includes copied rows.
    The all-valid condition removes the unknown penalized episode representative.
    """
    assert all(r['valid'] for r in rows)
    padded = rows + [rows[i] for i in copies]
    lengths = Counter(r['env'] for r in padded)
    episode_scores = {r['env']: r['episode_reward'] for r in padded}
    baseline = sum(episode_scores.values()) / len(episode_scores)
    groups = defaultdict(list)
    for r in padded:
        groups[r['prompt_sha256']].append(r['step_r'])
    return [(r['episode_reward'] - baseline) / lengths[r['env']]
            + r['step_r'] - sum(groups[r['prompt_sha256']]) / len(groups[r['prompt_sha256']])
            for r in padded]


def analyze():
    folder = Path(__file__).parent
    prior_path = folder / 'rl4-reward-audit-final.json'
    prior = json.loads(prior_path.read_text())
    source_files = ['gigpo/core_gigpo.py', 'verl/trainer/ppo/ray_trainer.py',
        'agent_system/multi_turn_rollout/utils.py', 'agent_system/reward_manager/episode.py',
        'agent_system/environments/strategic_sc2_env_manager_v2.py',
        'agent_system/multi_turn_rollout/rollout_loop.py']
    source_hashes = {name: hashlib.sha256(subprocess.check_output(
        ['git', 'show', f'{SHA}:{name}'], cwd=ROOT)).hexdigest() for name in source_files}
    batches = []
    for arm in ('A', 'B'):
        snapshot = Path(prior[arm]['source_snapshot'])
        base = audit(snapshot)
        old = prior[arm]
        assert {k: v for k, v in base.items() if k != 'batches'} == {
            k: v for k, v in old.items() if k != 'batches'}, arm
        assert len(base['batches']) == len(old['batches'])
        for new_batch, old_batch in zip(base['batches'], old['batches']):
            # Historical B3 mean differs by 7e-15; all raw rows/hashes are exact.
            assert abs(new_batch['mean_reward'] - old_batch['mean_reward']) < 1e-12
            assert dict(new_batch, mean_reward=old_batch['mean_reward']) == old_batch
        result = json.loads((snapshot / 'result.json').read_text())
        assert result == json.loads((snapshot / '.done').read_text())
        assert result['process_exit_code'] == 0 and result['checkpoint_complete']
        overrides = json.loads((snapshot / 'effective_config.json').read_text())['overrides']
        for key, expected in {'algorithm.gamma': '0.95', 'algorithm.gigpo.mode': 'mean_norm',
            'algorithm.gigpo.step_advantage_w': '1.0', 'algorithm.use_kl_in_reward': 'False',
            'actor_rollout_ref.actor.invalid_action_penalty_coef': '0.1',
            'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu': '1',
            'env.rollout.n': '8', 'env.history_length': '0', '+env.sc2_memory_mode': 'off'}.items():
            assert overrides[key] == expected, key
        console = (snapshot / 'console.log').read_text()
        assert "'enable_similarity': False" in console
        raw = {r['source_line']: r for r in jsonl(snapshot / 'prompt_response.jsonl')}
        assert not any(r['memory_context'] for r in raw.values())
        for batch in base['batches']:
            rows, selected, groups = [], [], defaultdict(list)
            for episode in batch['episodes']:
                first = first_phase_answer(episode, raw)
                for i, answer in enumerate(episode['answers']):
                    prompt = raw[answer['source_line']]['prompt']
                    assert hashlib.sha256(prompt.encode()).hexdigest() == answer['prompt_sha256']
                    row = dict(env=episode['env'], call=i+1, source_line=answer['source_line'],
                        valid=answer['valid'], prompt_sha256=answer['prompt_sha256'],
                        episode_reward=episode['reward'],
                        step_r=sum(a['credited_reward'] * .95**j
                                   for j, a in enumerate(episode['answers'][i:])) - .1*(not answer['valid']))
                    rows.append(row)
                    groups[row['prompt_sha256']].append(row['step_r'])
                    if i == first:
                        selected.append(dict(**row, seed=episode['seed'], result=episode['result'],
                            calls_total=episode['decisions'], answer=answer['answer'],
                            game_time=re.search(r'It is now (.*?) in the game', prompt)[1],
                            marine_count=(sum(int(n or 1) for n in re.findall(
                                r'(?m)^- marine(?: x (\d+))?\s*$', answer['answer'])) if answer['valid'] else None)))
            padding = (-len(rows)) % 4
            for row in selected:
                group = groups[row['prompt_sha256']]
                row['real_anchor_size'] = len(group)
                row['episode_component_bounds'] = episode_component_bounds(batch['episodes'], row['env'], padding)
                if padding == 0 or len(group) == 1:
                    step = row['step_r'] - sum(group)/len(group)
                    row['step_component'] = step
                    row['joint_scalar_bounds'] = [step+x for x in row['episode_component_bounds']]
            record = dict(arm=arm, global_step=batch['global_step'], wins=batch['wins'],
                decisions=len(rows), copied_rows=padding, logged_advantage=batch['logged_advantage'],
                first_phase_answers=selected,
                marine_counts=dict(Counter(str(r['marine_count']) for r in selected)),
                input_groups=dict(Counter(r['prompt_sha256'] for r in selected)))
            if arm == 'B' and batch['global_step'] == 4:
                assert padding == 1 and len(rows) == 31 and len(selected) == 8
                candidates = [scalar_advantages(rows, [i]) for i in range(len(rows))]
                matches = [i for i, values in enumerate(candidates) if
                    abs(min(values)-batch['logged_advantage'][0]) < .001 and
                    abs(max(values)-batch['logged_advantage'][1]) < .001]
                record['copy_enumeration'] = dict(
                    candidates=[dict(copied_source_line=rows[i]['source_line'],
                                     full_batch_range=[min(v), max(v)]) for i, v in enumerate(candidates)],
                    log_compatible_source_lines=[rows[i]['source_line'] for i in matches],
                    meaning='Log-constrained scalar inference; actual copy IDs/token tensors were not saved.')
                for row in selected:
                    idx = next(i for i, r in enumerate(rows) if r['source_line'] == row['source_line'])
                    row['all_copy_joint_bounds'] = [min(v[idx] for v in candidates), max(v[idx] for v in candidates)]
                    row['log_compatible_scalar_values'] = [candidates[i][idx] for i in matches]
            batches.append(record)
    return dict(runtime_sha=SHA, authority_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        source_file_hashes=source_hashes, batches=batches,
        limits=['First reinforcement phase is not necessarily the second model call.',
            'Whole-response scalar signal is not the gradient of the Marine count token or actual probability change.',
            'Copy IDs, balancing permutation, token masks and optimizer update tensors are absent from these snapshots.',
            'Winning/losing trajectories are not isolated causality of their first reinforcement answer.',
            'Rounded metric matching is an inference, not recovery of the original FP32/PPO tensors.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze()
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    for b in report['batches']:
        print(b['arm'], b['global_step'], 'wins', b['wins'], 'counts', b['marine_counts'])
