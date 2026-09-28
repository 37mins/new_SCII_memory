"""CPU scalar reconstruction of completed scene evaluations, before batch copying.

Run with PYTHONPATH pointing to frozen c684176 and its existing dependencies:
  CUDA_VISIBLE_DEVICES= python -B audit_scene_signal.py --runs-root RUNS --output NEW_JSON
No tokenizer/model loading. One synthetic valid token carries each answer's
constant GiGPO scalar; these are not saved PPO tensors or token-weighted metrics.
"""
import argparse
from collections import Counter
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import OmegaConf
import torch

from agent_system.reward_manager.episode import EpisodeRewardManager
from agent_system.sc2_episode import policy_transitions
from gigpo import core_gigpo
from verl import DataProto
from verl.trainer.ppo.ray_trainer import apply_invalid_action_penalty
from verl.trainer.ppo.sc2_task_signal import inspect_task_signal

SHA = 'c684176d41bc4debe830896572cf0a5c51e9a43a'
ROOT = Path(core_gigpo.__file__).resolve().parents[1]
SEEDS = (3974528993, 1183530000, 3446496864, 2996859384,
         3847406327, 1824802814, 175019443, 3122016241)
SOURCES = ('gigpo/core_gigpo.py', 'agent_system/reward_manager/episode.py',
           'agent_system/sc2_episode.py', 'agent_system/multi_turn_rollout/rollout_loop.py',
           'agent_system/environments/strategic_sc2_env_manager_v2.py',
           'verl/trainer/ppo/sc2_task_signal.py', 'verl/trainer/ppo/ray_trainer.py',
           'verl/trainer/config/ppo_trainer.yaml', 'verl/trainer/main_ppo.py')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path):
    return [dict(json.loads(line), source_line=i) for i, line in
            enumerate(path.read_text().splitlines(), 1)]


def effective_config(overrides):
    cfg = OmegaConf.load(ROOT/'verl/trainer/config/ppo_trainer.yaml')
    parser = OverridesParser.create()
    for key, value in overrides.items():
        key = key.lstrip('+')
        if key.startswith(('algorithm.', 'actor_rollout_ref.actor.', 'env.')):
            # Match Hydra CLI grammar; YAML dotlist would parse `off` as False.
            parsed = parser.parse_override(f'{key}={value}').value()
            OmegaConf.update(cfg, key, parsed, merge=True, force_add=True)
    return cfg


def reconstruct(raw, endings, width=1):
    """Keep natural-terminal credit and collector's env-major answer order."""
    episodes, selected = [], []
    for end in sorted(endings, key=lambda x: x['env_index']):
        assert end['terminated'] and not end['truncated']
        active = [r for r in raw if r['env_idx'] == end['env_index']
                  and r['step_index'] <= end['step_index']]
        assert active == sorted(active, key=lambda x: x['step_index'])
        # The collector accumulates environment rewards into an FP32 array.
        total = np.zeros(1, dtype=np.float32)
        for row in active:
            total += np.array([row['reward']])
        policy = policy_transitions([dict(r, active_masks=True,
            rewards=r['reward'], environment_done=r['done']) for r in active])
        assert len(policy) == end['policy_decision_count'] > 0
        assert abs(sum(r['rewards'] for r in policy) - sum(r['reward'] for r in active)) < 1e-8
        episodes.append(dict(env=end['env_index'], outcome=end['game_result'],
            reward=float(total[0]), calls=len(policy),
            terminal_polls=len(active)-len(policy)))
        for call, row in enumerate(policy, 1):
            assert row['memory_context'] == ''  # off: logged prompt equals anchor
            selected.append(dict(row, call=call, episode_reward=float(total[0]),
                                 episode_length=len(policy)))
    n = len(selected)
    batch = DataProto.from_dict(tensors={
        'prompts': torch.ones(n, 1, dtype=torch.long),
        'responses': torch.ones(n, width, dtype=torch.long),
        'input_ids': torch.ones(n, width+1, dtype=torch.long),
        'attention_mask': torch.ones(n, width+1, dtype=torch.long),
    }, non_tensors={
        'uid': np.array(['scene']*n, dtype=object),
        'traj_uid': np.array([f"env{r['env_idx']}" for r in selected], dtype=object),
        'anchor_obs': np.array([r['prompt'] for r in selected], dtype=object),
        'rewards': np.array([r['rewards'] for r in selected], dtype=object),
        'active_masks': np.ones(n, dtype=bool),
        'episode_rewards': np.array([r['episode_reward'] for r in selected], dtype=object),
        'episode_lengths': np.array([r['episode_length'] for r in selected], dtype=object),
        'is_action_valid': np.array([[r['is_action_valid']] for r in selected], dtype=object),
        'data_source': np.array(['sc2']*n, dtype=object),
    })
    return batch, episodes, selected


def decompose(batch, config):
    """Call production functions; no alternate reward, gamma or penalty."""
    reward = EpisodeRewardManager(SimpleNamespace(decode=lambda *a, **k: ''), 0,
                                 normalize_by_length=False)
    inspection = inspect_task_signal(batch, config, reward)
    inspection['synthetic_carrier_tokens'] = inspection.pop('effective_tokens')
    mask = batch.batch['attention_mask'][:, 1:]
    batch.batch['step_rewards'] = core_gigpo.compute_step_discounted_returns(batch, config.algorithm.gamma)
    environment_returns = batch.batch['step_rewards'].clone()
    batch.batch['token_level_scores'] = reward(batch)
    batch, _ = apply_invalid_action_penalty(batch, config.actor_rollout_ref.actor.invalid_action_penalty_coef)
    index, traj = batch.non_tensor_batch['uid'], batch.non_tensor_batch['traj_uid']
    groups = core_gigpo.build_step_group(batch.non_tensor_batch['anchor_obs'], index,
        config.algorithm.gigpo.enable_similarity, config.algorithm.gigpo.similarity_thresh)
    episode = core_gigpo.episode_norm_reward(batch.batch['token_level_scores'], mask, index, traj)
    step = core_gigpo.step_norm_reward(batch.batch['step_rewards'], mask, groups)
    joint = episode + config.algorithm.gigpo.step_advantage_w * step
    assert float(joint.abs().max()) == inspection['max_abs_task_advantage']
    # FP64 *centering of the same FP32 scores* identifies reduction residuals.
    # It does not change training precision or recompute a new discount path.
    ep64 = core_gigpo.episode_norm_reward(batch.batch['token_level_scores'].double(), mask, index, traj)
    step64 = core_gigpo.step_norm_reward(batch.batch['step_rewards'].double(), mask, groups)
    return dict(inspection=inspection, episode=episode[:, 0].tolist(), step=step[:, 0].tolist(),
        joint=joint[:, 0].tolist(), environment_returns=environment_returns.tolist(),
        penalized_returns=batch.batch['step_rewards'].tolist(),
        fp64_centering_joint=(ep64 + config.algorithm.gigpo.step_advantage_w * step64)[:, 0].tolist(),
        anchor_sizes=sorted(Counter(groups).values()))


def audit(run, seed):
    def read(name): return json.loads((run/name).read_text())
    result, manifest, load = read('result.json'), read('manifest.json'), read('checkpoint_evaluation.json')
    assert result == read('.done') and result['status'] == 'completed'
    assert result['process_exit_code'] == result['training_updates'] == 0
    assert manifest['source_commit'] == SHA and not manifest['dirty']
    assert manifest['mode'] == 'evaluation' and manifest['world_size'] == 4
    assert load['loaded_global_step'] == 4 and load['checkpoint_load_returned'] and load['val_only']
    assert load['feedback'] == '0' and load['game_seeds'] == [seed]*8
    assert [r['seeds'] for r in jsonl(run/'evaluation_seeds.jsonl')] == [[seed]*8]
    overrides = read('effective_config.json')['overrides']
    cfg = effective_config(overrides)
    assert cfg.algorithm.adv_estimator == 'gigpo' and cfg.algorithm.gamma == .95
    assert cfg.algorithm.gigpo.mode == 'mean_norm' and cfg.algorithm.gigpo.step_advantage_w == 1
    assert not cfg.algorithm.gigpo.enable_similarity and not cfg.algorithm.use_kl_in_reward
    assert cfg.actor_rollout_ref.actor.use_invalid_action_penalty
    assert cfg.actor_rollout_ref.actor.invalid_action_penalty_coef == .1
    assert cfg.env.sc2_memory_mode == 'off' and cfg.env.history_length == 0
    assert cfg.env.prompt_response_log_max_prompt_chars == 0 and cfg.env.rollout.n == 8
    raw, endings = jsonl(run/'prompt_response.jsonl'), result['episode_outcomes']['outcomes']
    assert len(endings) == 8
    batch, episodes, selected = reconstruct(raw, endings)
    parts = decompose(batch, cfg)
    # Scalar advantages must not depend on artificial response width.
    wide, _, _ = reconstruct(raw, endings, width=7)
    assert decompose(wide, cfg)['joint'] == parts['joint']
    assert abs(sum(e['reward'] for e in episodes)/8 - result['metrics']['val/text/test_score']) < .001
    for i, row in enumerate(selected):
        row['prompt_sha256'] = hashlib.sha256(row.pop('prompt').encode()).hexdigest()
        row.pop('memory_context')
        for key in ('episode', 'step', 'joint', 'environment_returns', 'penalized_returns', 'fp64_centering_joint'):
            row[key] = parts[key][i]
    summary = dict(seed=seed, wins=sum(e['outcome']=='Victory' for e in episodes),
        mean_reward=sum(e['reward'] for e in episodes)/8, calls=len(selected),
        invalid=sum(not r['is_action_valid'] for r in selected),
        per_game_calls=[e['calls'] for e in episodes], anchor_sizes=parts['anchor_sizes'],
        episode_range=[min(parts['episode']), max(parts['episode'])],
        step_range=[min(parts['step']), max(parts['step'])],
        joint_range=[min(parts['joint']), max(parts['joint'])],
        max_abs_joint=max(map(abs, parts['joint'])),
        max_abs_fp64_centering_joint=max(map(abs, parts['fp64_centering_joint'])),
        **parts['inspection'])
    files = ('manifest.json', 'effective_config.json', 'result.json', '.done',
             'checkpoint_evaluation.json', 'evaluation_seeds.jsonl',
             'prompt_response.jsonl', 'episode_outcomes.jsonl')
    return dict(run=str(run), summary=summary, episodes=episodes, decisions=selected,
                artifact_sha256={n:digest(run/n) for n in files})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    sources = {}
    for name in SOURCES:
        frozen = subprocess.check_output(['git', 'show', f'{SHA}:{name}'], cwd=ROOT)
        assert frozen == (ROOT/name).read_bytes(), name
        sources[name] = digest(ROOT/name)
    reports = []
    with contextlib.redirect_stdout(io.StringIO()):
        for seed in SEEDS:
            name = f'20260914-zero-task-signal-sampling-v1-spb2-A4-l6-screen-s{seed}-eval8-4gpu-r1'
            reports.append(audit(args.runs_root/name, seed))
    report = dict(runtime_sha=SHA, source_sha256=sources,
        analysis_script_sha256=digest(Path(__file__)), torch_version=torch.__version__, numpy_version=np.__version__,
        scope='CPU reconstruction of per-answer task advantages before copying/balancing; no PPO or model loaded',
        limits=['Eval never formed a PPO batch; no saved advantage/response-mask/copy-order tensors.',
                'Synthetic one-token carriers preserve scalar scores; not actual token counts or PPO weighting.',
                'FP64 centering is arithmetic attribution only, not a proposed precision change.',
                'Nonzero advantage is not evidence of better tactics, causal game gain, or learning.'], scenes=reports)
    with args.output.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps([r['summary'] for r in reports], ensure_ascii=False))


if __name__ == '__main__':
    main()
