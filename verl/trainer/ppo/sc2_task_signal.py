"""CPU gate for one real SC2 GiGPO group, before copying or model RPCs."""

from copy import deepcopy
from collections import defaultdict

import numpy as np
import torch


def inspect_task_signal(batch, config, reward_fn):
    # These are the production reward/advantage functions, not a new reward.
    from gigpo import core_gigpo
    from verl.trainer.ppo.ray_trainer import (
        apply_invalid_action_penalty, compute_advantage, compute_response_mask,
    )
    from verl.trainer.ppo.reward import compute_reward

    if batch.batch is None or not len(batch):
        raise RuntimeError("No real SC2 policy batch; zero-signal sampling cannot fabricate decisions")
    if len(set(batch.non_tensor_batch["uid"])) != 1:
        raise ValueError("Zero-signal candidate supports exactly one SC2 group per collection")
    # DataProto deepcopy invokes __getstate__, which consolidates/locks the
    # original TensorDict. Clone tensors directly to keep the accepted batch writable.
    data = type(batch).from_dict(
        tensors={k: v.clone() for k, v in batch.batch.items()},
        non_tensors=deepcopy(batch.non_tensor_batch), meta_info=deepcopy(batch.meta_info),
    )
    mask = compute_response_mask(data).bool()
    if not mask.any() or not mask.any(dim=-1).all():
        raise RuntimeError("SC2 policy response without effective tokens")
    data.batch["response_mask"] = mask
    data.batch["step_rewards"] = core_gigpo.compute_step_discounted_returns(data, config.algorithm.gamma)
    # Reward debug printing can consume NumPy RNG. Inspection must not alter
    # the later legacy copying/balancing randomness, including in the A arm.
    rng = np.random.get_state()
    try:
        scores, _ = compute_reward(data, reward_fn)
    finally:
        np.random.set_state(rng)
    data.batch["token_level_scores"] = scores
    if config.actor_rollout_ref.actor.get("use_invalid_action_penalty", True):
        data, _ = apply_invalid_action_penalty(
            data, config.actor_rollout_ref.actor.invalid_action_penalty_coef,
        )
    data.batch["token_level_rewards"] = data.batch["token_level_scores"]
    data = compute_advantage(
        data, adv_estimator=config.algorithm.adv_estimator, gamma=config.algorithm.gamma,
        step_advantage_w=config.algorithm.gigpo.step_advantage_w,
        gigpo_mode=config.algorithm.gigpo.mode,
        gigpo_enable_similarity=config.algorithm.gigpo.enable_similarity,
        gigpo_similarity_thresh=config.algorithm.gigpo.similarity_thresh,
    )
    active = data.batch["advantages"][mask]
    if not torch.isfinite(active).all():
        raise FloatingPointError("Nonfinite task advantage; not a zero-signal group")
    episodes = {}
    for uid, reward in zip(batch.non_tensor_batch["traj_uid"], batch.non_tensor_batch["episode_rewards"]):
        episodes.setdefault(str(uid), float(reward))

    # Test equality BEFORE mean reduction: equal FP32 inputs can acquire a
    # nonzero rounding residual in x - mean(x). No magnitude threshold is used.
    # Match episode_norm_reward: only the first score of each trajectory is used.
    episode_scores = {}
    for uid, score in zip(data.non_tensor_batch["traj_uid"],
                          data.batch["token_level_rewards"].sum(dim=-1).tolist()):
        episode_scores.setdefault(str(uid), score)
    episode_varies = len(set(episode_scores.values())) > 1
    step_groups = core_gigpo.build_step_group(
        data.non_tensor_batch["anchor_obs"], data.non_tensor_batch["uid"],
        config.algorithm.gigpo.enable_similarity, config.algorithm.gigpo.similarity_thresh,
    )
    step_scores = defaultdict(set)
    for uid, score in zip(step_groups, data.batch["step_rewards"].tolist()):
        step_scores[uid].add(score)
    varying_step_groups = sum(len(values) > 1 for values in step_scores.values())
    score_variation = episode_varies or (
        config.algorithm.gigpo.step_advantage_w != 0 and varying_step_groups > 0
    )
    nonzero = bool((active != 0).any())
    if not score_variation:
        kind = "reduction_residual_only" if nonzero else "exact_zero"
    elif not nonzero:
        kind = "zero_joint_advantage"
    else:
        kind = "episode_score_difference" if episode_varies else "step_score_difference_only"
    return {
        "has_task_signal": bool(score_variation and nonzero),
        "signal_kind": kind,
        "episode_return_varies": len(set(episodes.values())) > 1,
        "episode_score_varies": episode_varies,
        "step_groups_with_score_variation": varying_step_groups,
        "max_abs_task_advantage": float(active.abs().max()),
        "real_responses": len(batch), "effective_tokens": int(mask.sum()),
        "episode_returns": episodes,
        "criterion": "grouped_score_variation_and_nonzero_GiGPO_before_padding_v2",
    }
