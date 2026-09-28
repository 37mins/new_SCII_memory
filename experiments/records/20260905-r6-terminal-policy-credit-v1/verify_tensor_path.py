"""CPU-only check of the actual collector -> GiGPO returns -> reward container."""
import numpy as np
import torch
from types import SimpleNamespace

from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from agent_system.reward_manager.episode import EpisodeRewardManager
from gigpo.core_gigpo import compute_step_discounted_returns
from verl.trainer.ppo.metric_utils import compute_data_metrics


def sample(uid, reward, applied, done, valid=True):
    return dict(
        traj_uid=uid, uid="group", rewards=reward, active_masks=True,
        policy_action_applied=applied, environment_done=done, is_action_valid=valid,
        is_tool_query=False, data_source="sc2",
        prompts=torch.tensor([1, 2]), responses=torch.tensor([3]),
        input_ids=torch.tensor([1, 2, 3]), attention_mask=torch.ones(3, dtype=torch.long),
    )


rows = [
    [sample("zero", 20.0, False, True)],
    [sample("played", .5, True, False, False), sample("played", 20.0, False, True)],
    [sample("real-terminal", -50.0, True, True, False)],
]
batch = TrajectoryCollector.gather_rollout_data(
    None, rows, np.array([20., 20.5, -50.]), np.array([1, 2, 1]),
    {"success_rate": np.array([1., 1., 0.])},
    np.array(["zero", "played", "real-terminal"]), np.zeros(3),
)
assert len(batch) == 2
assert batch.non_tensor_batch["traj_uid"].tolist() == ["played", "real-terminal"]
assert batch.non_tensor_batch["policy_action_applied"].all()
assert not batch.non_tensor_batch["is_action_valid"].any()
assert batch.non_tensor_batch["episode_lengths"].tolist() == [1, 1]
assert np.allclose(batch.non_tensor_batch["success_rate"].astype(float), 2/3)
returns = compute_step_discounted_returns(batch, .95)
assert torch.equal(returns, torch.tensor([20.5, -50.]))
manager = EpisodeRewardManager(SimpleNamespace(decode=lambda *args, **kwargs: "tokens"), 0)
reward = manager(batch)
assert torch.equal(reward[:, -1], torch.tensor([20.5, -50.]))
batch.batch["token_level_scores"] = reward
batch.batch["token_level_rewards"] = reward.clone()
batch.batch["advantages"] = torch.ones_like(reward)
batch.batch["returns"] = reward.clone()
metrics = compute_data_metrics(batch, use_critic=False)
assert metrics["episode/length/mean"] == 1.0
assert metrics["episode/length/max"] == 1.0
assert metrics["episode/length/min"] == 1.0
assert all(np.isfinite(value) for value in metrics.values())
print("TENSOR_PATH_OK: 4 environment rows -> 2 consumed policy samples; "
      "terminal rewards [20.5, -50] retained; all-game success 2/3; "
      "actual compute_data_metrics finite; no GPU or SC2 run")
