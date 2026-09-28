# Terminal-policy credit: r1 and the scalar-type retry

## r1: failed after checkpoint saving

Run `20260905-r6-terminal-policy-credit-v1-qwen-instruct2507-l5-legacy-4gpu-rl1-s0-r1`
used source `89424474e739fca9bed76cdf5f092df8e99c35d8` on node 44 GPUs 4-7.
It started at 12:49:26 UTC and ended at 13:11:31 UTC on September 5, exit 1.
The original directory and all artifacts remain unchanged.

Eight games finished naturally: 7 wins, 1 defeat, no artificial truncation.
Win rate is 87.5% (Wilson 95% interval approximately 52.9%-97.8%); one pre-update
rollout batch cannot establish improved learning. The collector had 27 active
environment rows: 23 consumed answers and four unconsumed terminal replies.
Consumption IDs matched and the four terminal-only rows were excluded. The old
batch alignment then copied one real row to reach 24; duplicate weighting is
not fixed here. All four policy-participating games won; three other wins and
the only defeat had no Planner decision. Only 1/23 real answers passed the
full contract. Three invalid and one valid real terminal answer all received
20.5. The independent format-reward fix is NOT included in r2.

The training path reached actor update and saved all four model/optimizer/extra
state shards plus `global_step_1/data.pt`. It then failed before `logger.log`:

```text
compute_data_metrics -> episode_lengths[unique_idx].max().item()
AttributeError: 'int' object has no attribute 'item'
```

`result.json` reports `completed_steps=0`, `checkpoint_complete=true`; metrics
are `[]`. Its `metrics_finite=true` is vacuous for an empty list, not proof of
finite gradients/KL. No numerical PPO pass is claimed. The counter reflects
missing logged steps, not proof that no parameter update was executed.

This regression was introduced by our terminal-policy change: the recomputed
length used Python `sum(...)`, whereas the original collector used NumPy
`float32` lengths. Object-array collation preserves this scalar distinction;
metric max/min expect `.item()`. A CPU reproduction on the unchanged r1
snapshot failed at the identical line. Our prior CPU check covered collector,
returns and reward but omitted `compute_data_metrics`.

## r2: one-line type correction

The only production change from r1 is `np.float32(sum(...))` for the retained
decision count. It restores the existing scalar type without changing counted
actions, reward, validity or PPO math. A regression now executes the actual
collector → reward → `compute_data_metrics` path. The wrapper uses a new r2
RUN_ID and Ray directory; all non-path settings are frozen.

The user approved the same node 44 GPUs 4-7, original Qwen3-4B-Instruct-2507,
Level 5, eight rollouts and one update, replacing the proposed format gate.
R2 starts from fresh original weights, not r1's checkpoint. It uses the same
record and branch, preserving r1. Launch and verification evidence follow.

Source `76180970d2e139ef2da052df659546621a84539d` passed all 45 pinned Linux CPU
tests, including real data metrics. Local lightweight tests passed 40 tests;
one Torch/NumPy integration test was explicitly skipped locally and passed on
Linux. The frozen runtime is
`/root/new-scii-r6-terminal-policy-20260905/code-7618097`. Comparing its resolved
configuration to r1 after path normalization returned `{}`. Production source
diff against r1 contains one changed line in `rollout_loop.py`; format-reward
commit `2fa14ca` is not in this runtime.

Node-44 CPU log `cpu-tests-7618097.log` SHA-256:
`e659692b223fdeaa0f8a436ff634ef7a5de0174524aa0274ce702b01b022f733`.
Git bundle `new-scii-terminal-policy-r2-20260905.bundle`, saved locally and under
`/root` on node 44, SHA-256:
`b0b883ed04c1ba7d723b716e1fd22e8a30884ca567f0f3d9d0f52739f05aba70`.
R1's raw artifact hashes are preserved in [the hash record](r1-artifact-hashes.json).

R2 started at 13:18:59 UTC (21:18:59 Beijing) on September 5. The manifest is
formal/clean and model loading was observed at 13:19:34 UTC; see
[launch evidence](launch-r2-20260905.json).

## r2: completed live gate

Ended at 13:39:55 UTC, exit 0. One step was logged with finite metrics and all
four checkpoint shards complete. The prior scalar-type crash did not recur.
Saving the checkpoint took 744.467 seconds; actor update took 16.675 seconds.
[Completion JSON](r2-completion.json) preserves metrics, outcomes and hashes;
the remote source/run remain unchanged.

There were eight natural games: six wins, two defeats, no truncation. The 75%
win rate has a Wilson 95% interval of approximately 40.9%-92.9%; it is one
pre-update rollout batch, not an improvement estimate. Eighteen active rows
contain 14 consumed answers and four terminal-only replies. The consumption-ID
and reward-conservation gate passed; the four terminal-only rows are excluded.
All four policy-participating games won. Two wins and both losses had no model
decision. The legacy batch-copy alignment remains unchanged.

Logged pre-clip grad norm is 36.063, PPO clip fraction 0.044, PPO KL -0.058 and
reference KL loss 0.213. Logged reward mean 22.875 covers policy-participating
games; success rate 0.75 covers all eight games. Strict validity is zero and
response-length cap ratio is 0.875; neither issue is fixed by scalar correction.
These values establish a finite update, not a clean learning-capability result.

The user subsequently authorized preparing and conditionally launching one
format-reward gate. After r2 passed, that run started at 13:45:01 UTC from fresh
original weights on the same four GPUs. Relative to r2, only the terminal
format-bonus verdict changes; [its record](../20260905-r6-format-reward-v1/README.md)
contains the candidate and launch evidence. No longer training is queued.
