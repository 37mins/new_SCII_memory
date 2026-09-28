# SC2 episode termination

Planner does not decide when a game ends. This restores the behavior in
`SC2_online.zip` (SHA-256
`e40fb7dad98f4c7d336031f3b2047ee3e15113ce14925830cfb6540aaa5888ab`):
`get_action()` transports observations and actions, while `game_end()` is called
only after `run_game()` returns an SC2 result.

The SC2 environment uses the Gymnasium step API:

```text
observation, reward, terminated, truncated, info
```

This follows the [Gymnasium Env API](https://gymnasium.farama.org/api/env/),
which separates MDP terminal states from time limits and other external
truncation conditions.

`terminated=true` means SC2 returned `Victory`, `Defeat`, or `Tie`.
`truncated=true` means the RL wrapper reached an explicitly configured positive
interaction limit before SC2 returned a result. The compatibility boundary used
by the existing trajectory collector computes `done = terminated or truncated`;
it retains both original flags in `info` and in the outcome ledger.

Natural training restores the original package's uncapped Planner loop:

```bash
SC2_MAX_STEPS=0 bash scripts/train_qwen_level5.sh
```

Zero is the default and means no Planner-call limit. A positive value is reserved
for an explicit truncation test. Reaching such a configured limit records
`terminated=false`, `truncated=true`, `game_result=null`, and
`outcome_reward=0`; it is not counted as a loss. Because this GiGPO path has no
bootstrap value for incomplete trajectories, training stops before a PPO
update if any collected trajectory is truncated.

Every trajectory end appends one row to
`episode_outcomes.jsonl`. The run audit accepts a completed training step only
when every expected trajectory has a natural result and none is truncated.
SC2 transport or reset failures raise an error and are not converted into game
outcomes.

This change does not alter the r6 prompt, parser, build-order logic, Executor,
micro, automatic combat, step shaping, model settings, or optimizer.
