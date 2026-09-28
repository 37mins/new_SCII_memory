# Terminal responses are not policy actions

This candidate separates a consumed Planner answer from a terminal socket reply.
It does not change the model query, parser, Executor, combat, reward amounts,
generation budget or PPO loss. The comparison control is A's `legacy` shaping;
the independent `off` ablation is not silently promoted into this fix.

Each ACTION carries a game-local `policy_action_id`. Only
`SC2Planner.get_action()` records that ID when it receives the answer. The next
query or game-end payload echoes `last_consumed_policy_action_id`. SC2Env marks
`policy_action_applied=true` only if that acknowledgement matches this step's
sent ID. An ACTION ignored by the terminal-ACK loop does not acquire credit.
These are transport metadata, not evidence that every parsed BO succeeded.

Reset may receive a final game result before any Planner call. It now caches and
acknowledges that result. The existing full-width generation loop is retained,
so a dummy answer may still be generated; its terminal step is explicitly not
a policy action. Optimizing away that inference is outside this change.

Before GiGPO returns and PPO collation, `policy_transitions()` removes only
unconsumed terminal rows. Real actions remain even when invalid or terminal.
If a separate terminal poll follows a real action, its reward is added once to
that action's step reward; the episode total is unchanged. A zero-decision game
still contributes to the all-game win rate and outcome ledger, but contributes
no policy sample. A completely zero-decision batch fails before PPO rather than
inventing data or silently launching replacement games.

The original `active_masks` remains the environment-liveness mask. It is not
overwritten by the policy filter, so terminal outcomes remain visible to the
success evaluator. The policy batch's `episode_lengths` counts retained action
steps (excluding optional tool rows); the outcome ledger's existing
`decision_count` still counts environment interactions, with the new
`policy_decision_count` reporting consumed Planner answers. Current r6 uses
`normalize_by_length=False`; no reward coefficient is changed.

`episode/reward/mean` and `episode/length/*` are computed from PPO trajectories,
so after this filter they cover only games with consumed actions. They are not
directly comparable to the old eight-game means. Each collection logs
`SC2 policy credit:` followed by every game's original total reward, environment
interaction count, real decision count and excluded row count, including
zero-decision games. Use those all-game totals alongside the all-game win rate.

`prompt_response.jsonl` records the sent ID, acknowledged ID and applied flag.
No prompt substring, format-validity flag or win/loss value decides consumption.
Missing acknowledgements on ordinary nonterminal transitions fail explicitly.
Code and environment must be deployed together; historical socket producers
without these metadata are not supported by this candidate.

The public basis is Farama's
[2025 vector-environment guidance](https://farama.org/Vector-Autoreset-Mode):
non-action reset transitions must be excluded from training buffers or masked,
while real terminal transitions are retained. The upstream
[verl-agent collector](https://github.com/langfengQ/verl-agent/blob/master/agent_system/multi_turn_rollout/rollout_loop.py)
also filters inactive rows before collation. Our SC2 socket acknowledgement is
a project-specific adapter implementing that distinction, not an algorithm
copied verbatim from either source. Neither source claims this improves SC2 wins.

Verification covers the natural-game, no-tool-training r6 path. Optional
trainable memory-tool queries and infrastructure-failure recovery are not
promoted by these tests. The existing batch-copy alignment behavior, six-action
limit, 96-token clipping and Planner validity problems remain separate issues.

The first live gate exposed a scalar-type regression after checkpoint saving:
counted lengths were Python integers, but metric max/min used NumPy `.item()`.
The retry wraps the unchanged count in `np.float32`, preserving the original
collector's type. The CPU regression now includes actual data metrics. See
[the run results](../../experiments/records/20260905-r6-terminal-policy-credit-v1/results.md).
