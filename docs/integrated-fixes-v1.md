# Integrated Fixes v1

On the current diagnostic branch, the September16
[empty-ability query repair](../experiments/records/20260914-zero-task-signal-sampling-v1/empty-abilities-fix.md)
returns the existing `[Empty]` observation text when no fully built units remain.
CPU regression passes; real-game acceptance is pending. Existing deployments are
unchanged, and this observation-boundary repair is not a learning-signal fix.

The September11 [execution-failure feedback candidate](../experiments/records/20260911-execution-failure-feedback-v1/README.md)
is opt-in and not an accepted integration. It adds failure feedback and releases
recorded unsubmitted selections; no current deployment or RL configuration is changed.

On the diagnostic branch, the opt-in [command recorder](../experiments/records/20260911-execution-failure-feedback-v1/command-provenance.md)
observed Depot issuers' orders in the September12 r2 eight-game check, whose frozen
position matcher linked0/66 starts. The September13 recorder-only candidate uses
the observed issuer and completed command history. On the same recorded inputs,
the old/candidate callbacks link0/66 starts and0/64 completions respectively.
This CPU replay is not a new live-game or reward/RL acceptance. Existing deployments
and the published default remain unchanged; recording is still opt-in.

The opt-in [quota prompt candidate](../experiments/records/20260909-bo-segment-protocol-v1/quota-game-ab.md)
uses `SC2_CLARIFY_BO_QUOTA=1` only with the original-count protocol. It clarifies
cumulative quotas, the example and answer completion; default is off. A fixed-query
test improved valid answers from22/41 to28/41, but eight formerly valid answers
regressed. This is not a game/RL acceptance; parser, rules and training are unchanged.

The September9 opt-in [BO segment protocol candidate](../experiments/records/20260909-bo-segment-protocol-v1/README.md)
is CPU-validated only. `SC2_ORIGINAL_BO_SEGMENT=1` restores approximately15 list
entries and exact-name count expansion for valid answers, without an expanded-action
cap. The approved opening-Marine guard keeps its insertion but does not delete a
valid long plan to fit six slots. Invalid-answer rules, six-action default, rewards,
execution and training remain unchanged. No GPU/learning acceptance is claimed.

This is the September 6 integration of existing fixes, not a new planner or
learning algorithm. Its parent is `8cb4f1888d245dc06ea3d1f19d804d79af0fff05`,
which descends from private main `d42096e`. It excludes the unvalidated GPU
diagnostic hooks from `1e38a6a`; those remain on the diagnostic branch. A CPU-only
metric repair is not presented as proof of healthy model KL.

| Included change | Existing evidence | Default here |
|---|---|---|
| Natural outcome / truncation separation | [8 natural games, including a win past call 20](../experiments/records/20260903-r6-natural-termination-ab-v1/revalidation-20260905.json) | No Planner-call limit |
| Terminal policy credit and scalar statistics | [One finite update, 14 real decisions, 4 terminal rows excluded](../experiments/records/20260905-r6-terminal-policy-credit-v1/r2-completion.json) | Enabled |
| Full-answer format reward | [Invalid terminal answers receive no bonus](../experiments/records/20260905-r6-format-reward-v1/results.md) | Enabled |
| Qwen cached generation positions | [Native advancing positions, cache preserved](../experiments/records/20260905-qwen-generation-position-v1/README.md) | Enabled for Qwen3 |
| Exhausted unit-ID allocation | [10,000-ID regression and finite L6 update](../experiments/records/20260906-unit-id-exhaustion-v1/README.md) | Enabled |
| Per-call state bonus removal | [B completed; all 33 bonus logs zero](../experiments/records/20260905-r6-natural-step-shaping-ab-v1/results.md) | `SC2_STEP_SHAPING_MODE=off` |
| SpB SWA dropout | [Actor/ref configs both zero; cache true](../experiments/records/20260906-unit-id-exhaustion-v1/launch-spb-node47-r1.json) | `SPB_DISABLE_SWA_DROPOUT=1` |
| Reference KL reporting | [CPU real-update A/B preserves loss, gradients and weights](reference/kl-metric.md) | Collect every micro-batch; no loss change |

Relative to the parent, production changes are only launcher/config defaults:
turn on the already-tested off/SWA-zero options and make the generic launcher
select Instruct-2507. No Planner, parser, Executor, micro, rollout, reward formula,
optimizer or loss implementation is rewritten. Full query logs remain enabled.
The historical Qwen3-4B launcher remains available for explicit old-model use.

The protocol remains 6 BO / 96 response tokens, HF/eager with cache, no thinking,
canonical g2, GiGPO mean_norm, gamma .95, learning rate 1e-6 and clip 1.0.
Removing per-call shaping does not remove terminal state shaping, outcome rewards
or invalid-action penalties. `legacy` remains an explicit reward ablation;
`SPB_DISABLE_SWA_DROPOUT=0` retains the upstream config only for documented tests.
Neither override alone recreates historical r6/r7: use their frozen source and config.

The prior delta reward, BO-lifecycle behavior changes, full15/count-v3 protocol,
new micro, attack-wave changes and KL diagnostic forwards are not imported.
No checkpoint, SFT dataset, model asset or shared Python installation is altered.

The combined experiment validates compatibility and actual setting propagation,
not each fix's causal effect again. The [run record](../experiments/records/20260906-integrated-fixes-v1/README.md)
separates CPU results, live completion and learning conclusions. Long training
remains blocked on policy-consistency evidence and viable handling of empty
policy batches. Batch-copy weighting, BO completion semantics and high-level
combat effectiveness remain known limitations, not silently declared repaired.

On this diagnostic branch, the September 7 authorized
[empty-BO control-flow candidate](../experiments/records/20260906-integrated-fixes-v1/empty-bo-frame.md)
changes only the same-frame refresh loop from `while` to `if`. It is a CPU-stage
repair, not part of the earlier completed integration runs or a passed RL gate.

The later [child Python-RNG candidate](../experiments/records/20260906-integrated-fixes-v1/r4-python-rng-prefix.md)
adds `SC2_SEED_PYTHON_RANDOM=1` as an opt-in at the game-child entry. Default is off;
no current RL launch is changed. Two CPU-only, same-seed 120-second fixed-BO
prefixes reproduced all recorded frames. Full-game determinism and learning
remain unverified; this is not a KL or policy-update repair.

The September 7 [pending-addon placement candidate](../experiments/records/20260906-integrated-fixes-v1/r4-pending-addon-fix.md)
extends the existing addon reservation to SCV production-building orders before
construction starts. It changes only `BasePlayer.find_placement` and is active
on this candidate branch under the existing `avoid_production_addons=True`
default. CPU regression passed on the actual burnysc2 7.1.3 environment. One
same-seed Level6/g2 game subsequently observed three completed Factory Tech Labs
and three distinct tanks, reaching the first Planner at 10:00. It stopped there
without an answer or complete replay: this supports the production-chain repair,
not a win-rate or RL claim. It is not an accepted integration/release fix.
Executor selection, BO completion semantics and combat behavior remain unchanged.

The subsequent [Marine/Reactor occupancy candidate](../experiments/records/20260906-integrated-fixes-v1/r4-reactor-conflict.md)
changes only local Marine producer filtering to the free-slot condition from
burnysc2 7.1.3. Other commands, BO accounting and combat remain unchanged.
Eight CPU tests passed on the actual dependency. One same-seed Level6/g2 game
then observed three completed Reactors, no BO timeouts, three completed Factory
Tech Labs and four distinct tanks. All 103 Marine submissions had a free slot;
the previous run had 93 submissions to already-full slots. It stopped at the first
Planner with no answer or natural outcome: this passes that single-game mechanism
gate, not a win-rate/RL gate or an integration/release acceptance.

The September 8 [garrison ownership candidate](../experiments/records/20260908-garrison-wave-ownership-v1/README.md)
only excludes existing attack-wave members from garrison selection, including idle
units waiting at a rally point. It uses the existing group tags, without changing
scouting, rally positions, wave sizes or training. Eight targeted and 75 related
CPU tests passed. A same-seed Level6/g2 game then won before the first Planner
decision; replay auditing found no garrison overrides of active-wave members.
The evaluation process still exited 1 because the collector required policy
samples even for this natural win. This is a local mechanism result, not release
or RL acceptance; the original failed run remains unchanged.

The [standalone evaluation finish candidate](../experiments/records/20260908-garrison-wave-ownership-v1/evaluation-only-finish.md)
addresses that boundary on this branch. Only `trainer.val_only=True`, SC2 and
`is_train=False` use per-game metadata, including zero-decision games. No policy
tensors are invented. Training collection, credit filtering and empty-batch
rejection are unchanged; validation inside training is also outside this change.
CPU tests cover the collector, metric aggregation and result writer. A new
SpB Level6/g2 game at the same seed subsequently passed the full standalone entry:
exit0, natural Victory, reward20, one game, zero consumed policy decisions and
zero PPO updates. All 491 own commands and the tracker stream match the old
failed-evaluation game. This accepts that evaluation boundary only; no-policy
training remains rejected, RL and Viking survival are unverified, and the old
failed result is not relabelled.

A subsequent unchanged-runtime SpB game used the historical first seed2807211443:
seven consumed answers, six valid, exit0 and natural Defeat. Valid plan order
was preserved and 25 production commands matched the replay, but six MULE BOs
were skipped and a legal Viking selection timed out behind an unfinished
Starport. This verifies real Planner participation, not complete execution or
Level6 learning. Construction interruption remains for a separate read-only audit.

The September 8 [bounded no-policy training candidate](../experiments/records/20260908-sc2-empty-policy-batch-v1/README.md)
is opt-in, after Level5's first eight games ended without a model decision. It
retains those outcomes, advances the existing seed stream without an optimizer
update, and stops at explicit consecutive/total limits. Default rejection and
real-decision credit remain unchanged. This is a collection-boundary candidate,
not a policy-stability, learning or checkpoint-resume acceptance.

The September 8 [g2 Planner handoff candidate](../experiments/records/20260908-g2-planner-handoff-v1/README.md)
is opt-in (`SC2_FIXED_BO_PREFIX_LIMIT=138`), restricting only the initial fixed BO
queue to #0–#137. Default 0 retains all 237 entries. It preserves the existing
completion/skip boundary, Strategist source, prompt, tactics and subsequent Planner
queues. 106 CPU tests passed, including mocked-answer queue changes and launcher
configuration; no game/GPU validation or learning acceptance has been performed.

The September 9 [May-inspired opening A/B](../experiments/records/20260908-g2-planner-handoff-v1/may-opening-ab.md)
adds `SC2_FIXED_BO_PREFIX_LIMIT=-1`: no initial fixed actions, then the existing
Commander/Strategist/Planner path from the first callback. Default 0 and candidate
138 are unchanged. Current strategy, prompt and all downstream mechanisms stay
frozen; this is not a byte-exact May runtime recovery or an accepted improvement.

The authorized September 9 [Factory prerequisite candidate](../experiments/records/20260908-g2-planner-handoff-v1/factory-prerequisite-candidate.md)
adds a missing Factory to Planner candidates before filtering dependent tech
actions. Only that two-line branch and its source hash change; template, strategy,
execution and training remain frozen. CPU regression passes, but no new game or
PPO update has tested the candidate. Existing invalid-answer fallback may select
the added Factory; those actions are still rule-generated, not model choices.
