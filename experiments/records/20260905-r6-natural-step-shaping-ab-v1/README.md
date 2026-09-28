# Natural r6: per-call state-reward ablation

Status verified 2026-09-05 12:32 UTC: both arms completed one PPO update with
exit code 0, finite metrics and complete step-1 checkpoints. A won 7/8; B won
6/8. Both have terminal-policy attribution contamination; neither is promoted
as a clean learning baseline. See [completed results](results.md).
Four-GPU Linux regression passed 34/34 tests. Live source was
`c5c4037978a8a97ef788d38f905772736ccb1900`, verified on private GitHub before
launch. See [launch evidence](launch-20260905-four-gpu.json).

On September 5, the user approved the live A/B and private GitHub push. A fresh
resource check found no entirely idle eight-GPU node, so the user separately
approved both arms on A100-44 GPUs 4-7. Each arm still has eight rollouts and
one PPO update. Both arms use the same four-GPU launcher; the previous
eight-GPU result is historical context, not the matched A/B control.

The control is the validated `8ac1e19` natural-termination runtime, tested in
`20260905-r6-natural-termination-ab-v1-qwen-instruct2507-l5-uncapped-rl1-s0-r2`:
one complete PPO update, eight natural outcomes, seven wins, no truncations,
and one victory after 23 interactions. Original r6 and that deployment remain
unchanged. The new reward semantics are isolated on
`codex/fix-r6-step-shaping-off-v1`.

The hypothesis is that removing the per-call absolute state bonus eliminates
positive credit from unchanged states while preserving natural-game/PPO
execution. The sole A/B variable is `SC2_STEP_SHAPING_MODE=legacy|off`.
It is an ablation candidate, not an established improvement or complete remedy
for reward attribution. [Reward definition and public basis](../../../docs/reference/step-shaping.md)
explain the exact scope and why the earlier delta formula is not being merged.

Both proposed arms start from the same official Qwen3-4B-Instruct-2507 revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, not a trained checkpoint. They use
Level 5 Hard/Air, Flat128, g2, seed 0, eight rollouts, one GiGPO/PPO update,
gamma 0.95, mean_norm, learning rate 1e-6, six-action/96-token outputs, and the
same old micro and automatic combat. Torch 2.7.1+cu128, Transformers 4.55.2,
Ray 2.48.0 and Gymnasium 1.2.3 match the completed control. Model, game, map,
BO and frozen source hashes are verified by the unchanged asset verifier.

`run_pair_node44.sh` runs `run_arm.sh legacy` and `run_arm.sh off`
sequentially on A100-44 GPUs 4-7, each from a clean, backed-up deployment with
its own Ray directory and run ID. Estimated combined duration is 70-110 minutes;
each arm has a two-hour infrastructure timeout. A timeout or transport failure
is a failed/incomplete run, never an SC2 defeat. An existing run is refused.
No eight- or 96-update continuation is authorized by this record.

The mechanism test replays the same observations through the actual
`get_action()` method: legacy earns +10 over twenty stationary calls, off earns
zero from this term; all incoming rewards and terminal payloads must remain
identical. A source-boundary test requires every byte of Planner outside
`get_action()` to match `8ac1e19`, including the prompt and parsing functions.
The launcher dry-run must differ only in the shaping flag after normalizing
artifact paths.

The live gate requires natural outcomes for all eight games in each arm,
finite PPO metrics and complete step-1 checkpoints. Off must emit no legacy
state awards. Success count, reward, action validity, response clipping,
gradient norm and episode length are diagnostics; raw reward is not directly
comparable after deleting a term. Single-batch differences are not statistically
established improvements, and first-rollout wins do not measure post-update
learning. A gate failure stops promotion and does not overwrite the control.

New commits must be backed up and privately synchronized before a formal GPU
experiment. The user has now explicitly authorized synchronization to the
private `CeraveNew/new_SCII` repository; its privacy and the pushed SHA must
still be checked. No run is reported as launched before its process and manifest
exist.

The four-GPU adaptation changes only launch resource selection and checkpoint
auditing: `N_GPUS_PER_NODE` defaults to eight, and the manifest records the
actual world size so four checkpoint shards are not incorrectly judged against
an eight-shard requirement. Old manifests without this field retain their
historical eight-rank audit. Model, optimizer, reward, Planner and game logic
are not changed by this adaptation. The paired script stops before the second
arm if the first fails its completion audit or selected GPUs are occupied.

Local verification passed 13 `test_r6*.py` tests and 15 system-release tests
using `python3 -B -m unittest discover -s tests -p '<pattern>' -v`.
`bash -n` and `git diff --check` also passed. The local interpreter lacks
Gymnasium, so the additional four Planner contract tests require the pinned
Linux environment. The five new reward tests demonstrate the +10 versus 0
stationary-state case, preservation of incoming and terminal rewards, and the
frozen Planner boundary; launcher tests check flag propagation and reject an
unknown mode before run creation.

The original Linux dependency overlay passed all 32 tests on A100-44.
Both full experiment wrappers completed dry-runs; after removing artifact-path
differences, their configurations differ only in the shaping flag. The source
and test-log hashes are in [offline-validation.json](offline-validation.json).
Those results preceded GPU approval. The subsequent approved four-GPU
deployment passed 34/34 CPU tests and full model/SC2/input hashing. Its two
full-wrapper dry-runs still differed only in the shaping flag after normalizing
artifact paths. A's live manifest confirms four ranks, GPUs 4-7, eight rollouts,
one update, Instruct-2507 and `SC2_MAX_STEPS=0`; B was queued at that launch
snapshot and subsequently ran to completion.
