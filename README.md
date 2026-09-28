# SC2 Linear Attention

Compare linear/hybrid-attention models with Qwen in StarCraft II, using shared
observations, actions, execution and training budgets. We separate model learning
from scripted assistance; a high system win rate alone is not architecture evidence.

The default is now **Integrated Fixes v1**, combining tested repairs on Canonical
System-v1. Per-call state bonuses are off; SpB SWA dropout is disabled. The model
pair is original **Qwen3-4B-Instruct-2507 / SpikingBrain-2.0-instruct**, without
project SFT or prior RL. Individual repairs passed their recorded tests;
[combined Level 6 verification](experiments/records/20260906-integrated-fixes-v1/README.md)
completed four updates, but policy-consistency checks remain unresolved.
This is not yet a validated 96-step training baseline.

This diagnostic branch adds opt-in [PPO/generation observations](docs/reference/ppo-diagnostics.md)
on that integration; they are not part of the published default or a verified fix.

## Usage

Prepare the pinned Linux runtime and existing model/SC2 snapshots using
[Setup](docs/setup.md). On an available eight-A100 node:

```bash
export SC2PATH=/workspace/assets/StarCraftII
export TOTAL_EPOCHS=4 SAVE_FREQ=1 SC2_DIFFICULTY=Harder

MODEL_PATH=/workspace/assets/Qwen3-4B-Instruct-2507 RUN_ID=qwen-integrated-l6-s0 \
  bash scripts/train_qwen_instruct2507_level5.sh

MODEL_PATH=/workspace/assets/SpikingBrain-2.0-instruct RUN_ID=spb-integrated-l6-s0 \
  bash scripts/train_spb_level5.sh
```

These are alternative model runs; execute sequentially on the same GPUs.
The historical script names contain `level5`; `SC2_DIFFICULTY=Harder` selects
Level 6. For four available GPUs, also set `N_GPUS_PER_NODE=4` and the exact
`CUDA_VISIBLE_DEVICES`. Add `DRY_RUN=1 VERIFY_ASSETS=0` to inspect configuration
without starting a model or game. Training still stops if no game supplies
a real Planner action; do not bypass that check with artificial training samples.
Standalone SC2 evaluation has a [verified no-policy finish candidate](experiments/records/20260908-garrison-wave-ownership-v1/evaluation-only-finish.md):
CPU regression and one same-seed Level6 game passed, counting the game without
fabricating policy samples. This is an evaluation-boundary result, not RL acceptance.
A further frozen-version game consumed seven Planner answers (six valid),
preserved valid plans and verified production commands, but ended in defeat;
see the same record for the unfinished-Starport blockage. No RL was run.
The [SpB Level5 run](experiments/records/20260908-garrison-wave-ownership-v1/spb-level5-96.md)
stopped before its first update: all eight games won without a Planner decision.
An opt-in [bounded empty-batch candidate](experiments/records/20260908-sc2-empty-policy-batch-v1/README.md)
records those games and advances sampling without a PPO update; default behavior is unchanged.
A separate [g2 Planner handoff candidate](experiments/records/20260908-g2-planner-handoff-v1/README.md)
has passed CPU tests; after two startup failures, its r3 A/B entered SC2 on
September 9 using eight GPUs. This is pure evaluation, with no game-effect result yet.
A [May-inspired opening comparison](experiments/records/20260908-g2-planner-handoff-v1/may-opening-ab.md)
compares full fixed g2 versus Planner from the opening, preserving the current
strategy, prompt and system. Its eight-GPU node47 evaluation completed on
September9: full g2 won 2/4 and opening Planner won 3/4, with both arms exiting 0.
Only two distinct game seeds were used; this is neither RL validation nor recovery
of the missing May runtime.
The [SpB Level6 RL promotion](experiments/records/20260908-g2-planner-handoff-v1/opening-rl8.md)
was stopped before its first update: seven games won, but one entered an empty
candidate loop. A minimal [Factory prerequisite candidate](experiments/records/20260908-g2-planner-handoff-v1/factory-prerequisite-candidate.md)
passes CPU regression; real-game and RL validation have not restarted.
An opt-in [original BO-segment comparison](experiments/records/20260909-bo-segment-protocol-v1/README.md)
now passes CPU queue tests: approximately 15 list entries with count expansion,
not a 15-action cap. The node44 pure evaluation completed: six-action A won 4/4,
restored-segment B won 3/4. B's longer answers often reached the frozen 96-token
budget; this does not establish that the original protocol is worse. No PPO was run.

## Experiment guideline

The default-off [zero-task-signal sampling candidate](experiments/records/20260914-zero-task-signal-sampling-v1/README.md)
uses a bounded collection budget. Its [grouped-score revision](experiments/records/20260914-zero-task-signal-sampling-v1/grouped-score-v2.md)
excludes equal-score reduction residuals without changing rewards or GiGPO; it is not yet a validated learning improvement.

The opt-in [execution-failure feedback candidate](experiments/records/20260911-execution-failure-feedback-v1/README.md)
releases only recorded, unsubmitted selections for replanning. Default is off;
CPU and game acceptance are separate, and existing deployments are unchanged.
Its opt-in [command provenance recorder](experiments/records/20260911-execution-failure-feedback-v1/command-provenance.md)
passed transport/ACK checks in eight games. Its Depot order/lifecycle association
candidate also passed a new eight-game, single-scenario acceptance: 66 starts and
64 completions matched independent audit. It does not change rewards, establish
Marine birth attribution, or demonstrate model learning.

The opt-in [HF/FSDP layer-wrapping candidate](experiments/records/20260910-spb-fsdp-layer-wrap-v1/README.md)
targets the SpB backward-memory peak. It preserves the default HF path and is
opt-in: the four-GPU one-step SpB update and checkpoint passed; the eight-GPU
Level6 run ended after63/96 steps on September11 with an SC2 response timeout.
The separate [gamma-only A/B](experiments/records/20260910-spb-fsdp-layer-wrap-v1/gamma-ab.md)
is the current learning-signal diagnostic, not a restart of that long run.
The four-GPU Level7 continuation was stopped at
step31 at the user's request; step24 is its last complete checkpoint. Neither run
has established a learning improvement. The completed
[temperature-only paired evaluation](experiments/records/20260910-spb-fsdp-layer-wrap-v1/temperature-ab.md)
returned 4/8 wins in each arm across two diagnostic seeds; it did not establish
an improvement. No PPO or changes to the Level6 training task were made.
The opt-in [Level7 checkpoint-resume candidate](experiments/records/20260910-spb-fsdp-layer-wrap-v1/README.md)
adds SC2 seed/counter persistence and verified legacy-final import. The real step4
load and subsequent updates ran through step31; step8/16/24 include the new SC2
state. Stopping did not delete these checkpoints or the original step4.

Start from a clean, privately backed-up commit with a unique run ID. Keep the
two models' game, prompt, reward and interaction budgets aligned. Read the saved
effective configuration before comparing results; never edit a running deployment.
Outputs include original plans, outcomes, metrics and checkpoints. Completion
requires `result.json`/`.done` and a complete checkpoint, not just a process exit.

The earlier [Qwen Level6 step-8 experiment](experiments/records/20260906-integrated-fixes-v1/real-rl-stage8.md)
is a separate record, not the current SpB Level5 schedule. Known limits include
unresolved learning stability, empty-policy batches, legacy batch-copy weighting
and unverified high-level production/combat effectiveness. The current run adds
no early-stop and does not use a single KL threshold as failure. Its six-action/
96-token protocol is frozen; system changes and their evidence are in its record.

See [integrated scope and evidence](docs/integrated-fixes-v1.md),
[historical results](experiments/system-v1/README.md), [provenance](docs/provenance.md),
[method](docs/method.md), and [AGENTS.md](AGENTS.md). Historical r6/r7 results are
retained; they are not results of this new integration.
