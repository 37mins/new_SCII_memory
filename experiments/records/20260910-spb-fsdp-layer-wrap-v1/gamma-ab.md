# Gamma-only short RL A/B

Status (September11 afternoon verification): **r2 A and B completed**, each adding
eight global steps (5–12), exit0 and archived step12 checkpoint acceptance.
A ended11:09:23 and B13:34:28 Beijing. Both won39/64 training games. B preserved
more late opening variants but did not improve aggregate wins; gamma1 is not
promoted as a fix. See [Level6 source/log/CPU audit](l6-training-signal-followup.md).
The earlier09:01 snapshot had A sampling global5 and B pending; that is no longer
the current state.
**r1 failed before any game/update** and is retained. See
[failure and startup correction](gamma-startup-fix.md) and
[r2 actual-start record](gamma-ab-retry-launch.json).
The following queue description is the historical September10 launch record.
User approved this experiment on September10. Frozen source is
`f3803883bb4727b2f372c82b0d05b0f2ff81ea70`, privately backed up on node44.
The three CPU tests passed on the pinned node44 runtime, CUDA hidden (2.138s).
The queue started at22:58, waits at most30 minutes for all eight GPUs, and exits
without training if that slot remains unavailable. GPU0/1 were occupied by an
external vLLM task at verification; no process was stopped. Full command,
paths, test results and bundle hashes: [launch record](gamma-ab-launch.json).
Hypothesis: discounting by Planner-call count contributes to plan concentration
and loss of within-group learning signal. This is not a confirmed implementation bug.
The existing [CPU audit](sampling-reward-audit.md) establishes a fewer-call bias in
one same-reward group; gamma=1 removes that bias but cannot create reward diversity.

Both arms use the Level6 long-run **step4 model and optimizer**, eight GPUs, eight
new global steps (numbered **5–12**), eight games per step. A uses gamma=0.95, B=1.0.
The parent runtime is `6a1657f7e045dcf998495a3d0adfb331572bf60d`, run
`20260910-spb-fsdp-layer-wrap-v1-spb2-l6-g2-opening-quota-rl96-8gpu-s0-r1`, under
`/mnt/data0601/yangjinyue/new_SCII/experiments/20260910-spb-fsdp-layer-wrap-v1/runs`.
Its step4 contains 8 model, 8 optimizer, 8 extra-state shards and data.pt
(63,570,316,070 bytes including metadata). These are presence/size checks;
successful runtime loading must be recorded separately.

The parent has **no sc2_state.pt**. Neither arm claims uninterrupted SC2 resume.
Both create fresh memory-off game environments and use the same seed sequence:
2732, 43567, 42613, 52416, 45891, 21243, 30403, 32103; each is repeated for the
eight rollouts of that step. Existing checkpoint loading restores per-rank RNG,
model, optimizer, scheduler and dataloader. The shared experiment-only hook seeds
the missing driver RNG to 0 **after loading**, and records this boundary. It does
not change rewards, responses, model layers or optimizer updates. Actual seed
ledgers must match; differing subsequent sampling streams after A/B diverge are
not represented as identical trajectories. Reused training seeds are not held-out eval.

CPU inspection of the actual 1,947-byte data.pt with the same Torch RandomSampler
and eight DataLoader workers found the saved epoch was after its last yield,
before StopIteration. Eight naive outer epochs would yield `[0,1,1,1,1,1,1,1]`.
The shared launch hook closes this **already exhausted** iterator before starting
the eight new epochs; any unexpected real sample is an error, not silently dropped.
This fixes the new experiment's step budgeting, not a claimed historical learning bug.

Frozen in both arms: original SpB model provenance, original reference model,
FP32 parameters/temperature normalization, BF16 autocast, HF cache, eager,
layer FSDP, swa_dropout=0, temperature0.4/top_p0.9/top_k0, 192 tokens,
original approximately15-entry/count-expansion plus quota prompt, g2/opening
Planner, current strategy/filters/fallback/Executor/micro, history0/off,
reward/shaping-off, GiGPO mean_norm, learning rate1e-6 and all existing losses.
No Refinery forced-answer hook is included. The learning-rate schedule remains
budgeted for96; total_epochs=8 bounds **new** updates without rebuilding the schedule.

Run IDs use the parent experiment prefix followed by
`spb2-l6-g2-gamma0.95-s4plus8-8gpu-s0-r2` and
`spb2-l6-g2-gamma1.0-s4plus8-8gpu-s0-r2` for the approved September11 retry.
The unused B/r1 and failed A/r1 are not reused or overwritten.
Run A then B on node44 GPU0–7, total ceiling8 hours, no retry, no96-step promotion.
Save complete checkpoints at global8 and12 in each new run; expected new storage
about255GB, retaining parent and failure artifacts. Recheck all GPUs before each arm;
do not stop other tasks. Node43's separate Level6 run is not changed or restarted:
it exited on September11 03:43 after63 updates when SC2 stopped returning data.

Primary observations: semantic plan variants for identical opening queries;
within-group episode/step-return spread; groups with zero or near-zero advantages;
real update counts, valid actions and Planner calls. Also report clipping, KL,
gradient, truncation, true termination, original answers and game results.
Identical titles do not establish semantic identity, nor do different titles establish
diversity. Finite gradients and merely lower KL do not establish learning success.

After both arms complete, compare step4 and each step12 using identical held-out
game seeds and evaluation settings. Do not label training rollout wins independent
evaluation. Current entry points cover training only; do not silently schedule
extra eval or GPU budgets. Report stability/diversity changes separately from
game-performance evidence; eight steps cannot establish architecture superiority.

Launch from a clean, privately backed-up SHA:
`SOURCE_SHA=<sha> SC2_GAMMA_AB_AUTHORIZED=1 bash experiments/records/20260910-spb-fsdp-layer-wrap-v1/run_gamma_pair_node44.sh`.
Dry-run either arm with DRY_RUN=1, VERIFY_ASSETS=0, existing MODEL_PATH/SC2PATH,
and a new OUTPUT_DIR. No production source or historical deployment is edited.
