# r6 natural-termination A/B

Status: original A/B and September 5 one-update revalidation completed;
earlier eight-update follow-ups interrupted.

This experiment tests one correction on the healthy Canonical r6 source
(`52d1eaa`): a Planner-call budget must not be reported as an SC2 defeat. The
control preserves the historical `call20 -> failed episode` behavior. The
treatment removes that decision from Planner and applies the unchanged r6
20-interaction limit in the RL wrapper as Gymnasium `truncated`.

Both arms use the official `Qwen/Qwen3-4B-Instruct-2507` revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, Level 5 (`Hard`, `Air`), seed 0,
Canonical g2, eight rollouts, one GiGPO/PPO update, and the same r6 prompt,
parser, Executor, old micro, automatic combat, reward shaping, generation
budget, and optimizer. No BO lifecycle patch is present.

The primary result is termination attribution: counts of natural games and
interaction truncations. Win rate, reward, episode length, action validity,
response clipping, PPO KL, clip fraction, and gradient norm are diagnostics.
One update is an interface A/B, not evidence of learning quality or architecture
superiority.

## Runs

- `20260903-r6-natural-termination-ab-v1-qwen-instruct2507-l5-legacy-rl1-s0-r3`:
  historical semantics, `SC2_MAX_STEPS=20`; completed on A100-44 from `afd163e`.
  The reported success rate was 7/8 (87.5%), but the raw log contains seven
  `Result: Victory` events and one `Generate new plan... [20]` event with no SC2
  result. The eighth sample was therefore an artificial call-limit failure that
  nevertheless entered the PPO update. Mean reward was 16.556, valid-action
  ratio 16.1%, response clip ratio 42.9%, PPO KL -0.215, and pre-clip gradient
  norm 526.481. The run exited 0 with a complete step-1 checkpoint.
- `20260903-r6-natural-termination-ab-v1-qwen-instruct2507-l5-gymnasium-rl1-s0-r1`:
  corrected semantics, `SC2_MAX_STEPS=20`; completed on A100-44 from
  `0e0f953`. All eight trajectories ended with real SC2 results: six victories
  and two defeats, with zero truncations. Mean reward was 4.209, valid-action
  ratio 20.8%, response clip ratio 79.2%, PPO KL -0.157, and pre-clip gradient
  norm 949.628. The run exited 0 with a complete step-1 checkpoint.

The control proves that the old reported 7/8 mixed seven real wins with one
call-limit pseudo-result. The treatment proves that true SC2 results pass
through the Gymnasium termination path and can complete one PPO update without
changing the r6 interaction limit. No treatment trajectory reached 20 calls,
so the external-truncation path is covered by the server regression tests but
was not exercised by this stochastic batch. These one-step runs validate the
interface correction; their win rates and optimizer diagnostics are not a
model-quality comparison.

## Follow-up runs

Two follow-ups use the same corrected System-v1 behavior and official
Qwen3-4B-Instruct-2507 snapshot. No Planner, Executor, micro, automatic-combat,
reward, generation, optimizer, or termination setting is changed.

- The Level 5 fixed-BO gate evaluates g1-g5 once each at environment seed 0,
  with training disabled. It tests the model-below system independently of
  Planner quality. The five BO files are the previously registered artifacts
  in `configs/five-bo.json`.
- The Level 6 run changes only the opponent from `Hard` to `Harder` and the
  training budget from the one-step interface gate to eight GiGPO/PPO updates.
  It retains Canonical g2, eight rollouts per update, seed 0, the 20-interaction
  external truncation limit, and all r6 learning hyperparameters.

The valid Level 5 retry writes directly to the shared experiment run directory.
The Level 6 retry writes to a node-local unique directory because A100-47 could
not access the shared filesystem at launch time; it must be copied into the
canonical run archive and hash-verified after NFS recovery.

The first follow-up launch attempts did not reach model execution. The local
Level 5 fixed-BO attempt on A100-42 used an incomplete copied Python runtime;
the Level 6 attempt on A100-47 let the dataset cache fall back to that node's
full root filesystem. Both are retained as infrastructure failures with zero
SC2 outcomes and zero updates.

The valid retries are:

- `20260903-r6-natural-termination-ab-v1-qwen-instruct2507-l5-fivebo-r2-*`:
  running on A100-42 GPUs 0-4 from `c06e77f`, one seed-0 Level 5 evaluation
  game for each of g1-g5. It uses the standard shared `verl-agent-v7-exact`
  runtime and writes directly to the shared experiment run directory.
- `20260903-r6-natural-termination-ab-v1-qwen-instruct2507-l6-rl8-s0-r2`:
  running on A100-47 GPUs 0-7 from `da0ddb4`. The verified model, SC2 assets,
  source, cache, and output are node-local because A100-47 still cannot access
  the shared filesystem. Cache placement is the only change from its failed
  launch attempt; the experiment configuration is unchanged.

The later Level 5 run
`20260903-r6-natural-termination-ab-v1-qwen-instruct2507-l5-canonical-natural8-s0-r1`
completed one PPO update, then stopped before update two. Its 16 outcomes were
15 natural victories and one interaction-limit truncation. This confirmed that
the corrected safety gate works, while also showing that the inherited
20-Planner-call limit prevents otherwise healthy natural-game training.

`SC2_online.zip` contains no Planner-call termination limit: Planner waits for
SC2 `game_end()`. The uncapped follow-up therefore changes only this inherited
limit. `SC2_MAX_STEPS=0` restores the original natural loop; positive values
remain available solely for explicit truncation tests. The follow-up restarts
from the original Instruct-2507 weights and does not continue the one-step
checkpoint.

Run one arm with:

```bash
bash experiments/records/20260903-r6-natural-termination-ab-v1/run_arm.sh corrected
```

Run the eight-step uncapped follow-up with:

```bash
bash experiments/records/20260903-r6-natural-termination-ab-v1/run_l5_uncapped_8step.sh
```

The launcher refuses an existing run directory and records the effective
configuration, source hashes, model manifest, logs, checkpoint, replay, and
per-trajectory outcome ledger.

## 2026-09-05 revalidation

The user approved A100-44 GPUs 0-7 for one update with eight Level 5 rollouts
from the original Instruct-2507 snapshot. Runtime source stays at `8ac1e19`;
no new termination implementation or second behavioral change is introduced.
The r6 prompt, six-action/96-token budget, g2 opening, old micro, combat,
shaping, mean-norm GiGPO and optimizer are frozen. This tests natural outcome
transport and one finite PPO update, not improvement in win rate.

Read-only server inspection found that the earlier uncapped run exited 137
before any update. The September 4 remount retry exited 130 after four of eight
updates, with finite logged metrics, 39 recorded natural outcomes (35 victories)
and no truncations, but no final checkpoint. The 39 outcomes include the
unfinished next batch and are not a completed training result. Both used
Torch 2.8.0 / Transformers 4.57.3 / Ray 2.50.0, unlike the original A/B's
2.7.1+cu128 / 4.55.2 / 2.48.0. They cannot establish a one-variable comparison
against that A/B. No historical artifact was edited.

`run_l5_uncapped_1step.sh` reuses the original A/B dependency directory, checks
its versions, and targets the already-committed repair in a new clean deployment.
It uses the existing model and SC2 assets; it does not install dependencies or
download weights. Run ID:
`20260905-r6-natural-termination-ab-v1-qwen-instruct2507-l5-uncapped-rl1-s0-r2`.
Expected wall time is 30-90 minutes; an outer two-hour process timeout is a
failed/incomplete run, never an SC2 Defeat. Existing run directories are refused.

The gate requires eight natural outcomes, no call-count penalty, finite PPO
metrics and a complete step-1 checkpoint. If no game crosses 20 Planner calls,
the live boundary remains unexercised even when the gate completes; the existing
deterministic call-count regressions cover it separately. No long training is
authorized by this gate.

Local verification on September 5: the eight termination tests and twelve
system-release tests pass. The full discovery cannot import four Planner
contract tests because local Python lacks Gymnasium; those must run with the
original Linux dependency directory before launch.

The original Linux runtime passed all 24 source tests before the first launch.
The `r1` attempt failed at Ray initialization: its default temporary directory
included the long run ID and exceeded the 107-byte Unix-socket path limit.
It produced no SC2 outcome or PPO update and is preserved as an infrastructure
failure. The retry changes only the run/output identifiers and supplies
`RAY_TMPDIR=/dev/shm/r6_nat_0905_r2`. The launcher now requires this explicit
short path; a regression test rejects omission before model loading.
The historical and shared SC2 binary/map paths have identical registered
SHA-256 hashes. Apart from artifact paths, the effective configuration differs
from the original corrected one-update A/B only in `SC2_MAX_STEPS=20 → 0`.

Runtime source `8ac1e19` was already preserved on the tracked remote branch;
the new launcher/record commit has a local and server Git-bundle backup.
Pushing the new record was blocked by the safety reviewer pending verification
of the private remote and approval; it has not been pushed.

Retry `r2` passed the effective-configuration comparison, initialized Ray with
the original dependencies, and started eight SC2 workers. No PPO update had
completed at the 10:07 UTC snapshot. The exact source, model, budget, paths,
test results and backup hashes are in [the revalidation record](revalidation-20260905.json).

The retry completed at 10:30:33 UTC with exit code 0, eight natural results
(seven victories, one defeat), zero truncations, one finite PPO update and a
complete step-1 checkpoint. A victorious trajectory ended at interaction 23,
exercising the live boundary beyond the old limit. Mean episode reward was
18.685, action validity 11.1%, response clipping 38.9%, pre-clip gradient norm
344.053 and PPO KL -0.335. The terminal-interface gate passed; these pre-update
rollout outcomes do not establish improved learning or resolve the remaining
reward/Planner issues. All eight GPUs were free at the 10:44 UTC check.
