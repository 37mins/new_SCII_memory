# Paired five-BO evaluation

Status: running, 2026-08-31 11:23 CST. This is a functional evaluation of the current
integration, not the previously proposed one-update PPO smoke. Each original
Qwen3-4B and SpikingBrain-2.0-instruct snapshot plays g1–g5 once on Level 5
(Hard), Flat128, Terran/Air. Neither snapshot has project SFT or prior RL.

The five original files come from the read-only historical asset directory
`assets/repro/mse_micro_five_bo_20260711/fixed_bos` under the old project.
Filenames and exact SHA-256 values are in `configs/five-bo.json`; no BO content
is rewritten. g2 is byte-identical to the canonical two-base input.

## Meaning and controls

`LLMPlayer` loads the selected file into `bo_data` and initially executes it
directly. Only after this list is exhausted does `commander.renew_bo()` call
Planner. This is **fixed scripted opening followed by live planning**, not a
test in which the model makes every decision from game start. A game can end
before meaningful model participation. Record first Planner time and number
of requests separately from terminal notifications.

Keep the canonical prompt, strict validator, six-action/96-token output limit,
old micro, first-20/next-10 auto-attacks and 20 Planner-turn horizon. The shared
launcher only switches to `val_only=True`, one GPU/one trajectory per BO,
no checkpoint saving and greedy validation. No PPO update is executed. The
model may still allocate trainer worker/optimizer objects during startup.

Configuration seed is 0 in all ten runs. The unchanged validation sampler
uses RandomState(1000), so the first actual SC2 seed is **2807211443**, not 0.
Each game's `evaluation_seeds.jsonl` must confirm this; compare actual seeds,
not filenames. There is no new game-time cap beyond the frozen behavior.

The hypothesis is that the integrated system runs across the five openings;
failures can expose unsupported BO mappings, prerequisite/production failures,
or combat losses. Compare plans, execution logs and replay before attributing
failure to one layer. Report natural Victory/Defeat/Tie separately from
trajectory truncation, validation score and strict plan validity.

One game per BO cannot estimate a stable win rate, establish model superiority,
or verify learning/gradients. In particular, identical wins without Planner
participation support system functionality, not model quality.

## Run

Use a clean pushed integration commit and the pinned compatible runtime:

```bash
FIVEBO_ROOT=/path/to/fixed_bos SC2PATH=/path/to/StarCraftII \
MODEL_PATH=/path/to/Qwen3-4B RUN_ID=system-v1-qwen-fivebo-s0 \
bash scripts/eval_five_bo.sh

CONFIG_PATH=configs/spikingbrain2-level5.env \
FIVEBO_ROOT=/path/to/fixed_bos SC2PATH=/path/to/StarCraftII \
MODEL_PATH=/path/to/SpikingBrain-2.0-instruct RUN_ID=system-v1-spb-fivebo-s0 \
bash scripts/eval_five_bo.sh
```

Run models sequentially. Each suite uses GPU 0–4 in parallel; override with
`EVAL_GPU_IDS='0 1 2 3 4'` only after checking ownership. Each game has a
two-hour wall-clock safety timeout, no automatic retry. Historical five-game
Qwen old-micro evaluation took roughly ten minutes; SpB timing is not assumed.

Each game saves manifest/effective config, model and BO hashes, validation
seed ledger, full prompt/response text, SC2 logs/replay and `result.json`.
Completion requires a zero process exit, finite validation metrics, zero
observed training updates and one replay; no model decision is required for
a scripted-opening game. Validation `test_score` is the trainer's score,
not relabeled as summed episode reward. Keep failed runs and restart only
with a new run ID. The suite exit file is `suite.done`.

The isolated node-42 root is `/root/new-scii-system-v1-eval-20260831`.
Per-game IDs and runtime versions are recorded at launch; deployments and
historical runs are not modified. The code source is the integration branch,
not the old five-BO launcher's reconstructed runtime or v1524 micro.

## Initial launch evidence

Qwen suite `20260831-system-v1-qwen-fivebo-l5-s0-25b2ee85` started on node 42,
GPU 0–4, at 11:20:44 CST from clean pushed commit
`25b2ee85e1d0b462f26bfef378380e06fe11baa1`. All five games entered SC2;
their seed ledgers confirmed 2807211443. Both game binaries/map and every Qwen
model file passed the pinned hashes. The Qwen directory name contains
`full15_sft`, but the verified contents are the original public model, not SFT.

One packaging issue was observed: the unchanged Executor writes BO timing
diagnostics to `examination/`. This untracked output makes the next suite's
clean-source gate fail. The integration now ignores that generated directory;
no running deployment or Executor code is edited. The first suite is preserved,
and any subsequent launch must use a new clean deployment and unique run ID.

SpB is queued on the same GPU 0–4 using clean deployment
`c920d57c2ab67d0cbe9fd3ba226e725d8d7a6ab8`; suite ID is
`20260831-system-v1-spb-fivebo-l5-s0-c920d57c`. Relative to the Qwen commit,
only `.gitignore`, this record and the node wrapper's optional model selector
changed. All model/environment/Planner/Executor/validation code and configs
are byte-identical. The original SpB attempt in the used Qwen deployment is
expected to be rejected before GPU work; it must not be counted as a model
failure or silently removed.

The first completed Qwen games exposed an audit distinction: the environment
can deliver a terminal observation and mark its unused response valid. A raw
response record is not necessarily a Planner action. The initial deployed
`result.json` fields `plan_count`/`valid_plan_count` count raw records and must
not be interpreted as decision counts or BO validity. Read the per-record
`semantic_action_evaluable` flag. The reporting utility now separately counts
response records, evaluable actions and valid evaluable actions; this reporting
fix is regression-tested and does not modify the frozen running deployment.

## Qwen result, 11:30 CST

All five games completed with process exit 0, a saved replay and zero PPO
updates. Natural outcomes were four Victories and one Defeat; the mean
validation score was 9.2. This is five single-seed cases, not a stable win-rate
estimate or a comparison with historical training-rollout percentages.

| BO | Natural outcome | Game time | Planner requests | Strict-valid evaluable actions |
|---|---|---|---:|---:|
| g1 | Defeat | 12:07 | 0 | not applicable |
| g2 | Victory | 09:25 | 0 | not applicable |
| g3 | Victory | 09:58 | 0 | not applicable |
| g4 | Victory | 08:39 | 0 | not applicable |
| g5 | Victory | 13:57 | 18 | 1/18 |

g5 first invoked Planner at 12:11. The valid record is step index 12, not the
terminal record; the denominator uses `semantic_action_evaluable`. g1–g4
each have one non-evaluable terminal response, which is not a Planner BO.
Thus three of the four wins happened without model planning. g5's victory
also does not establish good protocol adherence (1/18 strict-valid actions).

At g1's terminal observation, 7955 minerals and 2559 gas remained, with one
SCV and one bunker. This localizes the failure before model takeover; the
specific BO/production/combat cause is not yet established by this summary.
No prompt or combat fix is inferred from this single failed game.

SpB's clean suite started around 11:30 CST. All five original model manifests
passed verification and loaded `SSESWAMoBAForCausalLM` (5.17B parameters).
No SpB win-rate result is available at this update.
