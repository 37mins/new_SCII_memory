# Server layout

The September16 combat-reward gamma candidate D uses a new isolated source and
control directory `/dev/shm/sc2-combat-gamma-20260916-r1` on node46, GPUs0–3,
with existing `/root/new-scii-spb-unitid-20260906/python` and `/usr/include`.
No old runtime or dependency is modified; models/A4 and outputs use shared storage.
The approved cap is four new global steps / 32 games / 90 minutes, no automatic retry.
See `experiments/records/20260916-combat-result-reward-v1/README.md` for actual launch state.

The September15 node46 v2 pair keeps its frozen training source `2e2cf8b` under
`/dev/shm/sc2-v2-pair-node46-20260915-r1/code`. The separately committed wait script
runs outside that source in tmux `sc2_v2_wait46_r1`, logs under `watch-r1`, and only
checks GPUs0–3. It may dispatch the existing pair once before the original deadline
1789492635; it neither extends the budget nor retries a failed launch. See
`experiments/records/20260914-zero-task-signal-sampling-v1/signal-supply-next.md`.

The September15 scene-training post-evaluation is approved on node47, eight GPUs,
one60-minute budget and24 games total. GPUs0–3 run A4 then SceneB8; GPUs4–7 run
SceneA8 concurrently. Independent source/control live under
`/dev/shm/sc2-scene-posteval-20260915-r1`, private source backup under shared
`new_SCII/staging/20260915-scene-posteval-r1`, and Ray/cache use separate
`/dev/shm/pe_<label>_0915_r1` directories. The verified B-r2 Python/include and
the three existing shared checkpoints are read-only. No new checkpoint or
large asset copy, process cleanup, automatic retry or training is authorized.
See `scene-training.md` for source, paths, actual startup and acceptance.

The September15 B-only scene-training retry is approved on node47 GPUs0–3,
with one2-hour ceiling including deployment and about130GiB new shared storage.
Node43 stages its pinned Python/include and complete original A4 (not the new A8)
under `/mnt/data0601/yangjinyue/new_SCII/staging/20260915-scene-B-r2`.
Node47's full system disk is not used: source, extracted dependencies and control
socket use `/dev/shm/sc2-scene-B-retry-20260915-r2`, Ray/cache use
`/dev/shm/sc2_scene_B_0915_r2`; training outputs use the unique shared scene-B-r2 run.
Only dependency/storage paths may differ from r1; old deployments, A8, failed B-r1
and other processes are preserved. No automatic retry or96-step continuation.
Actual staging/training acceptance is in `scene-training.md` and the new launch record.

The September14 zero-task-signal A/B is approved on node43 GPUs0–3 serially,
32 games per arm within one3-hour budget. Source and private Git backup are under
`/root/new-scii-zero-signal-20260914`, separate from the CPU staging and old deployments.
Control `pair-r1`, tmux `sc2_zero_signal_r1`, and `/dev/shm/zs_{A,B}_0914_r1`
are unique. Models/parent A4/dependencies use the existing node43 local assets
read-only; new logs/checkpoints use shared `new_SCII/experiments/20260914-zero-task-signal-sampling-v1/runs`.
No dependency deployment, checkpoint copy, retry or long-run promotion is included.
See that experiment record for actual startup and outcomes.

The separately approved September13 A4/scene-step5 paired evaluation reuses
those same node43 local models/checkpoints and dependencies read-only. A new
`code-<SHA>` uses control `a4-scene-eval-r1` and Ray directories
`/dev/shm/se_A4_0913_r1` / `/dev/shm/se_A5scene_0913_r1`. GPUs0–3 run the two
eight-game evaluations serially within one45-minute ceiling, no PPO, checkpoint,
large asset copy or retry. See `early-training-signal.md`; only the new run
artifacts are written, never the step5 training deployment or older runs.
The follow-up eight-seed panel received separate four-GPU/45-minute approval.
Runtime `e8d8de1` passed25 server CPU tests; the serial controller started at18:43
Beijing on September13, with a19:28 deadline. It uses `FEEDBACK_SCENE_SET=historical8`, a new
`a4-multiseed-eval-r1` control directory and `/dev/shm/me_<label>_0913_r1` Ray paths.
No rollout or training module changes; see the same experiment record for status.

The September13 authorized A4 scene-coverage training reuses node43 local assets,
four-shard A-r2 step4 and the same local feedback-experiment run root. Its new
`code-<SHA>`, `a4-scene1-r1` control and `/dev/shm/a4s_0913_r1` Ray/cache path are
independent. GPUs0–3, eight games at explicit seed1098341402, one new global step,
one full step5 checkpoint, at most45 minutes; no evaluation, retry or extension.
See `early-training-signal.md` for authorization and actual status. Old deployments,
results and environments are unchanged, and command tracing stays off.

The September13 online-correction evaluation reuses node43's existing local
assets and the feedback experiment runs root. Only two answers are replayed;
the third and later answers are online. Its independent `code-<SHA>` remains
under `/root/new-scii-execution-feedback-20260911`, control `online-from3-r1`,
Ray directory `/dev/shm/oc_0913_r1`. The approved ceiling is eight GPUs/eight
games/15 minutes, without PPO, new checkpoints or automatic retries. Existing
deployments and results remain read-only; see `supply-reward-candidate.md`
for the actual startup and acceptance status.

The September11 feedback RL r2 restart is authorized on node43 GPUs0–3,
sequential A then B within one three-hour deadline. It reuses verified local
assets and outputs under `new-scii-execution-feedback-local-20260911/shared`.
The new frozen source remains under `new-scii-execution-feedback-20260911/code-<SHA>`,
control is `rl4-serial-r2`, Ray/cache directories are
`/dev/shm/sc2_feedback_rl4_{A,B}_0911_r2`. The controller records host memory and
only starts B after A finishes with a complete step4 checkpoint and releases GPUs.
No old deployment/result is changed; no automatic retry or long-run promotion.

The approved September11 supply-order diagnostic uses node44's independent
`/root/new-scii-execution-feedback-20260911/code-<SHA>` and `supply-order-r1` control.
Its eight-GPU/eight-game evaluation has one45-minute ceiling and no PPO/retry.
Existing node44 Python overlay, system headers, shared original SpB/SC2 assets
are read-only; no large model copy is required. Ray/HF/Triton use
`/dev/shm/sc2_supply_order_0911_r1`, outputs a unique shared feedback-experiment run.
Node43's parallel four-step feedback RL pair remains unchanged and uses its own
already-verified local assets/output directories. See `supply-order.md`/`rl4.md`
for provenance and actual running state rather than inferring launch from layout.

The September11 execution-failure feedback A/B uses node47's new
`/root/new-scii-execution-feedback-20260911/code-<SHA>` and a complete private
Git bundle, never the original deployment. Existing
`/root/new-scii-spb2-level5-control/python` and
`/dev/shm/spb-unitid-20260906/include` are read-only. GPUs0–3 and4–7 each run
four evaluation games concurrently, within one shared hour, no PPO or retry.
Ray and generated caches use `/dev/shm/sc2_feedback_{A,B}_0911_r1`; the pair
control is `pair-r1` in the new root. Shared runs use the separate
`20260911-execution-failure-feedback-v1` experiment. Model step24/SC2 assets
are read in place. See that experiment record for actual startup and results;
this planned layout is not proof of successful loading or gameplay.

After node47 resource contention aborted r1 before gameplay, the user approved
the same pair on node43 for at most57 minutes. Its thin node43 wrapper reads
`/root/new-scii-spb-level5-96-20260908/{python,include}` in place, uses r2 RUN_IDs,
`/dev/shm/sc2_feedback_{A,B}_0911_r2` and `pair-r2` in node43's independent
`/root/new-scii-execution-feedback-20260911`. Neither node47 deployment nor
its failed runs are overwritten; no dependency or checkpoint copy is required.

After r2's NFS startup timeout, the user approved a bounded local-storage r3.
Only required assets are copied from44 into node43's new
`/root/new-scii-execution-feedback-local-20260911/shared`, preserving paths below
the old `/mnt/data0601` prefix. The record's `FEEDBACK_STORAGE_ROOT` redirects
model/SC2/FLA/MoBA/checkpoint and run/Hydra output together; Python/headers stay
at their existing read-only node43 locations. Source uses a new `code-<SHA>`
under the existing feedback root, with `pair-r3` and r3 Ray/RUN IDs. The shared
mount, old deployments and r1/r2 results are unchanged. Copy has one30-minute
deadline; only successful verification permits a new57-minute eight-GPU pair.
See the feedback record for actual completion, not just this planned layout.

The approved fixed-failure-input comparison reuses those verified node43 local
assets on GPUs0–3, original SpB then step24, within one20-minute deadline.
It starts no SC2/PPO. A new `code-<SHA>` and unique `fixed2x4-4gpu-r1` run isolate
the diagnostic from r3; temporary/cache files use `/dev/shm/sc2_fixed_feedback_0911_r1`.

The September10 SpB OOM candidate has an isolated CPU-only checkout at
`sc2-a100-44:/root/new-scii-spb-fsdp-20260910/code-08fa2e3` with a full private
Git bundle beside it. It reuses the existing Python overlay read-only and does
not alter the original quota run or deployment. Two CPU/Gloo ranks passed the
FSDP/HF regression with CUDA hidden. The subsequently approved one-step GPU run
uses a separate clean `code-7e5b1f9` in the same family, node44 GPUs4–7, eight
Level6 rollouts and a one-hour ceiling. Its tmux session is `sc2_spb_fsdp_rl1_r1`,
control logs live in `rl1-r1`, and shared outputs use the unique run under
`new_SCII/experiments/20260910-spb-fsdp-layer-wrap-v1/runs`. Existing deployments
remain untouched. Following one-step completion, the user separately authorized
node43 GPUs0–7, 96 steps from the original SpB, a 48-hour limit and a full checkpoint
every four steps. This uses node43's new
`/root/new-scii-spb-fsdp-20260910/code-6a1657f`, existing
`/root/new-scii-spb-level5-96-20260908/{python,include}` dependencies read-only,
and separate `rl96-node43-r1` control/Ray/run paths. No old deployment is modified.
See the experiment record for actual startup/acceptance state; deployment alone
is not evidence of model loading or training.

The separately approved Level7 short run reuses node44's pinned overlay and
headers read-only, with a new clean `code-<SHA>` under the same FSDP root.
Its control directory is `/root/new-scii-spb-fsdp-20260910/l7-rl4-r1`, tmux is
`sc2_spb_fsdp_l7_rl4_r1`, and Ray is `/dev/shm/sc2_spb_fsdp_l7_rl4_0910_r1`.
It uses GPUs4–7, four global steps/eight rollouts each, at most four hours,
saving at step4. Shared outputs use a unique Level7 run in the same experiment.
Node43's Level6 source, processes and budget are unchanged. Only log/checkpoint
storage is shared; neither launch guarantees absence of storage contention.

The approved September 8 g2-handoff A/B uses node44's separate
`/root/new-scii-g2-planner-handoff-20260908`: private bundle, one clean
`code-<SHA>` for both arms and unique `pair-r1` logs. GPUs0–3/4–7 each run four
SpB Level6 evaluation games; the whole suite has a 60-minute ceiling and no PPO.
Existing Python/headers and shared models/SC2 remain read-only. Shared results
use `new_SCII/experiments/20260908-g2-planner-handoff-v1/runs` and distinct Ray
directories `/dev/shm/g2handoff_A_r1`, `/dev/shm/g2handoff_B_r1`. See its `gpu-ab.md`
for time-sensitive startup and completion; this layout is not evidence of launch.
The all-eight-card launch on44 was rejected before startup when GPU0 was taken
by another job. The same budget is allocated as two independent four-GPU arms:
A on44 GPUs4–7, B on42 GPUs4–7, both using new commit-specific source under the
same separate root name. Node42 reuses its already-copied pinned dependencies
under `/root/new-scii-spb-level6-96-20260908`; there is no new environment copy.
Both timed arms share one absolute UTC deadline. The unused44 deployment and
CPU preflight records remain intact.

The September 8 failed-seed control reuses node44's frozen `code-e1e25ce` and
existing SpB dependencies read-only. Only a new orchestration checkout/bundle
under `/root/new-scii-spb-failed-seed-20260908` is deployed. After GPU0 was taken
by another task, the user authorized GPUs1–4 for eight original-model Level6
games at historical seed43567, without PPO or a
checkpoint substitute. See `failed-seed-control.md` in the garrison record.

The newly authorized node42 Level6 96-step run uses independent
`/root/new-scii-spb-level6-96-20260908`, a fresh copy of the verified Python
overlay/headers, clean `code-e1e25ce` and separately frozen orchestration. Shared
weights/results remain in their existing locations. See `spb-rl4.md` in the
September 8 garrison record. No other task or old deployment is changed.

The agent mistakenly treated the added Level5 run as replacing Level6;
that was not an explicit user cancellation. SpB Level5 ran on node43 GPUs0–7.
Its independent root is
`/root/new-scii-spb-level5-96-20260908`: clean `code-e1e25ce`, separately frozen
orchestration and a copy of node44's verified Python overlay/headers. Shared
weights, FLA/MoBA and SC2 remain read-only; the isolated dependency transfer is
staged under shared `new_SCII/staging/spb-l5-96-node43-20260908`. No original
environment is overwritten. See `spb-level5-96.md` in the same record for live status.

The September 8 SpB four-step RL suite reuses node44's read-only
`/root/new-scii-garrison-wave-20260908/code-e1e25ce` and existing dependencies.
Its new `orchestration-<SHA>` starts initial eight-game evaluation, four global
steps, then paired step-4 evaluation on GPUs6,7 after an ownership/free check.
Each stage gets a unique run under the same experiment and its own `/dev/shm`
Ray directory. No model/environment deployment or old-source edit is needed.
See that experiment's `spb-rl4.md` for the authorized bounds and live status.

The September 8 garrison-wave candidate uses a new node44 root
/root/new-scii-garrison-wave-20260908 with a private Git bundle and code-<SHA>.
One idle GPU1, one SpB Level6/g2 game at seed2621640384, at most30 minutes, no PPO
or automatic retry. Existing /root/new-scii-spb2-level5-control/python, system
Python3.12 headers and shared model/SC2/FLA/MoBA assets are read-only. Results use
shared new_SCII/experiments/20260908-garrison-wave-ownership-v1/runs; old source
and runs are untouched. Its record distinguishes preparation from live startup.

The September 8 opt-in air-options candidate uses separate node48 root
`/root/new-scii-planner-air-options-20260908`, with a Git bundle backup and clean
commit-specific source. It reuses the valid-BO candidate's pinned dependencies,
model and SC2 read-only. One GPU6 game, seed2621640384, at most30 minutes, no PPO
or retry. Outputs use shared `new_SCII/experiments/20260908-planner-air-options-v1/runs`.
Its experiment record defines model/rule attribution; original deployments and
results are unchanged. Deployment preparation is not evidence of game startup.

The September 8 valid-BO passthrough candidate uses independent node48 root
`/root/new-scii-valid-bo-20260908` for a Git bundle backup and commit-specific source.
It reuses existing `/root/new-scii-qwen-kl-metric-20260906/python` and `include`,
the original shared SpB weights and FLA/MoBA, without modifying those dependencies.
One GPU, one Level6/g2 game at seed2621640384, at most30 minutes; no PPO or automatic
retry. Runs live under shared `new_SCII/experiments/20260908-valid-bo-passthrough-v1/runs`.
Original node44 deployments/results remain read-only. Approval and live status are
recorded in that experiment's README, not inferred from this deployment plan.

The September 7 opt-in Python-RNG prefix candidate uses independent node44 source
`/root/new-scii-r4-python-rng-20260907/code-<SHA>`. Its two sequential games are
CPU-only, capped at 120 game seconds each; existing r4 dependencies/SC2 are read-only.
Separate repeat1/repeat2 runs retain their own logs and replay, with no model/Ray
startup. Original r4 deployment and evaluations are untouched. See the integrated
record `r4-python-rng-prefix.md` for budget, identities and verification status.
The authorized extension reuses that directory family with a new `code-<SHA>`:
two sequential CPU games stop on the first Planner request, maximum 600 wall
seconds each, separate `preplanner` runs. No model answer or PPO update is made;
the previous `code-7aad777` and its two-minute runs remain frozen.
The approved pending-addon acceptance reuses this directory family with a new
frozen SHA, one CPU-only g2/Level6 game at seed 2621640384 and a 600-second wall
limit. Its unique `pending-addon-g2-l6` run stops before any model answer; no
existing deployment, dependency or previous run is changed. See the same
integrated record, `r4-pending-addon-fix.md`.
The subsequent authorized Marine-slot acceptance uses `code-c15d02b` in this same
family and a unique `reactor-slots-g2-l6` run: one CPU-only game, same seed and
600-second limit. No baseline deployment or existing run is overwritten.
Its full-game Qwen paired evaluation subsequently completed using independent
`pair-8968d1d` orchestration and unchanged `code-836460a` / `code-c15d02b` gameplay
deployments, one GPU per arm. The authorized SpB add-on reuses `code-c15d02b`,
with a new `pair-<SHA>` orchestration directory and unique single-game run.
It reads the same pinned Python plus existing shared FLA/MoBA sources, keeps
generation KV cache on and SWA dropout off, and does no PPO. See the integrated
record `r4-reactor-paired-eval.md` for identities, results and actual live status.

The September 7 single-update r4 replay was subsequently authorized on idle node
47, retaining eight GPUs and the same 30-minute total budget. Its independent
source/staging directory is `/dev/shm/r4-first-update-node47-20260907`; it reads
the existing `/dev/shm/spb-unitid-20260906/python` dependencies. An isolated
screen session avoids the full root disk and absent tmux. Logs and parameter
evidence use the unique node47 run under the integrated experiment's shared runs.
No weight/environment copy or duplicate node46 job is scheduled. See
`r4-first-update-replay.md` for the frozen identities and actual execution state.

The September 7 r4 first-optimizer-update replay is prepared independently on
node 46 under `/root/new-scii-r4-first-update-20260907`. Its `runtime-ca7885d`
is a verified archive of r4's frozen source, and its diagnostic source receives
a separate clean commit. The existing `/root/new-scii-spb-unitid-20260906/python`
overlay is read-only. Outputs use a unique run in the integrated experiment;
temporary caches use `/dev/shm/sc2_r4_first_update_node46_r1`. User authorization
is eight GPUs for at most 30 minutes, one update, no game, allowing another idle
node in parallel without interfering with 44. Preparation is not a GPU launch;
see the integrated record's `r4-first-update-replay.md` for status.

The September 7 empty-BO control-flow repair uses a new source-only directory
`/root/new-scii-empty-bo-frame-20260907/code-<SHA>` on node 44 for CPU regression.
It reads the existing `/root/new-scii-spb2-level5-control/python` dependencies;
no model copy, dependency installation, SC2 launch or GPU experiment is authorized.
The r3 deployment and all run artifacts remain untouched.

On September 7 at 00:16 Beijing, the approved eight-GPU real-RL r3 suite started
on A100-44 from `/root/new-scii-real-rl-20260906/code-aa1d20d`, with tmux
`sc2_real_r3` and separate Ray paths `/dev/shm/sc2_real_r3_<stage>`.
It reuses the pinned Python overlay and shared original Instruct-2507 assets
read-only. Initial evaluation precedes eight global training steps and paired
step-8 evaluation; no step 9 is queued. The prior r2 failure/deployment remain.
Time-sensitive stage evidence is in the integrated record's `real-rl-r3-launch.json`.

On September 6 at 15:03 Beijing the user authorized an additional node-48 KL
probe without stopping either node-44 run. After external jobs reclaimed part
of 48, only GPUs 1–2 were idle; the r8 two-GPU launcher preserves global mini8,
eight rollouts and one update. It reuses the existing pinned Python and headers
under `/root/new-scii-qwen-kl-metric-20260906` read-only. New committed code lives
under `/root/new-scii-kl-integrated-20260906`, with tmux `qkl_int_l6_r8`, Ray
`/dev/shm/qkl_int_l6_r8` and its own shared run. See the KL record for actual
launch status; no existing run or external process is stopped.
Before launch, the user requested history_length=5; r8 remains held while its
actual SC2 history semantics are checked. The launcher is preparation, not a
running experiment. Existing node-44 experiments remain unchanged.

On September 6 at 14:16 Beijing, the user approved the KL one-step probe on
A100-44 GPUs 0–3 in parallel with the unchanged integrated four-step on GPUs
4–7. The unstarted serial KL r6 is cancelled. The new r7 uses its own committed
deployment under `/root/new-scii-kl-integrated-20260906`, tmux `qkl_int_l6_r7`,
Ray `/dev/shm/qkl_int_l6_r7`, and a unique shared run. Python/model assets remain
read-only; no running process or deployment is changed. See the KL record.

The original project remains a read-only source of evidence. Integrated Fixes v1
is prepared in a separate local release worktree; no historical server runtime
is overwritten. Time-sensitive deployment snapshots below are not availability
guarantees.

The approved September 6 combined Level-6 four-update gate uses A100-44 GPUs
4–7 only after the existing eight-GPU run has completed and those GPUs are free.
New code goes to `/root/new-scii-integrated-fixes-20260906/code-<SHA>`, reusing
the pinned `/root/new-scii-spb2-level5-control/python` read-only. New output uses
shared `new_SCII/experiments/20260906-integrated-fixes-v1/runs/<RUN_ID>`;
Ray uses `/dev/shm/sc2_int_l6_4_r1` and tmux `sc2_int_l6_4_r1`. See its record
for actual launch/completion; preparation alone is not a running experiment.

The approved follow-up KL single-update probe reuses only those same four GPUs
after successful completion/release. Its separate source is
`/root/new-scii-kl-integrated-20260906/code-1fc9f33`; the pinned Python overlay
and assets stay read-only. The exact launch gate, hashes and unique r6 run ID
are in `experiments/records/20260906-qwen-kl-metric-v1/ready-node44-r6.json`.
No other node, retry or long run is authorized by this follow-up.

Verified historical canonical archives are on `sc2-a100-197` under
`/data/yangjinyue/new-scii-qwen-level5-canonical/runs`; Qwen r6 and SpB r7
originally ran on node 198. Rush seed1/2 full runs are on `sc2-a100-42` under
`/root/new-scii-level5-rush-96/runs`. They must not be edited in place.

The user project's shared evidence root is
`/mnt/data0601/yangjinyue/new_SCII`. Use a bounded stat check before accessing
NFS. Refer to AGENTS.md for aliases, task-ownership checks and authorization;
no credentials belong here. Paths were read-checked on 2026-08-31, not guaranteed
permanent node availability.

The approved 2026-08-31 five-BO evaluation uses the separate node-42 root
`/root/new-scii-system-v1-eval-20260831`, with commit-specific `deployments`
and unique `runs`. Its pinned launch is
`experiments/system-v1/run_five_bo_node42.sh`. Existing model files and the
isolated `/root/new-scii-spb2-rl-smoke` Python/FLA/MoBA dependencies are read-only
inputs. The shared `/dev/shm/.../python` alternative had drifted to Torch 2.8 /
Transformers 4.57 and is not used; the chosen environment imports Torch 2.7.1,
Transformers 4.55.2 and Ray 2.48.0. No package installation is performed.

The approved September 5 natural-termination revalidation uses A100-44 under
`/root/new-scii-r6-termination-verify-20260905`. Its clean `code` deployment is
pinned to `8ac1e19`; the Git bundle and separately versioned launcher are saved
alongside it. The read-only dependency overlay is
`/root/new-scii-spb2-level5-control/python`, verified as Torch 2.7.1+cu128,
Transformers 4.55.2 and Ray 2.48.0. Outputs use a new run below
`/mnt/data0601/yangjinyue/new_SCII/experiments/20260903-r6-natural-termination-ab-v1/runs`.
Historical deployments and runs remain untouched.

The September 6 unit-ID exhaustion fix starts from the unchanged f6a784c
deployment at `/root/new-scii-qwen-generation-position-20260905/code-f6a784c`
on A100-44. Its new commit-specific deployment is staged separately under
`/root/new-scii-unit-id-exhaustion-20260906`, using the same read-only Python,
model and SC2 assets. The approved short test uses GPUs 4–7 and a unique run
under shared `experiments/20260906-unit-id-exhaustion-v1/runs`; no prior run or
runtime is overwritten. See that experiment record for current status.

The same record's September 6 SpB Level-5 96-update run uses A100-46, after
checking all eight GPUs and the shared assets. Its new isolated deployment root
is `/root/new-scii-spb-unitid-20260906`, with commit-specific code and a separately
approved copy of the pinned Python overlay. The original SpB model and historical
FLA/MoBA source directories remain read-only; outputs have a unique shared run ID.

During deployment, another Gemma service occupied six GPUs on 46. The user then
approved moving this SpB run to the still-idle A100-47. Because 47's root disk is
full, dependencies and writable caches use `/dev/shm/spb-unitid-20260906` and
code uses shared `new_SCII/deployments/spb-unitid-20260906/code-<sha>`; logs and
checkpoints remain in the same unique shared run. No old files are removed and
model KV-cache behavior is unchanged. Node 47 has no tmux; its existing screen
hosts `spb_l5_96_drop0_r1`. Inspect it with
`SCREENDIR=/dev/shm/spb-unitid-20260906/screen screen -ls`.

That SpB run failed at 11:09 Beijing on September 6, after one PPO update; the
next rollout had no consumed Planner actions. All artifacts remain; continuation
is held by user request. The approved Qwen KL diagnostic uses a separate shared
`new_SCII/deployments/qwen-kl-diagnostics-20260906/code-<SHA>`, reuses the pinned
47 `/dev/shm` dependencies read-only, and uses its own `/dev/shm/qkl_l6_trace_r5`
Ray directory and `qwen_kl_l6_trace_r5` screen. It is one update, not long training.

The next step-shaping ablation stages its committed code separately at
`/root/new-scii-r6-shaping-ab-20260905` on A100-44. Initially this is CPU
verification only. GPU approval and private source synchronization were obtained
before the subsequent four-GPU launch. The successful natural-termination deployment above
is not edited or reused as a writable worktree. On September 5 at 11:32 UTC,
GPUs 0-2 were occupied by other tasks; the user approved a four-GPU A/B on
GPUs 4-7 instead. The four-GPU revision uses a new `code-4gpu` deployment under
the same staging root, preserving the earlier eight-GPU `code` snapshot.
At 11:38 UTC it launched source `c5c4037` in tmux `r6_shaping4_ab_0905`; the
serial control log is `pair4.log`. Run IDs and launch evidence are recorded in
`experiments/records/20260905-r6-natural-step-shaping-ab-v1/launch-20260905-four-gpu.json`.

The terminal-policy-credit candidate is isolated under
`/root/new-scii-r6-terminal-policy-20260905` on A100-44. The `code-7fdab02`
snapshot passed CPU tests without GPUs. A new commit-specific `code-<sha>`
snapshot carries the approved thin launcher; the earlier snapshots remain
unchanged. The user approved GPUs 4-7, eight Level-5 rollouts and one update,
with legacy shaping retained. Its experiment record is
`20260905-r6-terminal-policy-credit-v1`; outputs use that experiment's unique
`runs` directory below the shared new_SCII root. No shared environment or model
files are modified.

The subsequent format-reward candidate uses a separate node-44 staging root,
`/root/new-scii-r6-format-reward-20260905`, with commit-specific deployments.
Only GPUs 4-7 are approved for its one-update gate after CPU verification and
completion of the terminal-policy control. The dependency overlay and model
assets above are reused read-only. Its output experiment is
`20260905-r6-format-reward-v1` below the shared new_SCII root.

## 2026-09-08 g2 handoff r2 on node46

After renewed user approval and an eight-GPU idle check, both four-GPU arms use
node46: A GPUs0–3, B GPUs4–7. New deployment
`/root/new-scii-g2-planner-handoff-20260908/code-a34e539` reuses
`/root/new-scii-spb-unitid-20260906/python` (Torch2.7.1+cu128, Transformers4.55.2,
Ray2.48.0, NumPy1.26.4) and system Python3.12 headers. No dependency/model copy.
The `g2_handoff_pair_r2` tmux and `pair-r2` logs share a 60-minute deadline;
new Ray paths `/dev/shm/g2handoff_A_r2` and `/dev/shm/g2handoff_B_r2` are isolated.
See the g2 handoff record for actual startup and results; no PPO is authorized.

The user-approved r3 retry uses separate `code-ea0055c`, `pair-r3`, tmux
`g2_handoff_pair_r3` and Ray paths `/dev/shm/g2handoff_A_r3`,
`/dev/shm/g2handoff_B_r3`. It retains the original September9 00:56:34 Beijing
deadline and the same dependency/model inputs. The r2 deployment is not edited.
At September9 09:39 Beijing this deadline had passed and all eight node46 GPUs
belonged to other jobs. The r3 paths/session are therefore planned only: no
deployment or experiment was started under the expired window.
The subsequent explicit authorization moves r3 to node44 for a fresh shared
50-minute window (A GPUs0–3, B GPUs4–7, four games each, no PPO). Node44 uses its
existing `/root/new-scii-spb2-level5-control/python`; new `code-ea0055c` and
`pair-r3` paths remain separate from old runs. See `launch_r3_node44.sh` in the
handoff record; the original expired node46 plan remains in commit4a903b3.

## 2026-09-08 node42 retired smoke checkpoint cleanup

User-authorized removal of two retired August20 SpB smoke model/optimizer
snapshots freed 118.378 GiB of `/dev/shm`; logs/configs/results and all other
checkpoints remain. Those two snapshot directories are no longer resumable.
See [exact deletion manifest and before/after state](checkpoint-cleanup-20260908.md).
No experiments or other users' processes were started/stopped.

## 2026-09-09 May-inspired opening A/B on node47

The requested pure-evaluation comparison uses a new root
`/root/new-scii-g2-planner-handoff-20260909`, commit-specific `code-<SHA>` and
`pair-may-r1`. Existing `/dev/shm/spb-unitid-20260906/python` and its
`include/python3.12:include` headers are reused without installation. At 10:49
Beijing all eight GPUs were empty; this must be rechecked at launch. Both arms
share a 50-minute deadline, four GPUs/four games each, no PPO. Outputs stay below
the existing shared new_SCII handoff experiment, with unique `may-opening` RUNs.
See [configuration, provenance and actual status](../../experiments/records/20260908-g2-planner-handoff-v1/may-opening-ab.md).
# 2026-09-09 开局 Planner RL 独立部署

44使用 `/root/new-scii-g2-planner-handoff-20260909`：`code-e3ffac9`为只读B组runtime，
`orchestration-<SHA>`为本次八卡step8入口。复用现有
`/root/new-scii-spb2-level5-control/python`，不改旧环境或旧47 A/B部署。
源码Git bundle单独备份于此目录，模型和每步完整checkpoint在共享盘。
具体资源、RUN_ID和验收边界见
[开局Planner RL记录](../../experiments/records/20260908-g2-planner-handoff-v1/opening-rl8.md)。

## 2026-09-09 BO segment protocol A/B on node44

User-approved pure evaluation uses new `/root/new-scii-bo-segment-20260909`,
independent `code-0d7ba09` and `code-a1f15af` snapshots, a full Git bundle backup
and `pair-r1` orchestration logs. It reuses node44's existing
`/root/new-scii-spb2-level5-control/python` and system Python3.12 headers; models
remain on shared storage. A uses GPUs0–3, B GPUs4–7, four games each, one shared
50-minute deadline. No PPO or automatic retry. Ray directories are
`/dev/shm/boseg_A_r1` and `/dev/shm/boseg_B_r1`. Runs are under the shared new_SCII
`20260909-bo-segment-protocol-v1` experiment, not any existing deployment/run.
See [configuration and actual status](../../experiments/records/20260909-bo-segment-protocol-v1/README.md).
# Level7 断点恢复候选（2026-09-10）

44的原`/root/new-scii-spb-fsdp-20260910/code-cfdd3b4`和`l7-rl4-r1`不修改。
续训使用同父目录下独立`code-<新SHA>`、控制目录`l7-resume96-r1`及tmux
`sc2_spb_l7_resume96_r1`，Ray为`/dev/shm/sc2_spb_l7_resume96_0910_r1`。
模型与依赖复用Level7原路径，不下载、不改环境。新RUN目录在
`/mnt/data0601/yangjinyue/new_SCII/experiments/20260910-spb-fsdp-layer-wrap-v1/runs/`
下的`20260910-spb-fsdp-layer-wrap-v1-spb2-l7-g2-opening-quota-resume4to96-4gpu-s0-r1`。
授权是44 GPU4–7、最长48小时、每8步完整保存、不自动重试；必须先核验前序完整
退出和GPU归属，不把该记录当作当前空闲证明。43任务不变。

## 2026-09-13 node43 phase-two air-options A/B

This approved four-GPU serial evaluation uses an independent `code-<SHA>` and
`phase2-air-pair-r1` under `/root/new-scii-execution-feedback-20260911`.
The existing A4 checkpoint, local asset mirror
`/root/new-scii-execution-feedback-local-20260911/shared`, Python overlay and
headers under `/root/new-scii-spb-level5-96-20260908` are reused, not modified.
New runs live in the mirror's `yangjinyue/new_SCII/experiments/20260913-phase2-air-options-v1/runs`;
Ray paths are `/dev/shm/p2_A_0913_r1` and `/dev/shm/p2_B_0913_r1`.
GPUs0–3, four games per arm, one shared45-minute limit, no PPO/retry. See the
[candidate record](../../experiments/records/20260913-phase2-air-options-v1/README.md).
## 2026-09-16 combat-reward preparation and relocation

The user approved waiting for sbh's SpB2 ALFWorld seed0 to release node197,
without interrupting it. The new isolated root is
`/data/yangjinyue/new-scii-combat-reward-20260916`; do not use or modify the August
SC2 deployments. The separately approved A4-only transfer writes
`staging/a4-r1/global_step_4` under that root, with both source and destination
SHA256 manifests retained. This is about60GiB of model/Adam state, not a new
original-model download or permission to copy additional large dependencies.
See [readiness and authorized budget](../../experiments/records/20260916-combat-result-reward-v1/README.md).
Preparation/transfer and a follow-up monitor do not mean training is running.
The user subsequently approved stopping the slow transfer, retaining all partial
artifacts, and moving both jobs to node47 GPUs4–7 serially, replacing the node197
budgets: a30-minute/48-answer probe followed by a3-hour RL A/B pair. Source/control
use `/dev/shm/sc2-combat-reward-20260916-r1`; a private Git bundle uses shared
`new_SCII/staging/20260916-combat-reward-r1`. A4 and dependencies reuse the existing
September15 scene-B-r2 paths read-only. Results use the new experiment's shared
`runs` directory, not old deployments. No other processes or large assets change.
