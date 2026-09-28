# r6 terminal policy credit

Status: r2 completed at 2026-09-05 13:39:55 UTC, exit zero, with one finite
logged PPO update and complete checkpoint. Its consumption gate passed: 14
real decisions, four terminal-only rows excluded, eight natural games (6 wins).
See [r2 completion evidence](r2-completion.json). This verifies the statistics
and attribution path, not Planner quality or learning improvement.

R1 failed at 2026-09-05 13:11:31 UTC in post-update metrics; see
[r1 results and r2 repair](results.md). The user approved r2 with only the
episode-length scalar type corrected; format reward remains unchanged.

R2 started at 13:18:59 UTC on the same four GPUs from clean/pushed `7618097`,
after 45 Linux CPU tests and identical non-path config comparison. See
[r2 launch evidence](launch-r2-20260905.json).

The original live validation started at 2026-09-05 12:49:26 UTC on node 44 GPUs 4-7,
from clean/pushed `8942447`, after explicit approval. Configuration normalization
against A found no non-path differences. See [launch evidence](launch-20260905.json).
R1 alone was not a complete live repair success. The original source change is
`7fdab0204f35f8451a225612628992d91e8c58d7`; later commits add only records and
the thin launch wrapper. No additional training behavior is introduced.
User authorized parallel repair on 2026-09-05. The A/B runtime and all prior
results remain unchanged. This training-semantic change is isolated in
`codex/fix-r6-terminal-policy-credit`, based on clean `8da48c4`; its parent
runtime was `c5c4037`. Rollback uses that parent and its recorded configuration.

A's run
`20260905-r6-natural-step-shaping-ab-v1-qwen-instruct2507-l5-legacy-4gpu-rl1-s0-r1`
on A100-44 completed one update with 7/8 natural wins. Its prompt log contains
23 active environment rows: five no-BO-context terminal replies, all marked
valid, and eighteen BO-context replies, none fully valid (three `depot`, fifteen
`tank` failures). The five terminal-only games include four wins and one loss.
Therefore 20.8% trainer validity is not a measurement of valid Planner plans,
and the all-game win rate is not Planner learning evidence. These are diagnostic
counts from the old log, not new consumption-ID measurements.

The falsifiable repair target is zero unconsumed terminal samples entering PPO,
with unchanged game outcomes/reward amounts and retained real final actions.
The implementation and public basis are in
[policy credit](../../../docs/reference/policy-credit.md). Only action/reward
attribution changes; legacy shaping, Qwen3-4B-Instruct-2507 revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, g2, Level 5 Hard/Air, Flat128,
seed 0, six-action/96-token generation and the original micro remain frozen.
No SpB, SFT, learning-rate, dropout or cache experiment is bundled here.

The regression first ran the actual old `gather_rollout_data()` with one
zero-decision win and one genuine action followed by a separate terminal poll:
old collation produced three samples instead of one. After the fix it keeps
one action, preserves its full reward and still reports both games' win rate.
Tests also cover initial terminal reset/ACK, consumed final actions, ignored
ACTIONs during terminal ACK, invalid real actions, cached terminal replies and
an all-zero-policy batch.

Local commands are `python3 -B -m unittest discover -s tests -p 'test_r6*.py' -v`
and the corresponding `test_system_release.py` suite. The local Python lacks
Torch, NumPy and Gymnasium: AST tests execute the real selected methods with
small fake boundary objects, not GPU kernels or a full game. Full pinned Linux
CPU tests and a future live gate must be reported separately.

The isolated Linux copy is
`/root/new-scii-r6-terminal-policy-20260905/code-7fdab02` on A100-44, with
`CUDA_VISIBLE_DEVICES=""` and read-only dependency overlay
`/root/new-scii-spb2-level5-control/python`. All 44 tests passed with
`/usr/bin/python3 -B -m unittest discover -s tests -v`; the deployment remained
clean. CPU log SHA-256:
`f42741331881b451f5e383242c1c694bb6dca46659094c4c2c81dcad0bcb7c7d`.
The source bundle is backed up locally and on node 44 as
`new-scii-terminal-policy-7fdab02.bundle`, SHA-256
`e072a14e9537573cc09e33fd5deade2d4399ab174d54977dfb186ac9e7b4b95c`.
The GitHub repository's private status and admin access were verified before
pushing the isolated branch; no weights, credentials or live run directories
were pushed.

`verify_tensor_path.py` also passed against that same source and pinned Linux
environment with CUDA disabled. It uses the actual TrajectoryCollector,
DataProto, GiGPO discounted returns and EpisodeRewardManager: four environment
rows become two consumed-action samples, rewards `[20.5, -50]` are retained,
and the all-three-game success fraction remains `2/3`. This is a synthetic CPU
data-path test, not a model parameter update or game experiment. Script SHA-256
is `3d2a8393d2c43e6f47e756b918fe0c6c2b8e38f30ac97086ba7f7395b56a5d77`;
`/root/new-scii-r6-terminal-policy-20260905/tensor-path-7fdab02.log` SHA-256 is
`7fc5123e0c8b10d1cdf3a72bdb5c812e17071c79bbd7f9e4d1b454007bf06ba4`.

To repeat the CPU data-path check from a pinned Linux environment:

```bash
CUDA_VISIBLE_DEVICES="" python3 -B experiments/records/20260905-r6-terminal-policy-credit-v1/verify_tensor_path.py
```

Local verification passed 23 r6 tests and 17 system-release tests, including
the updated source fingerprint manifest; `git diff --check` passed. The
fingerprint changes cover only five source files for this attribution boundary.
The baseline's reward manager, GiGPO/PPO, model code, launcher controls, BO,
Executor and micro hashes remain identical.

The rollout audit prints each of the eight original episode totals and both
interaction/decision counts. PPO's reward/length means exclude zero-decision
games after the filter, whereas success rate still includes all games. A change
in those means is not itself an improvement.

The authorized live gate compares A's legacy configuration with
this candidate, using eight rollouts and one update on the same four GPUs, fresh
initial weights, a new RUN_ID and a clean backed-up commit. All eight games
must have natural outcomes; policy sample counts must match consumption IDs;
PPO must be finite and its checkpoint complete. If none of the games invokes
Planner, report that and stop before PPO. Do not increase the rollout budget
to obtain more policy samples without permission. A one-update gate cannot
establish a learning curve or win-rate gain, and A's only failure had no Planner
decision, so filtering correctly may leave only winning policy trajectories.

The approved run ID is
`20260905-r6-terminal-policy-credit-v1-qwen-instruct2507-l5-legacy-4gpu-rl1-s0-r1`.
`run_node44.sh` selects GPUs 4-7 and a fresh run directory under this experiment
on the shared new_SCII root. It invokes the unchanged A/B arm launcher in
`legacy` mode: same original model, assets, optimizer, gamma, sampling, natural
termination and one-update budget. Expected duration is 25-45 minutes including
about twelve minutes for checkpoint I/O; the inherited infrastructure timeout
is two hours. There is no automatic retry or longer continuation.

Before launch, compare the dry-run effective configuration against completed A
after normalizing only deployment/run/Ray paths and the experiment name. Stop if
any other configuration differs. Completion additionally requires matching the
new consumption flags and per-game policy-credit audit with eight natural
outcomes; numerical completion alone does not certify action attribution.
