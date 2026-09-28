# r6 full-response format reward

Status: r1 completed September 5, 14:14:35 UTC (22:14 Beijing), exit 0,
one finite logged update and complete checkpoint. See [results](results.md),
[completion evidence](r1-completion.json) and [launch evidence](launch-20260905.json).
The user subsequently requested the next prepared short test: r2 changes only
prompt logging, as defined in [the interface investigation](planner-interface-preparation.md).

The live control is now `76180970d2e139ef2da052df659546621a84539d`, run
`20260905-r6-terminal-policy-credit-v1-qwen-instruct2507-l5-legacy-4gpu-rl1-s0-r2`.
Its scalar-type correction is merged into the existing format branch so the
only treatment relative to r2 is format reward. No current deployment is edited.
The former r1 control failed in post-update metrics and is not a numerical pass.
This branch originally started from record-only `263051a`. A verified
incremental backup is `/private/tmp/new-scii-pre-format-reward-263051a.bundle`,
SHA-256 `f7aa3cd6b4f5bcd7eb215699aa7e3c385f633725697695f732999d5f608bc98e`.
It requires base `c5c4037978a8a97ef788d38f905772736ccb1900`.

The only treatment is replacing the unconditional terminal format-bonus verdict
with the existing full-answer validator verdict. The
[interface and public basis](../../../docs/reference/plan-format-reward.md)
define its scope. The primary gate is zero positive format bonuses on invalid
last consumed answers, while valid last answers retain +0.5. A deterministic
paired CPU check must preserve model query and parsed executable plan for
identical inputs. Invalid real actions remain PPO samples and unconsumed
terminal replies remain excluded. One update cannot establish learning or
statistically meaningful win-rate improvement.

Live budget: node `sc2-a100-44`, GPUs 4-7, Qwen/Qwen3-4B-Instruct-2507 revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, original checkpoint (no project SFT
or RL), Level 5 Hard/Air, Flat128, canonical g2, seed 0, eight rollouts and one
GiGPO/PPO update. Model and SC2 assets, dependency overlay and all controls are
inherited from the control's [launch record](../20260905-r6-terminal-policy-credit-v1/launch-r2-20260905.json).
Legacy state shaping, uncapped natural games, six BO / 96 tokens, HF/eager,
no-thinking, learning rate 1e-6 and clip 1.0 are frozen. The old batch-copy
alignment remains; this is not a claim of perfect PPO sample weighting.

The new run ID is
`20260905-r6-format-reward-v1-qwen-instruct2507-l5-legacy-4gpu-rl1-s0-r1`.
`run_node44.sh` wraps the unchanged legacy arm with unique paths. Expect 25-45
minutes including checkpoint I/O; timeout remains two hours. Do not launch
until the control finishes and GPUs are free. Before launch, compare resolved
configs, allowing only code/run/Ray paths and experiment name to differ. Save
the clean pushed SHA, asset manifests and actual config with the run.

Completion requires eight natural outcomes, one finite PPO update, exit zero,
complete step-1 checkpoint and per-game bonus/consumption audit. Report all-game
win rate separately from policy-participating episode reward. Preserve failures
and all raw logs. A lower reward caused solely by removing false format bonuses
is expected, not evidence of weaker gameplay.

## Conditional continuation

The user requested preparing work in parallel and starting this single short
gate as soon as r2 is healthy. Thread heartbeat `sc2-r2` previously used a
two-minute interval; it is now PAUSED at the user's request on September 5.
This does not stop the running server-side experiment. It inspected the fixed r2
directory without starting a separate task or modifying r2. `check_control.py` verifies
completion, nonempty finite metrics, eight natural games and consumption/reward
conservation before allowing the next launch. A pending check does not consume
the authorization. A failed control blocks the launch and must be reported.

Before launch additionally verify the candidate SHA/clean state, matching
non-path config, empty target directory and free GPUs 4-7. Use the original
model snapshot, never r2's trained checkpoint. Launch at most once in tmux;
an existing target run or tmux session prevents a duplicate. Record evidence
and stop automatic continuation after the format gate finishes or fails.
No further fixes, reruns, larger budgets, other seeds or SpB runs are queued.

The prepared runtime is `46ce2ccde96b153e008731ca78d95168699dd0d4`, clean and
pushed, at `/root/new-scii-r6-format-reward-20260905/code-46ce2cc`. All 53 Linux
CPU tests passed, including the metric-type regression and conditional gate.
The paired 23-response fixtures still produce identical queries and executable
BO. Non-path configuration differences against r2 are `{}`, also verified in
the actual candidate manifest/config at 13:46:40 UTC. R2 completed with 6/8
natural wins, one finite logged update and a complete checkpoint; its 14 real
answers and four excluded terminal rows passed the consumption audit. This
does not resolve its zero full-contract validity or 87.5% response-length cap
rate. Exact paths, budgets and hashes are in [the handoff](handoff-ready.json).
Automatic monitoring remains disabled. The r1-only handoff has ended. A separate
user request authorizes the next prepared short test; it does not authorize any
automatic follow-on run or long training.

## Offline evidence

Source `2fa14ca2a531975080bcc9bf7199d136a8ea529c` is clean and pushed to the
private repository. The only production changes are 18 inserted / 3 removed
lines across manager, socket environment and Planner; validator, prompt,
parser, BO, micro and optimizer are unchanged. Node-44 deployment is
`/root/new-scii-r6-format-reward-20260905/code-2fa14ca`.

All 50 pinned Linux CPU tests passed. `verify_offline.py` compared old/new
Planner on identical fixtures with all 23 recorded real responses: every query
and executable BO matched. This is a deterministic fixture comparison, not
replaying the original games. Logged prompts are truncated, so historical
validity is taken from the full-input runtime verdict, not revalidated against
truncated text. The control had 1/23 fully valid responses. Its four real
terminal rewards replay as 20.5 → 20 for three invalid answers and 20.5 → 20.5
for the valid answer, total correction -1.5; no new wins are claimed.

The remote CPU log `cpu-tests-2fa14ca.log` SHA-256 is
`2640cbfd27005be693f44607023b640196c3e7874de2e1599383cee4919d755a`.
The adjacent `offline-2fa14ca.json` SHA-256 is
`090ae2f523d89d864a3bf5119c2307bdee1eaac66183c473b2ca2fdf67800d89`.
Source bundle `/root/new-scii-format-reward-20260905.bundle` SHA-256 is
`7f500ee155d993145f46bec4703f76ce85271feeb09cc6c62788d530e1c36d46`.
