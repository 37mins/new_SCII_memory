# v1524 micro on the rising canonical Level 5 baseline

This experiment tests the v1524 MuleBot-inspired micro path on the exact
canonical Level 5 system whose original 96-update run showed a rising success
curve. The paired baseline is
`20260818-qwen-canonical-plan-rl-v1-qwen-base-l5-full96-s0-r6`.

The only behavioral change is `SC2_ENABLE_NEWEXEC_MICRO=0 -> 1`. Planner,
prompt and legacy compact6 protocol, System-v1 `ScriptPlayer`, build order,
reward, Qwen3-4B checkpoint, seed, Flat128 `Hard` Air opponent, rollout budget,
generation budget, and GIGPO optimization settings remain fixed. The
`ScriptPlayer` stays byte-identical at SHA-256
`76354ff908b47f8c400e2cf279a3af5f49e98a3da6afb53018143b579f28c6da`;
the experiment only enables its existing v1524 path.

The run uses eight GPUs, eight rollouts per update, and 96 updates. It starts
from the same official Qwen3-4B snapshot with no project SFT and no prior
project RL. The primary comparison is the success and reward trajectory versus
the frozen r6 baseline; action validity, response clipping, numerical health,
and non-empty micro traces are secondary diagnostics.

## Runs

`20260830-v1524-micro-level5-rl-v1-qwen-base-l5-full96-s0-r1` started at
2026-08-30 03:44 CST on `sc2-a100-42`, GPUs 0-7. It runs from clean Git commit
`19b1202ce26289b63e19c69aac11ef6b158e64c5`; the immutable deployment archive
SHA-256 is
`0f4bb651c9566268868e1329efa2ef098054e98f5621e3bfb7cae3c9ce1ad3b8`.
The run root is
`runs/20260830-v1524-micro-level5-rl-v1/20260830-v1524-micro-level5-rl-v1-qwen-base-l5-full96-s0-r1`
on the shared filesystem. Status: running.
