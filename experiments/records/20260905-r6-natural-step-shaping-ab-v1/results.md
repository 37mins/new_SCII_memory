# Completed A/B, not a learning improvement claim

Verified 2026-09-05 12:32 UTC on A100-44. Both arms used clean `c5c4037`, GPUs
4-7, eight Level-5 rollouts and one fresh Instruct-2507 PPO update. Run IDs are
`20260905-r6-natural-step-shaping-ab-v1-qwen-instruct2507-l5-legacy-4gpu-rl1-s0-r1`
and the matching `off` ID. Raw artifacts remain below
`/mnt/data0601/yangjinyue/new_SCII/experiments/20260905-r6-natural-step-shaping-ab-v1/runs`.

| Metric | A: legacy | B: off |
|---|---:|---:|
| Completion / exit | 1/1, 0 | 1/1, 0 |
| Natural outcomes / truncations | 8 / 0 | 8 / 0 |
| Wins | 7/8 | 6/8 |
| Win-rate Wilson 95% interval | 52.9–97.8% | 40.9–92.9% |
| Episode reward mean | 13.312 | 3.750 |
| Environment interactions mean / max | 2.875 / 8 | 4.875 / 24 |
| Reported action validity | 20.8% | 35.0% |
| Fully valid BO-context replies | 0/18 | 8/33 |
| Non-evaluable terminal-only replies marked valid | 5 | 6 |
| Response budget clipping | 66.7% | 45.0% |
| Pre-clip gradient norm | 755.513 | 451.836 |
| actor/ppo_kl | -0.130 | 0.170 |
| actor/kl_loss | 0.261 | 1.152 |
| PPO clip fraction | 8.2% | 3.3% |
| Checkpoint save seconds | 738.438 | 730.361 |
| End UTC | 12:00:25.776720 | 12:29:39.774415 |

Both checkpoint audits passed; both `.done` files report completion. Reward
means are not directly comparable because B deletes one reward term. Clipping
and gradient changes are observations, not proof of improved training. These
wins were collected before each arm's only update, so they cannot demonstrate
learning from that update. The samples are small and game/generation trajectories
differed despite matching seed settings.

B's console contains 33 `Step shaping reward off: 0.000` lines and no legacy
`Step shaping reward:` lines. The state-bonus removal was applied; this does
not mean all other step rewards were disabled. A final 12:35 UTC check confirmed
B's `.done` as completed and all eight node-44 GPUs idle at that instant.

The major shared limitation is confirmed by prompt logs and frozen collection
code: A has five and B six no-BO-context terminal replies included as policy
samples. A's real BO-context replies fail on `depot` (3) and `tank` (15).
B has eight fully valid replies; its failures are `depot` (3), `sieget` (1),
`tank` (16), `x 23` (1), malformed canonical bullets (3), and excess bullets (1).
Full-answer rejection does not prove no valid prefix was executed.

Process completion is valid; interpreting these as clean policy-learning or
architecture results is not. The independent terminal-credit candidate does
not edit or replace either run and keeps A's legacy reward as its next control.

| Artifact SHA-256 | A | B |
|---|---|---|
| result.json | `5777cccd1f014e4ef6f0f8d10f64c63a8f16ba67775f7ad535af26d675f9f021` | `85c23371704927038922a78a6e5d37c1a4d7877c61f745a3da38d8bd4718e829` |
| metrics.json | `9052f8996e199842578efa2295633b450d7b77377b5ace6157f6421e3e1b4609` | `d72763214dc0507cb97e859c7dae38ac6ccd6cd58fd87b08131abe9f111169e6` |
| prompt_response.jsonl | `ac355146a8ebed0b0bad14a73ade6cb61e15c46b42bb29882980591e9d3f0d87` | `dfa0fa02aa4be6f10c1b2e7fc5493e09afd844ffd44eceabff7be784dd8b6313` |
| episode_outcomes.jsonl | `050f63a3d7bb175de8f5f8e33f7732cd643736ad80e05de1d860d283a436efe3` | `79a1c90eba6cabb5fc6b101b93f1eb2b4c0d4c2d315f9b328031d39dc28841ec` |
| console.log | `edd7c8be1e522c9b6af4e637daef45281c50f288dcfc006baa254a59e4befd32` | `cacbc24d5345865ab1a6d0f0ced3c7dd3632ee6b5c5936754f6d604dbf7e9729` |
