# Qwen3-4B-Instruct-2507 Level 5

This experiment substitutes the official non-thinking
`Qwen/Qwen3-4B-Instruct-2507` package for the released `Qwen/Qwen3-4B`
package. The Planner, action contract, System-v1 Executor, reward, Level 5
environment, GiGPO/PPO settings, rollout seeds and matched decoding parameters
remain unchanged.

Before either run, verify the downloaded package and its rendered generation
prefix:

```bash
python scripts/verify_assets.py --model /path/to/model \
  --model-manifest configs/qwen3-4b-instruct-2507.sha256 \
  --sc2 /path/to/StarCraftII
python scripts/check_qwen_instruct2507.py --model /path/to/model
```

The one-update smoke must finish with eight rollouts, finite training metrics,
no thinking marker and a complete checkpoint. Only then may the fresh 96-update
run start from the same official checkpoint. The primary run intentionally keeps
the released sampling parameters (`temperature=0.4`, `top_p=0.9`, `top_k=0`),
so the model package is the only method change. The model's recommended native
sampling (`0.7/0.8/20`) belongs to a separate system experiment.

```bash
RUN_ID=20260828-qwen3-4b-instruct-2507-level5-v1-smoke5-s0-r5 \
TOTAL_EPOCHS=1 SAVE_FREQ=1 \
MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507 \
SC2PATH=/path/to/StarCraftII \
OUTPUT_DIR=/path/to/run \
bash scripts/train_qwen_instruct2507_level5.sh
```

After the smoke gate passes, use a new output directory and set
`TOTAL_EPOCHS=96 SAVE_FREQ=96` for run
`20260828-qwen3-4b-instruct-2507-level5-v1-full96-s0-r6`.

The first smoke identifier is retained as an infrastructure failure: node 42
rejected GPU container creation before loading the model because its Docker
daemon had no NVIDIA runtime. The replacement runs use node 197. The runtime
starts from the exact historical r6 base image and applies the package versions
measured from the released image; the frozen recipe is under `runtime/` and its
resulting image digest is recorded in `manifest.json`.

The second smoke also exited before loading the model because its outer Docker
launcher did not override the inherited bash entrypoint. It is retained as an
infrastructure failure; smoke3 uses an explicit entrypoint with otherwise
identical inputs.

Smoke3 then exposed that Hydra still wrote its own metadata below the source
tree. The shared launcher now routes that metadata to `OUTPUT_DIR/hydra`, so
the deployment remains read-only; this is an operational fix shared by both
Qwen variants and does not alter the RL method.

Smoke4 reached model and SC2 worker initialization, then exposed the same issue
for per-environment observation logs. `SC2_RUNTIME_LOG_ROOT` is now explicitly
set to `OUTPUT_DIR/sc2-logs` and propagated to Ray workers before smoke5.

Smoke5 completed one update and passed the promotion gate: 8 terminal rollouts,
7 wins, mean episode reward 16.375, finite training metrics, no thinking marker,
and a complete eight-rank `global_step_1` checkpoint. Its pre-clip gradient norm
was 197.281 and 37.5% of responses reached the matched 96-token limit; these are
monitoring signals for the full run, not reasons to change the control protocol.
The exact result and artifact hashes are recorded in `smoke5-audit.json`.

The fresh 96-update run started on all eight GPUs of node 197 at
2026-08-28 08:37:54 UTC. It starts from the official checkpoint rather than
the smoke checkpoint; its run identifier and launch hashes are in
`manifest.json`.

Runs write to node-local ext4 while active and are archived to the experiment
root on `/data` only after completion. This avoids the stalled large-file
JuiceFS writes observed during setup without changing model or training logic.
