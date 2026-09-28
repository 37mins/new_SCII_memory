# Setup

The runtime target is Linux with Python 3.12, PyTorch 2.7.1/CUDA 12.8 and eight
A100 GPUs. No model weights, SC2 binaries, checkpoints or credentials belong in
Git. Use the same container paths at launch and throughout a run.

## Qwen runtime and assets

```bash
docker build -t sc2-system-v1 .
mkdir -p assets
docker run --rm -v "$PWD:/workspace" sc2-system-v1 \
  hf download Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --local-dir /workspace/assets/Qwen3-4B-Instruct-2507
```

SC2 uses client 4.10 (`Base75689`) and `Maps/Melee/Flat128.SC2Map`.
Follow the [Blizzard distribution instructions](https://github.com/Blizzard/s2client-proto#downloads);
extracting their password-protected archive accepts their AI/ML license.

```bash
curl -L https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip -o assets/SC2.4.10.zip
unzip -P iagreetotheeula assets/SC2.4.10.zip -d assets
curl -L https://blzdistsc2-a.akamaihd.net/MapPacks/Melee.zip -o assets/Melee.zip
mkdir -p assets/StarCraftII/Maps
unzip -P iagreetotheeula assets/Melee.zip -d assets/StarCraftII/Maps
```

Reuse an existing matching snapshot when available; the project servers already
have this model and need no new download. For an authorized short run after
checking that the whole node is available:

```bash
docker run --rm --gpus all --network=host --ipc=host --shm-size=128g \
  --ulimit memlock=-1 -v "$PWD:/workspace" \
  -e MODEL_PATH=/workspace/assets/Qwen3-4B-Instruct-2507 \
  -e SC2PATH=/workspace/assets/StarCraftII \
  -e RUN_ID=qwen-integrated-l6-s0 \
  -e SC2_DIFFICULTY=Harder -e TOTAL_EPOCHS=4 -e SAVE_FREQ=1 \
  sc2-system-v1 bash scripts/train_qwen_instruct2507_level5.sh
```

The host Git directory must be available inside `/workspace`; the launcher
requires a clean commit. Keep the shell attached or use an owned tmux session.
Select unused, unique output and Ray paths. Do not stop somebody else's Ray
cluster to make a node appear free.

## SpikingBrain

Use the registered [Panyuqi/SpikingBrain-2.0-instruct](https://modelscope.cn/models/Panyuqi/SpikingBrain-2.0-instruct)
snapshot. Download it outside Git to `assets/SpikingBrain-2.0-instruct` using
ModelScope's file download facility, or copy the registered original snapshot.
A mutable download is only acceptable if **every operational file matches**
`configs/spikingbrain2.sha256`; a mismatch stops launch rather than selecting
another weight silently. Do not supply compact6 SFT or an RL checkpoint.

The 26-file manifest covers all model code/config, tokenizer files and eleven
original PyTorch weight shards. The original 30-file manifest additionally
contained ModelScope bookkeeping, the model card and its service metadata;
these four files are not runtime inputs. No safetensors conversion is required.

SpB also needs FlashAttention 2.6.3 and the FLA/MoBA code from
`BICLab/SpikingBrain2.0@02561132682c2a66925bb0df9f6a1d3f9e816723`.
The optional build recipe is:

```bash
docker build -f Dockerfile.spb -t sc2-system-v1-spb .
```

This recipe is pinned to the historical dependencies but has **not yet been
rebuilt and GPU-validated as part of this integration**. An existing verified
r7-compatible runtime is the other supported preparation path. Do not treat an
offline launcher test as kernel validation.

Run the same Docker command as Qwen with image `sc2-system-v1-spb`, model path
`/workspace/assets/SpikingBrain-2.0-instruct`, a new run ID and script
`scripts/train_spb_level5.sh`. The launcher creates `runs/<id>/model`, copies
small model files, and links weight shards to the verified original snapshot.
It applies only the historical eight-line training-mode patch to two copied
files. The source weights/code stay unchanged and must remain mounted.
The default launcher additionally passes `swa_dropout=0.0` to actor/reference
configuration, retaining SWA on the HF path without disabling model cache.

## Optional controls

`SC2_ENV_SEED` selects the environment seed; both models default to 0.
`BO_PROFILE=canonical` is the default frozen two-base opening.
`BO_PROFILE=rush` uses the exact historical one-base/three-Barracks BO without
automatically choosing seed 2. For a paired Rush experiment, pass the same
profile and seed to both model scripts. The integrated optional profile does
not include the historical diagnostic production/combat monkey-patches and is
therefore not advertised as a byte-identical Rush replay.

`TOTAL_EPOCHS=1 SAVE_FREQ=1` is the short integration budget, not a 96-step
learning result. Changing the six-action/96-token protocol is out of scope.
The resolved overrides are recorded; custom overrides define a separate run,
not a replacement for a published result.

## Offline verification

Use the pinned Linux training environment for the complete CPU regression,
including real Torch update tests. Run without GPUs:

```bash
CUDA_VISIBLE_DEVICES="" python -B -m unittest discover -s tests -v
```

`DRY_RUN=1 VERIFY_ASSETS=0` emits the exact command and marks the manifest
non-formal. It does not import the model, create Ray/SC2 processes, or validate
GPU dependencies. Real training always verifies model, game and frozen source
hashes. A failed verification or launch leaves a failure record; reuse a new
run ID after fixing it.

A lightweight environment without Torch/Transformers runs only part of the suite;
skipped numerical tests do not count as full integration verification.
