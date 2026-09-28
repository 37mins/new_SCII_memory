# Provenance

## Current integration

[Integrated Fixes v1](integrated-fixes-v1.md) incorporates the tested repair chain
through `8cb4f18` and changes defaults to shaping-off / SpB SWA-zero. It is not
byte-identical to historical Canonical. The current source manifest and new run
record define this version; the sections below preserve original provenance.

## Historical Canonical integration

The shared Qwen/SpB launcher was integrated from main `d42096e`. It does not
change the twelve frozen system/trainer files in `configs/system-v1.sha256`.
The hashes below describe the historical r6 runtime; the checked-in freeze
manifest describes its already-cleaned release source. These are distinct
provenance objects and are not claimed byte-identical.

SpB uses the same registered generic snapshot as r7, content identifier
`de52e8ef993afc763f9d4fb498e8d8302ef0bc13bd7dfd53c6ae5f77f43698b5`.
The operational-file hashes come from the old 30-file checkpoint manifest at
`codex/experiment-level6-system-ab-v1`. Model card and download bookkeeping
are excluded, not weight, tokenizer or Python model files. The preparation
script reproduces `spb2_training_chunk_mode.patch` from
`experiments/records/20260815-spb2-rl-smoke-v1`: four training-mode selections
in each of `sse.py` and `sse_swa.py`. It saves the resulting file hashes
per run and never patches the original snapshot.

SpB upstream code is BICLab/SpikingBrain2.0 commit
`02561132682c2a66925bb0df9f6a1d3f9e816723` (Apache-2.0);
its FLA and MoBA subprojects declare MIT licenses. These kernels are installed
as optional dependencies rather than copied into this Git tree.
The shared launcher is new packaging, not a newly completed 96-step run.

## Completed Qwen r6

The release was reconstructed from the immutable runtime of completed run `20260818-qwen-canonical-plan-rl-v1-qwen-base-l5-full96-s0-r6`, not from the later development worktree. The run exited with code 0 on 2026-08-19 and produced a complete step-96 FSDP checkpoint: eight model shards, eight optimizer shards, eight extra-state shards, and `data.pt`.

The historical source commit was `e61a23b8c642cb1b88d67f11f1f6d9a96a65ef31`; the clean orchestration repository was at `cbf7424b1ec764d2b059fbd2ffe410f98469f9c7`. The final patched runtime had SHA-256 tree digests `72843f4ffbe1f8f430e9c23124fad47811d359ef1b75e893e84980e070ceba7e` for `verl` and `667c779e2f5299c0c3eca0d60503a9c65643a549308ed1976651e84a56d34818` for `gigpo`. The table records the historical active-file hashes. The release preserves their training behavior while removing inert backups, unrelated task environments, patch staging, generated output, demo credentials, and unused Planner-side client construction; it also repairs a parse-time error in an unused critic branch. The released contract is covered by the tests in `tests/`.

| Artifact | SHA-256 |
|---|---|
| Qwen snapshot manifest used by the historical run | `82fdeae910670892e5c70b177ca5ec30d3d69090416e34480be6d63880f7a56c` |
| Planner | `25cd70413db0d5846a5823093545f793092bc850d9eac1091a7d7d018f1bf3e5` |
| Strategic SC2 manager | `82f59d27f4ea4c5623e39beabc6d48597b33979cddee0b4bbf56cf145c60c345` |
| System-v1 ScriptPlayer | `76354ff908b47f8c400e2cf279a3af5f49e98a3da6afb53018143b579f28c6da` |
| Build order | `304b1f4ff12fe30c5e6d5f7b8eaef8ed6823ca65afed3a3706cae10e3639c494` |
| Training data | `7c8f6b07eacef5e06963b475c3c36e90196da870fae095b12c2ea0f7faf10912` |
| Validation data | `4dbe5e3ff68df3fbf76e91284b28b06c52bf7495193013a4bf4ed7ca417a0cd7` |
| SC2 `Base75689/SC2_x64` | `115f65a49191c504f6336f7d5b7818b23dbba3d7a45cacf9142c86c959abaa41` |
| `Flat128.SC2Map` | `3916dbd2bb662a6e9c9bd27089906630c8b0a42fdbab51fc4401cf1df7b9b212` |

The model is the official Qwen post-trained checkpoint at Hugging Face revision `1cfa9a7208912126459214e8b04321603b3df60c`. “No SFT/no RL” means that this project did not apply SC2 SFT or an earlier RL checkpoint before the run; it does not mean the Qwen checkpoint is a foundation-only pretrained model.

The historical run used Python 3.12.11, PyTorch 2.7.1+cu128, Transformers 4.55.2, and Ray 2.48.0 inside image `sha256:3a773ade07540703574ec33410e31f46799814daa65a23ade9510589546c54f9`. That private image is replaced here by the pinned official PyTorch CUDA 12.8 image plus `requirements.txt`; the Qwen path was verified not to import SpikingBrain FLA or MoBA kernels. StarCraft II remains an external licensed dependency and is never redistributed by this repository.
