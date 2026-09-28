# Qwen3-4B Level 5 r6

This is the authoritative record for the completed 96-update release run. It tested whether the official Qwen3-4B checkpoint, with no project SFT or prior project RL, could learn under the canonical Planner prompt and the frozen System-v1 Executor on Level 5.

The run completed all 96 GiGPO updates and saved a complete distributed checkpoint. It produced 625 successes in 768 training rollouts, for 81.38% success and mean shaped reward 9.90. The last 16 updates were stronger than the first 16 on both measurements, but the evidence is limited to one training seed and has no independent held-out evaluation. The result supports using this configuration as the Qwen system baseline; it does not isolate model capability from the scripted Executor and does not establish statistical superiority.

`manifest.json` contains the exact machine-readable method, provenance, and result. The public launcher is `scripts/train_qwen_level5.sh`; immutable small inputs are under `data/`, and external asset hashes are enforced by `scripts/verify_assets.py`.
