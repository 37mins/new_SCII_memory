# Opt-in original-package game-seed candidate

Parent runtime: `e13e1fbdf84171172d7eeace826fca599c41ea63`. Default behavior is unchanged.
Set `+ray_init.runtime_env.env_vars.SC2_ORIGINAL_GAME_SEEDS='1'` only for the approved
node46 C comparison: original SpB, Level6/g2, four steps of eight games, one final
complete checkpoint, at most two hours and 60GiB of new outputs, after existing A/B.
No automatic retry, continuation or change to node197.

Source: `SC2_online.zip` SHA256
`e40fb7dad98f4c7d336031f3b2047ee3e15113ce14925830cfb6540aaa5888ab`,
`envSC2.py:131–139`: each game draws `random.randint(0,100)`, inclusive, with replacement.
This candidate restores that draw distribution, not the missing historical trainer.
For auditability, a separate `random.Random(base_seed)` and the saved reset index
determine the draw sequence. It neither consumes model RNG nor guarantees unique seeds.
The existing seed log records the actual values sent to SC2 workers.

Keep data.train_batch_size=1 and env.rollout.n=8, including the existing GiGPO UID
grouping. These eight games therefore compare returns across game seeds; environment
difficulty can contribute to their differences. Nonzero advantages or a smoother win
curve do not establish improved learning. Changing to eight singleton groups would
also change the baseline and zero the episode-relative advantages; that is not this test.

No reward, loss, model, temperature, token budget, prompt, BO, micro, cache, precision,
optimizer or validation-seed change. No update to the historical runtime/deployment.
CPU reset-index replay is not full model/optimizer restore acceptance.

The separately requested C continuation reuses the same seed mechanism. Its only
runtime addition is a matching read-only `resume_state.json` seed preview, previously
hard-coded to grouped seeds. Actual sampling, checkpoint state and parameter updates
are unchanged. A direct checkpoint restore at reset_index=4 must report and actually
use [60,71,12,45,55,40,78,81] next, then match the full uninterrupted 96-batch stream.
The C short deployment stays frozen at 56cfc60. No legacy-state reconstruction is used
for C; only its complete model/Adam/RNG/dataloader/environment checkpoint is eligible.
Resource authorization and dispatch are recorded separately, not implied by this code.

CPU tests: `PYTHONPATH=tests:scripts python3 -B -m unittest test_original_game_seeds test_sc2_checkpoint_resume -v`.
GPU/SC2 and learning acceptance are pending and recorded by the separate orchestration.
