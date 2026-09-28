# Gamma A/B startup correction — September11

This repairs the new experiment launcher, **not** a claimed cause of historical
RL convergence. User approved diagnosing and fixing startup and retrying the
unchanged A/B with new run IDs. No prompt, model layer, precision, loss, seed
sequence, checkpoint, rollout count, learning rate, reward or game change.

## Failed r1

Frozen source `f3803883bb4727b2f372c82b0d05b0f2ff81ea70`, node44 GPU0–7.
The queue acquired eight idle GPUs at September10 23:06:03 Beijing. A failed at
23:07:37, exit1, before model/optimizer step4 restoration and before any game or
PPO update. `gamma_pairing.json` does not exist. B never started. Both `.done`
and `result.json` confirm failure. Console SHA256:
`6256da045cd7c4300a436f1546fb1d87905cf0fa635b948bbb7d9cea05a2e42e`.

The failing FSDP initialization reports:
`Duplicate GPU detected : rank 7 and rank 0 both on CUDA device 10000`.
Old runs and their deployments remain unchanged.

## Evidence and single correction

Old `gamma_ab.install()` immediately imports `RayPPOTrainer` inside Ray's
worker-process setup hook. On the frozen runtime, a CUDA-hidden CPU probe
intercepted `torch._C._cuda_getDeviceCount` before it could touch the driver.
The captured import chain was `gamma_ab.install → verl/protocol.py:37 →
verl/utils/device.py:28 → torch.cuda.is_available():174 → _cuda_getDeviceCount`.
This is a real early probe; `torch.cuda.is_initialized()==False` by itself does
not rule out this driver availability probe.

Corresponding public implementations:

- [Ray2.48.0 default_worker](https://github.com/ray-project/ray/blob/ray-2.48.0/python/ray/_private/workers/default_worker.py)
  executes setup hooks before `worker.main_loop()`.
- [Ray2.48.0 task execution](https://github.com/ray-project/ray/blob/ray-2.48.0/python/ray/_raylet.pyx)
  sets accelerator visibility in task execution, after process startup.
- [PyTorch2.7.1 is_available](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/cuda/__init__.py#L159-L174)
  explicitly documents that the default count call initializes the CUDA driver.

The minimal correction defers trainer import and `_load_checkpoint` wrapping
until `TaskRunner.run` calls `make_envs`. GPU worker setup no longer imports the
trainer or probes CUDA. This reuses the already-used environment callback boundary;
it does not force device IDs, disable cache, change NCCL settings or alter losses.
The existing pairing function and exhausted-epoch handling are unchanged.

## Acceptance

CPU regression must intercept both the driver count call and lazy initialization:
installing the new hook must make neither call and must not import the trainer.
Existing gamma-only config and checkpoint/epoch-pairing tests must still pass.

Then run `probe_gamma_devices.py` once on eight verified-idle GPUs, at most150s.
It uses the same setup hook, eight one-GPU Ray actors and one scalar NCCL sum.
Require eight distinct CUDA UUIDs and sum28 on all eight ranks. No model, game
or optimizer is loaded. This is bounded startup acceptance, not an RL experiment.
Do not retry the device probe or a failed RL run automatically.

If accepted, start r2 A then B from an independent clean frozen directory, at most
eight hours total, including startup acceptance. Each arm still adds exactly8
global steps (5–12). Confirm actual step4 load, eight recorded game seeds and SC2
progress before reporting training has started. CPU success alone is insufficient.

## September11 acceptance record

Frozen candidate `5ba3f2111dc5bd576d1e67c7da5fdeaffba9f24f` is clean and privately
backed up. Pinned node44 CPU tests: `CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:tests:/root/new-scii-spb2-level5-control/python
timeout 45s python3 -m unittest test_gamma_ab -v`: **4 passed,3.037s**.
Local config test passed;3 Torch-dependent tests skipped because Torch is unavailable.
Bash syntax and `git diff --check` passed.

At08:50:29 Beijing the approved r2 controller started on44 GPU0–7. At08:50:43,
the actual8-rank probe passed:8 distinct device UUIDs and all-reduce28 on every
rank. A then passed the formerly failing FSDP initialization and reached the
real parent step4 model/optimizer loader. Reading large optimizer shards has
been slow but progressing: rank0 read_bytes rose from4,137,144,320 at08:55:31
to6,070,718,464 at08:57:36. No extra large-checkpoint read was started.
This observation is not a new model/optimizer bug or an excuse to alter settings.
By09:01 the real model/optimizer loader had returned, `gamma_pairing.json`
confirmed global_step4/fresh reset0, actual first seeds were2732 in all8 games,
and all8 SC2 workers had called Planner. A is sampling its first new batch
(global_step5); this snapshot does not yet establish a completed PPO update.

Latest actual-start evidence, run IDs, controller/bundle hashes and the inclusive
eight-hour deadline are in [r2 launch record](gamma-ab-retry-launch.json).
The original r1 and node43 long-run results remain untouched.
