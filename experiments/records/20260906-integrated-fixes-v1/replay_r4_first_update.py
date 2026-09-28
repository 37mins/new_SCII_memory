"""Observe one r4 optimizer update from step1; no game or policy modification."""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch


def block_indices(size, width=4096, blocks=16):
    width = min(width, size)
    starts = torch.linspace(0, size - width, blocks, dtype=torch.float64).long()
    return torch.unique(torch.cat([torch.arange(int(s), int(s) + width) for s in starts]))


def capture_blocks(parameter, optimizer, indices):
    state = optimizer.state[parameter]
    take = lambda x: x.detach().flatten()[indices.to(x.device)].cpu().clone()
    return dict(parameter=take(parameter), exp_avg=take(state['exp_avg']),
                exp_avg_sq=take(state['exp_avg_sq']), step=int(state['step'].item()))


def observe_one_step(actor, output, rank):
    """Hooks return None and only copy detached values. Keep the real step method."""
    optimizer = actor.actor_optimizer
    parameters = [p for g in optimizer.param_groups for p in g['params']]
    if len(parameters) != 1:
        raise ValueError('Expected the recorded single FSDP flat shard per rank')
    parameter = parameters[0]
    indices = block_indices(parameter.numel())
    evidence = dict(indices=indices, shard_numel=parameter.numel(),
        before=capture_blocks(parameter, optimizer, indices),
        group={k: v for k, v in optimizer.param_groups[0].items() if k != 'params'},
        parameter_names=list(getattr(parameter, '_fqns', [])),
        parameter_shapes=[list(s) for s in getattr(parameter, '_shapes', [])],
        shard_parameter_info=[list(s) for s in getattr(parameter, '_shard_param_infos', [])])
    original_step = actor._optimizer_step
    original_forward = actor._forward_micro_batch
    calls = []

    def save_gradient(name, tensor):
        evidence[name] = tensor.detach().cpu().clone()
        # Returning None preserves the incoming gradient.

    def forward(*args, **kwargs):
        entropy, logp = original_forward(*args, **kwargs)
        if torch.is_grad_enabled() and logp.requires_grad:
            logp.register_hook(lambda g: save_gradient('loss_to_logprob', g))
            if entropy is not None and entropy.requires_grad:
                entropy.register_hook(lambda g: save_gradient('loss_to_entropy', g))
        return entropy, logp

    def pre_step(unused_optimizer, unused_args, unused_kwargs):
        evidence['gradient_after_global_clip'] = parameter.grad.detach().flatten()[indices.to(parameter.device)].cpu().clone()
        evidence['local_gradient_norm_after_clip'] = float(torch.linalg.vector_norm(parameter.grad.detach()))

    def post_step(unused_optimizer, unused_args, unused_kwargs):
        calls.append(1)
        evidence['after'] = capture_blocks(parameter, optimizer, indices)
        evidence['parameter_delta'] = evidence['after']['parameter'] - evidence['before']['parameter']

    def step():
        evidence['gradient_after_fsdp_before_clip'] = parameter.grad.detach().flatten()[indices.to(parameter.device)].cpu().clone()
        evidence['local_gradient_norm_before_clip'] = float(torch.linalg.vector_norm(parameter.grad.detach()))
        norm = original_step()
        evidence['global_gradient_norm_before_clip'] = float(norm)
        return norm

    handles = [optimizer.register_step_pre_hook(pre_step), optimizer.register_step_post_hook(post_step)]
    actor._optimizer_step, actor._forward_micro_batch = step, forward

    def finish():
        actor._optimizer_step, actor._forward_micro_batch = original_step, original_forward
        for handle in handles:
            handle.remove()
        evidence['observed_optimizer_steps'] = len(calls)
        with (output / f'parameter-chain-rank-{rank:02d}.pt').open('xb') as stream:
            torch.save(evidence, stream)
        if len(calls) != 1 or evidence['after']['step'] != evidence['before']['step'] + 1:
            raise RuntimeError('Replay did not perform exactly one optimizer update')
        return evidence

    return finish


def load_config(run, source, output):
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict
    overrides = json.loads((run / 'effective_config.json').read_text())['overrides']
    with initialize_config_dir(config_dir=str(source / 'verl/trainer/config'), version_base=None):
        config = compose(config_name='ppo_trainer', overrides=[f'{k}={v}' for k, v in overrides.items()])
    # This assignment is normally made by ray_trainer, not an LR schedule change.
    with open_dict(config.actor_rollout_ref.actor.optim):
        config.actor_rollout_ref.actor.optim.total_training_steps = config.trainer.total_training_steps
    with open_dict(config.actor_rollout_ref.actor):
        config.actor_rollout_ref.actor.ppo_diagnostics_dir = str(output / 'first-mini-trace')
    return config


def probability_summary(values, old, masks):
    difference = (values.double() - old.double())[masks]
    row = dict(tokens=int(masks.sum()), finite=bool(torch.isfinite(difference).all()))
    if row['finite']:
        row.update(k1_token_mean=float(-difference.mean()),
            k1_response_mean=float(torch.where(masks, old.double() - values.double(), 0.).sum(1).div(masks.sum(1)).mean()),
            abs_mean=float(difference.abs().mean()), abs_max=float(difference.abs().max()),
            delta_quantiles=torch.quantile(difference, torch.tensor([0., .01, .05, .5, .95, .99, 1.], dtype=torch.float64)).tolist(),
            drops_over20=int((difference < -20).sum()))
    return row


def main():
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from verl import DataProto
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.workers.actor.ppo_diagnostics import record_full_batch
    from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, load_fsdp_optimizer

    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('run', 'source', 'output', 'checkpoint'):
        parser.add_argument('--' + key, required=True, type=Path)
    args = parser.parse_args()
    rank = int(os.environ['RANK'])
    if int(os.environ['WORLD_SIZE']) != 8:
        raise ValueError('Historical r4 shards require exactly eight ranks')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.set_num_threads(1)
    stream = (args.output / f'rank-{rank:02d}.jsonl').open('x')

    def emit(event, **values):
        stream.write(json.dumps(dict(event=event, rank=rank,
            time=datetime.now(timezone.utc).isoformat(), **values)) + '\n')
        stream.flush()

    config = load_config(args.run, args.source, args.output)
    emit('configuration', config=OmegaConf.to_container(config.actor_rollout_ref, resolve=True),
         cuda_bf16_reduced_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
         tf32=torch.backends.cuda.matmul.allow_tf32)
    path = args.run / 'ppo_diagnostics' / f'update-0002-rank-{rank:02d}.pt'
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['responses'].shape != (23, 96):
        raise ValueError('Expected 23 recorded responses per rank')
    emit('input', sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
         samples=23, selected_offsets=[0], checkpoint=str(args.checkpoint))
    worker = ActorRolloutRefWorker(config.actor_rollout_ref, role='actor_rollout')
    worker.init_model()
    worker.load_checkpoint(str(args.checkpoint), del_local_after_load=False)
    actor, optimizer = worker.actor, worker.actor_optimizer
    if worker._is_offload_param:
        load_fsdp_model_to_gpu(worker.actor_module_fsdp)
    if worker._is_offload_optimizer:
        load_fsdp_optimizer(optimizer, device_id=torch.cuda.current_device())
    if (actor.config.ppo_mini_batch_size, actor.config.ppo_micro_batch_size_per_gpu,
            actor.config.ppo_epochs, actor.config.use_dynamic_bsz) != (1, 1, 1, False):
        raise ValueError('Replay requires the recorded local mini1/micro1/epoch1/static configuration')
    state_steps = [int(s['step'].item()) for s in optimizer.state.values()]
    if state_steps != [22]:
        raise ValueError(f'Expected r4 step1 Adam counter22, received {state_steps}')
    emit('restored', optimizer_steps=state_steps, scheduler=worker.actor_lr_scheduler.state_dict(),
         parameter_dtypes=sorted({str(p.dtype) for p in worker.actor_module_fsdp.parameters()}),
         attention_dropout=worker.actor_model_config.attention_dropout,
         rng_note='Restored checkpoint RNG; historical post-rollout RNG was not saved. No shuffle; Qwen dropout=0.')
    temperature = float(config.actor_rollout_ref.rollout.temperature)
    full = DataProto.from_dict(tensors=saved, meta_info={'temperature': temperature}).to(torch.cuda.current_device())
    actor.actor_module.train()
    trace = args.output / f'full-batch-rank-{rank:02d}.jsonl'
    with trace.open('x'):
        pass
    record_full_batch(trace, actor, full.batch, temperature, 'pre_update', 0)
    pre_rows = [json.loads(line) for line in trace.read_text().splitlines()]
    before = torch.tensor([r['log_prob'][0] for r in pre_rows])
    mask = saved['attention_mask'][:, -96:].bool()
    restored = probability_summary(before, saved['old_log_probs'], mask)
    emit('full_batch_before', samples_per_rank=23, pre_vs_old=restored)
    # Historical r4 pre-update actor-old gap was zero. A mismatch pauses this
    # replay before any update; it is not evidence of a training implementation bug.
    matches_old = torch.tensor(int(restored['finite'] and restored['abs_max'] == 0.), device='cuda')
    dist.all_reduce(matches_old, op=dist.ReduceOp.MIN)
    if not matches_old.item():
        raise RuntimeError('Restored effective-token probabilities differ from historical old; '
                           'no optimizer update performed. Inspect the saved pre-update evidence.')
    # No old/ref rebasing: use the saved historical probabilities and advantages.
    finish = observe_one_step(actor, args.output, rank)
    prefix = DataProto.from_dict(tensors={k: v[:1] for k, v in saved.items()},
                                meta_info={'temperature': temperature}).to(torch.cuda.current_device())
    with worker.ulysses_sharding_manager:
        prefix = worker.ulysses_sharding_manager.preprocess_data(data=prefix)
        metrics = actor.update_policy(prefix)
    evidence = finish()
    emit('optimizer_update_completed', optimizer_updates=1, adam_step=evidence['after']['step'],
         metrics=metrics, global_grad_norm=evidence['global_gradient_norm_before_clip'])
    # Do not advance the global-step scheduler after a partial batch.
    record_full_batch(trace, actor, full.batch, temperature, 'post_update', 1)
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    after = torch.tensor([r['log_prob'][0] for r in rows if r['phase'] == 'post_update'])
    local = dict(before=before, after=after, old=saved['old_log_probs'], mask=mask,
        grad_before=evidence['local_gradient_norm_before_clip'], grad_after=evidence['local_gradient_norm_after_clip'])
    gathered = [None] * 8
    dist.all_gather_object(gathered, local)
    if rank == 0:
        merged = {k: torch.cat([r[k] for r in gathered]) for k in ('before', 'after', 'old', 'mask')}
        result = dict(samples=184, optimizer_updates=1,
            pre_vs_old=probability_summary(merged['before'], merged['old'], merged['mask']),
            post_vs_old=probability_summary(merged['after'], merged['old'], merged['mask']),
            post_vs_pre=probability_summary(merged['after'], merged['before'], merged['mask']),
            gradient_norm_from_shards_before=sum(r['grad_before'] ** 2 for r in gathered) ** .5,
            gradient_norm_from_shards_after=sum(r['grad_after'] ** 2 for r in gathered) ** .5)
        (args.output / 'full-batch-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    emit('completed', optimizer_updates=1, samples_checked=23, checkpoint_saved=False,
         limitation='Parameter-chain tensors cover 16 deterministic blocks per flat shard, not all model parameters.')
    stream.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
