"""Observe existing backward/optimizer calls; no extra gradient, loss, or update."""
import torch
from verl.workers.actor.ppo_diagnostics import record_full_batch


def install_update_observers(actor, batch, temperature, emit, post_path):
    forward, step = actor._forward_micro_batch, actor._optimizer_step
    updates, sample_cursor = 0, 0

    def observed_forward(micro_batch, temperature, calculate_entropy=False):
        nonlocal sample_cursor
        entropy, log_prob = forward(micro_batch, temperature, calculate_entropy)
        if torch.is_grad_enabled() and log_prob.requires_grad:
            offset = sample_cursor % len(batch)
            sample_cursor += len(log_prob)
            mask = micro_batch['attention_mask'][:, -log_prob.shape[-1]:].bool().cpu()
            labels = micro_batch['responses'].detach().cpu()

            def record_gradient(gradient):
                active = gradient.detach().double().cpu()[mask]
                emit('token_loss_gradient', optimizer_updates_before=updates,
                     sample_offset=offset, tokens=active.numel(), finite=bool(active.isfinite().all()),
                     token_ids=labels[mask].tolist(), gradient=active.tolist(),
                     note='d(loss)/d(selected log_prob): PPO+reference KL; not the separate entropy branch')
                # Returning None preserves the gradient used by autograd.
            log_prob.register_hook(record_gradient)
        return entropy, log_prob

    def observed_step():
        nonlocal updates
        shards = []
        for name, parameter in actor.actor_module.named_parameters():
            if parameter.grad is not None:
                grad = parameter.grad.detach()
                if not grad.numel():
                    continue
                norm = float(grad.norm())
                maximum = float(grad.abs().max())
                shards.append(dict(parameter=name, elements=grad.numel(), l2=norm, abs_max=maximum))
        emit('parameter_gradient_shards', optimizer_updates_before=updates, shards=shards)
        norm = step()
        updates += 1
        emit('optimizer_update', optimizer_updates=updates, global_grad_norm=float(norm))
        # Same full batch after EVERY update. The existing observer protects RNG
        # and runs under no_grad; this does not add optimizer steps.
        record_full_batch(post_path, actor, batch, temperature, 'post_optimizer_update', updates)
        return norm

    actor._forward_micro_batch = observed_forward
    actor._optimizer_step = observed_step
