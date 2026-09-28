"""CPU analysis of the one-update evidence, without loading model checkpoints."""
import argparse
import json
from pathlib import Path

import torch


def difference(actual, expected):
    d = actual.double() - expected.double()
    return dict(finite=bool(d.isfinite().all()), abs_max=float(d.abs().max()),
                abs_mean=float(d.abs().mean()), equal_elements=int((d == 0).sum()), elements=d.numel())


def audit_adam(evidence):
    """Use the installed PyTorch AdamW on CPU slices, not a replacement training step."""
    before, after, group = evidence['before'], evidence['after'], evidence['group']
    parameter = torch.nn.Parameter(before['parameter'].clone())
    keys = ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize', 'foreach', 'capturable', 'differentiable', 'fused')
    optimizer = torch.optim.AdamW([parameter], **{k: group[k] for k in keys})
    optimizer.state[parameter] = dict(step=torch.tensor(float(before['step'])),
        exp_avg=before['exp_avg'].clone(), exp_avg_sq=before['exp_avg_sq'].clone())
    parameter.grad = evidence['gradient_after_global_clip'].clone()
    optimizer.step()
    state = optimizer.state[parameter]
    expected_delta = parameter.detach() - before['parameter']
    result = dict(step_before=before['step'], step_after=after['step'],
        observed_optimizer_steps=evidence['observed_optimizer_steps'],
        parameter=difference(after['parameter'], parameter.detach()),
        parameter_delta=difference(evidence['parameter_delta'], expected_delta),
        exp_avg=difference(after['exp_avg'], state['exp_avg']),
        exp_avg_sq=difference(after['exp_avg_sq'], state['exp_avg_sq']),
        actual_delta_abs_max=float(evidence['parameter_delta'].abs().max()),
        changed_parameters=int((evidence['parameter_delta'] != 0).sum()),
        global_gradient_norm_before_clip=evidence['global_gradient_norm_before_clip'],
        local_gradient_norm_before_clip=evidence['local_gradient_norm_before_clip'],
        local_gradient_norm_after_clip=evidence['local_gradient_norm_after_clip'],
        loss_to_logprob=evidence.get('loss_to_logprob', torch.empty(0)).tolist(),
        note='CPU slice reference; CUDA foreach arithmetic may round differently; no arbitrary pass threshold')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--historical-run', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if torch.__version__ != '2.7.1+cu128':
        raise RuntimeError('Use the recorded PyTorch 2.7.1 runtime for this CPU comparison')
    report = dict(torch=torch.__version__, ranks=[], worst_tokens=[])
    for rank in range(8):
        evidence = torch.load(args.output / f'parameter-chain-rank-{rank:02d}.pt', weights_only=True, map_location='cpu')
        report['ranks'].append(dict(rank=rank, **audit_adam(evidence)))
        saved = torch.load(args.historical_run / 'ppo_diagnostics' / f'update-0002-rank-{rank:02d}.pt', weights_only=True, map_location='cpu')
        rows = [json.loads(s) for s in (args.output / f'full-batch-rank-{rank:02d}.jsonl').read_text().splitlines()]
        pre = torch.tensor([r['log_prob'][0] for r in rows if r['phase'] == 'pre_update'], dtype=torch.float64)
        post = torch.tensor([r['log_prob'][0] for r in rows if r['phase'] == 'post_update'], dtype=torch.float64)
        mask = saved['attention_mask'][:, -96:].bool()
        delta = post - pre
        active = mask.nonzero()
        for index in delta[mask].abs().topk(min(3, len(active))).indices:
            sample, token = active[index].tolist()
            report['worst_tokens'].append(dict(rank=rank, local_sample=sample, token_index=token,
                used_in_first_update=sample == 0, token_id=int(saved['responses'][sample, token]),
                old=float(saved['old_log_probs'][sample, token]), pre=float(pre[sample, token]),
                post=float(post[sample, token]), delta=float(delta[sample, token]),
                ref=float(saved['ref_log_prob'][sample, token]), advantage=float(saved['advantages'][sample, token]),
                input_ids=saved['input_ids'][sample][saved['attention_mask'][sample].bool()].tolist(),
                response_ids=saved['responses'][sample][mask[sample]].tolist()))
    report['worst_tokens'].sort(key=lambda x: -abs(x['delta']))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
