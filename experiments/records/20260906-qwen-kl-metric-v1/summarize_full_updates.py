"""Read-only full-batch update analysis; no model load or acceptance threshold."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import torch


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines(keepends=True) if line.endswith('\n')]


def summarize(root):
    updates, counts, gradients = defaultdict(list), defaultdict(dict), defaultdict(list)
    config_rows, inputs, events = [], [], []
    for rank in range(4):
        rows = read_rows(root / f'rank-{rank:02d}.jsonl')
        events += rows
        config_rows += [r for r in rows if r['event']=='config']
        inputs += [r for r in rows if r['event']=='input']
        for r in rows:
            if r['event']=='optimizer_update':
                gradients[r['optimizer_updates']].append(r['global_grad_norm'])
        saved_path = root / 'ppo_diagnostics' / f'update-0001-rank-{rank:02d}.pt'
        if not saved_path.exists():
            continue
        saved = torch.load(saved_path, map_location='cpu', weights_only=True)
        for row in read_rows(root / f'post-each-rank-{rank:02d}.jsonl'):
            index, step = row['sample_offset'], row['optimizer_updates_before']
            mask = saved['attention_mask'][index, -saved['responses'].shape[1]:].bool()
            new = torch.tensor(row['log_prob'][0], dtype=torch.float64)[mask]
            old = saved['old_log_probs'][index, mask].double()
            a = saved['advantages'][index, mask].double()
            labels = saved['responses'][index, mask]
            positions = torch.nonzero(mask).flatten()
            delta = new-old
            updates[step].append(dict(rank=rank, sample=index, delta=delta, advantage=a,
                labels=labels, positions=positions, old=old, new=new))
            counts[step][rank] = counts[step].get(rank, 0)+1
    summaries=[]
    if not config_rows:
        return dict(run_id=root.name, optimizer_updates=[])
    actor_config=config_rows[0]['config']['actor']
    low=actor_config.get('clip_ratio_low')
    high=actor_config.get('clip_ratio_high')
    low=actor_config['clip_ratio'] if low is None else low
    high=actor_config['clip_ratio'] if high is None else high
    for step, rows in sorted(updates.items()):
        d=torch.cat([r['delta'] for r in rows]); a=torch.cat([r['advantage'] for r in rows])
        finite=bool(d.isfinite().all())
        ratio=d.clamp(-20,20).exp()
        below, above=ratio<1-low,ratio>1+high
        worst=max(rows, key=lambda r:float(r['delta'].abs().max()))
        j=int(worst['delta'].abs().argmax())
        summaries.append(dict(optimizer_updates=step, rank_sample_counts=counts[step],
            complete_all_ranks=counts[step]=={0:69,1:69,2:69,3:69}, tokens=len(d), finite=finite,
            abs_mean=float(d.abs().mean()),
            abs_quantiles={name:float(value) for name,value in zip(['p50','p90','p99','max'],torch.quantile(d.abs(),torch.tensor([.5,.9,.99,1.],dtype=torch.float64)))},
            ppo_k1_token_mean=float(-d.mean()), ratio_below_clip=float(below.double().mean()),
            ratio_above_clip=float(above.double().mean()),
            objective_clipfrac=float(((below&(a<0))|(above&(a>0))).double().mean()),
            exponent_clamped_tokens=int((d.abs()>20).sum()), global_grad_norm_by_rank=gradients[step],
            worst_token=dict(rank=worst['rank'],sample=worst['sample'],response_index=int(worst['positions'][j]),
                token_id=int(worst['labels'][j]),old=float(worst['old'][j]),new=float(worst['new'][j]),advantage=float(worst['advantage'][j]))))
    return dict(run_id=root.name, result=json.loads((root/'result.json').read_text()) if (root/'result.json').exists() else None,
        inputs=inputs, config_flags=[{k:r.get(k) for k in ['rank','param_dtype','fp32_normalization','model_fp32','rowwise_rms_norm','recompute_start','max_updates']} for r in config_rows],
        train_eval_gaps=[r['max_train_eval_gap_all_ranks'] for r in events if r['event']=='rebased_start'],
        optimizer_updates=summaries, token_gradient_rows=sum(r['event']=='token_loss_gradient' for r in events),
        token_gradients_finite=all(r['finite'] for r in events if r['event']=='token_loss_gradient'),
        note='Post-update statistics compare the SAME complete batch after each optimizer step; not training rollout or independent evaluation.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    print(json.dumps(summarize(parser.parse_args().run),indent=2))
