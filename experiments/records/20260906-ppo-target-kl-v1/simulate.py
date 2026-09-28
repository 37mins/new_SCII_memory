"""Read r32 probabilities on CPU; simulate decisions, never load/update a model."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import torch


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines(keepends=True) if s.endswith('\n')]


def token_stats(rows):
    delta = torch.cat([r[2] for r in rows])
    k3 = torch.expm1(delta) - delta
    return dict(samples=len(rows), tokens=len(delta), kl_sum=float(k3.sum()),
                kl_mean=float(k3.mean()), abs_mean=float(delta.abs().mean()),
                p99_abs=float(torch.quantile(delta.abs(), .99)), max_abs=float(delta.abs().max()))


def simulate(root):
    manifest = json.loads((root/'manifest.json').read_text())
    world_size = manifest['world_size']
    groups = {phase: defaultdict(list) for phase in ('pre_optimizer', 'full_post_optimizer')}
    rank_sizes = {}
    for rank in range(world_size):
        saved_path = root/'ppo_diagnostics'/f'update-0001-rank-{rank:02d}.pt'
        saved = torch.load(saved_path, map_location='cpu', weights_only=True)
        rank_sizes[rank] = len(saved['responses'])
        sources = [('pre_optimizer', [r for r in read_rows(saved_path.with_suffix('.jsonl')) if r['phase']=='update']),
                   ('full_post_optimizer', read_rows(root/f'post-each-rank-{rank:02d}.jsonl'))]
        for phase, rows in sources:
            for row in rows:
                i = row['sample_offset']
                mask = saved['attention_mask'][i, -saved['responses'].shape[-1]:].bool()
                new = torch.tensor(row['log_prob'][0], dtype=torch.float64)[mask]
                old = saved['old_log_probs'][i, mask].double()
                groups[phase][row['optimizer_updates_before']].append((rank, i, new-old))
    out = dict(run_id=root.name, source_commit=manifest['source_commit'], formula='expm1(new_logp-old_logp)-(new_logp-old_logp)',
               mask='saved response attention mask, including EOS; no reference probabilities or exponent/loss clipping',
               weighting='global valid-token sum / count; historical 2 copied rows retained, not a clean online weighting test')
    for phase, steps in groups.items():
        out[phase] = []
        for n, rows in sorted(steps.items()):
            by_rank = {rank: token_stats([r for r in rows if r[0]==rank]) for rank in range(world_size)}
            complete = phase=='pre_optimizer' or all(by_rank[r]['samples']==rank_sizes[r] for r in range(world_size))
            out[phase].append(dict(updates_completed=n, complete=complete, **token_stats(rows), rank=by_rank))
    out['threshold_simulations'] = []
    for target in (.001, .005, .01, .02, .05):
        item = dict(target_kl=target, stop_threshold=1.5*target)
        for phase in groups:
            hit = next((r for r in out[phase] if r['complete'] and r['kl_mean']>1.5*target), None)
            item[phase] = None if hit is None else {k:hit[k] for k in ('updates_completed','kl_mean','p99_abs','max_abs')}
        out['threshold_simulations'].append(item)
    return out


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    print(json.dumps(simulate(parser.parse_args().run), indent=2))
