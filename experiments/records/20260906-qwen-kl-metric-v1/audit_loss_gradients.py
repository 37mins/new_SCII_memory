"""Check recorded d(loss)/d(log_prob) against the frozen PPO/KL code on CPU."""
import argparse
import json
from pathlib import Path
import torch
from verl.trainer.ppo.core_algos import compute_policy_loss, kl_penalty, agg_loss
from summarize_full_updates import read_rows


def audit(root):
    output=[]
    world_size=json.loads((root/'manifest.json').read_text())['world_size']
    for rank in range(world_size):
        events=read_rows(root/f'rank-{rank:02d}.jsonl')
        if not events:
            continue
        cfg=events[0]['config']['actor']
        saved_path=root/'ppo_diagnostics'/f'update-0001-rank-{rank:02d}.pt'
        if not saved_path.exists():
            continue
        saved=torch.load(saved_path,map_location='cpu',weights_only=True)
        gradients={(r['optimizer_updates_before'],r['sample_offset']):r for r in events if r['event']=='token_loss_gradient'}
        for row in read_rows(saved_path.with_suffix('.jsonl')):
            if row['phase']!='update':
                continue
            i=row['sample_offset']; key=(row['optimizer_updates_before'],i)
            if key not in gradients:
                continue
            mask=saved['attention_mask'][i:i+1,-saved['responses'].shape[-1]:].bool()
            old=saved['old_log_probs'][i:i+1]; ref=saved['ref_log_prob'][i:i+1]
            adv=saved['advantages'][i:i+1]
            lp=torch.tensor(row['log_prob'],dtype=torch.float32,requires_grad=True)
            pg,clipfrac,ppo_k1,clipfrac_lower=compute_policy_loss(old,lp,adv,mask,cliprange=cfg['clip_ratio'],
                cliprange_low=cfg['clip_ratio_low'],cliprange_high=cfg['clip_ratio_high'],
                clip_ratio_c=cfg['clip_ratio_c'],loss_agg_mode=cfg['loss_agg_mode'])
            kld=kl_penalty(torch.where(mask,lp,0.),torch.where(mask,ref,0.),cfg['kl_loss_type'])
            kl=agg_loss(kld,mask,cfg['loss_agg_mode'])
            # config is emitted BEFORE worker.init_model normalizes global mini
            # size to the rank-local size. Match that recorded worker boundary.
            accumulation=cfg['ppo_mini_batch_size']//world_size//cfg['ppo_micro_batch_size_per_gpu']
            gp=torch.autograd.grad(pg/accumulation,lp,retain_graph=True)[0][mask]
            gk=torch.autograd.grad(kl*cfg['kl_loss_coef']/accumulation,lp)[0][mask]
            observed=torch.tensor(gradients[key]['gradient'],dtype=torch.float32)
            delta=(lp.detach()-old)[mask]; j=int(delta.abs().argmax())
            output.append(dict(rank=rank,sample=i,optimizer_updates_before=key[0],tokens=int(mask.sum()),
                pg_loss=float(pg.detach()),ppo_k1=float(ppo_k1.detach()),
                reference_capped_k3=float(kl.detach()),clipfrac=float(clipfrac.detach()),
                clipfrac_lower=float(clipfrac_lower.detach()),
                derivative_abs_error_max=float((gp+gk-observed).abs().max()),
                token_gradient_abs_max=float(observed.abs().max()),max_logprob_gap=float(delta.abs().max()),
                kl_saturated_tokens=int((kld.detach()[mask]>=10.).sum()),
                worst=dict(token_id=int(saved['responses'][i:i+1][mask][j]),
                    old=float(old[mask][j]),new=float(lp.detach()[mask][j]),advantage=float(adv[mask][j]),
                    ppo_gradient=float(gp[j]),kl_gradient=float(gk[j]),observed_gradient=float(observed[j]))))
    return dict(rows=len(output),max_derivative_error=max((r['derivative_abs_error_max'] for r in output),default=None),
        micro_batches=output,note='CPU check of recorded PPO+reference-KL token derivatives; excludes the separate entropy branch and does not validate the transformer Jacobian.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    print(json.dumps(audit(parser.parse_args().run),indent=2))
