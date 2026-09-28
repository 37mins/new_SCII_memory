"""Test valid-token distributions and same-batch accounting, not model quality."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires pinned Torch runtime')
class FullUpdateSummaryTest(unittest.TestCase):
    def test_loss_gradient_audit_uses_rank_local_accumulation(self):
        import torch
        path=Path(__file__).resolve().parents[1]/'experiments/records/20260906-qwen-kl-metric-v1/audit_loss_gradients.py'
        spec=importlib.util.spec_from_file_location('gradient_audit_test',path)
        module=importlib.util.module_from_spec(spec)
        with patch.object(sys,'path',[str(path.parent),*sys.path]):
            spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'ppo_diagnostics').mkdir()
            (root/'manifest.json').write_text(json.dumps({'world_size':4}))
            config={'ppo_mini_batch_size':8,'ppo_micro_batch_size_per_gpu':1,
                'clip_ratio':.2,'clip_ratio_low':.2,'clip_ratio_high':.28,'clip_ratio_c':3.,
                'loss_agg_mode':'token-mean','kl_loss_type':'low_var_kl','kl_loss_coef':.01}
            events=[dict(event='config',config={'actor':config}),dict(event='token_loss_gradient',
                optimizer_updates_before=0,sample_offset=0,gradient=[-.05,-.05])]
            (root/'rank-00.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
            saved=dict(responses=torch.tensor([[1,2,0]]),attention_mask=torch.tensor([[1,1,1,0]]),
                old_log_probs=torch.zeros(1,3),ref_log_prob=torch.zeros(1,3),advantages=torch.full((1,3),.2))
            path=root/'ppo_diagnostics'/'update-0001-rank-00.pt'; torch.save(saved,path)
            path.with_suffix('.jsonl').write_text(json.dumps(dict(phase='update',sample_offset=0,
                optimizer_updates_before=0,log_prob=[[0.,0.,0.]]))+'\n')
            result=module.audit(root)
        self.assertEqual(result['rows'],1)
        self.assertLess(result['max_derivative_error'],1e-8)

    def test_uses_effective_clip_bounds_and_excludes_response_padding(self):
        import torch
        path=Path(__file__).resolve().parents[1]/'experiments/records/20260906-qwen-kl-metric-v1/summarize_full_updates.py'
        spec=importlib.util.spec_from_file_location('full_summary_test',path)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'ppo_diagnostics').mkdir()
            for rank in range(4):
                config=dict(event='config',rank=rank,config={'actor':{'clip_ratio':.2,'clip_ratio_low':None,'clip_ratio_high':.4}})
                (root/f'rank-{rank:02d}.jsonl').write_text(json.dumps(config)+'\n')
                saved=dict(responses=torch.tensor([[1,2,0]]),attention_mask=torch.tensor([[1,1,1,0]]),
                    old_log_probs=torch.zeros(1,3),advantages=torch.ones(1,3))
                torch.save(saved,root/'ppo_diagnostics'/f'update-0001-rank-{rank:02d}.pt')
                row=dict(sample_offset=0,optimizer_updates_before=1,log_prob=[[-.25,.3,1000.]])
                (root/f'post-each-rank-{rank:02d}.jsonl').write_text(json.dumps(row)+'\n')
            result=module.summarize(root)['optimizer_updates'][0]
        self.assertEqual(result['tokens'],8)
        self.assertAlmostEqual(result['abs_quantiles']['max'],.3)
        self.assertAlmostEqual(result['abs_mean'],.275)
        self.assertEqual(result['ratio_below_clip'],.5)
        self.assertEqual(result['ratio_above_clip'],0.)
        self.assertEqual(result['objective_clipfrac'],0.)
        self.assertEqual(result['exponent_clamped_tokens'],0)
        self.assertFalse(result['complete_all_ranks'])


if __name__=='__main__':
    unittest.main()
