"""CPU checks preserve Qwen casting/order; --gpu separately tests the actual kernel."""
import importlib.util
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires pinned Torch runtime')
class RowwiseNormTest(unittest.TestCase):
    def test_wrapper_preserves_formula_casts_parameters_and_gradient(self):
        import torch
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
        from verl.models.transformers.qwen3_rowwise_rms_norm import install_qwen3_rowwise_rms_norm

        def reference_kernel(x, weight, bias, eps):
            self.assertEqual(x.dtype, torch.float32)
            self.assertTrue(torch.equal(weight, torch.ones_like(weight)))
            self.assertIsNone(bias)
            return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)

        torch.manual_seed(5)
        for dtype in (torch.float32, torch.bfloat16):
            original = Qwen3RMSNorm(128)
            candidate = Qwen3RMSNorm(128)
            candidate.load_state_dict(original.state_dict())
            before = candidate.weight
            self.assertEqual(install_qwen3_rowwise_rms_norm(candidate), 1)
            self.assertIs(before, candidate.weight)
            x = torch.randn(2, 7, 128).to(dtype).requires_grad_()
            y = x.detach().clone().requires_grad_()
            with patch.dict(sys.modules, {'flash_attn.ops.triton.layer_norm': SimpleNamespace(rms_norm_fn=reference_kernel)}):
                a, b = original(x), candidate(y)
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                a.square().sum().backward(); b.square().sum().backward()
            torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
            torch.testing.assert_close(original.weight.grad, candidate.weight.grad, rtol=0, atol=0)


def gpu_kernel_check():
    import json
    import torch
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    from verl.models.transformers.qwen3_rowwise_rms_norm import install_qwen3_rowwise_rms_norm
    torch.manual_seed(5)
    results = []
    for size in (128, 2560):
        for dtype in (torch.float32, torch.bfloat16):
            old, new = Qwen3RMSNorm(size).cuda(), Qwen3RMSNorm(size).cuda()
            new.load_state_dict(old.state_dict())
            install_qwen3_rowwise_rms_norm(new)
            x = torch.randn(2, 7, size, device='cuda').to(dtype).requires_grad_()
            y = x.detach().clone().requires_grad_()
            a, b = old(x), new(y)
            upstream = torch.randn_like(a)
            (a * upstream).sum().backward(); (b * upstream).sum().backward()
            values = {}
            for name, left, right in [('output', a, b), ('input_grad', x.grad, y.grad), ('weight_grad', old.weight.grad, new.weight.grad)]:
                diff = (left.float() - right.float()).detach()
                rel = diff.norm() / left.float().norm().clamp_min(1e-12)
                assert torch.isfinite(diff).all()
                assert rel < (2e-2 if dtype == torch.bfloat16 else 1e-5), (size, dtype, name, rel)
                values[name] = dict(abs_max=diff.abs().max().item(), relative_l2=rel.item())
            with torch.no_grad():
                row = y[:, 2:3].clone()
                torch.testing.assert_close(new(y)[:, 2:3], new(row), rtol=0, atol=0)
            results.append(dict(size=size, dtype=str(dtype), values=values, row_batch_equal=True))
    print(json.dumps(dict(status='passed', tests=results), indent=2))


if __name__ == '__main__':
    if '--gpu' in sys.argv:
        gpu_kernel_check()
    else:
        unittest.main()
