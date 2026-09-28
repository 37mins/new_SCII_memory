"""Exercise the actual fixed-text actor forward at BF16 rounding boundaries."""
import importlib.util
from types import SimpleNamespace
from typing import Tuple
import unittest

from test_r6_kl_metrics import ACTOR, function


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires pinned Linux Torch runtime")
class TemperaturePrecisionTest(unittest.TestCase):
    def test_forward_and_gradient_match_explicit_precision(self):
        import torch
        torch.set_num_threads(1)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([128., 127.5, 125., 120.]))

            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(logits=self.weight.to(torch.bfloat16).expand(
                    input_ids.shape[0], input_ids.shape[1], -1).clone())

        def logp(logits, labels):
            return logits.float().log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        def entropy(logits):
            logs = logits.float().log_softmax(-1)
            return -(logs.exp() * logs).sum(-1)

        forward = function(ACTOR.read_text(), "_forward_micro_batch", torch=torch, Tuple=Tuple,
                           logprobs_from_logits=logp, verl_F=SimpleNamespace(entropy_from_logits=entropy))
        micro = dict(input_ids=torch.tensor([[1, 2, 1, 0, 0]]),
                     attention_mask=torch.ones(1, 5), position_ids=torch.arange(5).unsqueeze(0),
                     responses=torch.tensor([[1, 0, 0]]))
        outputs = []
        for flag in (False, True):
            model = Model()
            config = SimpleNamespace(get=lambda key, default=None: flag if key == "fp32_normalization" else default)
            actor = SimpleNamespace(config=config, device_name="cpu", actor_module=model,
                                    use_remove_padding=False, use_fused_kernels=False)
            actual_entropy, actual = forward(actor, micro, .4, True)
            (-actual.mean() - .001 * actual_entropy.mean()).backward()
            gradient = model.weight.grad.clone()
            reference = Model()
            logits = reference(**micro).logits[:, 1:-1]
            expected_logits = logits.float() / .4 if flag else (logits / .4)
            expected, expected_entropy = logp(expected_logits, micro['responses']), entropy(expected_logits)
            (-expected.mean() - .001 * expected_entropy.mean()).backward()
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(actual_entropy, expected_entropy, atol=0, rtol=0)
            torch.testing.assert_close(gradient, reference.weight.grad, atol=0, rtol=0)
            self.assertTrue(torch.isfinite(gradient).all())
            outputs.append(actual.detach())
        self.assertGreater((outputs[0] - outputs[1]).abs().max().item(), .01)


if __name__ == "__main__":
    unittest.main()
