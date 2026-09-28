"""Actual HF sampling must be identical with/without selected-token observation."""
import contextlib
import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
                     "requires pinned Linux Torch/Transformers")
class HfLogProbDiagnosticsTest(unittest.TestCase):
    def test_same_samples_rng_masks_and_cached_probabilities(self):
        import torch
        from omegaconf import OmegaConf
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from verl import DataProto
        from verl.utils.model import compute_position_id_with_mask
        from verl.workers.rollout.hf_rollout import HFRollout

        torch.set_num_threads(1)
        torch.manual_seed(123)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, attention_dropout=0., bos_token_id=1, eos_token_id=63,
            pad_token_id=0, max_position_embeddings=128)
        config._attn_implementation = "eager"
        model = Qwen3ForCausalLM(config).eval()
        ids = torch.tensor([[0, 0, 3, 4, 5, 6], [0, 3, 4, 5, 6, 7]])
        mask = (ids != 0).long()
        prompts = DataProto.from_dict(tensors=dict(input_ids=ids, attention_mask=mask,
            position_ids=compute_position_id_with_mask(mask)),
            meta_info=dict(eos_token_id=63, pad_token_id=0))

        def generate(capture):
            torch.manual_seed(321)
            rollout = HFRollout(model, OmegaConf.create(dict(do_sample=True, temperature=.4,
                top_p=.6, top_k=0, response_length=8, n=1, capture_log_probs=capture)))
            with patch("verl.workers.rollout.hf_rollout.torch.autocast", return_value=contextlib.nullcontext()), \
                 patch("verl.workers.rollout.hf_rollout.get_torch_device"):
                result = rollout.generate_sequences(prompts)
            return result.batch, torch.get_rng_state()

        plain, plain_rng = generate(False)
        traced, traced_rng = generate(True)
        self.assertTrue(torch.equal(plain_rng, traced_rng))
        self.assertEqual(set(traced.keys()) - set(plain.keys()),
            {"rollout_unwarped_log_probs", "rollout_sampling_log_probs"})
        for key in plain.keys():
            self.assertTrue(torch.equal(plain[key], traced[key]), key)
        model.eval()
        with torch.no_grad():
            full = model(input_ids=traced["input_ids"], attention_mask=traced["attention_mask"],
                position_ids=traced["position_ids"], use_cache=False).logits[:, 5:-1].float() / .4
        expected = torch.log_softmax(full, -1).gather(2, traced["responses"].unsqueeze(-1)).squeeze(-1)
        active = traced["attention_mask"][:, -8:].bool()
        torch.testing.assert_close(traced["rollout_unwarped_log_probs"][active], expected[active],
                                   atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.isfinite(traced["rollout_sampling_log_probs"][active]).all())
        self.assertGreater(float((traced["rollout_sampling_log_probs"] -
                                  traced["rollout_unwarped_log_probs"])[active].abs().max()), .01)

    def test_eos_padding_is_zero_and_raw_logits_are_not_modified(self):
        import torch
        from verl.workers.rollout.hf_diagnostics import selected_log_probs
        raw = torch.tensor([[1., 2., 3.], [3., 2., 1.]])
        scores = torch.tensor([[-float("inf"), 4., 6.], [6., 4., -float("inf")]])
        before = raw.clone()
        output = SimpleNamespace(logits=(raw, raw), scores=(scores, scores))
        responses = torch.tensor([[2, 0, 0], [0, 1, 0]])
        mask = torch.tensor([[1, 0, 0], [1, 1, 0]])
        unwarped, sampling = selected_log_probs(output, responses, mask, .5)
        self.assertTrue(torch.equal(raw, before))
        self.assertTrue(torch.isfinite(sampling).all())
        self.assertTrue(torch.equal(sampling[~mask.bool()], torch.zeros(3)))
        self.assertAlmostEqual(unwarped[0, 0].item(), torch.log_softmax(raw[0] / .5, -1)[2].item())


if __name__ == "__main__":
    unittest.main()
