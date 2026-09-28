"""CPU checks for the opt-in HF/FSDP wrapping boundary, not GPU OOM acceptance."""
import ast
import contextlib
import copy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def selected_policy(enabled, rollout=True, name="hf"):
    tree = ast.parse((ROOT / "verl/workers/fsdp_workers.py").read_text())
    gate = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                and "fsdp_layer_wrap" in ast.unparse(node.test))
    config = dict(name=name)
    if enabled is not None:
        config["fsdp_layer_wrap"] = enabled
    class Config(dict):
        __getattr__ = dict.__getitem__
    policy = object()
    scope = dict(self=SimpleNamespace(_is_rollout=rollout,
                 config=SimpleNamespace(rollout=Config(config))), auto_wrap_policy=policy)
    exec(compile(ast.Module(body=[gate], type_ignores=[]), "wrap_gate", "exec"), scope)
    return scope["auto_wrap_policy"] is policy


def cpu_rank(rank, init_file):
    import datetime
    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from transformers import Qwen3Config, Qwen3ForCausalLM, LogitsProcessor
    from verl import DataProto
    from verl.utils.fsdp_utils import get_fsdp_wrap_policy
    from verl.utils.model import compute_position_id_with_mask
    from verl.workers.rollout.hf_rollout import HFRollout

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + init_file, rank=rank,
                            world_size=2, timeout=datetime.timedelta(seconds=30))
    try:
        torch.manual_seed(17)
        cfg = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, attention_dropout=0., bos_token_id=1, eos_token_id=31,
            pad_token_id=0, max_position_embeddings=64, tie_word_embeddings=True)
        cfg._attn_implementation = "eager"
        source = Qwen3ForCausalLM(cfg).float()
        baseline = copy.deepcopy(source)
        source.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        wrapped = FSDP(source, device_id=torch.device("cpu"), use_orig_params=False,
                       auto_wrap_policy=get_fsdp_wrap_policy(source, {"min_num_params": 0}))
        assert len(FSDP.fsdp_modules(wrapped)) == 3
        ids = torch.tensor([[1, 3, 5, 7]])
        mask = torch.ones_like(ids)
        positions = compute_position_id_with_mask(mask)
        optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-6)
        reference_optimizer = torch.optim.AdamW(baseline.parameters(), lr=1e-6)
        # Two micro-batches reproduce gradient accumulation before one Adam step.
        for _ in range(2):
            actual = wrapped(input_ids=ids, attention_mask=mask, use_cache=False).logits
            expected = baseline(input_ids=ids, attention_mask=mask, use_cache=False).logits
            torch.testing.assert_close(actual, expected)
            (actual.float().square().mean() / 2).backward()
            (expected.float().square().mean() / 2).backward()
        fsdp_norm = wrapped.clip_grad_norm_(1.)
        reference_norm = torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1.)
        torch.testing.assert_close(fsdp_norm, reference_norm)
        optimizer.step()
        reference_optimizer.step()
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        with FSDP.summon_full_params(wrapped):
            for actual, expected in zip(source.parameters(), baseline.parameters()):
                torch.testing.assert_close(actual, expected)

        class EndAtDifferentTimes(LogitsProcessor):
            def __call__(self, input_ids, scores):
                token = 31 if input_ids.shape[1] >= 4 + rank * 2 else 3
                scores.fill_(-float("inf"))
                scores[:, token] = 0
                return scores

        original_generate = source.generate
        def generate(**kwargs):
            assert kwargs["use_cache"] is True
            return original_generate(**kwargs, logits_processor=[EndAtDifferentTimes()])
        source.generate = generate
        rollout = HFRollout(wrapped, OmegaConf.create(dict(do_sample=False, temperature=.4,
            top_p=.9, top_k=0, response_length=5, n=1, fsdp_layer_wrap=True)))
        prompt = DataProto.from_dict(tensors=dict(input_ids=ids, attention_mask=mask,
            position_ids=positions), meta_info=dict(eos_token_id=31, pad_token_id=0))
        # The production adapter chooses CUDA autocast; this tiny test runs CPU only.
        with patch("verl.workers.rollout.hf_rollout.torch.autocast", return_value=contextlib.nullcontext()), \
                patch("verl.workers.rollout.hf_rollout.get_torch_device"):
            result = rollout.generate_sequences(prompt)
        active = int(result.batch["attention_mask"][:, -5:].sum())
        assert active == 1 + rank * 2, (rank, active)
        assert not torch.cuda.is_initialized()
        print(f"CPU rank={rank}: nested_fsdp=3, backward/Adam/cache/EOS passed, response_tokens={active}", flush=True)
    finally:
        dist.destroy_process_group()


class WrapGateTest(unittest.TestCase):
    def test_default_and_explicit_false_preserve_legacy(self):
        self.assertFalse(selected_policy(None))
        self.assertFalse(selected_policy(False))

    def test_opt_in_preserves_existing_layer_policy(self):
        self.assertTrue(selected_policy(True))

    def test_non_hf_and_reference_are_unchanged(self):
        self.assertTrue(selected_policy(None, name="vllm"))
        self.assertTrue(selected_policy(None, rollout=False))

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
                         "requires pinned Torch/Transformers; CPU only")
    def test_two_cpu_ranks_backward_adam_and_unequal_eos(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as tmp:
            context = mp.spawn(cpu_rank, args=(str(Path(tmp)/"gloo"),), nprocs=2, join=False)
            try:
                import time
                deadline = time.monotonic() + 75
                while not context.join(timeout=1):
                    if time.monotonic() >= deadline:
                        self.fail("CPU FSDP test exceeded 75 seconds")
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
