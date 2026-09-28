"""CPU tests for the diagnostic token forcing, not a model-equivalence gate."""
import importlib.util
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires pinned Linux Torch runtime")
class FixedTokenReplayTest(unittest.TestCase):
    def test_model_fp32_decorator_covers_forward_and_checkpoint_recompute(self):
        import torch
        from types import SimpleNamespace
        from transformers import Qwen3Config, Qwen3ForCausalLM
        path = Path(__file__).resolve().parents[1] / "experiments/records/20260906-qwen-kl-metric-v1/replay_fixed_tokens.py"
        node = next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.Assign)
                    and ast.unparse(n.targets[0]) == "worker.actor_module.forward")
        torch.manual_seed(1)
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, attention_dropout=0.)
        config._attn_implementation = 'eager'
        original, candidate = Qwen3ForCausalLM(config), Qwen3ForCausalLM(config)
        candidate.load_state_dict(original.state_dict())
        parameters = list(candidate.parameters())
        def cpu_context(device, enabled):
            self.assertEqual(device, 'cuda')
            self.assertFalse(enabled)
            return torch.autocast('cpu', enabled=False)
        scope = dict(worker=SimpleNamespace(actor_module=candidate), torch=SimpleNamespace(autocast=cpu_context))
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<real-source>', 'exec'), scope)
        tokens = torch.tensor([[1, 3, 4, 8]])
        upstream = torch.randn(1, 4, 32)
        for model in (original, candidate):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        expected = original(tokens, use_cache=False).logits
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = candidate(tokens, use_cache=False).logits
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        (expected * upstream).sum().backward(); (actual * upstream).sum().backward()
        for before, after, ref in zip(parameters, candidate.parameters(), original.parameters()):
            self.assertIs(before, after)
            torch.testing.assert_close(after.grad, ref.grad, rtol=0, atol=0)

    def test_layer_prediction_alignment_and_nonmutation(self):
        import torch
        from test_r6_kl_metrics import function
        path = Path(__file__).resolve().parents[1] / "experiments/records/20260906-qwen-kl-metric-v1/replay_fixed_tokens.py"
        states = function(path.read_text(), "prediction_states")
        gaps = function(path.read_text(), "layer_gaps", torch=torch)
        value = torch.arange(1 * 12 * 4).reshape(1, 12, 4).float()
        before = value.clone()
        full = states(value, "hf_full_fp32_normalization", 5, 3)
        self.assertTrue(torch.equal(full, value[:, 6:9]))
        cached = [states((value[:, :end], None), "hf_cached_generate", 5, 3) for end in (7, 8, 9)]
        result = gaps({"full": {"layer0": full}, "cached": {"layer0": cached}})
        self.assertEqual(result[0]['abs_max'], 0)
        self.assertEqual(result[0]['shape'], [1, 3, 4])
        self.assertTrue(torch.equal(value, before))

    def test_finished_rank_extra_forwards_are_not_counted_as_answer_tokens(self):
        import torch
        path = Path(__file__).resolve().parents[1] / "experiments/records/20260906-qwen-kl-metric-v1/replay_fixed_tokens.py"
        node = next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.Assign)
                    and ast.unparse(n.targets[0]) == "head_cached[:, :length]")
        scope = dict(torch=torch, head_cached=torch.zeros(1, 96), length=3,
                     head_values={"cached": [torch.tensor([[float(i)]]) for i in range(27)]})
        with self.assertRaises(RuntimeError):
            exec('head_cached[:, :length] = torch.cat(head_values["cached"], dim=1)', scope)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<real-source>", "exec"), scope)
        self.assertTrue(torch.equal(scope['head_cached'][0, :3], torch.arange(3).float()))
        self.assertTrue(torch.equal(scope['head_cached'][0, 3:], torch.zeros(93)))

    def test_shadow_head_is_fp32_and_does_not_mutate_inputs(self):
        import torch
        from test_r6_kl_metrics import function
        path = Path(__file__).resolve().parents[1] / "experiments/records/20260906-qwen-kl-metric-v1/replay_fixed_tokens.py"
        selected = function(path.read_text(), "selected")
        shadow = function(path.read_text(), "fp32_head_logp", torch=torch, selected=selected)
        torch.manual_seed(3)
        head = torch.nn.Linear(4, 5)
        hidden = torch.randn(1, 3, 4).bfloat16()
        labels = torch.tensor([[1, 2, 3]])
        before = hidden.clone(); weight = head.weight.detach().clone()
        expected = selected(head(hidden.float()), labels, .4)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = shadow(head, hidden, labels, .4)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(before, hidden))
        self.assertTrue(torch.equal(weight, head.weight))

    def test_new_start_keeps_historical_tensors_and_freezes_reference(self):
        import torch
        path = Path(__file__).resolve().parents[1] / "experiments/records/20260906-qwen-kl-metric-v1/replay_fixed_tokens.py"
        from test_r6_kl_metrics import function
        rebase = function(path.read_text(), "rebased_tensors")
        original = {"old_log_probs": torch.ones(2, 3), "ref_log_prob": torch.zeros(2, 3),
                    "input_ids": torch.ones(2, 5), "advantages": torch.randn(2, 3)}
        saved = {key: value.clone() for key, value in original.items()}
        fresh = torch.randn(2, 3, requires_grad=True)
        result = rebase(original, fresh)
        self.assertTrue(torch.equal(result['old_log_probs'], fresh))
        self.assertTrue(torch.equal(result['ref_log_prob'], fresh))
        result['old_log_probs'].add_(1)
        self.assertTrue(torch.equal(result['ref_log_prob'], fresh))
        self.assertFalse(result['ref_log_prob'].requires_grad)
        for key in original:
            self.assertTrue(torch.equal(original[key], saved[key]))
        self.assertIs(result['input_ids'], original['input_ids'])
        self.assertIs(result['advantages'], original['advantages'])

    def test_precision_ab_changes_only_actor_param_dtype(self):
        from omegaconf import OmegaConf
        source = Path(__file__).resolve().parents[1]
        path = source / "experiments/records/20260906-qwen-kl-metric-v1/replay_fixed_tokens.py"
        spec = importlib.util.spec_from_file_location("fixed_replay_config", path)
        replay = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(replay)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "effective_config.json").write_text(json.dumps({"overrides": {}}))
            original = OmegaConf.to_container(replay.load_config(run, source), resolve=True)
            baseline = OmegaConf.to_container(replay.load_config(run, source, "bf16"), resolve=True)
            candidate = OmegaConf.to_container(replay.load_config(run, source, "fp32"), resolve=True)
            normalized = OmegaConf.to_container(replay.load_config(run, source, "fp32", True), resolve=True)
        self.assertTrue(normalized['actor_rollout_ref']['actor'].pop('fp32_normalization'))
        self.assertEqual(normalized, candidate)
        self.assertEqual(baseline['actor_rollout_ref']['actor']['fsdp_config'].pop('mixed_precision'), {'param_dtype': 'bf16'})
        self.assertEqual(candidate['actor_rollout_ref']['actor']['fsdp_config'].pop('mixed_precision'), {'param_dtype': 'fp32'})
        self.assertEqual(baseline, original)
        self.assertEqual(candidate, original)
        arguments = [str(path), '--run', '/historical', '--source', str(source),
                     '--output', '/not-created', '--param-dtype', 'fp32', '--max-updates', '4']
        with patch.object(sys, 'argv', arguments), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            replay.main()
        self.assertEqual(stopped.exception.code, 2)
        arguments = [str(path), '--run', '/historical', '--source', str(source),
                     '--output', '/not-created', '--hf-full-precision', '--max-updates', '4']
        with patch.object(sys, 'argv', arguments), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            replay.main()
        self.assertEqual(stopped.exception.code, 2)

    def test_update_prefix_matches_worker_device_boundary(self):
        source = Path(__file__).resolve().parents[1]
        replay = source / "experiments/records/20260906-qwen-kl-metric-v1/replay_fixed_tokens.py"
        tree = ast.parse(replay.read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        body = next(node.body for node in main.body if isinstance(node, ast.If) and ast.unparse(node.test) == "count")
        update = next(i for i, node in enumerate(body) if ast.unparse(node) == "metrics = actor.update_policy(prefix)")
        self.assertIn("prefix = prefix.to(torch.cuda.current_device())", [ast.unparse(n) for n in body[:update]])
        worker = (source / "verl/workers/fsdp_workers.py").read_text()
        self.assertIn("data = data.to(get_torch_device().current_device())", worker)

    def test_torchrun_script_arguments_are_not_abbreviated_as_launcher_options(self):
        from torch.distributed.run import get_args_parser
        script = "replay_fixed_tokens.py"
        launcher = ["--standalone", "--nnodes=1", "--nproc_per_node=4"]
        arguments = ["--source", "/source", "--run", "/historical", "--output", "/new", "--max-updates", "4", "--skip-paths"]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            get_args_parser().parse_args(launcher + [script] + arguments)
        parsed = get_args_parser().parse_args(launcher + ["--", script] + arguments)
        self.assertEqual(parsed.training_script, script)
        self.assertEqual(parsed.training_script_args, arguments)

    def test_cached_forced_answer_preserves_raw_probabilities(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM

        path = Path(__file__).resolve().parents[1] / (
            "experiments/records/20260906-qwen-kl-metric-v1/replay_fixed_tokens.py")
        spec = importlib.util.spec_from_file_location("fixed_replay", path)
        replay = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(replay)
        torch.set_num_threads(1)
        torch.manual_seed(0)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, bos_token_id=1, eos_token_id=63, pad_token_id=0,
            max_position_embeddings=128, attention_dropout=0.)
        config._attn_implementation = "eager"
        model = Qwen3ForCausalLM(config).eval()
        prompt = torch.tensor([[0, 0, 3, 4, 5]])
        response = torch.tensor([[6, 7, 63]])
        forcing = replay.FixedTokens(response, 5)
        scores = torch.randn(1, 64)
        before = scores.clone()
        self.assertEqual(forcing(prompt, scores).argmax(-1).item(), 6)
        self.assertTrue(torch.equal(scores, before))
        with torch.no_grad():
            generated = model.generate(input_ids=prompt, attention_mask=(prompt != 0).long(),
                do_sample=True, temperature=.4, top_p=.9, top_k=0, use_cache=True,
                max_new_tokens=8, logits_processor=[forcing],
                output_logits=True, return_dict_in_generate=True)
            self.assertTrue(torch.equal(generated.sequences[:, 5:], response))
            full = torch.cat([prompt, response], 1)
            mask = (full != 0).long()
            positions = (mask.cumsum(-1) - 1).clamp_min(0)
            logits = model(input_ids=full, attention_mask=mask, position_ids=positions,
                           use_cache=False).logits[:, 4:-1]
            expected = replay.selected(logits, response, .4)
            actual = torch.stack([replay.selected(value, response[:, index], .4)
                for index, value in enumerate(generated.logits)], 1)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
