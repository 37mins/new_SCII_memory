"""The actual actor update must be unchanged when diagnostic passes are enabled."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from test_r6_kl_metrics import ACTOR, append_function, function


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires pinned Linux Torch runtime")
class PpoDiagnosticsTest(unittest.TestCase):
    def test_actual_update_rng_gradients_parameters_and_metrics_are_unchanged(self):
        import torch
        from verl import DataProto
        from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty

        torch.set_num_threads(1)
        responses = torch.tensor([[1, 2, 3], [2, 3, 1], [3, 1, 2], [1, 3, 2]])
        data = DataProto.from_dict(tensors=dict(responses=responses,
            input_ids=torch.cat([torch.ones(4, 2, dtype=torch.long), responses], 1),
            attention_mask=torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0],
                                         [1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]),
            position_ids=torch.arange(5).repeat(4, 1),
            old_log_probs=torch.full((4, 3), -1.0), ref_log_prob=torch.full((4, 3), -.9),
            rollout_unwarped_log_probs=torch.full((4, 3), -1.05),
            rollout_sampling_log_probs=torch.full((4, 3), -.7),
            advantages=torch.tensor([[.4]*3, [-.1]*3, [.2]*3, [-.3]*3])),
            meta_info={"temperature": .4})

        def run(directory, full_observer=False):
            torch.manual_seed(812)
            model = torch.nn.Linear(1, 1, bias=False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            config = SimpleNamespace(use_kl_loss=True, ppo_mini_batch_size=2,
                ppo_micro_batch_size_per_gpu=1, ppo_epochs=2, use_dynamic_bsz=False,
                clip_ratio=.2, clip_ratio_low=.2, clip_ratio_high=.28, entropy_coeff=0.,
                loss_agg_mode="token-mean", policy_loss={"loss_mode": "vanilla"},
                kl_loss_type="low_var_kl", kl_loss_coef=.01, ppo_diagnostics_dir=directory)
            config.get = lambda key, default=None: getattr(config, key, default)
            losses, gradients = [], []

            def forward(micro_batch, temperature, calculate_entropy):
                # Deliberately stochastic: extra diagnostic forwards must not change RNG.
                return None, -torch.nn.functional.softplus(model.weight.reshape(()) *
                    micro_batch["responses"].float() / temperature + .1 * torch.rand(()))

            def aggregate(**kwargs):
                value = agg_loss(**kwargs)
                losses.append(value.detach().clone())
                return value

            def step():
                gradients.append(model.weight.grad.detach().clone())
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                return norm

            actor = SimpleNamespace(config=config, actor_module=model,
                actor_optimizer=optimizer, _forward_micro_batch=forward, _optimizer_step=step)
            if full_observer:
                observer_path = Path(__file__).resolve().parents[1] / 'experiments/records/20260906-qwen-kl-metric-v1/full_update_observers.py'
                spec = importlib.util.spec_from_file_location('full_update_observers_test', observer_path)
                observer = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(observer)
                def emit(event, **values):
                    with (Path(directory)/'observer-events.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(event=event, **values))+'\n')
                observer.install_update_observers(actor, data.batch, .4, emit, Path(directory)/'post-each.jsonl')
            update = function(ACTOR.read_text(), "update_policy", DataProto=DataProto,
                torch=torch, get_torch_device=lambda: SimpleNamespace(current_device=lambda: "cpu"),
                compute_policy_loss=compute_policy_loss, agg_loss=aggregate,
                kl_penalty=kl_penalty, append_to_dict=append_function())
            metrics = update(actor, data)
            return metrics, losses, gradients, model.weight.detach().clone(), torch.get_rng_state()

        with tempfile.TemporaryDirectory() as tmp:
            plain, traced = run(None), run(tmp)
            self.assertEqual(plain[0], traced[0])
            for before, after in zip(plain[1] + plain[2] + list(plain[3:]),
                                     traced[1] + traced[2] + list(traced[3:])):
                self.assertTrue(torch.equal(before, after))
            rows = [json.loads(line) for line in (Path(tmp)/"update-0001-rank-00.jsonl").read_text().splitlines()]
            self.assertEqual([sum(r["phase"] == phase for r in rows)
                              for phase in ("pre_update", "update", "post_update")], [4, 8, 4])
            self.assertEqual({r["optimizer_updates_before"] for r in rows if r["phase"] == "pre_update"}, {0})
            self.assertEqual({r["optimizer_updates_before"] for r in rows if r["phase"] == "post_update"}, {4})
            saved = torch.load(Path(tmp)/"update-0001-rank-00.pt", weights_only=True)
            self.assertTrue(torch.equal(saved["responses"], responses))
            self.assertTrue(torch.equal(saved["rollout_unwarped_log_probs"],
                                        data.batch["rollout_unwarped_log_probs"]))
            self.assertAlmostEqual(rows[0]["rollout_unwarped_eval_abs_gap_mean"], .05, places=6)
            self.assertAlmostEqual(rows[0]["rollout_sampling_eval_abs_gap_mean"], .3, places=6)
        with tempfile.TemporaryDirectory() as tmp:
            observed = run(tmp, full_observer=True)
            self.assertEqual(plain[0], observed[0])
            for before, after in zip(plain[1] + plain[2] + list(plain[3:]),
                                     observed[1] + observed[2] + list(observed[3:])):
                self.assertTrue(torch.equal(before, after))
            events = [json.loads(line) for line in (Path(tmp)/'observer-events.jsonl').read_text().splitlines()]
            self.assertEqual([e['optimizer_updates'] for e in events if e['event']=='optimizer_update'], [1,2,3,4])
            gradients = [e for e in events if e['event']=='token_loss_gradient']
            self.assertEqual(len(gradients), 8)
            self.assertTrue(all(e['finite'] for e in gradients))
            self.assertEqual([e['sample_offset'] for e in gradients], [0,1,2,3,0,1,2,3])
            posts = [json.loads(line) for line in (Path(tmp)/'post-each.jsonl').read_text().splitlines()]
            self.assertEqual(len(posts), 16)
            self.assertEqual(sorted({e['optimizer_updates_before'] for e in posts}), [1,2,3,4])

    def test_mask_zero_gap_and_ratio_outside_are_distinct_from_objective_clip(self):
        import torch
        from verl.workers.actor.ppo_diagnostics import record_micro_batch
        actor = SimpleNamespace(config=SimpleNamespace(clip_ratio=.2, clip_ratio_low=.2, clip_ratio_high=.28))
        old = torch.tensor([[-1., float("nan")]])
        data = dict(responses=torch.tensor([[7, 8]]), attention_mask=torch.tensor([[1, 1, 0]]),
                    old_log_probs=old, ref_log_prob=old, advantages=torch.tensor([[.3, 0.]]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"trace.jsonl"
            record_micro_batch(path, actor, data, old, "pre_update", 0, 0)
            record_micro_batch(path, actor, data, old - .271, "update", 1, 0)
            zero, moved = map(json.loads, path.read_text().splitlines())
            self.assertEqual(zero["abs_logprob_gap_max"], 0.)
            self.assertEqual(zero["tokens"], 1)
            self.assertAlmostEqual(moved["ppo_k1_mean"], .271, places=6)
            self.assertEqual(moved["ratio_below_clip"], 1.)
            self.assertEqual(moved["objective_clipfrac"], 0.)


if __name__ == "__main__":
    unittest.main()
