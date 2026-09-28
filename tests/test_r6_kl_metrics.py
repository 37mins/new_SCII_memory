"""KL reporting regression; the optional CPU test runs the actual update loop."""
import ast
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
ACTOR = ROOT / "verl/workers/actor/dp_actor.py"
OLD = 'metrics["actor/kl_loss"] = kl_loss.detach().item()'
NEW = 'append_to_dict(metrics, {"actor/kl_loss": kl_loss.detach().item()})'


def function(source, name, **scope):
    node = next(n for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<real-source>", "exec"), scope)
    return scope[name]


def append_function():
    return function((ROOT / "verl/utils/py_functional.py").read_text(),
                    "append_to_dict", Dict=dict)


class KlMetricTest(unittest.TestCase):
    def test_kl_loss_block_stays_identical_to_metric_fix(self):
        source = ACTOR.read_text()
        self.assertEqual(source.count(NEW), 1)
        block = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.If)
                     and ast.unparse(n.test) == 'self.config.use_kl_loss' and len(n.body) > 2)
        self.assertEqual(hashlib.sha256(ast.unparse(block).encode()).hexdigest(),
                         "b1fb2fcffcc5cf82ea425993ed8c7f3219058d64103b30a2efdf9be6cd73c1ca")

    def test_later_microbatch_does_not_overwrite_earlier_values(self):
        for statement, expected in ((OLD, 3.333), (NEW, [0.1, 3.333])):
            scope = dict(metrics={}, append_to_dict=append_function())
            for value in (0.1, 3.333):
                scope["kl_loss"] = SimpleNamespace(
                    detach=lambda value=value: SimpleNamespace(item=lambda: value))
                exec(statement, scope)
            self.assertEqual(scope["metrics"]["actor/kl_loss"], expected)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"),
                         "requires pinned Linux Torch/NumPy runtime")
    def test_real_cpu_update_preserves_gradients_and_parameters(self):
        import torch
        from verl import DataProto
        from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
        from verl.utils.metric.utils import reduce_metrics

        torch.set_num_threads(1)
        responses = torch.tensor([[1, 2, 3], [2, 3, 1], [3, 1, 2], [1, 3, 2]])
        masks = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0],
                              [1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        data = DataProto.from_dict(tensors=dict(
            responses=responses, input_ids=torch.cat([torch.ones(4, 2, dtype=torch.long), responses], 1),
            attention_mask=masks, position_ids=torch.arange(5).repeat(4, 1),
            old_log_probs=torch.full((4, 3), -1.0), ref_log_prob=torch.full((4, 3), -0.9),
            advantages=torch.tensor([[.4] * 3, [-.1] * 3, [.2] * 3, [-.3] * 3])),
            meta_info={"temperature": .4})

        def run(source):
            model = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                model.weight.fill_(.2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            config = SimpleNamespace(use_kl_loss=True, ppo_mini_batch_size=2,
                ppo_micro_batch_size_per_gpu=1, ppo_epochs=2, use_dynamic_bsz=False,
                clip_ratio=.2, clip_ratio_low=.2, clip_ratio_high=.28, entropy_coeff=0.,
                loss_agg_mode="token-mean", policy_loss={"loss_mode": "vanilla"},
                kl_loss_type="low_var_kl", kl_loss_coef=.01)
            config.get = lambda key, default=None: getattr(config, key, default)
            observed, gradients = [], []

            def forward(micro_batch, temperature, calculate_entropy):
                return None, -torch.nn.functional.softplus(
                    model.weight.reshape(()) * micro_batch["responses"].float() / temperature)

            def aggregate(**kwargs):
                value = agg_loss(**kwargs)
                observed.append(value.detach().item())
                return value

            def step():
                gradients.append(model.weight.grad.detach().clone())
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                return norm

            actor = SimpleNamespace(actor_module=model, actor_optimizer=optimizer,
                config=config, _forward_micro_batch=forward, _optimizer_step=step)
            update = function(source, "update_policy", DataProto=DataProto, torch=torch,
                get_torch_device=lambda: SimpleNamespace(current_device=lambda: "cpu"),
                compute_policy_loss=compute_policy_loss, agg_loss=aggregate,
                kl_penalty=kl_penalty, append_to_dict=append_function())
            metrics = update(actor, data)
            return metrics, observed, gradients, model.weight.detach().clone()

        new_source = ACTOR.read_text()
        old, old_values, old_grads, old_weight = run(new_source.replace(NEW, OLD))
        new, values, grads, weight = run(new_source)
        self.assertEqual(len(values), 8)  # 2 epochs x 2 minibatches x 2 microbatches.
        self.assertEqual(old_values, values)
        self.assertEqual(old["actor/kl_loss"], values[-1])
        self.assertEqual(new["actor/kl_loss"], values)
        mean = reduce_metrics({"actor/kl_loss": values.copy()})["actor/kl_loss"]
        self.assertAlmostEqual(mean, sum(values) / len(values))
        self.assertNotAlmostEqual(mean, old["actor/kl_loss"])
        for before, after in zip(old_grads, grads):
            self.assertTrue(torch.equal(before, after))
        self.assertTrue(torch.equal(old_weight, weight))
        self.assertEqual({k: v for k, v in old.items() if k != "actor/kl_loss"},
                         {k: v for k, v in new.items() if k != "actor/kl_loss"})


if __name__ == "__main__":
    unittest.main()
