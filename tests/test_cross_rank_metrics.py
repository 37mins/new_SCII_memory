"""Exercise the real concat/collector; never start Ray, a model or a game."""
import copy
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires pinned Linux Torch runtime")
class CrossRankMetricsTest(unittest.TestCase):
    def test_four_rank_lists_scalars_and_inputs_are_preserved(self):
        from verl import DataProto
        from verl.utils.metric.utils import reduce_metrics

        values = [4.8661622541, 5.3275671893, 7.4880328060, 6.2912067818]
        packets = [DataProto(meta_info={"temperature": .4, "metrics": {
            "actor/ppo_kl": [value] * 69, "actor/lr": 1e-6,
            "actor/grad_norm": [rank + .5] * 35,
            "perf/max_memory_allocated_gb": 10. + rank}})
            for rank, value in enumerate(values)]
        before = copy.deepcopy([p.meta_info for p in packets])
        result = DataProto.concat(packets)
        metrics = result.meta_info["metrics"]
        self.assertEqual(len(metrics["actor/ppo_kl"]), 276)
        self.assertEqual(len(metrics["actor/grad_norm"]), 140)
        self.assertEqual(len(metrics["actor/lr"]), 4)
        reduced = reduce_metrics(metrics)
        self.assertAlmostEqual(reduced["actor/ppo_kl"], 5.9932422578)
        self.assertEqual(reduced["perf/max_memory_allocated_gb"], 13.)
        self.assertEqual(reduced["actor/lr"], 1e-6)
        self.assertEqual([p.meta_info for p in packets], before)

    def test_different_microbatch_counts_and_recursive_concat(self):
        from verl import DataProto
        from verl.utils.metric.utils import reduce_metrics

        packets = [DataProto(meta_info={"metrics": {"loss": values}})
                   for values in ([1., 3.], [5.], [7., 9., 11.])]
        result = DataProto.concat([DataProto.concat(packets[:2]), packets[2]])
        self.assertEqual(result.meta_info["metrics"]["loss"], [1., 3., 5., 7., 9., 11.])
        self.assertEqual(reduce_metrics(result.meta_info["metrics"])["loss"], 6.)

    def test_nonmetric_tensor_and_numpy_payloads_unchanged(self):
        import numpy as np
        import torch
        from verl import DataProto

        packets = [DataProto.from_dict(tensors={"x": torch.tensor([[rank]])},
            non_tensors={"id": np.array([str(rank)], dtype=object)},
            meta_info={"temperature": .4}) for rank in range(4)]
        result = DataProto.concat(packets)
        self.assertTrue(torch.equal(result.batch["x"], torch.arange(4).reshape(-1, 1)))
        self.assertEqual(result.non_tensor_batch["id"].tolist(), ["0", "1", "2", "3"])
        self.assertIs(result.meta_info, packets[0].meta_info)

    def test_actual_dp_collector_reaches_all_rank_metrics(self):
        from verl import DataProto
        from verl.single_controller.base.decorator import collect_dp_compute_data_proto
        from verl.single_controller.base.worker_group import WorkerGroup
        from verl.utils.metric.utils import reduce_metrics

        packets = [DataProto(meta_info={"metrics": {"loss": [float(i)]}})
                   for i in range(4)]
        group = WorkerGroup(None)
        group._workers = [None] * 4
        result = collect_dp_compute_data_proto(group, packets)
        self.assertEqual(reduce_metrics(result.meta_info["metrics"])["loss"], 1.5)


if __name__ == "__main__":
    unittest.main()
