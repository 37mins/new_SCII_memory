"""CPU seed selection and actual SC2 reset dispatch, without Ray or games."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from test_r6_natural_termination import ROOT, method

RECORD = ROOT / 'experiments/records/20260908-garrison-wave-ownership-v1'
sys.path.insert(0, str(RECORD))
from spb_failed_seed_eval import select_seed


class FailedSeedEvaluationTest(unittest.TestCase):
    def raw(self):
        return SimpleNamespace(is_train=False, seed=1000, num_processes=8, reset_index=0,
                               validation_rng=np.random.RandomState(1000))

    def test_exact_eight_seeds_dispatched_and_logged_by_original_reset(self):
        raw = self.raw()
        calls = []
        def reset(seed):
            calls.append(seed)
            return ('obs', {})
        raw.workers = [SimpleNamespace(reset=SimpleNamespace(remote=reset)) for _ in range(8)]
        record = select_seed(SimpleNamespace(envs=raw))
        self.assertEqual(record['seeds'], [43567] * 8)
        fn = method(ROOT/'agent_system/environments/env_package/sc2_online_restored/envSC2.py',
                    'SC2MultiProcessEnv', 'reset', np=np, os=os, json=json,
                    ray=SimpleNamespace(get=lambda values: values))
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp)/'evaluation_seeds.jsonl'
            with patch.dict(os.environ, {'SC2_EVAL_SEEDS_LOG_PATH': str(log)}):
                observations, _ = fn(raw)
            self.assertEqual(len(observations), 8)
            self.assertEqual(calls, [43567] * 8)
            self.assertEqual(json.loads(log.read_text())['seeds'], calls)
            with self.assertRaises(ValueError):
                fn(raw)
            self.assertEqual(len(calls), 8)

    def test_wrong_environment_is_rejected(self):
        for field, value in [('is_train', True), ('seed', 0), ('num_processes', 4), ('reset_index', 1)]:
            with self.subTest(field=field):
                raw = self.raw()
                setattr(raw, field, value)
                with self.assertRaises(ValueError):
                    select_seed(SimpleNamespace(envs=raw))

    def test_hook_cannot_be_used_for_training_or_checkpoint_substitution(self):
        sys.path.insert(0, str(ROOT/'experiments/records/20260906-integrated-fixes-v1'))
        from spb_failed_seed_eval import install
        environments = SimpleNamespace(make_envs=lambda cfg: self.fail('must reject before creating envs'))
        with patch.dict(sys.modules, {'agent_system.environments': environments}):
            install()
            for val_only, resume in [(False, 'disable'), (True, 'resume_path')]:
                config = SimpleNamespace(trainer=SimpleNamespace(val_only=val_only, resume_mode=resume))
                with self.assertRaises(ValueError):
                    environments.make_envs(config)


if __name__ == '__main__':
    unittest.main()
