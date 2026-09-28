"""CPU-only tests of the real reset function; no SC2/Ray/model imports."""
import ast
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT/'agent_system/environments/env_package/sc2_online_restored/envSC2.py'


def reset_method():
    cls = next(n for n in ast.parse(ENV.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'SC2MultiProcessEnv')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'reset')
    ns = dict(os=os, json=json, random=random, ray=NS(get=lambda rows: rows),
              np=NS(repeat=lambda values,n: NS(tolist=lambda: [x for x in values for _ in range(n)])))
    exec(compile(ast.Module(body=[fn], type_ignores=[]),str(ENV),'exec'), ns)
    return ns['reset']


def environment(index=0, train=True):
    return NS(is_train=train, seed=0, reset_index=index, env_num=1, group_n=8, num_processes=8,
              training_rng=NS(randint=lambda *args,**kw: [2732]),
              validation_rng=NS(randint=lambda *args,**kw: NS(tolist=lambda: list(range(1000,1008)))),
              workers=[NS(reset=NS(remote=lambda seed: ({'seed':seed}, {}))) for _ in range(8)])


class OriginalGameSeedsTest(unittest.TestCase):
    def setUp(self):
        self.env_patch = patch.dict(os.environ, SC2_TRAIN_SEEDS_LOG_PATH='', SC2_EVAL_SEEDS_LOG_PATH='')
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.reset = reset_method()

    def seeds(self, obj):
        return [obs['seed'] for obs in self.reset(obj)[0]]

    def test_default_keeps_grouped_seeds(self):
        with patch.dict(os.environ, SC2_ORIGINAL_GAME_SEEDS='0'):
            self.assertEqual(self.seeds(environment()), [2732]*8)

    def test_original_range_exact_four_batches_and_global_rng_unchanged(self):
        expected = [[49,97,53,5,33,65,62,51], [100,38,61,45,74,27,64,17],
                    [36,17,96,12,79,32,68,90], [77,18,39,12,93,9,87,42]]
        obj = environment()
        before = random.getstate()
        with patch.dict(os.environ, SC2_ORIGINAL_GAME_SEEDS='1'):
            self.assertEqual([self.seeds(obj) for _ in range(4)], expected)
        self.assertEqual(obj.reset_index, 4)
        self.assertEqual(random.getstate(), before)

    def test_reset_index_replays_same_next_games(self):
        with patch.dict(os.environ, SC2_ORIGINAL_GAME_SEEDS='1'):
            obj = environment()
            for _ in range(3): self.seeds(obj)
            self.assertEqual(self.seeds(obj), self.seeds(environment(index=3)))

    def test_evaluation_unchanged(self):
        with patch.dict(os.environ, SC2_ORIGINAL_GAME_SEEDS='1'):
            self.assertEqual(self.seeds(environment(train=False)), list(range(1000,1008)))

    def test_no_deduplication(self):
        with patch.dict(os.environ, SC2_ORIGINAL_GAME_SEEDS='1'), patch.object(random, 'Random') as rng:
            rng.return_value.randint.return_value = 42
            self.assertEqual(self.seeds(environment()), [42]*8)


if __name__ == '__main__': unittest.main()
