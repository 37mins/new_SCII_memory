"""CPU characterization of the frozen scheduler; does not install a candidate.

Run from this checkout with the existing decoder Python. The real frozen method
runs against tiny unit collections; no model, game, CUDA, or timing simulation.
"""
import ast
import asyncio
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
SHA = '9ae8668c4c6045f5834d197c72f7495c175c1c73'
PATH = 'agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/players/script_player.py'
SOURCE = subprocess.check_output(['git','show',f'{SHA}:{PATH}'],cwd=ROOT)
assert hashlib.sha256(SOURCE).hexdigest() == '89ed4006cf35af238b0962409bd5a183780d8676af5a49af35422d3077030069'
cls = next(n for n in ast.parse(SOURCE).body if isinstance(n,ast.ClassDef) and n.name=='ScriptPlayer')
node = next(n for n in cls.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='execute_auto_attack')
scope = {}
exec(compile(ast.Module(body=[node],type_ignores=[]),f'{SHA}:{PATH}','exec'),scope)
SCHEDULE = scope['execute_auto_attack']


class Units(list):
    @property
    def exists(self): return bool(self)
    @property
    def amount(self): return len(self)
    def filter(self, predicate): return Units(u for u in self if predicate(u))


def player(threshold, count):
    units = Units(SimpleNamespace(tag=i) for i in range(count))
    calls = []
    async def launch(target, selected): calls.append([u.tag for u in selected])
    return SimpleNamespace(units=units, _get_all_combat_units=lambda:units,
        active_scout_unit_tag=None,total_attack_groups={},enemy_structures=Units(),
        enemy_start_locations=[(24.5,125.5)],enemy_townhall_types=set(),
        attack_config=dict(first_attack_threshold=threshold,subsequent_attack_threshold=10,harassment_triggers=[]),
        has_launched_first_attack=False,harassment_wave_index=0,launch_total_attack=launch,calls=calls)


def tick(p):
    with contextlib.redirect_stdout(io.StringIO()): asyncio.run(SCHEDULE(p))


class BoundaryTest(unittest.TestCase):
    def test_current_scheduler_launches_without_model_attack(self):
        p=player(20,20);tick(p)
        self.assertEqual(p.calls,[list(range(20))])
        self.assertTrue(p.has_launched_first_attack)

    def test_scout_is_excluded_from_twenty_available(self):
        p=player(20,20);p.active_scout_unit_tag=0;tick(p)
        self.assertEqual(p.calls,[])
        self.assertFalse(p.has_launched_first_attack)

    def test_proposed_forty_waits_but_does_not_change_later_ten(self):
        for threshold in (20,40):
            for count in (threshold-1,threshold):
                p=player(threshold,count);tick(p)
                self.assertEqual(len(p.calls),int(count==threshold))
            for count in (9,10):
                p=player(threshold,count);p.has_launched_first_attack=True;tick(p)
                self.assertEqual(len(p.calls),int(count==10))

    def test_pretending_first_attack_happened_immediately_relaunches(self):
        p=player(20,20);p.has_launched_first_attack=True;tick(p)
        self.assertEqual(p.calls,[list(range(20))])

    def test_busy_wave_units_are_excluded(self):
        p=player(20,30);p.has_launched_first_attack=True
        p.total_attack_groups={1:dict(unit_tags=set(range(20)))};tick(p)
        self.assertEqual(p.calls,[list(range(20,30))])

    def test_retained_answers_remain_valid_and_have_no_attack(self):
        import sys
        sys.path[:0]=[str(ROOT/'tests'),str(ROOT)]
        from test_r6_format_reward import strict_validator
        from unittest.mock import patch
        fixture=json.loads((HERE/'first-attack-inputs.json').read_text())
        self.assertEqual(fixture['source_sha'],SHA)
        self.assertEqual(fixture['seed'],42613)
        for row in fixture['rows']:
            self.assertEqual(hashlib.sha256(row['prompt'].encode()).hexdigest(),row['prompt_sha256'])
            with patch.dict('os.environ',SC2_ORIGINAL_BO_SEGMENT='1'):
                self.assertTrue(strict_validator()(row['prompt'],row['answer'])[0])
            self.assertNotIn('OFFENSE',row['answer'])


if __name__=='__main__': unittest.main()
