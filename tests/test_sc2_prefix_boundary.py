import ast
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
tree = ast.parse((ROOT / 'experiments/records/20260906-integrated-fixes-v1/r4_rng_prefix.py').read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'record_first_planner')
scope = {'json': json}
exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual boundary recorder>', 'exec'), scope)


class BoundaryTest(unittest.TestCase):
    def test_request_recording_preserves_payload_bo_and_random_state(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            send = Mock(return_value='sent')
            planner = SimpleNamespace(_send=send, ct=1)
            player = SimpleNamespace(commander=SimpleNamespace(planner=planner),
                trace={'9': {'minerals': 150}}, state=SimpleNamespace(game_loop=400),
                time=17.85, time_formatted='00:17', current_bo_index=237,
                bo_data=[{'action': 'scv'}] * 237)
            rng = random.getstate()
            scope['record_first_planner'](player, output)
            data = dict(done=False, obs='original query', last_consumed_policy_action_id=None)
            self.assertEqual(planner._send(data), 'sent')
            self.assertIs(send.call_args.args[0], data)
            self.assertEqual(random.getstate(), rng)
            self.assertEqual((player.current_bo_index, len(player.bo_data)), (237, 237))
            self.assertEqual(json.loads((output/'boundary_trace.json').read_text()), player.trace)
            self.assertEqual(json.loads((output/'first_planner.json').read_text())['request'], data)

    def test_terminal_is_forwarded_without_inventing_request(self):
        with tempfile.TemporaryDirectory() as folder:
            planner = SimpleNamespace(_send=Mock())
            player = SimpleNamespace(commander=SimpleNamespace(planner=planner))
            send = planner._send
            scope['record_first_planner'](player, Path(folder))
            data = {'done': True}
            planner._send(data)
            send.assert_called_once_with(data)
            self.assertEqual(list(Path(folder).iterdir()), [])
