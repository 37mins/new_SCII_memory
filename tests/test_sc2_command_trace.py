"""Pinned burnysc2 commands/protobuf with fake RPC; no SC2 game or model."""
import ast
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from sc2.bot_ai import BotAI
from sc2.client import Client
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.protocol import ProtocolError
from sc2.unit import Unit
from sc2.unit_command import UnitCommand
from sc2.units import Units
from s2clientprotocol import raw_pb2, sc2api_pb2

from agent_system.sc2_command_trace import CommandTrace
from agent_system.sc2_episode import gymnasium_transition
from test_r6_natural_termination import ARENA, ROOT, method
from test_r6_format_reward import planner_scope, strict_validator
from test_valid_bo_passthrough import CONTROLS, STRATEGY

FROZEN = 'dc1b22e96aef9046531e97bac79f4d4801a2e6cb'


def load_hooks(parent=BotAI):
    tree = ast.parse((ARENA/'players/llm_player.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LLMPlayer')
    names = {'do', 'on_start', 'on_step', 'on_end', 'on_building_construction_started',
             'on_building_construction_complete', 'on_unit_created'}
    cls.body = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    scope = dict(ScriptPlayer=parent, os=os, UnitTypeId=UnitTypeId)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), 'real-player-hooks', 'exec'), scope)
    return scope['LLMPlayer']


def unit(bot, tag, kind, position=(25, 25), progress=1):
    proto = raw_pb2.Unit(tag=tag, unit_type=kind.value, build_progress=progress,
                        alliance=raw_pb2.Self)
    proto.pos.x, proto.pos.y = position
    return Unit(proto, bot)


def bot_with_trace(directory):
    bot = load_hooks()()
    bot.state = SimpleNamespace(game_loop=100)
    bot.commander = SimpleNamespace(planner=SimpleNamespace(last_consumed_policy_action_id=1))
    bot.game_data = SimpleNamespace(abilities={
        AbilityId.TERRANBUILD_SUPPLYDEPOT.value: SimpleNamespace(_proto=SimpleNamespace(target=2)),
        AbilityId.BARRACKSTRAIN_MARINE.value: SimpleNamespace(_proto=SimpleNamespace(target=1)),
    })
    bot.client = Client.__new__(Client)
    bot.client._execute = AsyncMock(side_effect=lambda **kw: sc2api_pb2.Response(
        action=sc2api_pb2.ResponseAction(result=[1]*len(kw['action'].actions))))
    bot.test_rpc = bot.client._execute
    bot.command_trace = CommandTrace(directory, lambda: dict(
        game_loop=bot.state.game_loop,
        latest_policy_action_id=bot.commander.planner.last_consumed_policy_action_id))
    bot.command_trace.attach_client(bot.client)
    return bot


def prepare_plan(trace, answer_id=1, origin='model', parsed=('supplydepot',), final=None, valid=True):
    trace.answer('frozen query', dict(policy_action_id=answer_id, value='\n'.join(parsed),
                                     plan_format_valid=valid, trace_origin=origin))
    trace.planned(list(parsed), list(parsed if final is None else final), passthrough=valid)
    trace.queue_ready(list(parsed if final is None else final), phase_index=1)


class CommandTraceTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bot = bot_with_trace(self.temp.name)
        self.trace = self.bot.command_trace
        self.addCleanup(self.trace.stream.close)
        self.worker = unit(self.bot, 11, UnitTypeId.SCV)

    def issue_depot(self, position=(25, 25), origin=True):
        self.trace.active_bo = self.trace.bo_origin(0, True, 'supplydepot') if origin else None
        command = UnitCommand(AbilityId.TERRANBUILD_SUPPLYDEPOT, self.worker, Point2(position))
        self.assertTrue(self.bot.do(command))
        self.trace.active_bo = None
        return command

    async def send(self):
        result = await self.bot.client.actions(self.bot.actions)
        self.bot.actions.clear()
        return result

    def observe_depot_order(self, position=(25, 25), target_tag=None):
        worker = unit(self.bot, self.worker.tag, UnitTypeId.SCV)
        order = worker._proto.orders.add(ability_id=319)
        if target_tag is None:
            order.target_world_space_pos.x, order.target_world_space_pos.y = position
        else:
            order.target_unit_tag = target_tag
        self.bot.workers = Units([worker], self.bot)
        return worker

    async def test_real_command_ack_and_tag_survive_next_planner_answer(self):
        prepare_plan(self.trace)
        self.issue_depot()
        self.assertIsNone(self.trace.commands[0]['accepted'])
        self.assertEqual(await self.send(), [])
        self.assertTrue(self.trace.commands[0]['accepted'])
        self.observe_depot_order()
        depot = unit(self.bot, 101, UnitTypeId.SUPPLYDEPOT, progress=.01)
        await self.bot.on_building_construction_started(depot)
        prepare_plan(self.trace, answer_id=2, parsed=('marine',))
        self.bot.commander.planner.last_consumed_policy_action_id = 2
        await self.bot.on_building_construction_complete(unit(self.bot, 101, UnitTypeId.SUPPLYDEPOT))
        start = next(r for r in self.trace.pending if r['event'] == 'depot_started')
        complete = self.trace.pending[-1]
        self.assertEqual(start['origin']['policy_action_id'], 1)
        self.assertEqual(complete['latest_policy_action_id'], 2)
        self.assertEqual(start['command_id'], complete['command_id'])

    async def test_half_grid_match_uses_order_and_preserves_original_answer_source(self):
        prepare_plan(self.trace, origin='human_order_intervention')
        self.issue_depot(position=(28.5, 22.5))
        await self.send()
        command_id = self.trace.commands[0]['event_id']
        worker = unit(self.bot, 11, UnitTypeId.SCV, position=(28, 23))
        order = worker._proto.orders.add(ability_id=319, progress=.125)
        order.target_world_space_pos.x, order.target_world_space_pos.y = 29, 23
        self.bot.workers = Units([worker], self.bot)
        before = worker._proto.SerializeToString()
        commands = copy.deepcopy(self.trace.commands)
        prepare_plan(self.trace, answer_id=2, parsed=('marine',))
        await self.bot.on_building_construction_started(
            unit(self.bot, 101, UnitTypeId.SUPPLYDEPOT, position=(29, 23), progress=.01))
        start = self.trace.pending[-1]
        self.assertEqual(start['issuer_observations'], [dict(
            issuer_tag=11, command_ids=[command_id], observed=True, position=[28, 23],
            orders=[dict(ability_id=319, progress=.125, target_kind='target_world_space_pos',
                         target_position=[29, 23], target_tag=None)])])
        self.assertEqual(start['command_id'], command_id)
        self.assertEqual(start['origin'], commands[0]['origin'])
        commands[0].update(building_tag=101, building_position=[29, 23])
        self.assertEqual(self.trace.commands, commands)
        self.assertEqual(self.trace.commands[0]['origin']['policy_action_id'], 1)
        self.assertEqual(self.trace.commands[0]['origin']['transport_origin'], 'human_order_intervention')
        self.assertEqual(worker._proto.SerializeToString(), before)
        self.assertEqual(self.bot.actions, [])
        self.assertEqual(self.bot.test_rpc.await_count, 1)
        json.dumps(start)  # No live Unit/Point/protobuf object is left in the log.

    async def test_order_tag_zero_point_and_absent_target_are_distinct(self):
        prepare_plan(self.trace)
        self.issue_depot(position=(28.5, 22.5))
        await self.send()
        worker = unit(self.bot, 11, UnitTypeId.SCV)
        worker._proto.orders.add(ability_id=319, target_unit_tag=101)
        zero = worker._proto.orders.add(ability_id=319)
        zero.target_world_space_pos.x, zero.target_world_space_pos.y = 0, 0
        worker._proto.orders.add(ability_id=319)
        self.trace.construction_started(
            unit(self.bot, 101, UnitTypeId.SUPPLYDEPOT, position=(29, 23)), [worker])
        orders = self.trace.pending[-1]['issuer_observations'][0]['orders']
        self.assertEqual([(o['target_kind'], o['target_tag'], o['target_position']) for o in orders],
                         [('target_unit_tag', 101, None), ('target_world_space_pos', None, [0, 0]),
                          (None, None, None)])
        self.assertEqual(self.trace.pending[-1]['command_id'], self.trace.commands[0]['event_id'])

    async def test_missing_issuer_is_not_idle_and_unrelated_workers_are_not_borrowed(self):
        prepare_plan(self.trace)
        self.issue_depot(position=(28.5, 22.5))
        self.worker = unit(self.bot, 12, UnitTypeId.SCV)
        self.issue_depot(position=(30.5, 22.5))
        await self.send()
        idle = unit(self.bot, 11, UnitTypeId.SCV)
        unrelated = unit(self.bot, 99, UnitTypeId.SCV, position=(29, 23))
        unrelated._proto.orders.add(ability_id=319, target_unit_tag=101)
        self.trace.construction_started(
            unit(self.bot, 101, UnitTypeId.SUPPLYDEPOT, position=(29, 23)), [idle, unrelated])
        observed, missing = self.trace.pending[-1]['issuer_observations']
        self.assertEqual(observed['issuer_tag'], 11)
        self.assertEqual(observed['orders'], [])
        self.assertTrue(observed['observed'])
        self.assertEqual(missing['issuer_tag'], 12)
        self.assertIsNone(missing['orders'])
        self.assertIsNone(missing['position'])
        self.assertFalse(missing['observed'])
        self.assertIsNone(self.trace.pending[-1]['command_id'])

    async def test_failed_ack_is_preserved_even_if_issuer_targets_building(self):
        prepare_plan(self.trace)
        self.issue_depot(position=(28.5, 22.5))
        self.bot.test_rpc.side_effect = lambda **kw: sc2api_pb2.Response(
            action=sc2api_pb2.ResponseAction(result=[2]))
        await self.send()
        worker = unit(self.bot, 11, UnitTypeId.SCV)
        worker._proto.orders.add(ability_id=319, target_unit_tag=101)
        self.trace.construction_started(
            unit(self.bot, 101, UnitTypeId.SUPPLYDEPOT, position=(29, 23)), [worker])
        self.assertFalse(self.trace.commands[0]['accepted'])
        self.assertEqual(len(self.trace.pending[-1]['issuer_observations']), 1)
        self.assertIsNone(self.trace.pending[-1]['command_id'])

    async def test_disabled_start_hook_delegates_without_reading_workers(self):
        class Parent:
            @property
            def workers(self):
                raise AssertionError('Disabled recorder must not read worker orders')

            async def on_building_construction_started(self, depot):
                self.parent_depot = depot

        bot = load_hooks(Parent)()
        bot.command_trace = None
        depot = unit(self.bot, 101, UnitTypeId.SUPPLYDEPOT)
        await bot.on_building_construction_started(depot)
        self.assertIs(bot.parent_depot, depot)

    def test_exact_position_without_order_no_longer_bypasses_issuer_evidence(self):
        source = subprocess.check_output(['git', 'show',
            '8042871c5b2b3c4b45c0992d141c480906b10e2c:agent_system/sc2_command_trace.py'], cwd=ROOT).decode()
        scope = {}
        exec(compile(source, 'frozen-command-trace', 'exec'), scope)
        for target in ([28.5, 22.5], [29, 23]):
            results = []
            for cls in (scope['CommandTrace'], CommandTrace):
                trace = cls(self.temp.name, lambda: {'game_loop': 324})
                self.addCleanup(trace.stream.close)
                trace.commands = [dict(event_id=5, ability='TERRANBUILD_SUPPLYDEPOT', issuer_tag=11,
                                       wire_attempted=True, accepted=True, building_tag=None, completed=False,
                                       target_position=target, origin={'policy_action_id': 1})]
                depot = unit(self.bot, 101, UnitTypeId.SUPPLYDEPOT, position=(29, 23))
                if cls is CommandTrace:
                    trace.construction_started(depot, [])
                    self.assertEqual(trace.pending[-1]['issuer_observations'][0]['command_ids'], [5])
                    trace.pending[-1].pop('issuer_observations')
                else:
                    trace.construction_started(depot)
                trace.construction_completed(depot)
                results.append((trace.pending[1:], trace.commands, trace.buildings))
            self.assertIsNone(results[1][0][0]['command_id'])
            self.assertIsNone(results[1][0][1]['command_id'])
            if target == [29, 23]:
                self.assertEqual(results[0][0][0]['command_id'], 5)
            else:
                self.assertIsNone(results[0][0][0]['command_id'])

    async def test_queued_is_not_acknowledged_and_ack_is_not_completed(self):
        prepare_plan(self.trace)
        self.issue_depot()
        self.assertFalse(any(r['event'] == 'depot_completed' for r in self.trace.pending))
        self.trace.construction_started(unit(self.bot, 101, UnitTypeId.SUPPLYDEPOT), self.bot.workers)
        self.assertIsNone(self.trace.pending[-1]['command_id'])  # No wire proof yet.

    async def test_rpc_rejection_does_not_link_construction(self):
        prepare_plan(self.trace)
        self.issue_depot()
        # Preserve the installed observer; change its original AsyncMock behavior.
        self.bot.test_rpc.side_effect = lambda **kw: sc2api_pb2.Response(action=sc2api_pb2.ResponseAction(result=[2]))
        result = await self.send()
        self.assertEqual(result[0].value, 2)
        self.assertFalse(self.trace.commands[0]['accepted'])
        self.trace.construction_started(unit(self.bot, 101, UnitTypeId.SUPPLYDEPOT), self.bot.workers)
        self.assertIsNone(self.trace.pending[-1]['command_id'])

    async def test_client_swallowed_protocol_error_is_not_success(self):
        prepare_plan(self.trace)
        self.issue_depot()
        self.bot.test_rpc.side_effect = ProtocolError('synthetic not-in-game')
        self.assertEqual(await self.send(), [])  # The pinned Client itself catches this error.
        self.assertIsNone(self.trace.commands[0]['accepted'])
        self.assertEqual(self.trace.pending[-1]['error_type'], 'ProtocolError')

    async def test_two_same_position_commands_are_ambiguous_not_nearest_or_latest(self):
        prepare_plan(self.trace)
        self.issue_depot()
        prepare_plan(self.trace, answer_id=2)
        self.issue_depot()
        await self.send()
        self.observe_depot_order()
        self.trace.construction_started(unit(self.bot, 101, UnitTypeId.SUPPLYDEPOT), self.bot.workers)
        self.assertIsNone(self.trace.pending[-1]['command_id'])
        self.assertGreater(len(self.trace.pending[-1]['candidate_command_ids']), 1)

    async def test_human_rule_and_fixed_bo_are_not_relabelled_model(self):
        prepare_plan(self.trace, origin='human')
        self.issue_depot()
        await self.send()
        self.observe_depot_order()
        self.trace.construction_started(unit(self.bot, 101, UnitTypeId.SUPPLYDEPOT), self.bot.workers)
        self.assertEqual(self.trace.pending[-1]['origin']['transport_origin'], 'human')
        self.assertEqual(self.trace.bo_origin(0, False, 'supplydepot')['selection_source'], 'fixed_bo')
        prepare_plan(self.trace, parsed=(), final=('supplydepot',), valid=False)
        self.assertEqual(self.trace.queue[0]['selection_source'], 'postprocess_or_salvage')
        self.trace.queue_ready(['supplydepot', 'marine'], 1)
        self.assertEqual(self.trace.queue[1]['selection_source'], 'commander_rule')

    async def test_direct_wire_command_has_no_borrowed_model_origin(self):
        prepare_plan(self.trace)
        command = UnitCommand(AbilityId.TERRANBUILD_SUPPLYDEPOT, self.worker, Point2((25, 25)))
        await self.bot.client.actions([command])  # Bypass BotAI.do on purpose.
        self.observe_depot_order()
        self.trace.construction_started(unit(self.bot, 101, UnitTypeId.SUPPLYDEPOT), self.bot.workers)
        self.assertEqual(self.trace.pending[-1]['origin']['selection_source'], 'unobserved_local_command')

    async def test_duplicate_start_and_completion_only_without_start(self):
        prepare_plan(self.trace)
        self.issue_depot()
        await self.send()
        depot = unit(self.bot, 101, UnitTypeId.SUPPLYDEPOT)
        self.trace.construction_started(depot, self.bot.workers)
        self.trace.construction_started(depot, self.bot.workers)
        self.assertEqual(self.trace.pending[-1]['event'], 'duplicate_construction_callback')
        self.trace.construction_completed(unit(self.bot, 102, UnitTypeId.SUPPLYDEPOT))
        self.assertIsNone(self.trace.pending[-1]['command_id'])
        self.trace.construction_completed(depot)
        self.trace.construction_completed(depot)
        self.assertTrue(self.trace.pending[-1]['duplicate'])

    async def test_marine_birth_remains_unknown_even_with_one_accepted_train(self):
        prepare_plan(self.trace, parsed=('marine',))
        barracks = unit(self.bot, 21, UnitTypeId.BARRACKS)
        self.trace.active_bo = self.trace.bo_origin(0, True, 'marine')
        self.bot.do(UnitCommand(AbilityId.BARRACKSTRAIN_MARINE, barracks))
        self.trace.active_bo = None
        await self.send()
        self.bot.structures = Units([barracks], self.bot)
        await self.bot.on_unit_created(unit(self.bot, 301, UnitTypeId.MARINE))
        self.assertIsNone(self.trace.pending[-1]['command_id'])
        self.assertEqual(self.trace.pending[-1]['match_kind'], 'unresolved_birth_to_command')

    async def test_real_executor_records_final_placement_and_preserves_behavior(self):
        outcomes = []
        source = subprocess.check_output(['git', 'show', f'{FROZEN}:{(ARENA/"players/base_player.py").relative_to(ROOT)}'], cwd=ROOT).decode()
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'BasePlayer')
        old_node = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'run_actions')
        scope = dict(AbilityId=AbilityId, Point2=Point2, json=json)
        exec(compile(ast.Module(body=[old_node], type_ignores=[]), 'frozen-run-actions', 'exec'), scope)
        old_method = scope['run_actions']
        new_method = method(ARENA/'players/base_player.py', 'BasePlayer', 'run_actions', **{
            k: v for k, v in scope.items() if k != 'run_actions'})
        for fn, enabled in ((old_method, False), (new_method, False), (new_method, True)):
            bot = bot_with_trace(self.temp.name)
            trace = bot.command_trace
            self.addCleanup(trace.stream.close)
            prepare_plan(trace)
            if not enabled:
                bot.command_trace = None
            bot.planner_queue_active = True
            bot.current_bo_index = 0
            bot.bo_data = [{'action': 'supplydepot'}]
            bot.get_unit_by_id = lambda _: unit(bot, 11, UnitTypeId.SCV)
            bot.check_action = lambda _: (True, '')
            bot._get_target_base_for_building = lambda _: Point2((10, 10))
            bot.find_placement = AsyncMock(return_value=Point2((27, 29)))
            bot.calculate_cost = lambda _: SimpleNamespace(minerals=100, vespene=0)
            bot.resource_cost, bot.last_action = 0, []
            action = {'action': 'TERRANBUILD_SUPPLYDEPOT', 'units': [7], 'target_position': [90, 90]}
            with contextlib.redirect_stdout(io.StringIO()):
                await fn(bot, [action])
            await bot.client.actions(bot.actions)
            outcomes.append((copy.deepcopy(action), bot.resource_cost, bot.last_action,
                             bot.actions[0].ability, tuple(bot.actions[0].target),
                             bot.test_rpc.call_args.kwargs['action'].SerializeToString()))
            if enabled:
                self.assertEqual(trace.commands[0]['target_position'], [27, 29])
                self.assertEqual(trace.commands[0]['origin']['policy_action_id'], 1)
                self.assertIsNone(trace.active_bo)
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[1], outcomes[2])

    async def test_on_start_is_opt_in_and_shares_sidecar_without_queue_mutation(self):
        bot = self.bot
        with patch.dict(os.environ, SC2_COMMAND_TRACE_DIR=''):
            await bot.on_start()
        self.assertIs(bot.command_trace, self.trace)
        with patch.dict(os.environ, SC2_COMMAND_TRACE_DIR=self.temp.name):
            await bot.on_start()
        self.addCleanup(bot.command_trace.stream.close)
        self.assertIs(bot.command_trace, bot.commander.command_trace)
        self.assertIs(bot.command_trace, bot.commander.planner.command_trace)

    async def test_frame_exception_propagates_after_flush_and_end_closes_trace(self):
        class Parent(BotAI):
            async def on_step(self, iteration):
                raise RuntimeError('synthetic frame failure')

            async def on_end(self, result):
                self.original_end_result = result
        bot = load_hooks(Parent)()
        bot.command_trace = self.trace
        with self.assertRaisesRegex(RuntimeError, 'synthetic frame failure'):
            await bot.on_step(100)
        self.assertEqual(self.trace.pending, [])
        self.assertTrue(self.trace.path.read_text())
        await bot.on_end('test-result')
        self.assertEqual(bot.original_end_result, 'test-result')
        self.assertTrue(self.trace.stream.closed)

    async def test_short_rpc_result_list_stays_unknown(self):
        prepare_plan(self.trace)
        self.issue_depot()
        self.bot.test_rpc.side_effect = lambda **kw: sc2api_pb2.Response(action=sc2api_pb2.ResponseAction(result=[]))
        await self.send()
        self.assertIsNone(self.trace.commands[0]['accepted'])

    def test_new_trace_does_not_change_global_sampling_rng_or_overwrite_files(self):
        state = random.getstate()
        other = CommandTrace(self.temp.name, lambda: {})
        self.addCleanup(other.stream.close)
        self.assertEqual(state, random.getstate())
        self.assertNotEqual(other.path, self.trace.path)
        self.trace.flush()
        rows = [json.loads(line) for line in self.trace.path.read_text().splitlines()]
        self.assertEqual(rows[0]['event'], 'trace_start')


class PlannerTraceTest(unittest.TestCase):
    def test_optional_transport_origin_never_changes_answer_or_reward(self):
        step = method(ROOT/'agent_system/environments/env_package/sc2_online_restored/envSC2.py',
                      'SC2Env', 'step', gymnasium_transition=gymnasium_transition)
        for origin in (None, 'human', 'model'):
            sent = []
            data = dict(obs='next', done=False, reward=1.0)
            env = SimpleNamespace(reset_failed=False, game_end_flag=False, _next_policy_action_id=0,
                pre_action_terminal_drain_timeout=0, _receive_pending_from_a=lambda **kw: None,
                _send_to_a=sent.append, _receive_from_a=lambda: data,
                _mark_terminal_if_done=lambda _: None, _policy_transition=lambda *args: args)
            payload = dict(text='- supplydepot', plan_format_valid=True)
            if origin:
                payload['trace_origin'] = origin
            actual = step(env, payload)
            expected = dict(msg='ACTION', value='- supplydepot', policy_action_id=1, plan_format_valid=True)
            if origin:
                expected['trace_origin'] = origin
            self.assertEqual(sent, [expected])
            self.assertEqual(actual[0], data)

    def test_actual_planner_prompt_reward_and_plan_unchanged(self):
        path = ARENA/'agents/planner.py'
        old = ast.parse(subprocess.check_output(['git', 'show', f'{FROZEN}:{path.relative_to(ROOT)}'], cwd=ROOT).decode())
        old.body = [n for n in old.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        base = planner_scope()
        scope = dict(base)
        exec(compile(old, 'frozen-planner', 'exec'), scope)
        for answer in ('- supplydepot x 2', '- marine x 21', '- not_a_unit', '- supplydepot\n- barracks'):
            outputs = []
            for cls, enabled in ((scope['SC2Planner'], False), (base['SC2Planner'], False), (base['SC2Planner'], True)):
                with tempfile.TemporaryDirectory() as directory:
                    trace = CommandTrace(directory, lambda: {})
                    try:
                        p = cls.__new__(cls)
                        p.ct, p.correct_format, p.exceed, p.last_consumed_policy_action_id = 0, False, False, None
                        p.command_trace = trace if enabled else None
                        sent = []
                        p._send = sent.append
                        p._recv = lambda: dict(msg='ACTION', value=answer, policy_action_id=42,
                            plan_format_valid=strict_validator()(sent[-1]['obs'], answer)[0])
                        controls = dict(CONTROLS, SC2_ORIGINAL_BO_SEGMENT='1', SC2_PRESERVE_VALID_BO='1',
                                        SC2_CLARIFY_BO_QUOTA='1')
                        with patch.dict(os.environ, controls), contextlib.redirect_stdout(io.StringIO()):
                            result = p.generate_plan('It is now 04:00 in the game. We have 4 barracks and 1 refinery.',
                                                     {'marine': 42, 'supplydepot': 4, 'barracks': 2}, STRATEGY, .25)
                        outputs.append((sent, result, p.correct_format, p.exceed))
                        if enabled:
                            self.assertEqual(trace.transport_origin, 'unverified_transport')
                            self.assertEqual(trace.plan[0]['policy_action_id'], 42)
                    finally:
                        trace.stream.close()
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual(outputs[1], outputs[2])


if __name__ == '__main__':
    unittest.main()
