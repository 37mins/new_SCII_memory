"""Opt-in Depot/Marine provenance, observing commands without selecting actions."""
import hashlib
import json
import os
from pathlib import Path
import uuid

from sc2.ids.ability_id import AbilityId

ABILITIES = {a.value: a.name for a in
             (AbilityId.TERRANBUILD_SUPPLYDEPOT, AbilityId.BARRACKSTRAIN_MARINE)}


class CommandTrace:
    def __init__(self, directory, state):
        self.episode_id = f"{os.getpid()}-{uuid.uuid4().hex}"
        self.path = Path(directory) / f"game-{self.episode_id}-command-trace.jsonl"
        self.stream = self.path.open('x')
        self.state = state
        self.pending = []
        self.sequence = 0
        self.answer_id = None
        self.transport_origin = 'unverified_transport'
        self.plan = []
        self.queue = []
        self.active_bo = None
        self.commands = []
        self.buildings = {}
        self.completed_tags = set()
        self.emit('trace_start', game_pid=os.getpid(), run_id=os.getenv('RUN_ID'),
                  episode_id=self.episode_id)

    def emit(self, event, **fields):
        self.sequence += 1
        row = dict(event_id=self.sequence, event=event, **self.state(), **fields)
        self.pending.append(row)
        return row

    def flush(self):
        for row in self.pending:
            self.stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        self.stream.flush()
        self.pending.clear()

    def answer(self, query, response):
        self.answer_id = response['policy_action_id']
        self.transport_origin = response.get('trace_origin', 'unverified_transport')
        self.emit('received_answer', policy_action_id=self.answer_id,
                  transport_origin=self.transport_origin,
                  prompt_sha256=hashlib.sha256(query.encode()).hexdigest(),
                  answer=response['value'], valid=bool(response.get('plan_format_valid', False)))

    def planned(self, parsed, final, passthrough):
        remaining = list(enumerate(parsed))
        self.plan = []
        for action in final:
            match = next((p for p in remaining if p[1] == action), None)
            if passthrough and match is not None:
                remaining.remove(match)
                source, parsed_index = 'validated_answer', match[0]
            else:
                source, parsed_index = 'postprocess_or_salvage', None
            self.plan.append(dict(action=action, selection_source=source, parsed_index=parsed_index,
                                  policy_action_id=self.answer_id, transport_origin=self.transport_origin))
        self.emit('parsed_plan', policy_action_id=self.answer_id, parsed=list(parsed),
                  final=list(final), origins=[dict(item) for item in self.plan])

    def queue_ready(self, queue, phase_index):
        remaining = list(self.plan)
        self.queue = []
        for index, action in enumerate(queue):
            match = next((p for p in remaining if p['action'] == action), None)
            if match is not None:
                remaining.remove(match)
                item = dict(match)
            else:
                # Commander may insert its opening Marine after Planner returns.
                item = dict(action=action, selection_source='commander_rule', parsed_index=None,
                            policy_action_id=self.answer_id, transport_origin=self.transport_origin)
            self.queue.append(dict(item, queue_index=index, phase_index=phase_index))
        self.emit('executor_queue', origins=[dict(item) for item in self.queue])

    def bo_origin(self, index, planner_active, action):
        if not planner_active:
            return dict(action=action, queue_index=index, selection_source='fixed_bo',
                        policy_action_id=None, transport_origin=None)
        if index < len(self.queue) and self.queue[index]['action'] == action:
            return dict(self.queue[index])
        return dict(action=action, queue_index=index, selection_source='unknown_queue',
                    policy_action_id=None, transport_origin=None)

    def command(self, command, queued):
        if command.ability.name not in ('TERRANBUILD_SUPPLYDEPOT', 'BARRACKSTRAIN_MARINE'):
            return
        target = command.target
        position = list(target) if isinstance(target, tuple) else None
        target_tag = getattr(target, 'tag', None)
        row = self.emit('local_command', ability=command.ability.name, ability_id=command.ability.value,
                        issuer_tag=command.unit.tag, target_position=position, target_tag=target_tag,
                        queue=command.queue, locally_queued=queued,
                        origin=dict(self.active_bo) if self.active_bo is not None else
                        dict(selection_source='system_or_unscoped', policy_action_id=None),
                        orders_before=[dict(ability=o.ability.id.value, progress=o.progress)
                                       for o in command.unit.orders])
        # Mutable diagnostic state is separate from the immutable emitted event.
        self.commands.append(dict(row, wire_attempted=False, accepted=None, building_tag=None, completed=False))

    def wire_request(self, request):
        groups = []
        for index, action in enumerate(request.actions):
            if not action.HasField('action_raw') or not action.action_raw.HasField('unit_command'):
                continue
            command = action.action_raw.unit_command
            if command.ability_id not in ABILITIES:
                continue
            position = ([command.target_world_space_pos.x, command.target_world_space_pos.y]
                        if command.HasField('target_world_space_pos') else None)
            target_tag = command.target_unit_tag if command.HasField('target_unit_tag') else None
            matches = [r for r in self.commands if r['locally_queued'] and not r['wire_attempted']
                       and r['ability_id'] == command.ability_id and r['issuer_tag'] in command.unit_tags
                       and r['target_position'] == position and r['target_tag'] == target_tag
                       and r['queue'] == command.queue_command]
            unique = all(sum(r['issuer_tag'] == tag for r in matches) <= 1 for tag in command.unit_tags)
            for tag in command.unit_tags:
                if not any(r['issuer_tag'] == tag for r in matches):
                    # Direct client calls can bypass BotAI.do; do not borrow an old BO's origin.
                    row = self.emit('unscoped_wire_command', ability=ABILITIES[command.ability_id],
                                    ability_id=command.ability_id, issuer_tag=tag,
                                    target_position=position, target_tag=target_tag, queue=command.queue_command,
                                    origin=dict(selection_source='unobserved_local_command', policy_action_id=None))
                    row = dict(row, wire_attempted=False, accepted=None, building_tag=None, completed=False,
                               locally_queued=False)
                    self.commands.append(row)
                    matches.append(row)
            for row in matches:
                row['wire_attempted'] = True
            groups.append(dict(action_index=index, command_ids=[r['event_id'] for r in matches],
                               unit_tags=list(command.unit_tags), ability_id=command.ability_id,
                               local_mapping_unique=unique))
        if not groups:
            return None
        return self.emit('wire_request', action_count=len(request.actions), groups=groups)

    def wire_result(self, batch, response=None, error=None):
        codes = list(response.action.result) if response is not None else []
        aligned = error is None and len(codes) == batch['action_count']
        self.emit('wire_result', request_event_id=batch['event_id'], result_codes=codes,
                  results_aligned=aligned, error_type=error)
        by_id = {r['event_id']: r for r in self.commands}
        for group in batch['groups']:
            # SC2 ActionResult.Success == 1. A group ACK is not unit completion.
            accepted = (codes[group['action_index']] == 1
                        if aligned and group['local_mapping_unique'] else None)
            for command_id in group['command_ids']:
                by_id[command_id]['accepted'] = accepted

    def attach_client(self, client):
        original = client._execute

        async def observed(*args, **kwargs):
            batch = self.wire_request(kwargs['action']) if 'action' in kwargs else None
            try:
                response = await original(*args, **kwargs)
            except Exception as error:
                if batch is not None:
                    self.wire_result(batch, error=type(error).__name__)
                raise
            if batch is not None:
                self.wire_result(batch, response=response)
            return response

        client._execute = observed

    def construction_started(self, unit, workers):
        if unit.type_id.name != 'SUPPLYDEPOT':
            return
        if unit.tag in self.buildings:
            self.emit('duplicate_construction_callback', tag=unit.tag)
            return
        pending_issuers = {}
        for command in self.commands:
            if command['ability'] == 'TERRANBUILD_SUPPLYDEPOT' and command['wire_attempted']:
                pending_issuers.setdefault(command['issuer_tag'], []).append(command['event_id'])
        observed_workers = {worker.tag: worker for worker in workers}
        issuer_observations = []
        for tag, command_ids in pending_issuers.items():
            worker = observed_workers.get(tag)
            snapshot = dict(issuer_tag=tag, command_ids=command_ids, observed=worker is not None,
                            position=None, orders=None)
            if worker is not None:
                # Read the pinned SDK's raw orders to preserve absent vs point vs tag targets.
                snapshot['position'] = list(worker.position)
                snapshot['orders'] = [dict(
                    ability_id=o.ability_id, progress=o.progress, target_kind=o.WhichOneof('target'),
                    target_position=([o.target_world_space_pos.x, o.target_world_space_pos.y]
                                     if o.HasField('target_world_space_pos') else None),
                    target_tag=o.target_unit_tag if o.HasField('target_unit_tag') else None,
                ) for o in worker._proto.orders]
            issuer_observations.append(snapshot)
        # Orders identify the observed builder; a unique unfinished command identifies its source.
        # Retire only verified completions, not merely ACKed or already-started commands.
        issuers = [s['issuer_tag'] for s in issuer_observations if s['observed'] and any(
            o['ability_id'] == AbilityId.TERRANBUILD_SUPPLYDEPOT.value and (
                o['target_kind'] == 'target_world_space_pos' and o['target_position'] == list(unit.position)
                or o['target_kind'] == 'target_unit_tag' and o['target_tag'] == unit.tag
            ) for o in s['orders'])]
        candidates = [r for r in self.commands if len(issuers) == 1
                      and r['ability'] == 'TERRANBUILD_SUPPLYDEPOT' and r['wire_attempted']
                      and r['issuer_tag'] == issuers[0] and not r['completed']]
        match = (candidates[0] if len(candidates) == 1 and candidates[0]['accepted'] is True
                 and candidates[0]['building_tag'] is None else None)
        if match is not None:
            match['building_tag'] = unit.tag
            match['building_position'] = list(unit.position)
        command_id = match['event_id'] if match is not None else None
        self.buildings[unit.tag] = command_id
        self.emit('depot_started', tag=unit.tag, position=list(unit.position),
                  command_id=command_id, candidate_command_ids=[r['event_id'] for r in candidates],
                  match_kind='unique_accepted_issuer_order_after_completed_history' if match else 'unresolved',
                  origin=dict(match['origin']) if match else None,
                  issuer_observations=issuer_observations)

    def construction_completed(self, unit):
        if unit.type_id.name not in ('SUPPLYDEPOT', 'SUPPLYDEPOTLOWERED'):
            return
        self.emit('depot_completed', tag=unit.tag, command_id=self.buildings.get(unit.tag),
                  position=list(unit.position), progress=unit.build_progress,
                  duplicate=unit.tag in self.completed_tags)
        command_id = self.buildings.get(unit.tag)
        if command_id is not None and unit.build_progress == 1 and unit.tag not in self.completed_tags:
            command = next(r for r in self.commands if r['event_id'] == command_id)
            if list(unit.position) == command['building_position']:
                command['completed'] = True
        self.completed_tags.add(unit.tag)

    def marine_created(self, unit, barracks):
        if unit.type_id.name != 'MARINE':
            return
        self.emit('marine_created', tag=unit.tag, position=list(unit.position), command_id=None,
                  match_kind='unresolved_birth_to_command',
                  barracks=[dict(tag=b.tag, position=list(b.position),
                    orders=[dict(ability=o.ability.id.value, progress=o.progress) for o in b.orders])
                    for b in barracks])
