import sys
import os
import csv
import json
import hashlib
import math
import random
from collections import defaultdict

# ==========================================
# 动态添加项目根目录到 sys.path
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../tools
project_root = os.path.dirname(current_dir)              # .../ (项目根目录)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ==========================================

from .script_player import ScriptPlayer
from agents.commander import SC2Commander

from sc2.unit import Unit
from sc2.units import Units
# from agents import AdjustAgent, MicroAgent, WorldModelAgent, TopAgent
from agents import AdjustAgent, MicroAgent, WorldModelAgent
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.ids.buff_id import BuffId
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
from typing import Dict, Any, Set, List, Optional
from tools.bo_reader import load_bo_data, check_and_get_bo_hints

from sc2.dicts.unit_train_build_abilities import TRAIN_INFO
from sc2.dicts.unit_research_abilities import RESEARCH_INFO
from sc2.dicts.unit_trained_from import UNIT_TRAINED_FROM
from sc2.dicts.upgrade_researched_from import UPGRADE_RESEARCHED_FROM
from sc2.dicts.unit_tech_alias import UNIT_TECH_ALIAS

class LLMPlayer(ScriptPlayer):
    def __init__(self, config, bo_address, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        agent_config = {
            "model_name": self.model_name,
            "generation_config": self.generation_config,
            "llm_client": self.llm_client,
        }

        self.adjust_agent = AdjustAgent(log_dir=self.log_path, **agent_config)
        self.micro_agent = MicroAgent(**agent_config)
        self.world_model_agent = WorldModelAgent(**agent_config)
        self.commander = SC2Commander(port=config.port)
        self.command_trace = None

        # [新增] 实例化 TopAgent
        # self.top_agent = TopAgent(**agent_config)

        self.plan_verifier = "llm" if config.enable_plan_verifier else None
        self.action_verifier = self.verify_actions if getattr(self.config, 'enable_action_verifier', False) else None

        self.next_decision_time = -1
        self.flag_test = True

        self.bo_address = bo_address
        self.bo_data = load_bo_data(self.bo_address)
        prefix_limit = int(os.getenv("SC2_FIXED_BO_PREFIX_LIMIT", "0"))
        if prefix_limit:
            if prefix_limit not in (-1, 138):
                raise ValueError("SC2_FIXED_BO_PREFIX_LIMIT supports only 0 (full BO), -1 (Planner from start), or 138 (g2 handoff)")
            with open(self.bo_address, "rb") as stream:
                bo_sha256 = hashlib.sha256(stream.read()).hexdigest()
            if bo_sha256 != "304b1f4ff12fe30c5e6d5f7b8eaef8ed6823ca65afed3a3706cae10e3639c494":
                raise ValueError("g2 handoff requires the frozen canonical g2 BO")
            retained_count = max(0, prefix_limit)
            print("Fixed BO handoff: " + json.dumps({
                "bo_sha256": bo_sha256,
                "retained_prefix_count": retained_count,
                "withheld_indices_half_open": [retained_count, len(self.bo_data)],
            }))
            # Only the initial queue is bounded; renew_bo still owns every later plan.
            self.bo_data = self.bo_data[:retained_count]
        self.delay_time = []

        self.notified_bo_indices = set()
        self.current_bo_index = 0
        self.current_bo_parsed_state = None
        self.execution_feedback_enabled = os.getenv("SC2_EXECUTION_FAILURE_FEEDBACK", "0") == "1"
        self.planner_queue_active = False
        self.pending_execution_feedback = None
        self.last_bo_block_reason = None
        self.current_bo_pre_move_attempted = False

        self.ability_descriptions_text = self._load_ability_data()
        self.ability_requirements_map = self._build_ability_requirements()

        # [新增] 维护当前生效的策略参数，初始化为默认值
        # self.strategy_params = {
        #     "first_attack_threshold": 40,
        #     "subsequent_attack_threshold": 15,
        #     "harassment_triggers": [[3, 3], [5, 5]], # 👈 这里改成方括号
        #     "enable_early_scv_defense": False,
        #     "early_scv_defense_threshold": 5
        # }

        self.strategy_params = {
                    "first_attack_threshold": 20,
                    "subsequent_attack_threshold": 10,
                    "harassment_triggers": [],
                    "enable_early_scv_defense": False,
                    "early_scv_defense_threshold": 5
                }

        # ==========================================
        # [新增] 用于控制 TopAgent 触发机制的变量
        # ==========================================
        self.enemy_base_scouted = False           # 标记是否已经首次侦察到敌方基地
        self.enemy_base_scout_time = -1           # 记录首次侦察到敌方基地的游戏时间（秒）
        # self.top_agent_triggered = False          # 标记 TopAgent 是否已经完成了一次触发
        self.all_scouting_history = []            # 用于累积保存全局所有的侦察信息

        self.history_research = []
        self.research_table = {
            'ARMORYRESEARCH_TERRANSHIPPLATINGLEVEL1': 'terranshipplatinglevel1',
            'ARMORYRESEARCH_TERRANSHIPPLATINGLEVEL2': 'terranshipplatinglevel2',
            'ARMORYRESEARCH_TERRANSHIPPLATINGLEVEL3': 'terranshipplatinglevel3',
            'ARMORYRESEARCH_TERRANSHIPWEAPONSLEVEL1': 'terranshipweaponslevel1',
            'ARMORYRESEARCH_TERRANSHIPWEAPONSLEVEL2': 'terranshipweaponslevel2',
            'ARMORYRESEARCH_TERRANSHIPWEAPONSLEVEL3': 'terranshipweaponslevel3',
            'ARMORYRESEARCH_TERRANVEHICLEPLATINGLEVEL1': 'terranvehicleplatinglevel1',
            'ARMORYRESEARCH_TERRANVEHICLEPLATINGLEVEL2': 'terranvehicleplatinglevel2',
            'ARMORYRESEARCH_TERRANVEHICLEPLATINGLEVEL3': 'terranvehicleplatinglevel3',
            'ARMORYRESEARCH_TERRANVEHICLEWEAPONSLEVEL1': 'terranvehicleweaponslevel1',
            'ARMORYRESEARCH_TERRANVEHICLEWEAPONSLEVEL2': 'terranvehicleweaponslevel2',
            'ARMORYRESEARCH_TERRANVEHICLEWEAPONSLEVEL3': 'terranvehicleweaponslevel3',
            'ARMORYRESEARCH_TERRANVEHICLEANDSHIPPLATINGLEVEL1': 'terranvehicleandshipplatinglevel1',
            'ARMORYRESEARCH_TERRANVEHICLEANDSHIPPLATINGLEVEL2': 'terranvehicleandshipplatinglevel2',
            'ARMORYRESEARCH_TERRANVEHICLEANDSHIPPLATINGLEVEL3': 'terranvehicleandshipplatinglevel3',
            'BARRACKSTECHLABRESEARCH_STIMPACK': 'stimpack',
            'ENGINEERINGBAYRESEARCH_TERRANINFANTRYARMORLEVEL1': 'terraninfantryarmorlevel1',
            'ENGINEERINGBAYRESEARCH_TERRANINFANTRYARMORLEVEL2': 'terraninfantryarmorlevel2',
            'ENGINEERINGBAYRESEARCH_TERRANINFANTRYARMORLEVEL3': 'terraninfantryarmorlevel3',
            'ENGINEERINGBAYRESEARCH_TERRANINFANTRYWEAPONSLEVEL1': 'terraninfantryweaponslevel1',
            'ENGINEERINGBAYRESEARCH_TERRANINFANTRYWEAPONSLEVEL2': 'terraninfantryweaponslevel2',
            'ENGINEERINGBAYRESEARCH_TERRANINFANTRYWEAPONSLEVEL3': 'terraninfantryweaponslevel3',
            'RESEARCH_SMARTSERVOS': 'smartservos',
            'RESEARCH_CYCLONELOCKONDAMAGE': 'cyclonelockondamage',
            'RESEARCH_BANSHEECLOAKINGFIELD': 'bansheecloakingfield',
            'RESEARCH_BANSHEEHYPERFLIGHTROTORS': 'bansheehyperflightrotors',
            'RESEARCH_BATTLECRUISERWEAPONREFIT': 'battlecruiserweaponrefit',
            'RESEARCH_COMBATSHIELD': 'combatshield',
            'RESEARCH_CONCUSSIVESHELLS': 'concussiveshells',
            'RESEARCH_DRILLINGCLAWS': 'drillingclaws',
            'RESEARCH_HISECAUTOTRACKING': 'hisecautotracking',
            'RESEARCH_INFERNALPREIGNITER': 'infernalpreigniter',
            'RESEARCH_PERSONALCLOAKING': 'personalcloaking',
            'RESEARCH_RAVENCORVIDREACTOR': 'ravencorvidreactor',
            'RESEARCH_TERRANSTRUCTUREARMORUPGRADE': 'terranstructurearmorupgrade',
            'STARPORTTECHLABRESEARCH_RESEARCHDURABLEMATERIALS': 'researchdurablematerials',
            'STARPORTTECHLABRESEARCH_RESEARCHMEDIVACENERGYUPGRADE': 'researchmedivacenergyupgrade',
            'STARPORTTECHLABRESEARCH_RESEARCHRAVENINTERFERENCEMATRIX': 'researchraveninterferencematrix',
            'FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL1': 'protossgroundweaponslevel1',
            'FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL2': 'protossgroundweaponslevel2',
            'FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL3': 'protossgroundweaponslevel3',
            'FORGERESEARCH_PROTOSSGROUNDARMORLEVEL1': 'protossgroundarmorlevel1',
            'FORGERESEARCH_PROTOSSGROUNDARMORLEVEL2': 'protossgroundarmorlevel2',
            'FORGERESEARCH_PROTOSSGROUNDARMORLEVEL3': 'protossgroundarmorlevel3',
            'FORGERESEARCH_PROTOSSSHIELDSLEVEL1': 'protossshieldslevel1',
            'FORGERESEARCH_PROTOSSSHIELDSLEVEL2': 'protossshieldslevel2',
            'FORGERESEARCH_PROTOSSSHIELDSLEVEL3': 'protossshieldslevel3',
            'RESEARCH_ZERGMELEEWEAPONSLEVEL1': 'zergmeleeweaponslevel1',
            'RESEARCH_ZERGMELEEWEAPONSLEVEL2': 'zergmeleeweaponslevel2',
            'RESEARCH_ZERGMELEEWEAPONSLEVEL3': 'zergmeleeweaponslevel3',
            'RESEARCH_ZERGGROUNDARMORLEVEL1': 'zerggroundarmorlevel1',
            'RESEARCH_ZERGGROUNDARMORLEVEL2': 'zerggroundarmorlevel2',
            'RESEARCH_ZERGGROUNDARMORLEVEL3': 'zerggroundarmorlevel3',
            'RESEARCH_ZERGMISSILEWEAPONSLEVEL1': 'zergmissileweaponslevel1',
            'RESEARCH_ZERGMISSILEWEAPONSLEVEL2': 'zergmissileweaponslevel2',
            'RESEARCH_ZERGMISSILEWEAPONSLEVEL3': 'zergmissileweaponslevel3',
            'RESEARCH_ZERGFLYERATTACKLEVEL1': 'zergflyerattacklevel1',
            'RESEARCH_ZERGFLYERATTACKLEVEL2': 'zergflyerattacklevel2',
            'RESEARCH_ZERGFLYERATTACKLEVEL3': 'zergflyerattacklevel3',
            'RESEARCH_ZERGFLYERARMORLEVEL1': 'zergflyerarmorlevel1',
            'RESEARCH_ZERGFLYERARMORLEVEL2': 'zergflyerarmorlevel2',
            'RESEARCH_ZERGFLYERARMORLEVEL3': 'zergflyerarmorlevel3',
            'RESEARCH_ADAPTIVETALONS': 'adaptivetalons',
            'RESEARCH_ANABOLICSYNTHESIS': 'anabolicsynthesis',
            'RESEARCH_CENTRIFUGALHOOKS': 'centrifugalhooks',
            'RESEARCH_GLIALREGENERATION': 'glialregeneration',
            'RESEARCH_ZERGLINGMETABOLICBOOST': 'zerglingmetabolicboost',
            'RESEARCH_PNEUMATIZEDCARAPACE': 'pneumatizedcarapace',
            'RESEARCH_MUSCULARAUGMENTS': 'muscularaugments',
            'RESEARCH_GROOVEDSPINES': 'groovedspines',
            'RESEARCH_NEURALPARASITE': 'neuralparasite',
            'RESEARCH_CHITINOUSPLATING': 'chitinousplating',
            'CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL1': 'protossairweaponslevel1',
            'CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL2': 'protossairweaponslevel2',
            'CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL3': 'protossairweaponslevel3',
            'CYBERNETICSCORERESEARCH_PROTOSSAIRARMORLEVEL1': 'protossairarmorlevel1',
            'CYBERNETICSCORERESEARCH_PROTOSSAIRARMORLEVEL2': 'protossairarmorlevel2',
            'CYBERNETICSCORERESEARCH_PROTOSSAIRARMORLEVEL3': 'protossairarmorlevel3',
            'RESEARCH_WARPGATE': 'warpgate',
            'RESEARCH_ADEPTRESONATINGGLAIVES': 'adeptresonatingglaives',
            'RESEARCH_BURROW': 'burrow',
            'RESEARCH_TUNNELINGCLAWS': 'tunnelingclaws',
            'RESEARCH_CHARGE': 'charge',
            'RESEARCH_BLINK': 'blink',
            'RESEARCH_ADVANCEDBALLISTICS': 'advancedballistics',
        }

    async def on_start(self):
        await super().on_start()
        combat_scale = os.getenv("SC2_COMBAT_REWARD_SCALE")
        if combat_scale is not None:
            from agent_system.sc2_combat_reward import CombatResultReward
            self.commander.planner.combat_result_reward = CombatResultReward(float(combat_scale))
            self.commander.planner.combat_score = lambda: (self.state.score, self.state.game_loop)
        directory = os.getenv("SC2_COMMAND_TRACE_DIR")
        if directory:
            from agent_system.sc2_command_trace import CommandTrace
            self.command_trace = CommandTrace(directory, lambda: dict(
                game_loop=int(self.state.game_loop), time_seconds=self.time,
                supply_cap=self.supply_cap, supply_used=self.supply_used,
                minerals=self.minerals, vespene=self.vespene,
                latest_policy_action_id=self.commander.planner.last_consumed_policy_action_id,
            ))
            self.commander.command_trace = self.command_trace
            self.commander.planner.command_trace = self.command_trace
            self.command_trace.attach_client(self.client)

    def do(self, action, *args, **kwargs):
        before = len(self.actions)
        result = super().do(action, *args, **kwargs)
        if self.command_trace is not None and not isinstance(action, bool):
            self.command_trace.command(action, queued=any(item is action for item in self.actions[before:]))
        return result

    async def on_step(self, iteration):
        try:
            return await super().on_step(iteration)
        finally:
            if self.command_trace is not None:
                self.command_trace.flush()

    async def on_end(self, game_result):
        try:
            return await super().on_end(game_result)
        finally:
            if self.command_trace is not None:
                self.command_trace.flush()
                self.command_trace.stream.close()

    async def on_building_construction_started(self, unit):
        await super().on_building_construction_started(unit)
        if self.command_trace is not None:
            self.command_trace.construction_started(unit, self.workers)

    async def on_building_construction_complete(self, unit):
        await super().on_building_construction_complete(unit)
        if self.command_trace is not None:
            self.command_trace.construction_completed(unit)

    async def on_unit_created(self, unit):
        await super().on_unit_created(unit)
        if self.command_trace is not None:
            self.command_trace.marine_created(unit, self.structures(UnitTypeId.BARRACKS))

    def _looks_like_attack_bo(self, bo_command: str) -> bool:
        lowered = str(bo_command).strip().lower()
        if not lowered or "no attack" in lowered or "do not attack" in lowered:
            return False
        attack_markers = (
            "attacking instruction",
            "launch offense",
            "attack with",
            "attack enemy",
            "attack the enemy",
        )
        return lowered.startswith("attack ") or any(marker in lowered for marker in attack_markers)


    def log_current_iteration(self, iteration: int):
        print(f"================ iteration {iteration} ================")
        self.logging("iteration", iteration, save_trace=True)
        self.logging("time_seconds", int(self.time), save_trace=True)
        self.logging("time_formatted", self.time_formatted, save_trace=True)
        self.logging("minerals", self.minerals, save_trace=True)
        self.logging("vespene", self.vespene, save_trace=True)

        self.logging("cumulative_minerals_collected", self.state.score.collected_minerals, save_trace=True)
        self.logging("cumulative_vespene_collected", self.state.score.collected_vespene, save_trace=True)

        unit_mineral_value, unit_vespene_value = 0, 0
        for unit in self.units:
            unit_value = self.calculate_unit_value(unit.type_id)
            unit_mineral_value += unit_value.minerals
            unit_vespene_value += unit_value.vespene
        self.logging("unit_mineral_value", unit_mineral_value, save_trace=True)
        self.logging("unit_vespene_value", unit_vespene_value, save_trace=True)

        structure_mineral_value, structure_vespene_value = 0, 0
        for structure in self.structures:
            structure_value = self.calculate_unit_value(structure.type_id)
            structure_mineral_value += structure_value.minerals
            structure_vespene_value += structure_value.vespene
        self.logging("structure_mineral_value", structure_mineral_value, save_trace=True)
        self.logging("structure_vespene_value", structure_vespene_value, save_trace=True)

        self.logging("cumulative_unit_value", self.state.score.total_value_units, save_trace=True)
        self.logging("cumulative_structure_value", self.state.score.total_value_structures, save_trace=True)

        self.logging("supply_army", self.supply_army, save_trace=True)
        self.logging("supply_workers", self.supply_workers, save_trace=True)
        self.logging("supply_left", self.supply_left, save_trace=True)
        self.logging("n_structures", len(self.structures), save_trace=True)
        self.logging("n_visible_enemy_units", len(self.enemy_units), save_trace=True)
        self.logging("n_visible_enemy_structures", len(self.enemy_structures), save_trace=True)
        unit_types = set(unit.type_id for unit in self.units)
        structure_types = set(unit.type_id for unit in self.structures)
        self.logging("n_unit_types", len(unit_types), save_trace=True)
        self.logging("n_structure_types", len(structure_types), save_trace=True)

    def _load_ability_data(self):
        csv_path = os.path.join(project_root, 'knowledge', 'NEW_ability.csv')
        if not os.path.exists(csv_path):
            print(f"Error: CSV file not found at {csv_path}")
            return ""

        abilities_text = ""
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row = {k.strip(): v for k, v in row.items()}
                    if 'enabled' in row and str(row['enabled']).strip() == '0':
                        continue
                    name = row.get('ability', '').strip()
                    desc = row.get('description', '').strip()
                    if name:
                        abilities_text += f"Ability: {name}, Description: {desc}\n"
        except Exception as e:
            print(f"Error loading ability CSV: {e}")
        return abilities_text.strip()

    def _build_ability_requirements(self):
        req_map = {}
        for producer, products in TRAIN_INFO.items():
            for product, info in products.items():
                ability = info.get('ability')
                if ability:
                    req_map[ability] = {
                        'producer': producer,
                        'required_building': info.get('required_building'),
                        'requires_techlab': info.get('requires_techlab', False),
                        'product': product,
                        'type': 'train'
                    }
        for producer, upgrades in RESEARCH_INFO.items():
            for upgrade, info in upgrades.items():
                ability = info.get('ability')
                if ability:
                    req_map[ability] = {
                        'producer': producer,
                        'required_building': info.get('required_building'),
                        'required_upgrade': info.get('required_upgrade'),
                        'product': upgrade,
                        'type': 'research'
                    }
        return req_map

    def _has_structure_or_alias(self, unit_type: UnitTypeId) -> bool:
        if self.structures(unit_type).ready.exists:
            return True
        for structure in self.structures.ready:
            aliases = UNIT_TECH_ALIAS.get(structure.type_id, set())
            if unit_type in aliases:
                return True
        return False

    def _validate_resources_and_supply(self, ability_enum, req_info) -> str:
        product = req_info.get('product') if req_info else None
        if product:
            cost = self.calculate_cost(product)
        else:
            cost = self.game_data.calculate_ability_cost(ability_enum)

        minerals_needed = cost.minerals
        gas_needed = cost.vespene
        supply_cost = 0
        if isinstance(product, UnitTypeId):
            supply_cost = self.calculate_supply_cost(product)

        if supply_cost > 0 and self.supply_left < supply_cost:
            supply_provider = "SupplyDepot"
            if self.config.own_race == "Protoss": supply_provider = "Pylon"
            elif self.config.own_race == "Zerg": supply_provider = "Overlord"
            return (f"**Action Failed**: Supply blocked (Available: {self.supply_left} < Required: {supply_cost}).\n"
                    f"**Suggestion**: Construct a {supply_provider} immediately.")

        if self.vespene < gas_needed:
             gas_provider = "Refinery"
             return (f"**Action Failed**: Insufficient Vespene Gas ({self.vespene}/{gas_needed}).\n"
                      f"**Suggestion**: Build a {gas_provider}.")

        if self.minerals < minerals_needed:
            return (f"**Action Failed**: Insufficient Minerals ({self.minerals}/{minerals_needed}).\n"
                    f"**Suggestion**: Wait for resources.")
        return None

    def _validate_prerequisites(self, ability_enum, req_info, target_ability_name) -> str:
        ability_upper = target_ability_name.upper()
        safe_req_info = req_info if req_info else {}
        producer_type = safe_req_info.get('producer')

        if not producer_type:
            if "BARRACKS" in ability_upper: producer_type = UnitTypeId.BARRACKS
            elif "FACTORY" in ability_upper: producer_type = UnitTypeId.FACTORY
            elif "STARPORT" in ability_upper: producer_type = UnitTypeId.STARPORT
            elif "COMMANDCENTER" in ability_upper: producer_type = UnitTypeId.COMMANDCENTER
            elif "SCV" in ability_upper: producer_type = UnitTypeId.COMMANDCENTER

        if not req_info and not producer_type:
            return (f"**Action Failed**: Ability '{target_ability_name}' unknown.\n"
                    f"**Reason**: No prerequisite data found.")

        my_producers = None
        townhall_types = {UnitTypeId.COMMANDCENTER, UnitTypeId.ORBITALCOMMAND, UnitTypeId.PLANETARYFORTRESS}
        target_product = safe_req_info.get('product')

        if target_product == UnitTypeId.SCV or (producer_type in townhall_types):
            my_producers = self.townhalls
            producer_name_for_log = "TownHall"
        elif producer_type:
            my_producers = self.structures(producer_type) | self.units(producer_type)
            producer_name_for_log = producer_type.name
        else:
             return f"**Action Failed**: Unknown producer."

        if my_producers is not None:
            if not my_producers.exists:
                if producer_type == UnitTypeId.WARPGATE and self.structures(UnitTypeId.GATEWAY).exists:
                    pass
                else:
                    return (f"**Action Failed**: Missing facility.\n"
                            f"**Reason**: Required '{producer_name_for_log}' not found.\n"
                            f"**Suggestion**: Check if need Build a {producer_name_for_log}.")

            my_ready_producers = my_producers.ready
            if not my_ready_producers.exists:
                return (f"**Action Failed**: Facility under construction.\n"
                        f"**Reason**: {producer_name_for_log} is not ready.\n"
                        f"**Suggestion**: Wait for completion.")

            grounded = my_ready_producers.filter(lambda u: not u.is_flying)
            if not grounded.exists:
                return (f"**Action Failed**: Facility unavailable.\n"
                        f"**Reason**: All {producer_name_for_log}s are flying.\n"
                        f"**Suggestion**: Land structure.")

        if safe_req_info.get('requires_techlab', False) and producer_type:
            specific_producers = self.structures(producer_type)
            producers_with_techlab = specific_producers.filter(lambda u: u.has_techlab and u.is_ready)
            if not producers_with_techlab.exists:
                return (f"**Action Failed**: Tech Lab required.\n"
                        f"**Reason**: No {producer_type.name} with Tech Lab available.\n"
                        f"**Suggestion**: Check if need Build a Tech Lab.")

        required_building = safe_req_info.get('required_building')
        if required_building:
            if not self._has_structure_or_alias(required_building):
                if self.structures(required_building).exists:
                    return (f"**Action Failed**: Prerequisite pending.\n"
                            f"**Reason**: '{required_building.name}' is constructing.\n"
                            f"**Suggestion**: Wait.")
                return (f"**Action Failed**: Prerequisite missing.\n"
                        f"**Reason**: Requires '{required_building.name}'.\n"
                        f"**Suggestion**: Construct {required_building.name}.")

        return None

    async def validate_action_and_get_suggestion(self, target_ability_name: str, current_available_abilities: set) -> str:
        try:
            ability_enum = AbilityId[target_ability_name]
        except KeyError:
            return (f"**Action Failed**: Ability '{target_ability_name}' unknown.")

        req_info = self.ability_requirements_map.get(ability_enum)

        if ability_enum in current_available_abilities:
            return self._validate_resources_and_supply(ability_enum, req_info)
        else:
            prereq_error = self._validate_prerequisites(ability_enum, req_info, target_ability_name)
            if prereq_error:
                return prereq_error
            return (f"**Action Failed**: Ability '{target_ability_name}' not currently available.\n"
                    f"**Suggestion**: Wait.")

    def _finish_current_bo(self):
        print(f"[{self.time_formatted}] >>> BO #{self.current_bo_index} Completed/Skipped. <<<\n")
        self.current_bo_index += 1
        self.current_bo_parsed_state = None
        self.current_bo_wait_count = 0
        self.pending_pre_move = None
        self.last_bo_block_reason = None
        self.current_bo_pre_move_attempted = False

    def _fail_current_bo(self, reason):
        """Only called before a target command is submitted, on map failure/timeout."""
        trace = getattr(self, "command_trace", None)
        if trace is not None:
            trace.emit('bo_failed', reason=reason, origin=trace.bo_origin(
                self.current_bo_index, self.planner_queue_active,
                self.bo_data[self.current_bo_index]['action']))
        if not self.execution_feedback_enabled or not self.planner_queue_active:
            self._finish_current_bo()
            return
        index = self.current_bo_index
        feedback = {
            "source_policy_action_id": self.commander.planner.last_consumed_policy_action_id,
            "phase_index": self.commander.strategy.get("_phase_index"),
            "processed_prefix_count": index,
            "queue_source": "parsed_answer_plus_existing_rules",
            "failed": {
                "action": self.bo_data[index]["action"],
                "reason": reason,
                "target_command_submitted": False,
                "pre_move_command_attempted": self.current_bo_pre_move_attempted,
            },
            "unsubmitted_suffix": [item["action"] for item in self.bo_data[index + 1:]],
            "restored_selectable_quota": self.commander.release_unsubmitted_bo(index),
        }
        feedback["replan_scheduled"] = bool(self.commander.waiting_bo)
        if not feedback["replan_scheduled"]:
            feedback["skip_reason"] = "no_selectable_quota"
        self.logging("execution_failure_feedback", feedback, save_trace=True, print_log=True)
        if not feedback["replan_scheduled"]:
            self._finish_current_bo()
            return
        self.pending_execution_feedback = feedback
        self.current_bo_parsed_state = None
        self.current_bo_wait_count = 0
        # Forget the local cache, but do not cancel an already-issued MOVE or any
        # processed-prefix SC2 command. The next game frame owns the new query.
        self.pending_pre_move = None
        self.last_bo_block_reason = None
        self.current_bo_pre_move_attempted = False

    def _check_bo_delay(self, current_bo_item, is_skipped):
        if "time" in current_bo_item and "seconds" in current_bo_item:
            real_time = None if is_skipped else int(self.time)
            BO_time = current_bo_item["seconds"]
            BO_action = current_bo_item["action"]
            The_game = self.bo_address.strip(".json").split("/")[-1]
            one_BO_delay = {
                "BO_time": BO_time,
                "real_time": real_time,
                "BO_action": BO_action
            }
            self.delay_time.append(one_BO_delay)
            os.makedirs(f"examination/{The_game}", exist_ok=True)
            time_label = getattr(self, "time_str", None) or getattr(self, "time_formatted", "unknown")
            time_label = str(time_label).replace(":", "-")
            with open(f"examination/{The_game}/{time_label}_delay.json", "w", encoding="utf-8") as f:
                    json.dump(self.delay_time, f, indent=2, ensure_ascii=False)

    async def _get_scripted_build_position(self, ability_name: str) -> Optional[Point2]:
        try:
            ability = AbilityId[ability_name]
            if self.townhalls.exists:
                start_pos = self.townhalls.random.position
            else:
                start_pos = self.start_location

            pos = await self.find_placement(ability, start_pos, max_distance=30, placement_step=2)
            return pos
        except Exception as e:
            print(f"[Warning] Failed to find placement for {ability_name}: {e}")
            return None

    def _switch_build_order(self, new_bo_address: str):
        """用于中途动态切换 BO"""
        print(f"\n[{self.time_formatted}] 🔄 TopAgent initiated BO Switch! Old: {self.bo_address} -> New: {new_bo_address}")
        try:
            self.bo_address = new_bo_address
            self.bo_data = load_bo_data(self.bo_address)
            self.current_bo_index = 0
            self.current_bo_parsed_state = None
            self.pending_pre_move = None
        except Exception as e:
            print(f"[Error] Failed to switch BO: {e}")

    def regular_unit_name(self, data):
        names = defaultdict(int)
        for u in data:
            n = u.name.lower()
            if n == 'scv':
                names['SCV'] += 1
            elif 'flying' in n:
                names[n.replace('flying', '')] += 1
            elif 'lowered' in n:
                names[n.replace('lowered', '')] += 1
            elif 'sieged' in n:
                names[n.replace('sieged', '')] += 1
            elif 'burrowed' in n:
                names[n.replace('burrowed', '')] += 1
            elif n == 'liberatorag':
                names['liberator'] += 1
            elif n == 'thorap':
                names['thor'] += 1
            elif n in ['vikingassault', 'vikingfighter']:
                names['viking'] += 1
            elif 'reactor' in n:
                name = n.replace('reactor', '').lower()
                names[f"{name} with reactor"] += 1
            elif 'techlab' in n:
                name = n.replace('techlab','').lower()
                names[f"{name} with techlab"] += 1
            else:
                names[u.name.lower()] += 1

        names = sorted(names.items())
        return names

    def get_obs(self):
        # ===== own =====
        # unit string
        if len(self.units) == 0:
            unit_string = "no units currently"
        else:
            names = self.regular_unit_name(self.units)
            units_text = []
            for element, count in names:
                if count == 0:
                    continue
                units_text.append(f"{count} {element}")

            unit_string = ", ".join(units_text)

        # structure string
        if len(self.structures) == 0:
            structure_string = "no structure currently"
        else:
            names = self.regular_unit_name(self.structures)

            bases = ["commandcenter", 'orbitalcommand', 'planetaryfortress']

            units_text = []
            for name, count in names:
                if name in bases:
                    base_str = f"{count} bases in the form of {name}"
                    units_text.append(base_str)

            for name, count in names:
                if name not in bases:
                    base_str = f"{count} {name}"
                    units_text.append(base_str)
            structure_string = ', '.join(units_text)

        # ===== enemy =====
        # unit string
        if len(self.enemy_units) == 0:
            enemy_unit_string = "no units currently"
        else:
            names = self.regular_unit_name(self.enemy_units)
            units_text = []
            for element, count in names:
                if count == 0:
                    continue
                units_text.append(f"{count} {element}")

            enemy_unit_string = ", ".join(units_text)

        # structure string
        if len(self.enemy_structures) == 0:
            enemy_structure_string = "no structure currently"
        else:
            names = self.regular_unit_name(self.enemy_structures)

            bases = ["commandcenter", 'orbitalcommand', 'planetaryfortress']

            units_text = []
            for name, count in names:
                if name in bases:
                    base_str = f"{count} bases in the form of {name}"
                    units_text.append(base_str)

            for name, count in names:
                if name not in bases:
                    base_str = f"{count} {name}"
                    units_text.append(base_str)
            enemy_structure_string = ', '.join(units_text)

        research_string = ', '.join(self.history_research) if len(self.history_research) else 'no research currently'

        obs_text = f"It is now {self.time_formatted} in the game, \
and we are playing as Terran with {self.minerals} minerals and {self.vespene} vespene gas, \
while our current supply is {self.supply_used}, of which {self.supply_army} is allocated to the army, {self.supply_workers} to workers, and {self.supply_left} remains unused on a 128x128 sized map. \
\nIn terms of our military forces, we have {unit_string}. In terms of the structures, we have {structure_string}. \
\nIn terms of research, we have researched {research_string}.\
\nAs for the visible enemy presence, there are {enemy_unit_string}. As for the visible enemy structures, there are {enemy_structure_string}."
        return obs_text

    # async def on_end(self, game_result):
    #     game_result = game_result.name
    #     self.logging("game_result", game_result, save_trace=True)
    #     self.logging("SBR", round(self.sbr.mean, 4), save_trace=True)

    #     time_cost = self.time_formatted.split(":")
    #     time_cost = int(time_cost[0]) * 60 + int(time_cost[1])
    #     self.logging("time_cost", time_cost, save_trace=True)
    #     self.logging("RUR", round(self.resource_cost / time_cost, 4), save_trace=True)

    #     with open(f"{self.log_path}/trace.json", "w", encoding="utf-8") as f:
    #         json.dump(self.trace, f, indent=2, ensure_ascii=False)
    #     self.commander.planner.game_end(self.get_obs(), game_result)

    async def run(self, iteration: int):
        # 1. Run maintenance tasks, unpack current strategy parameters, and get new scouting info
        new_scouting_info = await self.run_maintenance_tasks(
            iteration,
            **self.strategy_params
        )

        # ==========================================
        # TopAgent Trigger Logic (Check Switch Only)
        # ==========================================
        # Safely initialize state variables
        # if not hasattr(self, 'all_scouting_history'):
        #     self.all_scouting_history = []
        #     self.enemy_base_scouted = False
        #     self.enemy_base_scout_time = -1
        #     self.top_agent_triggered = False

        # # Accumulate all incoming scouting information
        # if new_scouting_info:
        #     self.all_scouting_history.extend(new_scouting_info)

        # # Check if the scout has finalized its mission
        # if not self.enemy_base_scouted:
        #     if hasattr(self, 'scout_state') and self.scout_state == "DONE":
        #         self.enemy_base_scouted = True
        #         self.enemy_base_scout_time = self.time # Record the game time (in seconds)
        #         print(f"[{self.time_formatted}] ⚠️ Final scout confirmed! Triggering TopAgent...")

        # # Check if conditions are met to trigger TopAgent: Scouted AND never triggered before
        # if self.enemy_base_scouted and not self.top_agent_triggered:
        #     scouting_text = "\n".join(self.all_scouting_history)
        #     print(f"[{self.time_formatted}] 📡 TopAgent analyzing enemy style...")

        #     # Lock the trigger so it only executes once per game
        #     self.top_agent_triggered = True

        #     # 调用 TopAgent 获取决策（内部已完成映射计算）
        #     decision = await self.top_agent.decide_strategy(
        #         scouting_info=scouting_text,
        #         game_time=self.time_formatted,
        #         current_bo_path=self.bo_address
        #     )

        #     predicted_style = decision.get("predicted_style")
        #     print(f"[{self.time_formatted}] 🧠 Enemy style classified as: {predicted_style}")

        #     # LLM player 只检测有没有切换
        #     if decision.get("switch_bo"):
        #         new_bo_path = decision.get("bo_path")
        #         print(f"[{self.time_formatted}] 🔄 Switch Triggered! Changing BO to counter {predicted_style}.")
        #         self._switch_build_order(new_bo_path)
        #     else:
        #         # 如果不切换，则维持原策略执行
        #         print(f"[{self.time_formatted}] 🛡️ No switch needed. Maintaining current default strategy.")

        #     # 更新攻防阈值参数
        #     new_params = decision.get("params", {})
        #     if new_params:
        #         for key, value in new_params.items():
        #             self.strategy_params[key] = value
        #         print(f"[{self.time_formatted}] ⚙️ Macro parameters synced: {self.strategy_params}")
        # # ==========================================

        if iteration % 10 == 0:
            self.log_current_iteration(iteration)

        if self.pending_execution_feedback is not None:
            self.bo_data = self.commander.renew_bo(
                self.get_obs(), execution_feedback=self.pending_execution_feedback,
            )
            self.pending_execution_feedback = None
            self.current_bo_index = 0
            self.current_bo_wait_count = 0
            self.current_bo_wait_start = None
            self.last_bo_checked_index = -1
            self.last_bo_check_time = -999
            self.logging('refresh build order', ', '.join([p['action'] for p in self.bo_data]))
        elif self.current_bo_index >= len(self.bo_data):
            self.bo_data = self.commander.renew_bo(self.get_obs())
            self.current_bo_index = 0
            self.planner_queue_active = True
            self.logging('refresh build order', ', '.join([p['action'] for p in self.bo_data]))

        if self.current_bo_index < len(self.bo_data):

            if not hasattr(self, 'current_bo_wait_start') or self.current_bo_wait_start is None:
                self.current_bo_wait_start = iteration

            if not hasattr(self, 'current_bo_wait_count'):
                self.current_bo_wait_count = 0

            current_bo_item = self.bo_data[self.current_bo_index]
            bo_action = current_bo_item['action']

            if str(bo_action).startswith(">>>"):
                print(f"[{self.time_formatted}] Skipping Phase Marker: {bo_action}")
                self._finish_current_bo()
                return

            bo_command_str = f"{bo_action}"

            # --- Phase A: Parsing ---
            if self.current_bo_parsed_state is None:
                print(f"\n[{self.time_formatted}] >>> Start Processing BO #{self.current_bo_index}: {bo_command_str} <<<")

                parsed_state = {
                    "type": "OPERATION",
                    "attack_cmds": [],
                    "ability_name": None
                }

                # add attacking instruction
                if self._looks_like_attack_bo(bo_command_str):
                    classified_results = self.adjust_agent.run([bo_command_str])
                    self.logging(f"llm_decision_classify_{self.current_bo_index}", classified_results, save_trace=True, print_log=True)
                else:
                    classified_results = {
                        "standard_attack_commands": [],
                        "empty_tasks": [],
                        "other_tasks": [bo_command_str],
                    }

                if classified_results.get("standard_attack_commands"):
                    parsed_state["type"] = "ATTACK"
                    parsed_state["attack_cmds"] = classified_results["standard_attack_commands"]
                # end attacking instruction
                else:
                    if not self.ability_descriptions_text:
                        self.ability_descriptions_text = self._load_ability_data()

                    ability_result = await self.micro_agent.bo_to_ability(
                        bo_step=bo_command_str,
                        ability_descriptions=self.ability_descriptions_text
                    )
                    self.logging(f"llm_decision_bo_map_{self.current_bo_index}", ability_result, save_trace=True, print_log=True)

                    if ability_result:
                        parsed_state["ability_name"] = ability_result[0]
                        print(f"    -> Parsed Ability: {parsed_state['ability_name']}")
                    else:
                        print(f"    -> [Warning] Failed to map BO to Ability Name. Skipping.")
                        self._fail_current_bo("mapping_failed")
                        return

                self.current_bo_parsed_state = parsed_state

            # --- Phase B: Execution ---
            state = self.current_bo_parsed_state

            if state["type"] == "ATTACK":
                # Keep attack BO accounting, but leave actual attack timing to script_player.
                # This matches the restored baseline executor and avoids duplicate small waves.
                self._finish_current_bo()
                return

            if state["type"] == "OPERATION" and state["ability_name"]:
                target_ability_name = state["ability_name"]

                if target_ability_name == "CALLDOWNMULE_CALLDOWNMULE":
                    print(f"[{self.time_formatted}] BO Step is MULE, skipping.")
                    self._finish_current_bo()
                    return

                # Frequency control
                current_check_time = iteration
                last_check_time = getattr(self, 'last_bo_check_time', -999)
                last_bo_index = getattr(self, 'last_bo_checked_index', -1)

                if self.current_bo_index == last_bo_index:
                    if (current_check_time - last_check_time) < 22:
                        return
                    else:
                        self.current_bo_wait_count += 1
                else:
                    self.current_bo_wait_count = 1

                self.last_bo_check_time = current_check_time
                self.last_bo_checked_index = self.current_bo_index

                timeout_threshold = 15 if self.time < 300 else 10

                if self.current_bo_wait_count > timeout_threshold:
                    warning_msg = f"[Skip] BO #{self.current_bo_index} skipped after {self.current_bo_wait_count} attempts (Threshold: {timeout_threshold})."
                    print(warning_msg)
                    self.logging(f"bo_skipped_{self.current_bo_index}", {"reason": "timeout", "action": bo_command_str}, save_trace=True)
                    await self.chat_send(f"Skipping stuck order: {bo_action}")
                    self._check_bo_delay(current_bo_item, True)
                    self._fail_current_bo("timeout: " + (self.last_bo_block_reason or "target command not submitted"))
                    return

                if (self.pending_pre_move and
                    self.pending_pre_move["bo_index"] == self.current_bo_index and
                    self.pending_pre_move["ability"] == target_ability_name):

                    cached = self.pending_pre_move

                    try:
                        unit_id = int(cached["unit_id"])
                    except (ValueError, TypeError):
                         print(f"    -> [Error] Cached unit tag is invalid: {cached['unit_id']}")
                         unit_id = -1

                    unit = self.get_unit_by_id(unit_id)

                    if unit:
                        ability_enum = AbilityId[target_ability_name]
                        req_info = self.ability_requirements_map.get(ability_enum)
                        resource_error = self._validate_resources_and_supply(ability_enum, req_info)

                        if resource_error:
                            self.last_bo_block_reason = resource_error
                            return

                        print(f"    -> [Execute Cached] Resources Ready! Using Pre-moved Unit {unit.tag} to Build at {cached['target_pos']}")

                        final_action = {
                            "action": target_ability_name,
                            "units": [cached["unit_id"]]
                        }

                        if target_ability_name in ["TERRANBUILD_REFINERY", "PROTOSSBUILD_ASSIMILATOR", "ZERGBUILD_EXTRACTOR"]:
                            geysers = self.vespene_geyser.closer_than(1.0, cached["target_pos"])
                            if geysers.exists:
                                target_geyser = geysers.first
                                final_action["target_unit"] = self.tag_to_id(target_geyser.tag)
                            else:
                                print(f"    -> [Error] No Vespene Geyser found at {cached['target_pos']} for Refinery.")
                                self.pending_pre_move = None
                                return
                        else:
                            final_action["target_position"] = [int(cached["target_pos"].x), int(cached["target_pos"].y)]

                        await self.run_actions([final_action])
                        self._check_bo_delay(current_bo_item, False)
                        self._finish_current_bo()
                        return
                    else:
                        print(f"    -> [Execute Cached Failed] Pre-moved unit {unit_id} is dead or missing. Falling back to LLM.")
                        self.pending_pre_move = None

                # Normal flow
                all_own_units = self.units + self.structures
                raw_abilities_list = await self.get_available_abilities(all_own_units, ignore_resource_requirements=True)
                current_available_abilities_set = {ab for sublist in raw_abilities_list for ab in sublist}

                error_prompt = await self.validate_action_and_get_suggestion(target_ability_name, current_available_abilities_set)
                self.last_bo_block_reason = error_prompt or None

                if error_prompt:
                    is_resource_error = "Insufficient" in error_prompt or "Supply blocked" in error_prompt
                    is_terran_build = target_ability_name.startswith("TERRANBUILD_")

                    should_start_pre_move = (
                        self.time > 270 and
                        is_resource_error and
                        is_terran_build and
                        "REFINERY" not in target_ability_name and
                        (self.pending_pre_move is None or self.pending_pre_move["bo_index"] != self.current_bo_index)
                    )

                    if should_start_pre_move:
                        print(f"    -> [Pre-Move] Resources missing for {target_ability_name}. Initiating Move-First logic.")

                        valid_units_text = await self.units_to_text_wo_contructing(self.workers)

                        executor_tags = await self.micro_agent.select_executor(
                            bo_step=bo_command_str,
                            ability_name=target_ability_name,
                            available_units_text=valid_units_text,
                            future_bo_list="None"
                        )
                        self.logging(f"llm_pre_move_executor_{self.current_bo_index}", executor_tags, save_trace=True)

                        if executor_tags:
                            target_pos = await self._get_scripted_build_position(target_ability_name)

                            if target_pos:
                                try:
                                    raw_tag = executor_tags[0]
                                    if isinstance(raw_tag, list):
                                        raw_tag = raw_tag[0]
                                    unit_id = int(raw_tag)
                                except (ValueError, TypeError, IndexError) as e:
                                    print(f"    -> [Error] Failed to parse unit tag from {executor_tags}: {e}")
                                    unit_id = -1

                                unit = self.get_unit_by_id(unit_id)

                                if unit:
                                    print(f"    -> [Pre-Move] Commanding Unit {unit.tag} to Move to {target_pos}")

                                    move_action = {
                                        "action": "MOVE_MOVE",
                                        "units": [unit_id],
                                        "target_position": [int(target_pos.x), int(target_pos.y)]
                                    }

                                    await self.run_actions([move_action])
                                    self.current_bo_pre_move_attempted = True

                                    self.pending_pre_move = {
                                        "bo_index": self.current_bo_index,
                                        "unit_id": unit_id,
                                        "target_pos": target_pos,
                                        "ability": target_ability_name
                                    }
                                else:
                                    print(f"    -> [Pre-Move Failed] Selected unit {unit_id} not found in self.units (Total: {len(self.units)}).")
                            else:
                                print("    -> [Pre-Move Failed] Script could not find a build position.")
                        else:
                            print("    -> [Pre-Move Failed] LLM could not select a worker.")

                    print(f"    -> [Wait] Action blocked: {error_prompt.splitlines()[0]}")
                    return

                ability_enum = AbilityId[target_ability_name]
                valid_executors_tags = []
                for unit, available_abilities in zip(all_own_units, raw_abilities_list):
                    if ability_enum in available_abilities:
                        valid_executors_tags.append(unit.tag)

                valid_executors = all_own_units.filter(lambda u: u.tag in valid_executors_tags)
                if target_ability_name == "BARRACKSTRAIN_MARINE" and self.micro_agent._local_fallback_enabled():
                    # burnysc2 7.1.3 BotAI.train: use a free production slot, not a waiting queue.
                    valid_executors = valid_executors.filter(
                        lambda u: len(u.orders) < 1 + int(u.add_on_tag in self.reactor_tags)
                    )

                actions = None
                if valid_executors:
                    valid_units_text = await self.units_to_text_wo_contructing(valid_executors)
                    raw_obs_text_dict = await self.obs_to_text_dict()

                    if hasattr(self, "get_structures_summary"):
                        print(self.get_structures_summary())
                    map_info = raw_obs_text_dict['Map information']
                    own_units_text = await self.units_to_text(self.units)
                    own_structures_text = await self.structures_to_text(self.structures)

                    executor_result = await self.micro_agent.select_executor(
                        bo_step=bo_command_str,
                        ability_name=target_ability_name,
                        available_units_text=valid_units_text,
                        future_bo_list="None"
                    )

                    self.logging(f"llm_decision_executor_{self.current_bo_index}_{iteration}", executor_result, save_trace=True, print_log=True)

                    if executor_result:
                        target_type = self.get_action_target_type(target_ability_name)

                        actions = await self.micro_agent.get_final_action(
                            ability_name=target_ability_name,
                            executor_tags=executor_result,
                            map_info=map_info,
                            own_units_text=own_units_text,
                            own_structures_text=own_structures_text,
                            target_type=target_type
                        )

                        self.logging(f"llm_decision_final_action_{self.current_bo_index}_{iteration}", actions, save_trace=True, print_log=True)
                    else:
                        print(f"    -> [Wait] Agent returned no executor for {target_ability_name}")
                        self.last_bo_block_reason = "No executor returned for " + target_ability_name
                else:
                    print(f"    -> [Wait] No unit capable of {target_ability_name}")
                    self.last_bo_block_reason = "No available executor for " + target_ability_name

                if actions:
                    print(f"[{self.time_formatted}] >>> BO #{bo_command_str} action condition <<<\n")
                    await self.run_actions(actions)
                    self._check_bo_delay(current_bo_item, False)
                    self._finish_current_bo()
                else:
                     if not self.pending_pre_move:
                        print(f"    -> [Wait] Generated empty action for {target_ability_name}")
                        self.last_bo_block_reason = self.last_bo_block_reason or "Empty action for " + target_ability_name
