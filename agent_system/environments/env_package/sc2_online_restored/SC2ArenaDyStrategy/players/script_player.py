# script_player.py
from .base_player import BasePlayer
from sc2.unit import Unit
from sc2.units import Units
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId
from sc2.ids.buff_id import BuffId
from sc2.ids.effect_id import EffectId
from sc2.position import Point2
from typing import Dict, Any, Set, List, Optional
import math
import random
import copy
import os
from collections import Counter


def read_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def read_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default

class ScriptPlayer(BasePlayer):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        # --- [从 LLMPlayer 移动来的初始化变量] ---
        self.pending_pre_move: Optional[Dict[str, Any]] = None

        # SCV auto-attack settings
        self.scv_auto_attack_distance = 4
        self.scv_auto_attack_time = 240

        # 自动防御 map
        self.active_defense_map = {}

        # 侦察逻辑变量
        self.active_scout_unit_tag = None
        self.scouting_information = {}
        self.scouted_locations = set()
        self.scv_scout_sent = False
        self.scout_target_location = None
        self.known_enemy_tags_in_vision = set()

        self.structure_rally_points = {}

        # 驻防逻辑变量
        self.GARRISON_PERPENDICULAR_DISTANCE = 15
        self.GARRISON_EXTENSION_DISTANCE = 10
        self.GARRISON_CHECK_RADIUS = 4
        self.GARRISON_DEFENSE_ZONE_RADIUS = 30
        self.last_garrison_check_time = -1

        # 总攻系统
        self.total_attack_groups: Dict[int, Dict[str, Any]] = {}
        self.total_attack_wave_id_counter: int = 0
        # 需要追加进攻兵力的指令列表
        self.reinforcements_commands = []
        self.TIMER = max(1, read_int_env("SC2_REINFORCEMENT_TIMER", 88))
        self.llm_attack_min_units = max(1, read_int_env("SC2_LLM_ATTACK_MIN_UNITS", 5))
        self.first_llm_attack_min_units = max(
            self.llm_attack_min_units,
            read_int_env("SC2_FIRST_LLM_ATTACK_MIN_UNITS", self.llm_attack_min_units),
        )
        self.reinforcement_min_units = max(1, read_int_env("SC2_REINFORCEMENT_MIN_UNITS", 5))
        self.explicit_attack_all_available_threshold = max(
            1,
            read_int_env("SC2_EXPLICIT_ATTACK_ALL_AVAILABLE_THRESHOLD", 40),
        )
        self.explicit_attack_overrequest_multiplier = max(
            1,
            read_int_env("SC2_EXPLICIT_ATTACK_OVERREQUEST_MULTIPLIER", 2),
        )
        self.explicit_attack_supply_slack = max(
            0,
            read_int_env("SC2_EXPLICIT_ATTACK_SUPPLY_SLACK", 4),
        )
        self.attack_merge_max_units = max(
            self.reinforcement_min_units,
            read_int_env("SC2_ATTACK_MERGE_MAX_UNITS", 16),
        )
        self.newexec_attack_min_units_without_tech = max(
            0,
            read_int_env("SC2_NEWEXEC_ATTACK_MIN_UNITS_WITHOUT_TECH", 0),
        )
        self.newexec_mixed_basic_attack_min_units = max(
            0,
            read_int_env("SC2_NEWEXEC_MIXED_BASIC_ATTACK_MIN_UNITS", 0),
        )
        self.newexec_pure_basic_attack_min_units = max(
            0,
            read_int_env("SC2_NEWEXEC_PURE_BASIC_ATTACK_MIN_UNITS", 0),
        )
        self.newexec_complete_explicit_direct_passthrough = (
            os.getenv("SC2_NEWEXEC_COMPLETE_EXPLICIT_DIRECT_PASSTHROUGH", "0") == "1"
        )
        self.newexec_complete_explicit_delayed_release = (
            os.getenv("SC2_NEWEXEC_COMPLETE_EXPLICIT_DELAYED_RELEASE", "1") == "1"
        )
        self.newexec_fill_explicit_to_ready = (
            os.getenv("SC2_NEWEXEC_FILL_EXPLICIT_TO_READY", "1") == "1"
        )

        # Compatibility with the restored training baseline: the new execution
        # layer is used, but the old strategy thresholds must still drive the
        # scripted attack scheduler.
        self.enable_early_scv_defense = True
        self.early_scv_defense_threshold = 5
        self.attack_config = {
            "first_attack_threshold": 60,
            "subsequent_attack_threshold": 20,
            "harassment_triggers": [],
        }
        self.harassment_wave_index = 0
        self.has_launched_first_attack = False
        self.triggered_harassment_levels = set()
        self.auto_train_marines_enabled = os.getenv("SC2_AUTO_TRAIN_MARINES", "1") != "0"
        self.enable_newexec_micro = os.getenv("SC2_ENABLE_NEWEXEC_MICRO", "0") == "1"
        self.micro_basic_hooligan_enabled = os.getenv("SC2_NEWEXEC_BASIC_HOOLIGAN", "0") == "1"
        self.micro_hooligan_bio_enabled = os.getenv("SC2_NEWEXEC_HOOLIGAN_BIO", "1") != "0"
        self.micro_hooligan_ally_power_radius = max(
            3.5, read_float_env("SC2_NEWEXEC_HOOLIGAN_ALLY_POWER_RADIUS", 3.5)
        )
        self.micro_hooligan_retreat_enabled = (
            os.getenv("SC2_NEWEXEC_HOOLIGAN_RETREAT", "1") != "0"
        )
        self.micro_special_abilities_enabled = os.getenv("SC2_NEWEXEC_SPECIAL_ABILITIES", "1") != "0"
        self.micro_special_bio_enabled = os.getenv("SC2_NEWEXEC_SPECIAL_BIO", "0") == "1"
        self.micro_fight_enabled = os.getenv("SC2_NEWEXEC_FIGHT_MICRO", "1") != "0"
        self.micro_bio_fight_enabled = os.getenv("SC2_NEWEXEC_BIO_FIGHT", "0") == "1"
        self.micro_require_tech_unit = os.getenv("SC2_NEWEXEC_REQUIRE_TECH_UNIT", "0") != "0"
        self.micro_defense_enabled = os.getenv("SC2_NEWEXEC_DEFENSE_MICRO", "0") == "1"
        self.micro_static_setup_min_support = max(
            0, read_int_env("SC2_NEWEXEC_STATIC_SETUP_MIN_SUPPORT", 0)
        )
        self.micro_static_setup_min_group = max(
            0, read_int_env("SC2_NEWEXEC_STATIC_SETUP_MIN_GROUP", 0)
        )
        self.micro_static_setup_ghost_radius = max(
            0, read_int_env("SC2_NEWEXEC_STATIC_SETUP_GHOST_RADIUS", 18)
        )
        self.micro_ghost_snipe_min_hp = max(
            1, read_int_env("SC2_NEWEXEC_GHOST_SNIPE_MIN_HP", 60)
        )
        self.micro_ghost_emp_min_targets = max(
            1, read_int_env("SC2_NEWEXEC_GHOST_EMP_MIN_TARGETS", 2)
        )
        self.micro_trace_interval = max(22, read_int_env("SC2_MICRO_TRACE_INTERVAL", 224))
        self.enable_attack_workers = os.getenv("SC2_ENABLE_ATTACK_WORKERS", "0") == "1"
        self.allow_explicit_small_attacks = os.getenv("SC2_ALLOW_EXPLICIT_SMALL_ATTACKS", "0") == "1"
        self.allow_all_available_workers = os.getenv("SC2_ALLOW_ALL_AVAILABLE_WORKERS", "0") == "1"
        self.allow_explicit_worker_support = os.getenv("SC2_ALLOW_EXPLICIT_WORKER_SUPPORT", "0") == "1"
        self.max_explicit_attack_workers = max(0, read_int_env("SC2_MAX_EXPLICIT_ATTACK_WORKERS", 0))
        attack_target_mode = os.getenv("SC2_ATTACK_TARGET_UNITS_FIRST", "auto").strip().lower()
        self.attack_target_units_first_mode = attack_target_mode
        self.attack_target_units_first = attack_target_mode in {"1", "true", "yes", "on", "always", "unit", "units"}
        self.attack_target_units_first_auto = attack_target_mode in {"", "auto", "adaptive", "tech"}
        self.attack_target_units_first_mixed_ratio = min(
            1.0, max(0.0, read_float_env("SC2_ATTACK_TARGET_UNITS_FIRST_MIXED_RATIO", 0.2))
        )
        self.hooligan_structure_focus_enabled = os.getenv("SC2_HOOLIGAN_STRUCTURE_FOCUS", "guarded").lower() not in {"0", "false", "off", "never"}
        self.attack_macro_retarget_interval = max(8, read_int_env("SC2_ATTACK_MACRO_RETARGET_INTERVAL", 48))
        self.attack_fallback_order_interval = max(8, read_int_env("SC2_ATTACK_FALLBACK_ORDER_INTERVAL", 48))
        self.attack_engaged_retarget_idle_only = (
            os.getenv("SC2_ATTACK_ENGAGED_RETARGET_IDLE_ONLY", "0") == "1"
        )

        self.enemy_townhall_types = {
            UnitTypeId.COMMANDCENTER,
            UnitTypeId.ORBITALCOMMAND,
            UnitTypeId.PLANETARYFORTRESS,
            UnitTypeId.NEXUS,
            UnitTypeId.HATCHERY,
            UnitTypeId.LAIR,
            UnitTypeId.HIVE,
        }

        # =====================================================================
        # [新增] 战斗微操系统变量 (参考 army.py 的 fight/dodge 机制)
        # =====================================================================

        # 帧控制: 每个单位的"下次可接受指令的帧"，防止指令覆盖过快
        # 参考 army.py 的 listenframe_of_unit
        self.unit_listenframe: Dict[int, int] = {}

        # 战斗状态: 每个单位"预期战斗结束帧"，标识单位正在交战中
        # 参考 army.py 的 ease
        self.unit_ease: Dict[int, int] = {}
        self.current_micro_iteration: int = 0

        # 血量快照: 上一帧的血量，用于检测掉血触发react
        # 参考 army.py 的 last_health
        self.unit_last_health: Dict[int, float] = {}

        # 闪避系统: 需要闪避的延迟效果集合 (effect_id, pos, radius, start_frame, end_frame)
        # 参考 army.py 的 delayed_effect
        self.dodge_effects: Set = set()

        # 闪避去重: 已记录的效果位置，避免重复添加
        # 参考 army.py 的 delayed_pos
        self.dodge_positions: Set = set()

        # 当前正在闪避的单位tags，闪避优先级最高
        # 参考 army.py 的 dodging
        self.dodging_tags: Set[int] = set()

        # 扫描系统: 每个单位上次扫描附近敌人的帧
        # 参考 army.py 的 scanned
        self.micro_scan_frame: Dict[int, int] = {}

        # 我能打到的敌人tags (per unit)
        # 参考 army.py 的 ha_scan
        self.micro_ha_scan: Dict[int, Set[int]] = {}

        # 能打到我的敌人tags (per unit)
        # 参考 army.py 的 ho_scan
        self.micro_ho_scan: Dict[int, Set[int]] = {}

        # Auditable micro state. These counters prove which unit types and
        # abilities actually reached the new attack layer during a rollout.
        self.micro_event_counts: Counter = Counter()
        self.micro_trace_last_frame = 0
        self.micro_special_scan_frame: Dict[int, int] = {}
        self.micro_last_enemy_frame: Dict[int, int] = {}

        # Marine 基础射程常量
        self.MARINE_RANGE = 5.0
        self.MARINE_RADIUS = 0.375

        # 微操帧间隔常量 (约0.13秒, 参考army.py的listenframe+3)
        self.MICRO_LISTEN_INTERVAL = 3

        # 战斗状态持续帧数 (约3秒, 参考army.py的ease 3*seconds)
        self.MICRO_EASE_DURATION = 66

        # 扫描间隔帧数 (约2秒, 参考army.py的scanned 2*seconds)
        self.MICRO_SCAN_INTERVAL = 44

        # =====================================================================
        # [新增结束] 战斗微操系统变量
        # =====================================================================

    def _freeze_attack_key_value(self, value):
        if isinstance(value, dict):
            return tuple(sorted((str(k), self._freeze_attack_key_value(v)) for k, v in value.items()))
        if isinstance(value, list):
            return tuple(self._freeze_attack_key_value(v) for v in value)
        return value

    def _reinforcement_command_key(self, command: Dict[str, Any]):
        return (
            self._freeze_attack_key_value(command.get("target_position", [])),
            str(command.get("target_unit", "")),
            self._freeze_attack_key_value(command.get("unit_type_counts", {})),
        )

    def _store_reinforcement_command(self, command: Dict[str, Any], timer: Optional[int] = None) -> bool:
        stored_command = copy.deepcopy(command)
        if timer is not None:
            stored_command["timer"] = timer
        elif "timer" not in stored_command:
            stored_command["timer"] = self.TIMER

        new_key = self._reinforcement_command_key(stored_command)
        new_count = max(1, stored_command.get("unit_count", 0))
        new_min_units = stored_command.get("min_units_to_launch")
        for pending in self.reinforcements_commands:
            if self._reinforcement_command_key(pending) != new_key:
                continue
            old_count = max(1, pending.get("unit_count", 0))
            merged_count = min(self.attack_merge_max_units, old_count + new_count)
            pending["unit_count"] = max(old_count, merged_count)
            if stored_command.get("all_available"):
                pending["all_available"] = True
            incoming_counts = stored_command.get("unit_type_counts") or {}
            if isinstance(incoming_counts, dict):
                pending_counts = pending.setdefault("unit_type_counts", {})
                for unit_name, unit_count in incoming_counts.items():
                    try:
                        normalized_count = max(0, int(unit_count))
                    except (TypeError, ValueError):
                        continue
                    pending_counts[unit_name] = max(0, int(pending_counts.get(unit_name, 0))) + normalized_count
                total_counts = sum(max(0, int(v)) for v in pending_counts.values())
                if total_counts > self.attack_merge_max_units:
                    scale = self.attack_merge_max_units / float(total_counts)
                    scaled_counts = {}
                    for unit_name, unit_count in pending_counts.items():
                        scaled_counts[unit_name] = max(1, int(round(max(0, int(unit_count)) * scale)))
                    while sum(scaled_counts.values()) > self.attack_merge_max_units:
                        largest = max(scaled_counts, key=scaled_counts.get)
                        scaled_counts[largest] -= 1
                        if scaled_counts[largest] <= 0:
                            del scaled_counts[largest]
                    pending["unit_type_counts"] = scaled_counts
                    pending["unit_count"] = sum(scaled_counts.values())
            if new_min_units is not None:
                pending["min_units_to_launch"] = max(
                    max(1, pending.get("min_units_to_launch", 1)),
                    max(1, int(new_min_units)),
                )
            if "timer" not in pending:
                pending["timer"] = stored_command["timer"]
            else:
                pending["timer"] = min(pending["timer"], stored_command["timer"])
            self.logging(
                "reinforcement_queue",
                f"合并重复攻击指令，当前等待投入 {pending['unit_count']} 单位。",
            )
            return False

        self.reinforcements_commands.append(stored_command)
        return True

    def _attack_uses_all_available(self, command: Dict[str, Any], requested_unit_count: int) -> bool:
        if bool(command.get("all_available")):
            return True
        raw_counts = command.get("unit_type_counts")
        has_explicit_counts = isinstance(raw_counts, dict) and bool(raw_counts)
        return requested_unit_count >= 100 and not has_explicit_counts

    def _relax_overlarge_explicit_attack(
        self,
        command: Dict[str, Any],
        requested_unit_count: int,
        available_count: int,
        context: str,
    ) -> Dict[str, Any]:
        if bool(command.get("all_available")):
            return command

        explicit_type_counts = self._command_unit_type_counts(command)
        if not explicit_type_counts:
            return command

        explicit_total = sum(explicit_type_counts.values())
        supply_used = max(
            1,
            int(getattr(self, "supply_used", 0) or (self.supply_army + self.supply_workers)),
        )
        effective_available = max(1, available_count)
        too_large_for_current_wave = (
            explicit_total >= self.explicit_attack_all_available_threshold
            and explicit_total > effective_available * self.explicit_attack_overrequest_multiplier
        )
        impossible_for_current_supply = explicit_total > supply_used + self.explicit_attack_supply_slack

        if not (too_large_for_current_wave or impossible_for_current_supply):
            return command

        relaxed = copy.deepcopy(command)
        relaxed["all_available"] = True
        relaxed["unit_count"] = max(200, requested_unit_count)
        relaxed["unit_type_counts"] = {}
        relaxed["_relaxed_overlarge_explicit"] = True
        self.logging(
            "MULEMICRO_ATTACK_SELECTION",
            (
                f"{context} relaxed overlarge explicit request "
                f"{command.get('unit_type_counts')} -> all_available "
                f"(requested_total={explicit_total}, available={available_count}, supply_used={supply_used})"
            ),
            "warning",
        )
        return relaxed

    ATTACK_UNIT_TYPE_ALIASES = {
        "marine": UnitTypeId.MARINE,
        "marauder": UnitTypeId.MARAUDER,
        "ghost": UnitTypeId.GHOST,
        "scv": UnitTypeId.SCV,
        "reaper": UnitTypeId.REAPER,
        "hellion": UnitTypeId.HELLION,
        "hellbat": UnitTypeId.HELLIONTANK,
        "sieged tank": UnitTypeId.SIEGETANK,
        "siegetank": UnitTypeId.SIEGETANK,
        "siege tank": UnitTypeId.SIEGETANK,
        "siegetanksieged": UnitTypeId.SIEGETANK,
        "medivac": UnitTypeId.MEDIVAC,
        "viking": UnitTypeId.VIKINGFIGHTER,
        "vikingfighter": UnitTypeId.VIKINGFIGHTER,
        "vikingassault": UnitTypeId.VIKINGASSAULT,
        "banshee": UnitTypeId.BANSHEE,
        "liberator": UnitTypeId.LIBERATOR,
        "raven": UnitTypeId.RAVEN,
        "battlecruiser": UnitTypeId.BATTLECRUISER,
        "battle cruiser": UnitTypeId.BATTLECRUISER,
        "widowmine": UnitTypeId.WIDOWMINE,
        "widow mine": UnitTypeId.WIDOWMINE,
        "cyclone": UnitTypeId.CYCLONE,
        "thor": UnitTypeId.THOR,
    }

    ATTACK_UNIT_TYPE_EQUIVALENTS = {
        UnitTypeId.SIEGETANK: {UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED},
        UnitTypeId.SIEGETANKSIEGED: {UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED},
        UnitTypeId.LIBERATOR: {UnitTypeId.LIBERATOR, UnitTypeId.LIBERATORAG},
        UnitTypeId.LIBERATORAG: {UnitTypeId.LIBERATOR, UnitTypeId.LIBERATORAG},
        UnitTypeId.VIKINGFIGHTER: {UnitTypeId.VIKINGFIGHTER, UnitTypeId.VIKINGASSAULT},
        UnitTypeId.VIKINGASSAULT: {UnitTypeId.VIKINGFIGHTER, UnitTypeId.VIKINGASSAULT},
        UnitTypeId.WIDOWMINE: {UnitTypeId.WIDOWMINE, UnitTypeId.WIDOWMINEBURROWED},
        UnitTypeId.WIDOWMINEBURROWED: {UnitTypeId.WIDOWMINE, UnitTypeId.WIDOWMINEBURROWED},
    }

    MICRO_STATIONARY_TYPES = {
        UnitTypeId.SIEGETANKSIEGED,
        UnitTypeId.LIBERATORAG,
        UnitTypeId.WIDOWMINEBURROWED,
    }

    MICRO_BASIC_A_MOVE_TYPES = {
        UnitTypeId.SCV,
    }

    MICRO_BASIC_HOOLIGAN_TYPES = {
        UnitTypeId.MARINE,
        UnitTypeId.MARAUDER,
        UnitTypeId.REAPER,
    }

    MICRO_SPECIAL_ABILITY_TYPES = {
        UnitTypeId.MARINE,
        UnitTypeId.MARAUDER,
        UnitTypeId.SCV,
        UnitTypeId.GHOST,
        UnitTypeId.BANSHEE,
        UnitTypeId.MEDIVAC,
        UnitTypeId.SIEGETANK,
        UnitTypeId.SIEGETANKSIEGED,
        UnitTypeId.LIBERATOR,
        UnitTypeId.LIBERATORAG,
        UnitTypeId.VIKINGFIGHTER,
        UnitTypeId.VIKINGASSAULT,
        UnitTypeId.RAVEN,
        UnitTypeId.WIDOWMINE,
        UnitTypeId.WIDOWMINEBURROWED,
        UnitTypeId.BATTLECRUISER,
        UnitTypeId.REAPER,
        UnitTypeId.THOR,
    }

    MICRO_TECH_ANCHOR_TYPES = {
        UnitTypeId.GHOST,
        UnitTypeId.BANSHEE,
        UnitTypeId.MEDIVAC,
        UnitTypeId.SIEGETANK,
        UnitTypeId.SIEGETANKSIEGED,
        UnitTypeId.LIBERATOR,
        UnitTypeId.LIBERATORAG,
        UnitTypeId.VIKINGFIGHTER,
        UnitTypeId.VIKINGASSAULT,
        UnitTypeId.RAVEN,
        UnitTypeId.WIDOWMINE,
        UnitTypeId.WIDOWMINEBURROWED,
        UnitTypeId.BATTLECRUISER,
        UnitTypeId.THOR,
    }

    MICRO_STRUCTURE_FIRST_ONLY_TYPES = {
        UnitTypeId.MARINE,
        UnitTypeId.REAPER,
    }

    MICRO_TARGET_UNITS_FIRST_FORCE_TYPES = {
        UnitTypeId.GHOST,
        UnitTypeId.BANSHEE,
        UnitTypeId.RAVEN,
        UnitTypeId.BATTLECRUISER,
        UnitTypeId.THOR,
    }

    SCV_HIGH_VALUE_REPAIR_TYPES = {
        UnitTypeId.SIEGETANK,
        UnitTypeId.SIEGETANKSIEGED,
        UnitTypeId.THOR,
        UnitTypeId.BATTLECRUISER,
        UnitTypeId.RAVEN,
        UnitTypeId.LIBERATOR,
        UnitTypeId.LIBERATORAG,
        UnitTypeId.VIKINGFIGHTER,
        UnitTypeId.VIKINGASSAULT,
        UnitTypeId.MEDIVAC,
        UnitTypeId.BANSHEE,
    }

    def _command_unit_type_counts(self, command: Dict[str, Any]) -> Dict[UnitTypeId, int]:
        raw_counts = command.get("unit_type_counts") or {}
        parsed_counts: Dict[UnitTypeId, int] = {}
        if not isinstance(raw_counts, dict):
            return parsed_counts

        for raw_name, raw_count in raw_counts.items():
            try:
                count = int(raw_count)
            except (TypeError, ValueError):
                continue
            if count <= 0:
                continue
            normalized = str(raw_name).lower().replace("_", " ").replace("-", " ").strip()
            compact = normalized.replace(" ", "")
            unit_type = self.ATTACK_UNIT_TYPE_ALIASES.get(normalized) or self.ATTACK_UNIT_TYPE_ALIASES.get(compact)
            if unit_type is None:
                continue
            parsed_counts[unit_type] = parsed_counts.get(unit_type, 0) + count
        return parsed_counts

    def _prepare_attack_command(self, command: Dict[str, Any], context: str) -> Optional[Dict[str, Any]]:
        if self.enable_attack_workers:
            return command

        raw_counts = command.get("unit_type_counts") or {}
        if not isinstance(raw_counts, dict) or not raw_counts:
            return command

        parsed_counts = self._command_unit_type_counts(command)
        requested_workers = parsed_counts.get(UnitTypeId.SCV, 0)
        requested_combat = sum(
            count for unit_type, count in parsed_counts.items()
            if unit_type != UnitTypeId.SCV
        )
        if requested_workers <= 0:
            return command

        if (
            bool(command.get("all_available"))
            and self.allow_explicit_worker_support
            and self.max_explicit_attack_workers > 0
        ):
            kept = copy.deepcopy(command)
            kept["_explicit_worker_support"] = True
            self.logging(
                "MULEMICRO_ATTACK_SELECTION",
                (
                    f"{context} kept all-available worker support "
                    f"workers={requested_workers} request={raw_counts}"
                ),
                "warning",
            )
            return kept

        if (
            self.allow_explicit_worker_support
            and requested_combat > 0
            and not bool(command.get("all_available"))
            and 0 < requested_workers <= self.max_explicit_attack_workers
        ):
            kept = copy.deepcopy(command)
            kept["_explicit_worker_support"] = True
            self.logging(
                "MULEMICRO_ATTACK_SELECTION",
                (
                    f"{context} kept explicit worker support "
                    f"workers={requested_workers} combat={requested_combat} request={raw_counts}"
                ),
                "warning",
            )
            return kept

        cleaned_counts = {}
        removed_workers = 0
        saw_worker = False
        for raw_name, raw_count in raw_counts.items():
            try:
                count = int(raw_count)
            except (TypeError, ValueError):
                continue
            if count <= 0:
                continue
            normalized = str(raw_name).lower().replace("_", " ").replace("-", " ").strip()
            compact = normalized.replace(" ", "")
            unit_type = self.ATTACK_UNIT_TYPE_ALIASES.get(normalized) or self.ATTACK_UNIT_TYPE_ALIASES.get(compact)
            if unit_type == UnitTypeId.SCV:
                saw_worker = True
                removed_workers += count
                continue
            cleaned_counts[raw_name] = count

        if not saw_worker:
            return command

        if not cleaned_counts:
            self.logging(
                "MULEMICRO_ATTACK_SELECTION",
                f"{context} skipped worker-only attack request {raw_counts}",
                "warning",
            )
            return None

        cleaned = copy.deepcopy(command)
        cleaned["unit_type_counts"] = cleaned_counts
        cleaned["unit_count"] = sum(cleaned_counts.values())
        self.logging(
            "MULEMICRO_ATTACK_SELECTION",
            f"{context} stripped {removed_workers} worker units from attack request {raw_counts} -> {cleaned_counts}",
            "warning",
        )
        return cleaned

    def _command_requests_explicit_worker_support(self, command: Dict[str, Any]) -> bool:
        if self.enable_attack_workers:
            return UnitTypeId.SCV in self._command_unit_type_counts(command)
        if not self.allow_explicit_worker_support:
            return False
        counts = self._command_unit_type_counts(command)
        requested_workers = counts.get(UnitTypeId.SCV, 0)
        if bool(command.get("all_available")):
            return requested_workers > 0 and self.max_explicit_attack_workers > 0
        requested_combat = sum(
            count for unit_type, count in counts.items()
            if unit_type != UnitTypeId.SCV
        )
        return (
            requested_combat > 0
            and 0 < requested_workers <= self.max_explicit_attack_workers
        )

    def _select_attack_units(self, current_available_units, command: Dict[str, Any], anchor_pos, unit_count: int):
        sorted_units = current_available_units.sorted(lambda u: u.distance_to(anchor_pos))
        if bool(command.get("all_available")):
            requested_type_counts = self._command_unit_type_counts(command)
            requested_workers = requested_type_counts.get(UnitTypeId.SCV, 0)
            combat_units = sorted_units
            if bool(command.get("_relaxed_overlarge_explicit")) or not self.allow_all_available_workers:
                combat_units = sorted_units.filter(
                    lambda u: u.type_id not in {UnitTypeId.SCV, UnitTypeId.MULE}
                )
            selected_units = combat_units.take(unit_count)
            if (
                command.get("_explicit_worker_support")
                and requested_workers > 0
                and self.allow_explicit_worker_support
                and self.max_explicit_attack_workers > 0
            ):
                worker_units = sorted_units.filter(lambda u: u.type_id == UnitTypeId.SCV)
                selected_units = selected_units | worker_units.take(
                    min(requested_workers, self.max_explicit_attack_workers)
                )
            return selected_units

        requested_type_counts = self._command_unit_type_counts(command)
        if not requested_type_counts:
            return sorted_units.take(unit_count)

        selected_tags = set()
        for unit_type, requested_count in requested_type_counts.items():
            type_pool = sorted_units.filter(
                lambda u, unit_type=unit_type: (
                    u.type_id in self.ATTACK_UNIT_TYPE_EQUIVALENTS.get(unit_type, {unit_type})
                    and u.tag not in selected_tags
                )
            )
            for unit in type_pool.take(requested_count):
                selected_tags.add(unit.tag)

        selected_units = sorted_units.filter(lambda u: u.tag in selected_tags)
        return self._balance_explicit_worker_support(selected_units, requested_type_counts)

    def _balance_explicit_worker_support(self, selected_units, requested_type_counts):
        requested_workers = requested_type_counts.get(UnitTypeId.SCV, 0)
        requested_combat = sum(
            count for unit_type, count in requested_type_counts.items()
            if unit_type != UnitTypeId.SCV
        )
        if requested_workers <= 0 or requested_combat <= 0:
            return selected_units

        selected_combat = sum(1 for unit in selected_units if unit.type_id != UnitTypeId.SCV)
        allowed_workers = min(
            requested_workers,
            self.max_explicit_attack_workers,
            selected_combat * requested_workers // requested_combat,
        )
        worker_tags = [
            unit.tag for unit in selected_units if unit.type_id == UnitTypeId.SCV
        ]
        keep_worker_tags = set(worker_tags[:allowed_workers])
        return selected_units.filter(
            lambda unit: unit.type_id != UnitTypeId.SCV or unit.tag in keep_worker_tags
        )

    def _unit_type_counts_from_units(self, units) -> Dict[str, int]:
        counts: Counter = Counter()
        for unit in units:
            counts[unit.type_id.name.lower()] += 1
        return dict(sorted(counts.items()))

    def _newexec_attack_has_tech_anchor(self, units) -> bool:
        return any(unit.type_id in self.MICRO_TECH_ANCHOR_TYPES for unit in units)

    def _newexec_attack_is_mixed_basic_group(self, units) -> bool:
        basic_types = {
            unit.type_id
            for unit in units
            if unit.type_id in self.MICRO_BASIC_HOOLIGAN_TYPES
        }
        return len(basic_types) >= 2

    def _newexec_attack_is_pure_basic_group(self, units) -> bool:
        has_basic = False
        for unit in units:
            if unit.type_id in self.MICRO_BASIC_HOOLIGAN_TYPES:
                has_basic = True
                continue
            if unit.type_id != UnitTypeId.SCV:
                return False
        return has_basic

    def _newexec_attack_is_marine_only_group(self, units) -> bool:
        has_marine = False
        for unit in units:
            if unit.type_id == UnitTypeId.MARINE:
                has_marine = True
                continue
            if unit.type_id != UnitTypeId.SCV:
                return False
        return has_marine

    def _newexec_full_army_is_marine_only(self) -> bool:
        combat_units = self._get_all_combat_units()
        return combat_units.exists and all(
            unit.type_id == UnitTypeId.MARINE for unit in combat_units
        )

    def _newexec_micro_active_for_current_army(self) -> bool:
        return (
            self.enable_newexec_micro
            and not self._newexec_full_army_is_marine_only()
        )

    def _newexec_complete_explicit_should_delay(self, selected_units) -> bool:
        if self._newexec_attack_has_tech_anchor(selected_units):
            return True
        if self._newexec_attack_is_mixed_basic_group(selected_units):
            return True
        return False

    def _newexec_min_units_for_selected_attack(self, units, base_min_units: int) -> int:
        # Micro must not change when the policy's attack is allowed to launch.
        # Raising this threshold delayed small mixed-army attacks (notably g3/g5),
        # which confounded macro execution with the post-launch micro comparison.
        return max(1, int(base_min_units))

    def _newexec_find_mergeable_attack_wave(
        self,
        units_to_launch,
        final_position: Optional[Point2],
        target_tag: Optional[int],
    ) -> Optional[int]:
        if not self._newexec_micro_active_for_current_army() or not self.total_attack_groups:
            return None
        if not units_to_launch.exists:
            return None

        incoming_tags = {unit.tag for unit in units_to_launch}
        incoming_count = len(incoming_tags)
        current_alive_tags = {unit.tag for unit in self.units}
        candidate_tags = set(incoming_tags)
        for attack_data in self.total_attack_groups.values():
            if attack_data.get("state") == "GATHERING":
                candidate_tags.update(attack_data.get("unit_tags", set()))
        if (
            self.micro_require_tech_unit
            and not self._micro_combat_has_tech_unit(candidate_tags)
        ):
            return None
        best_candidate = None
        best_score = None

        for wave_id, attack_data in self.total_attack_groups.items():
            state = attack_data.get("state")
            # Once a wave is moving, keep later reinforcements independent.
            # Absorbing them into ATTACKING suppressed the baseline's repeated
            # attack-wave launches and left mixed armies replenishing piecemeal.
            if state != "GATHERING":
                continue

            existing_tags = set(attack_data.get("unit_tags", set())) & current_alive_tags
            if not existing_tags or existing_tags & incoming_tags:
                continue

            merged_count = len(existing_tags) + incoming_count
            if merged_count > self.attack_merge_max_units:
                continue

            existing_target_tag = attack_data.get("target_tag")
            same_target = bool(target_tag and existing_target_tag == target_tag)
            if not same_target:
                existing_final = attack_data.get("final_position")
                if not existing_final or not final_position:
                    continue
                if existing_final.distance_to(final_position) > 32:
                    continue

            score = (merged_count,)
            if best_score is None or score < best_score:
                best_candidate = wave_id
                best_score = score

        return best_candidate

    def _defer_attack_until_ready(
        self,
        command: Dict[str, Any],
        min_units_to_launch: int,
        timer: Optional[int] = None,
    ) -> None:
        rein_command = copy.deepcopy(command)
        rein_command["timer"] = self.TIMER if timer is None else timer
        rein_command["unit_count"] = max(
            max(1, int(rein_command.get("unit_count", 0) or 0)),
            max(1, int(min_units_to_launch)),
        )
        rein_command["min_units_to_launch"] = max(1, int(min_units_to_launch))
        self._store_reinforcement_command(rein_command, timer=rein_command["timer"])

    def _explicit_attack_is_fully_selected(self, command: Dict[str, Any], selected_units) -> bool:
        requested = self._command_unit_type_counts(command)
        if not requested or not selected_units.exists:
            return False
        return not self._remaining_unit_type_counts(command, selected_units)

    def _defer_complete_explicit_delayed_release(
        self,
        command: Dict[str, Any],
        selected_units,
        context: str,
    ) -> None:
        rein_command = copy.deepcopy(command)
        rein_command["_newexec_complete_explicit_waited"] = True
        selected_count = max(1, len(selected_units))
        rein_command["timer"] = self.TIMER
        rein_command["unit_count"] = max(
            selected_count,
            sum(self._command_unit_type_counts(command).values()) or selected_count,
        )
        rein_command["min_units_to_launch"] = selected_count
        self.logging(
            "MULEMICRO_ATTACK_SELECTION",
            (
                f"{context} complete explicit group below newexec ready threshold; "
                f"delayed release queued with original selected count {selected_count}"
            ),
            "warning",
        )
        self._store_reinforcement_command(rein_command, timer=self.TIMER)

    def _fill_explicit_attack_to_ready_threshold(
        self,
        current_available_units,
        selected_units,
        anchor_pos,
        adjusted_min_units: int,
        context: str,
    ):
        if len(selected_units) >= adjusted_min_units:
            return selected_units
        selected_tags = {unit.tag for unit in selected_units}
        extra_needed = adjusted_min_units - len(selected_units)
        extras = current_available_units.filter(
            lambda unit: (
                unit.tag not in selected_tags
                and unit.type_id not in {UnitTypeId.SCV, UnitTypeId.MULE}
            )
        ).sorted(lambda unit: unit.distance_to(anchor_pos)).take(extra_needed)
        if not extras.exists:
            return selected_units
        filled_units = selected_units | extras
        self.logging(
            "MULEMICRO_ATTACK_SELECTION",
            (
                f"{context} filled complete explicit group to newexec ready threshold "
                f"{len(selected_units)}->{len(filled_units)}/{adjusted_min_units}; "
                f"extras={self._unit_type_counts_from_units(extras)}"
            ),
            "warning",
        )
        return filled_units

    async def _launch_direct_attack_passthrough(
        self,
        targets,
        units_to_commit,
        available_unit_tags,
        context: str,
    ) -> None:
        if not units_to_commit.exists:
            return
        num_targets = max(1, len(targets))
        units_per_target = max(1, math.ceil(len(units_to_commit) / num_targets))
        units_list = list(units_to_commit)
        curr_idx = 0
        for target in targets:
            sub_group_list = units_list[curr_idx : curr_idx + units_per_target]
            if not sub_group_list:
                break
            attack_target = target if isinstance(target, Unit) else target
            for unit in sub_group_list:
                available_unit_tags.discard(unit.tag)
                unit.attack(attack_target)
            target_pos = target.position if isinstance(target, Unit) else target
            self.logging(
                "MULEMICRO_ATTACK_SELECTION",
                (
                    f"{context} complete explicit squad direct passthrough: "
                    f"target={target_pos.rounded}, units={len(sub_group_list)}"
                ),
            )
            curr_idx += units_per_target

    def _remaining_unit_type_counts(self, command: Dict[str, Any], selected_units) -> Dict[str, int]:
        requested = command.get("unit_type_counts") or {}
        if not isinstance(requested, dict) or not requested:
            return {}
        remaining = {}
        for raw_name, raw_count in requested.items():
            try:
                count = max(0, int(raw_count))
            except (TypeError, ValueError):
                continue
            normalized = str(raw_name).lower().replace("_", " ").replace("-", " ").strip()
            compact = normalized.replace(" ", "")
            unit_type = self.ATTACK_UNIT_TYPE_ALIASES.get(normalized) or self.ATTACK_UNIT_TYPE_ALIASES.get(compact)
            compatible_types = self.ATTACK_UNIT_TYPE_EQUIVALENTS.get(unit_type, {unit_type}) if unit_type else set()
            selected_count = sum(1 for unit in selected_units if unit.type_id in compatible_types)
            if count > selected_count:
                remaining[raw_name] = count - selected_count
        return remaining

    def _select_scv_repair_target(self, scv, candidates, assignment_counts):
        eligible = [
            target for target in candidates
            if target.tag != scv.tag
            and getattr(target, "is_mechanical", False)
            and target.type_id in self.SCV_HIGH_VALUE_REPAIR_TYPES
            and getattr(target, "health_percentage", 1.0) < 0.98
            and assignment_counts.get(target.tag, 0) < 3
        ]
        if not eligible:
            return None
        return min(
            eligible,
            key=lambda target: (
                target.health_percentage,
                scv.distance_to(target),
            ),
        )

    def _record_micro_event(self, event: str, unit=None) -> None:
        self.micro_event_counts[event] += 1
        if unit is not None:
            self.micro_event_counts[f"unit:{unit.type_id.name.lower()}"] += 1

    def _flush_micro_trace(self, iteration: int) -> None:
        if iteration - self.micro_trace_last_frame < self.micro_trace_interval:
            return
        self.micro_trace_last_frame = iteration
        payload = ",".join(f"{key}={value}" for key, value in sorted(self.micro_event_counts.items()))
        self.logging("MULEMICRO_TRACE", payload or "no_events")

    async def _available_micro_abilities(self, unit) -> Set[AbilityId]:
        try:
            return set(await self.get_available_abilities(unit))
        except Exception as exc:
            self.logging("MULEMICRO_TRACE", f"ability_query_failed:{unit.type_id.name}:{exc}", "warning")
            return set()

    # ==========================================
    # [新增] 战斗微操系统 - 4个核心函数
    # 参考 army.py 的 fight() + do_dodge() 机制
    # ==========================================

    def micro_update_health_snapshot(self):
        """
        [新增] 更新血量快照，用于检测掉血。
        参考 army.py 的 last_health / current_health 机制。
        必须在所有微操逻辑之前调用。
        """
        current_tags = set()
        for unit in self.units:
            current_tags.add(unit.tag)
            # 上一帧的血量已经在 unit_last_health 中
            # 这里不更新，让 micro_fight 读取后再更新

        # 清理死亡单位的过期数据
        dead_tags = []
        for tag in self.unit_last_health:
            if tag not in current_tags:
                dead_tags.append(tag)
        for tag in dead_tags:
            del self.unit_last_health[tag]

        # 同时清理其他微操字典中的死亡单位
        for tag in dead_tags:
            self.unit_ease.pop(tag, None)
            self.unit_listenframe.pop(tag, None)
            self.micro_scan_frame.pop(tag, None)
            self.micro_ha_scan.pop(tag, None)
            self.micro_ho_scan.pop(tag, None)
            self.micro_special_scan_frame.pop(tag, None)
            self.micro_last_enemy_frame.pop(tag, None)

    def _micro_get_combat_tags(self, iteration: int) -> Set[int]:
        """
        [新增] 辅助函数：返回当前所有"正在执行攻击任务"的战斗单位tags集合。
        包括：
        1. 总攻编队中状态为 ATTACKING 的单位
        2. 自动防御中的 defender 单位
        """
        combat_tags = set()

        # Keep the tech gate local to each wave. A specialist in one squad must
        # not switch unrelated basic-bio squads away from their old-like path.
        for wave_id, attack_data in self.total_attack_groups.items():
            if attack_data["state"] == "ATTACKING":
                wave_tags = set(attack_data["unit_tags"])
                if self.micro_require_tech_unit and not self._micro_tags_have_tech_anchor(wave_tags):
                    continue
                combat_tags.update(wave_tags)

        # Defense assignments already receive stable attack orders from
        # automatic_defense(). Keep them old-like unless defense micro is
        # explicitly enabled; otherwise one specialist can pull the whole
        # response group into the Mulobot loop before an attack wave launches.
        if self.micro_defense_enabled:
            for defender_tag in self.active_defense_map:
                combat_tags.add(defender_tag)

        # Keep workers out of the Mulobot-style micro loop unless explicitly
        # requested. BO attack execution may still issue normal A-move orders,
        # but SCVs/MULEs should not be scanned for special combat micro by default.
        if not (self.enable_attack_workers or self.allow_explicit_worker_support) and combat_tags:
            worker_tags = {
                unit.tag
                for unit in self.units.of_type({UnitTypeId.SCV, UnitTypeId.MULE})
            }
            combat_tags.difference_update(worker_tags)

        return combat_tags

    def _micro_tags_have_tech_anchor(self, combat_tags: Set[int]) -> bool:
        return any(
            unit.tag in combat_tags and unit.type_id in self.MICRO_TECH_ANCHOR_TYPES
            for unit in self.units
        )

    def _micro_combat_has_tech_unit(self, combat_tags: Set[int]) -> bool:
        """Enable specialist micro only for a combat set containing its anchor."""
        if not self.micro_require_tech_unit:
            return True
        return self._micro_tags_have_tech_anchor(combat_tags)

    def _micro_goal_for_unit_tag(self, tag: int) -> Optional[Point2]:
        for attack_data in self.total_attack_groups.values():
            if attack_data["state"] == "ATTACKING" and tag in attack_data["unit_tags"]:
                return attack_data.get("final_position") or attack_data.get("rally_point")
        enemy_tag = self.active_defense_map.get(tag)
        if enemy_tag:
            enemy = self.enemy_units.find_by_tag(enemy_tag)
            if enemy:
                return enemy.position
        return None

    def _unit_can_attack_seen_enemy(self, unit, enemy) -> bool:
        if enemy.is_flying:
            return bool(getattr(unit, "can_attack_air", False))
        return bool(getattr(unit, "can_attack_ground", False))

    def _enemy_threat_range_against(self, enemy, unit) -> float:
        if unit.is_flying:
            return getattr(enemy, "air_range", 0) if getattr(enemy, "can_attack_air", False) else 0
        return getattr(enemy, "ground_range", 0) if getattr(enemy, "can_attack_ground", False) else 0

    def _micro_has_static_setup_support(self, unit, combat_allies) -> bool:
        if self.micro_static_setup_min_support <= 0:
            return True

        combat_attackers = [
            ally for ally in combat_allies
            if ally.type_id not in {UnitTypeId.SCV, UnitTypeId.MULE}
            and (
                getattr(ally, "can_attack_ground", False)
                or getattr(ally, "can_attack_air", False)
            )
        ]
        if self.micro_static_setup_ghost_radius > 0 and any(
            ally.type_id == UnitTypeId.GHOST
            and ally.tag != unit.tag
            and ally.distance_to(unit) <= self.micro_static_setup_ghost_radius
            for ally in combat_attackers
        ):
            return True

        support = 0
        for ally in combat_attackers:
            if ally.tag == unit.tag:
                continue
            if ally.distance_to(unit) <= 8:
                support += 1
        if (
            self.micro_static_setup_min_group > 0
            and len(combat_attackers) >= self.micro_static_setup_min_group
            and support >= max(1, self.micro_static_setup_min_support // 2)
        ):
            return True
        return support >= self.micro_static_setup_min_support

    def _basic_hooligan_target_key(self, enemy):
        high_value = {
            UnitTypeId.SIEGETANK,
            UnitTypeId.SIEGETANKSIEGED,
            UnitTypeId.MEDIVAC,
            UnitTypeId.GHOST,
            UnitTypeId.BANSHEE,
            UnitTypeId.RAVEN,
            UnitTypeId.WIDOWMINE,
            UnitTypeId.WIDOWMINEBURROWED,
            UnitTypeId.LIBERATOR,
            UnitTypeId.LIBERATORAG,
        }
        enemy_range = max(getattr(enemy, "ground_range", 0), getattr(enemy, "air_range", 0))
        rank = 0 if enemy.type_id in high_value else 1
        return (rank, -enemy_range, enemy.health + enemy.shield)

    def _micro_basic_hooligan_unit_enabled(self, unit_type: UnitTypeId) -> bool:
        if not self.micro_basic_hooligan_enabled:
            return False
        if unit_type in self.MICRO_BASIC_HOOLIGAN_TYPES:
            return self.micro_hooligan_bio_enabled
        return False

    def _micro_basic_hooligan_attack(self, unit, iteration: int, goal: Optional[Point2]) -> bool:
        if not self._micro_basic_hooligan_unit_enabled(unit.type_id):
            return False

        tag = unit.tag
        pos = unit.position
        my_ground_range = getattr(unit, "ground_range", 0) if getattr(unit, "can_attack_ground", False) else 0
        my_air_range = getattr(unit, "air_range", 0) if getattr(unit, "can_attack_air", False) else 0
        myrange = max(my_ground_range, my_air_range)
        if myrange <= 0:
            return False

        myradius = getattr(unit, "radius", 0.5)
        shootable_units = []
        nearby_threat_power = 0.0
        ally_power = 3.0
        away_x = 0.0
        away_y = 0.0
        away_n = 0

        for ally in self.units:
            if ally.tag == tag or not getattr(ally, "can_attack", False):
                continue
            if ally.type_id in {UnitTypeId.SCV, UnitTypeId.MULE}:
                continue
            if ally.distance_to(unit) < self.micro_hooligan_ally_power_radius:
                ally_power += math.sqrt(max(1.0, ally.health + ally.shield))

        ignored = {
            UnitTypeId.LARVA,
            UnitTypeId.EGG,
            UnitTypeId.OVERLORD,
            UnitTypeId.OBSERVER,
            UnitTypeId.OBSERVERSIEGEMODE,
        }
        for enemy in self.enemy_units:
            if not enemy.is_visible or enemy.type_id in ignored:
                continue
            dist = unit.distance_to(enemy)
            eradius = getattr(enemy, "radius", 0.5)
            radii = myradius + eradius
            if self._unit_can_attack_seen_enemy(unit, enemy) and dist <= myrange + radii:
                shootable_units.append(enemy)

            threat_range = self._enemy_threat_range_against(enemy, unit)
            if threat_range > 0 and dist <= threat_range + radii + 2.0:
                nearby_threat_power += math.sqrt(max(1.0, enemy.health + enemy.shield))
                if self._unit_can_attack_seen_enemy(unit, enemy) and threat_range < myrange:
                    away = enemy.position.towards(pos, 1.0)
                    away_x += away.x - enemy.position.x
                    away_y += away.y - enemy.position.y
                    away_n += 1

        shootable_structures = []
        nearest_visible_enemy_unit_dist = min((unit.distance_to(enemy) for enemy in self.enemy_units if enemy.is_visible and enemy.type_id not in ignored), default=999.0)
        structure_exclusion_distance = (
            22.0
            if self.attack_target_units_first
            else max(myrange + myradius + 3.0, 8.0)
        )
        allow_structure_focus = (
            self.hooligan_structure_focus_enabled
            and (nearest_visible_enemy_unit_dist > structure_exclusion_distance)
            and (goal is None or unit.distance_to(goal) <= max(myrange + myradius + 2.0, 7.0))
        )
        if allow_structure_focus and getattr(unit, "can_attack_ground", False):
            for structure in self.enemy_structures:
                if structure.is_flying:
                    continue
                dist = unit.distance_to(structure)
                radii = myradius + getattr(structure, "radius", 1.0)
                if dist <= myrange + radii:
                    shootable_structures.append(structure)

        shootables = sorted(shootable_units, key=self._basic_hooligan_target_key)
        if shootables and unit.weapon_cooldown <= 3:
            unit.attack(shootables[0])
            self.unit_listenframe[tag] = iteration + self.MICRO_LISTEN_INTERVAL
            self._record_micro_event("hooligan:focus_unit", unit)
            return True

        if shootable_structures and not shootables and unit.weapon_cooldown <= 3:
            target = min(shootable_structures, key=lambda structure: structure.health + structure.shield)
            unit.attack(target)
            self.unit_listenframe[tag] = iteration + self.MICRO_LISTEN_INTERVAL
            self._record_micro_event("hooligan:focus_structure", unit)
            return True

        if shootables or nearby_threat_power > 0:
            outnumbered = nearby_threat_power >= ally_power
            cooldown_kite = bool(shootables) and unit.weapon_cooldown > 3 and outnumbered
            should_step = self.micro_hooligan_retreat_enabled and (
                cooldown_kite or outnumbered
            )
            if should_step:
                if away_n > 0:
                    step = Point2((pos.x + 0.8 * away_x / away_n, pos.y + 0.8 * away_y / away_n))
                elif goal is not None and unit.distance_to(goal) > 4:
                    step = pos.towards(goal, 0.5)
                else:
                    step = pos.towards(self.start_location, 0.7)
                unit.move(self._clamp_position_to_map_bounds(step))
                self.unit_listenframe[tag] = iteration + self.MICRO_LISTEN_INTERVAL
                self._record_micro_event("hooligan:step", unit)
                return True

        return False

    async def micro_dodge(self, iteration: int):
        """
        [新增] 闪避危险效果 - 最高优先级。
        参考 army.py 的 dodge_delayed() + do_dodge()。

        处理流程：
        1. 维护延迟效果表（核弹、腐蚀胆汁）
        2. 标记需闪避的单位
        3. 计算合成逃离向量并执行移动
        """
        # 帧率转换: sc2 约22.4帧/秒, 这里的iteration是on_step的iteration
        # army.py中使用 self.seconds 作为帧率乘数，这里简化处理
        SECONDS = 22  # 近似帧率

        # ---- Step 1: 维护延迟效果表 ----
        # 清理过期效果
        expired = set()
        for effect_tuple in self.dodge_effects:
            (effect_id, pos, radius, start_frame, end_frame) = effect_tuple
            if iteration > end_frame:
                expired.add(effect_tuple)
        for effect_tuple in expired:
            (effect_id, pos, radius, start_frame, end_frame) = effect_tuple
            self.dodge_positions.discard(pos)
        self.dodge_effects -= expired

        # 扫描新的延迟效果 (参考 army.py dodge_delayed)
        for effect in self.state.effects:
            if effect.id == EffectId.RAVAGERCORROSIVEBILECP:
                for pos in effect.positions:
                    if pos not in self.dodge_positions:
                        self.dodge_positions.add(pos)
                        # 胆汁: 约1.3秒后落地, 持续约2.7秒
                        pair = (effect.id, pos, effect.radius, iteration + 30, iteration + 60)
                        self.dodge_effects.add(pair)
            if effect.id == EffectId.NUKEPERSISTENT:
                for pos in effect.positions:
                    if pos not in self.dodge_positions:
                        self.dodge_positions.add(pos)
                        # 核弹: 约10秒后落地
                        pair = (effect.id, pos, effect.radius, iteration + 10 * SECONDS, iteration + 14 * SECONDS)
                        self.dodge_effects.add(pair)

        # ---- Step 2: 收集当前需要闪避的区域 ----
        # 参考 army.py 的 tododge
        tododge = set()  # (pos, radius)

        # 延迟效果中正在生效的
        for effect_tuple in self.dodge_effects:
            (effect_id, epos, radius, start_frame, end_frame) = effect_tuple
            if start_frame < iteration < end_frame:
                tododge.add((epos, radius))

        # 即时效果 (解放者锁定区等)
        delayed_effect_types = {EffectId.RAVAGERCORROSIVEBILECP, EffectId.NUKEPERSISTENT}
        for effect in self.state.effects:
            if effect.id not in delayed_effect_types:
                if effect.is_enemy:
                    for epos in effect.positions:
                        tododge.add((epos, effect.radius))

        # ---- Step 3: 标记需要闪避的单位 ----
        self.dodging_tags = set()
        margin = 2.0  # 安全边距, 参考 army.py

        if not tododge:
            return  # 没有危险效果，直接返回

        combat_tags = self._micro_get_combat_tags(iteration)
        if not combat_tags:
            return
        if not self._micro_combat_has_tech_unit(combat_tags):
            return

        non_combat_types = {UnitTypeId.SCV, UnitTypeId.MULE}
        for unit in self.units:
            if unit.tag not in combat_tags:
                continue
            if unit.type_id in non_combat_types:
                continue
            pos = unit.position
            for (epos, radius) in tododge:
                if pos.distance_to(epos) < radius + margin:
                    self.dodging_tags.add(unit.tag)
                    break  # 只要在一个危险区内就标记

        # ---- Step 4: 执行闪避移动 ----
        # 参考 army.py do_dodge 中的 delta 向量迭代计算
        for tag in self.dodging_tags:
            unit = None
            for u in self.units:
                if u.tag == tag:
                    unit = u
                    break
            if unit is None:
                continue

            # 帧控制：避免重复下令
            if self.unit_listenframe.get(tag, 0) > iteration:
                continue

            pos = unit.position
            # 迭代计算逃离向量 (最多3次迭代, 参考army.py)
            moves = 0
            moveit = False
            changed = True
            calc_pos = Point2((pos.x, pos.y))

            while moves < 3 and changed:
                changed = False
                moves += 1
                delta = Point2((0, 0))
                for (epos, radius) in tododge:
                    if calc_pos.distance_to(epos) < radius + margin:
                        changed = True
                        moveit = True
                        if calc_pos.distance_to(epos) < 0.2:
                            # 太近了, 往基地方向跑
                            gopos = calc_pos.towards(self.start_location, radius + margin)
                        else:
                            # 沿反方向逃离, 偏向基地方向
                            # 参考 army.py: shadowpos = pos.towards(ourmain, 2)
                            shadowpos = calc_pos.towards(self.start_location, 2)
                            gopos = epos.towards(shadowpos, radius + margin)
                        delta = Point2((delta.x + gopos.x - calc_pos.x,
                                        delta.y + gopos.y - calc_pos.y))
                if changed:
                    calc_pos = Point2((calc_pos.x + delta.x, calc_pos.y + delta.y))

            if moveit:
                unit.move(calc_pos)
                self.unit_listenframe[tag] = iteration + self.MICRO_LISTEN_INTERVAL
                self._record_micro_event("dodge", unit)
                # 闪避中的单位清除hist_order, 避免总攻系统覆盖闪避指令
                # (如果 hist_order 存在于 base_player 中)

    async def micro_special_abilities(self, iteration: int):
        """Mulebot-inspired combat abilities for units used by the fixed BO suite."""
        if not self.micro_special_abilities_enabled:
            return
        combat_tags = self._micro_get_combat_tags(iteration)
        if not combat_tags:
            return
        if not self._micro_combat_has_tech_unit(combat_tags):
            return

        visible_enemies = [enemy for enemy in self.enemy_units if enemy.is_visible]
        repair_assignments = Counter()
        combat_allies = [ally for ally in self.units if ally.tag in combat_tags]
        has_stim = UpgradeId.STIMPACK in self.state.upgrades

        for unit in self.units:
            tag = unit.tag
            if tag not in combat_tags or tag in self.dodging_tags:
                continue
            if unit.type_id not in self.MICRO_SPECIAL_ABILITY_TYPES:
                continue
            if self.micro_special_scan_frame.get(tag, 0) > iteration:
                continue
            if self.unit_listenframe.get(tag, 0) > iteration:
                continue
            if unit.type_id in self.MICRO_BASIC_HOOLIGAN_TYPES and not self.micro_special_bio_enabled:
                self._record_micro_event("policy:bio_special_passthrough", unit)
                continue
            self.micro_special_scan_frame[tag] = iteration + 11

            nearby = sorted(visible_enemies, key=lambda enemy: unit.distance_to(enemy))
            ground_near = [enemy for enemy in nearby if not enemy.is_flying and unit.distance_to(enemy) <= 15]
            air_near = [enemy for enemy in nearby if enemy.is_flying and unit.distance_to(enemy) <= 15]
            if ground_near or air_near:
                self.micro_last_enemy_frame[tag] = iteration

            abilities = (
                set()
                if unit.type_id in self.MICRO_BASIC_A_MOVE_TYPES
                or self._micro_basic_hooligan_unit_enabled(unit.type_id)
                else await self._available_micro_abilities(unit)
            )
            issued = False

            if unit.type_id in self.MICRO_BASIC_HOOLIGAN_TYPES:
                self._record_micro_event(
                    "policy:basic_hooligan_active"
                    if self._micro_basic_hooligan_unit_enabled(unit.type_id)
                    else "policy:basic_hooligan_passthrough",
                    unit,
                )
            elif unit.type_id in self.MICRO_BASIC_A_MOVE_TYPES:
                self._record_micro_event("policy:worker_amove_passthrough", unit)

            if unit.type_id == UnitTypeId.SCV:
                repair_target = self._select_scv_repair_target(
                    unit, combat_allies, repair_assignments
                )
                if repair_target is not None and self.minerals > 0:
                    unit(AbilityId.EFFECT_REPAIR_SCV, repair_target)
                    repair_assignments[repair_target.tag] += 1
                    self._record_micro_event("ability:scv_repair", unit)
                    issued = True

            elif unit.type_id in {UnitTypeId.MARINE, UnitTypeId.MARAUDER}:
                stim_ability = (
                    AbilityId.EFFECT_STIM_MARINE
                    if unit.type_id == UnitTypeId.MARINE
                    else getattr(AbilityId, "EFFECT_STIM_MARAUDER", None)
                )
                if (
                    nearby
                    and unit.distance_to(nearby[0]) <= 10
                    and has_stim
                    and stim_ability is not None
                    and unit.health >= 29
                    and not unit.has_buff(BuffId.STIMPACK)
                ):
                    unit(stim_ability)
                    self._record_micro_event("ability:stim", unit)
                    issued = True

            elif unit.type_id == UnitTypeId.GHOST:
                snipe_targets = [
                    enemy for enemy in nearby
                    if unit.distance_to(enemy) <= 10.5
                    and getattr(enemy, "is_biological", False)
                    and enemy.health + enemy.shield >= self.micro_ghost_snipe_min_hp
                ]
                emp_targets = [
                    enemy for enemy in nearby
                    if unit.distance_to(enemy) <= 10.5 and enemy.energy + enemy.shield >= 90
                ]
                if AbilityId.EFFECT_GHOSTSNIPE in abilities and snipe_targets:
                    target = max(snipe_targets, key=lambda enemy: enemy.health + enemy.shield + enemy.energy)
                    unit(AbilityId.EFFECT_GHOSTSNIPE, target)
                    self._record_micro_event("ability:ghost_snipe", unit)
                    issued = True
                elif (
                    AbilityId.EMP_EMP in abilities
                    and len(emp_targets) >= self.micro_ghost_emp_min_targets
                ):
                    target = max(emp_targets, key=lambda enemy: enemy.energy + enemy.shield)
                    unit(AbilityId.EMP_EMP, target.position)
                    self._record_micro_event("ability:ghost_emp", unit)
                    issued = True

            elif unit.type_id == UnitTypeId.BANSHEE:
                air_threat = any(enemy.can_attack_air and unit.distance_to(enemy) <= 10 for enemy in nearby)
                took_damage = self.unit_last_health.get(tag, unit.health + unit.shield) > unit.health + unit.shield
                if (
                    not unit.is_cloaked
                    and (air_threat or took_damage)
                    and AbilityId.BEHAVIOR_CLOAKON_BANSHEE in abilities
                    and unit.energy >= 40
                ):
                    unit(AbilityId.BEHAVIOR_CLOAKON_BANSHEE)
                    self._record_micro_event("ability:banshee_cloak_on", unit)
                    issued = True
                elif (
                    unit.is_cloaked
                    and not air_threat
                    and iteration - self.micro_last_enemy_frame.get(tag, iteration) > 220
                    and AbilityId.BEHAVIOR_CLOAKOFF_BANSHEE in abilities
                ):
                    unit(AbilityId.BEHAVIOR_CLOAKOFF_BANSHEE)
                    self._record_micro_event("ability:banshee_cloak_off", unit)
                    issued = True

            elif unit.type_id == UnitTypeId.MEDIVAC:
                patients = [
                    ally for ally in self.units
                    if ally.tag != tag
                    and ally.tag in combat_tags
                    and getattr(ally, "is_biological", False)
                    and ally.health_percentage < 0.999
                ]
                if patients:
                    patient = min(patients, key=lambda ally: (ally.health_percentage, unit.distance_to(ally)))
                    if unit.distance_to(patient) <= 4.5 and AbilityId.MEDIVACHEAL_HEAL in abilities:
                        unit(AbilityId.MEDIVACHEAL_HEAL, patient)
                        self._record_micro_event("ability:medivac_heal", unit)
                    else:
                        unit.move(patient.position.towards(self.start_location, 3.5))
                        self._record_micro_event("move:medivac_follow", unit)
                    issued = True

            elif unit.type_id == UnitTypeId.SIEGETANK:
                siege_targets = [enemy for enemy in ground_near if 4.0 < unit.distance_to(enemy) <= 13]
                siege_structures = [
                    structure for structure in self.enemy_structures
                    if not structure.is_flying and 4.0 < unit.distance_to(structure) <= 13
                ]
                should_siege = (
                    bool(siege_structures)
                    or bool(siege_targets)
                    or len(siege_targets) >= 2
                    or any(enemy.is_armored or enemy.health + enemy.shield >= 120 for enemy in siege_targets)
                )
                if should_siege and AbilityId.SIEGEMODE_SIEGEMODE in abilities:
                    if self._micro_has_static_setup_support(unit, combat_allies):
                        unit(AbilityId.SIEGEMODE_SIEGEMODE)
                        self._record_micro_event("ability:tank_siege", unit)
                        issued = True
                    else:
                        self._record_micro_event("policy:static_setup_wait_support", unit)

            elif unit.type_id == UnitTypeId.SIEGETANKSIEGED:
                nearby_structures = [
                    structure for structure in self.enemy_structures
                    if not structure.is_flying and unit.distance_to(structure) <= 13
                ]
                if (
                    not ground_near
                    and not nearby_structures
                    and iteration - self.micro_last_enemy_frame.get(tag, iteration) > 88
                    and AbilityId.UNSIEGE_UNSIEGE in abilities
                ):
                    unit(AbilityId.UNSIEGE_UNSIEGE)
                    self._record_micro_event("ability:tank_unsiege", unit)
                    issued = True
                elif ground_near and unit.weapon_cooldown <= 6:
                    victims = [enemy for enemy in ground_near if 2 < unit.distance_to(enemy) < 13]
                    if victims:
                        target = max(
                            victims,
                            key=lambda enemy: (
                                sum(other.distance_to(enemy) < 1.25 for other in victims),
                                int(enemy.is_armored),
                                enemy.health + enemy.shield,
                            ),
                        )
                        unit.attack(target)
                        self._record_micro_event("attack:tank_focus", unit)
                        issued = True

            elif unit.type_id == UnitTypeId.LIBERATOR:
                close_ground = [enemy for enemy in ground_near if unit.distance_to(enemy) <= 13]
                if close_ground and AbilityId.MORPH_LIBERATORAGMODE in abilities:
                    if self._micro_has_static_setup_support(unit, combat_allies):
                        target = min(close_ground, key=lambda enemy: unit.distance_to(enemy))
                        unit(AbilityId.MORPH_LIBERATORAGMODE, target.position)
                        self._record_micro_event("ability:liberator_ag", unit)
                        issued = True
                    else:
                        self._record_micro_event("policy:static_setup_wait_support", unit)

            elif unit.type_id == UnitTypeId.LIBERATORAG:
                if (
                    not ground_near
                    and iteration - self.micro_last_enemy_frame.get(tag, iteration) > 88
                    and AbilityId.MORPH_LIBERATORAAMODE in abilities
                ):
                    unit(AbilityId.MORPH_LIBERATORAAMODE)
                    self._record_micro_event("ability:liberator_aa", unit)
                    issued = True

            elif unit.type_id == UnitTypeId.VIKINGFIGHTER:
                close_ground = [enemy for enemy in ground_near if unit.distance_to(enemy) <= 6]
                if not air_near and close_ground and AbilityId.MORPH_VIKINGASSAULTMODE in abilities:
                    if self._micro_has_static_setup_support(unit, combat_allies):
                        unit(AbilityId.MORPH_VIKINGASSAULTMODE)
                        self._record_micro_event("ability:viking_assault", unit)
                        issued = True
                    else:
                        self._record_micro_event("policy:static_setup_wait_support", unit)

            elif unit.type_id == UnitTypeId.VIKINGASSAULT:
                if (air_near or not ground_near) and AbilityId.MORPH_VIKINGFIGHTERMODE in abilities:
                    unit(AbilityId.MORPH_VIKINGFIGHTERMODE)
                    self._record_micro_event("ability:viking_fighter", unit)
                    issued = True

            elif unit.type_id == UnitTypeId.RAVEN:
                matrix_targets = [
                    enemy for enemy in nearby
                    if unit.distance_to(enemy) <= 9
                    and (getattr(enemy, "is_mechanical", False) or enemy.is_armored)
                ]
                if matrix_targets and AbilityId.EFFECT_INTERFERENCEMATRIX in abilities:
                    target = max(matrix_targets, key=lambda enemy: enemy.health + enemy.shield)
                    unit(AbilityId.EFFECT_INTERFERENCEMATRIX, target)
                    self._record_micro_event("ability:raven_matrix", unit)
                    issued = True
                elif ground_near and AbilityId.BUILDAUTOTURRET_AUTOTURRET in abilities:
                    target = min(ground_near, key=lambda enemy: unit.distance_to(enemy))
                    turret_pos = target.position.towards(unit.position, 2)
                    unit(AbilityId.BUILDAUTOTURRET_AUTOTURRET, turret_pos)
                    self._record_micro_event("ability:raven_turret", unit)
                    issued = True

            elif unit.type_id == UnitTypeId.WIDOWMINE:
                if ground_near and AbilityId.BURROWDOWN_WIDOWMINE in abilities:
                    if self._micro_has_static_setup_support(unit, combat_allies):
                        unit(AbilityId.BURROWDOWN_WIDOWMINE)
                        self._record_micro_event("ability:widowmine_burrow", unit)
                        issued = True
                    else:
                        self._record_micro_event("policy:static_setup_wait_support", unit)

            elif unit.type_id == UnitTypeId.WIDOWMINEBURROWED:
                if (
                    not ground_near
                    and iteration - self.micro_last_enemy_frame.get(tag, iteration) > 132
                    and AbilityId.BURROWUP_WIDOWMINE in abilities
                ):
                    unit(AbilityId.BURROWUP_WIDOWMINE)
                    self._record_micro_event("ability:widowmine_unburrow", unit)
                    issued = True

            elif unit.type_id == UnitTypeId.BATTLECRUISER:
                if unit.health_percentage < 0.3 and AbilityId.EFFECT_TACTICALJUMP in abilities:
                    unit(AbilityId.EFFECT_TACTICALJUMP, self.start_location)
                    self._record_micro_event("ability:bc_tactical_jump", unit)
                    issued = True
                else:
                    yamato_targets = [
                        enemy for enemy in nearby
                        if unit.distance_to(enemy) <= 10 and enemy.health + enemy.shield >= 200
                    ]
                    if yamato_targets and AbilityId.YAMATO_YAMATOGUN in abilities:
                        target = max(yamato_targets, key=lambda enemy: enemy.health + enemy.shield)
                        unit(AbilityId.YAMATO_YAMATOGUN, target)
                        self._record_micro_event("ability:bc_yamato", unit)
                        issued = True

            elif unit.type_id == UnitTypeId.REAPER:
                grenade_targets = [enemy for enemy in ground_near if unit.distance_to(enemy) <= 5]
                if grenade_targets and AbilityId.KD8CHARGE_KD8CHARGE in abilities:
                    target = min(grenade_targets, key=lambda enemy: unit.distance_to(enemy))
                    unit(AbilityId.KD8CHARGE_KD8CHARGE, target.position)
                    self._record_micro_event("ability:reaper_grenade", unit)
                    issued = True

            elif unit.type_id == UnitTypeId.THOR:
                if air_near and AbilityId.MORPH_THORHIGHIMPACTMODE in abilities:
                    unit(AbilityId.MORPH_THORHIGHIMPACTMODE)
                    self._record_micro_event("ability:thor_high_impact", unit)
                    issued = True

            if issued:
                self.unit_listenframe[tag] = iteration + self.MICRO_LISTEN_INTERVAL

    async def micro_fight(self, iteration: int):
        if not self.micro_fight_enabled:
            return
        """
        [新增] 战斗微操核心函数 - 走位、甩枪、兴奋剂/幽灵隐身。
        参考 army.py 的 fight() 函数。

        只处理满足以下条件的作战单位:
        - 属于ATTACKING状态的总攻编队 或 自动防御中的单位
        - 不在闪避中 (dodging_tags)
        - 帧控制允许 (listenframe < iteration)

        处理流程:
        A. 触发判断 (react) - 是否进入战斗状态
        B. 兴奋剂释放
        C. 判断是否在战斗中 (fighting)
        D. 战斗走位 (wishdist计算)
        E. 射击目标选择
        F. 执行最终指令
        """
        # 获取需要微操的战斗单位
        combat_tags = self._micro_get_combat_tags(iteration)
        if not combat_tags:
            return
        if not self._micro_combat_has_tech_unit(combat_tags):
            return

        # 构建敌人信息缓存 (避免在循环内重复访问)
        # 参考 army.py 的 enemy_unit_mem
        enemy_info = {}  # tag -> (type_id, position, health, shield, is_flying, ground_range, air_range, radius)
        all_enemies = self.enemy_units | self.enemy_structures
        for ene in self.enemy_units:
            if ene.is_visible:
                e_ground_range = 0
                e_air_range = 0
                # 简化射程获取: 使用 sc2 API
                if ene.can_attack_ground:
                    e_ground_range = ene.ground_range
                if ene.can_attack_air:
                    e_air_range = ene.air_range
                e_radius = ene.radius
                enemy_info[ene.tag] = (
                    ene.type_id, ene.position, ene.health, ene.shield,
                    ene.is_flying, e_ground_range, e_air_range, e_radius
                )

        # 遍历所有战斗单位执行微操
        for unit in self.units:
            tag = unit.tag

            # 过滤条件
            if tag not in combat_tags:
                continue
            if unit.type_id in self.MICRO_STATIONARY_TYPES:
                continue
            if tag in self.dodging_tags:
                continue
            if self.unit_listenframe.get(tag, 0) > iteration:
                continue
            if self._micro_basic_hooligan_unit_enabled(unit.type_id):
                goal = self._micro_goal_for_unit_tag(tag)
                if self._micro_basic_hooligan_attack(unit, iteration, goal):
                    continue
            if unit.type_id in self.MICRO_BASIC_HOOLIGAN_TYPES and not self.micro_bio_fight_enabled:
                self._record_micro_event("policy:bio_fight_passthrough", unit)
                continue
            if unit.type_id in self.MICRO_BASIC_A_MOVE_TYPES:
                continue

            my_ground_range = getattr(unit, "ground_range", 0) if unit.can_attack_ground else 0
            my_air_range = getattr(unit, "air_range", 0) if unit.can_attack_air else 0
            myrange = max(my_ground_range, my_air_range)
            if myrange <= 0:
                continue
            myradi = getattr(unit, "radius", 0.5)
            my_is_flying = unit.is_flying
            my_can_hit_ground = my_ground_range > 0
            my_can_hit_air = my_air_range > 0
            pos = unit.position

            # ---- Step A: 触发判断 (react) ----
            # 参考 army.py: weapon_cooldown > 0 / danger >= 0.3 / 掉血
            react = False

            if unit.weapon_cooldown > 0:
                react = True

            # 检测掉血
            last_hp = self.unit_last_health.get(tag)
            current_hp = unit.health + unit.shield
            if last_hp is not None and current_hp < last_hp:
                react = True

            # 检测附近是否有敌人 (简化版danger检测)
            nearby_enemy_count = 0
            for etag, einfo in enemy_info.items():
                (etyp, epos, ehp, eshield, efly, egr, ear, erad) = einfo
                if pos.distance_to(epos) < 10:
                    nearby_enemy_count += 1
            if nearby_enemy_count > 0:
                react = True

            # 更新血量快照 (在react判断之后更新，确保本帧的判断基于上一帧数据)
            self.unit_last_health[tag] = current_hp

            if react:
                # 检查是否是首次进入战斗 (参考 army.py 的 wasatease)
                wasatease = True
                if tag in self.unit_ease:
                    if iteration < self.unit_ease[tag]:
                        wasatease = False

                # 延长战斗状态窗口
                self.unit_ease[tag] = iteration + self.MICRO_EASE_DURATION

                # ---- Step B: 特殊战斗技能 (首次进入战斗时触发) ----
                # 参考 army.py: Marine/Marauder 开兴奋剂，Ghost 可隐身。
                if wasatease and unit.type_id == UnitTypeId.GHOST:
                    cloak_ability = getattr(AbilityId, "BEHAVIOR_CLOAKON_GHOST", None)
                    if (
                        cloak_ability is not None
                        and UpgradeId.PERSONALCLOAKING in self.state.upgrades
                        and not unit.is_cloaked
                        and unit.energy >= 30
                    ):
                        unit(cloak_ability)
                        self.unit_ease[tag] = iteration + 7 * 22
                        self.unit_listenframe[tag] = iteration + self.MICRO_LISTEN_INTERVAL
                        continue

            else:
                # 没有react, 更新血量快照后跳过
                self.unit_last_health[tag] = current_hp
                continue

            # ---- Step C: 判断是否在战斗中 (fighting) ----
            # 参考 army.py: ease窗口内 且 danger < 1.5
            fighting = False
            if tag in self.unit_ease:
                if iteration < self.unit_ease[tag]:
                    fighting = True

            # 威胁过高时退出战斗 (参考 army.py: danger >= 1.5)
            # 简化: 附近敌人远多于友军时视为太危险
            if fighting and nearby_enemy_count > 0:
                nearby_ally_count = 0
                for ally in self.units:
                    if ally.tag != tag and ally.can_attack:
                        if ally.type_id != UnitTypeId.SCV and ally.type_id != UnitTypeId.MULE:
                            if pos.distance_to(ally.position) < 8:
                                nearby_ally_count += 1
                if nearby_enemy_count > (nearby_ally_count + 1) * 3:
                    fighting = False

            if not fighting:
                # 不在战斗中, 让总攻系统的A-Move继续生效
                self.unit_last_health[tag] = current_hp
                continue

            self._record_micro_event("fight:engaged", unit)

            # ---- Step D: 战斗走位 ----
            # 参考 army.py fight() 的核心走位逻辑

            tomove = Point2((0, 0))
            moveit = False
            mayshoot = True  # 战斗中允许射击

            # D.1 扫描附近敌人 (每 MICRO_SCAN_INTERVAL 帧刷新)
            # 参考 army.py 的 ha_scan / ho_scan
            if self.micro_scan_frame.get(tag, 0) <= iteration:
                self.micro_scan_frame[tag] = iteration + self.MICRO_SCAN_INTERVAL
                self.micro_ha_scan[tag] = set()
                self.micro_ho_scan[tag] = set()

                for etag, einfo in enemy_info.items():
                    (etyp, epos, ehp, eshield, efly, egr, ear, erad) = einfo
                    dist = pos.distance_to(epos)

                    # ho_scan: 能打到我的敌人
                    enemy_effective_range = ear if my_is_flying else egr
                    if enemy_effective_range > 0 and dist < max(myrange + 5, enemy_effective_range + 5):
                        self.micro_ho_scan[tag].add(etag)

                    # ha_scan: 我能打到的敌人
                    can_hit_enemy = (efly and my_can_hit_air) or ((not efly) and my_can_hit_ground)
                    if can_hit_enemy and dist < max(myrange + 5, 10):
                        self.micro_ha_scan[tag].add(etag)

            # D.2 远离太近的敌人 (参考 army.py 的 wishdist 计算)
            ho_scan = self.micro_ho_scan.get(tag, set())
            ha_scan = self.micro_ha_scan.get(tag, set())

            for etag in ho_scan:
                if etag not in enemy_info:
                    continue
                (etyp, epos, ehp, eshield, efly, egr, ear, erad) = enemy_info[etag]

                # 确定敌人的有效射程 (按我方是否为空军切换)
                erange = ear if my_is_flying else egr
                if erange <= 0:
                    erange = 0.1  # 近战单位

                # 计算理想距离 (参考 army.py 的三种情况)
                radii = myradi + erad

                if etyp == UnitTypeId.SIEGETANKSIEGED:
                    # 架设坦克: 贴脸打 (在其最小射程内)
                    wishdist_min = radii
                    wishdist = radii
                    wishdist_max = myrange + radii - 0.1
                elif myrange <= erange:
                    # 我射程短或相等: 靠近打
                    wishdist_min = radii
                    wishdist = myrange / 2 + radii
                    wishdist_max = myrange + radii - 0.1
                else:
                    # 我射程长 (Marine vs 近战): 拉打
                    wishdist_min = (myrange + erange) / 2 + radii
                    wishdist = 0.75 * myrange + erange / 4 + radii
                    wishdist_max = myrange + radii - 0.1

                dist = pos.distance_to(epos)

                if dist < wishdist_min:
                    # 太近了, 需要后退
                    if dist < 0.1:
                        # 重叠了, 往基地方向跑
                        togo = pos.towards(self.start_location, 3)
                    else:
                        # 正常后退到理想距离
                        togo = epos.towards(pos, wishdist)
                    tomove = Point2((tomove.x + togo.x - pos.x,
                                     tomove.y + togo.y - pos.y))
                    moveit = True

            # D.3 靠近可打的敌人 (如果没在逃跑)
            # 参考 army.py: 找 ha_scan 中最近的在 wishdist_max 之外的敌人
            if not moveit:
                mindist = myrange * 2
                best_togo = None
                for etag in ha_scan:
                    if etag not in enemy_info:
                        continue
                    (etyp, epos, ehp, eshield, efly, egr, ear, erad) = enemy_info[etag]

                    erange = ear if my_is_flying else egr
                    if erange <= 0:
                        erange = 0.1

                    radii = myradi + erad

                    if etyp == UnitTypeId.SIEGETANKSIEGED:
                        wishdist = radii
                        wishdist_max = myrange + radii - 0.1
                    elif myrange <= erange:
                        wishdist = myrange / 2 + radii
                        wishdist_max = myrange + radii - 0.1
                    else:
                        wishdist = 0.75 * myrange + erange / 4 + radii
                        wishdist_max = myrange + radii - 0.1

                    dist = pos.distance_to(epos)

                    if wishdist_max < dist < mindist:
                        # 在射程外但最近, 向其靠近
                        mindist = dist
                        best_togo = epos.towards(pos, wishdist)

                if best_togo is not None:
                    tomove = Point2((best_togo.x - pos.x, best_togo.y - pos.y))
                    moveit = True

            # D.4 攻击建筑时微步靠近 (参考 army.py)
            if len(ha_scan) == 0 and len(ho_scan) == 0:
                if unit.weapon_cooldown >= 3:
                    for order in unit.orders:
                        if order.ability.exact_id == AbilityId.ATTACK_ATTACK:
                            if type(order.target) == int:
                                for stru in self.enemy_structures:
                                    if stru.tag == order.target:
                                        stru_togo = stru.position.towards(pos, stru.radius + 2)
                                        stru_dist = pos.distance_to(stru_togo)
                                        if stru_dist >= 1.5:
                                            stru_togo = pos.towards(stru_togo, 1.0)
                                            tomove = Point2((stru_togo.x - pos.x, stru_togo.y - pos.y))
                                            moveit = True
                                        elif stru_dist >= 0.5:
                                            tomove = Point2((stru_togo.x - pos.x, stru_togo.y - pos.y))
                                            moveit = True

            # ---- Step E & F: 最终指令 - 移动或射击 ----
            # 参考 army.py fight() 的最终 moveit/mayshoot 逻辑
            if moveit:
                goal = Point2((pos.x + tomove.x, pos.y + tomove.y))

                if mayshoot and unit.weapon_cooldown <= 3:
                    # 武器即将冷却完毕, 停下来射击!
                    # 选择最高价值目标 (参考 army.py: eenergy - ehealth)
                    victim_worth = -99999
                    victim_unit = None

                    for etag in ha_scan:
                        if etag not in enemy_info:
                            continue
                        (etyp, epos, ehp, eshield, efly, egr, ear, erad) = enemy_info[etag]

                        dist = pos.distance_to(epos)
                        radii = myradi + erad

                        # 必须在射程内才能打
                        if dist < myrange + radii:
                            # 价值评估: 参考 army.py 的 eworth = eenergy - ehealth
                            # 对于没有能量的单位, 优先打低血量的
                            eworth = -(ehp + eshield)

                            # 额外优先级: 高价值目标加分
                            if etyp in {UnitTypeId.MEDIVAC, UnitTypeId.SIEGETANKSIEGED,
                                        UnitTypeId.HIGHTEMPLAR, UnitTypeId.INFESTOR,
                                        UnitTypeId.VIPER, UnitTypeId.GHOST,
                                        UnitTypeId.WIDOWMINE, UnitTypeId.WIDOWMINEBURROWED,
                                        UnitTypeId.BANSHEE, UnitTypeId.RAVEN}:
                                eworth += 200
                            elif etyp in {UnitTypeId.QUEEN, UnitTypeId.HYDRALISK,
                                          UnitTypeId.STALKER, UnitTypeId.SENTRY}:
                                eworth += 50

                            if eworth > victim_worth:
                                victim_worth = eworth
                                # 找到实际的unit对象用于attack命令
                                for ene in self.enemy_units:
                                    if ene.tag == etag:
                                        victim_unit = ene
                                        break

                    if victim_unit is not None:
                        # 停下射击高价值目标
                        unit.attack(victim_unit)
                        self._record_micro_event("fight:focus_fire", unit)
                    else:
                        # 没找到可射击目标, 继续走位
                        unit.move(goal)
                        self._record_micro_event("fight:kite_move", unit)
                else:
                    # 武器冷却中, 走位
                    unit.move(goal)
                    self._record_micro_event("fight:kite_move", unit)

                self.unit_listenframe[tag] = iteration + self.MICRO_LISTEN_INTERVAL

            # 即使没有移动指令, 也尝试选择更好的射击目标
            elif mayshoot and unit.weapon_cooldown <= 3 and len(ha_scan) > 0:
                victim_worth = -99999
                victim_unit = None

                for etag in ha_scan:
                    if etag not in enemy_info:
                        continue
                    (etyp, epos, ehp, eshield, efly, egr, ear, erad) = enemy_info[etag]

                    dist = pos.distance_to(epos)
                    radii = myradi + erad

                    if dist < myrange + radii:
                        eworth = -(ehp + eshield)
                        if etyp in {UnitTypeId.MEDIVAC, UnitTypeId.SIEGETANKSIEGED,
                                    UnitTypeId.HIGHTEMPLAR, UnitTypeId.INFESTOR,
                                    UnitTypeId.VIPER, UnitTypeId.GHOST,
                                    UnitTypeId.WIDOWMINE, UnitTypeId.WIDOWMINEBURROWED,
                                    UnitTypeId.BANSHEE, UnitTypeId.RAVEN}:
                            eworth += 200
                        elif etyp in {UnitTypeId.QUEEN, UnitTypeId.HYDRALISK,
                                      UnitTypeId.STALKER, UnitTypeId.SENTRY}:
                            eworth += 50

                        if eworth > victim_worth:
                            victim_worth = eworth
                            for ene in self.enemy_units:
                                if ene.tag == etag:
                                    victim_unit = ene
                                    break

                if victim_unit is not None:
                    unit.attack(victim_unit)
                    self._record_micro_event("fight:focus_fire", unit)
                    self.unit_listenframe[tag] = iteration + self.MICRO_LISTEN_INTERVAL

    # ==========================================
    # [新增结束] 战斗微操系统
    # ==========================================

    # ==========================================
    # 1. 工人与资源管理脚本
    # ==========================================
    async def distribute_workers(self, resource_ratio: float = 2.0) -> None:
        """
        根据全局矿气比分配工人。
        1. 优先保证气矿（除非气体 > 1000，此时释放一半采气工人）。
        2. 基地间平衡：将过饱和基地的 SCV 转移到 缺人基地。
        3. 微操优化：就近调配。
        """
        # 过滤掉侦查兵
        active_workers = self.workers.filter(lambda w: w.tag != self.active_scout_unit_tag)

        if not self.townhalls.ready or not active_workers:
            return

        # 1. 收集所有基地周围的矿点和气矿
        mineral_patches = {
            m for nexus in self.townhalls.ready
            for m in self.mineral_field.closer_than(12, nexus)
        }
        gas_refineries = {
            g for nexus in self.townhalls.ready
            for g in self.gas_buildings.ready.closer_than(12, nexus)
            if g.has_vespene
        }

        # 2. 派遣MULE（如果是人族）
        if self.config.own_race == "Terran":
            await self._deploy_mules(mineral_patches)

        # ==========================================
        # 3. [新逻辑] 气体存量检查 & 释放多余工人
        # ==========================================
        available_idle_workers = list(active_workers.idle)

        # 检查是否需要限制采气 (气体储备 > 1000)
        limit_gas_mining = self.vespene > 1000

        for gas_site in gas_refineries:
            target_gas_workers = 1 if limit_gas_mining else gas_site.ideal_harvesters

            # 获取当前在这个气矿工作的工人 (通过距离判断)
            current_gas_workers = [
                w for w in self.workers.gathering
                if w.distance_to(gas_site) < 2
            ]

            # 如果当前人数超过目标
            if len(current_gas_workers) > target_gas_workers:
                excess_count = len(current_gas_workers) - target_gas_workers

                # 释放多余工人
                for i in range(excess_count):
                    worker_to_reassign = current_gas_workers[i]
                    available_idle_workers.append(worker_to_reassign)

        # ==========================================
        # 4. [修复] 基地间平衡 (Base Balancing)
        # ==========================================
        if len(self.townhalls.ready) >= 2:
            rich_bases = [th for th in self.townhalls.ready if th.surplus_harvesters > 0]
            # 找出缺人的基地
            poor_bases_list = [th for th in self.townhalls.ready if th.surplus_harvesters < 0]

            # 【修复点】：使用局部字典记录缺口，而不是修改 unit 属性
            poor_base_deficits = {th: -th.surplus_harvesters for th in poor_bases_list}

            for rich_base in rich_bases:
                if not poor_bases_list:
                    break

                transfer_count = int(rich_base.surplus_harvesters)

                local_miners = [
                    w for w in self.workers.gathering
                    if w.distance_to(rich_base) < 10 and w not in available_idle_workers
                ]

                transferred = 0
                for w in local_miners:
                    if transferred >= transfer_count or not poor_bases_list:
                        break

                    target_base = poor_bases_list[0]

                    # 执行转移
                    target_minerals = self.mineral_field.closer_than(10, target_base)
                    if target_minerals:
                        w.gather(target_minerals.closest_to(w))

                        transferred += 1

                        # 【修复点】：更新局部字典的缺口数据
                        poor_base_deficits[target_base] -= 1

                        # 如果该基地不再缺人，从列表中移除
                        if poor_base_deficits[target_base] <= 0:
                            poor_bases_list.pop(0)

        # ==========================================
        # 5. 统计缺工任务 (Gas & Minerals)
        # ==========================================
        gas_tasks = {}
        mineral_tasks = {}

        if not limit_gas_mining:
            for g in gas_refineries:
                missing = max(0, -g.surplus_harvesters)
                if missing > 0:
                    gas_tasks[g] = missing

        for m in mineral_patches:
            worker_count = 0
            for worker in self.workers.gathering:
                if (worker.distance_to(m) < 2 and
                    worker not in available_idle_workers):
                    worker_count += 1
            need = max(0, 2 - worker_count)
            if need:
                mineral_tasks[m] = need

        # ==========================================
        # 6. 优先填补气矿
        # ==========================================
        for gas_site in list(gas_tasks.keys()):
            needed = gas_tasks[gas_site]
            if needed <= 0: continue

            # A. 从附近矿区调人
            nearby_mineral_workers = []
            for mineral in mineral_patches:
                if mineral.distance_to(gas_site) < 10:
                    for worker in self.workers.gathering:
                        if (worker.distance_to(mineral) < 2 and
                            worker not in available_idle_workers):
                            nearby_mineral_workers.append((worker, worker.distance_to(gas_site)))

            nearby_mineral_workers.sort(key=lambda x: x[1])

            reassigned = 0
            for worker, _ in nearby_mineral_workers:
                if reassigned >= needed: break
                worker.gather(gas_site)
                reassigned += 1

            needed -= reassigned

            # B. 用 idle workers 补充
            if needed > 0 and available_idle_workers:
                available_idle_workers.sort(key=lambda w: w.distance_to(gas_site))
                assigned = 0
                workers_to_remove = []
                for worker in available_idle_workers:
                    if assigned >= needed: break
                    worker.gather(gas_site)
                    workers_to_remove.append(worker)
                    assigned += 1

                for worker in workers_to_remove:
                    available_idle_workers.remove(worker)
                needed -= assigned

            gas_tasks[gas_site] = needed
            if gas_tasks[gas_site] <= 0: del gas_tasks[gas_site]

        # ==========================================
        # 7. 用剩余 idle workers 填补矿区
        # ==========================================
        for worker in available_idle_workers:
            if not mineral_tasks:
                # 如果没有缺人的矿了，就往最近的基地交货或者采最近的矿
                if self.townhalls.ready:
                    closest_base = self.townhalls.ready.closest_to(worker)
                    m_field = self.mineral_field.closer_than(10, closest_base)
                    if m_field:
                        worker.gather(m_field.closest_to(worker))
                continue

            target = min(mineral_tasks.keys(), key=lambda s: s.distance_to(worker))
            worker.gather(target)

            mineral_tasks[target] -= 1
            if mineral_tasks[target] <= 0:
                del mineral_tasks[target]

        # ==========================================
        # 8. 兜底逻辑
        # ==========================================
        if gas_tasks and not limit_gas_mining:
            for gas_site, needed in gas_tasks.items():
                if needed <= 0: continue

                # 扩大搜索范围到 15 码
                distant_mineral_workers = []
                for mineral in mineral_patches:
                    if mineral.distance_to(gas_site) < 15:
                        for worker in self.workers.gathering:
                            if (worker.distance_to(mineral) < 2 and
                                worker not in available_idle_workers):
                                distant_mineral_workers.append((worker, worker.distance_to(gas_site)))

                distant_mineral_workers.sort(key=lambda x: x[1])

                reassigned = 0
                for worker, _ in distant_mineral_workers:
                    if reassigned >= needed: break
                    worker.gather(gas_site)
                    reassigned += 1

    async def _deploy_mules(self, mineral_patches) -> None:
        """
        部署MULE到合适的矿点
        """
        mule_units = self.units(UnitTypeId.MULE).idle
        for mule in mule_units:
            nearby_minerals = [m for m in mineral_patches if m.distance_to(mule) < 12]
            best_mineral = self._select_best_mineral_for_mule(nearby_minerals, mule)
            if best_mineral:
                mule.gather(best_mineral)

    def _select_best_mineral_for_mule(self, mineral_patches, orbital_command):
        """
        为MULE选择最佳的矿点
        优先选择：
        1. 资源量较多的矿点
        2. 距离基地较近的矿点
        3. 当前采集单位较少的矿点
        4. 没有MULE的矿点
        """
        if not mineral_patches:
            return None

        best_mineral = None
        best_score = -1

        for mineral in mineral_patches:
            # 计算该矿点的评分
            score = 0

            # 资源量权重（剩余资源越多越好）
            resource_weight = mineral.mineral_contents / 1800  # 1800是矿点的初始资源量
            score += resource_weight * 40

            # 距离权重（距离越近越好）
            distance_weight = 1 - (mineral.distance_to(orbital_command) / 12)
            score += distance_weight * 20

            # 采集单位数量权重（采集单位越少越好）
            current_harvesters = 0
            for unit in self.units:
                if (hasattr(unit, 'order_target') and unit.order_target == mineral.tag and
                    unit.type_id in [UnitTypeId.SCV, UnitTypeId.MULE]):
                    current_harvesters += 1

            harvester_weight = max(0, 1 - current_harvesters / 4)  # 最多4个采集单位(2个SCV + 2个MULE)
            score += harvester_weight * 30

            # 检查是否已经有MULE在这个矿点
            mule_count = 0
            for unit in self.units.filter(lambda u: u.type_id == UnitTypeId.MULE):
                if hasattr(unit, 'order_target') and unit.order_target == mineral.tag:
                    mule_count += 1

            if mule_count >= 1:  # 每个矿点最多1个MULE
                score -= 50

            # 优先选择资源量充足的矿点
            if mineral.mineral_contents < 500:  # 资源量过低的矿点降低优先级
                score -= 20

            if score > best_score:
                best_score = score
                best_mineral = mineral
        return best_mineral if best_score > 0 else None


    async def _attempt_auto_calldown_mule(self) -> tuple[bool, str]:
        """
        [独立脚本] 自动执行砸矿骡逻辑 (直接执行版)
        只要有能量和合适的目标就直接下达指令
        """
        # 1. 获取所有已就绪的轨道司令部
        orbitals = self.structures(UnitTypeId.ORBITALCOMMAND).ready
        if not orbitals.exists:
            return False, ""

        # 2. 寻找有能量的施法者 (任意一个能量 >= 50 的即可)
        caster = None
        for oc in orbitals:
            if oc.energy >= 50:
                caster = oc
                break

        if not caster:
            return False, "" # 能量不足，静默失败

        # 3. 确定投送的目标矿区 (找采集工人数最冗余的基地，如果没有则以自己为准)
        bases = self.townhalls.ready
        if bases.exists:
            target_base = min(bases, key=lambda b: b.surplus_harvesters)
        else:
            target_base = caster

        # 4. 在目标基地附近寻找最佳矿点
        minerals_near_base = self.mineral_field.closer_than(10, target_base)
        if not minerals_near_base.exists:
             return False, ""

        target_mineral = self._select_best_mineral_for_mule(minerals_near_base, target_base)

        if not target_mineral:
            return False, ""

        # ==========================================================
        # [核心修改] 直接执行动作，不再返回 action 字典
        # ==========================================================
        caster(AbilityId.CALLDOWNMULE_CALLDOWNMULE, target=target_mineral)

        success_msg = (f"Auto-MULE: Caster [{caster.tag}] -> "
                       f"Mineral near [{target_base.position.rounded}]")

        # 返回 True 和日志消息即可，不需要返回 actions 列表了
        return True, success_msg

    # ==========================================
    # 2. 自动防御脚本
    # ==========================================
    def get_enemy_units_near_structures(self, distance: float=15.0):
        """
        筛选出距离任意我方建筑在指定距离内的敌方单位

        :param distance: 距离阈值
        :return: 符合条件的敌方单位集合
        """
        if not self.structures:
            # 如果没有建筑,返回空集合
            return self.enemy_units.subgroup([])

        if not self.enemy_units:
            # 如果没有敌方单位,返回空集合
            return self.enemy_units

        # 使用 in_distance_of_group 方法筛选
        return self.enemy_units.in_distance_of_group(self.structures, distance)

    async def automatic_defense(self, base_defense_radius: float = 15.0, response_radius: float = 30.0):
        """
        自动指派战斗单位去防御基地。
        [新] 使用 self.active_defense_map {defender_tag: enemy_tag} 来跟踪防御/威胁对。
        当一个特定的威胁消失（死亡或离开）时，对应的防御单位会停止追击。
        """

        # 1. 查找所有可用于防御的单位
        non_combat_types = {"SCV", "PROBE", "DRONE", "MULE"}
        if self.supply_army > 0:
            available_defenders = self.units.filter(
                lambda unit: unit.can_attack and unit.name.upper() not in non_combat_types
            )
        else:   # wlx修改：当没有兵力时（可能最初期或最后背水一战阶段，不能解决前方兵力在攻击，后方无兵力被袭击的情况），把工人也用上防御
            available_defenders = self.units.filter(
                lambda unit: unit.can_attack
            )

        # 2. 查找进入基地防御圈的敌人
        enemies_near_base = self.get_enemy_units_near_structures(base_defense_radius)
        # 当前在防御圈内的所有敌人 tag
        current_threat_tags = enemies_near_base.tags
        # 视野中所有存活的敌人 tag
        all_living_enemy_tags = self.enemy_units.tags

        # 3. [Check] 维护和清理 self.active_defense_map
        # 遍历当前的防御记录，判断是否终止
        new_defense_map = {} # 用一个新的 map 来存储仍然有效的防御记录
        stopped_units = 0

        # 遍历当前记录在案的"我方防御单位"和"其锁定的敌方单位"
        for defender_tag, enemy_tag in self.active_defense_map.items():
            defender = self.units.find_by_tag(defender_tag)

            # 检查我方防御单位是否存活
            if not defender:
                continue # 防御单位死亡，自动从 new_map 中移除，无需操作

            # 检查敌方单位是否存活
            enemy_is_alive = enemy_tag in all_living_enemy_tags
            if not enemy_is_alive:
                # [判断是否终止] 对应的敌方单位消失（死亡）
                continue # 敌方单位死亡，防御任务结束，自动移除

            # 检查敌方单位是否仍在威胁区
            enemy_is_still_a_threat = enemy_tag in current_threat_tags

            if enemy_is_still_a_threat:
                # 敌人存活且仍在威胁区，保持防御记录
                new_defense_map[defender_tag] = enemy_tag
            else:
                # [判断是否终止] 敌人存活，但已离开威胁区
                # 停止对应的防御单位
                if defender.orders and defender.orders[0].ability.id in {AbilityId.ATTACK, AbilityId.ATTACK_ATTACK}:
                    defender.stop()
                    stopped_units += 1
                # 不将此记录添加到 new_map 中，防御结束

        self.active_defense_map = new_defense_map # [删除防御记录] 更新 map，移除了所有已完成的防御

        if stopped_units > 0:
            # print(f"Automatic Defense: Stopped {stopped_units} units from pursuit as their targets left.")
            self.logging("Automatic Defense", f"Stopped {stopped_units} units from pursuit as their targets left.")
            await self.chat_send(f"Automatic Defense: {stopped_units} units disengaging.")

        # 4. [指派新的防御]
        if not enemies_near_base.exists:
            return # 基地安全，也无需指派新单位

        threat_center = enemies_near_base.center
        # 找出在响应范围内的所有可用单位
        responding_units = available_defenders.closer_than(response_radius, threat_center)

        if not responding_units.exists:
            return # 没有单位能响应

        # 找出尚未分配任务的单位
        unassigned_defenders = responding_units.filter(lambda u: u.tag not in self.active_defense_map and not u.is_attacking)
        # wlx新修改：发现有些单位第一次触发防御后，之后跟敌人交火时，敌人全部在防区外，其实没触发防御或是留在防御列表里，全靠"本能"跟敌人交火，所以会被新的防御触发拉走，现在加上一个假设，即附近响应单位如果在攻击就是在跟敌人交火，就不选择他们，选择没在攻击的兵力。

        new_assignments = 0
        for unit in unassigned_defenders:
            # 命令每个单位攻击距离它自己最近的那个威胁
            target = enemies_near_base.closest_to(unit)
            if target:
                unit.attack(target)
                # [记录防御行为]
                # 将"我方单位"和"它去攻击的敌方单位"绑定
                self.active_defense_map[unit.tag] = target.tag
                new_assignments += 1

        if new_assignments > 0:
            # print(f"Automatic Defense: Assigned {new_assignments} new units to defend.")
            self.logging("Automatic Defense", f"Assigned {new_assignments} new units to defend.")
            await self.chat_send(f"Automatic Defense: Engaging base threat!")

    # ==========================================
    # 3. 集结与驻防脚本
    # ==========================================
    async def set_terran_combat_rally_points(self):
            """
            为每个人族攻击单位生产建筑（兵营、工厂、星港）设置集结点。

            集结点基于最近的指挥中心（或其升级形态），
            设置在该指挥中心与地图中心连线上，距离指挥中心15个单位的位置。
            [V2: 使用 self.structure_rally_points 缓存避免重复设置]
            """

            # 1. 定义人族的相关建筑

            # (A) 生产攻击单位的建筑
            combat_structures = {
                UnitTypeId.BARRACKS,
                UnitTypeId.FACTORY,
                UnitTypeId.STARPORT,
            }

            # (B) 用来计算集结点的"中心"建筑 (指挥中心及其升级)
            rally_centers = {
                UnitTypeId.COMMANDCENTER,
                UnitTypeId.ORBITALCOMMAND,
                UnitTypeId.PLANETARYFORTRESS
            }

            # 2. 找到所有已建成的"集结中心"建筑 (B)
            all_ready_rally_centers = self.structures(rally_centers).ready

            if not all_ready_rally_centers:
                # 如果一个"中心"都没有，则无法设置
                return

            # 3. 获取地图中心
            map_center = self.game_info.map_center

            # 4. 预先计算所有"集结中心"的集结点
            rally_points_by_center_tag = {}
            for center in all_ready_rally_centers:
                rally_point = center.position.towards(map_center, 15)
                rally_points_by_center_tag[center.tag] = rally_point

            # 5. [新增] 获取当前所有相关生产建筑
            ready_combat_structures = self.structures(combat_structures).ready
            current_combat_tags = {s.tag for s in ready_combat_structures}

            # 6. [新增] 清理缓存中已不存在(被摧毁)的建筑
            # (使用 list() 来允许在迭代时修改字典)
            tags_to_remove = [tag for tag in self.structure_rally_points if tag not in current_combat_tags]
            for tag in tags_to_remove:
                del self.structure_rally_points[tag]
                print(f"Rally cache: Removed destroyed structure [Tag:{tag}]")

            # 7. 遍历所有已建成的生产建筑 (A)，并设置集结点
            for structure in ready_combat_structures:

                # 找到最近的"集结中心" (B)
                closest_center = all_ready_rally_centers.closest_to(structure.position)

                # 获取这个"中心"对应的集结点
                target_rally_point = rally_points_by_center_tag[closest_center.tag]

                # 8. [修改] 检查缓存的集结点是否与目标集结点不同
                cached_point = self.structure_rally_points.get(structure.tag)

                if cached_point != target_rally_point:

                    # *** (1) 执行设置集结点的动作 ***
                    structure(AbilityId.RALLY_UNITS, target_rally_point)

                    # *** (2) [新增] 更新缓存 ***
                    self.structure_rally_points[structure.tag] = target_rally_point

                    # *** (3) 按照您的要求，发送通知 ***
                    message = (
                        f"已更新集结点: {structure.type_id.name} [Tag:{structure.tag}] "
                        f"-> {target_rally_point.rounded} "
                        f"(基于 {closest_center.type_id.name} [Tag:{closest_center.tag}])"
                    )
                    print(message)
                    await self.chat_send(message)

    def _clamp_position_to_map_bounds(self, p: Point2) -> Point2:
            """
            确保坐标点在地图边界内，且 X 和 Y 至少为 1。
            """
            map_width = self.game_info.map_size.width
            map_height = self.game_info.map_size.height

            # 用户要求：如果计算出 0 或者 负数位置 取 1
            x = max(1, p.x)
            y = max(1, p.y)

            # 额外保护：确保不会超出地图边界 (减1是因为坐标从0开始)
            x = min(x, map_width - 1)
            y = min(y, map_height - 1)

            return Point2((x, y))

    async def manage_garrison(self):
            """
            管理基地的自动驻防。
            为每个基地计算4个驻防点，并按比例分配防区内的空闲战斗单位。
            [修改] 只有在基地 10.0 半径内没有敌情时才会执行。
            """

            # 1. [修改] 检查是否有敌情 (使用 automatic_defense 的 10.0 半径)
            # 任何建筑附近 10.0 范围内有敌人，则跳过驻防调整
            enemies_near_base = self.get_enemy_units_near_structures(10.0)
            if enemies_near_base.exists:
                # print("敌情威胁中，暂停驻防调整。")
                return # 存在敌情，不调整驻防

            # 2. 获取所有空闲战斗单位
            non_combat_types = {UnitTypeId.SCV, UnitTypeId.PROBE, UnitTypeId.DRONE, UnitTypeId.MULE}
            # 集结点等待中的单位仍属于进攻编队，不由驻防重新分配。
            attack_unit_tags = {
                tag for group in self.total_attack_groups.values() for tag in group["unit_tags"]
            }
            idle_combat_units = self.units.filter(
                lambda u: u.can_attack
                and u.type_id not in non_combat_types
                and u.is_idle
                and u.tag not in attack_unit_tags
            )
            if not idle_combat_units.exists:
                # print("没有空闲的战斗单位，跳过驻防。")
                return # 没有单位可分配

            # 3. 遍历每个基地 (center) 来管理其驻防
            rally_centers_types = {
                UnitTypeId.COMMANDCENTER,
                UnitTypeId.ORBITALCOMMAND,
                UnitTypeId.PLANETARYFORTRESS
            }
            all_ready_rally_centers = self.structures(rally_centers_types).ready
            if not all_ready_rally_centers.exists:
                return # 没有基地

            map_center = self.game_info.map_center
            total_units_moved = 0

            for center in all_ready_rally_centers:

                # --- A. 计算 4 个驻防点 ---

                # 逻辑同 set_terran_combat_rally_points
                main_rally_point = center.position.towards(map_center, 15)

                # Point 1 (50%): The main rally point
                p1 = main_rally_point

                # Point 2 (10%): 在 指挥中心(center) 和 集结点(p1) 连线上，靠近指挥中心
                p2 = main_rally_point.towards(center.position, self.GARRISON_EXTENSION_DISTANCE)

                # [修改] Points 3 & 4 (20% each): 在 指挥中心(center) 和 P1(main_rally_point) 连线的中点 做垂线
                vector = main_rally_point.position - center.position

                # [修复] 替换 .lerp()，手动计算中点 (P1 + P2) / 2
                mid_point = (center.position + main_rally_point.position) / 2 # 计算中点

                if not vector.length: # 避免除零
                    perp_vector = Point2((self.GARRISON_PERPENDICULAR_DISTANCE, 0))
                else:
                    # [修复] 替换 .rotated(90)，手动计算垂直向量
                    # 2D 向量 (x, y) 旋转 90 度为 (-y, x)
                    norm_vec = vector.normalized
                    rotated_vec = Point2((-norm_vec.y, norm_vec.x))
                    perp_vector = rotated_vec * self.GARRISON_PERPENDICULAR_DISTANCE

                p3 = mid_point + perp_vector
                p4 = mid_point - perp_vector

                # [新增] 确保所有驻防点坐标合法（大于0且在地图内）
                p1 = self._clamp_position_to_map_bounds(p1)
                p2 = self._clamp_position_to_map_bounds(p2)
                p3 = self._clamp_position_to_map_bounds(p3)
                p4 = self._clamp_position_to_map_bounds(p4)

                # [修改] 按优先级排序：P1, P2, P3, P4
                garrison_quotas_points = [
                    (0.50, p1), # P1 (50%)
                    (0.10, p2), # P2 (10%)
                    (0.20, p3), # P3 (20%)
                    (0.20, p4)  # P4 (20%)
                ]

                # --- B. 分配单位 ---

                # 1. 获取防区内的所有空闲单位 (只操作本防区的兵力)
                # [注意] GARRISON_DEFENSE_ZONE_RADIUS (30) 仍然用于定义 *我方* 防区范围
                units_in_zone = idle_combat_units.closer_than(self.GARRISON_DEFENSE_ZONE_RADIUS, center.position)
                total_units_in_zone = units_in_zone.amount

                if total_units_in_zone == 0:
                    continue # 这个基地防区没兵，跳过

                available_units = list(units_in_zone) # 可供分配的单位
                units_moved_this_base = 0

                # 2. 按优先级遍历驻防点
                for quota, point in garrison_quotas_points:
                    needed_count = math.ceil(total_units_in_zone * quota)

                    # 检查有多少单位 *已经* 在这个点
                    units_at_point = []
                    remaining_available = []

                    for u in available_units:
                        if u.distance_to(point) <= self.GARRISON_CHECK_RADIUS:
                            units_at_point.append(u)
                        else:
                            remaining_available.append(u)

                    num_at_point = len(units_at_point)
                    num_to_move = max(0, needed_count - num_at_point)

                    # a. 将已在位置的单位(最多needed_count个)标记为"已分配"
                    # (我们只关心更新 available_units 列表)
                    # final_assigned_at_point = units_at_point[:needed_count]

                    # b. 将多余的单位 和 不在位置的单位 放回"可用"池
                    excess_units = units_at_point[needed_count:]
                    available_units = remaining_available + excess_units

                    # c. 如果还需要单位，从"可用"池中调拨 (如果兵力不够, available_units会变空, 自动满足 P1 > P2 > P3 > P4)
                    if num_to_move > 0 and available_units:
                        # 排序，找到最近的
                        available_units.sort(key=lambda u: u.distance_to(point))

                        units_to_move = available_units[:num_to_move]
                        available_units = available_units[num_to_move:] # 更新"可用"池

                        for u in units_to_move:
                            u.move(point)
                            units_moved_this_base += 1

                # 3. 将所有剩余未分配的单位派往主集结点 (P1)
                for u in available_units:
                    if u.distance_to(p1) > self.GARRISON_CHECK_RADIUS:
                        u.move(p1)
                        units_moved_this_base += 1

                if units_moved_this_base > 0:
                    print(f"Garrisoning Base {center.tag}: Reassigned {units_moved_this_base} units.")
                    total_units_moved += units_moved_this_base

    # ==========================================
    # 4. 侦察脚本
    # ==========================================
    def _update_scouting_information(self):
        """
        [V3] 主动检查视野, 记录新发现的单位。
        (根据用户需求, 移除了 "丢失视野" 的打印)
        此函数不重写 BotAI 事件, 而是由 manage_scouting 调用。
        """

        current_visible_tags = {u.tag for u in self.enemy_units}

        # --- 1. 处理新进入视野的单位 ---
        new_tags = current_visible_tags - self.known_enemy_tags_in_vision

        if new_tags:
            current_time_int = int(self.time)
            if current_time_int not in self.scouting_information:
                self.scouting_information[current_time_int] = []

            for tag in new_tags:
                unit = self.enemy_units.find_by_tag(tag)
                if unit:
                    info = f"发现 {unit.type_id.name} (Tag: {unit.tag}) 位于 {unit.position.rounded}"
                    if unit.is_structure:
                        info += " (建筑)"

                    # [V3] 侦察报告只打印新发现的单位
                    print(f"[侦察信息] {info}")

                    # 添加到我们的信息字典中
                    self.scouting_information[current_time_int].append(info)

                    # 额外记录到日志
                    self.logging("scout_discovery", info)

        self.known_enemy_tags_in_vision = current_visible_tags


    async def manage_scouting(self):
        """
        管理侦察单位和信息收集。
        在每个 on_step 周期被 run 方法调用。
        """

        # [V3] 首先, 更新我们的视野信息
        self._update_scouting_information()

        # 0. 如果当前被选择的侦察单位处于任意攻击队列里，取消其侦察任务
        if any(self.active_scout_unit_tag in group["unit_tags"] for group in self.total_attack_groups.values()):
            self.active_scout_unit_tag = None

        # 1. 检查当前侦察单位的状态
        if self.active_scout_unit_tag:
            scout_unit = self.units.find_by_tag(self.active_scout_unit_tag)

            # 侦察单位死亡或消失
            if not scout_unit:
                print(f"侦察单位 (Tag: {self.active_scout_unit_tag}) 丢失 (判定为死亡).")
                self.active_scout_unit_tag = None
                self.scout_target_location = None

            # [V3 修改] 侦察单位变为空闲 (已到达目的地), 立即派遣到下一个最近的地点
            elif scout_unit.is_idle:
                print(f"侦察单位 {scout_unit.type_id.name} (Tag: {self.active_scout_unit_tag}) 已变为空闲, 寻找下一个目标...")

                # [V3 新] 寻找离侦察兵 *当前位置* 最近的下一个目标
                next_target = self._get_scout_target(from_position=scout_unit.position)

                if next_target:
                    # [V3 新] 派遣到新目标
                    self._send_scout(scout_unit, next_target)
                else:
                    # [V3 新] 找不到目标, 释放侦察兵
                    print(f"侦察单位 {scout_unit.type_id.name} (Tag: {self.active_scout_unit_tag}) 已完成所有任务, 释放。")
                    self.active_scout_unit_tag = None
                    self.scout_target_location = None

            # 否则, 单位还在路上, 什么都不做
            else:
                pass

        # 2. 如果当前没有侦察单位, 尝试派遣一个新的
        if not self.active_scout_unit_tag:
            await self._find_and_dispatch_scout()

    async def _find_and_dispatch_scout(self):
        """
        按照 飞机 > Marine > SCV 的优先级, 寻找一个空闲单位并派遣它。
        """

        # [V3 修改] 默认从基地出发寻找第一个目标
        target = self._get_scout_target(from_position=self.start_location)
        if not target:
            # print("没有需要侦察的目标。")
            return # 没有可侦察的目标

        scout_unit = None

        # 优先级 3: 飞机 (任何空闲的飞行单位)
        air_types = {UnitTypeId.VIKINGFIGHTER, UnitTypeId.BANSHEE, UnitTypeId.MEDIVAC, UnitTypeId.LIBERATOR, UnitTypeId.RAVEN}
        idle_aircraft = self.units.filter(lambda u: u.type_id in air_types and not any(u.tag in group["unit_tags"] for group in self.total_attack_groups.values()) and u.is_idle)
        if idle_aircraft.exists:
            scout_unit = idle_aircraft.random
            print(f"派遣 [飞机]{scout_unit.name} 执行侦察任务。")
            await self.chat_send(f"派遣 [飞机]{scout_unit.name} 执行侦察任务。")

        # 优先级 2: Marine/Reaper (如果没有可用的飞机)
        army_types = {UnitTypeId.MARINE, UnitTypeId.REAPER}
        if not scout_unit:
            idle_army = self.units.filter(lambda u: u.type_id in army_types and not any(u.tag in group["unit_tags"] for group in self.total_attack_groups.values()) and u.is_idle)
            if idle_army.exists:
                scout_unit = idle_army.random
                print(f"派遣 [{scout_unit.name}] 执行侦察任务。")
                await self.chat_send(f"派遣 [{scout_unit.name}] 执行侦察任务。")

        # 优先级 1: SCV (仅限开局一次)
        if not scout_unit and not self.scv_scout_sent and self.time > 15:
            scv_pool = self.workers.idle  # <-- 1. 优先尝试找空闲的
            if not scv_pool.exists:
                # <-- 2. 如果找不到空闲的, 就从矿工里找
                scv_pool = self.workers.filter(
                    lambda w: w.is_gathering
                    and w.order_target in self.mineral_field.tags
                )
            if scv_pool.exists: # <-- 3. 只要池子里有单位 (无论是空闲的还是采矿的)
                scout_unit = scv_pool.random
                self.scv_scout_sent = True
                print("派遣 [SCV] (从工人中抽取)...")
                await self.chat_send("派遣 [SCV] (从工人中抽取)执行侦察任务。")

        # [V3 修正]
        # 统一下发派遣命令 (原代码中此缩进错误, 导致飞机和Marine无法被派遣)
        if scout_unit:
            self._send_scout(scout_unit, target)

    def _get_scout_target(self, from_position=None):
        """
        决定下一个侦察目标点。
        [V3 修改] 增加 from_position 参数, 用于计算最近距离。

        优先级:
        1. 未侦察过的敌方出生点。
        2. 未侦察过的矿区 (优先去近的)。
        3. 已知的敌方基地 (如果所有点都侦察过了)。
        """

        # [V3 新] 确定距离计算的基准点
        base_position = from_position if from_position else self.start_location

        # 1. 敌方出生点
        if self.enemy_start_locations:
            enemy_start_loc = self.enemy_start_locations[0]
            if enemy_start_loc not in self.scouted_locations:
                return enemy_start_loc

        # 2. 未侦察过的矿区
        unscouted_expansions = []
        for exp_loc in self.expansion_locations_list:
            if exp_loc not in self.scouted_locations:
                unscouted_expansions.append(exp_loc)

        if unscouted_expansions:
            # [V3 修改] 返回距离 (base_position) 最近的那个未侦察矿区
            return min(unscouted_expansions, key=lambda loc: loc.distance_to(base_position))

        # 3. 如果所有点都侦察过了, 重置侦察列表, 重新侦察敌方基地
        if self.enemy_start_locations:
            print("所有侦察点已完成, 循环重置。")
            self.scouted_locations.clear()
            return self.enemy_start_locations[0]

        # 备用方案 (几乎不会触发)
        return self.game_info.map_center

    def _send_scout(self, unit, target):
        """
        发送侦察单位并更新追踪变量的辅助函数。
        """
        unit.move(target)
        self.active_scout_unit_tag = unit.tag
        self.scout_target_location = target

        # 当我们派遣单位时就标记该地点, 防止重复派遣
        self.scouted_locations.add(target)

        print(f"正在派遣 {unit.type_id.name} (Tag: {unit.tag}) 侦察 {target.rounded}")
        self.logging("scouting", f"Dispatching {unit.type_id.name} to {target.rounded}")

    # ==========================================
    # 5. 战斗辅助与总攻管理脚本
    # ==========================================
    def _get_all_combat_units(self):
        """
        辅助函数: 获取所有非农民、非MULE的战斗单位
        """
        non_combat_types = {"SCV", "PROBE", "DRONE", "MULE"}
        return self.units.filter(
            lambda unit: unit.can_attack and unit.name.upper() not in non_combat_types
        )


    async def rally_units_to_point(self, units_to_rally, target_point):
        """
        将一个 Units 集合中的所有单位集结 (A-Move) 到一个目标点。

        :param units_to_rally: 一个 sc2.units.Units 集合 (由调用方确保)
        :param target_point: sc2.position.Point2 目标点
        """

        # 既然 units_to_rally 总是 Units 集合，
        # 我们可以直接检查 .exists 属性 (这是 Units 集合特有的)
        # 以确保集合非空，防止发送无效指令。
        if units_to_rally.exists:

            # 使用 A-Move (攻击性移动)
            # 【修复】 必须遍历集合中的每一个单位
            for unit in units_to_rally:
                unit.attack(target_point)

            # --- 备选方案 ---
            # 如果你只想让它们"移动" (M-Move)，忽略敌人，
            # 使用下面这行代替:
            # units_to_rally.move(target_point)


    # 原版: def launch_total_attack(self, target_enemy_unit, units_to_launch):
    def _attack_wave_should_target_units_first(self, live_units_in_group: Units) -> bool:
        if (
            self._newexec_micro_active_for_current_army()
            and self.micro_require_tech_unit
            and self._newexec_attack_is_marine_only_group(live_units_in_group)
        ):
            return False
        if self.attack_target_units_first:
            return True
        if not self.attack_target_units_first_auto or not self.enemy_units.exists:
            return False

        type_counts = Counter(unit.type_id for unit in live_units_in_group)
        has_force_anchor = any(
            unit_type in self.MICRO_TARGET_UNITS_FIRST_FORCE_TYPES for unit_type in type_counts
        )
        if not has_force_anchor:
            mixed_count = sum(
                count
                for unit_type, count in type_counts.items()
                if unit_type not in self.MICRO_STRUCTURE_FIRST_ONLY_TYPES
            )
            total_count = sum(type_counts.values())
            if total_count <= 0 or mixed_count / total_count < self.attack_target_units_first_mixed_ratio:
                return False

        if not has_force_anchor and mixed_count <= 0:
            return False

        squad_center = live_units_in_group.center
        closest_enemy_unit = self.enemy_units.closest_to(squad_center)
        if closest_enemy_unit.distance_to(squad_center) > 22:
            return False

        if self.enemy_structures.exists:
            closest_structure = self.enemy_structures.closest_to(squad_center)
            if closest_enemy_unit.distance_to(squad_center) > closest_structure.distance_to(squad_center) + 10:
                return False

        return True

    def _newexec_units_first_initial_target(self, units_to_commit):
        if (
            not units_to_commit.exists
            or not self.enemy_units.exists
            or not self._attack_wave_should_target_units_first(units_to_commit)
        ):
            return None
        return self._select_attack_compatible_enemy_unit(units_to_commit)

    def _select_attack_compatible_enemy_unit(self, live_units_in_group: Units):
        if not live_units_in_group.exists or not self.enemy_units.exists:
            return None

        squad_center = live_units_in_group.center
        attackers = [unit for unit in live_units_in_group if getattr(unit, "can_attack", False)]
        worker_types = {
            UnitTypeId.SCV,
            UnitTypeId.PROBE,
            UnitTypeId.DRONE,
            UnitTypeId.MULE,
        }
        if not attackers:
            return self.enemy_units.closest_to(squad_center)

        scored_targets = []
        for enemy in self.enemy_units:
            compatible_count = sum(
                1 for unit in attackers if self._unit_can_attack_seen_enemy(unit, enemy)
            )
            scored_targets.append(
                (compatible_count, -enemy.distance_to(squad_center), enemy)
            )

        best_compatible_count = max(score[0] for score in scored_targets)
        if best_compatible_count <= 0:
            return self.enemy_units.closest_to(squad_center)

        compatible_targets = [
            score for score in scored_targets if score[0] > 0
        ]
        combat_targets = [
            score
            for score in compatible_targets
            if score[2].type_id not in worker_types
        ]
        nearest_compatible_distance = min(-score[1] for score in compatible_targets)
        local_combat_targets = [
            score
            for score in combat_targets
            if -score[1] <= nearest_compatible_distance + 10
        ]
        target_pool = local_combat_targets or compatible_targets

        return max(
            target_pool,
            key=lambda score: (
                score[0],
                score[1],
            ),
        )[2]

    async def launch_total_attack(
        self,
        target,
        units_to_launch,
        min_units_to_launch: Optional[int] = None,
        respect_min_units: bool = False,
    ):
        """
        ...
        :param target: sc2.unit.Unit 或 sc2.position.Point2 - 最终进攻的目标
        ...
        """
        # self.logger.info(f"launch_total_attack: 收到总攻发起指令！...")
        self.logging("launch_total_attack", "收到总攻发起指令！...")

        if not units_to_launch.exists:
            return

        # rally_point = self.game_info.map_center
        # self.rally_units_to_point(units_to_launch, rally_point)

        unit_tags_set = {unit.tag for unit in units_to_launch}
        base_min_units_to_launch = max(
            1,
            int(min_units_to_launch or min(self.reinforcement_min_units, units_to_launch.amount)),
        )
        min_units_to_launch = self._newexec_min_units_for_selected_attack(
            units_to_launch, base_min_units_to_launch
        ) if not respect_min_units else base_min_units_to_launch

        # --- [移动至此] 解析目标类型：区分 Unit 和 Point2 ---
        target_tag = None
        final_position = None
        if isinstance(target, Point2):
            final_position = target
        elif isinstance(target, Unit): # 假设 target 是 Unit 对象
            target_tag = target.tag
            final_position = target.position
        else:
            # 备用，防止传入错误类型
            # self.logger.error(f"launch_total_attack 收到未知的目标类型: {type(target)}")
            self.logging("launch_total_attack", f"收到未知的目标类型: {type(target)}", "error")
            final_position = self.enemy_start_locations[0]
        # --- 结束 ---

        if self.enable_newexec_micro and not self._newexec_micro_active_for_current_army():
            self.logging(
                "MULEMICRO_MARINE_ONLY_FALLBACK",
                (
                    "strict full-army Marine-only composition uses exact old attack path; "
                    f"units={len(units_to_launch)}, "
                    f"target={final_position.rounded}"
                ),
            )

        merge_wave_id = self._newexec_find_mergeable_attack_wave(
            units_to_launch, final_position, target_tag
        )
        if merge_wave_id is not None:
            attack_data = self.total_attack_groups[merge_wave_id]
            old_count = len(set(attack_data.get("unit_tags", set())) & {unit.tag for unit in self.units})
            attack_data["unit_tags"].update(unit_tags_set)
            attack_data["min_units_to_launch"] = max(
                int(attack_data.get("min_units_to_launch", self.reinforcement_min_units)),
                min_units_to_launch,
            )
            if target_tag:
                attack_data["target_tag"] = target_tag
            if final_position:
                attack_data["final_position"] = final_position

            state = attack_data.get("state")
            if state == "GATHERING":
                await self.rally_units_to_point(units_to_launch, attack_data["rally_point"])
            else:
                attack_target = attack_data.get("final_position") or final_position
                if attack_target:
                    for unit in units_to_launch:
                        unit.attack(attack_target)

            self.logging(
                "MULEMICRO_ATTACK_MERGE",
                {
                    "wave_id": merge_wave_id,
                    "state": state,
                    "incoming": self._unit_type_counts_from_units(units_to_launch),
                    "old_count": old_count,
                    "new_count": old_count + units_to_launch.amount,
                    "min_units_to_launch": attack_data["min_units_to_launch"],
                },
            )
            return

        if units_to_launch.amount < min_units_to_launch:
            self.logging(
                "launch_total_attack",
                f"跳过低于最低出击规模的攻击波: {units_to_launch.amount}/{min_units_to_launch}",
                "warning",
            )
            return

        self.total_attack_wave_id_counter += 1
        wave_id = self.total_attack_wave_id_counter

        # --- [修改] 计算集结点 ---
        # 原逻辑: rally_point = self.game_info.map_center
        # 新逻辑: 取小队中心点与target.position(即 final_position)的中点
        rally_point = (units_to_launch.center + final_position) / 2
        await self.rally_units_to_point(units_to_launch, rally_point)
        await self.chat_send(f"Gathering together! Wave: {wave_id}, Location: {rally_point.rounded}! Let's do this!")

        self.total_attack_groups[wave_id] = {
            "unit_tags": unit_tags_set,
            "target_tag": target_tag,         # 最终目标 (可能为 None)
            "final_position": final_position, # (新) 最终目标坐标
            "rally_point": rally_point,
            "state": "GATHERING",
            "min_units_to_launch": min_units_to_launch,
            "last_macro_order_loop": 0,
            "last_fallback_order_loop": 0,
            "last_ordered_target_tag": target_tag,
        }

        # self.logger.info(f"launch_total_attack: 已创建总攻波次 {wave_id}，状态: GATHERING，集结点: {rally_point.rounded}。")
        self.logging("launch_total_attack", f"已创建总攻波次 {wave_id}，状态: GATHERING，集结点: {rally_point.rounded}，最低出击: {min_units_to_launch}。")

    async def manage_total_attack_groups(self):
        """
        【维护总攻编队】(在 on_step 中每帧调用)

        (V4: 严格按优先级索敌)
        1. (剔除死亡)
        2. (删除全灭记录)
        3. (状态机)
        - if GATHERING: ...
        - if ATTACKING:
            - 检查 (目标失效 OR 有单位闲置)
            - (是) 整个小队自动索敌 (优先建筑 > 单位)，并 A-Move 攻击目标位置

        [修改] ATTACKING 状态下重新索敌时，跳过正在微操中的单位，
        避免覆盖 micro_fight 的精细走位指令。
        """
        if not self.total_attack_groups:
            return

        current_alive_unit_tags = {unit.tag for unit in self.units}
        waves_to_delete = []

        for wave_id, attack_data in list(self.total_attack_groups.items()):

            group_unit_tags = attack_data["unit_tags"]

            # --- 1. 剔除死亡单位 ---
            dead_tags = group_unit_tags - current_alive_unit_tags
            if dead_tags:
                group_unit_tags.difference_update(dead_tags)

            # --- 2. 检查编队是否全灭 ---
            if not group_unit_tags:
                waves_to_delete.append(wave_id)
                # self.logger.info(f"manage_total_attack_groups: 总攻波次 {wave_id} 已全灭。")
                self.logging("manage_total_attack_groups", f"总攻波次 {wave_id} 已全灭。")
                continue

            # --- 3. 状态机逻辑 ---

            # 获取这个波次所有"存活"的单位对象
            live_units_in_group = self.units.filter(lambda u: u.tag in group_unit_tags)
            if not live_units_in_group.exists:
                continue # (安全检查)

            current_state = attack_data["state"]

            # Marine-only waves deliberately bypass gathering, target
            # switching, grouping, and Mulobot micro. Keep their original
            # old-like A-move alive without reissuing it every frame.
            if current_state == "DIRECT":
                current_iteration = getattr(self, "current_micro_iteration", 0)
                if current_iteration <= 0:
                    current_iteration = getattr(self.state, "game_loop", 0)
                last_order_loop = int(attack_data.get("last_macro_order_loop", 0))
                idle_units = live_units_in_group.idle
                if (
                    idle_units.exists
                    and current_iteration - last_order_loop
                    >= self.attack_macro_retarget_interval
                ):
                    direct_target = attack_data.get("final_position")
                    if direct_target is None:
                        direct_target = self.enemy_start_locations[0]
                    for unit in idle_units:
                        unit.attack(direct_target)
                    attack_data["last_macro_order_loop"] = current_iteration
                continue

            # --- 状态: GATHERING (集结中) ---
            if current_state == "GATHERING":
                rally_point = attack_data["rally_point"]
                min_units_to_launch = max(1, int(attack_data.get("min_units_to_launch", self.reinforcement_min_units)))

                units_not_at_rally = live_units_in_group.further_than(5.0, rally_point)

                if not units_not_at_rally.exists:
                    if live_units_in_group.amount < min_units_to_launch:
                        waves_to_delete.append(wave_id)
                        self.logging(
                            "manage_total_attack_groups",
                            f"总攻波次 {wave_id}: 集结后仅剩 {live_units_in_group.amount} 单位，低于最低出击 {min_units_to_launch}，取消本波等待补兵。",
                            "warning",
                        )
                        continue

                    # --- (触发) 所有单位都已抵达集结点 ---
                    # self.logger.info(f"总攻波次 {wave_id}: 集结完毕，发动进攻！")
                    self.logging("manage_total_attack_groups", f"总攻波次 {wave_id}: 集结完毕，发动进攻！")
                    attack_data["state"] = "ATTACKING"

                    # --- 【V3 修复：Plan B 逻辑】 ---
                    final_target_pos = attack_data.get("final_position")
                    if not final_target_pos:
                        target_tag = attack_data.get("target_tag")
                        final_target = None
                        if target_tag:
                            if self.enemy_units.exists:
                                final_target = self.enemy_units.find_by_tag(target_tag)
                            if not final_target and self.enemy_structures.exists:
                                final_target = self.enemy_structures.find_by_tag(target_tag)

                        if final_target:
                            final_target_pos = final_target.position
                        else:
                            # self.logger.info(f"总攻波次 {wave_id}: 目标丢失, A-Move至敌方出生点。")
                            self.logging("manage_total_attack_groups", f"总攻波次 {wave_id}: 目标丢失, A-Move至敌方出生点。")
                            final_target_pos = self.enemy_start_locations[0]

                    # 统一 A-Move 到最终坐标
                    for unit in live_units_in_group:
                        unit.attack(final_target_pos)
                    current_iteration = getattr(self, "current_micro_iteration", 0)
                    if current_iteration <= 0:
                        current_iteration = getattr(self.state, "game_loop", 0)
                    attack_data["last_macro_order_loop"] = current_iteration
                    attack_data["last_ordered_target_tag"] = attack_data.get("target_tag")
                    self.logging("ATTACK", f"Launch attack to {final_target_pos} with {live_units_in_group.amount} units.")
                    await self.chat_send(f"Launch attack! Wave: {wave_id}, Target: {final_target_pos.rounded}! Move!Move!Move!")
                    # --- 【V3 修复结束】 ---

                else:
                    # --- (维持) 仍在集结中 ---
                    idle_and_lost = live_units_in_group.idle.further_than(5.0, rally_point)
                    if idle_and_lost.exists:
                        await self.rally_units_to_point(idle_and_lost, rally_point)

            # --- 状态: ATTACKING (进攻中) ---
            elif current_state == "ATTACKING":

                # --- 【!! 自动切换目标 V4: 闲置/目标丢失 自动索敌 !!】 ---

                target_tag = attack_data.get("target_tag")
                target_alive = False

                # 1. 检查原目标是否存活
                if target_tag:
                    # 检查目标是否在可见的敌方单位或建筑中
                    if self.enemy_units.find_by_tag(target_tag) or self.enemy_structures.find_by_tag(target_tag):
                        target_alive = True

                # 2. 检查小队是否有闲置单位(任何单位闲置都触发重新索敌，保持进攻性)
                has_idle_units = live_units_in_group.idle.exists

                current_iteration = getattr(self, "current_micro_iteration", 0)
                if current_iteration <= 0:
                    current_iteration = getattr(self.state, "game_loop", 0)

                # 如果目标丢失 或者 有单位闲置，则寻找最近目标
                if not target_alive or has_idle_units:
                    squad_center = live_units_in_group.center
                    new_target = None

                    # Micro-friendly targeting: fight the army first when
                    # visible, then clean up structures.
                    if (
                        self.enemy_units.exists
                        and self._attack_wave_should_target_units_first(live_units_in_group)
                    ):
                        new_target = self._select_attack_compatible_enemy_unit(
                            live_units_in_group
                        )
                    elif self.enemy_structures.exists:
                        new_target = self.enemy_structures.closest_to(squad_center)
                    elif self.enemy_units.exists:
                        new_target = self.enemy_units.closest_to(squad_center)

                    if new_target:
                        # 仅当目标发生变化时打印日志(避免闲置刷新时刷屏)
                        target_changed = target_tag != new_target.tag
                        last_order_loop = int(attack_data.get("last_macro_order_loop", 0))
                        should_refresh_order = (
                            target_changed
                            or current_iteration - last_order_loop >= self.attack_macro_retarget_interval
                            or (
                                has_idle_units
                                and current_iteration - last_order_loop
                                >= max(self.MICRO_LISTEN_INTERVAL, self.attack_macro_retarget_interval // 2)
                            )
                        )

                        if target_changed:
                            # self.logger.info(f"总攻波次 {wave_id}: 锁定新目标 {new_target.name} (Tag: {new_target.tag})，攻击位置: {new_target.position.rounded}。")
                            self.logging("manage_total_attack_groups", f"总攻波次 {wave_id}: 锁定新目标 {new_target.name} (Tag: {new_target.tag})，攻击位置: {new_target.position.rounded}。")

                        # 【核心修改】: 统一A-Move到新目标的【位置】，而非攻击单位Tag
                        # 这样即使目标移动或进入战争迷雾，单位也会攻击沿途敌人
                        target_pos = new_target.position

                        # =====================================================
                        # [修改] 跳过正在微操中的单位，避免覆盖micro_fight指令
                        # 参考方案: 只对不在战斗ease窗口内的单位重新下达A-Move
                        # =====================================================
                        if should_refresh_order:
                            issued_any = False
                            for unit in live_units_in_group:
                                in_micro_window = self.unit_ease.get(unit.tag, 0) > current_iteration
                                if unit.tag in self.dodging_tags:
                                    self._record_micro_event("attack_state:preserve_dodge", unit)
                                    continue
                                if (
                                    self.attack_engaged_retarget_idle_only
                                    and self.enemy_units.exists
                                    and not unit.is_idle
                                ):
                                    self._record_micro_event("attack_state:preserve_combat", unit)
                                    continue
                                if in_micro_window and not unit.is_idle:
                                    self._record_micro_event("attack_state:preserve_micro", unit)
                                    continue
                                if in_micro_window and unit.is_idle and not target_changed:
                                    unit.attack(target_pos)
                                    issued_any = True
                                    self._record_micro_event("attack_state:idle_retarget", unit)
                                    continue
                                unit.attack(target_pos)
                                issued_any = True
                                self._record_micro_event("attack_state:macro_retarget", unit)
                            if issued_any:
                                attack_data["last_macro_order_loop"] = current_iteration
                                attack_data["last_ordered_target_tag"] = new_target.tag
                        # =====================================================

                        # Keep the logical target aligned with an order that was
                        # actually issued. When every unit is already engaged,
                        # changing only the stored tag makes mobile/fog targets
                        # churn every frame and sends newly idle units in
                        # different directions.
                        if should_refresh_order and issued_any:
                            attack_data["target_tag"] = new_target.tag
                            attack_data["final_position"] = target_pos

                    else:
                        # 如果视野内无敌军，且有单位闲置，则向最后已知位置或敌方出生点A-Move
                        if has_idle_units:
                            final_target_pos = attack_data.get("final_position", self.enemy_start_locations[0])
                            last_fallback_loop = int(attack_data.get("last_fallback_order_loop", 0))
                            if current_iteration - last_fallback_loop >= self.attack_fallback_order_interval:
                                # self.logger.info(f"总攻波次 {wave_id}: 无视野目标, 闲置单位推进至{final_target_pos.rounded}。")
                                self.logging("manage_total_attack_groups", f"总攻波次 {wave_id}: 无视野目标, 闲置单位推进至{final_target_pos.rounded}。")
                                for unit in live_units_in_group.idle:
                                    if unit.tag in self.dodging_tags:
                                        self._record_micro_event("attack_state:preserve_dodge", unit)
                                        continue
                                    unit.attack(final_target_pos)
                                    self._record_micro_event("attack_state:fallback_order", unit)
                                attack_data["last_fallback_order_loop"] = current_iteration

                # (非闲置且目标存活时，继续执行原有指令，无需操作)
                pass


        # --- 循环外：执行清理 (删除全灭的编队) ---
        for wave_id in waves_to_delete:
            if wave_id in self.total_attack_groups:
                del self.total_attack_groups[wave_id]

    async def manage_attack(self):
        """
        【攻击总指挥】
        (V2: 仅维护总攻, 触发逻辑移至 execute_llm_attacks)

        根据游戏情况决定并执行一种攻击策略。
        此函数在 on_step 中被调用, 且在 automatic_defense (自动防御) 之后。
        """

        # --- 1. (必须) 持续维护所有已发起的"总攻"编队 ---
        # (此函数每帧运行，以更新现有攻击波次的状态,
        # 例如，在 GATHERING -> ATTACKING 切换, 或在目标丢失后自动寻找新目标)
        await self.manage_total_attack_groups()

        # --- 2. (已移除) ---
        # (自动触发总攻的硬编码逻辑已被移除)
        # (现在，攻击的发起由 AdjustAgent 的输出
        #  并通过 run() -> execute_llm_attacks() 触发)

    async def _resolve_attack_targets(self, target_pos_data, target_unit_name) -> list:
            """
            [新] 解析 AdjustAgent 攻击指令的目标。
            返回一个 [target_object_1, target_object_2, ...] 列表 (Point2 或 Unit)
            """
            final_targets = []

            # 优先级 1: target_position
            if target_pos_data:
                try:
                    # 检查数据是 [x,y] 还是 [[x1,y1], [x2,y2]]
                    if isinstance(target_pos_data[0], list):
                        # 多个位置: [[x1,y1], [x2,y2]]
                        for pos in target_pos_data:
                            final_targets.append(Point2((pos[0], pos[1])))
                    elif len(target_pos_data) == 2 and isinstance(target_pos_data[0], int):
                        # 单个位置: [x,y]
                        final_targets.append(Point2((target_pos_data[0], target_pos_data[1])))
                    else:
                        # self.logger.warning(f"无法解析 target_position 数据: {target_pos_data}")
                        self.logging("resolve_attack_targets", f"无法解析 target_position 数据: {target_pos_data}", "warning")
                except Exception as e:
                    # self.logger.error(f"解析 target_position 时出错: {e}. 数据: {target_pos_data}")
                    self.logging("resolve_attack_targets", f"解析 target_position 时出错: {e}. 数据: {target_pos_data}", "error")

            # 优先级 2: target_unit (if position was empty)
            elif target_unit_name:
                # 查找与 target_unit_name "最像" 的敌方单位
                all_enemies = self.enemy_units | self.enemy_structures
                if all_enemies.exists:

                    target_unit_name_lower = target_unit_name.lower()
                    best_match_unit = None

                    # 1. 尝试精确匹配
                    for unit in all_enemies:
                        if unit.name.lower() == target_unit_name_lower:
                            best_match_unit = unit
                            break

                    # 2. 如果没有精确匹配, 尝试包含匹配
                    if not best_match_unit:
                        for unit in all_enemies:
                            if target_unit_name_lower in unit.name.lower():
                                best_match_unit = unit
                                break # 找到第一个包含的就用

                    if best_match_unit:
                        # self.logger.info(f"已解析 target_unit '{target_unit_name}' -> 敌方 {best_match_unit.name} (Tag: {best_match_unit.tag})")
                        self.logging("resolve_attack_targets", f"已解析 target_unit '{target_unit_name}' -> 敌方 {best_match_unit.name} (Tag: {best_match_unit.tag})")
                        final_targets.append(best_match_unit)
                    else:
                        # self.logger.warning(f"在视野中未找到匹配 '{target_unit_name}' 的敌方单位。")
                        self.logging("resolve_attack_targets", f"在视野中未找到匹配 '{target_unit_name}' 的敌方单位。", "warning")

            # --- [!! 修改后的备选方案逻辑 !!] ---
            # 备选方案: 如果未找到任何目标...
            if not final_targets:
                # self.logger.warning(f"未解析到目标 '{target_pos_data}'/'{target_unit_name}'。 尝试寻找备选目标...")
                self.logging("resolve_attack_targets", f"未解析到目标 '{target_pos_data}'/'{target_unit_name}'。 尝试寻找备选目标...", "warning")

                # 1. [新] 备选方案 A: 寻找最近的敌方基地
                # (使用 __init__ 中定义的 self.enemy_townhall_types)
                enemy_townhalls = self.enemy_structures.filter(
                    lambda structure: structure.type_id in self.enemy_townhall_types
                )

                if enemy_townhalls.exists:
                    # 找到离 *我方出生点* 最近的敌方基地
                    closest_enemy_townhall = enemy_townhalls.closest_to(self.start_location)
                    final_targets.append(closest_enemy_townhall)
                    # self.logger.warning(f"备选方案A: 锁定最近的敌方基地 {closest_enemy_townhall.name} (Tag: {closest_enemy_townhall.tag})。")
                    self.logging("resolve_attack_targets", f"备选方案A: 锁定最近的敌方基地 {closest_enemy_townhall.name} (Tag: {closest_enemy_townhall.tag})。", "warning")

                # 2. [原] 备选方案 B: 寻找敌方出生点
                elif self.enemy_start_locations:
                    final_targets.append(self.enemy_start_locations[0])
                    # self.logger.warning(f"备选方案B: 未找到敌方基地, 默认攻击敌方出生点。")
                    self.logging("resolve_attack_targets", f"备选方案B: 未找到敌方基地, 默认攻击敌方出生点。", "warning")

                # 3. 最终失败
                else:
                    # self.logger.error("无法解析任何目标, 且敌方出生点未知。")
                    self.logging("resolve_attack_targets", "无法解析任何目标, 且敌方出生点未知。", "error")
            # --- [!! 修改结束 !!] ---

            return final_targets

    async def execute_llm_attacks(self, llm_attack_commands: list):
        """
        [新] 处理来自 AdjustAgent 的所有标准攻击指令。
        统一使用launch_total_attack(集结 -> 进攻)的流程。
        """
        # self.logger.info(f"收到 {len(llm_attack_commands)} 条来自 AdjustAgent 的攻击指令。")
        self.logging("execute_llm_attacks", f"收到 {len(llm_attack_commands)} 条来自 AdjustAgent 的攻击指令。")


        # 1. 获取本轮所有可用的战斗单位
        # (即不在任何已发起的 "总攻" 编队中的单位)
        busy_unit_tags = set()
        if self.total_attack_groups:
            for attack_data in self.total_attack_groups.values():
                busy_unit_tags.update(attack_data["unit_tags"])
        # 把正在侦察的单位也排除
        busy_unit_tags.update([self.active_scout_unit_tag])

        available_combat_units = self._get_all_combat_units()
        if any(self._command_requests_explicit_worker_support(command) for command in llm_attack_commands):
            # Workers stay outside every automatic combat path. They are added
            # only when the explicit BO attack composition requests SCVs.
            available_combat_units = available_combat_units | self.units.of_type({UnitTypeId.SCV})
        available_combat_units = available_combat_units.filter(
            lambda unit: unit.tag not in busy_unit_tags
        )

        # (重要) 转换为 Tag 集合, 以便高效管理和防止重复分配
        available_unit_tags = {u.tag for u in available_combat_units}

        for command in llm_attack_commands:
            command = self._prepare_attack_command(command, "initial")
            if command is None:
                continue
            requested_unit_count = max(1, command.get("unit_count", 0))
            command = self._relax_overlarge_explicit_attack(
                command,
                requested_unit_count,
                len(available_unit_tags),
                "initial",
            )
            requested_unit_count = max(1, command.get("unit_count", 0))
            use_all_available = self._attack_uses_all_available(command, requested_unit_count)
            explicit_type_counts = self._command_unit_type_counts(command)
            if explicit_type_counts and not use_all_available:
                requested_unit_count = sum(explicit_type_counts.values())
            min_attack_units = (
                self.first_llm_attack_min_units
                if self.total_attack_wave_id_counter == 0 and not self.total_attack_groups
                else self.llm_attack_min_units
            )
            unit_count = (
                len(available_unit_tags)
                if use_all_available
                else (
                    requested_unit_count
                    if explicit_type_counts
                    else max(requested_unit_count, min_attack_units)
                )
            )
            min_units_to_launch = (
                min_attack_units
                if use_all_available or not explicit_type_counts
                else max(1, requested_unit_count)
            )

            # 如果本轮已没有可用单位, 停止处理后续攻击指令
            if not available_unit_tags:
                # self.logger.warning("已无更多可用单位分配, 剩余的 LLM 攻击指令将存储。")
                self.logging("execute_llm_attacks", "已无更多可用单位分配, 剩余的 LLM 攻击指令将存储。", "warning")
                rein_command = copy.deepcopy(command)
                rein_command["timer"] = self.TIMER
                rein_command["min_units_to_launch"] = min_units_to_launch
                self._store_reinforcement_command(rein_command, timer=self.TIMER)
                continue

            target_pos_data = command.get("target_position", [])
            target_unit_name = command.get("target_unit", "")
            if len(available_unit_tags) < min_units_to_launch:
                self.logging(
                    "execute_llm_attacks",
                    f"可用单位 {len(available_unit_tags)} < 最小出击 {min_units_to_launch}, 当前 LLM 攻击指令将先存储等待续兵。",
                    "warning",
                )
                rein_command = copy.deepcopy(command)
                rein_command["timer"] = self.TIMER
                rein_command["min_units_to_launch"] = min_units_to_launch
                self._store_reinforcement_command(rein_command, timer=self.TIMER)
                continue

            # --- 2. 解析目标 ---
            target_list = await self._resolve_attack_targets(target_pos_data, target_unit_name)
            if not target_list and not target_pos_data and not target_unit_name:
                reference_pos = self.units.center if self.units.exists else self.start_location
                fallback_target = None
                if self.enemy_structures.exists:
                    fallback_target = self.enemy_structures.closest_to(reference_pos)
                elif self.enemy_units.exists:
                    fallback_target = self.enemy_units.closest_to(reference_pos)
                elif self.enemy_start_locations:
                    fallback_target = self.enemy_start_locations[0]
                if fallback_target:
                    target_list = [fallback_target]
                    self.logging("execute_llm_attacks", f"使用默认攻击目标: {fallback_target}", "warning")

            if not target_list:
                # self.logger.warning(f"无法解析攻击指令的目标: {command}")
                self.logging("execute_llm_attacks", f"无法解析攻击指令的目标: {command}", "warning")
                continue # 跳过此指令

            # --- 3. 统一攻击逻辑(Gather -> Attack) ---

            # 3.1 确定基准点(用于选择最近的兵)
            # 默认以第一个目标为锚点进行距离排序
            anchor_target = target_list[0]
            anchor_pos = anchor_target.position if isinstance(anchor_target, Unit) else anchor_target

            # 3.2 筛选并排序当前可用单位
            current_available_units = self.units.filter(lambda u: u.tag in available_unit_tags)
            if not current_available_units.exists:
                # self.logger.warning("已无更多可用单位分配, 当前 LLM 攻击指令将存储。")
                self.logging("execute_llm_attacks", "已无更多可用单位分配, 当前 LLM 攻击指令将存储。", "warning")
                rein_command = copy.deepcopy(command)
                rein_command["timer"] = self.TIMER
                rein_command["min_units_to_launch"] = min_units_to_launch
                self._store_reinforcement_command(rein_command, timer=self.TIMER)
                continue

            # 3.3 确定投入兵力
            units_to_commit = None

            units_to_commit = self._select_attack_units(
                current_available_units, command, anchor_pos, unit_count
            )
            selected_type_counts = self._unit_type_counts_from_units(units_to_commit)
            self.logging(
                "MULEMICRO_ATTACK_SELECTION",
                f"requested={command.get('unit_type_counts') or {'any': unit_count}} selected={selected_type_counts}",
            )
            # self.logger.info(f"指令请求 {unit_count} 单位，执行Attack (投入 {len(units_to_commit)} 单位)。")
            self.logging("execute_llm_attacks", f"指令请求 {unit_count} 单位，执行Attack (投入 {len(units_to_commit)} 单位)。")
            await self.chat_send(f"@ HQ: Request {unit_count} units to attack, send {len(units_to_commit)} units.")
            if not units_to_commit.exists:
                # self.logger.warning("未能成功投入兵力, 当前 LLM 攻击指令将存储。")
                self.logging("execute_llm_attacks", "未能成功投入兵力, 当前 LLM 攻击指令将存储。", "warning")
                self._defer_attack_until_ready(command, min_units_to_launch, timer=self.TIMER)
                continue
            adjusted_min_units = self._newexec_min_units_for_selected_attack(
                units_to_commit, min_units_to_launch
            )
            respect_explicit_min_units = False
            if len(units_to_commit) < adjusted_min_units:
                if (
                    explicit_type_counts
                    and self._explicit_attack_is_fully_selected(command, units_to_commit)
                ):
                    self.logging(
                        "MULEMICRO_ATTACK_SELECTION",
                        (
                            "initial respect complete explicit group size "
                            f"{len(units_to_commit)} instead of ready threshold {adjusted_min_units}"
                        ),
                        "warning",
                    )
                    adjusted_min_units = max(1, len(units_to_commit))
                    respect_explicit_min_units = True
                if (
                    self.newexec_fill_explicit_to_ready
                    and explicit_type_counts
                    and self._explicit_attack_is_fully_selected(command, units_to_commit)
                    and not respect_explicit_min_units
                ):
                    units_to_commit = self._fill_explicit_attack_to_ready_threshold(
                        current_available_units,
                        units_to_commit,
                        anchor_pos,
                        adjusted_min_units,
                        "initial",
                    )
                    if len(units_to_commit) >= adjusted_min_units:
                        selected_type_counts = self._unit_type_counts_from_units(units_to_commit)
                        self.logging(
                            "MULEMICRO_ATTACK_SELECTION",
                            f"initial filled selected={selected_type_counts}",
                        )
                    else:
                        self.logging(
                            "MULEMICRO_ATTACK_SELECTION",
                            (
                                "selected complete explicit group still below newexec ready threshold "
                                f"{len(units_to_commit)}/{adjusted_min_units}; queued"
                            ),
                            "warning",
                        )
                        self._defer_attack_until_ready(command, adjusted_min_units, timer=self.TIMER)
                        continue
                if (
                    self.newexec_complete_explicit_delayed_release
                    and explicit_type_counts
                    and self._explicit_attack_is_fully_selected(command, units_to_commit)
                    and self._newexec_complete_explicit_should_delay(units_to_commit)
                    and len(units_to_commit) < adjusted_min_units
                    and not respect_explicit_min_units
                ):
                    if command.get("_newexec_complete_explicit_waited"):
                        self.logging(
                            "MULEMICRO_ATTACK_SELECTION",
                            (
                                "selected complete explicit group below newexec ready threshold "
                                f"{len(units_to_commit)}/{adjusted_min_units}; delayed release"
                            ),
                            "warning",
                        )
                        adjusted_min_units = max(1, len(units_to_commit))
                    else:
                        self._defer_complete_explicit_delayed_release(
                            command, units_to_commit, "initial"
                        )
                        continue
                if (
                    self.newexec_complete_explicit_direct_passthrough
                    and explicit_type_counts
                    and self._explicit_attack_is_fully_selected(command, units_to_commit)
                ):
                    self.logging(
                        "MULEMICRO_ATTACK_SELECTION",
                        (
                            "selected complete explicit group below newexec ready threshold "
                            f"{len(units_to_commit)}/{adjusted_min_units}; direct passthrough"
                        ),
                        "warning",
                    )
                    await self._launch_direct_attack_passthrough(
                        target_list, units_to_commit, available_unit_tags, "initial"
                    )
                    continue
                if adjusted_min_units <= len(units_to_commit):
                    pass
                else:
                    self.logging(
                        "MULEMICRO_ATTACK_SELECTION",
                        (
                            "selected group below newexec ready threshold "
                            f"{len(units_to_commit)}/{adjusted_min_units}; queued"
                        ),
                        "warning",
                    )
                    self._defer_attack_until_ready(command, adjusted_min_units, timer=self.TIMER)
                    continue
            min_units_to_launch = adjusted_min_units
            if explicit_type_counts and len(units_to_commit) < min_units_to_launch and not self.allow_explicit_small_attacks:
                self.logging(
                    "MULEMICRO_ATTACK_SELECTION",
                    f"exact composition below launch minimum {len(units_to_commit)}/{min_units_to_launch}; queued",
                    "warning",
                )
                self._defer_attack_until_ready(command, min_units_to_launch, timer=self.TIMER)
                continue
            if not target_pos_data and not target_unit_name:
                units_first_target = self._newexec_units_first_initial_target(units_to_commit)
                if units_first_target is not None:
                    target_list = [units_first_target]
                    self.logging(
                        "MULEMICRO_ATTACK_SELECTION",
                        f"initial units-first target={units_first_target.name} at {units_first_target.position.rounded}",
                    )
            if (not use_all_available) and unit_count > len(units_to_commit):   # 指令要求兵力大于可投入兵力，需要追加
                # self.logger.warning(f"投入兵力不足, 将存储当前 LLM 攻击指令的剩余部分。仍需投入：{unit_count - len(units_to_commit)} 单位。")
                self.logging("execute_llm_attacks", f"投入兵力不足, 将存储当前 LLM 攻击指令的剩余部分。仍需投入：{unit_count - len(units_to_commit)} 单位。", "warning")
                rein_command = copy.deepcopy(command)
                rein_command["timer"] = self.TIMER
                rein_command["unit_count"] = unit_count - len(units_to_commit)
                remaining_counts = self._remaining_unit_type_counts(command, units_to_commit)
                if remaining_counts:
                    rein_command["unit_type_counts"] = remaining_counts
                    rein_command["unit_count"] = sum(remaining_counts.values())
                rein_command["min_units_to_launch"] = max(self.reinforcement_min_units, min_units_to_launch)
                self._store_reinforcement_command(rein_command, timer=self.TIMER)

            # 3.4 分配并执行
            # 如果有多个目标，将兵力平分
            num_targets = len(target_list)
            count_committed = len(units_to_commit)
            units_per_target = math.ceil(count_committed / num_targets)
            if units_per_target < 1:
                units_per_target = 1

            units_list_committed = list(units_to_commit)
            curr_idx = 0

            for target in target_list:
                # 切片取出分队
                sub_group_list = units_list_committed[curr_idx : curr_idx + units_per_target]
                if not sub_group_list:
                    break
                # 构造 Units 对象 (launch_total_attack需要)
                sub_group = self.units.filter(lambda u: u.tag in {u.tag for u in sub_group_list})
                if sub_group.exists:
                    # 从总可用池移除已分配的单位
                    for u in sub_group:
                        available_unit_tags.discard(u.tag)

                    # --- 核心修改 ---
                    # 调用 unified logic: launch_total_attack
                    # 这将触发: 创建编队 -> 集结(GATHERING) -> 进攻(ATTACKING)
                    # 进攻时会自动指向 target 的位置(如果是 Unit 则攻击其坐标)，确保 A-Move 攻击路上的敌人
                    await self.launch_total_attack(
                        target,
                        sub_group,
                        min_units_to_launch=min_units_to_launch,
                        respect_min_units=respect_explicit_min_units,
                    )

                    t_str = target.position.rounded if isinstance(target, Unit) else target.rounded
                    # self.logger.info(f" -> 发起进攻波次: 目标 {t_str}，单位数 {len(sub_group)}")
                    self.logging("execute_llm_attacks", f" -> 发起进攻波次: 目标 {t_str}，单位数 {len(sub_group)}")
                    await self.chat_send(f"@ Attacking Command from HQ: Attack {t_str} with {len(sub_group)} units! Copy that!")

                curr_idx += units_per_target

    async def execute_reinforcement_attacks(self):
        """
        [wlx] 处理所有存储的追加攻击指令。
        统一使用launch_total_attack(集结 -> 进攻)的流程。
        """

        # 0. 更新并管理追加进攻指令列表
        for command in self.reinforcements_commands:
            if "timer" in command:
                command["timer"] -= 1 # 函数每帧调用 timer进行计时
            else:
                command["timer"] = self.TIMER # 防止结构不对没有timer的命令
        remaining_commands = [command for command in self.reinforcements_commands if command["timer"] != 0]
        executing_commands = [command for command in self.reinforcements_commands if command["timer"] == 0]
        self.reinforcements_commands = remaining_commands
        if len(executing_commands) == 0:
            return

        # 1. 获取本轮所有可用的战斗单位
        # (即不在任何已发起的 "总攻" 编队中的单位)
        busy_unit_tags = set()
        if self.total_attack_groups:
            for attack_data in self.total_attack_groups.values():
                busy_unit_tags.update(attack_data["unit_tags"])
        # 把正在侦察的单位也排除
        busy_unit_tags.update([self.active_scout_unit_tag])

        available_combat_units = self._get_all_combat_units()
        if any(self._command_requests_explicit_worker_support(command) for command in executing_commands):
            available_combat_units = available_combat_units | self.units.of_type({UnitTypeId.SCV})
        available_combat_units = available_combat_units.filter(
            lambda unit: unit.tag not in busy_unit_tags
        )

        # (重要) 转换为 Tag 集合, 以便高效管理和防止重复分配
        available_unit_tags = {u.tag for u in available_combat_units}

        for command in executing_commands:
            command = self._prepare_attack_command(command, "reinforcement")
            if command is None:
                continue
            requested_unit_count = max(1, command.get("unit_count", 0))
            command = self._relax_overlarge_explicit_attack(
                command,
                requested_unit_count,
                len(available_unit_tags),
                "reinforcement",
            )
            requested_unit_count = max(1, command.get("unit_count", 0))
            use_all_available = self._attack_uses_all_available(command, requested_unit_count)
            explicit_type_counts = self._command_unit_type_counts(command)
            if explicit_type_counts and not use_all_available:
                requested_unit_count = sum(explicit_type_counts.values())
            first_wave_pending = (
                self.total_attack_wave_id_counter == 0 and not self.total_attack_groups
            )
            base_min_units = (
                self.first_llm_attack_min_units
                if first_wave_pending
                else self.reinforcement_min_units
            )
            unit_count = (
                len(available_unit_tags)
                if use_all_available
                else (
                    requested_unit_count
                    if explicit_type_counts
                    else max(requested_unit_count, base_min_units)
                )
            )
            target_pos_data = command.get("target_position", [])
            target_unit_name = command.get("target_unit", "")
            stored_min_units = max(1, int(command.get("min_units_to_launch", 1)))
            min_units_to_launch = (
                max(base_min_units, stored_min_units)
                if use_all_available or not explicit_type_counts
                else max(1, min(requested_unit_count, stored_min_units))
            )

            # 如果可用单位不够一个有效波次, 放回存储列表继续攒兵。
            if len(available_unit_tags) < min_units_to_launch:
                self.logging(
                    "execute_reinforcement_attacks",
                    f"可用单位 {len(available_unit_tags)} < 最小出击 {min_units_to_launch}, 本次的 LLM 追加攻击指令将重新存储。",
                    "warning",
                )
                command["timer"] = self.TIMER
                self._store_reinforcement_command(command, timer=self.TIMER)
                continue

            # --- 2. 解析目标 ---
            target_list = await self._resolve_attack_targets(target_pos_data, target_unit_name)
            if not target_list and not target_pos_data and not target_unit_name:
                reference_pos = self.units.center if self.units.exists else self.start_location
                fallback_target = None
                if self.enemy_structures.exists:
                    fallback_target = self.enemy_structures.closest_to(reference_pos)
                elif self.enemy_units.exists:
                    fallback_target = self.enemy_units.closest_to(reference_pos)
                elif self.enemy_start_locations:
                    fallback_target = self.enemy_start_locations[0]
                if fallback_target:
                    target_list = [fallback_target]
                    self.logging("execute_reinforcement_attacks", f"使用默认攻击目标: {fallback_target}", "warning")

            if not target_list:
                # self.logger.warning(f"无法解析攻击指令的目标: {command}")
                self.logging("execute_reinforcement_attacks", f"无法解析攻击指令的目标: {command}", "warning")
                continue # 跳过此指令

            # --- 3. 统一攻击逻辑(Gather -> Attack) ---

            # 3.1 确定基准点(用于选择最近的兵)
            # 默认以第一个目标为锚点进行距离排序
            anchor_target = target_list[0]
            anchor_pos = anchor_target.position if isinstance(anchor_target, Unit) else anchor_target

            # 3.2 筛选并排序当前可用单位
            current_available_units = self.units.filter(lambda u: u.tag in available_unit_tags)
            if not current_available_units.exists:
                # self.logger.warning("已无更多可用单位分配, 本次的 LLM 追加 攻击指令将重新存储。")
                self.logging("execute_reinforcement_attacks", "已无更多可用单位分配, 本次的 LLM 追加 攻击指令将重新存储。", "warning")
                command["timer"] = self.TIMER
                self._store_reinforcement_command(command, timer=self.TIMER)
                continue

            # 3.3 确定投入兵力
            units_to_commit = None

            units_to_commit = self._select_attack_units(
                current_available_units, command, anchor_pos, unit_count
            )
            selected_type_counts = self._unit_type_counts_from_units(units_to_commit)
            self.logging(
                "MULEMICRO_ATTACK_SELECTION",
                f"reinforcement requested={command.get('unit_type_counts') or {'any': unit_count}} selected={selected_type_counts}",
            )
            # self.logger.info(f"指令请求 {unit_count} 单位，执行Attack (投入 {len(units_to_commit)} 单位)。")
            self.logging("execute_reinforcement_attacks", f"指令请求 {unit_count} 单位，执行Attack (投入 {len(units_to_commit)} 单位)。")
            await self.chat_send(f"--- Reinforce: Request {unit_count} units to attack, send {len(units_to_commit)} units.")
            if not units_to_commit.exists:
                # self.logger.warning("未能成功投入兵力, 本次的 LLM 追加 攻击指令将重新存储。")
                self.logging("execute_reinforcement_attacks", "未能成功投入兵力, 本次的 LLM 追加 攻击指令将重新存储。", "warning")
                self._defer_attack_until_ready(command, min_units_to_launch, timer=self.TIMER)
                continue
            adjusted_min_units = self._newexec_min_units_for_selected_attack(
                units_to_commit, min_units_to_launch
            )
            respect_explicit_min_units = False
            if len(units_to_commit) < adjusted_min_units:
                if (
                    explicit_type_counts
                    and self._explicit_attack_is_fully_selected(command, units_to_commit)
                ):
                    self.logging(
                        "MULEMICRO_ATTACK_SELECTION",
                        (
                            "reinforcement respect complete explicit group size "
                            f"{len(units_to_commit)} instead of ready threshold {adjusted_min_units}"
                        ),
                        "warning",
                    )
                    adjusted_min_units = max(1, len(units_to_commit))
                    respect_explicit_min_units = True
                if (
                    self.newexec_fill_explicit_to_ready
                    and explicit_type_counts
                    and self._explicit_attack_is_fully_selected(command, units_to_commit)
                    and not respect_explicit_min_units
                ):
                    units_to_commit = self._fill_explicit_attack_to_ready_threshold(
                        current_available_units,
                        units_to_commit,
                        anchor_pos,
                        adjusted_min_units,
                        "reinforcement",
                    )
                    if len(units_to_commit) >= adjusted_min_units:
                        selected_type_counts = self._unit_type_counts_from_units(units_to_commit)
                        self.logging(
                            "MULEMICRO_ATTACK_SELECTION",
                            f"reinforcement filled selected={selected_type_counts}",
                        )
                    else:
                        self.logging(
                            "MULEMICRO_ATTACK_SELECTION",
                            (
                                "reinforcement selected complete explicit group still below newexec ready threshold "
                                f"{len(units_to_commit)}/{adjusted_min_units}; queued"
                            ),
                            "warning",
                        )
                        self._defer_attack_until_ready(command, adjusted_min_units, timer=self.TIMER)
                        continue
                if (
                    self.newexec_complete_explicit_delayed_release
                    and explicit_type_counts
                    and self._explicit_attack_is_fully_selected(command, units_to_commit)
                    and self._newexec_complete_explicit_should_delay(units_to_commit)
                    and len(units_to_commit) < adjusted_min_units
                    and not respect_explicit_min_units
                ):
                    if command.get("_newexec_complete_explicit_waited"):
                        self.logging(
                            "MULEMICRO_ATTACK_SELECTION",
                            (
                                "reinforcement selected complete explicit group below newexec ready threshold "
                                f"{len(units_to_commit)}/{adjusted_min_units}; delayed release"
                            ),
                            "warning",
                        )
                        adjusted_min_units = max(1, len(units_to_commit))
                    else:
                        self._defer_complete_explicit_delayed_release(
                            command, units_to_commit, "reinforcement"
                        )
                        continue
                if (
                    self.newexec_complete_explicit_direct_passthrough
                    and explicit_type_counts
                    and self._explicit_attack_is_fully_selected(command, units_to_commit)
                ):
                    self.logging(
                        "MULEMICRO_ATTACK_SELECTION",
                        (
                            "reinforcement selected complete explicit group below newexec ready threshold "
                            f"{len(units_to_commit)}/{adjusted_min_units}; direct passthrough"
                        ),
                        "warning",
                    )
                    await self._launch_direct_attack_passthrough(
                        target_list, units_to_commit, available_unit_tags, "reinforcement"
                    )
                    continue
                if adjusted_min_units <= len(units_to_commit):
                    pass
                else:
                    self.logging(
                        "MULEMICRO_ATTACK_SELECTION",
                        (
                            "reinforcement selected group below newexec ready threshold "
                            f"{len(units_to_commit)}/{adjusted_min_units}; queued"
                        ),
                        "warning",
                    )
                    self._defer_attack_until_ready(command, adjusted_min_units, timer=self.TIMER)
                    continue
            min_units_to_launch = adjusted_min_units
            if explicit_type_counts and len(units_to_commit) < min_units_to_launch and not self.allow_explicit_small_attacks:
                self.logging(
                    "MULEMICRO_ATTACK_SELECTION",
                    f"reinforcement exact composition below launch minimum {len(units_to_commit)}/{min_units_to_launch}; queued",
                    "warning",
                )
                self._defer_attack_until_ready(command, min_units_to_launch, timer=self.TIMER)
                continue
            if not target_pos_data and not target_unit_name:
                units_first_target = self._newexec_units_first_initial_target(units_to_commit)
                if units_first_target is not None:
                    target_list = [units_first_target]
                    self.logging(
                        "MULEMICRO_ATTACK_SELECTION",
                        f"reinforcement units-first target={units_first_target.name} at {units_first_target.position.rounded}",
                    )
            if (not use_all_available) and unit_count > len(units_to_commit):   # 指令要求兵力大于可投入兵力，需要追加
                # self.logger.warning(f"投入兵力不足, 本次的 LLM 追加 攻击指令将继续存储剩余部分。仍需投入：{unit_count - len(units_to_commit)} 单位。")
                self.logging("execute_reinforcement_attacks", f"投入兵力不足, 本次的 LLM 追加 攻击指令将继续存储剩余部分。仍需投入：{unit_count - len(units_to_commit)} 单位。", "warning")
                command["timer"] = self.TIMER
                command["unit_count"] = unit_count - len(units_to_commit)
                remaining_counts = self._remaining_unit_type_counts(command, units_to_commit)
                if remaining_counts:
                    command["unit_type_counts"] = remaining_counts
                    command["unit_count"] = sum(remaining_counts.values())
                command["min_units_to_launch"] = max(self.reinforcement_min_units, min_units_to_launch)
                self._store_reinforcement_command(command, timer=self.TIMER)

            # 3.4 分配并执行
            # 如果有多个目标，将兵力平分
            num_targets = len(target_list)
            count_committed = len(units_to_commit)
            units_per_target = math.ceil(count_committed / num_targets)
            if units_per_target < 1:
                units_per_target = 1

            units_list_committed = list(units_to_commit)
            curr_idx = 0

            for target in target_list:
                # 切片取出分队
                sub_group_list = units_list_committed[curr_idx : curr_idx + units_per_target]
                if not sub_group_list:
                    break
                # 构造 Units 对象 (launch_total_attack需要)
                sub_group = self.units.filter(lambda u: u.tag in {u.tag for u in sub_group_list})
                if sub_group.exists:
                    # 从总可用池移除已分配的单位
                    for u in sub_group:
                        available_unit_tags.discard(u.tag)

                    # --- 核心修改 ---
                    # 调用 unified logic: launch_total_attack
                    # 这将触发: 创建编队 -> 集结(GATHERING) -> 进攻(ATTACKING)
                    # 进攻时会自动指向 target 的位置(如果是 Unit 则攻击其坐标)，确保 A-Move 攻击路上的敌人
                    await self.launch_total_attack(
                        target,
                        sub_group,
                        min_units_to_launch=min_units_to_launch,
                        respect_min_units=respect_explicit_min_units,
                    )

                    t_str = target.position.rounded if isinstance(target, Unit) else target.rounded
                    # self.logger.info(f" -> 发起进攻波次: 目标 {t_str}，单位数 {len(sub_group)}")
                    self.logging("execute_reinforcement_attacks", f" -> 发起进攻波次: 目标 {t_str}，单位数 {len(sub_group)}")
                    await self.chat_send(f"--- Reinforcement Attacking Command: Attack {t_str} with {len(sub_group)} units! Support our comrades!")

                curr_idx += units_per_target

    async def execute_auto_attack(self):
        """
        Restored baseline scripted attack scheduler.

        The new execution layer still handles launch/gather/attack, but the
        old clean baseline relied on these thresholds from LLMPlayer.strategy_params.
        Dropping this path changes the experiment protocol and made the repro
        runs sit/passively fail.
        """
        busy_unit_tags = set()
        if self.total_attack_groups:
            for attack_data in self.total_attack_groups.values():
                busy_unit_tags.update(attack_data["unit_tags"])

        all_combat_units = self._get_all_combat_units()
        scout_tag = self.active_scout_unit_tag
        available_combat_units = all_combat_units.filter(
            lambda unit: unit.tag not in busy_unit_tags and unit.tag != scout_tag
        )
        available_count = available_combat_units.amount

        target = None
        enemy_bases = self.enemy_structures.filter(lambda s: s.type_id in self.enemy_townhall_types)
        if enemy_bases.exists:
            target = enemy_bases.closest_to(self.start_location).position
        elif self.enemy_start_locations:
            target = self.enemy_start_locations[0]
        else:
            target = self.game_info.map_center

        if not target:
            return

        units_to_launch = None
        if not self.has_launched_first_attack:
            first_threshold = self.attack_config["first_attack_threshold"]
            if available_count >= first_threshold:
                self.has_launched_first_attack = True
                units_to_launch = available_combat_units
                print(f"[Auto-Attack] 首次总攻触发！当前兵力 {available_count} >= {first_threshold}")
            else:
                triggers = self.attack_config["harassment_triggers"]
                if self.harassment_wave_index < len(triggers):
                    req_threshold, send_amount = triggers[self.harassment_wave_index]
                    if available_count >= req_threshold:
                        units_to_launch = list(available_combat_units)[:send_amount]
                        self.harassment_wave_index += 1
                        print(
                            f"[Auto-Attack] 触发第 {self.harassment_wave_index} 波袭扰！"
                            f"(阈值 {req_threshold}, 派出 {send_amount})"
                        )
        else:
            sub_threshold = self.attack_config["subsequent_attack_threshold"]
            if available_count >= sub_threshold:
                units_to_launch = available_combat_units
                print(f"[Auto-Attack] 后续波次触发！当前积累 {available_count} >= {sub_threshold}")

        if units_to_launch and len(units_to_launch) > 0:
            if scout_tag:
                units_to_launch = [u for u in units_to_launch if u.tag != scout_tag]
            if not units_to_launch:
                return
            final_units = self.units.filter(lambda u: u.tag in {x.tag for x in units_to_launch})
            await self.launch_total_attack(target, final_units)


    # ==========================================
    # [新增] 统一维护接口 (已修改: 集成微操系统)
    # ==========================================
    async def run_maintenance_tasks(
        self,
        iteration: int,
        enable_early_scv_defense: Optional[bool] = None,
        early_scv_defense_threshold: Optional[int] = None,
        first_attack_threshold: Optional[int] = None,
        subsequent_attack_threshold: Optional[int] = None,
        harassment_triggers: Optional[List[tuple]] = None,
        **_unused_strategy_params,
    ):
        """
        统一执行所有脚本化的维护任务。
        LLMPlayer 的 run() 方法应调用此方法。

        [修改] 新增微操系统调用:
        - micro_update_health_snapshot(): 在最开头更新血量快照
        - micro_dodge(): 在manage_attack之后执行闪避
        - micro_fight(): 在闪避之后执行战斗微操
        """
        if enable_early_scv_defense is not None:
            self.enable_early_scv_defense = enable_early_scv_defense
        if early_scv_defense_threshold is not None:
            self.early_scv_defense_threshold = early_scv_defense_threshold
        if first_attack_threshold is not None:
            self.attack_config["first_attack_threshold"] = first_attack_threshold
        if subsequent_attack_threshold is not None:
            self.attack_config["subsequent_attack_threshold"] = subsequent_attack_threshold
        if harassment_triggers is not None:
            if self.attack_config.get("harassment_triggers") != harassment_triggers:
                self.attack_config["harassment_triggers"] = harassment_triggers
                self.harassment_wave_index = 0
        _ = _unused_strategy_params
        self.current_micro_iteration = iteration

        # =====================================================
        # [新增] 0. 更新血量快照 (必须在所有逻辑之前)
        # 参考 army.py 的 last_health 机制
        # =====================================================
        if self._newexec_micro_active_for_current_army():
            self.micro_update_health_snapshot()

        # 1. 资源管理
        await self.distribute_workers()
        if self.config.own_race == "Terran":
            mule_success, mule_msg = await self._attempt_auto_calldown_mule()
            if mule_success:
                print(f"[{self.time_formatted}] [Auto-Maintenance] {mule_msg}")

            # 2. 建筑集结点维护
            await self.set_terran_combat_rally_points()

            # 3. 自动防御
            await self.automatic_defense()

            # 4. 自动侦察
            new_scouting_info = await self.manage_scouting()

            # 5. 恢复旧 baseline 的自动进攻阈值逻辑，再处理追加兵力
            await self.execute_auto_attack()
            await self.execute_reinforcement_attacks()

            # 6. 攻击编队状态维护 (总攻状态机: 索敌、A-Move 宏观方向)
            await self.manage_attack()

            # =====================================================
            # [新增] 6.5 战斗微操系统 (在总攻维护之后执行)
            # 执行顺序说明:
            #   manage_attack() 先运行 → 总攻状态机正常推进，给单位基础A-Move
            #   micro_dodge()   紧随其后 → 闪避危险效果，最高优先级
            #   micro_fight()   最后运行 → 对ATTACKING状态的单位覆盖精细走位/射击
            # 这样就是"总攻给方向，微操给细节"的覆盖关系
            # 参考 army.py 中 do_attack_goalofarmy → fight 的逻辑
            # =====================================================
            if self._newexec_micro_active_for_current_army():
                await self.micro_dodge(iteration)
                await self.micro_special_abilities(iteration)
                await self.micro_fight(iteration)
                self._flush_micro_trace(iteration)
            # =====================================================

            # 7. 驻防逻辑 (低频)
            garrison_check_interval = 66
            if iteration - self.last_garrison_check_time > garrison_check_interval:
                await self.manage_garrison()
                self.last_garrison_check_time = iteration

            return new_scouting_info
