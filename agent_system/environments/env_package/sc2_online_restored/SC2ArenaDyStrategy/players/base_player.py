from sc2.bot_ai import BotAI
from sc2.unit import Unit
from sc2.units import Units
from sc2.position import Point2
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId

import time
import os
import json
import math
import pandas as pd
import random

from tools.logger import setup_logger
from tools.format import extract_code, extract_first_number
from tools.ops import IterativeMean


class TargetType:
    NONE = "None"
    POINT = "Point"
    UNIT = "Unit"
    POINT_OR_UNIT = "PointOrUnit"


def load_knowledge():
    knowledge_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "knowledge"))
    TerranAbilityData = pd.read_csv(os.path.join(knowledge_dir, "TerranAbility.csv"))
    with open(os.path.join(knowledge_dir, "data.json"), "r") as f:
        game_data = json.load(f)

    TerranAbility = {}
    for idx, item in TerranAbilityData.iterrows():
        ability = item["ability"]
        description = item["description"]
        ability_data = [item for item in game_data["Ability"] if item["name"] == ability]
        if len(ability_data) == 0:
            target = TargetType.NONE
        else:
            target = ability_data[0]["target"]
        if not isinstance(target, str):
            if "Build" in target:
                target = TargetType.POINT
            elif "BuildOnUnit" in target or "Unit" in target:
                target = TargetType.UNIT
            else:
                target = TargetType.NONE
        TerranAbility[ability] = {
            "enabled": item["enabled"],
            "description": description,
            "target": target,  # None, Point, Unit, PointOrUnit
        }
    return TerranAbility


TerranAbility = load_knowledge()


class BasePlayer(BotAI):
    def __init__(self, config, player_name, model_name, generation_config, llm_client, log_path="logs", enable_logging=True):
        super().__init__()

        self.config = config
        self.player_name = player_name
        self.model_name = model_name
        self.generation_config = generation_config
        self.llm_client = llm_client
        map_size = str(extract_first_number(config.map_name))
        self.map_name = f"{map_size}x{map_size}"

        time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        self.real_model_name = self.model_name.split("/")[-1]

        self.enable_logging = enable_logging
        if enable_logging:
            self.log_path = f"{log_path}/{self.real_model_name}/{time_str}"
            os.makedirs(f"{self.log_path}/observation", exist_ok=True)
            self.logger = setup_logger(f"{player_name}_{self.real_model_name}", log_dir=self.log_path)

        self._tag_to_id = {}
        self._id_to_tag = {}
        self._id_to_abilities = {}
        self.next_id = 1

        self.last_action = []
        self.trace = {}
        self.tag_to_health = {}

        self.sbr = IterativeMean()
        self.resource_cost = 0

        self.miner_units = ["SCV", "Probe", "Drone"]

    def logging(self, key: str, value, level="info", save_trace=False, save_file=False, print_log=True):
        if not self.enable_logging:
            return
        idx = self.state.game_loop // 4
        if level in ["info", "warning", "error"] and print_log:
            text = f"({idx}) {key}: {str(value)}"
            if level == "info":
                self.logger.info(text)
            elif level == "warning":
                self.logger.warning(text)
            elif level == "error":
                self.logger.error(text)

        if save_trace:
            if idx not in self.trace:
                self.trace[idx] = {}
            self.trace[idx][key] = value
            if idx % 500 == 0:
                with open(f"{self.log_path}/trace.json", "w", encoding="utf-8") as f:
                    json.dump(self.trace, f, indent=2, ensure_ascii=False)

        if save_file:
            with open(f"{self.log_path}/observation/{idx}-{key}.txt", "w", encoding="utf-8") as f:
                if isinstance(value, list) or isinstance(value, dict):
                    value = json.dumps(value, indent=2, ensure_ascii=False)
                f.write(value)

    async def on_end(self, game_result):
        game_result = game_result.name
        self.logging("game_result", game_result, save_trace=True)
        self.logging("SBR", round(self.sbr.mean, 4), save_trace=True)

        time_cost = self.time_formatted.split(":")
        time_cost = int(time_cost[0]) * 60 + int(time_cost[1])
        self.logging("time_cost", time_cost, save_trace=True)
        self.logging("RUR", round(self.resource_cost / time_cost, 4), save_trace=True)

        with open(f"{self.log_path}/trace.json", "w", encoding="utf-8") as f:
            json.dump(self.trace, f, indent=2, ensure_ascii=False)

    def update_tag_to_health(self):
        self.tag_to_health = {unit.tag: unit.health for unit in self.units}
        self.tag_to_health.update({unit.tag: unit.health for unit in self.structures})

    def get_lowest_health_enemy(self, units: Units):
        """Get the enemy unit with the lowest health."""
        if not units.exists:
            return None
        return min(units, key=lambda unit: unit.health + unit.shield)

    def _can_build(self, unit_type):
        """辅助函数，检查是否可以且尚未开始建造某个单位/建筑。"""
        return self.can_afford(unit_type) and not self.already_pending(unit_type)

    def get_total_amount(self, unit_type: UnitTypeId):
        """获取指定单位类型的总数量，包括正在建造的和已完成的。"""
        unit_amount = self.units(unit_type).amount
        structures_amount = self.structures(unit_type).amount
        pending_amount = self.already_pending(unit_type)
        return unit_amount + structures_amount + pending_amount

    async def on_step(self, iteration: int):
        if len(self.units) == 0 or len(self.townhalls) == 0:
            return
        self.sbr.update(int(self.supply_used == self.supply_cap))

        #### before run
        await self.run(iteration)
        #### after run

        if iteration % 15 == 0:
            self.update_tag_to_health()

    async def run(self, iteration: int):
        raise NotImplementedError

    def verify_actions(self, actions):
        if isinstance(actions, str):
            try:
                actions = json.loads(extract_code(actions))
            except json.JSONDecodeError:
                return False, "Action must be a json list wrapped with triple backticks and without comments"
        if not isinstance(actions, list):
            return False, "Action must be a list"

        errors = []
        cost_minerals, cost_vespene, cost_supply = 0, 0, 0
        for action in actions:
            ok, message_or_cost = self.check_action(action)
            if not ok:
                errors.append(json.dumps(action, indent=2, ensure_ascii=False) + "\n>> Error: " + message_or_cost)
            else:
                cost_minerals += message_or_cost[0]
                cost_vespene += message_or_cost[1]
                cost_supply += message_or_cost[2]

        if self.minerals < cost_minerals:
            errors.append(">>>> Total actions error: minerals is not enough for executing all actions")
        if self.vespene < cost_vespene:
            errors.append(">>>> Total actions error: vespene is not enough for executing all actions")
        if self.supply_left < cost_supply:
            errors.append(">>>> Total actions error: supply is not enough for executing all actions")

        if errors:
            return False, "\n\n".join(errors)
        return True, ""

    def check_action(self, action: dict):
        if not isinstance(action, dict):
            return False, "Action must be a dictionary"

        ### required keys checks
        # action check
        base_keys = ["action", "units"]
        for key in base_keys:
            if key not in action:
                return False, f"Missing required key: {key}"
        action_name = action["action"]
        if action_name not in TerranAbility:
            return False, f"Unknown action: {action['action']}"
        # target check
        target_type = TerranAbility[action_name]["target"]
        if target_type == TargetType.NONE:
            required_keys = base_keys
        elif target_type == TargetType.POINT:
            required_keys = base_keys + ["target_position"]
        elif target_type == TargetType.UNIT:
            required_keys = base_keys + ["target_unit"]

        if target_type != TargetType.POINT_OR_UNIT:
            unused_keys = [key for key in action.keys() if key not in required_keys]
            for key in required_keys:
                if key not in action:
                    return False, f"Missing required key: {key}"
        else:
            if "target_position" in action and "target_unit" in action:
                return False, "Cannot have both `target_position` and `target_unit`"
            if "target_position" not in action and "target_unit" not in action:
                return False, "Missing required key: target_position or target_unit"
            unused_keys = [key for key in action.keys() if key not in base_keys + ["target_position", "target_unit"]]
        ### unused keys check
        if unused_keys:
            # unused_keys = [key for key in action.keys() if key not in ["action", "units"]]
            return False, f"Unused keys: {unused_keys}"

        ### value check
        if not isinstance(action_name, str):
            return False, "`action` must be a string"
        if not (isinstance(action["units"], list) and len(action["units"]) > 0):
            return False, "`units` must be a non-empty list of integers"
        if "target_position" in action:
            if not (
                isinstance(action["target_position"], list)
                and len(action["target_position"]) == 2
                and all(isinstance(i, int) for i in action["target_position"])
            ):
                return False, "`target_position` must be a list of two integers"
        if "target_unit" in action:
            if not isinstance(action["target_unit"], int):
                return False, "`target_unit` must be an integer"
            if action["target_unit"] not in self._id_to_tag:
                return False, f"Unit with id {action['target_unit']} not found"
            target_unit = self.get_unit_by_id(action["target_unit"])
            if target_unit is None:
                return False, f"Unit with id {action['target_unit']} not found"

        ### unit checks
        if len(action["units"]) == 0:
            return False, "`units` must not be an empty list"
        for unit_id in action["units"]:
            if not isinstance(unit_id, int):
                return False, "`units` must be a list of integers"
            if unit_id not in self._id_to_tag:
                return False, f"Unit with id {unit_id} not found"
            if unit_id not in self._id_to_abilities:
                return False, f"Unit with id {unit_id} not found"
            unit = self.get_unit_by_id(unit_id)
            if not unit:
                return False, f"Unit {unit_id} doesn't exist"
            if not unit.is_mine:
                return False, f"Unit {unit_id} is not mine"
            if action_name not in self._id_to_abilities[unit_id]:
                return False, f"[{unit_id}]{unit.name} cannot perform action {action_name}"
            if unit.is_constructing_scv:
                return False, f"[{unit_id}]{unit.name} is constructing, cannot perform other actions"

        ### action_name check
        building_units = self.get_building_units()
        building_units = [name.lower() for name in building_units]
        # if action_name == "TERRANBUILD_SUPPLYDEPOT":
        #     if self.supply_cap - self.supply_used >= 8:
        #         return False, "There is still enough supply count, no need to build new Supply Depot."
        # if action_name == "PROTOSSBUILD_PYLON":
        #     if self.supply_cap - self.supply_used >= 8:
        #         return False, "There is still enough supply count, no need to build new Pylon."
        # if action_name == "LARVATRAIN_OVERLORD":
        #     if self.supply_cap - self.supply_used >= 8:
        #         return False, "There is still enough supply count, no need to build new Overlord."

        ### resource check
        cost = self.calculate_cost(AbilityId[action_name]) * len(action["units"])
        supply_cost = 0
        if self.minerals < cost.minerals:
            return False, f"Minerals is not enough for action {action_name}"
        if self.vespene < cost.vespene:
            return False, f"Vespene is not enough for action {action_name}"
        try:
            supply_cost = self.calculate_supply_cost(AbilityId[action_name]) * len(action["units"])
            if self.supply_left < supply_cost:
                return False, f"Supply is not enough for action {action_name}"
        except KeyError:
            pass

        ### Protoss Pylon check
        if self.config.own_race == "Protoss":
            if action_name == "PROTOSSBUILD_PYLON":
                pylons = self.units(UnitTypeId.PYLON)
                close_pylons = pylons.closer_than(5, action["target_position"])

                if close_pylons:
                    close_pylons_pos = [f"({int(p.position.x)}, {int(p.position.y)})" for p in close_pylons]
                    return False, f"Too close to other Pylons at positions: " + ", ".join(close_pylons_pos)

        return True, [cost.minerals, cost.vespene, supply_cost]

    def get_building_units(self):
        building_units = []
        for unit in self.units:
            if unit.build_progress < 1.0:
                building_units.append(unit)
        return [unit.name for unit in building_units]

    ################ tag id mapping
    def tag_to_id(self, tag: int):
        if tag not in self._tag_to_id:
            # Keep legacy IDs until full; mappings are never removed during a game.
            if len(self._id_to_tag) >= 1000:
                next_id = len(self._id_to_tag)
            else:
                next_id = tag % 1000
                while next_id in self._id_to_tag:
                    next_id = (next_id + 1) % 1000
            self._tag_to_id[tag] = next_id
            self._id_to_tag[next_id] = tag
        return self._tag_to_id[tag]

    def id_to_tag(self, _id: int):
        return self._id_to_tag[_id]

    def get_unit_by_tag(self, tag: int):
        unit = self.all_units.find_by_tag(tag)
        return unit

    def get_unit_by_id(self, _id: int):
        tag = self.id_to_tag(_id)
        return self.get_unit_by_tag(tag)

    # =========================================================
    # [新增方法 1] 根据 SCV 位置寻找最佳建造基地
    # 逻辑：
    # 1. 找到所有已就绪的基地。
    # 2. 按距离当前干活的 SCV 的远近进行排序。
    # 3. 依次检查基地附近的拥挤程度。
    # 4. 优先返回：距离 SCV 最近 且 不拥挤的基地。
    # 5. 兜底方案：如果都拥挤，返回相对最不拥挤的那个。
    # =========================================================
    def _get_target_base_for_building(self, scv_position: Point2) -> Point2:
        if not self.townhalls.ready:
            return self.start_location

        # 1. 按距离 SCV 的远近排序基地
        sorted_bases = self.townhalls.ready.sorted_by_distance_to(scv_position)

        # 2. 定义拥挤阈值 (例如：15格范围内超过 8 个建筑算拥挤，可根据实际体验调整)
        CROWDED_THRESHOLD = 8

        best_fallback_base = None
        min_structure_count = float('inf')

        # 3. 依次检查 (从近到远)
        for base in sorted_bases:
            # 统计该基地周围 15 格距离内的建筑数量
            # 注意：排除不需要占地太多的单位或特定逻辑，这里粗略统计所有 structures
            count = self.structures.closer_than(15, base.position).amount

            # 如果当前基地不拥挤，它就是我们要找的“最近且可用”的基地
            if count < CROWDED_THRESHOLD:
                return base.position

            # 记录最空的基地作为备选 (如果所有基地都满了，就去最空的那个，避免卡死)
            if count < min_structure_count:
                min_structure_count = count
                best_fallback_base = base

        # 4. 兜底：如果所有基地都超过了阈值，返回最空的那个；如果还没找到，返回最近的
        if best_fallback_base:
            return best_fallback_base.position

        return sorted_bases[0].position

    # =========================================================
    # [新增方法 2] 寻找距离 SCV 最近且未被采集的气矿
    # =========================================================
    def _find_available_geyser(self, scv_position: Point2):
        if not self.townhalls.ready:
            return None

        valid_geysers = []

        # 1. 收集所有基地附近可用的气矿
        for base in self.townhalls.ready:
            # 查找基地附近的瓦斯泉
            geysers = self.vespene_geyser.closer_than(15, base.position)
            for geyser in geysers:
                # 检查该气矿上是否已经有建筑 (Refinery/Assimilator/Extractor)
                has_building = self.structures.filter(
                    lambda s: s.distance_to(geyser) < 1.0 and
                    s.type_id in {
                        UnitTypeId.REFINERY, UnitTypeId.REFINERYRICH,
                        UnitTypeId.ASSIMILATOR, UnitTypeId.ASSIMILATORRICH,
                        UnitTypeId.EXTRACTOR, UnitTypeId.EXTRACTORRICH
                    }
                ).exists

                # 只有上面没建筑的才算有效
                if not has_building:
                    valid_geysers.append(geyser)

        # 2. 如果有可用气矿，返回距离 SCV 最近的一个
        if valid_geysers:
            return min(valid_geysers, key=lambda g: g.distance_to(scv_position))

        return None

    # =========================================================
    # [辅助方法] 计算点 p 到线段 ab 的垂直距离 (用于矿路检查)
    # =========================================================
    def _distance_to_segment(self, p: Point2, a: Point2, b: Point2) -> float:
        d_x = b.x - a.x
        d_y = b.y - a.y
        if d_x == 0 and d_y == 0:
            return p.distance_to(a)

        t = ((p.x - a.x) * d_x + (p.y - a.y) * d_y) / (d_x * d_x + d_y * d_y)
        t = max(0, min(1, t))

        closest_x = a.x + t * d_x
        closest_y = a.y + t * d_y
        return p.distance_to(Point2((closest_x, closest_y)))

    # =========================================================
    # [辅助方法] 检查位置是否阻挡了基地到资源的采集路径
    # =========================================================
    def _is_blocking_mining_path(self, pos: Point2, buffer: float = 3.0) -> bool:
        nearby_bases = self.townhalls.ready.closer_than(15, pos)
        if not nearby_bases.exists:
            return False

        for base in nearby_bases:
            # 检查 基地 <-> 矿
            for mineral in self.mineral_field.closer_than(10, base):
                if self._distance_to_segment(pos, base.position, mineral.position) < buffer:
                    return True
            # 检查 基地 <-> 气
            for geyser in self.vespene_geyser.closer_than(10, base):
                if self._distance_to_segment(pos, base.position, geyser.position) < buffer:
                    return True
            for refinery in self.gas_buildings.closer_than(10, base):
                if self._distance_to_segment(pos, base.position, refinery.position) < buffer:
                    return True
        return False

    # =========================================================
    # [选址核心逻辑] find_placement
    # =========================================================
    # async def find_placement(
    #     self,
    #     building,
    #     near: Point2,
    #     max_distance: int = 20,
    #     random_alternative: bool = True,
    #     placement_step: int = 2,
    #     addon_place: bool = False,
    # ):
    #     assert isinstance(building, (AbilityId, UnitTypeId))
    #     assert isinstance(near, Point2), f"{near} is no Point2 object"

    #     if isinstance(building, UnitTypeId):
    #         building_ability = self.game_data.units[building.value].creation_ability.id
    #     else:
    #         building_ability = building

    #     addon_check_ability = None
    #     if addon_place:
    #         # 使用补给站的建造指令来模拟检查挂件位置是否可建
    #         addon_check_ability = self.game_data.units[UnitTypeId.SUPPLYDEPOT.value].creation_ability.id

    #     # 1. 检查原点
    #     if await self.can_place_single(building_ability, near):
    #         if not self._is_blocking_mining_path(near):
    #             if not addon_place or await self.can_place_single(addon_check_ability, near.offset((2.5, -0.5))):
    #                 return near

    #     if max_distance == 0:
    #         return None

    #     # 2. 螺旋搜索
    #     for distance in range(placement_step, max_distance, placement_step):
    #         possible_positions = [
    #             Point2(p).offset(near).to2
    #             for p in (
    #                 [(dx, -distance) for dx in range(-distance, distance + 1, placement_step)]
    #                 + [(dx, distance) for dx in range(-distance, distance + 1, placement_step)]
    #                 + [(-distance, dy) for dy in range(-distance, distance + 1, placement_step)]
    #                 + [(distance, dy) for dy in range(-distance, distance + 1, placement_step)]
    #             )
    #         ]

    #         # 3. 引擎查询 + 矿路过滤
    #         res = await self.client._query_building_placement_fast(building_ability, possible_positions)
    #         possible = [p for r, p in zip(res, possible_positions) if r]
    #         possible = [p for p in possible if not self._is_blocking_mining_path(p)]

    #         if not possible:
    #             continue

    #         # 4. 挂件位置检查
    #         if addon_place:
    #             res = await self.client._query_building_placement_fast(
    #                 addon_check_ability,
    #                 [p.offset((2.5, -0.5)) for p in possible],
    #             )
    #             possible = [p for r, p in zip(res, possible) if r]

    #         if not possible:
    #             continue

    #         if random_alternative:
    #             return random.choice(possible)
    #         return min(possible, key=lambda p: p.distance_to_point2(near))

    #     return None
    async def find_placement(
        self,
        building,
        near: Point2,
        max_distance: int = 20,
        random_alternative: bool = True,
        placement_step: int = 2,
        addon_place: bool = False,
        avoid_production_addons: bool = True,
    ):
        assert isinstance(building, (AbilityId, UnitTypeId))
        assert isinstance(near, Point2), f"{near} is no Point2 object"

        if isinstance(building, UnitTypeId):
            building_ability = self.game_data.units[building.value].creation_ability.id
        else:
            building_ability = building

        addon_check_ability = None
        if addon_place:
            addon_check_ability = self.game_data.units[UnitTypeId.SUPPLYDEPOT.value].creation_ability.id

        # =========================================================
        # [逻辑 1]：预留现有及已下令待开工生产建筑的挂件空间
        # =========================================================
        reserved_addon_locations = []
        if avoid_production_addons:
            prod_types = {UnitTypeId.BARRACKS, UnitTypeId.FACTORY, UnitTypeId.STARPORT}
            for structure in self.structures.of_type(prod_types):
                # 挂件的标准偏移量：x + 2.5, y - 0.5
                reserved_addon_locations.append(structure.position.offset((2.5, -0.5)))
            # SCV en route: the order exists before the building enters structures.
            production_builds = {
                AbilityId.TERRANBUILD_BARRACKS, AbilityId.TERRANBUILD_FACTORY,
                AbilityId.TERRANBUILD_STARPORT,
            }
            for worker in self.workers:
                for order in worker.orders:
                    if order.ability.exact_id in production_builds and isinstance(order.target, Point2):
                        reserved_addon_locations.append(order.target.offset((2.5, -0.5)))
        # =========================================================

        # 1. 检查原点
        if await self.can_place_single(building_ability, near):
            if not self._is_blocking_mining_path(near):
                is_blocking_addon = False
                if avoid_production_addons:
                    for reserved in reserved_addon_locations:
                        # [修改点]：这里也同步增加到 4.5，确保原点也不会太靠近挂件区
                        if near.distance_to(reserved) < 4.5:
                            is_blocking_addon = True
                            break

                if not is_blocking_addon:
                    if not addon_place or await self.can_place_single(addon_check_ability, near.offset((2.5, -0.5))):
                        return near

        if max_distance == 0:
            return None

        # 2. 螺旋搜索
        for distance in range(placement_step, max_distance, placement_step):
            possible_positions = [
                Point2(p).offset(near).to2
                for p in (
                    [(dx, -distance) for dx in range(-distance, distance + 1, placement_step)]
                    + [(dx, distance) for dx in range(-distance, distance + 1, placement_step)]
                    + [(-distance, dy) for dy in range(-distance, distance + 1, placement_step)]
                    + [(distance, dy) for dy in range(-distance, distance + 1, placement_step)]
                )
            ]

            # =========================================================
            # [逻辑 2]：过滤掉阻挡挂件的点
            # =========================================================
            if avoid_production_addons and reserved_addon_locations:
                # [核心修改点]：
                # 原值 3 -> 新值 4.5 (增加了 50% 的安全距离)
                # 这不仅仅是留出挂件位置，还留出了大概 2.5 格的缝隙，大型单位可以通过
                possible_positions = [
                    p for p in possible_positions
                    if all(p.distance_to(r) >= 4.5 for r in reserved_addon_locations)
                ]

            if not possible_positions:
                continue
            # =========================================================

            # 3. 引擎查询 + 矿路过滤
            res = await self.client._query_building_placement_fast(building_ability, possible_positions)
            possible = [p for r, p in zip(res, possible_positions) if r]
            possible = [p for p in possible if not self._is_blocking_mining_path(p)]

            if not possible:
                continue

            # 4. 挂件位置检查
            if addon_place:
                res = await self.client._query_building_placement_fast(
                    addon_check_ability,
                    [p.offset((2.5, -0.5)) for p in possible],
                )
                possible = [p for r, p in zip(res, possible) if r]

            if not possible:
                continue

            if random_alternative:
                return random.choice(possible)
            return min(possible, key=lambda p: p.distance_to_point2(near))

        return None

    # =========================================================
    # [核心执行逻辑] run_actions
    # =========================================================
    async def run_actions(self, actions):
        for action in actions:
            # --- 1. 安全过滤 ---
            action_name = action.get("action", "")
            action_name_upper = action_name.upper()

            # 屏蔽攻击指令；MOVE 需要保留给 BO 预走位逻辑使用。
            if "ATTACK" in action_name_upper:
                try:
                    self.logging("blocked_action", f"Blocked command: {action_name}", save_trace=True, print_log=True)
                    action["is_valid"] = False
                    action["error"] = "Attack command blocked by filter."
                except Exception as e:
                    self.logging("block_filter_error", str(e), level="error", print_log=True)
                continue

            try:
                # --- 2. 基础检查 ---
                action_check_result, action_check_msg = self.check_action(action)
                if not action_check_result:
                    action["is_valid"] = False
                    action["error"] = action_check_msg
                else:
                    # --- 3. 执行单元循环 ---
                    for unit_id in action["units"]:
                        ability = AbilityId[action["action"]]
                        target = None
                        curr_unit = self.get_unit_by_id(unit_id)

                        # 确保单位存在
                        if not curr_unit:
                            continue

                        # --- 4. 智能目标选择与选址 (核心修改) ---

                        # 判断是否为建造指令 (排除挂件升级，因为挂件是直接点建筑升级，不需要选址)
                        is_build_cmd = "BUILD" in action_name and not ("BUILD_TECHLAB" in action_name or "BUILD_REACTOR" in action_name)

                        # A. 基地类建筑：遵循全局扩张逻辑
                        if action_name in ["TERRANBUILD_COMMANDCENTER", "PROTOSSBUILD_NEXUS", "ZERGBUILD_HATCHERY"]:
                            target = await self.get_next_expansion()
                            assert target is not None, "No available expansion location found."

                        # B. 气矿类建筑：寻找离 SCV 最近的可用气矿
                        elif action_name in ["TERRANBUILD_REFINERY", "PROTOSSBUILD_ASSIMILATOR", "ZERGBUILD_EXTRACTOR"]:
                            target = self._find_available_geyser(curr_unit.position)
                            assert target is not None, "No available vespene geyser found."

                        # C. 普通生产/科技建筑：基于 SCV 位置的负载均衡选址
                        elif is_build_cmd:
                            # 1. 找到目标基地 (离 SCV 最近且不拥挤)
                            base_pos = self._get_target_base_for_building(curr_unit.position)

                            # 2. 检查是否需要为人族挂件预留空间
                            need_addon = ability in [AbilityId.TERRANBUILD_BARRACKS, AbilityId.TERRANBUILD_FACTORY, AbilityId.TERRANBUILD_STARPORT]

                            # 3. 在目标基地附近寻找空位
                            target = await self.find_placement(
                                ability,
                                base_pos,
                                max_distance=30,
                                random_alternative=True,
                                addon_place=need_addon
                            )
                            assert target is not None, f"No valid placement found for {action_name} near base {base_pos}."

                        # D. 非建造类指令 (训练、研究、修理等)
                        else:
                            if "target_unit" in action:
                                target = self.get_unit_by_id(action["target_unit"])
                                assert target is not None, f"Unit with id {action['target_unit']} not found"
                            elif "target_position" in action:
                                target = Point2(action["target_position"])
                                assert target is not None, f"Invalid target position: {action['target_position']}"

                        # --- 5. 下达指令 ---
                        trace = getattr(self, "command_trace", None)
                        if trace is not None:
                            trace.active_bo = trace.bo_origin(
                                self.current_bo_index, self.planner_queue_active,
                                self.bo_data[self.current_bo_index]['action'],
                            )
                        try:
                            run_state = curr_unit(ability=ability, target=target)
                        finally:
                            if trace is not None:
                                trace.active_bo = None

                        # --- 6. 更新资源统计 ---
                        cost = self.calculate_cost(ability)
                        self.resource_cost += cost.minerals + cost.vespene

            except Exception as e:
                action["is_valid"] = False
                action["error"] = str(e)

            if not action.get("is_valid", True):
                print(
                    f"SC2 action failed: {action.get('action', 'unknown')}: "
                    f"{action.get('error', 'unknown error')}"
                )

        # --- 7. 更新历史动作 ---
        valid_actions = [action for action in actions if action.get("is_valid", True)]
        valid_actions_dump = [json.dumps(action, ensure_ascii=False) for action in valid_actions]
        self.last_action.extend(valid_actions_dump)

    ############### run actions shy 屏蔽 all attack command

    ################ obs to text
    async def obs_to_text(self):
        obs = {}
        obs["Round state"] = self.round_state_to_text()
        obs["Own units"] = await self.units_to_text(self.units)
        obs["Unit abilities"] = await self.abilities_to_text(self.units)
        obs["Own structures"] = await self.structures_to_text(self.structures)
        obs["Structure abilities"] = await self.abilities_to_text(self.structures)
        obs["Visible enemy units"] = await self.units_to_text(self.enemy_units)
        obs["Visible enemy structures"] = await self.structures_to_text(self.enemy_structures)
        obs["Action history"] = self.action_history_to_text()
        # --- 获取下一个可用扩张点（分矿）的位置 ---
        expansion_text = ""
        # next_expansion_loc = await self.get_next_expansion()
        # expansion_text = f"Next available expansion location: ({int(next_expansion_loc.x)}, {int(next_expansion_loc.y)})\n"
        try:
            # get_next_expansion() 会自动找到距离出生点最近的、未被占领的矿区
            next_expansion_loc = await self.get_next_expansion()

            if next_expansion_loc:
                # 格式化坐标为整数
                expansion_text = f"Next available expansion location: ({int(next_expansion_loc.x)}, {int(next_expansion_loc.y)})\n"
            else:
                # 明确告诉 LLM 没有可用分矿了
                expansion_text = "Next available expansion location: None\n"
        except Exception as e:
            # 增加一个保护，以防 get_next_expansion 失败
            self.logger.error(f"Error getting next expansion: {e}")
            expansion_text = "Next available expansion location: Error\n"

        # --- 组合所有地图信息 ---
        obs["Map information"] = (
            self.miner_to_text() + "\n" +
            self.gas_to_text()
#             + "\n" +
#             f"""Only construct a new base if the Harvester status shows: no more harvesters accepted, otherwise, focus on developing the existing base and its related structures.
# no more harvesters accepted condition will be show like this in # Own units:

# [xxx]CommandCenter
# Position: (xxx, xxx)
# Health: xxx/xxx
# Harvesters: 17/16 (no more harvesters accepted, surplus 1)

# [xxx]OrbitalCommand
# Position: (xxx, xxx)
# Health: xxx/xxx
# Harvesters: 26/6 (no more harvesters accepted, surplus 20)

# Command you should do, if this condition is appearing:
# TERRANBUILD_COMMANDCENTER(target: Point): Build a Command Center, the Terran base. Cost: 400 minerals.
# {expansion_text}
#             """
        )
        obs["Ability description"] = self.get_ability_desc(obs["Unit abilities"] + obs["Structure abilities"])
        obs_text = "\n\n".join([f"# {key}\n{value}" for key, value in obs.items()])

        self.logging("obs", obs, save_trace=True, print_log=False)
        if self.enable_logging:
            self.logging("obs_text", obs_text, save_file=True, print_log=False)
        return obs_text

    async def obs_to_text_dict(self):
        obs = {}
        obs["Round state"] = self.round_state_to_text()
        obs["Own units"] = await self.units_to_text(self.units)
        obs["Unit abilities"] = await self.abilities_to_text(self.units)
        obs["Own structures"] = await self.structures_to_text(self.structures)
        obs["Structure abilities"] = await self.abilities_to_text(self.structures)
        obs["Visible enemy units"] = await self.units_to_text(self.enemy_units)
        obs["Visible enemy structures"] = await self.structures_to_text(self.enemy_structures)
        obs["Action history"] = self.action_history_to_text()
        # --- 获取下一个可用扩张点（分矿）的位置 ---
        expansion_text = ""
        # next_expansion_loc = await self.get_next_expansion()
        # expansion_text = f"Next available expansion location: ({int(next_expansion_loc.x)}, {int(next_expansion_loc.y)})\n"
        try:
            # get_next_expansion() 会自动找到距离出生点最近的、未被占领的矿区
            next_expansion_loc = await self.get_next_expansion()

            if next_expansion_loc:
                # 格式化坐标为整数
                expansion_text = f"Next available expansion location: ({int(next_expansion_loc.x)}, {int(next_expansion_loc.y)})\n"
            else:
                # 明确告诉 LLM 没有可用分矿了
                expansion_text = "Next available expansion location: None\n"
        except Exception as e:
            # 增加一个保护，以防 get_next_expansion 失败
            self.logger.error(f"Error getting next expansion: {e}")
            expansion_text = "Next available expansion location: Error\n"

        # --- 组合所有地图信息 ---
        obs["Map information"] = (
            self.miner_to_text() + "\n" +
            self.gas_to_text()
            # + "\n" +
#             f"""Only construct a new base if the Harvester status shows: no more harvesters accepted, otherwise, focus on developing the existing base and its related structures.
# no more harvesters accepted condition will be show like this in # Own units:

# [xxx]CommandCenter
# Position: (xxx, xxx)
# Health: xxx/xxx
# Harvesters: 17/16 (no more harvesters accepted, surplus 1)

# [xxx]OrbitalCommand
# Position: (xxx, xxx)
# Health: xxx/xxx
# Harvesters: 26/6 (no more harvesters accepted, surplus 20)

# Command you should do, if this condition is appearing:
# TERRANBUILD_COMMANDCENTER(target: Point): Build a Command Center, the Terran base. Cost: 400 minerals.
# {expansion_text}
#             """
        )
        obs["Ability description"] = self.get_ability_desc(obs["Unit abilities"] + obs["Structure abilities"])
        obs_text = "\n\n".join([f"# {key}\n{value}" for key, value in obs.items()])

        self.logging("obs", obs, save_trace=True, print_log=False)
        if self.enable_logging:
            self.logging("obs_text", obs_text, save_file=True, print_log=False)
        return obs

    def get_ability_desc(self, text: str):
        desc = []
        for action in TerranAbility:
            if TerranAbility[action].get("enabled", False) and action in text:
                action_desc = TerranAbility[action]["description"]
                action_keys = TerranAbility[action]["target"]
                desc.append(f"{action}(target: {action_keys}): {action_desc}")
                try:
                    cost = self.units[0]._bot_object.game_data.calculate_ability_cost(AbilityId[action])
                    if cost.minerals and cost.vespene:
                        desc[-1] += f" Cost: {cost.minerals} minerals, {cost.vespene} vespene."
                    elif cost.vespene:
                        desc[-1] += f" Cost: {cost.vespene} vespene."
                    elif cost.minerals:
                        desc[-1] += f" Cost: {cost.minerals} minerals."
                except Exception as e:
                    pass
        return "\n".join(desc)

    def round_state_to_text(self):
        text = ""
        text += "Time: {}\n".format(self.time_formatted)
        text += "Race: {}\n".format(self.race.name)
        text += "Minerals: {}\n".format(self.minerals)
        text += "Vespene: {}\n".format(self.vespene)
        text += "Supply army: {}\n".format(self.supply_army)
        text += "Supply workers: {}\n".format(self.supply_workers)
        text += "Supply unused: {}\n".format(self.supply_cap - self.supply_used)
        text += "Map size: {}".format(self.map_name)

        return text.strip()

    def action_history_to_text(self):
        if len(self.last_action) == 0:
            return "[Empty]"
        return "\n".join(self.last_action[-10:])

    async def units_to_text(self, units: Units):
        if len(units) == 0:
            return "[Empty]"

        units_text = []

        other_units = units
        if units.first.is_mine:
            for mining_type in self.miner_units:
                mining_judge = lambda unit: unit.name == mining_type and not (unit.is_constructing_scv or unit.is_repairing or unit.is_attacking)
                mining_units = units.filter(mining_judge)
                if len(mining_units) > 0:
                    mining_ids = [self.tag_to_id(unit.tag) for unit in mining_units]
                    mining_ids = ", ".join(map(str, mining_ids))
                    mining_text = f"[{mining_ids}]{mining_type}\nState: collecting resources automatically"
                    units_text.append(mining_text)
                other_units = [unit for unit in other_units if not mining_judge(unit)]

                attacking_judge = lambda unit: unit.name == mining_type and unit.is_attacking
                attacking_units = units.filter(attacking_judge)
                if len(attacking_units) > 0:
                    attacking_ids = [self.tag_to_id(unit.tag) for unit in attacking_units]
                    attacking_ids = ", ".join(map(str, attacking_ids))
                    attacking_text = f"[{attacking_ids}]{mining_type}\nState: attacking enemies automatically"
                    units_text.append(attacking_text)
                other_units = [unit for unit in other_units if not attacking_judge(unit)]

        distance_to_start = lambda unit: int((unit.position.x - self.start_location.x) ** 2 + (unit.position.y - self.start_location.y) ** 2) // 4
        other_units = sorted(other_units, key=lambda unit: (distance_to_start(unit), unit.name))
        units_text += [await self.unit_to_text(unit) for unit in other_units]
        units_text = "\n".join(units_text)
        return units_text

    async def units_to_text_wo_contructing(self, units: Units):
        if len(units) == 0:
            return "[Empty]"

        units_text = []

        other_units = units
        if units.first.is_mine:
            for mining_type in self.miner_units:
                # 1. 筛选正在采矿的 (逻辑保持不变)
                mining_judge = lambda unit: unit.name == mining_type and not (unit.is_constructing_scv or unit.is_repairing or unit.is_attacking)
                mining_units = units.filter(mining_judge)
                if len(mining_units) > 0:
                    mining_ids = [self.tag_to_id(unit.tag) for unit in mining_units]
                    mining_ids = ", ".join(map(str, mining_ids))
                    mining_text = f"[{mining_ids}]{mining_type}\nState: collecting resources automatically"
                    units_text.append(mining_text)
                # 从剩余列表中移除矿工
                other_units = [unit for unit in other_units if not mining_judge(unit)]

                # 2. 筛选正在攻击的工兵 (逻辑保持不变)
                attacking_judge = lambda unit: unit.name == mining_type and unit.is_attacking
                attacking_units = units.filter(attacking_judge)
                if len(attacking_units) > 0:
                    attacking_ids = [self.tag_to_id(unit.tag) for unit in attacking_units]
                    attacking_ids = ", ".join(map(str, attacking_ids))
                    attacking_text = f"[{attacking_ids}]{mining_type}\nState: attacking enemies automatically"
                    units_text.append(attacking_text)
                # 从剩余列表中移除攻击工兵
                other_units = [unit for unit in other_units if not attacking_judge(unit)]

        # --- 新增步骤：从剩余单位中剔除正在建造的 SCV ---
        # is_constructing_scv 是 python-sc2 中 SCV 特有的属性
        other_units = [unit for unit in other_units if not unit.is_constructing_scv]
        # ----------------------------------------------

        # 3. 处理剩下的单位 (排序并生成文本)
        distance_to_start = lambda unit: int((unit.position.x - self.start_location.x) ** 2 + (unit.position.y - self.start_location.y) ** 2) // 4
        other_units = sorted(other_units, key=lambda unit: (distance_to_start(unit), unit.name))

        # 将剩下的单位转为文本
        units_text += [await self.unit_to_text(unit) for unit in other_units]

        units_text = "\n".join(units_text)
        return units_text

    async def structures_to_text(self, structures: Units):
        if len(structures) == 0:
            return "[Empty]"
        structures = [s for s in structures]
        sorted_structures = []
        current_x, current_y = self.start_location.x, self.start_location.y
        while structures:
            closest_structure = min(structures, key=lambda s: math.sqrt((s.position.x - current_x) ** 2 + (s.position.y - current_y) ** 2))
            sorted_structures.append(closest_structure)
            structures.remove(closest_structure)
            current_x, current_y = closest_structure.position.x, closest_structure.position.y
        return "\n".join([await self.unit_to_text(structure) for structure in sorted_structures])

    async def unit_to_text(self, unit: Unit):
        text = ""

        upgrading = False
        new_unit = ""
        if unit.is_structure:
            morph_orders = unit.orders
            for unit_order in morph_orders:
                if "Morph " in unit_order.ability.friendly_name and unit.name == "CommandCenter":
                    new_unit = unit_order.ability.friendly_name[6:]
                    upgrading = True

        if unit.build_progress == 1.0:
            if upgrading and new_unit:
                text += f"[{self.tag_to_id(unit.tag)}]{unit.name}(morphing into {new_unit})\n"
            else:
                text += f"[{self.tag_to_id(unit.tag)}]{unit.name}\n"
        else:
            text += f"[{self.tag_to_id(unit.tag)}]{unit.name}(building {int(unit.build_progress * 100)}%)\n"
        text += f"Position: ({int(unit.position.x)}, {int(unit.position.y)})\n"

        if unit.build_progress == 1.0:
            if int(unit.health_max) and unit.build_progress == 1.0:
                text += f"Health: {int(unit.health)}/{int(unit.health_max)} ({int(unit.health_percentage * 100)}%)\n"
            if unit.shield_max > 0.0:
                text += f"Shield: {int(unit.shield)}/{int(unit.shield_max)}\n"
            if unit.energy_max > 0.0:
                text += f"Energy: {int(unit.energy)}/{int(unit.energy_max)}\n"
            if unit.name == 'Liberator':
                text += f"You can only attack air forces in this formation. Use MORPH_LIBERATORAGMODE first if you want to attach ground unit.\n"

            if unit.is_mine:
                states = self.unit_state_to_text(unit)
                if states:
                    text += f"State: {states}\n"
                if hasattr(unit, "cargo_max") and unit.cargo_max > 0.0:
                    if hasattr(unit, "has_cargo") and unit.has_cargo:
                        used = unit.cargo_used
                        max = unit.cargo_max
                        text += f"Cargo: {used}/{max}\n"
                    if hasattr(unit, "passengers") and len(unit.passengers) > 0:
                        passengers = []
                        for p in unit.passengers:
                            pas = f"[{self.tag_to_id(p.tag)}]{p.name}"
                            passengers.append(pas)
                        text += f"Passengers: ({', '.join(passengers)})\n"

                if unit.is_structure:
                    # Supply information
                    assigned = unit.assigned_harvesters
                    ideal = unit.ideal_harvesters
                    surplus = unit.surplus_harvesters
                    if ideal > 0:
                        if surplus > 0:
                            text += f"Harvesters: {assigned}/{ideal} (no more harvesters accepted, surplus {surplus})\n"
                        elif surplus == 0:
                            text += f"Harvesters: {assigned}/{ideal} (no more harvesters accepted)\n"
                        else:
                            text += f"Harvesters: {assigned}/{ideal}\n"

                # Production list
                production_list = []
                unit_orders = unit.orders
                for unit_order in unit_orders:
                    if "Train " in unit_order.ability.friendly_name:
                        production_list.append(unit_order.ability.friendly_name[6:])
                if production_list:
                    text += f"Production list: {', '.join(production_list)}\n"

                # Add-on
                if unit.has_add_on:
                    add_on_tag = unit.add_on_tag
                    if add_on_tag != 0:     # There truly has an add-on
                        text += f"This structure has an add-on: [{self.tag_to_id(add_on_tag)}]{self.get_unit_by_tag(add_on_tag).name}\n"

        return text.strip()

    async def abilities_to_text(self, units: Units):
        units = [unit for unit in units if unit.build_progress == 1.0]
        if not units:
            return "[Empty]"
        n_units = len(units)
        units_ability_ids = await self.get_available_abilities(units, ignore_resource_requirements=True)
        units_ability_names = [[ability_id.name for ability_id in units_ability_ids[i]] for i in range(n_units)]
        unit_hash_table = {}
        for i in range(n_units):
            unit = units[i]
            ability_names = units_ability_names[i]
            ability_names = [name for name in ability_names if name != "NULL_NULL"]
            unknown_abilities = [name for name in ability_names if name not in TerranAbility]
            if unknown_abilities:
                print(f"Unit {unit.name} has unknown abilities: {unknown_abilities}")
                ability_names = [name for name in ability_names if name in TerranAbility]
            if unit.name in self.miner_units:
                ability_names = [name for name in ability_names if name not in ["ATTACK_ATTACK"]]
            ability_names = [name for name in ability_names if TerranAbility[name].get("enabled", False)]
            self._id_to_abilities[self.tag_to_id(unit.tag)] = ability_names

            unit_hash = unit.name + "|" + ", ".join(ability_names)
            if unit_hash not in unit_hash_table:
                unit_hash_table[unit_hash] = []
            unit_hash_table[unit_hash].append(str(self.tag_to_id(unit.tag)))

        text = ""
        for unit_hash, ids in unit_hash_table.items():
            unit_name, abilities = unit_hash.split("|")
            ids = ", ".join(ids)
            if abilities:
                text += f"{unit_name}[{ids}]: {abilities}\n"

        text = text.strip()
        if not text:
            text = "[Empty]"
        return text

    def unit_state_to_text(self, unit: Unit):
        order_target = unit.order_target or ""
        order_target_name = ""
        if order_target:
            if isinstance(order_target, Point2):
                order_target = f"({int(order_target.x)}, {int(order_target.y)})"
            elif isinstance(order_target, int):
                target_unit = self.get_unit_by_tag(order_target)
                if target_unit:
                    order_target_name = self.get_unit_by_tag(order_target).name
                    order_target = self.tag_to_id(order_target)

        states = []
        if unit.is_moving:
            if order_target:
                states.append(f"moving to [{order_target}]{order_target_name}")
            else:
                states.append("moving")
        if unit.is_attacking:
            if order_target:
                states.append(f"attacking [{order_target}]{order_target_name}")
            else:
                states.append("attacking")
        if unit.is_repairing:
            if order_target:
                states.append(f"repairing [{order_target}]{order_target_name}")
            else:
                states.append("repairing")

        if unit.is_idle:
            states.append("idle")
        if unit.is_flying:
            states.append("flying")
        if unit.is_transforming:
            states.append("transforming")
        if unit.is_patrolling:
            states.append("patrolling")
        if unit.tag in self.tag_to_health and unit.health < self.tag_to_health[unit.tag]:
            states.append("under attack")

        if unit.is_constructing_scv:
            if order_target:
                states.append(f"constructing [{order_target}]{order_target_name}")
            else:
                states.append("constructing")

        return "|".join(states)

    def miner_to_text(self):
        center = self.start_location
        miners = []
        num_workers = len([unit for unit in self.units if unit.name in self.miner_units])
        cloest_miners = self.mineral_field.closest_n_units(center, 100)
        cloest_miners = [mineral for mineral in cloest_miners if mineral.mineral_contents > 0]
        cloest_miners = cloest_miners[: 2 * num_workers]
        for mineral in cloest_miners:
            miners.append(f"[{self.tag_to_id(mineral.tag)}]({int(mineral.position.x)}, {int(mineral.position.y)})")
        if len(miners) == 0:
            return "No mineral fields found"
        return "Closest mineral fields: " + ", ".join(miners)

    def gas_to_text(self):
        gases = []
        cloest_gases = self.vespene_geyser.closest_n_units(self.start_location, 100)
        cloest_gases = [gas for gas in cloest_gases if gas.vespene_contents > 0]
        cloest_gases = cloest_gases[:10]
        for gas in cloest_gases:
            # # if there is a structure on the position, ignore it
            # if self.structures.closer_than(0.5, gas.position):
            #     continue
            gases.append(f"[{self.tag_to_id(gas.tag)}]({int(gas.position.x)}, {int(gas.position.y)})")
        if len(gases) == 0:
            return "No vespene geysers found"
        return "Closest vespene geysers: " + ", ".join(gases)

    def rock_to_text(self):
        if not self.destructables:
            return "No destructable obstacles found"

        rocks = []
        all_rocks = [rock for rock in self.destructables if rock.health > 0]
        for rock in all_rocks:
            text = ""
            text += f"[{self.tag_to_id(rock.tag)}]{rock.name}\n"
            text += f"Position: ({int(rock.position.x)}, {int(rock.position.y)})\n"
            text += f"Health: {int(rock.health)}/{int(rock.health_max)} ({int(rock.health_percentage * 100)}%)\n"
            rocks.append(text)
        if len(rocks) == 0:
            return "No destructable obstacles found"
        return "Destructable obstacles:\n" + "\n".join(rocks)

#### shy ####

    def get_action_target_type(self, action_name: str) -> str:
            """
            [封装工具函数] 获取指定动作的目标类型分类。
            用于判断该 action 属于哪一类：
            - TargetType.NONE ("None"): 不需要目标
            - TargetType.POINT ("Point"): 需要 target_position
            - TargetType.UNIT ("Unit"): 需要 target_unit
            - TargetType.POINT_OR_UNIT ("PointOrUnit"): 需要其中之一
            """
            # TerranAbility 是 base_player.py 中的全局变量，存储了动作知识库
            if action_name not in TerranAbility:
                self.logger.warning(f"get_action_target_type: Unknown action '{action_name}'")
                return None

            if action_name == "CALLDOWNMULE_CALLDOWNMULE":
                return "UNIT"

            # 返回其 target 字段，例如 "Point" 或 "Unit"
            target_type = TerranAbility[action_name]["target"]
            if target_type == TargetType.NONE:
                return "None"
            elif target_type == TargetType.POINT:
                return "POINT"
            elif target_type == TargetType.UNIT:
                return "UNIT"
            elif target_type == TargetType.POINT_OR_UNIT:
                return "UNIT"
