from agents.base_agent import BaseAgent
from tools.format import extract_code
import json
import os
import re

BO_ABILITY_FALLBACK = {
    "scv": "COMMANDCENTERTRAIN_SCV",
    "supplydepot": "TERRANBUILD_SUPPLYDEPOT",
    "barracks": "TERRANBUILD_BARRACKS",
    "commandcenter": "TERRANBUILD_COMMANDCENTER",
    "marine": "BARRACKSTRAIN_MARINE",
    "orbitalcommand": "UPGRADETOORBITAL_ORBITALCOMMAND",
    "refinery": "TERRANBUILD_REFINERY",
    "mule": "CALLDOWNMULE_CALLDOWNMULE",
    "engineeringbay": "TERRANBUILD_ENGINEERINGBAY",
    "planetaryfortress": "UPGRADETOPLANETARYFORTRESS_PLANETARYFORTRESS",
    "factory": "TERRANBUILD_FACTORY",
    "techlabbarracks": "BUILD_TECHLAB_BARRACKS",
    "reactorbarracks": "BUILD_REACTOR_BARRACKS",
    "techlabfactory": "BUILD_TECHLAB_FACTORY",
    "reactorfactory": "BUILD_REACTOR_FACTORY",
    "techlabstarport": "BUILD_TECHLAB_STARPORT",
    "reactorstarport": "BUILD_REACTOR_STARPORT",
    "missileturret": "TERRANBUILD_MISSILETURRET",
    "starport": "TERRANBUILD_STARPORT",
    "siegetank": "FACTORYTRAIN_SIEGETANK",
    "liberator": "STARPORTTRAIN_LIBERATOR",
    "vikingfighter": "STARPORTTRAIN_VIKINGFIGHTER",
    "terraninfantryarmorlevel1": "ENGINEERINGBAYRESEARCH_TERRANINFANTRYARMORLEVEL1",
    "terraninfantryarmorlevel2": "ENGINEERINGBAYRESEARCH_TERRANINFANTRYARMORLEVEL2",
    "terraninfantryweaponslevel1": "ENGINEERINGBAYRESEARCH_TERRANINFANTRYWEAPONSLEVEL1",
    "terraninfantryweaponslevel2": "ENGINEERINGBAYRESEARCH_TERRANINFANTRYWEAPONSLEVEL2",
    "stimpack": "BARRACKSTECHLABRESEARCH_STIMPACK",
    "combatshield": "RESEARCH_COMBATSHIELD",
    "ghostacademy": "TERRANBUILD_GHOSTACADEMY",
    "ghost": "BARRACKSTRAIN_GHOST",
    "personalcloaking": "RESEARCH_PERSONALCLOAKING",
    "armory": "TERRANBUILD_ARMORY",
    "widowmine": "FACTORYTRAIN_WIDOWMINE",
    "medivac": "STARPORTTRAIN_MEDIVAC",
    "banshee": "STARPORTTRAIN_BANSHEE",
    "bansheecloakingfield": "RESEARCH_BANSHEECLOAKINGFIELD",
    "marauder": "BARRACKSTRAIN_MARAUDER",
    "concussiveshells": "RESEARCH_CONCUSSIVESHELLS",
    "hellion": "FACTORYTRAIN_HELLION",
    "bunker": "TERRANBUILD_BUNKER",
    "fusioncore": "TERRANBUILD_FUSIONCORE",
    "battlecruiser": "STARPORTTRAIN_BATTLECRUISER",
}

class MicroAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.think = []
        self.chat_history = []
        self._bo_ability_map = None

    def _local_fallback_enabled(self) -> bool:
        return os.getenv("SC2_LOCAL_EXECUTOR_FALLBACK", "0") == "1"

    def _load_bo_ability_map(self) -> dict:
        if self._bo_ability_map is not None:
            return self._bo_ability_map

        self._bo_ability_map = {}
        knowledge_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "knowledge", "data_BO2Action_1.json")
        )
        try:
            with open(knowledge_path, "r") as f:
                data = json.load(f)
            for item in data:
                bo_name = self._normalize_bo_key(item.get("bo_name", ""))
                action_name = str(item.get("action_name", "")).strip()
                if bo_name and action_name:
                    self._bo_ability_map[bo_name] = action_name
        except Exception as e:
            print(f"Local BO ability map unavailable: {e}")
        return self._bo_ability_map

    def _normalize_bo_key(self, bo_step: str) -> str:
        text = str(bo_step).strip().lower()
        text = re.sub(r"\([^)]*\)", "", text)
        text = re.sub(r"^\s*[-*]\s*", "", text)
        text = re.sub(r"\s*x\s*\d+\s*$", "", text)
        text = re.sub(r"^(build|train|research|morph|upgrade)\s+", "", text)
        return re.sub(r"[^a-z0-9]+", "", text)

    def _fallback_bo_to_ability(self, bo_step: str) -> list:
        bo_key = self._normalize_bo_key(bo_step)
        ability = self._load_bo_ability_map().get(bo_key) or BO_ABILITY_FALLBACK.get(bo_key)
        if ability:
            print(f"Local BO ability fallback: {bo_step} -> {ability}")
            return [ability]
        return []

    def _fallback_select_executor(self, available_units_text: str) -> list:
        match = re.search(r"\[(\d+)", str(available_units_text))
        if match:
            tag = int(match.group(1))
            print(f"Local executor fallback selected tag: {tag}")
            return [tag]
        return []

    def _fallback_target_unit(self, ability_name: str, map_info: str, own_units_text: str, own_structures_text: str):
        if ability_name == "TERRANBUILD_REFINERY":
            sources = [
                "\n".join(line for line in str(map_info).splitlines() if "vespene" in line.lower() or "geyser" in line.lower()),
                str(map_info),
            ]
        elif ability_name == "CALLDOWNMULE_CALLDOWNMULE":
            sources = [
                "\n".join(line for line in str(map_info).splitlines() if "mineral" in line.lower()),
                str(map_info),
            ]
        else:
            sources = [f"{own_units_text}\n{own_structures_text}"]
        for source in sources:
            match = re.search(r"\[(\d+)", source)
            if match:
                return int(match.group(1))
        return None

    async def bo_to_ability(self, bo_step: str, ability_descriptions: str) -> list:
        """
        Step 1: 将 Build Order (BO) 文本指令转换为唯一的 SC2 Ability Name (API ID)。
        """
        self.think = []
        self.chat_history = []

        if self._local_fallback_enabled():
            fallback = self._fallback_bo_to_ability(bo_step)
            if fallback:
                self.think.append(["Step 1: BO to Ability", json.dumps(fallback)])
                return fallback
            print(f"Local BO ability fallback unavailable; skip LLM for: {bo_step}")
            return []

        # print(ability_descriptions)

        # print(bo_step)

        prompt = f"""
You are a StarCraft II expert agent.
Your task is to translate a specific "Build Order" instruction into the exact, unique Ability Name (API ID) required to execute it.

### Build Order Instruction
"{bo_step}"

### Available Abilities & Descriptions
{ability_descriptions}

### Instructions
1. Analyze the BO instruction to understand the intent (e.g., build a supply depot, train a marine, research an upgrade).
2. Find the EXACT corresponding Ability Name from the provided list.
3. Return ONLY the Ability Name inside a JSON list.
4. No need to consider the quantity, just output the names of the corresponding abilities, only a list is needed.
5. If the BO requires you to train multiple units, ignore the quantity and only consider building one.
6. If there is no ability to execute the corresponding BO, return []. Do not generate an intruction by yourself.

**Output Format:**
```json
["ABILITY_NAME"]
```
""".strip()

        response, messages = self.llm_client.call(prompt=prompt, **self.generation_config)

        print("BO:")
        print(bo_step)

        print("ABILITY_NAME:")
        print(response)

        self.chat_history.append(messages)
        self.think.append(["Step 1: BO to Ability", response])

        try:
            ability_list = json.loads(extract_code(response))
            if isinstance(ability_list, list) and len(ability_list) > 0:
                return ability_list
        except Exception as e:
            print(f"Error parsing ability from LLM: {e}")

        if self._local_fallback_enabled():
            fallback = self._fallback_bo_to_ability(bo_step)
            if fallback:
                return fallback

        return []

    async def select_executor(self, ability_name: str, available_units_text: str, bo_step: str, future_bo_list: str = "None") -> list:
        """
        Step 2: 从候选列表中选择最佳执行单位 (Executor)。
        """
        if self._local_fallback_enabled():
            fallback = self._fallback_select_executor(available_units_text)
            if fallback:
                self.think.append(["Step 2: Select Executor", json.dumps(fallback)])
                return fallback
            print(f"Local executor fallback unavailable; skip LLM for ability={ability_name}")
            return []

        prompt = f"""
You are a StarCraft II professional player agent.
Your task is to select the BEST unit(s) `tag` to execute the ability: **{ability_name}**.

### 1. Current Context
**Current Build Order Instruction:** "{bo_step}"
**Future Build Order (Next 5 steps):**
{future_bo_list}

### 2. Candidate Executors (Available Units)
{available_units_text}

### 3. Decision Logic (Follow strictly in order)

#### PRIORITY A: Future Planning (Crucial!)
*Check the 'Future Build Order'. If a structure update is coming soon, PRESERVE the idle structure.*
- **Add-on/Upgrade Reservation:** If the future BO contains "TechLab", "Reactor", "OrbitalCommand", or "PlanetaryFortress" within the next 1-3 steps:
    - You MUST leave at least one corresponding structure (Barracks/Factory/Starport/CommandCenter) **IDLE**.
    - **Do NOT** select an idle structure for training units if it's the only one available for that upcoming upgrade.
    - Instead, queue the unit in a structure that is already busy (has a production list), or choose a structure with a Reactor.
    - *Example:* If future is "Build Tech Lab", do NOT train a Marine in the only idle Barracks. Train it in a busy one, or wait.

#### PRIORITY B: Production Efficiency (Reactors)
- If training basic units (Marine, Reaper, Hellion, Viking, Medivac), prioritize structures with **Reactors**.
- **Reason:** Reactors can train 2 units simultaneously.
- Look for: "This structure has an add-on: [xxx]Reactor".

#### PRIORITY C: Load Balancing & Saturation (For SCVs/Workers)
- **Saturation Rule:** Check `Harvesters: X/Y`.
    - **Best:** Select bases where `X < Y` (Unsaturated).
    - **Avoid:** Do NOT select bases where `X >= Y` (Saturated), unless all bases are saturated.
    - **Spread:** Do not queue multiple SCVs in one Command Center if another one is empty. Avoid "Production list: SCV, SCV, SCV" if possible.

#### PRIORITY D: Standard Selection
- For constructing buildings: Prefer idle workers or those mining minerals.
- For training units: Distribute the queue. Don't stack all units in one building while others are idle.

### 4. Output
- Return the `tag` of the selected unit in a JSON list.
- Select ONLY ONE tag.

**Output Format:**
```json
[xxx]
```
""".strip()
        # print(prompt)

        response, messages = self.llm_client.call(prompt=prompt, **self.generation_config)

        print("!!! available_units:")
        print(available_units_text)
        print("!!! future BO")
        print(future_bo_list)
        print("!!! selected one:")
        print(response)

        self.chat_history.append(messages)
        self.think.append(["Step 2: Select Executor", response])

        try:
            tags = json.loads(extract_code(response))
            if isinstance(tags, list) and len(tags) > 0:
                return tags
        except Exception as e:
            print(f"Error parsing executor tags from LLM: {e}")

        if self._local_fallback_enabled():
            fallback = self._fallback_select_executor(available_units_text)
            if fallback:
                return fallback

        return []

    def _create_target_unit_prompt(self, ability_name: str, context_text: str,map_info: str, own_units_text: str, own_structures_text: str) -> str:
        if ability_name in ("CALLDOWNMULE_CALLDOWNMULE", "TERRANBUILD_REFINERY"):
            return f"""
Select the Target Unit for ability: **{ability_name}**.

## Game Context (Map)
{map_info}

### Instructions
1. Identify the appropriate target tag for **{ability_name}**.
   - If `CALLDOWNMULE_CALLDOWNMULE`: Target a Mineral fields
   - If `BUILD_REFINERY`: Target a Vespene Geyser.
2. Output the single target tag as a JSON list containing one integer.

**Output Format:**
```json
[xxx]
```
""".strip()
        else:
            return f"""
Select the Target Unit for ability: **{ability_name}**.

## Game Context (Units/Structures)
### Units
{own_units_text}

### Structures
{own_structures_text}

### Instructions
1. Identify the appropriate target tag for **{ability_name}**.
   - If `REPAIR`: Target a damaged structure/unit.
2. Output the single target tag as a JSON list containing one integer.

**Output Format:**
```json
[xxx]
```
""".strip()

    async def get_final_action(self, ability_name: str, executor_tags: list, map_info: str, own_units_text: str, own_structures_text: str, target_type:str) -> list:
        """
        Step 3: 确定目标 (Target) 并生成最终的 Action 字典。
        """
        # target_type = self._determine_target_type(ability_name)

        final_action = {
            "action": ability_name,
            "units": executor_tags
        }

        # --- 填充目标信息 ---
        if target_type == "POINT":
            # 自动填充占位符，具体的找点逻辑会在 BasePlayer.run_actions 或 find_placement 中处理
            final_action["target_position"] = [1, 1]

        elif target_type == "UNIT":
            if self._local_fallback_enabled():
                target_tag = self._fallback_target_unit(
                    ability_name, map_info, own_units_text, own_structures_text
                )
                if target_tag is not None:
                    final_action["target_unit"] = target_tag
                    self.think.append(["Step 3: Target Unit", str(target_tag)])
                    print(f"Local target fallback selected tag: {target_tag}")
                    return [final_action]
                print(f"Local target fallback unavailable; skip LLM for ability={ability_name}")
                self.think.append(["Step 3: Target Unit", "Local fallback unavailable"])
                return [final_action]

            # 构建 Prompt 让 LLM 从上下文中选择目标
            context_text = f"--- Map Info ---\n{map_info}\n\n--- Own Assets ---{own_structures_text}"


            prompt_target = self._create_target_unit_prompt(ability_name, context_text,
                                                            map_info=map_info,
                                                            own_units_text= own_units_text,
                                                            own_structures_text=own_structures_text)

            # print(prompt_target)

            resp_target, msg_target = self.llm_client.call(prompt=prompt_target, **self.generation_config)

            print("available_units:")
            print(context_text)
            print("selected one:")
            print(resp_target)

            self.chat_history.append(msg_target)

            try:
                target_list = json.loads(extract_code(resp_target))
                if target_list and isinstance(target_list, list) and len(target_list) > 0:
                    final_action["target_unit"] = target_list[0]
                    self.think.append(["Step 3: Target Unit", str(target_list[0])])
                else:
                    self.think.append(["Step 3: Target Unit", "No target found by LLM"])
                    # 如果需要 Target 但没找到，动作可能会失败。这里返回不带 target_unit 的 action 供上层处理。
            except Exception as e:
                print(f"Error parsing target unit from LLM: {e}")
                self.think.append(["Step 3: Target Unit", f"Error: {e}"])

        return [final_action]

    async def decide_recovery_action(self, suggestion: str, recent_actions: list[str]) -> dict:
        """
        Step 4 (Optional): 当动作被验证器拒绝时，结合[报错建议]与[最近历史]判断下一步。

        Args:
            suggestion (str): validate_action_and_get_suggestion 返回的错误详情与建议。
            recent_actions (list[str]): 最近执行过的 10 个动作列表 (Action Names)。

        Returns:
            dict: 包含决策结果的字典。
        """
        if self._local_fallback_enabled():
            return {
                "decision": "WAIT",
                "reason": "Local executor fallback mode skips LLM recovery decisions",
            }

        # 格式化历史记录
        history_text = "\n".join([f"- {i+1}. {act}" for i, act in enumerate(recent_actions)])

        # [修改] 全新的 Prompt，覆盖了资源、前置建筑、挂件、人口、忙碌等所有场景
        prompt = f"""
You are a StarCraft II logic arbiter.
A planned action failed verification. Your goal is to decide whether to **WAIT** or **EXECUTE** a fix based on the specific error reason.

### The Verifier's Report (Analyze the "Reason" and "Suggestion" carefully)
{suggestion}

### Recent Executed Actions (Last 10)
{history_text}

### Logic Decision Tree
1. **CASE: Missing Producer or Structure** (Report mentions "production facility" or "prerequisite structure")
    - Identify the missing building name (e.g., Barracks, GhostAcademy).
    - Check History: Did we execute "Build<Name>" recently?
    - If YES -> **WAIT** (It is under construction).
    - If NO  -> **EXECUTE** "Build<Name>" (e.g., "BuildBarracks").

2. **CASE: Missing Tech Lab or Reactor** (Report mentions "requires a Tech Lab" or "requires a Reactor")
   - Check History: Did we execute an add-on build for the required facility recently?
   - If YES -> **WAIT**.
   - If NO -> **EXECUTE** the specific add-on command targeting the correct facility:
     - For Barracks: `BUILD_TECHLAB_BARRACKS` or `BUILD_REACTOR_BARRACKS`
     - For Factory: `BUILD_TECHLAB_FACTORY` or `BUILD_REACTOR_FACTORY`
     - For Starport: `BUILD_TECHLAB_STARPORT` or `BUILD_REACTOR_STARPORT`

3. **CASE: Missing Upgrade** (Report mentions "prerequisite upgrade")
    - **EXECUTE** the specific research command if known, otherwise **WAIT**.

4. **CASE: Busy / Flying / Unpowered / Resources** (Report mentions "busy", "flying", "unpowered", "Insufficient")
    - **WAIT**. (We cannot force units to work faster or spawn resources).

### Output Format
Return a JSON object wrapped in triple backticks:
```json
{{
    "decision": "WAIT" or "EXECUTE",
    "action": "The specific Ability Name to execute" (Required if decision is EXECUTE, e.g., 'BuildBarracks', 'BuildTechLab', 'BuildSupplyDepot'),
    "reason": "Brief explanation of the logic used"
}}```
""".strip()

        response, messages = self.llm_client.call(prompt=prompt, **self.generation_config)

        # 记录思考过程
        self.chat_history.append(messages)
        self.think.append(["Step 4: Recovery Decision", response])

        try:
            decision = json.loads(extract_code(response))
            return decision
        except Exception as e:
            print(f"Error parsing recovery decision: {e}")
            return {"decision": "WAIT", "reason": "Error parsing LLM response"}
