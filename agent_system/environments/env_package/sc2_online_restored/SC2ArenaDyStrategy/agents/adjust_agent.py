from __future__ import annotations
import json
import os
import re
from agents.base_agent import BaseAgent
from tools.format import extract_code
from tools.logger import setup_logger

# 获取 logger 实例
# logger = setup_logger("AdjustAgent", log_dir=None)

ATTACK_KEYWORDS = (
    "attacking instruction",
    "launch offense",
    "attack with",
    "attack enemy",
    "attack the enemy",
)

UNIT_NAMES = (
    "marine",
    "marauder",
    "scv",
    "reaper",
    "hellion",
    "siegetank",
    "siege tank",
    "medivac",
    "viking",
    "cyclone",
    "thor",
)

NEGATIVE_ATTACK_HINTS = (
    "no attack",
    "do not attack",
    "don't attack",
    "without attacking",
)


def local_executor_fallback_enabled() -> bool:
    return os.getenv("SC2_LOCAL_EXECUTOR_FALLBACK", "0") == "1"


def rule_based_attack_command(plan: str) -> dict | None:
    """
    Fast path for explicit attack BO strings.
    The planner often emits short commands like "Attack with available army";
    routing those through the LLM classifier can misclassify them as Other Task.
    """
    if not isinstance(plan, str):
        return None

    text = plan.strip()
    lowered = text.lower()
    if not text or any(hint in lowered for hint in NEGATIVE_ATTACK_HINTS):
        return None

    has_attack_keyword = any(keyword in lowered for keyword in ATTACK_KEYWORDS)
    if not has_attack_keyword and not lowered.startswith("attack "):
        return None

    positions = [
        [int(x), int(y)]
        for x, y in re.findall(r"\((\d+)\s*,\s*(\d+)\)", text)
    ]

    unit_count = 0
    unit_pattern = "|".join(re.escape(name) for name in sorted(UNIT_NAMES, key=len, reverse=True))
    for number, _unit_name in re.findall(
        rf"\b(\d+)\s+({unit_pattern})s?\b",
        text,
        flags=re.IGNORECASE,
    ):
        unit_count += int(number)

    use_all_available = any(
        word in lowered
        for word in (
            "available army",
            "available forces",
            "all available",
            "full army",
            "all army",
            "all forces",
        )
    )
    if unit_count <= 0 or use_all_available:
        unit_count = 200

    target_unit = ""
    target_match = re.search(r"targeting\s+([^.;\n]+)", text, flags=re.IGNORECASE)
    if target_match:
        raw_target = re.sub(r"\([^)]*\)", "", target_match.group(1)).strip(" ,")
        if raw_target and not raw_target.lower().startswith("enemy unit at"):
            target_unit = raw_target

    return {
        "unit_count": unit_count,
        "all_available": use_all_available,
        "target_position": positions,
        "target_unit": target_unit,
    }

# --- 1. 第一阶段: 基础分类 (Plan -> Task Type) ---

def build_plan_prompt(plan: str) -> str:
    """
    构建用于 LLM 分类的英文提示（针对单个 plan，要求 JSON 列表输出）。
    (此函数基于您的需求描述)
    """

    # 英文提示，要求 LLM 严格返回 *包含单个元素* 的 JSON 列表
    prompt = f"""
Here is an instruction from StarCraft:
"{plan}"

Please classify this instruction into one of the following categories: ['Attack Task', 'Empty Task', 'Other Task'].

- 'Attack Task': Assigning our units to attack an enemy unit or location.
- 'Empty Task': No action is required; do nothing (e.g., "Do nothing and just wait").
- 'Other Task': Any task that is not an 'Attack Task' or 'Empty Task' (e.g., training units, building, researching, using non-attack abilities).

Please output *only* a valid JSON list containing the *single* corresponding task category.

Example Input 1:
"Train one SCV from the Command Center (4657)."

Example Output 1:
```json
[
    "Other Task"
]
```

Example Input 2:
"Use the Ghost at unit ID 4945 to snipe the enemy Ghost at unit ID 4961."

Example Output 2:
```json
[
    "Attack Task"
]
```

Example Input 3:
"Do nothing and just wait"

Example Output 3:
```json
[
    "Empty Task"
]
```
"""
    return prompt

def extract_json_list(text: str) -> list | None:
    """
    从 LLM 的原始文本输出中提取 JSON 列表。
    [修改] 严格依赖 `extract_code` (来自 tools.format) 提取 ```json ... ``` 块。
    """
    try:
        # 1. 严格使用 extract_code 提取
        code_block = extract_code(text)

        if not code_block:
             # 如果 extract_code 没有返回任何内容
             print(f"No code block (e.g., ```json ... ```) found in LLM output: '{text}'")
             return None

        # 2. 解析提取到的 code_block
        parsed_json = json.loads(code_block)

        # 3. 验证类型
        if isinstance(parsed_json, list):
            return parsed_json
        else:
            print(f"Parsed JSON from code block is not a list. Type: {type(parsed_json)}. Text: '{text}'")
            return None

    except json.JSONDecodeError as e:
        print(f"Could not decode extracted JSON list from code block: {e}. Code block: '{code_block}'. Raw Text: '{text}'")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during JSON list extraction: {e}. Text: '{text}'")
        return None

# --- 2. 第二阶段: 攻击分类 (Attack Plan -> Standard/Special) ---

def build_attack_prompt(command: str) -> str:
    """
    构建用于第二阶段攻击分类的 Prompt。
    (此函数基于您的需求描述)
    """
    return f"""Here are the definitions:
1.  **Standard Attack Command**: A basic instruction for a unit or units to attack a target.
2.  **Special Ability/Mode Switch Command**: Any other command that involves activating a specific, named ability or changing the unit's operational mode.

**Notice**: Standard Attack Command basically start with phrase "Attacking Instruction", if you see the phrase "Attacking Instruction", categorize it as a Standard Attack Command.

Your task is to analyze the command provided below.

-   If the command is a 'Special Ability/Mode Switch Command', you MUST respond with the exact JSON string: `[]`
-   If the command is a 'Standard Attack Command', you MUST respond with a JSON object with the following English keys and specific format:
    -   `"unit_count"`: (Type: `int`) The number of units being commanded. For singular nouns like 'the Marine' or 'the Cyclone', this is always `1`. If the Standard Attack Command does not specify the number of units, and terms such as 'full army' appear, representing all forces, please directly set this 'unit_count' to 200.
    -   `"target_position"`: (Type: `list`) A list of two integers `[x, y]` representing the **target's** coordinates.
    -   `"target_unit"`: (Type: `str`) A string containing **only the name** of the target unit (e.g., 'Barracks', 'Oracle'). Do not include descriptors like 'the enemy' or 'under construction'.
-   If the command only contains a single word like 'marine', it means training a marine instead of launch an attack with marine.

Here are some examples:

**Example 1 (Special Ability):**
Command: "Use Stimpack on Marine unit 4833 to increase damage and movement speed."
Your Output:
```json
[]
```

**Example 2 (Standard Attack):**
Command: "Launch OFFENSE with 3 Marine targeting Enemy Unit at (117, 43) and Enemy Unit at (35, 33)"
Your Output:
```json
{{"unit_count": 3, "target_position": [[117, 43],[35,33]], "target_unit": ""}}
```

**Example 3 (Standard Attack):**
Command: "Launch OFFENSE with 3 Marauder, 5 Marine, 1 SCV targeting SupplyDepotLowered"
Your Output:
```json
{{"unit_count": 9, "target_position": [], "target_unit": "SupplyDepotLowered"}}
```

**Example 4 (Standard Attack):**
Command: "Launch OFFENSE with 5 Marauder, 1 Marine, targeting OrbitalCommand"
Your Output:
```json
{{"unit_count": 6, "target_position": [], "target_unit": "OrbitalCommand"}}
```

**Example 5 (Other/Speical):**
Command: "marine"
Your Output:
```json
[]
```

---

Now, analyze this command:

Command: "{command}"
Your Output:
"""

def extract_attack_json_obj(text: str) -> dict | list | None:
    """
    从 LLM 的原始文本输出中提取 JSON 对象或列表。
    这是为第二阶段（攻击分类）定制的。
    [修改] 严格依赖 `extract_code` (来自 tools.format) 提取 ```json ... ``` 块。
    """
    try:
        # 1. 严格使用 extract_code 提取
        code_block = extract_code(text)

        if not code_block:
             # 如果 extract_code 没有返回任何内容
             print(f"No code block (e.g., ```json ... ```) found in LLM output: '{text}'")
             return None

        # 2. 解析提取到的 code_block
        parsed_json = json.loads(code_block)

        # 3. 验证类型
        if isinstance(parsed_json, (dict, list)):
            return parsed_json
        else:
            print(f"Parsed JSON from code block is not a dict or list. Type: {type(parsed_json)}. Text: '{text}'")
            return None

    except json.JSONDecodeError as e:
        print(f"Could not decode extracted JSON object/list from code block: {e}. Code block: '{code_block}'. Raw Text: '{text}'")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during JSON object/list extraction: {e}. Text: '{text}'")
        return None

# --- 3. JSON 日志保存 ---

def save_json(data: any, file_path: str):
    """
    将数据以 JSON 格式保存到文件，并确保目录存在。
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving to {file_path}: {e}")

# --- 4. AdjustAgent 类 ---

class AdjustAgent(BaseAgent):
    def __init__(self, log_dir: str = "./logs/classification_logs", *args, **kwargs):
        """
        初始化 AdjustAgent。
        """
        super().__init__(*args, **kwargs)
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # [修改] 实例属性，用于累积所有分类
        self.total_attack_tasks_raw = []
        self.total_standard_attack_commands = []
        self.total_empty_tasks = []
        self.total_other_tasks = []

        print(f"AdjustAgent initialized. Classification logs will be saved to {self.log_dir}")

    def save_json_logs(self):
        """
        [修改] 将 [所有累积] 的分类结果保存到 JSON 文件中。
        (不再需要参数)
        """
        try:
            # 保存第一阶段识别出的所有 Attack Task 字符串
            save_json(self.total_attack_tasks_raw, os.path.join(self.log_dir, "attack_tasks_raw.json"))
            # 保存第二阶段解析出的 Standard Attack (JSON 对象)
            save_json(self.total_standard_attack_commands, os.path.join(self.log_dir, "standard_attack_commands.json"))
            # 保存 Empty Task
            save_json(self.total_empty_tasks, os.path.join(self.log_dir, "empty_tasks.json"))
            # 保存 Other Task (现在也包括了特殊技能)
            save_json(self.total_other_tasks, os.path.join(self.log_dir, "other_tasks.json"))

            print(f"Cumulative classification logs saved to {self.log_dir}")
        except Exception as e:
            print(f"Failed to save cumulative classification logs: {e}")

    def classify_plan(self, plan: str):
        """
        [第一阶段] 调用 LLM 对单个 plan 进行基础分类。
        """
        if rule_based_attack_command(plan):
            return "Attack Task"

        if local_executor_fallback_enabled():
            lowered = str(plan).strip().lower()
            if not lowered or lowered in ("[]", "none", "null"):
                return "Empty Task"
            if "do nothing" in lowered or "wait" == lowered:
                return "Empty Task"
            print(f"Local classification fallback: treating non-attack plan as Other Task: {plan}")
            return "Other Task"

        prompt = build_plan_prompt(plan)

        try:
            response, messages = self.llm_client.call(
                **self.generation_config,
                prompt=prompt,
                need_json=True
            )

            categories_list = extract_json_list(response)

            category = None

            if categories_list and isinstance(categories_list, list) and len(categories_list) == 1:
                category = str(categories_list[0]).strip()
            else:
                print(f"Expected JSON list with 1 element, but got: {categories_list}. Raw: '{response}'")

            if category == "Attack Task":
                return "Attack Task"
            elif category == "Empty Task":
                return "Empty Task"
            elif category == "Other Task":
                return "Other Task"
            else:
                if category is not None:
                    print(f"Unknown category '{category}' for plan: '{plan}'. Defaulting to 'Other Task'.")
                else:
                    print(f"Parse failed for plan: '{plan}'. Defaulting to 'Other Task'.")
                return "Other Task"

        except Exception as e:
            print(f"Error during LLM call or classification for plan '{plan}': {e}")
            return "Other Task"

    def classify_attack_detail(self, plan: str):
        """
        [第二阶段] 调用 LLM 对 'Attack Task' plan 进行详细分类。
        返回解析后的 JSON 对象 (dict) 或列表 (list)。
        """
        rule_based_result = rule_based_attack_command(plan)
        if rule_based_result:
            return rule_based_result

        if local_executor_fallback_enabled():
            print(f"Local attack-detail fallback unavailable; skipping LLM for: {plan}")
            return None

        prompt = build_attack_prompt(plan)

        try:
            response, messages = self.llm_client.call(
                **self.generation_config,
                prompt=prompt,
                need_json=True
            )

            # 使用为第二阶段定制的解析器
            parsed_output = extract_attack_json_obj(response)

            return parsed_output # 可能是 dict, list, 或 None

        except Exception as e:
            print(f"Error during LLM call for attack detail classification '{plan}': {e}")
            return None # 失败时返回 None

    def run(self, plans: list[str]):
        """
        [修改] 执行两阶段分类任务。

        Args:
            plans (list[str]): PlanAgent 生成的自然语言规划列表。

        Returns:
            dict: [本轮] 分类后规划的字典。
                  e.g., {"standard_attack_commands": [{}], "empty_tasks": [], "other_tasks": []}
        """
        print(f"Starting 2-stage classification for {len(plans)} plans...")

        # 1. 为 [本轮] 初始化临时列表
        # 阶段1：基础分类
        current_attack_tasks_raw = []
        current_empty_tasks = []
        current_other_tasks = []

        # 阶段2：详细分类
        current_standard_attack_commands = [] # (新变量，存储 JSON 对象)

        for plan in plans:
            if not isinstance(plan, str) or not plan.strip():
                print(f"Skipping empty or invalid plan: {plan}")
                continue

            # --- 阶段 1 ---
            category = self.classify_plan(plan)

            # 2. 将结果附加到 [本轮] 列表
            if category == "Empty Task":
                current_empty_tasks.append(plan)
            elif category == "Other Task":
                current_other_tasks.append(plan)
            elif category == "Attack Task":
                # 标记为攻击任务，准备进入阶段 2
                current_attack_tasks_raw.append(plan)

                # --- 阶段 2 ---
                attack_detail = self.classify_attack_detail(plan)

                # 检查返回的是否是包含数据的 dict
                if isinstance(attack_detail, dict) and attack_detail:
                    # 这是 'Standard Attack Command'
                    current_standard_attack_commands.append(attack_detail)
                else:
                    # 这是 'Special Ability' (返回 '[]') 或解析失败 (返回 None)
                    # 按要求归类到 'Other Task'
                    current_other_tasks.append(plan)

        # 3. 将 [本轮] 结果累积到 [总] 实例属性中
        self.total_attack_tasks_raw.extend(current_attack_tasks_raw)
        self.total_standard_attack_commands.extend(current_standard_attack_commands)
        self.total_empty_tasks.extend(current_empty_tasks)
        self.total_other_tasks.extend(current_other_tasks)

        # 4. 保存 [总] 日志
        self.save_json_logs() # (不需要参数)

        # 5. 返回 [本轮] 结果
        result = {
            "standard_attack_commands": current_standard_attack_commands,
            "empty_tasks": current_empty_tasks,
            "other_tasks": current_other_tasks
        }

        print(f"Classification complete [Current Run]. Standard Attacks: {len(current_standard_attack_commands)}, Empty: {len(current_empty_tasks)}, Other/Special: {len(current_other_tasks)}.")
        print(f"Classification complete [Total]. Standard Attacks: {len(self.total_standard_attack_commands)}, Empty: {len(self.total_empty_tasks)}, Other/Special: {len(self.total_other_tasks)}.")

        return result
