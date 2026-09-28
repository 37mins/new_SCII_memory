from agents.base_agent import BaseAgent
from tools.format import extract_code
import json

class WorldModelAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def diagnose_action_failure(self, action_name: str, producer_type_name: str, relevant_units_text: str, recent_actions: list) -> dict:
        """
        当动作无法执行时，诊断原因。
        现在结合了 'recent_actions' 来判断是否是指令已下达但未生效的情况。
        """

        # 将列表转为字符串供 Prompt 阅读，处理空列表情况
        recent_history_str = str(recent_actions) if recent_actions else "None"

        print("WorldModelAgent diagnose_action_failure")
        print("!!! action_name")
        print(action_name)
        print("!!! producer_type_name")
        print(producer_type_name)
        print("!!! relevant_units_text")
        print(relevant_units_text)
        print("!!! recent_actions")
        print(recent_actions)

        prompt = f"""
You are a StarCraft II Game State Diagnostician.
The player wants to execute **{action_name}**, but the command is currently invalid.
Your task is to analyze the observation and history to determine why.

### Context
- **Desired Action**: {action_name}
- **Required Producer**: {producer_type_name}

### Observation (Relevant Units)
{relevant_units_text}

### Recent Action History (Last 10 commands sent)
{recent_history_str}

### Diagnostic Logic (Execute in Order)

**Step 1: Analyze Observation (Availability & State Check)**
Scan "Observation" for **{producer_type_name}**.

1.  **Identify Strict Candidates**:
    * Find all instances that **exactly match** `{producer_type_name}`.
    * *Note*: If looking for `CommandCenter`, do NOT count `OrbitalCommand` or `PlanetaryFortress` as candidates.
    * *Add-on Rule*: If the action is to build an add-on, ignore structures that already have one.

2.  **Check Status (The "BUSY" Check)**:
    Iterate through the found candidates. Even if a candidate exists, it is **BUSY** if:
    * **Condition A (Construction)**: It has `(building X%)`.
        -> Meaning: The structure itself is still being built.
    * **Condition B (Production Queue)**: Its `Production list` is NOT empty (e.g., contains "Marine", "SCV", "TechLab", etc.).
        -> **CRITICAL FOR UPGRADES**: A structure CANNOT transform/upgrade (e.g., CC -> Orbital) while it is producing units (e.g., making an SCV).
        -> Meaning: The structure is occupied with a production task.

    * **Diagnosis Decision**:
        * If you find a candidate but it matches Condition A or B -> Diagnosis: **"BUSY"**, Suggestion: **"Wait"**.
        * If you find a candidate that has NO construction progress AND an EMPTY production list -> It is available (Pass).

**Step 2: Analyze History (Pending Check)**
Only execute if NO valid `{producer_type_name}` exists at all (or all are fully upgraded/occupied permanently).
* **Condition C**: Command found in history to build the parent structure.
    -> Diagnosis: "PENDING", Suggestion: "Wait".

**Step 3: Default (Missing)**
If no candidates exist in Observation AND no pending build in History.
* **Condition D**:
    -> Diagnosis: "MISSING"
    -> Reason: "No {producer_type_name} available".
    -> Suggestion: "Construct {producer_type_name}"

### Output Format
Return a JSON object:
```json
{{
    "diagnosis": "BUSY" or "PENDING" or "MISSING",
    "reason": "Specific explanation. Examples: 'CommandCenter [219] is busy producing SCVs', 'Barracks is under construction (50%)', 'No Command Center found'.",
    "suggestion": "Wait" or "Construct <UnitName>"
}}
```
""".strip()

        response, _ = self.llm_client.call(prompt=prompt, **self.generation_config)

        try:
            result = json.loads(extract_code(response))
            # 默认建议设为 Wait，防止解析缺失
            if "suggestion" not in result:
                result["suggestion"] = "Wait"
            return result
        except Exception as e:
            print(f"[WorldModelAgent] Error parsing response: {e}")
            return {
                "diagnosis": "UNKNOWN",
                "reason": f"LLM parsing error: {response}",
                "suggestion": "Wait"
            }