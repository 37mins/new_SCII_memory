from collections import Counter
import re
import os
import json

import gymnasium as gym
from gymnasium import spaces
import socket
import pickle
import multiprocessing as mp
import time

DEFEAT_PENALTY = -50.0
LEGACY_STRICT_NAMES_ONLY = True

numbered_system_prompt = """
You are an expert in StarCraft II. Rank the next build-order segment for the current phase.
You will be given a numbered remaining BO list for this phase. Your output must choose only from that numbered list.
Return only bullet lines, with at most 15 lines. A line like `#1 x 2` still counts as one bullet line.
If the phased attacking instruction says no attacks are needed, do not output any attack instruction.
If an attack is needed, use one bullet line in the form `Launch OFFENSE with 14 marine`.
"""

legacy_segment_instruction = """Output at most {bullet_max} action bullets. Every build bullet must be exactly `- canonicalname`, where `canonicalname` is copied verbatim from the current remaining BO list.
For repeated actions, repeat the complete bullet line. Do not use `x N` in the answer.
Do not add timing, quantities, verbs, explanations, arrows, commas, aliases, plurals, upgrades, or multiple actions to one bullet."""

original_segment_instruction = """Provide the next segment's BO list, approximately 15 entries in length. This refers to list lines/entries, NOT the total count of units or expanded actions. A single attack instruction is one entry.
Every build bullet must be exactly `- canonicalname` or `- canonicalname x N`, where `canonicalname` is copied verbatim from the current remaining BO list and N is a positive integer within its remaining count.
For repeated actions, you may repeat the complete bullet line or use `x N`. The parser expands each `x N` entry into N actions, preserving list order; it still counts as one list entry. There is no hard limit of 15 expanded actions.
Do not add timing, verbs, explanations, arrows, commas, aliases, plurals, upgrades, or multiple action names to one bullet."""

legacy_system_prompt = """You are an expert in StarCraft II. Select the next near-term build-order actions from the current remaining BO list.
{segment_instruction}
Do not output an action merely because it appears in the global canonical table: it must also appear in the current remaining BO list.
An attack bullet is allowed only when an exact `Launch OFFENSE ...` entry appears in the current remaining BO list; otherwise output no attack.

## Optional Operations/Units:
These are the only canonical build-action names. Spelling is exact:
scv, supplydepot, barracks, commandcenter, marine, orbitalcommand, refinery, engineeringbay, planetaryfortress, factory, techlab factory, siegetank, liberator, raven, missileturret, techlab barracks, terraninfantryweaponslevel1, widowmine, fusioncore, terranstructurearmorupgrade, thor, marauder, concussiveshells, hi-secautotracking, mule, starport, reactor barracks, vikingfighter, techlab starport, combatshield, stimpack, armory, ghostacademy, terraninfantryarmorlevel1, ghost, terraninfantryweaponslevel2, personalcloaking, banshee, bansheecloakingfield, terraninfantryarmorlevel2, medivac, terraninfantryweaponslevel3, terraninfantryarmorlevel3, turret, bunker, reaper, hellion, battlecruiser, reactor starport.
"""

# question_prompt = """%s
# Now the phased strategic instructions are:
# %s
# Current observations are:
# %s
# The remaining BO list in this phase:
# %s
# Please give your answer following the example format:

# Your reasoning process.
# </think>
# <answer>
# - Build Order:
#   - supplydepot
#   - ...
# </answer>

# **The output format is not optional and must be followed precisely. Do not add any other text outside of these tags.**
# """

# question_prompt = """%s
# Now the phased strategic instructions are:
# %s
# Current observations are:
# %s
# The remaining BO list in this phase:
# %s
# Please give your answer following the example format:

# Your reasoning process.
# </think>
# <answer>
# Here are the Build Orders with attack instructions:
# - supplydepot
# - marine x 2
# - ...
# - Launch OFFENSE with 2 Marine, 1 Marauder
# - ...
# </answer>

# **The output format is not optional and must be followed precisely. Do not add any other text outside of these tags.**
# """

numbered_question_prompt = """%s
Now the phased strategic instructions are:
%s
Current observations are:
%s
The remaining BO list in this phase:
%s
Please output only the build-order section by choosing entries from the numbered remaining BO list.
Rules:
- Do not include reasoning, markdown fences, XML/HTML tags, placeholder text, or ellipsis lines.
- Every build-order line must start with "- ".
- Prefer the compact form "- #N x K" where #N is a choice number from the remaining BO list.
- You may also use the exact lowercase item name from the remaining BO list.
- Output Launch OFFENSE only when it appears as a numbered entry in the remaining BO list.
- If the numbered list contains a Launch OFFENSE entry, choose that numbered entry when the current army is ready to attack.
- For large remaining counts, output only the near-term subset instead of copying the full count.
- Do not invent generic words such as worker, builder, soldier, racer, extractor, or building.
- Never output a # number that is not shown in the remaining BO list.
- Stop immediately after the last selected bullet line.
- Output at most 15 bullet lines.
- If no attack is requested, do not output any attack line.

Here are the Build Orders with attack instructions:
"""

legacy_question_prompt = """%s
Now the phased strategic instructions are:
%s
Current observations are:
%s
The remaining BO list in this phase:
%s
Use only exact canonical names that occur in the remaining BO list above.
The example below is generated from that same current list. Follow its shape exactly:

<answer>
Here are the Build Orders with attack instructions:
%s
</answer>

**The output format is not optional and must be followed precisely. Do not add any other text outside of these tags.**
"""

instruction_prompt = """- Current Stage: %s %s
- Detailed Instruction: %s
- Attacking Instruction: %s
"""

def clarify_bo_quota(query):
    """Apply the fixed-query candidate without changing the BO contract."""
    return query.replace(
        "The example below is generated from that same current list. Follow its shape exactly:",
        """For each action name, add its quantities across ALL lines in your answer: a line without `x N` counts as 1. This cumulative total must not exceed that action's count in the remaining BO list.
The example below demonstrates output format only; it is not an additional task or an extra quota. Do not append the remaining list to a copied example and count both separately.
Aim for approximately 15 list entries, but do not repeat actions just to fill that target. Once this near-term segment is complete, close </answer> and stop; do not restart the list. Repeated names and `x N` remain allowed within the cumulative quota.
The example below is generated from that same current list. Follow its format:""",
    )

def evenly_insert(lst1, lst2):
    n = len(lst2)
    m = len(lst1)

    # 计算每个分段的长度
    segment_size = m // n
    remainder = m % n

    result = []
    start_idx = 0

    for i in range(n):
        # 计算当前段的长度，余数均匀分配给前几个段
        current_segment_size = segment_size + (1 if i < remainder else 0)

        # 添加当前段的元素
        result.extend(lst1[start_idx:start_idx + current_segment_size])

        # 添加对应的第二个列表元素
        result.append(lst2[i])

        # 更新起始索引
        start_idx += current_segment_size

    return result

def wash_attack(s):
    processed_text = re.sub(r'^At \d{1,2}:\d{2},?\s*', '', s)

    # 将第一个单词的首字母大写
    if processed_text:
        # 找到第一个单词的边界
        words = processed_text.split()
        if words:
            # 将第一个单词首字母大写
            words[0] = words[0].capitalize()
            processed_text = ' '.join(words)

    return f"Attacking Instruction: {processed_text}"

def get_bo(action):
    action = action.split('<answer>')[-1]

    pattern = r'-\s+([^-]+?)(?:\s*x\s*(\d+))?\s*$'

    result = []
    for line in action.split('\n'):
        match = re.search(pattern, line)
        if match:
            item = match.group(1).strip()
            count_str = match.group(2)

            if count_str:
                # 有重复次数
                count = int(count_str)
                # result.extend([item] * count)
                result.extend([item] * count)
            else:
                # 没有重复次数，默认1次
                result.append(item)
    return result

# def parse_build_order(text):
#     # 提取所有项目（包含 x N 格式）
#     pattern = r'-\s*([a-zA-Z]+)(?:\s*x\s*(\d+))?'
#     matches = re.findall(pattern, text)

#     result = []
#     for item, count in matches:
#         if count:
#             # 如果有数量（如 x 2），则重复添加相应次数
#             result.extend([item] * int(count))
#         else:
#             # 如果没有数量，直接添加
#             result.append(item)

#     return result

BUILD_ITEM_ALIASES = {
    "supply depot": "supplydepot",
    "supply depots": "supplydepot",
    "command center": "commandcenter",
    "command centers": "commandcenter",
    "orbital command": "orbitalcommand",
    "orbital commands": "orbitalcommand",
    "engineering bay": "engineeringbay",
    "engineering bays": "engineeringbay",
    "planetary fortress": "planetaryfortress",
    "planetary fortresses": "planetaryfortress",
    "techlab": "techlab barracks",
    "tech lab": "techlab barracks",
    "research lab": "techlab barracks",
    "research labs": "techlab barracks",
    "barracks with techlab": "techlab barracks",
    "barracks with tech lab": "techlab barracks",
    "barracks techlab": "techlab barracks",
    "barracks tech lab": "techlab barracks",
    "refineries": "refinery",
    "refiner": "refinery",
    "command center upgrade": "orbitalcommand",
    "command center upgrades": "orbitalcommand",
    "siege tank": "siegetank",
    "siege tanks": "siegetank",
    "missile turret": "missileturret",
    "missile turrets": "missileturret",
    "viking fighter": "vikingfighter",
    "viking fighters": "vikingfighter",
    "tech lab factory": "techlab factory",
    "tech lab starport": "techlab starport",
    "combat shield": "combatshield",
    "ghost academy": "ghostacademy",
    "banshee cloaking field": "bansheecloakingfield",
    "battle cruiser": "battlecruiser",
    "battle cruisers": "battlecruiser",
}

ALLOWED_BUILD_ITEMS = {
    "scv", "supplydepot", "barracks", "commandcenter", "marine",
    "orbitalcommand", "refinery", "engineeringbay", "planetaryfortress",
    "factory", "techlab factory", "siegetank", "liberator", "raven",
    "missileturret", "techlab barracks", "terraninfantryweaponslevel1",
    "widowmine", "fusioncore", "terranstructurearmorupgrade", "thor",
    "marauder", "concussiveshells", "hi-secautotracking", "mule", "starport",
    "reactor barracks", "vikingfighter", "techlab starport", "combatshield",
    "stimpack", "armory", "ghostacademy", "terraninfantryarmorlevel1",
    "ghost", "terraninfantryweaponslevel2", "personalcloaking", "banshee",
    "bansheecloakingfield", "terraninfantryarmorlevel2", "medivac",
    "terraninfantryweaponslevel3", "terraninfantryarmorlevel3", "turret",
    "bunker", "reaper", "hellion", "battlecruiser", "reactor starport",
}

def normalize_build_item(item):
    item = item.strip()
    if item.lower().startswith("launch offense"):
        return item
    item = re.sub(r"`|<[^>]+>", "", item)
    item = re.sub(r"\s*\([^)]*\)\s*", " ", item)
    item = item.replace("_", " ").replace("-", " ")
    item = re.sub(r"[^a-zA-Z0-9 ]+", " ", item)
    item = re.sub(r"\s+", " ", item).strip().lower()
    item = re.sub(r"^\d+\s+", "", item)
    item = BUILD_ITEM_ALIASES.get(item, item)
    if item in ALLOWED_BUILD_ITEMS:
        return item
    if item.endswith("s") and item[:-1] in ALLOWED_BUILD_ITEMS:
        return item[:-1]
    return None

BO_HEURISTIC_PRIORITY = [
    "supplydepot",
    "barracks",
    "refinery",
    "marine",
    "commandcenter",
    "orbitalcommand",
    "techlab barracks",
    "engineeringbay",
    "missileturret",
    "marauder",
    "factory",
    "starport",
    "vikingfighter",
    "techlab factory",
    "techlab starport",
    "stimpack",
    "combatshield",
    "siegetank",
    "medivac",
    "raven",
    "planetaryfortress",
]

ATTACK_BO_PRIORITY = [
    "supplydepot",
    "barracks",
    "refinery",
    "marine",
    "engineeringbay",
    "missileturret",
    "reactor barracks",
    "techlab barracks",
    "marauder",
    "stimpack",
    "combatshield",
    "medivac",
    "siegetank",
    "factory",
    "starport",
    "vikingfighter",
    "scv",
    "commandcenter",
    "orbitalcommand",
]

BO_SEGMENT_CAPS = {
    "scv": 6,
    "supplydepot": 2,
    "barracks": 2,
    "commandcenter": 1,
    "orbitalcommand": 1,
    "refinery": 2,
    "techlab barracks": 1,
    "engineeringbay": 1,
    "missileturret": 2,
    "factory": 1,
    "starport": 1,
    "techlab factory": 1,
    "techlab starport": 1,
    "marine": 6,
    "marauder": 3,
    "siegetank": 2,
    "medivac": 2,
    "raven": 1,
    "vikingfighter": 2,
}

def get_plan_segment_max(default=96):
    try:
        value = int(os.getenv("SC2_PLAN_SEGMENT_MAX", str(default)))
    except ValueError:
        value = default
    return max(1, min(value, 512))


def get_plan_bullet_max(default=15):
    try:
        value = int(os.getenv("SC2_PLAN_BULLET_MAX", str(default)))
    except ValueError:
        value = default
    return max(1, min(value, 15))


def get_planner_protocol():
    value = os.getenv("SC2_PLANNER_PROTOCOL", "numbered").strip().lower()
    return "legacy" if value in {"legacy", "legacy_names", "names"} else "numbered"

def parse_build_order(text, choice_map=None, strict_names_only=False, original_segment=False):
    """
    Parse a multi-line build order string where each line starts with '-' or '*'.
    In strict legacy mode, accepts only `- canonicalname` lines copied exactly
    from the current choice map. Other modes retain the historical compact
    count and numbered-choice parsing.
    original_segment is used only after whole-answer validation; it expands
    exact-name counts without a segment cap. Invalid answers use the old path.
    Returns a list of build items in order.
    """
    lines = text.strip().split('\n')
    result = []
    choice_map = choice_map or {}
    allowed_choices = {
        str(value).strip().lower() for value in choice_map.values()
    }
    allowed_exact_choices = {
        str(value).strip() for value in choice_map.values()
    }
    max_items = get_plan_segment_max()

    def repeat_count(raw_count, remaining_budget):
        if remaining_budget <= 0:
            return 0
        try:
            count = int(raw_count) if raw_count else 1
        except (TypeError, ValueError, OverflowError):
            count = 1
        count = max(1, count)
        if count > remaining_budget:
            print(f"[Planner] Capping repeated BO count {count} to remaining segment budget {remaining_budget}.")
            count = remaining_budget
        return count

    def is_allowed_choice(item):
        if not choice_map:
            return True
        return str(item).strip().lower() in allowed_choices

    for line in lines:
        if not original_segment and len(result) >= max_items:
            break
        line = line.strip()
        if not line or line[0] not in ('-', '*'):
            continue
        if strict_names_only:
            if not line.startswith("- "):
                continue
            item = line[2:].strip()
            if original_segment:
                count_match = re.fullmatch(r"(.+?)\s*x\s*(\d+)", item)
                count = int(count_match.group(2)) if count_match else 1
                item = count_match.group(1).strip() if count_match else item
                if item in allowed_exact_choices:
                    result.extend([item] * count)
                continue
            if item in allowed_exact_choices:
                result.append(item)
            continue
        item = line[1:].strip()
        choice_match = re.match(r"#?\s*(\d+)(?:\s*[.)])?(?:\s+x\s*(\d+))?\s*$", item, flags=re.IGNORECASE)
        if choice_match:
            choice = choice_map.get(choice_match.group(1))
            if choice is None:
                continue
            count = repeat_count(choice_match.group(2), max_items - len(result))
            result.extend([choice] * count)
            continue
        count_match = re.match(r"(.+?)\s+x\s*(\d+)\s*$", item, flags=re.IGNORECASE)
        if count_match:
            base = normalize_build_item(count_match.group(1))
            if base is None or not is_allowed_choice(base):
                continue
            count = repeat_count(count_match.group(2), max_items - len(result))
            result.extend([base] * count)
            continue
        item = normalize_build_item(item)
        if item is None or not is_allowed_choice(item):
            continue
        result.append(item)
    return result

def attack_instruction_allows_attack(attack_instruction):
    text = str(attack_instruction or "").lower()
    if not text.strip():
        return False
    no_attack_patterns = (
        "no attack",
        "no attacks",
        "no attacking",
        "no early attack",
        "no early attacks",
        "no immediate attack",
        "no immediate attacks",
        "do not attack",
        "without attacking",
    )
    if any(pattern in text for pattern in no_attack_patterns):
        return False
    if re.search(r"\b(do not|don't|dont)\s+(launch|start|commit)?\s*attacks?\b", text):
        return False
    if re.search(r"\bkeep\b.*\bhome\b", text):
        return False
    return True

def normalize_attack_order(text):
    text = re.sub(r"^\s*[-*]\s*", "", str(text)).strip()
    text = re.sub(r"^attacking instruction\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    if not text:
        return None
    if re.search(
        r"\b(launch\s+offense|attack\s+with|strike|commit|dispatch|assault|harass|pressure)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return text
    return None

def extract_attack_orders(text):
    attacks = []
    for line in str(text or "").splitlines():
        body = line.strip()
        if not body.startswith(("-", "*")):
            continue
        attack = normalize_attack_order(body)
        if attack:
            attacks.append(attack)
    return attacks

def default_attack_order(attack_instruction):
    attack = normalize_attack_order(attack_instruction)
    if attack:
        return attack
    return "Attack with available army"

def attack_threshold_from_instruction(attack_instruction, default=8):
    text = str(attack_instruction or "").lower()
    patterns = (
        r"as soon as\s+(\d+)\s+(?:combat\s+units?|marines?|units?)",
        r"once\s+(\d+)\s+(?:combat\s+units?|marines?|units?)",
        r"with\s+(\d+)\s+(?:combat\s+units?|marines?|units?)",
    )
    for pattern in patterns:
        threshold = _extract_number(pattern, text)
        if threshold:
            return threshold
    return default

def army_supply_from_obs(obs):
    own_text = str(obs or "").lower().split("as for the visible enemy")[0]
    return (
        _extract_number(r"supply army:\s*(\d+)", own_text)
        or _extract_number(r"of which\s+(\d+)\s+is allocated to the army", own_text)
    )

def attack_option_for_state(obs, attack_instruction, attack_ct=0):
    if not attack_instruction_allows_attack(attack_instruction):
        return None
    threshold = attack_threshold_from_instruction(attack_instruction)
    army_supply = army_supply_from_obs(obs)
    try:
        issued_attacks = max(0, int(attack_ct or 0))
    except (TypeError, ValueError):
        issued_attacks = 0
    if os.getenv("SC2_ATTACK_OPTION_ACCOUNT_ISSUED", "1") != "0":
        required_army_supply = threshold * (issued_attacks + 1)
    else:
        required_army_supply = threshold
    if army_supply < required_army_supply:
        return None
    return f"Launch OFFENSE with {threshold} marine"

ANTI_AIR_BO_PRIORITY = [
    "supplydepot",
    "barracks",
    "marine",
    "barracks",
    "marine",
    "engineeringbay",
    "missileturret",
    "barracks",
    "marine",
    "orbitalcommand",
    "supplydepot",
    "marine",
    "refinery",
    "reactor barracks",
    "factory",
    "starport",
    "vikingfighter",
    "techlab factory",
    "marine",
    "scv",
]

ANTI_AIR_ATTACK_BO_PRIORITY = [
    "marine",
    "marine",
    "marine",
    "supplydepot",
    "missileturret",
    "marine",
    "refinery",
    "reactor barracks",
    "factory",
    "starport",
    "vikingfighter",
    "vikingfighter",
    "siegetank",
    "scv",
]

def build_scheduled_sequence(counts, max_items=15, attack_focus=False):
    counts = Counter(counts)
    result = []
    anti_air_focus = (
        counts["engineeringbay"] > 0
        or counts["missileturret"] > 0
        or counts["vikingfighter"] > 0
    )
    if attack_focus and anti_air_focus:
        priority = ANTI_AIR_ATTACK_BO_PRIORITY
    elif attack_focus:
        priority = ATTACK_BO_PRIORITY
    elif anti_air_focus:
        priority = ANTI_AIR_BO_PRIORITY
    else:
        priority = BO_HEURISTIC_PRIORITY

    if max_items <= 6:
        if attack_focus and anti_air_focus:
            small_segment_priority = ANTI_AIR_ATTACK_BO_PRIORITY
        elif attack_focus:
            small_segment_priority = [
                "supplydepot",
                "barracks",
                "refinery",
                "marine",
                "engineeringbay",
                "missileturret",
                "barracks",
                "reactor barracks",
                "techlab barracks",
                "marauder",
                "marine",
                "factory",
                "starport",
                "vikingfighter",
            ]
        elif anti_air_focus:
            if counts["vikingfighter"] > 0:
                small_segment_priority = [
                    "marine",
                    "marine",
                    "refinery",
                    "factory",
                    "starport",
                    "vikingfighter",
                    "marine",
                    "vikingfighter",
                    "missileturret",
                    "reactor barracks",
                    "supplydepot",
                    "scv",
                ]
            elif counts["engineeringbay"] > 0:
                small_segment_priority = [
                    "supplydepot",
                    "barracks",
                    "marine",
                    "barracks",
                    "marine",
                    "engineeringbay",
                    "missileturret",
                    "marine",
                    "missileturret",
                    "supplydepot",
                    "barracks",
                    "orbitalcommand",
                ]
            elif counts["missileturret"] > 0:
                small_segment_priority = [
                    "supplydepot",
                    "barracks",
                    "marine",
                    "refinery",
                    "marine",
                    "missileturret",
                    "barracks",
                    "marine",
                    "refinery",
                    "orbitalcommand",
                ]
            else:
                small_segment_priority = [
                    "supplydepot",
                    "barracks",
                    "marine",
                    "marine",
                    "refinery",
                    "orbitalcommand",
                    "barracks",
                    "marine",
                    "scv",
                ]
        else:
            small_segment_priority = ["scv"] + BO_HEURISTIC_PRIORITY
        for item in small_segment_priority:
            if len(result) >= max_items:
                break
            if counts[item] <= 0:
                continue
            result.append(item)
            counts[item] -= 1

        for item in list(counts):
            while counts[item] > 0 and len(result) < max_items:
                result.append(item)
                counts[item] -= 1

        return result

    while len(result) < max_items and sum(counts.values()) > 0:
        placed = False

        if not attack_focus and not anti_air_focus and counts["scv"] > 0:
            result.append("scv")
            counts["scv"] -= 1
            placed = True
            if len(result) >= max_items:
                break

        for item in priority:
            if counts[item] <= 0:
                continue
            result.append(item)
            counts[item] -= 1
            placed = True
            break

        if placed:
            continue

        for item in list(counts):
            if counts[item] <= 0:
                continue
            result.append(item)
            counts[item] -= 1
            placed = True
            break

        if not placed:
            break

    return result

def format_bo_todo(bo_todo, max_items=15, attack_focus=False, attack_option=None):
    lines = []
    counts = Counter()
    choice_map = {}
    next_idx = 1
    if attack_option:
        lines.append(f"{next_idx}. {attack_option}")
        choice_map[str(next_idx)] = attack_option
        next_idx += 1
    for name, count in bo_todo.items():
        item = normalize_build_item(str(name))
        if item is None:
            continue
        try:
            count_int = int(float(count))
        except (TypeError, ValueError):
            count_int = 1
        if count_int <= 0:
            continue
        lines.append(f"{next_idx}. {item} x {count_int}")
        choice_map[str(next_idx)] = item
        next_idx += 1
        counts[item] += count_int
    sequence = build_scheduled_sequence(
        counts,
        max_items=min(sum(counts.values()), 80),
        attack_focus=attack_focus,
    )
    return lines, sequence, choice_map

def constrain_bo_to_todo(bo, bo_todo_sequence, max_items=15, attack_focus=False):
    if not bo_todo_sequence:
        return bo[:max_items], False

    todo_counts = Counter(bo_todo_sequence)
    selected_counts = Counter()
    anti_air_focus = any(
        item in todo_counts for item in ("engineeringbay", "missileturret", "vikingfighter")
    )
    required_counts = Counter(
        build_scheduled_sequence(todo_counts, max_items=max_items, attack_focus=attack_focus)
    )

    def item_cap(item):
        per_item_segment_cap = max(1, max_items // 2)
        if item == "scv" and max_items <= 4 and any(k != "scv" for k in todo_counts):
            per_item_segment_cap = 1
        return min(todo_counts[item], BO_SEGMENT_CAPS.get(item, max_items), per_item_segment_cap)

    def flexible_cap(item):
        per_item_segment_cap = max(1, max_items // 2)
        return min(BO_SEGMENT_CAPS.get(item, 1), per_item_segment_cap)

    for item in bo:
        if item.lower().startswith("launch offense"):
            continue
        if item in todo_counts:
            cap = item_cap(item)
        else:
            continue
        if selected_counts[item] >= cap:
            continue
        selected_counts[item] += 1

    used_fallback = False
    if not selected_counts:
        for item in bo_todo_sequence:
            if sum(selected_counts.values()) >= max_items:
                break
            if selected_counts[item] >= item_cap(item):
                continue
            selected_counts[item] += 1
        used_fallback = True
    else:
        for item in bo_todo_sequence:
            if sum(selected_counts.values()) >= max_items:
                break
            if selected_counts[item] >= item_cap(item):
                continue
            selected_counts[item] += 1

    for item, count in required_counts.items():
        selected_counts[item] = max(selected_counts[item], min(count, item_cap(item)))

    return build_scheduled_sequence(
        selected_counts,
        max_items=max_items,
        attack_focus=attack_focus,
    ), used_fallback

def format_choice_map(choice_map):
    return ", ".join(f"#{idx}={item}" for idx, item in sorted(choice_map.items(), key=lambda kv: int(kv[0])))


def _own_structure_count_from_obs(obs, name):
    if not isinstance(obs, str):
        return 0
    own_text = obs.lower().split("as for the visible enemy")[0]
    own_structures = _extract_markdown_section(obs, "Own structures")
    aliases = {
        "techlab barracks": ("techlab barracks", "barracks with techlab"),
        "reactor barracks": ("reactor barracks", "barracks with reactor"),
        "techlab factory": ("techlab factory", "factory with techlab"),
        "techlab starport": ("techlab starport", "starport with techlab"),
    }
    names = aliases.get(name, (name,))
    prose_count = max((_own_unit_count(alias, own_text) for alias in names), default=0)
    return prose_count or _count_structured_entities(own_structures, names)


def filter_bo_todo_for_prereqs(bo_todo, obs):
    filtered = dict(bo_todo or {})
    normalized_keys = {normalize_build_item(str(key)): key for key in filtered}
    barracks_count = _own_structure_count_from_obs(obs, "barracks")
    factory_count = _own_structure_count_from_obs(obs, "factory")
    starport_count = _own_structure_count_from_obs(obs, "starport")
    refinery_count = _own_structure_count_from_obs(obs, "refinery")
    barracks_techlab_count = _own_structure_count_from_obs(obs, "techlab barracks")

    def remove_item(item):
        key = normalized_keys.get(item)
        if key is not None:
            filtered.pop(key, None)

    gas_prereq_items = {
        "reactor barracks",
        "techlab barracks",
        "marauder",
        "concussiveshells",
        "stimpack",
        "combatshield",
        "factory",
        "starport",
        "vikingfighter",
        "siegetank",
        "medivac",
    }
    wants_gas = any(item in normalized_keys for item in gas_prereq_items)
    if wants_gas and refinery_count <= 0 and "refinery" not in normalized_keys:
        filtered["refinery"] = 1
        normalized_keys["refinery"] = "refinery"

    barracks_prereq_items = {
        "reactor barracks",
        "techlab barracks",
        "marauder",
        "concussiveshells",
        "stimpack",
        "combatshield",
    }
    wants_barracks_addon = any(item in normalized_keys for item in barracks_prereq_items)
    if wants_barracks_addon and barracks_count <= 0 and "barracks" not in normalized_keys:
        filtered["barracks"] = 1
        normalized_keys["barracks"] = "barracks"

    techlab_prereq_items = {"marauder", "concussiveshells", "stimpack", "combatshield"}
    wants_barracks_techlab = any(item in normalized_keys for item in techlab_prereq_items)
    if (
        wants_barracks_techlab
        and barracks_techlab_count <= 0
        and "techlab barracks" not in normalized_keys
    ):
        filtered["techlab barracks"] = 1
        normalized_keys["techlab barracks"] = "techlab barracks"

    tech_items = {"factory", "starport", "vikingfighter", "siegetank"}
    wants_tech = any(item in normalized_keys for item in tech_items)
    if wants_tech and refinery_count <= 0 and "refinery" not in normalized_keys:
        filtered["refinery"] = 1

    if factory_count <= 0:
        if wants_tech and "factory" not in normalized_keys:
            filtered["factory"] = 1
        remove_item("starport")
        remove_item("vikingfighter")
        remove_item("siegetank")
    elif starport_count <= 0:
        remove_item("vikingfighter")

    return filtered


def _extract_number(pattern, text, default=0):
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return default
    try:
        return int(float(match.group(1)))
    except (TypeError, ValueError):
        return default


def _own_unit_count(name, own_text):
    pattern = rf"(\d+)\s+{re.escape(name)}\b"
    values = []
    for value in re.findall(pattern, own_text, flags=re.IGNORECASE):
        try:
            values.append(int(float(value)))
        except (TypeError, ValueError):
            continue
    return max(values) if values else 0


def _extract_markdown_section(text, title):
    pattern = rf"#\s*{re.escape(title)}\s*\n(.*?)(?=\n#\s|\Z)"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else ""


def _count_structured_entities(section, names):
    if isinstance(names, str):
        names = [names]
    total = 0
    normalized_names = {name.lower() for name in names}
    for line in section.splitlines():
        match = re.match(r"\[([0-9,\s]+)\]\s*([A-Za-z]+)", line.strip())
        if not match:
            continue
        entity = normalize_build_item(match.group(2)) or match.group(2).lower()
        if entity not in normalized_names:
            continue
        ids = [value for value in match.group(1).split(",") if value.strip()]
        total += max(1, len(ids))
    return total


def final_state_shaping_reward(obs):
    if not isinstance(obs, str):
        return 0.0

    text = obs.lower()
    own_text = text.split("as for the visible enemy")[0]
    army_supply = (
        _extract_number(r"supply army:\s*(\d+)", own_text)
        or _extract_number(r"of which\s+(\d+)\s+is allocated to the army", own_text)
    )
    worker_supply = (
        _extract_number(r"supply workers:\s*(\d+)", own_text)
        or _extract_number(r"army,\s+(\d+)\s+to workers", own_text)
    )
    minerals = (
        _extract_number(r"minerals:\s*(\d+)", own_text)
        or _extract_number(r"with\s+(\d+)\s+minerals", own_text)
    )
    gas = (
        _extract_number(r"vespene:\s*(\d+)", own_text)
        or _extract_number(r"and\s+(\d+)\s+vespene gas", own_text)
    )

    own_units = _extract_markdown_section(obs, "Own units")
    own_structures = _extract_markdown_section(obs, "Own structures")

    structured_bases = _count_structured_entities(
        own_structures,
        ["commandcenter", "orbitalcommand", "planetaryfortress"],
    )
    bases = _extract_number(r"we have\s+(\d+)\s+bases", own_text) or structured_bases

    barracks = _own_unit_count("barracks", own_text) or _count_structured_entities(own_structures, "barracks")
    refinery = _own_unit_count("refinery", own_text) or _count_structured_entities(own_structures, "refinery")
    supplydepot = (
        _own_unit_count("supplydepot", own_text)
        or _count_structured_entities(own_structures, ["supplydepot", "supplydepotlowered"])
    )
    techlab = _own_unit_count("techlab barracks", own_text) or _count_structured_entities(own_structures, "techlab")
    factory = _own_unit_count("factory", own_text) or _count_structured_entities(own_structures, "factory")
    starport = _own_unit_count("starport", own_text) or _count_structured_entities(own_structures, "starport")
    marines = _own_unit_count("marine", own_text) or _count_structured_entities(own_units, "marine")
    marauders = _own_unit_count("marauder", own_text) or _count_structured_entities(own_units, "marauder")

    reward = 0.0
    reward += min(worker_supply, 24) / 24.0 * 0.45
    reward += min(army_supply, 16) / 16.0 * 0.45
    reward += min(bases, 2) / 2.0 * 0.25
    reward += min(barracks, 3) / 3.0 * 0.30
    reward += min(refinery, 2) / 2.0 * 0.20
    reward += min(supplydepot, 3) / 3.0 * 0.15
    reward += min(techlab, 2) / 2.0 * 0.15
    reward += min(factory + starport, 2) / 2.0 * 0.15
    reward += min(marines + marauders, 12) / 12.0 * 0.25

    if minerals > 900:
        reward -= 0.08
    if gas > 900:
        reward -= 0.05

    return max(0.0, min(2.0, round(reward, 3)))


def get_attack(action):
    action = action.split('<answer>')[-1]
    if 'No attacking needed in this phase' in action:
        return []

    action = action.replace('\n</answer>', '').split('- Attacking Instructions:\n')[1]

    result = []
    for line in action.split('\n'):
        line = line.strip('  -')
        result.append(wash_attack(line))

    return result

class SC2Planner():
    def __init__(self, host='127.0.0.1', port=9999) -> None:
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((host, port))
        self.ct = 0
        self.last_consumed_policy_action_id = None

        self.correct_format = False
        self.exceed = False
        self.terminal_ack_timeout = float(os.getenv("SC2_TERMINAL_ACK_TIMEOUT", "5.0"))

        msg = self._recv()

    def _send(self, data):
        payload = pickle.dumps(data)
        self.client_socket.sendall(len(payload).to_bytes(4, byteorder='big'))
        self.client_socket.sendall(payload)

    def _recv(self):
        header = self._recv_exact(4)
        if not header: return None
        length = int.from_bytes(header, byteorder='big')
        payload = self._recv_exact(length)
        if not payload:
            return None
        return pickle.loads(payload)

    def _recv_exact(self, length):
        chunks = []
        bytes_recvd = 0
        while bytes_recvd < length:
            chunk = self.client_socket.recv(length - bytes_recvd)
            if not chunk:
                return None
            chunks.append(chunk)
            bytes_recvd += len(chunk)
        return b''.join(chunks)

    def game_end(self, obs, result, attack_ct=0, telemetry=None):
        from agent_system.sc2_episode import outcome_fields
        reward = 0

        if self.correct_format:
            reward += 0.5
        print(f"Plan format reward: {0.5 if self.correct_format else 0.0:.1f}")

        # if not self.exceed:
        #     reward += 0.5

        print(f"Result: {result}")
        outcome = outcome_fields(result)
        reward += outcome['outcome_reward']
        if result != 'Victory':
            state_reward = final_state_shaping_reward(obs)
            reward += state_reward
            print(f"Outcome shaping reward: {state_reward:.3f}")

        from agent_system.sc2_combat_reward import consume_combat_reward
        reward += consume_combat_reward(self, "terminal")
        terminal_data = {"obs": obs, "reward": reward, "done": True, "success": result=='Victory', "attack_ct": attack_ct, "correct_format":self.correct_format, "exceed":self.exceed}
        if telemetry is not None:
            terminal_data["telemetry"] = telemetry
        terminal_data.update(outcome)
        terminal_data["last_consumed_policy_action_id"] = self.last_consumed_policy_action_id
        self._send(terminal_data)
        self.game_end_flag = True
        self._wait_terminal_ack(terminal_data)
        print("Game end.")

    def _wait_terminal_ack(self, terminal_data):
        if self.terminal_ack_timeout <= 0:
            return
        previous_timeout = self.client_socket.gettimeout()
        deadline = time.time() + self.terminal_ack_timeout
        try:
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    print(f"Terminal ack timed out after {self.terminal_ack_timeout}s.", flush=True)
                    return
                self.client_socket.settimeout(remaining)
                ack = self._recv()
                if ack and ack.get("msg") == "TERMINAL_ACK":
                    print("Terminal ack received.", flush=True)
                    return
                if ack and ack.get("msg") == "ACTION":
                    print("Terminal received ACTION after done; resending terminal data.", flush=True)
                    self._send(terminal_data)
                    continue
                print(f"Terminal ack missing or unexpected: {ack}", flush=True)
                return
        except socket.timeout:
            print(f"Terminal ack timed out after {self.terminal_ack_timeout}s.", flush=True)
        except Exception as exc:
            print(f"Terminal ack wait failed: {exc}", flush=True)
        finally:
            self.client_socket.settimeout(previous_timeout)

    def get_action(self, obs, reward=0.0, done=False, attack_ct=0, telemetry=None):
        """这就是信号发送/接收端 B"""

        # if done:
        #     reward += -50

        if not done and os.getenv("SC2_STEP_SHAPING_MODE", "legacy") == "legacy":
            state_reward = final_state_shaping_reward(obs)
            if state_reward > 0:
                step_reward = min(0.5, round(0.25 * state_reward, 3))
                reward += step_reward
                print(f"Step shaping reward: {step_reward:.3f}")
        elif not done:
            print("Step shaping reward off: 0.000")

        from agent_system.sc2_combat_reward import consume_combat_reward
        reward += consume_combat_reward(self, "planner")
        data = {"obs": obs, "reward": reward, "done": done, "success": False, "attack_ct": attack_ct, "correct_format":self.correct_format, "exceed":self.exceed,
            "last_consumed_policy_action_id": self.last_consumed_policy_action_id}
        if telemetry is not None:
            data["telemetry"] = telemetry
        self._send(data)
        response = self._recv()

        if response and response["msg"] == "ACTION":
            self.last_consumed_policy_action_id = response["policy_action_id"]
            self.correct_format = bool(response.get("plan_format_valid", False))
            trace = getattr(self, "command_trace", None)
            if trace is not None:
                trace.answer(obs, response)
            return response["value"]

        return ''

    def run(self):
        # 业务逻辑：等待 Reset 信号后开始
        msg = self._recv()
        if msg and msg["msg"] == "RESET":
            self._send([0.0] * 10) # 返回初始 obs

            # 环境 A 的主循环
            for i in range(100):
                # 模拟环境计算...
                obs = [0.1 * i] * 10
                action = self.get_action(obs, reward=float(i), done=(i==99))
                if i == 99: break

    def generate_plan(self, obs, bo_todo, strategy, reward, done=False, attack_ct=0,
                      execution_feedback=None, telemetry=None):
        segment_max = get_plan_segment_max()
        bullet_max = get_plan_bullet_max()
        attack_allowed = attack_instruction_allows_attack(strategy.get("attack_instruction", ""))
        bo_todo = filter_bo_todo_for_prereqs(bo_todo, obs)
        attack_option = attack_option_for_state(obs, strategy.get("attack_instruction", ""), attack_ct=attack_ct)
        bo_todo_, bo_todo_sequence, choice_map = format_bo_todo(
            bo_todo,
            max_items=segment_max,
            attack_focus=attack_allowed,
            attack_option=attack_option,
        )
        planner_protocol = get_planner_protocol()
        original_segment = (planner_protocol == "legacy"
                            and os.getenv("SC2_ORIGINAL_BO_SEGMENT", "0") == "1")
        if planner_protocol == "legacy":
            legacy_lines = []
            legacy_example_lines = []
            for line in bo_todo_:
                match = re.match(r"^\s*\d+\.\s*(.+?)\s*$", line)
                if match:
                    legacy_lines.append(f"- {match.group(1)}")
                example_match = re.match(r"^\s*\d+\.\s*(.+?)\s+x\s+\d+\s*$", line)
                if example_match and len(legacy_example_lines) < min(3, bullet_max):
                    example_item = example_match.group(1)
                    if original_segment:
                        count = int(re.search(r"x\s+(\d+)\s*$", line).group(1))
                        example_item += f" x {min(2, count)}"
                    legacy_example_lines.append(f"- {example_item}")
            if not legacy_example_lines and attack_option:
                legacy_example_lines.append(f"- {attack_option}")
            bo_todo_str = '\n'.join(legacy_lines) if legacy_lines else 'none'
            legacy_example = '\n'.join(legacy_example_lines)
            prompt_template = legacy_question_prompt
            prompt_system = legacy_system_prompt.format(segment_instruction=(
                original_segment_instruction if original_segment
                else legacy_segment_instruction.format(bullet_max=bullet_max)))
        else:
            bo_todo_str = '\n'.join(bo_todo_) if bo_todo_ else 'none'
            prompt_template = numbered_question_prompt
            prompt_system = numbered_system_prompt

        instruction = instruction_prompt % (
            strategy['phase_summary'],
            strategy['timing'],
            strategy['instruction'],
            strategy['attack_instruction']
        )
        if planner_protocol == "legacy":
            query = prompt_template % (
                prompt_system, instruction, obs, bo_todo_str, legacy_example
            )
        else:
            query = prompt_template % (prompt_system, instruction, obs, bo_todo_str)
        if original_segment and os.getenv("SC2_CLARIFY_BO_QUOTA", "0") == "1":
            query = clarify_bo_quota(query)
        if not original_segment:
            query += (
                f"\nAdditional constraint: output at most {bullet_max} bullet lines for this single near-term segment. "
                f"Output at most {segment_max} executable actions in total. "
                "Each bullet must contain only one exact current-list name; repeat bullets for repeated actions. "
                "Do not use `x N` or try to finish the whole phase in one answer.\n"
            )
        # query = '\nYou are an expert in StarCraft II. We now need you to analyze the current situation based on the present observation and a given phased strategic instruction.\nIn fact, you may need to complete the given phased strategy instructions segment by segment. ou will also be provided with a build list that remains to be build in this phase, to help you decide what to continue building next. \n1. Based on the above instructions, please provide the next segment\'s BO list (approximately 15 entries in length) along with detailed attack instructions. IMPORTANT: Note that the "15 entries" requirement refers to the number of lines/entries in your output list, NOT the total count of units. Even if a single entry involves building multiple units or structures, it counts as only one entry towards the 15-entry target. A single attack instructions is one entry.\n2. As for attack instructions, use format like `Launch OFFENSE with 2 Marine, 1 Marauder`. You should consider when to start an attack and how many units should involved.\n## Optional Operations/Units:\nFor your build order, below is a list of all available units/buildings/upgrade objects you can use. You may only select corresponding units/buildings/upgrade objects from this list:\nsupplydepot, barracks, commandcenter, marine, orbitalcommand, refinery, engineeringbay, planetaryfortress, factory, techlab factory, siegetank, liberator, raven, missileturret, techlab barracks, terraninfantryweaponslevel1, widowmine, fusioncore, terranstructurearmorupgrade, thor, marauder, concussiveshells, hi-secautotracking, mule, starport, reactor barracks, vikingfighter, techlab starport, combatshield, stimpack, armory, ghostacademy, terraninfantryarmorlevel1, ghost, terraninfantryweaponslevel2, personalcloaking, banshee, bansheecloakingfield, terraninfantryarmorlevel2, medivac, terraninfantryweaponslevel3, terraninfantryarmorlevel3, turret, bunker, reaper, hellion, battlecruiser, reactor starport.\n\nNow the phased strategic instructions are:\n- Current Stage: Rapid Expansion and Defense Setup (00:00 - 03:03) \n- Detailed Instruction: Prioritize early Command Centers for economic advantage and convert one into Planetary Fortress for defensive security. Establish Barracks with Tech Labs to enable Marauder production and Concussive Shells research. Build Engineering Bay for future upgrades and Refineries to support tech transitions.\n- Attacking Instruction: No attacks in this phase; focus on economic foundation and initial unit production.\n\nCurrent observations are:\nIt is now 03:11 in the game, and we are playing as Terran with 20 minerals and 464 vespene gas, while our current supply is 19, of which 1 is allocated to the army, 17 to workers, and 19 remains unused on a 128x128 sized map. \nIn terms of our military forces, we have 15 SCV. In terms of the structures, we have 2 bases in the form of commandcenter, 3 barracks, 2 refinery, 1 supplydepot. \nIn terms of research, we have researched no research currently.\nAs for the visible enemy presence, there are no units currently. As for the visible enemy structures, there are 2 bases in the form of commandcenter, 1 barracks, 1 supplydepot.\nThe remaining BO list in this phase:\n- SCV x 5\n- orbitalcommand x 1\n- refinery x 1\n- techlab barracks x 1\nPlease give your answer following the example format:\n\n<answer>\nHere are the Build Orders with attack instructions:\n- supplydepot\n- marine x 2\n- ...\n- Launch OFFENSE with 2 Marine, 1 Marauder\n- ...\n</answer>\n\n**The output format is not optional and must be followed precisely. Do not add any other text outside of these tags.**\n'
        # query = "Do you know alice and bob?"
        if execution_feedback is not None:
            # Do not feed plan counts into observation-based prerequisite parsing.
            query += (
                "\nExecution feedback for the previous segment (not owned units/buildings):\n"
                + json.dumps(execution_feedback, ensure_ascii=False)
                + "\nThe failed target and unsubmitted suffix were not submitted as production commands. "
                "Only recorded selection quotas were released; previously processed commands are not cancelled. "
                "These are not additional tasks or required retries. Use the current observation and "
                "remaining BO list above to choose the next segment, in the same answer format.\n"
            )
        print(f"Generate new plan... [{self.ct}]")
        self.ct += 1
        bo_ori = self.get_action(
            query, reward=reward, done=done, attack_ct=attack_ct, telemetry=telemetry
        )

        def check_format(text):
            """
            判断字符串是否符合格式要求：
            - 第一行必须为 "Here are the Build Orders with attack instructions:"
            - 后续每一行必须以 "- " 开头（即行首为连字符加空格）
            - 条目行数（即后续行数）不超过15
            - 不能有任何其他行（包括空行）
            """
            target_first_line = "Here are the Build Orders with attack instructions:"
            lines = text.splitlines()  # 按行分割，保留原始内容（不含换行符）

            # 至少有一行
            if not lines:
                return False

            # 检查第一行
            if lines[0] != target_first_line:
                return False

            # 检查后续每一行是否以 "- " 开头
            for line in lines[1:]:
                if not line.startswith("- "):
                    return False

            # 检查条目数量（后续行数）是否超过15
            if len(lines) - 1 > 15:
                return False

            return True

        def truncate_entries(text, max_entries=15):
            """
            将字符串中的条目（以 '-' 开头的行）截断至最多 max_entries 个。
            非条目行（如标题、空行等）全部保留。
            """
            lines = text.splitlines()          # 按行分割，保留原内容（不含换行符）
            result = []
            count = 0

            exceed = False

            for line in lines:
                # 去除行首空白后判断是否以 '-' 开头（避免缩进影响）
                if line.lstrip().startswith(('-', '*')):
                    if count < max_entries:     # 前15个条目保留
                        result.append(line)
                        count += 1
                    # 超过的条目直接跳过
                    else:
                        exceed = True
                else:
                    result.append(line)         # 非条目行始终保留

            return exceed, '\n'.join(result)            # 重新组合成字符串

        print(f"BO ori: {bo_ori}")
        bo_ori = bo_ori.split("</think>")[-1]
        # correct_format = check_format(bo_ori)
        attack_choice_available = bool(attack_option)
        raw_attack_orders = extract_attack_orders(bo_ori) if attack_allowed and attack_choice_available else []
        self.original_segment_active = (original_segment and self.correct_format
                                        and os.getenv("SC2_PRESERVE_VALID_BO", "0") == "1")
        if self.original_segment_active:
            exceed, bo = False, bo_ori
        else:
            exceed, bo = truncate_entries(bo_ori, max_entries=bullet_max)
        attack_orders = raw_attack_orders or (
            extract_attack_orders(bo) if attack_allowed and attack_choice_available else []
        )
        try:
            max_forced_attacks = int(os.getenv("SC2_MAX_FORCED_ATTACKS", "3"))
        except ValueError:
            max_forced_attacks = 3
        if attack_allowed and not attack_orders and attack_ct < max_forced_attacks:
            attack_orders = [default_attack_order(strategy.get("attack_instruction", ""))]

        parsed_bo = parse_build_order(
            bo,
            choice_map=choice_map,
            strict_names_only=(planner_protocol == "legacy" and LEGACY_STRICT_NAMES_ONLY),
            original_segment=self.original_segment_active,
        )
        parsed_attack_orders = []
        non_attack_bo = []
        for item in parsed_bo:
            attack_order = normalize_attack_order(item)
            if attack_order:
                parsed_attack_orders.append(attack_order)
            else:
                non_attack_bo.append(item)
        if attack_choice_available:
            attack_orders = (parsed_attack_orders if self.original_segment_active
                             else attack_orders or parsed_attack_orders)
        bo = non_attack_bo
        # Opt-in ablation: trust the existing whole-answer validator, not a
        # valid prefix salvaged from an invalid answer. Keep attack handling below.
        if (os.getenv("SC2_PRESERVE_VALID_BO", "0") == "1"
                and planner_protocol == "legacy" and self.correct_format):
            used_fallback = False
        else:
            bo, used_fallback = constrain_bo_to_todo(
                bo,
                bo_todo_sequence,
                max_items=segment_max,
                attack_focus=attack_allowed,
            )
        if attack_orders:
            attack_position = os.getenv("SC2_ATTACK_ORDER_POSITION", "after_build").strip().lower()
            build_segment = bo if self.original_segment_active else bo[:max(0, segment_max - 1)]
            if attack_position == "before_build":
                bo = [attack_orders[0]] + build_segment
            else:
                bo = build_segment + [attack_orders[0]]
        if used_fallback:
            print("BO fallback to remaining list.")
        if exceed:
            print("Max entries exceeded.")

        self.exceed = exceed

        # attack = get_attack(answer)

        # if len(attack):
        #     bo = evenly_insert(bo, attack)
        bo_ori_log = bo_ori.replace(chr(10), chr(44))
        print(f"BO ori: {bo_ori_log}, BO: {bo}")
        print(f"BO choices: {format_choice_map(choice_map)}")

        bo_attack = []
        for i in bo:
            if normalize_attack_order(i):
                i = 'Attacking Instruction: ' + i
            bo_attack.append(i)

        bo = bo_attack
        trace = getattr(self, "command_trace", None)
        if trace is not None:
            trace.planned(
                [('Attacking Instruction: ' + item) if normalize_attack_order(item) else item
                 for item in parsed_bo], bo,
                passthrough=(os.getenv("SC2_PRESERVE_VALID_BO", "0") == "1"
                             and planner_protocol == "legacy" and self.correct_format),
            )
        return bo
