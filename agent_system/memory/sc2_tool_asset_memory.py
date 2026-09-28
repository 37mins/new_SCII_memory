import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .sc2_compact_asset_memory import (
    SC2CompactAssetMemory,
    _action_tags,
    _as_bool,
    _as_float,
    _dedup_keep_order,
    _diagnostic_text,
    _fact_tags,
    _shorten,
    _to_text,
)
from .sc2_reflection_distiller import SC2ReflectionDistiller
from .strategic_memory_v2 import (
    extract_sc2_facts as _base_extract_sc2_facts,
    infer_failure_reason,
    is_attack_action,
    is_unit_train_action,
)


SC2_MEMORY_TOOL_NAMES = {"query_skill", "query_graph_skill", "query_world_model"}
_EXPLANATION_MARKERS = (
    "given the current",
    "breakdown",
    "recommend",
    "recommendation",
    "strategy",
    "situation",
    "here are",
    "here's",
    "we should",
    "you should",
    "because",
    "therefore",
)


def _load_json(path: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return json.loads(json.dumps(fallback))
    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return json.loads(json.dumps(fallback))


def _save_json(path: str, data: Dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def _append_jsonl(path: str, item: Dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")


def _slug(text: str, prefix: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text or "").strip().lower()).strip("._-")
    return f"{prefix}_{clean[:96] or 'asset'}"
_ACTION_STARTERS = (
    "build",
    "train",
    "produce",
    "construct",
    "make",
    "add",
    "expand",
    "attack",
    "launch",
    "offense",
    "scv",
    "worker",
    "marine",
    "marauder",
    "siegetank",
    "siege tank",
    "supplydepot",
    "supply depot",
    "barracks",
    "refinery",
    "factory",
    "engineeringbay",
    "engineering bay",
    "commandcenter",
    "command center",
    "bunker",
    "missileturret",
    "missile turret",
)
_LOW_LEVEL_ACTIONS = (
    ("BARRACKSTRAIN_MARINE", "marine"),
    ("BARRACKSTRAIN_MARAUDER", "marauder"),
    ("COMMANDCENTERTRAIN_SCV", "scv"),
    ("ORBITALCOMMANDTRAIN_SCV", "scv"),
    ("TERRANBUILD_SUPPLYDEPOT", "supplydepot"),
    ("TERRANBUILD_BARRACKS", "barracks"),
    ("TERRANBUILD_REFINERY", "refinery"),
    ("TERRANBUILD_FACTORY", "factory"),
    ("TERRANBUILD_ENGINEERINGBAY", "engineeringbay"),
    ("TERRANBUILD_COMMANDCENTER", "commandcenter"),
    ("TERRANBUILD_BUNKER", "bunker"),
    ("TERRANBUILD_MISSILETURRET", "missileturret"),
    ("FACTORYTRAIN_SIEGETANK", "siegetank"),
    ("ATTACK", "Launch OFFENSE with 8 marine"),
)


def _tokens(text: Any) -> set:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_>=.-]+", _to_text(text))
        if len(token) >= 2
    }


def _append_action(actions: List[str], action: str, max_actions: int) -> None:
    action = _to_text(action)
    if not action or len(actions) >= max_actions:
        return
    if actions and actions[-1] == action:
        return
    actions.append(action)


def _split_action_segments(text: str) -> List[str]:
    normalized = (
        text.replace("\r\n", "\n")
        .replace("->", "\n")
        .replace(";", "\n")
        .replace(",", "\n")
    )
    return [part.strip() for part in normalized.splitlines() if part.strip()]


def _strip_action_prefix(segment: str) -> str:
    cleaned = re.sub(r"^[\s\-*#>\d.)]+", "", segment.strip())
    cleaned = cleaned.strip("`'\" ")
    if ":" in cleaned:
        prefix, rest = cleaned.split(":", 1)
        if prefix.strip().lower() in _ACTION_STARTERS:
            return prefix.strip()
        lower_rest = rest.lower()
        if any(marker in lower_rest for marker in _EXPLANATION_MARKERS):
            return prefix.strip()
    return cleaned


def _starts_like_action(lower: str) -> bool:
    return lower.startswith(_ACTION_STARTERS)


def _canonical_action_from_low_level(text: str) -> List[str]:
    upper = text.upper()
    actions = []
    for token, action in _LOW_LEVEL_ACTIONS:
        if token in upper:
            actions.append(action)
    return _dedup_keep_order(actions)


def _canonicalize_sc2_action_segment(segment: str) -> str:
    cleaned = _strip_action_prefix(segment)
    if not cleaned:
        return ""
    lower = cleaned.lower()
    low_level = _canonical_action_from_low_level(cleaned)
    if low_level:
        return low_level[0]

    if any(marker in lower for marker in _EXPLANATION_MARKERS) and not _starts_like_action(lower):
        return ""
    if len(lower) > 80 and not _starts_like_action(lower):
        return ""

    if "launch offense" in lower or "attacking instruction" in lower or re.search(r"\battack\b", lower):
        unit = "marine"
        if "marauder" in lower:
            unit = "marauder"
        elif "siege" in lower or "tank" in lower:
            unit = "siegetank"
        count_match = re.search(r"\b(\d{1,3})\s*(?:x\s*)?(marine|marauder|siege\s*tank|siegetank|tank)?", lower)
        count = count_match.group(1) if count_match else "8"
        return f"Launch OFFENSE with {count} {unit}"

    aliases = (
        (("supplydepot", "supply depot"), "supplydepot"),
        (("commandcenter", "command center", "expand", "expansion"), "commandcenter"),
        (("engineeringbay", "engineering bay"), "engineeringbay"),
        (("missileturret", "missile turret"), "missileturret"),
        (("siegetank", "siege tank"), "siegetank"),
        (("barracks", "rax"), "barracks"),
        (("refinery",), "refinery"),
        (("factory",), "factory"),
        (("bunker",), "bunker"),
        (("marauder",), "marauder"),
        (("marine",), "marine"),
        (("scv", "worker"), "scv"),
    )
    action_like = _starts_like_action(lower) or len(lower) <= 35
    if not action_like:
        return ""
    for names, action in aliases:
        if any(re.search(rf"\b{re.escape(name)}\b", lower) for name in names):
            return action
    return ""


def extract_legal_sc2_actions(action: Any, max_actions: int = 5) -> List[str]:
    actions: List[str] = []

    def visit(value: Any) -> None:
        if len(actions) >= max_actions:
            return
        if isinstance(value, dict):
            for key in ("action", "ability", "ability_id", "name", "bo", "build_order"):
                if key in value:
                    visit(value[key])
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)
                if len(actions) >= max_actions:
                    break
            return

        text = _to_text(value)
        if not text:
            return
        for low_level_action in _canonical_action_from_low_level(text):
            _append_action(actions, low_level_action, max_actions)
        for segment in _split_action_segments(text):
            canonical = _canonicalize_sc2_action_segment(segment)
            _append_action(actions, canonical, max_actions)

    visit(action)
    return actions[:max_actions]


def extract_tool_sc2_facts(obs: Any) -> Dict[str, int]:
    text = _to_text(obs)
    facts = dict(_base_extract_sc2_facts(text))
    time_match = re.search(r"(?:game\s*time|time|it\s*is\s*now|time_formatted)\s*[:=]?\s*(\d{1,2}):(\d{2})", text, re.IGNORECASE)
    if time_match:
        facts["time_s"] = int(time_match.group(1)) * 60 + int(time_match.group(2))
    patterns = (
        ("minerals", r"\bminerals?\b\s*[:=]\s*(\d+)"),
        ("vespene", r"\b(?:vespene|gas)\b\s*[:=]\s*(\d+)"),
        ("army", r"\b(?:army(?:\s+supply)?|army_supply|supply_army)\b\s*[:=]\s*(\d+)"),
        ("workers", r"\b(?:workers?|worker_count|worker\s+count|supply_workers)\b\s*[:=]\s*(\d+)"),
        ("supply_left", r"\b(?:supply[_\s-]*left|free\s+supply|remaining\s+supply)\b\s*[:=]\s*(\d+)"),
        ("enemy_units", r"\b(?:enemy[_\s-]*units|n_visible_enemy_units)\b\s*[:=]\s*(\d+)"),
        ("enemy_structures", r"\b(?:enemy[_\s-]*structures|n_visible_enemy_structures)\b\s*[:=]\s*(\d+)"),
    )
    for key, pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            facts[key] = int(match.group(1))
    return facts


def format_tool_sc2_facts(facts: Dict[str, int]) -> str:
    if not facts:
        return "unknown"
    parts = []
    if "time_s" in facts:
        parts.append(f"time={facts['time_s'] // 60:02d}:{facts['time_s'] % 60:02d}")
    for key in ("minerals", "vespene", "army", "workers", "supply_left", "enemy_units", "enemy_structures"):
        if key in facts:
            parts.append(f"{key}={facts[key]}")
    return ", ".join(parts)


def _generalized_tool_condition(
    facts: Dict[str, int],
    tags: List[str] = None,
    action: Any = "",
    attack_ready_min_army: int = 8,
) -> str:
    tags = set(tags or [])
    parts = []
    supply_left = facts.get("supply_left")
    minerals = facts.get("minerals")
    army = facts.get("army")
    workers = facts.get("workers")
    if supply_left is not None:
        if supply_left <= 0:
            parts.append("supply_left=0")
        elif supply_left <= 1:
            parts.append("supply_left<=1")
        elif "production" in tags:
            parts.append("supply_left>0")
    if army is not None:
        if army < attack_ready_min_army:
            parts.append(f"army < {attack_ready_min_army}")
        else:
            parts.append(f"army >= {attack_ready_min_army}")
    lower_action = _to_text(action).lower()
    if "barracks" in tags or "barracks" in lower_action:
        parts.append("minerals >= 150 for barracks")
    elif minerals is not None:
        if minerals < 50:
            parts.append("minerals < 50")
        elif minerals < 150 and ("resource_rule" in tags or "minerals_low" in tags):
            parts.append("minerals < 150")
        elif minerals >= 300:
            parts.append("minerals >= 300")
    if workers is not None and workers < 18:
        parts.append("workers < 18")
    return ", ".join(_dedup_keep_order(parts)) or "general SC2 decision state"


def _normalize_reflection_condition(condition: Any, facts: Dict[str, int], tags: List[str], action: Any = "") -> str:
    text = _to_text(condition)
    text = re.sub(r"(?:^|,\s*)time\s*=\s*\d{1,2}:\d{2}\s*,?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*,\s*,", ",", text).strip(" ,")
    if text and len(text) <= 120:
        return text
    return _generalized_tool_condition(facts, tags, action)


def _is_specific_failure_reason(reason: Any) -> bool:
    lower = _to_text(reason).lower()
    if not lower:
        return False
    generic = (
        "episode ended without win",
        "invalid action",
        "unknown",
        "none",
        "no win",
        "failed",
    )
    if lower in generic:
        return False
    markers = (
        "insufficient",
        "supply",
        "too early",
        "army",
        "blocked",
        "cannot",
        "unavailable",
        "not enough",
        "requires",
        "timeout",
    )
    return any(marker in lower for marker in markers)


def _is_generic_terminal_failure(reason: Any) -> bool:
    lower = _to_text(reason).lower()
    return not lower or lower in {"episode ended without win", "invalid action", "unknown", "none"}


def _ordinary_single_action(action: Any) -> bool:
    if isinstance(action, (list, tuple)):
        canonical = [str(item).strip() for item in action if str(item).strip()][:2]
    else:
        canonical = extract_legal_sc2_actions(action, max_actions=2)
    if len(canonical) != 1:
        return False
    lower = canonical[0].lower()
    return lower in {"scv", "marine", "supplydepot", "barracks", "refinery", "factory", "marauder", "siegetank"}


def _chain_actions(chain: Any, max_actions: int = 5) -> List[str]:
    if isinstance(chain, (list, tuple)):
        text = " -> ".join(str(item) for item in chain)
    else:
        text = _to_text(chain)
    return extract_legal_sc2_actions(text, max_actions=max_actions)


def _has_attack(actions: List[str]) -> bool:
    return any("offense" in action.lower() or "attack" in action.lower() for action in actions)


def _meaningful_do_chain(actions: List[str]) -> bool:
    if len(actions) == 1 and _has_attack(actions):
        return True
    if len(actions) >= 3 and len(set(action.lower() for action in actions)) >= 2:
        return True
    return len(actions) >= 2 and _has_attack(actions)


def _meaningful_avoid_chain(actions: List[str], failure_reason: Any, tags: List[str]) -> bool:
    if _is_specific_failure_reason(failure_reason):
        return True
    tag_set = set(tags or [])
    if tag_set & {"attack_too_early", "supply_block_rule", "resource_rule", "invalid_action", "supply_blocked"}:
        return True
    return len(actions) >= 2 and _has_attack(actions)


def _as_confidence(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _as_text_items(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        candidates = value.values()
    elif isinstance(value, (list, tuple, set)):
        candidates = value
    else:
        candidates = [value]
    return [text for item in candidates if (text := _to_text(item))]


def _as_reflection_items(value: Any, content_keys: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """Normalize loose LLM JSON fields into a list of reflection item dicts.

    The training model usually follows the requested schema, but it can emit a
    single object, a keyed object, or plain text. Plain text is intentionally
    dropped because it lacks confidence/conditions and tends to add noise.
    """
    if value is None:
        return []
    if isinstance(value, dict):
        if any(key in value for key in content_keys):
            return [value]
        items: List[Dict[str, Any]] = []
        for child in value.values():
            items.extend(_as_reflection_items(child, content_keys))
        return items
    if isinstance(value, (list, tuple, set)):
        items = []
        for child in value:
            items.extend(_as_reflection_items(child, content_keys))
        return items
    return []


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    cleaned = _to_text(text)
    if not cleaned:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1)
    tagged = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", cleaned, re.DOTALL | re.IGNORECASE)
    if tagged:
        cleaned = tagged.group(1)
    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(cleaned)):
        char = cleaned[idx]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(cleaned[start : idx + 1])
                except Exception:
                    return None
                return value if isinstance(value, dict) else None
    return None


def parse_sc2_memory_tool_call(text: Any) -> Tuple[Optional[Dict[str, Any]], str]:
    raw = _to_text(text)
    obj = _extract_json_object(raw)
    if obj is None:
        match = re.search(r"\b(query_skill|query_graph_skill|query_world_model)\s*\((.*?)\)", raw, re.DOTALL)
        if not match:
            return None, "missing tool call JSON"
        query_match = re.search(r"query\s*=\s*['\"]([^'\"]+)['\"]", match.group(2), re.DOTALL)
        query = query_match.group(1).strip() if query_match else match.group(2).strip()
        obj = {"name": match.group(1), "arguments": {"query": query}}

    name = str(obj.get("name") or obj.get("tool") or obj.get("tool_name") or "").strip()
    if name not in SC2_MEMORY_TOOL_NAMES:
        return None, f"unsupported memory tool: {name or 'empty'}"
    args = obj.get("arguments", obj.get("args", {}))
    if not isinstance(args, dict):
        args = {}
    query = str(args.get("query") or obj.get("query") or "").strip()
    if not query:
        return None, "missing query"
    return {"name": name, "arguments": {**args, "query": query}}, ""


def is_sc2_memory_tool_call(text: Any) -> bool:
    raw = _to_text(text).lower()
    return "<tool_call>" in raw or any(f"{name}(" in raw or f'"{name}"' in raw for name in SC2_MEMORY_TOOL_NAMES)


class SC2ToolAssetMemory(SC2CompactAssetMemory):
    """AGIAgent-style tool-call retrieval memory for SC2.

    Assets are still written from environment transitions, but retrieval is
    exposed through explicit memory tools instead of unconditional prompt
    prefixing.
    """

    def __init__(
        self,
        *args,
        seed_static_assets: bool = True,
        online_update: bool = False,
        compact_assets_path: str = "",
        write_compact_assets_path: str = "",
        evidence_audit_path: str = "",
        asset_links_path: str = "",
        freeze_retrieval_after_init: bool = True,
        dynamic_quality_threshold: float = 0.8,
        reflection_enabled: bool = False,
        reflection_model: str = "",
        reflection_api_key: str = "",
        reflection_api_base: str = "",
        reflection_log_path: str = "",
        reflection_scene_log_path: str = "",
        reflection_min_quality: float = 0.65,
        reflection_fallback_to_rules: bool = True,
        curiosity_write_threshold: float = 0.45,
        curiosity_reflect_threshold: float = 0.65,
        curiosity_event_window: int = 12,
        curiosity_max_reflections_per_episode: int = 8,
        training_model_reflection: bool = False,
        training_reflection_max_items_per_step: int = 8,
        attack_ready_min_army: int = 8,
        **kwargs,
    ):
        self.online_update = _as_bool(online_update)
        self.compact_assets_path = compact_assets_path or ""
        self.write_compact_assets_path = write_compact_assets_path or self.compact_assets_path
        self.evidence_audit_path = evidence_audit_path or ""
        self.asset_links_path = asset_links_path or ""
        self.freeze_retrieval_after_init = _as_bool(freeze_retrieval_after_init)
        self.dynamic_quality_threshold = float(dynamic_quality_threshold)
        self.reflection_min_quality = float(reflection_min_quality)
        self.reflection_fallback_to_rules = _as_bool(reflection_fallback_to_rules)
        self.reflection_scene_log_path = reflection_scene_log_path or ""
        self.curiosity_write_threshold = float(curiosity_write_threshold)
        self.curiosity_reflect_threshold = float(curiosity_reflect_threshold)
        self.curiosity_event_window = int(curiosity_event_window)
        self.curiosity_max_reflections_per_episode = int(curiosity_max_reflections_per_episode)
        self.training_model_reflection = _as_bool(training_model_reflection)
        self.training_reflection_max_items_per_step = int(training_reflection_max_items_per_step)
        self.attack_ready_min_army = max(0, int(attack_ready_min_army or 0))
        self.reflection_distiller = SC2ReflectionDistiller(
            enabled=reflection_enabled,
            model=reflection_model,
            api_key=reflection_api_key,
            api_base=reflection_api_base,
            log_path=reflection_log_path,
        )
        if self.reflection_scene_log_path:
            os.makedirs(os.path.dirname(self.reflection_scene_log_path) or ".", exist_ok=True)
        self._persisting_seed_assets = False
        self._retrieval_asset_keys = None
        self._recent_events: List[List[Dict[str, Any]]] = []
        self._reflections_this_episode: List[int] = []
        self._pending_training_reflection_scenes: List[List[Dict[str, Any]]] = []
        self._active_training_reflection_scenes: List[Optional[Dict[str, Any]]] = []
        self._seen_scene_keys = set()
        super().__init__(*args, **kwargs)
        self._load_persistent_compact_assets()
        if seed_static_assets:
            self._persisting_seed_assets = True
            self._seed_static_assets()
            self._persisting_seed_assets = False
        if self.freeze_retrieval_after_init:
            self._freeze_retrieval_assets()

    def store(self, record: Dict[str, List[Any]]):
        if self.keys is None:
            self.keys = list(record.keys())
        assert self.keys == list(record.keys())
        if not self.online_update:
            return
        for env_idx in range(self.batch_size):
            rec = {key: record[key][env_idx] for key in self.keys}
            self._step_counts[env_idx] += 1
            step_num = self._step_counts[env_idx]
            self._remember_event(env_idx, step_num, rec)
            for asset in self._make_assets(env_idx, step_num, rec):
                stored_asset = self._upsert_asset(asset)
                if stored_asset is not None:
                    self._write_asset_log(env_idx, step_num, stored_asset)
            if _as_bool(rec.get("done")):
                self._recent_events[env_idx] = []
                self._recent_chains[env_idx] = []
                self._reflections_this_episode[env_idx] = 0

    def reset(self, batch_size: int):
        super().reset(batch_size)
        self._recent_events = [[] for _ in range(batch_size)]
        self._reflections_this_episode = [0 for _ in range(batch_size)]
        self._pending_training_reflection_scenes = [[] for _ in range(batch_size)]
        self._active_training_reflection_scenes = [None for _ in range(batch_size)]

    def _load_persistent_compact_assets(self) -> None:
        if not self.compact_assets_path:
            return
        doc = _load_json(self.compact_assets_path, {"meta": {"version": 0}, "world_facts": [], "action_skills": []})
        for item in doc.get("world_facts", []) or []:
            asset = self._internal_asset_from_compact(item, kind="world_fact")
            if asset:
                super()._upsert_asset(asset)
        for item in doc.get("action_skills", []) or []:
            asset = self._internal_asset_from_compact(item, kind="action_skill")
            if asset:
                super()._upsert_asset(asset)

    def _freeze_retrieval_assets(self) -> None:
        self._retrieval_asset_keys = {asset.get("key") for asset in self._assets if asset.get("key")}

    def _upsert_asset(self, asset: Dict[str, Any]):
        stored = super()._upsert_asset(asset)
        if stored is not None and self.write_compact_assets_path:
            compact_asset = dict(stored)
            compact_asset["count"] = int(asset.get("count", 1) or 1)
            source = asset.get("created_by") or ("seed_static" if self._persisting_seed_assets else "online_transition")
            self._upsert_compact_asset(compact_asset, source=source)
        return stored

    def _internal_asset_from_compact(self, item: Dict[str, Any], kind: str) -> Dict[str, Any]:
        if not isinstance(item, dict):
            return {}
        if kind == "world_fact":
            text = item.get("fact") or item.get("text") or ""
            polarity = "fact"
            action_chain = ""
        else:
            text = item.get("intent") or item.get("text") or item.get("observed_effect") or ""
            polarity = str(item.get("skill_polarity") or item.get("polarity") or "do").strip().lower() or "do"
            chain = item.get("action_chain") or []
            if isinstance(chain, str):
                action_chain = chain
            else:
                action_chain = " -> ".join(str(part).strip() for part in chain if str(part).strip())
        tags = [str(tag).strip() for tag in (item.get("tags") or []) if str(tag).strip()]
        asset = {
            "kind": kind,
            "polarity": polarity,
            "text": _shorten(text, 160),
            "condition": _shorten(item.get("condition", ""), 120),
            "tags": _dedup_keep_order(tags),
            "quality": float(item.get("quality", self.quality_threshold) or self.quality_threshold),
            "count": int(item.get("count", len(item.get("evidence_ids", []) or []) or 1) or 1),
            "action_chain": _shorten(action_chain, 180),
            "failure_signals": [_shorten(signal, 80) for signal in (item.get("failure_signals") or []) if signal],
            "last_step": int(item.get("last_step", 0) or 0),
            "created_by": str(item.get("created_by") or ""),
        }
        asset["key"] = self._asset_key(asset)
        return asset

    def _compact_doc(self) -> Dict[str, Any]:
        return _load_json(
            self.write_compact_assets_path,
            {
                "meta": {
                    "version": 0,
                    "source": "verl_agent_sc2_tool_asset_memory",
                    "schema": "agiagent_compact_assets_v1",
                },
                "world_facts": [],
                "action_skills": [],
            },
        )

    def _upsert_compact_asset(self, asset: Dict[str, Any], source: str) -> None:
        if not self.write_compact_assets_path:
            return
        doc = self._compact_doc()
        doc.setdefault("meta", {})
        doc["meta"].setdefault("source", "verl_agent_sc2_tool_asset_memory")
        doc["meta"].setdefault("schema", "agiagent_compact_assets_v1")
        if asset.get("kind") == "world_fact":
            section = "world_facts"
            entry = self._asset_to_world_fact(asset, source)
        else:
            section = "action_skills"
            entry = self._asset_to_action_skill(asset, source)
        items = doc.setdefault(section, [])
        merged = False
        for existing in items:
            if existing.get("merge_key") == entry.get("merge_key") or existing.get("id") == entry.get("id"):
                self._merge_compact_entry(existing, entry)
                merged = True
                self._audit_compact_asset(asset, existing, "merged", source)
                break
        if not merged:
            existing_ids = {str(item.get("id") or "") for item in items}
            base_id = entry.get("id") or _slug(entry.get("merge_key", ""), "asset")
            unique_id = base_id
            suffix = 2
            while unique_id in existing_ids:
                unique_id = f"{base_id}_{suffix}"
                suffix += 1
            entry["id"] = unique_id
            items.append(entry)
            self._audit_compact_asset(asset, entry, "created", source)
        doc["meta"]["version"] = int(doc["meta"].get("version", 0) or 0) + 1
        doc["meta"]["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        doc["meta"]["world_fact_count"] = len(doc.get("world_facts", []) or [])
        doc["meta"]["action_skill_count"] = len(doc.get("action_skills", []) or [])
        _save_json(self.write_compact_assets_path, doc)
        self._save_empty_links_doc()

    def _asset_to_world_fact(self, asset: Dict[str, Any], source: str) -> Dict[str, Any]:
        merge_key = self._asset_key(asset)
        evidence_id = f"sc2_{source}_{int(asset.get('last_step', 0) or 0)}"
        return {
            "id": _slug(merge_key, "fact"),
            "asset_type": "world_fact",
            "merge_key": merge_key,
            "condition": asset.get("condition", ""),
            "fact": asset.get("text", ""),
            "fact_type": "sc2_rule" if source == "seed_static" else "verified_transition",
            "evidence": asset.get("failure_signals", [])[:4] or [asset.get("condition", "")],
            "evidence_ids": [evidence_id],
            "tags": asset.get("tags", [])[:12],
            "quality": float(asset.get("quality", 0.0) or 0.0),
            "quality_rationale": asset.get("quality_rationale")
            or ("Seeded SC2 rule." if source == "seed_static" else "Derived from SC2 transition feedback."),
            "count": int(asset.get("count", 1) or 1),
            "created_by": source,
            "last_step": int(asset.get("last_step", 0) or 0),
        }

    def _asset_to_action_skill(self, asset: Dict[str, Any], source: str) -> Dict[str, Any]:
        merge_key = self._asset_key(asset)
        chain = extract_legal_sc2_actions(asset.get("action_chain", ""), max_actions=self.max_chain_actions)
        polarity = str(asset.get("polarity") or "do").strip().lower() or "do"
        evidence_id = f"sc2_{source}_{int(asset.get('last_step', 0) or 0)}"
        return {
            "id": _slug(merge_key, "skill"),
            "asset_type": "action_skill",
            "merge_key": merge_key,
            "intent": asset.get("text", ""),
            "condition": asset.get("condition", ""),
            "preconditions": asset.get("preconditions") or self._preconditions_from_asset(asset),
            "action_chain": chain,
            "observed_effect": asset.get("observed_effect") or self._observed_effect_from_asset(asset),
            "effect": asset.get("observed_effect") or self._observed_effect_from_asset(asset),
            "failure_signals": asset.get("failure_signals", [])[:6],
            "transfer_scope": "local",
            "skill_polarity": polarity,
            "skill_type": self._skill_type(asset),
            "usage_warning": asset.get("usage_warning") or self._usage_warning_from_asset(asset),
            "quality": float(asset.get("quality", 0.0) or 0.0),
            "quality_rationale": asset.get("quality_rationale")
            or ("Seeded SC2 rule." if source == "seed_static" else "Derived from short legal SC2 action chain."),
            "evidence_ids": [evidence_id],
            "tags": asset.get("tags", [])[:12],
            "count": int(asset.get("count", 1) or 1),
            "created_by": source,
            "last_step": int(asset.get("last_step", 0) or 0),
        }

    def _preconditions_from_asset(self, asset: Dict[str, Any]) -> List[str]:
        tags = set(asset.get("tags", []) or [])
        preconditions = []
        if "attack" in tags:
            preconditions.append(f"army >= {self.attack_ready_min_army} before attack")
        if "production" in tags or "marine" in tags:
            preconditions.append("supply_left > 0 for unit production")
        if "barracks" in tags:
            preconditions.append("minerals >= 150 for barracks")
        return preconditions[:4]

    def _observed_effect_from_asset(self, asset: Dict[str, Any]) -> str:
        polarity = str(asset.get("polarity") or "")
        if polarity == "avoid":
            return "Avoids known SC2 failure boundary."
        if polarity == "diagnostic":
            return "Provides decision precheck for current SC2 state."
        if "win_pattern" in set(asset.get("tags", []) or []):
            return "Associated with winning SC2 transition."
        if "positive_reward" in set(asset.get("tags", []) or []):
            return "Associated with positive SC2 reward."
        return "Reusable SC2 action hint."

    def _usage_warning_from_asset(self, asset: Dict[str, Any]) -> str:
        if str(asset.get("polarity") or "") == "avoid":
            return "Negative boundary; do not execute as an action script."
        if str(asset.get("polarity") or "") == "diagnostic":
            return "Diagnostic only; use as a check before choosing an action."
        return "Use only if current observation satisfies preconditions."

    def _merge_compact_entry(self, existing: Dict[str, Any], incoming: Dict[str, Any]) -> None:
        existing["quality"] = max(float(existing.get("quality", 0.0) or 0.0), float(incoming.get("quality", 0.0) or 0.0))
        existing["count"] = int(existing.get("count", 1) or 1) + int(incoming.get("count", 1) or 1)
        existing["last_step"] = max(int(existing.get("last_step", 0) or 0), int(incoming.get("last_step", 0) or 0))
        for key in ("tags", "evidence", "evidence_ids", "failure_signals", "preconditions", "action_chain"):
            merged = []
            seen = set()
            for value in (existing.get(key, []) or []) + (incoming.get(key, []) or []):
                text = str(value).strip()
                marker = text.lower()
                if text and marker not in seen:
                    merged.append(value)
                    seen.add(marker)
            if merged:
                existing[key] = merged[:8]
        for key in ("condition", "fact", "intent", "observed_effect", "effect", "usage_warning", "quality_rationale"):
            if incoming.get(key) and float(incoming.get("quality", 0.0) or 0.0) >= float(existing.get("quality", 0.0) or 0.0):
                existing[key] = incoming[key]

    def _audit_compact_asset(self, internal_asset: Dict[str, Any], compact_entry: Dict[str, Any], status: str, source: str) -> None:
        if not self.evidence_audit_path:
            return
        _append_jsonl(
            self.evidence_audit_path,
            {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": status,
                "source": source,
                "asset_id": compact_entry.get("id", ""),
                "asset_type": compact_entry.get("asset_type", ""),
                "quality": compact_entry.get("quality", 0.0),
                "merge_key": compact_entry.get("merge_key", ""),
                "internal_asset": internal_asset,
            },
        )

    def _save_empty_links_doc(self) -> None:
        if not self.asset_links_path:
            return
        if os.path.exists(self.asset_links_path):
            return
        _save_json(self.asset_links_path, {"links": []})

    def _remember_event(self, env_idx: int, step_num: int, rec: Dict[str, Any]) -> None:
        obs_before = rec.get("text_obs")
        obs_after = rec.get("next_text_obs", "")
        action = rec.get("action")
        valid = _as_bool(rec.get("is_action_valid"))
        failure_reason = infer_failure_reason(action, obs_before, valid, rec.get("failure_reason"))
        actions = extract_legal_sc2_actions(action, max_actions=self.max_chain_actions)
        facts_before = extract_tool_sc2_facts(obs_before)
        facts_after = extract_tool_sc2_facts(obs_after)
        event = {
            "step_num": step_num,
            "action": _to_text(action),
            "canonical_actions": actions,
            "obs_before_excerpt": _shorten(obs_before, 700),
            "obs_after_excerpt": _shorten(obs_after, 700),
            "facts_before": facts_before,
            "facts_after": facts_after,
            "reward": _as_float(rec.get("reward")),
            "done": _as_bool(rec.get("done")),
            "won": _as_float(rec.get("won")),
            "is_action_valid": valid,
            "failure_reason": failure_reason,
            "tags": _dedup_keep_order(_fact_tags(facts_before) + _action_tags(" ".join(actions))),
        }
        events = self._recent_events[env_idx]
        events.append(event)
        if len(events) > self.curiosity_event_window:
            del events[: len(events) - self.curiosity_event_window]
        for action_text in actions:
            self._remember_action(env_idx, action_text)

    def _remember_action(self, env_idx: int, action: Any) -> None:
        chain = self._recent_chains[env_idx]
        for canonical_action in extract_legal_sc2_actions(action, max_actions=self.max_chain_actions):
            if chain and chain[-1] == canonical_action:
                continue
            chain.append(canonical_action)
            if len(chain) > self.max_chain_actions:
                del chain[: len(chain) - self.max_chain_actions]

    def _make_assets(self, env_idx: int, step_num: int, rec: Dict[str, Any]) -> List[Dict[str, Any]]:
        obs = rec.get("text_obs")
        next_obs = rec.get("next_text_obs", "")
        action = rec.get("action")
        reward = _as_float(rec.get("reward"))
        done = _as_bool(rec.get("done"))
        won = _as_float(rec.get("won"))
        valid = _as_bool(rec.get("is_action_valid"))
        failure_reason = infer_failure_reason(action, obs, valid, rec.get("failure_reason"))
        facts = extract_tool_sc2_facts(obs)
        next_facts = extract_tool_sc2_facts(next_obs)
        legal_actions = extract_legal_sc2_actions(action, max_actions=self.max_chain_actions)
        tags = _dedup_keep_order(_fact_tags(facts) + _action_tags(" ".join(legal_actions)))
        chain = self._chain_text(env_idx)

        assets = []
        curiosity = self._score_curiosity(
            env_idx=env_idx,
            facts=facts,
            next_facts=next_facts,
            action=action,
            legal_actions=legal_actions,
            reward=reward,
            done=done,
            won=won,
            valid=valid,
            failure_reason=failure_reason,
            tags=tags,
        )
        queued_training_reflection = False
        verified_outcome_assets = self._verified_positive_outcome_assets(
            step_num=step_num,
            facts=facts,
            action=action,
            tags=tags,
            chain=chain,
            done=done,
            won=won,
        )
        if curiosity["should_write"]:
            scene = self._reflection_scene(
                env_idx=env_idx,
                step_num=step_num,
                obs=obs,
                next_obs=next_obs,
                action=action,
                reward=reward,
                done=done,
                won=won,
                valid=valid,
                failure_reason=failure_reason,
                facts=facts,
                next_facts=next_facts,
                legal_actions=legal_actions,
                tags=tags,
                chain=chain,
                curiosity=curiosity,
            )
            if self._should_store_scene(scene):
                self._write_reflection_scene(scene)
            if curiosity["should_reflect"] and self.training_model_reflection:
                self._queue_training_reflection(env_idx, scene)
                queued_training_reflection = True
            elif curiosity["should_reflect"] and self.reflection_distiller.enabled:
                reflection = self.reflection_distiller.distill(scene)
                if reflection.get("status") == "success":
                    assets.extend(
                        self._assets_from_reflection(
                            reflection.get("reflection", {}),
                            scene=scene,
                            facts=facts,
                            fallback_tags=tags,
                            step_num=step_num,
                        )
                    )
                    if env_idx < len(self._reflections_this_episode):
                        self._reflections_this_episode[env_idx] += 1
                elif not self.reflection_fallback_to_rules:
                    return self._quality_gate_assets(
                        self._critical_rule_assets(step_num, facts, action, legal_actions, tags, chain, valid, failure_reason)
                    )

        if not assets:
            if queued_training_reflection and not self.reflection_fallback_to_rules:
                return self._quality_gate_assets(verified_outcome_assets)
            assets.extend(
                self._fallback_rule_assets(
                    step_num=step_num,
                    facts=facts,
                    action=action,
                    legal_actions=legal_actions,
                    tags=tags,
                    chain=chain,
                    reward=reward,
                    done=done,
                    won=won,
                    valid=valid,
                    failure_reason=failure_reason,
                )
            )
        else:
            assets.extend(self._critical_rule_assets(step_num, facts, action, legal_actions, tags, chain, valid, failure_reason))
            assets.extend(verified_outcome_assets)

        return self._quality_gate_assets(assets)

    def _should_reflect_scene(
        self,
        rec: Dict[str, Any],
        facts: Dict[str, int],
        failure_reason: str,
        valid: bool,
        done: bool,
        won: float,
        reward: float,
        action: Any,
    ) -> bool:
        del rec
        if done:
            return True
        if not valid and _is_specific_failure_reason(failure_reason):
            return True
        if is_attack_action(action) and facts.get("army", 99) < 8:
            return True
        if facts.get("supply_left") == 0 and is_unit_train_action(action):
            return True
        if _is_specific_failure_reason(failure_reason):
            return True
        return abs(float(reward)) >= 15.0 or won > 0

    def _score_curiosity(
        self,
        env_idx: int,
        facts: Dict[str, int],
        next_facts: Dict[str, int],
        action: Any,
        legal_actions: List[str],
        reward: float,
        done: bool,
        won: float,
        valid: bool,
        failure_reason: str,
        tags: List[str],
    ) -> Dict[str, Any]:
        current_events = self._recent_events[env_idx] if env_idx < len(self._recent_events) else []
        chain = self._candidate_causal_chain(env_idx, facts, tags, failure_reason, done, won)
        chain_actions = chain.get("action_chain", []) or []
        specific_failure = _is_specific_failure_reason(failure_reason)
        state_delta = self._state_delta(facts, next_facts)

        novelty = 0.15
        if state_delta:
            novelty += 0.25
        if legal_actions and (not current_events or current_events[-1].get("canonical_actions") != legal_actions):
            novelty += 0.15
        if specific_failure:
            novelty += 0.15

        task_relevance = 0.10
        if done or won > 0:
            task_relevance += 0.45
        if is_attack_action(action) or "attack" in tags:
            task_relevance += 0.20
        if "production" in tags or "supply_blocked" in tags or "minerals_low" in tags:
            task_relevance += 0.15
        if not valid or specific_failure:
            task_relevance += 0.20

        progress_potential = 0.10
        if float(reward) > 0:
            progress_potential += min(0.35, float(reward) / 50.0)
        if won > 0:
            progress_potential += 0.45
        if is_attack_action(action) and facts.get("army", 99) < 8:
            progress_potential += 0.25
        if facts.get("supply_left") == 0 and is_unit_train_action(action):
            progress_potential += 0.25
        if specific_failure:
            progress_potential += 0.20

        replayability = 0.10
        if _meaningful_do_chain(chain_actions) or _meaningful_avoid_chain(chain_actions, failure_reason, tags):
            replayability += 0.35
        if _generalized_tool_condition(facts, tags, action, self.attack_ready_min_army) != "general SC2 decision state":
            replayability += 0.20
        if _ordinary_single_action(chain_actions):
            replayability -= 0.25
        if _is_generic_terminal_failure(failure_reason) and done and won <= 0:
            replayability -= 0.20
        if chain.get("evidence_event_count", 0) >= 2:
            replayability += 0.15

        novelty = max(0.0, min(1.0, novelty))
        task_relevance = max(0.0, min(1.0, task_relevance))
        progress_potential = max(0.0, min(1.0, progress_potential))
        replayability = max(0.0, min(1.0, replayability))
        score = (
            0.20 * novelty
            + 0.25 * task_relevance
            + 0.25 * progress_potential
            + 0.30 * replayability
        )
        reflection_count = self._reflections_this_episode[env_idx] if env_idx < len(self._reflections_this_episode) else 0
        under_reflection_budget = (
            self.curiosity_max_reflections_per_episode < 0
            or reflection_count < self.curiosity_max_reflections_per_episode
        )
        force_reflect = bool(won > 0 or (not valid and specific_failure) or (is_attack_action(action) and facts.get("army", 99) < 8))
        should_write = score >= self.curiosity_write_threshold or force_reflect
        should_reflect = under_reflection_budget and (score >= self.curiosity_reflect_threshold or force_reflect)
        return {
            "score": round(float(score), 4),
            "novelty": round(float(novelty), 4),
            "task_relevance": round(float(task_relevance), 4),
            "progress_potential": round(float(progress_potential), 4),
            "replayability": round(float(replayability), 4),
            "should_write": bool(should_write),
            "should_reflect": bool(should_reflect),
            "trigger_reasons": self._curiosity_reasons(
                facts=facts,
                next_facts=next_facts,
                action=action,
                reward=reward,
                done=done,
                won=won,
                valid=valid,
                failure_reason=failure_reason,
                state_delta=state_delta,
            ),
            "candidate_causal_chain": chain,
        }

    def _state_delta(self, facts: Dict[str, int], next_facts: Dict[str, int]) -> Dict[str, int]:
        delta = {}
        for key in ("minerals", "vespene", "army", "workers", "supply_left", "enemy_units", "enemy_structures"):
            if key in facts and key in next_facts:
                diff = int(next_facts[key]) - int(facts[key])
                if diff:
                    delta[key] = diff
        return delta

    def _curiosity_reasons(
        self,
        facts: Dict[str, int],
        next_facts: Dict[str, int],
        action: Any,
        reward: float,
        done: bool,
        won: float,
        valid: bool,
        failure_reason: str,
        state_delta: Dict[str, int],
    ) -> List[str]:
        reasons = []
        if won > 0:
            reasons.append("win")
        elif done:
            reasons.append("terminal")
        if not valid:
            reasons.append("invalid_action")
        if _is_specific_failure_reason(failure_reason):
            reasons.append("specific_failure")
        if is_attack_action(action) and facts.get("army", 99) < 8:
            reasons.append("attack_too_early")
        if facts.get("supply_left") == 0 and is_unit_train_action(action):
            reasons.append("supply_blocked")
        if next_facts and state_delta:
            reasons.append("state_changed")
        if abs(float(reward)) >= 15:
            reasons.append("large_reward")
        return reasons[:8]

    def _candidate_causal_chain(
        self,
        env_idx: int,
        facts: Dict[str, int],
        tags: List[str],
        failure_reason: str,
        done: bool,
        won: float,
    ) -> Dict[str, Any]:
        events = self._recent_events[env_idx] if env_idx < len(self._recent_events) else []
        if not events:
            return {}
        selected = []
        for event in events[-min(len(events), self.curiosity_event_window) :]:
            if self._event_is_chain_step(event):
                selected.append(event)
        if not selected:
            selected = events[-min(len(events), self.max_chain_actions) :]
        selected = selected[-self.max_chain_actions :]
        actions: List[str] = []
        for event in selected:
            for action in event.get("canonical_actions", []) or []:
                if not actions or actions[-1] != action:
                    actions.append(action)
        actions = actions[-self.max_chain_actions :]
        if won > 0:
            chain_role = "progress_chain"
        elif done or _is_specific_failure_reason(failure_reason):
            chain_role = "failure_chain"
        else:
            chain_role = "diagnostic_chain"
        return {
            "action_chain": actions,
            "chain_role": chain_role,
            "skill_type": chain_role,
            "intent_hint": self._chain_intent_hint(chain_role, actions, facts, tags),
            "precondition_hints": self._chain_preconditions(facts, tags),
            "observed_effect_hint": self._chain_observed_effect(selected[-1] if selected else {}),
            "failure_signal_hints": self._chain_failure_signals(selected, failure_reason),
            "source_event_keys": [
                f"step={event.get('step_num')}|action={';'.join(event.get('canonical_actions', []) or [])}"
                for event in selected
            ],
            "evidence_event_count": len(selected),
            "composition_rule": "recent trusted SC2 events ending at curiosity anchor; max 2-5 grounded actions",
        }

    def _event_is_chain_step(self, event: Dict[str, Any]) -> bool:
        if event.get("won") or event.get("done") or not event.get("is_action_valid", True):
            return True
        if _as_float(event.get("reward")) != 0.0:
            return True
        if _is_specific_failure_reason(event.get("failure_reason")):
            return True
        actions = event.get("canonical_actions", []) or []
        if not actions:
            return False
        action_text = " ".join(actions).lower()
        if "offense" in action_text or "attack" in action_text:
            return True
        return any(name in action_text for name in ("supplydepot", "barracks", "factory", "refinery", "marine", "marauder", "siegetank"))

    def _chain_intent_hint(self, chain_role: str, actions: List[str], facts: Dict[str, int], tags: List[str]) -> str:
        if chain_role == "progress_chain":
            return "reuse a short SC2 progress chain under matching production/attack conditions"
        if chain_role == "failure_chain":
            return "avoid a repeated SC2 failure boundary under matching state conditions"
        if "supply_blocked" in tags or facts.get("supply_left") == 0:
            return "diagnose supply block before choosing production"
        if "army_low" in tags:
            return "diagnose army size before attack"
        return "diagnose current SC2 build-order state before action"

    def _chain_preconditions(self, facts: Dict[str, int], tags: List[str]) -> List[str]:
        preconditions = []
        if "attack" in tags:
            preconditions.append(f"army >= {self.attack_ready_min_army} before attack")
        if "production" in tags:
            preconditions.append("supply_left > 0 for unit production")
        if "barracks" in tags:
            preconditions.append("minerals >= 150 for barracks")
        if facts.get("supply_left") == 0:
            preconditions.append("resolve supply block first")
        return _dedup_keep_order(preconditions)[:5]

    def _chain_observed_effect(self, event: Dict[str, Any]) -> str:
        if not event:
            return ""
        effects = []
        reward = _as_float(event.get("reward"))
        if reward:
            effects.append(f"reward={reward:g}")
        if event.get("won"):
            effects.append("won=true")
        if event.get("done"):
            effects.append("done=true")
        delta = self._state_delta(event.get("facts_before", {}) or {}, event.get("facts_after", {}) or {})
        if delta:
            effects.append("state_delta=" + json.dumps(delta, ensure_ascii=False, sort_keys=True))
        return "; ".join(effects)[:180]

    def _chain_failure_signals(self, events: List[Dict[str, Any]], failure_reason: str) -> List[str]:
        signals = []
        if failure_reason:
            signals.append(failure_reason)
        for event in events:
            reason = event.get("failure_reason")
            if reason:
                signals.append(reason)
        return _dedup_keep_order([_shorten(signal, 80) for signal in signals if signal])[:5]

    def _should_store_scene(self, scene: Dict[str, Any]) -> bool:
        key = json.dumps(
            {
                "condition": scene.get("generalized_condition"),
                "action": scene.get("current_action"),
                "outcome": scene.get("outcome"),
                "failure": scene.get("failure_reason"),
                "chain": scene.get("candidate_causal_chain", {}).get("action_chain"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if key in self._seen_scene_keys:
            return False
        self._seen_scene_keys.add(key)
        return True

    def _reflection_scene(
        self,
        env_idx: int,
        step_num: int,
        obs: Any,
        next_obs: Any,
        action: Any,
        reward: float,
        done: bool,
        won: float,
        valid: bool,
        failure_reason: str,
        facts: Dict[str, int],
        next_facts: Dict[str, int],
        legal_actions: List[str],
        tags: List[str],
        chain: str,
        curiosity: Dict[str, Any],
    ) -> Dict[str, Any]:
        candidate_chain = curiosity.get("candidate_causal_chain") or {}
        return {
            "scene_id": f"env{env_idx}_step{step_num}",
            "facts": facts,
            "next_facts": next_facts,
            "generalized_condition": _generalized_tool_condition(facts, tags, action, self.attack_ready_min_army),
            "rule_tags": tags[:10],
            "recent_action_chain": _chain_actions(chain, max_actions=self.max_chain_actions),
            "candidate_causal_chain": candidate_chain,
            "current_action": legal_actions,
            "reward": reward,
            "done": done,
            "won": won,
            "is_action_valid": valid,
            "failure_reason": failure_reason,
            "outcome": "win" if won > 0 else ("loss" if done else "transition"),
            "curiosity": {key: value for key, value in curiosity.items() if key != "candidate_causal_chain"},
            "observation_before_excerpt": _shorten(obs, 900),
            "observation_after_excerpt": _shorten(next_obs, 900),
            "observation_excerpt": _shorten(obs, 1200),
        }

    def _write_reflection_scene(self, scene: Dict[str, Any]) -> None:
        if not self.reflection_scene_log_path:
            return
        row = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scene": scene,
        }
        _append_jsonl(self.reflection_scene_log_path, row)

    def _queue_training_reflection(self, env_idx: int, scene: Dict[str, Any]) -> None:
        if env_idx >= len(self._pending_training_reflection_scenes):
            return
        queue = self._pending_training_reflection_scenes[env_idx]
        queue.append(scene)
        max_keep = max(1, self.training_reflection_max_items_per_step * 2)
        if len(queue) > max_keep:
            del queue[: len(queue) - max_keep]

    def has_pending_training_reflections(self) -> bool:
        return bool(
            self.training_model_reflection
            and any(queue for queue in self._pending_training_reflection_scenes)
        )

    def build_training_reflection_prompts(self) -> List[str]:
        prompts = []
        self._active_training_reflection_scenes = [None for _ in range(self.batch_size)]
        scheduled = 0
        for env_idx in range(self.batch_size):
            scene = None
            if scheduled < self.training_reflection_max_items_per_step and self._pending_training_reflection_scenes[env_idx]:
                scene = self._pending_training_reflection_scenes[env_idx].pop(0)
                self._active_training_reflection_scenes[env_idx] = scene
                scheduled += 1
            if scene is None:
                prompts.append("Return exactly this JSON: {\"world_facts\":[],\"action_skills\":[]}")
            else:
                prompts.append(self._build_training_reflection_prompt(scene))
        return prompts

    def apply_training_reflection_outputs(self, outputs: List[str]) -> Dict[str, int]:
        stats = {"parsed": 0, "stored": 0, "failed": 0}
        for env_idx, output in enumerate(outputs):
            scene = self._active_training_reflection_scenes[env_idx] if env_idx < len(self._active_training_reflection_scenes) else None
            if not scene:
                continue
            parsed = _extract_json_object(output)
            if not isinstance(parsed, dict):
                stats["failed"] += 1
                self._log_training_reflection(env_idx, scene, output, None, "parse_failed")
                continue
            stats["parsed"] += 1
            facts = scene.get("facts", {}) if isinstance(scene, dict) else {}
            tags = scene.get("rule_tags", []) if isinstance(scene, dict) else []
            step_num = int(str(scene.get("scene_id", "0")).split("step")[-1]) if "step" in str(scene.get("scene_id", "")) and str(scene.get("scene_id", "")).split("step")[-1].isdigit() else 0
            assets = self._quality_gate_assets(
                self._assets_from_reflection(
                    reflection=parsed,
                    scene=scene,
                    facts=facts,
                    fallback_tags=tags,
                    step_num=step_num,
                )
            )
            for asset in assets:
                asset["created_by"] = "training_model_reflection"
                stored = self._upsert_asset(asset)
                if stored is not None:
                    stats["stored"] += 1
                    self._write_asset_log(env_idx, step_num, stored)
            if assets and env_idx < len(self._reflections_this_episode):
                self._reflections_this_episode[env_idx] += 1
            self._log_training_reflection(env_idx, scene, output, parsed, "success" if assets else "no_assets")
        self._active_training_reflection_scenes = [None for _ in range(self.batch_size)]
        return stats

    def _build_training_reflection_prompt(self, scene: Dict[str, Any]) -> str:
        compact_scene = {
            "facts": scene.get("facts"),
            "next_facts": scene.get("next_facts"),
            "condition": scene.get("generalized_condition"),
            "tags": scene.get("rule_tags"),
            "chain": scene.get("candidate_causal_chain"),
            "action": scene.get("current_action"),
            "reward": scene.get("reward"),
            "won": scene.get("won"),
            "done": scene.get("done"),
            "valid": scene.get("is_action_valid"),
            "failure": scene.get("failure_reason"),
            "curiosity": scene.get("curiosity"),
            "obs_before": _shorten(scene.get("observation_before_excerpt") or scene.get("observation_excerpt"), 450),
            "obs_after": _shorten(scene.get("observation_after_excerpt"), 450),
        }
        return (
            "Distill this SC2 scene into compact reusable memory. Output minified JSON only.\n"
            "Schema: {\"world_facts\":[{\"fact\":\"...\",\"condition\":\"...\",\"tags\":[\"...\"],\"confidence\":0.0,\"quality_rationale\":\"...\"}],"
            "\"action_skills\":[{\"polarity\":\"do|avoid|diagnostic\",\"intent\":\"...\",\"condition\":\"...\",\"action_chain\":[\"...\"],"
            "\"preconditions\":[\"...\"],\"failure_signals\":[\"...\"],\"tags\":[\"...\"],\"confidence\":0.0,\"quality_rationale\":\"...\"}]}\n"
            "Rules: max 1 world_fact and 1 action_skill; no single scv/marine do skill; avoid needs a specific observed failure; "
            "if won=true, emit one do skill from the candidate causal chain and preserve its grounded action order; "
            "do not convert a winning scene into an avoid skill; collapse repeated boundary warnings into one general rule; "
            f"use conditions like army < {self.attack_ready_min_army}, army >= {self.attack_ready_min_army}, supply_left=0, minerals < 150; "
            "confidence<0.65 if not replayable.\n"
            f"Scene: {json.dumps(compact_scene, ensure_ascii=False, default=str)}"
        )

    def _log_training_reflection(
        self,
        env_idx: int,
        scene: Dict[str, Any],
        output: str,
        parsed: Optional[Dict[str, Any]],
        status: str,
    ) -> None:
        if not self.evidence_audit_path:
            return
        _append_jsonl(
            self.evidence_audit_path,
            {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": status,
                "source": "training_model_reflection",
                "env_idx": env_idx,
                "scene_id": scene.get("scene_id", ""),
                "raw_response": _shorten(output, 2000),
                "parsed": parsed,
            },
        )

    def _assets_from_reflection(
        self,
        reflection: Dict[str, Any],
        scene: Dict[str, Any],
        facts: Dict[str, int],
        fallback_tags: List[str],
        step_num: int,
    ) -> List[Dict[str, Any]]:
        if not isinstance(reflection, dict):
            return []
        assets: List[Dict[str, Any]] = []
        selected_evidence = [
            _shorten(item, 80)
            for item in _as_text_items(reflection.get("selected_evidence"))
            if _to_text(item)
        ][:4]

        world_fact_items = _as_reflection_items(
            reflection.get("world_facts"),
            ("fact", "condition", "confidence", "quality_rationale", "text"),
        )
        for item in world_fact_items[:2]:
            confidence = _as_confidence(item.get("confidence"), 0.0)
            fact = _shorten(item.get("fact") or item.get("text") or "", 140)
            if confidence < self.reflection_min_quality or not fact:
                continue
            item_tags = _dedup_keep_order(fallback_tags + _as_text_items(item.get("tags")))
            asset = self._asset(
                kind="world_fact",
                polarity="fact",
                text=fact,
                condition=_normalize_reflection_condition(item.get("condition"), facts, item_tags),
                tags=item_tags + ["llm_reflection"],
                quality=confidence,
                action_chain="",
                failure_signals=selected_evidence,
                step_num=step_num,
            )
            asset["created_by"] = "llm_reflection"
            asset["quality_rationale"] = _shorten(item.get("quality_rationale", "LLM reflection with rule filtering."), 160)
            assets.append(asset)

        action_skill_items = _as_reflection_items(
            reflection.get("action_skills"),
            (
                "polarity",
                "skill_polarity",
                "intent",
                "text",
                "condition",
                "action_chain",
                "confidence",
                "quality_rationale",
            ),
        )
        for item in action_skill_items[:2]:
            polarity = str(item.get("polarity") or item.get("skill_polarity") or "do").strip().lower()
            if polarity not in {"do", "avoid", "diagnostic"}:
                continue
            confidence = _as_confidence(item.get("confidence"), 0.0)
            intent = _shorten(item.get("intent") or item.get("text") or "", 140)
            if confidence < self.reflection_min_quality or not intent:
                continue
            chain_actions = _chain_actions(item.get("action_chain") or [], max_actions=self.max_chain_actions)
            failure_signals = [
                _shorten(signal, 80)
                for signal in (_as_text_items(item.get("failure_signals")) or selected_evidence)
                if _to_text(signal)
            ][:4]
            item_tags = _dedup_keep_order(
                fallback_tags
                + _as_text_items(item.get("tags"))
                + ["llm_reflection"]
            )
            if polarity == "do" and not _meaningful_do_chain(chain_actions):
                continue
            if polarity == "avoid" and not _meaningful_avoid_chain(chain_actions, "; ".join(failure_signals), item_tags):
                continue
            if polarity == "diagnostic" and confidence < max(self.reflection_min_quality, 0.75):
                continue

            asset = self._asset(
                kind="action_skill",
                polarity=polarity,
                text=intent,
                condition=_normalize_reflection_condition(item.get("condition"), facts, item_tags),
                tags=item_tags,
                quality=confidence,
                action_chain=" -> ".join(chain_actions),
                failure_signals=failure_signals,
                step_num=step_num,
            )
            asset["created_by"] = "llm_reflection"
            asset["preconditions"] = [
                _shorten(value, 70)
                for value in _as_text_items(item.get("preconditions"))
                if _to_text(value)
            ][:4]
            asset["observed_effect"] = _shorten(item.get("observed_effect") or item.get("effect") or "", 140)
            asset["quality_rationale"] = _shorten(item.get("quality_rationale", "LLM reflection with replayability filter."), 160)
            assets.append(asset)
        return assets

    def _critical_rule_assets(
        self,
        step_num: int,
        facts: Dict[str, int],
        action: Any,
        legal_actions: List[str],
        tags: List[str],
        chain: str,
        valid: bool,
        failure_reason: str,
    ) -> List[Dict[str, Any]]:
        assets: List[Dict[str, Any]] = []
        condition = _generalized_tool_condition(facts, tags, action, self.attack_ready_min_army)
        action_summary = " -> ".join(legal_actions[:3])

        if not valid and (_is_specific_failure_reason(failure_reason) or action_summary):
            reason = failure_reason or "invalid action"
            assets.append(
                self._asset(
                    kind="action_skill",
                    polarity="avoid",
                    text=f"avoid {action_summary or 'invalid SC2 action'}; reason: {_shorten(reason, 70)}",
                    condition=condition,
                    tags=tags + ["invalid_action"],
                    quality=0.9 if _is_specific_failure_reason(reason) else 0.72,
                    action_chain=chain if _meaningful_avoid_chain(_chain_actions(chain), reason, tags) else "",
                    failure_signals=[reason],
                    step_num=step_num,
                )
            )

        if is_attack_action(action) and facts.get("army", 99) < self.attack_ready_min_army:
            assets.append(
                self._asset(
                    kind="action_skill",
                    polarity="avoid",
                    text=f"avoid attack before army >= {self.attack_ready_min_army}",
                    condition=f"army < {self.attack_ready_min_army}",
                    tags=tags + ["attack_too_early"],
                    quality=0.9,
                    action_chain="",
                    failure_signals=[f"army below attack readiness threshold {self.attack_ready_min_army}"],
                    step_num=step_num,
                )
            )

        if facts.get("supply_left") == 0 and is_unit_train_action(action):
            assets.append(
                self._asset(
                    kind="world_fact",
                    polarity="fact",
                    text="supply_left=0 blocks unit production; build supply depot first",
                    condition="supply_left=0",
                    tags=tags + ["supply_block_rule"],
                    quality=0.9,
                    action_chain="",
                    failure_signals=["supply blocked"],
                    step_num=step_num,
                )
            )

        if failure_reason and "insufficient minerals" in failure_reason.lower():
            assets.append(
                self._asset(
                    kind="world_fact",
                    polarity="fact",
                    text=f"{_shorten(failure_reason, 70)}; choose affordable production or wait",
                    condition=condition,
                    tags=tags + ["resource_rule"],
                    quality=0.82,
                    action_chain="",
                    failure_signals=[failure_reason],
                    step_num=step_num,
                )
            )
        return assets

    def _fallback_rule_assets(
        self,
        step_num: int,
        facts: Dict[str, int],
        action: Any,
        legal_actions: List[str],
        tags: List[str],
        chain: str,
        reward: float,
        done: bool,
        won: float,
        valid: bool,
        failure_reason: str,
    ) -> List[Dict[str, Any]]:
        assets = self._critical_rule_assets(step_num, facts, action, legal_actions, tags, chain, valid, failure_reason)
        assets.extend(
            self._verified_positive_outcome_assets(
                step_num=step_num,
                facts=facts,
                action=action,
                tags=tags,
                chain=chain,
                done=done,
                won=won,
            )
        )
        condition = _generalized_tool_condition(facts, tags, action, self.attack_ready_min_army)
        chain_actions = _chain_actions(chain, max_actions=self.max_chain_actions)

        if done and won > 0:
            return assets
        if done and not _is_generic_terminal_failure(failure_reason) and _meaningful_avoid_chain(chain_actions, failure_reason, tags):
            assets.append(
                self._asset(
                    kind="action_skill",
                    polarity="avoid",
                    text="avoid repeating failed action chain under matching boundary",
                    condition=condition,
                    tags=tags + ["loss_pattern"],
                    quality=0.82,
                    action_chain=" -> ".join(chain_actions),
                    failure_signals=[failure_reason],
                    step_num=step_num,
                )
            )
        elif reward >= 15 and _meaningful_do_chain(chain_actions):
            assets.append(
                self._asset(
                    kind="action_skill",
                    polarity="do",
                    text="reuse short positive-reward action chain",
                    condition=condition,
                    tags=tags + ["positive_reward"],
                    quality=0.82,
                    action_chain=" -> ".join(chain_actions),
                    failure_signals=[],
                    step_num=step_num,
                )
            )
        elif reward <= -20 and _meaningful_avoid_chain(chain_actions, failure_reason, tags):
            assets.append(
                self._asset(
                    kind="action_skill",
                    polarity="avoid",
                    text="avoid short negative-reward action chain under matching boundary",
                    condition=condition,
                    tags=tags + ["negative_reward"],
                    quality=0.82,
                    action_chain=" -> ".join(chain_actions),
                    failure_signals=[failure_reason or f"reward={reward:.1f}"],
                    step_num=step_num,
                )
            )
        return assets

    def _verified_positive_outcome_assets(
        self,
        step_num: int,
        facts: Dict[str, int],
        action: Any,
        tags: List[str],
        chain: str,
        done: bool,
        won: float,
    ) -> List[Dict[str, Any]]:
        if not done or won <= 0:
            return []
        chain_actions = _chain_actions(chain, max_actions=self.max_chain_actions)
        if not _meaningful_do_chain(chain_actions):
            return []
        condition = _generalized_tool_condition(
            facts,
            tags,
            action,
            self.attack_ready_min_army,
        )
        asset = self._asset(
            kind="action_skill",
            polarity="do",
            text="reuse verified winning action chain: " + " -> ".join(chain_actions),
            condition=condition,
            tags=tags + ["verified_win", "win_pattern", "learned_success"],
            quality=0.97,
            action_chain=" -> ".join(chain_actions),
            failure_signals=[],
            step_num=step_num,
        )
        asset["created_by"] = "verified_win_transition"
        asset["observed_effect"] = "won=true"
        asset["quality_rationale"] = (
            "Deterministically extracted from a valid terminal winning transition."
        )
        return [asset]

    def _quality_gate_assets(self, assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered = []
        for asset in assets:
            source = str(asset.get("created_by") or "")
            threshold = max(self.quality_threshold, self.reflection_min_quality) if source in {"llm_reflection", "training_model_reflection", "model_reflection"} else max(
                self.quality_threshold,
                self.dynamic_quality_threshold,
            )
            if float(asset.get("quality", 0.0) or 0.0) < threshold:
                continue
            if asset.get("kind") == "action_skill":
                polarity = str(asset.get("polarity") or "do")
                actions = _chain_actions(asset.get("action_chain"), max_actions=self.max_chain_actions)
                if polarity == "do":
                    if not _meaningful_do_chain(actions):
                        continue
                    if (
                        _ordinary_single_action(actions)
                        and source != "verified_win_transition"
                    ):
                        continue
                if polarity == "avoid" and not _meaningful_avoid_chain(actions, "; ".join(asset.get("failure_signals", []) or []), asset.get("tags", [])):
                    continue
            filtered.append(asset)
        return filtered

    def _seed_static_assets(self) -> None:
        seeds = [
            self._asset(
                kind="world_fact",
                polarity="fact",
                text="supply_left=0 blocks unit production; build supply depot before more units",
                condition="supply_left=0",
                tags=["supply_blocked", "production", "supply_block_rule"],
                quality=0.98,
                action_chain="",
                failure_signals=["supply blocked"],
                step_num=0,
            ),
            self._asset(
                kind="action_skill",
                polarity="avoid",
                text=f"avoid attack before army >= {self.attack_ready_min_army}",
                condition=f"army < {self.attack_ready_min_army}",
                tags=["army_low", "attack", "attack_too_early"],
                quality=0.95,
                action_chain="",
                failure_signals=[f"army below attack readiness threshold {self.attack_ready_min_army}"],
                step_num=0,
            ),
            self._asset(
                kind="action_skill",
                polarity="avoid",
                text="avoid building barracks when minerals < 150",
                condition="minerals < 150",
                tags=["minerals_low", "barracks", "resource_rule"],
                quality=0.9,
                action_chain="",
                failure_signals=["insufficient minerals for barracks"],
                step_num=0,
            ),
            self._asset(
                kind="action_skill",
                polarity="do",
                text="keep marine production and attack after army is ready",
                condition=f"army >= {self.attack_ready_min_army} and supply_left > 0",
                tags=["army_attack_ready", "marine", "production", "attack"],
                quality=0.92,
                action_chain=f"marine -> marine -> Launch OFFENSE with {self.attack_ready_min_army} marine",
                failure_signals=[],
                step_num=0,
            ),
            self._asset(
                kind="action_skill",
                polarity="diagnostic",
                text="check supply_left, minerals, and army before choosing production or attack",
                condition="any SC2 decision state",
                tags=["diagnostic_check", "production", "attack"],
                quality=0.85,
                action_chain="",
                failure_signals=[],
                step_num=0,
            ),
        ]
        for asset in seeds:
            self._upsert_asset(asset)

    def no_confident(self, message: str, tool_name: str = "query_skill") -> Dict[str, Any]:
        base = {
            "status": "success",
            "memory_source": "sc2_tool_assets",
            "tool_name": tool_name,
            "message": f"no confident match: {message}",
            "no_confident_match": True,
        }
        if tool_name == "query_world_model":
            base.update({"matches": [], "relevant_rules": [], "linked_skills": []})
        elif tool_name == "query_graph_skill":
            base.update({"graph_skills_count": 0, "graph_skills": []})
        else:
            base.update({"skills_count": 0, "skills": []})
        return base

    def run_tool(self, tool_call: Dict[str, Any], env_idx: int, current_observation: Any) -> Dict[str, Any]:
        name = str(tool_call.get("name") or "").strip()
        args = tool_call.get("arguments") or {}
        if name == "query_skill":
            return self.query_skill(
                query=args.get("query", ""),
                env_name=args.get("env_name", "SC2"),
                current_observation=args.get("current_observation") or current_observation,
                env_idx=env_idx,
            )
        if name == "query_graph_skill":
            return self.query_graph_skill(
                query=args.get("query", ""),
                env_name=args.get("env_name", "SC2"),
                current_observation=args.get("current_observation") or current_observation,
                env_idx=env_idx,
            )
        if name == "query_world_model":
            return self.query_world_model(
                query=args.get("query", ""),
                env_name=args.get("env_name", "SC2"),
                current_observation=args.get("current_observation") or current_observation,
                env_idx=env_idx,
            )
        return self.no_confident(f"unsupported memory tool {name}", name or "query_skill")

    def query_skill(
        self,
        query: str,
        env_name: str = "SC2",
        current_observation: Any = "",
        top_k: int = None,
        env_idx: int = 0,
    ) -> Dict[str, Any]:
        del env_idx
        matches = self._retrieve_assets_for_query(query, current_observation, view="skills", top_k=top_k)
        if not matches:
            return self.no_confident("no action skill matches current SC2 state", "query_skill")
        skills = [
            skill
            for skill in [self._format_skill(asset, current_observation, score) for score, asset in matches]
            if skill.get("match_status") != "not_applicable"
        ]
        if not skills:
            return self.no_confident("retrieved skills are not applicable to current SC2 state", "query_skill")
        match_counts: Dict[str, int] = {}
        for skill in skills:
            status = str(skill.get("match_status") or "")
            match_counts[status] = match_counts.get(status, 0) + 1
        return {
            "status": "success",
            "memory_source": "sc2_tool_assets",
            "tool_name": "query_skill",
            "env_name": env_name or "SC2",
            "query": _shorten(query, 160),
            "message": "SC2 compact action skill retrieval",
            "skills_count": len(skills),
            "skills": skills,
            "avoid_actions": [skill for skill in skills if skill.get("skill_polarity") == "avoid"],
            "has_avoid_actions": any(skill.get("skill_polarity") == "avoid" for skill in skills),
            "diagnostic_actions": [skill for skill in skills if skill.get("skill_polarity") == "diagnostic"],
            "match_status_counts": match_counts,
            "ready_skill_count": match_counts.get("ready", 0),
            "prep_needed_skill_count": match_counts.get("prep_needed", 0),
            "diagnostic_only_skill_count": match_counts.get("diagnostic_only", 0),
            "executable_skill_count": sum(1 for skill in skills if skill.get("executable_now")),
            "no_confident_match": False,
            "retrieval_guidance": self._retrieval_guidance(skills),
        }

    def query_graph_skill(
        self,
        query: str,
        env_name: str = "SC2",
        current_observation: Any = "",
        top_k: int = None,
        env_idx: int = 0,
    ) -> Dict[str, Any]:
        del env_idx
        matches = self._retrieve_assets_for_query(query, current_observation, view="all", top_k=top_k)
        if not matches:
            return self.no_confident("no linked SC2 assets match current state", "query_graph_skill")
        graph_skills = [self._format_linked_asset(asset, current_observation, score) for score, asset in matches]
        return {
            "status": "success",
            "memory_source": "sc2_tool_assets",
            "tool_name": "query_graph_skill",
            "env_name": env_name or "SC2",
            "query": _shorten(query, 160),
            "message": "SC2 linked compact asset retrieval",
            "graph_skills_count": len(graph_skills),
            "graph_skills": graph_skills,
            "no_confident_match": False,
        }

    def query_world_model(
        self,
        query: str,
        env_name: str = "SC2",
        current_observation: Any = "",
        top_k_states: int = None,
        env_idx: int = 0,
        **kwargs,
    ) -> Dict[str, Any]:
        del env_idx, kwargs
        matches = self._retrieve_assets_for_query(query, current_observation, view="facts", top_k=top_k_states)
        if not matches:
            return self.no_confident("no SC2 world facts match current state", "query_world_model")
        facts = [self._format_world_fact(asset, score) for score, asset in matches]
        return {
            "status": "success",
            "memory_source": "sc2_tool_assets",
            "tool_name": "query_world_model",
            "env_name": env_name or "SC2",
            "query": _shorten(query, 160),
            "matches": facts,
            "summary": "; ".join(item["fact"] for item in facts[:2]),
            "relevant_rules": facts,
            "linked_skills": [],
            "no_confident_match": False,
        }

    def _retrieve_assets_for_query(
        self,
        query: Any,
        obs: Any,
        view: str,
        top_k: int,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        facts = extract_tool_sc2_facts(obs)
        current_tags = set(_fact_tags(facts))
        query_tokens = _tokens(query)
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for asset in self._assets:
            if self._retrieval_asset_keys is not None and asset.get("key") not in self._retrieval_asset_keys:
                continue
            if view == "skills" and asset.get("kind") != "action_skill":
                continue
            if view == "facts" and asset.get("kind") != "world_fact":
                continue
            quality = float(asset.get("quality", 0.0) or 0.0)
            if quality < self.quality_threshold:
                continue
            match_status = ""
            executable = False
            if asset.get("kind") == "action_skill":
                match_status, executable, _ = self._match_status(asset, obs)
                if match_status == "not_applicable":
                    continue
            asset_tags = set(asset.get("tags", []) or [])
            asset_tokens = _tokens(
                " ".join(
                    [
                        str(asset.get("text", "")),
                        str(asset.get("condition", "")),
                        " ".join(asset.get("failure_signals", []) or []),
                        " ".join(asset.get("tags", []) or []),
                    ]
                )
            )
            tag_overlap = len(current_tags & asset_tags)
            token_overlap = len(query_tokens & asset_tokens)
            if tag_overlap <= 0 and token_overlap <= 0:
                continue
            tag_score = tag_overlap / max(1, min(len(asset_tags), 4))
            token_score = token_overlap / max(1, min(len(query_tokens), 8))
            score = 0.45 * quality + 0.35 * tag_score + 0.20 * token_score
            source = str(asset.get("created_by") or "")
            if source and source != "seed_static":
                score += 0.08
            if executable:
                score += 0.06
            if score >= self.match_threshold:
                scored.append((score, asset))
        scored.sort(
            key=lambda item: (
                item[0],
                float(item[1].get("quality", 0.0) or 0.0),
                int(item[1].get("count", 1) or 1),
            ),
            reverse=True,
        )
        deduplicated = []
        seen = set()
        for score, asset in scored:
            actions = tuple(
                action.lower()
                for action in _chain_actions(
                    asset.get("action_chain"),
                    max_actions=self.max_chain_actions,
                )
            )
            signature = (
                str(asset.get("kind") or ""),
                str(asset.get("polarity") or ""),
                str(asset.get("condition") or "").strip().lower(),
                actions,
                str(asset.get("text") or "").strip().lower() if not actions else "",
            )
            if signature in seen:
                continue
            seen.add(signature)
            deduplicated.append((score, asset))
        return deduplicated[: max(1, int(top_k or self.max_prompt_items))]

    def _format_skill(self, asset: Dict[str, Any], obs: Any, score: float) -> Dict[str, Any]:
        status, executable, missing = self._match_status(asset, obs)
        sequence_actions = self._sequence_actions(asset)
        actions = sequence_actions[:4]
        return {
            "asset_id": asset.get("key", ""),
            "skill_polarity": asset.get("polarity", "do"),
            "skill_type": self._skill_type(asset),
            "intent": _shorten(asset.get("text", ""), 140),
            "condition": _shorten(asset.get("condition", ""), 120),
            "quality": float(asset.get("quality", 0.0) or 0.0),
            "created_by": str(asset.get("created_by") or "unknown"),
            "live_match_score": round(float(score), 4),
            "match_status": status,
            "executable_now": executable,
            "missing_preconditions": missing,
            "next_grounded_actions": actions[:4] if executable else actions[:2],
            "sequence_actions": sequence_actions,
            "failure_signals": asset.get("failure_signals", [])[:4],
            "decisive_boundary_precheck": asset.get("failure_signals", [])[:3],
            "chain_execution_advice": self._chain_advice(asset, status, actions),
        }

    def _format_linked_asset(self, asset: Dict[str, Any], obs: Any, score: float) -> Dict[str, Any]:
        if asset.get("kind") == "world_fact":
            return self._format_world_fact(asset, score)
        return self._format_skill(asset, obs, score)

    def _format_world_fact(self, asset: Dict[str, Any], score: float) -> Dict[str, Any]:
        return {
            "asset_id": asset.get("key", ""),
            "asset_type": "world_fact",
            "fact": _shorten(asset.get("text", ""), 150),
            "condition": _shorten(asset.get("condition", ""), 120),
            "quality": float(asset.get("quality", 0.0) or 0.0),
            "live_match_score": round(float(score), 4),
            "failure_signals": asset.get("failure_signals", [])[:4],
        }

    def _skill_type(self, asset: Dict[str, Any]) -> str:
        polarity = str(asset.get("polarity") or "do")
        if polarity in {"avoid", "diagnostic"}:
            return polarity
        if asset.get("action_chain"):
            return "progress_chain"
        return "single_action"

    def _match_status(self, asset: Dict[str, Any], obs: Any) -> Tuple[str, bool, List[str]]:
        polarity = str(asset.get("polarity") or "do")
        facts = extract_tool_sc2_facts(obs)
        condition = str(asset.get("condition") or "").lower()
        if polarity == "avoid":
            army_lt = re.search(r"army\s*<\s*(\d+)", condition)
            if army_lt and facts.get("army", 0) >= int(army_lt.group(1)):
                return "not_applicable", False, []
            if "minerals < 150" in condition and facts.get("minerals", 0) >= 150:
                return "not_applicable", False, []
            if "supply_left=0" in condition and facts.get("supply_left", 1) > 0:
                return "not_applicable", False, []
            return "diagnostic_only", False, []
        if polarity == "diagnostic":
            return "diagnostic_only", False, []
        tags = set(asset.get("tags", []) or [])
        missing = []
        action_chain = _chain_actions(asset.get("action_chain"), max_actions=self.max_chain_actions)
        is_attack_skill = (
            "attack" in tags
            or "offense" in condition
            or "attack" in condition
            or _has_attack(action_chain)
        )
        if is_attack_skill and facts.get("army", 99) < self.attack_ready_min_army:
            missing.append(f"army >= {self.attack_ready_min_army} before attack")
        if "production" in tags and facts.get("supply_left", 1) <= 0:
            missing.append("supply_left > 0 or build supply depot")
        if "barracks" in tags and facts.get("minerals", 0) < 150:
            missing.append("minerals >= 150 for barracks")
        if missing:
            return "prep_needed", False, missing
        return "ready", True, []

    def _next_grounded_actions(self, asset: Dict[str, Any]) -> List[str]:
        return self._sequence_actions(asset)[:4]

    def _sequence_actions(self, asset: Dict[str, Any]) -> List[str]:
        chain = _to_text(asset.get("action_chain", ""))
        if not chain:
            return []
        return extract_legal_sc2_actions(
            chain,
            max_actions=self.max_chain_actions,
        )

    def _chain_advice(self, asset: Dict[str, Any], status: str, actions: List[str]) -> str:
        polarity = str(asset.get("polarity") or "do")
        if polarity == "avoid":
            return "Treat this as a negative constraint; do not execute the failed chain."
        if polarity == "diagnostic":
            return "Use this as a state check, not as an action script."
        if status == "prep_needed":
            return "Satisfy missing_preconditions before replaying this chain."
        if actions:
            return "If current observation still matches, prioritize next_grounded_actions[0] and stop on failure feedback."
        return "Use as a short decision hint only; current observation has priority."

    def _retrieval_guidance(self, skills: List[Dict[str, Any]]) -> str:
        if any(skill.get("skill_polarity") == "avoid" for skill in skills):
            return "Avoid skills are negative boundaries, not actions to execute."
        if any(skill.get("executable_now") for skill in skills):
            return "Ready do skill retrieved; verify current state, then use the first grounded action."
        if any(skill.get("match_status") == "prep_needed" for skill in skills):
            return "Relevant skill found but preconditions are missing; satisfy prep first."
        return "Use retrieved memory as advisory context only."


def compact_tool_result_for_prompt(result: Dict[str, Any]) -> Dict[str, Any]:
    if result.get("no_confident_match"):
        return {
            "tool_name": result.get("tool_name", ""),
            "no_confident_match": True,
            "message": result.get("message", ""),
        }
    tool_name = result.get("tool_name", "")
    if tool_name == "query_world_model":
        return {
            "tool_name": tool_name,
            "no_confident_match": False,
            "relevant_rules": [
                {
                    "fact": item.get("fact", ""),
                    "condition": item.get("condition", ""),
                }
                for item in result.get("relevant_rules", [])[:2]
            ],
            "summary": result.get("summary", ""),
        }
    if tool_name == "query_graph_skill":
        return {
            "tool_name": tool_name,
            "no_confident_match": False,
            "graph_skills": _compact_skills(result.get("graph_skills", [])[:2]),
        }
    return {
        "tool_name": tool_name,
        "no_confident_match": False,
        "do": _compact_skills([item for item in result.get("skills", []) if item.get("skill_polarity") == "do"][:2]),
        "avoid": _compact_skills([item for item in result.get("skills", []) if item.get("skill_polarity") == "avoid"][:2]),
        "diagnostic": _compact_skills([item for item in result.get("skills", []) if item.get("skill_polarity") == "diagnostic"][:1]),
        "retrieval_guidance": result.get("retrieval_guidance", ""),
    }


def _compact_skills(skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact = []
    for skill in skills:
        status = skill.get("match_status", "")
        next_actions = []
        for action in skill.get("next_grounded_actions", []) or []:
            next_actions.extend(extract_legal_sc2_actions(action, max_actions=2 - len(next_actions)))
            if len(next_actions) >= 2:
                break
        compact.append(
            {
                "polarity": skill.get("skill_polarity", skill.get("polarity", "")),
                "status": status,
                "intent": _shorten(skill.get("intent", skill.get("fact", "")), 90),
                "next": next_actions[:2] if status == "ready" else [],
                "missing": skill.get("missing_preconditions", [])[:2],
                "avoid_if": skill.get("failure_signals", [])[:2],
            }
        )
    return compact
