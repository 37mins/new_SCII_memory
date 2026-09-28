import os
import re
from typing import Any, Dict, List, Tuple

from .base import BaseMemory


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").strip()


def _shorten(value: Any, limit: int) -> str:
    text = _to_text(value)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def _extract_int(pattern: str, text: str):
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(float(match.group(1)))
    except (TypeError, ValueError):
        return None


def extract_sc2_facts(obs: Any) -> Dict[str, int]:
    text = _to_text(obs)
    facts = {}
    time_match = re.search(r"(?:It is now|time_formatted:)\s*(\d{1,2}):(\d{2})", text, flags=re.IGNORECASE)
    if time_match:
        facts["time_s"] = int(time_match.group(1)) * 60 + int(time_match.group(2))

    patterns = (
        ("minerals", r"minerals:?\s*(\d+)"),
        ("vespene", r"(?:vespene|gas):?\s*(\d+)"),
        ("army", r"(?:supply_army:|allocated to the army,?)\s*(\d+)"),
        ("workers", r"(?:supply_workers:|to workers,?)\s*(\d+)"),
        ("supply_left", r"supply_left:?\s*(\d+)"),
        ("enemy_units", r"n_visible_enemy_units:?\s*(\d+)"),
        ("enemy_structures", r"n_visible_enemy_structures:?\s*(\d+)"),
    )
    for key, pattern in patterns:
        value = _extract_int(pattern, text)
        if value is not None:
            facts[key] = value
    return facts


def format_sc2_facts(facts: Dict[str, int]) -> str:
    if not facts:
        return "unknown"
    parts = []
    if "time_s" in facts:
        parts.append(f"time={facts['time_s'] // 60:02d}:{facts['time_s'] % 60:02d}")
    for key in ("minerals", "vespene", "army", "workers", "supply_left", "enemy_units", "enemy_structures"):
        if key in facts:
            parts.append(f"{key}={facts[key]}")
    return ", ".join(parts)


def _has_action_token(action: Any, tokens: Tuple[str, ...]) -> bool:
    lower = _to_text(action).lower()
    return any(token in lower for token in tokens)


def is_attack_action(action: Any) -> bool:
    return _has_action_token(action, ("attack", "launch offense", "attacking instruction"))


def is_unit_train_action(action: Any) -> bool:
    return _has_action_token(action, ("train", "marine", "scv", "commandcentertrain", "barrackstrain"))


def infer_failure_reason(action: Any, obs: Any, is_valid: bool = False, explicit_reason: Any = None) -> str:
    reason = _shorten(explicit_reason, 120)
    if reason:
        return reason
    if is_valid:
        return ""

    facts = extract_sc2_facts(obs)
    lower = _to_text(action).lower()
    minerals = facts.get("minerals")
    supply_left = facts.get("supply_left")
    army = facts.get("army")

    if supply_left is not None and supply_left <= 0 and is_unit_train_action(action):
        return "supply blocked"
    if "commandcentertrain_scv" in lower or re.search(r"\bscv\b", lower):
        if minerals is not None and minerals < 50:
            return "insufficient minerals for SCV"
    if "barracks" in lower:
        if minerals is not None and minerals < 150 and "train" not in lower:
            return "insufficient minerals for barracks"
    if "marine" in lower and minerals is not None and minerals < 50:
        return "insufficient minerals for marine"
    if is_attack_action(action) and army is not None and army < 8:
        return "attack likely too early"
    return "invalid action"


def tactical_diagnosis(facts: Dict[str, int], recent: List[Dict[str, Any]]) -> str:
    notes = []
    supply_left = facts.get("supply_left")
    minerals = facts.get("minerals")
    army = facts.get("army")
    workers = facts.get("workers")

    if supply_left is not None and supply_left <= 1:
        notes.append("supply blocked risk")
    if army is not None and army < 8:
        notes.append("army too small for reliable attack")
    elif army is not None and army >= 20:
        notes.append("army ready for strong attack")
    if minerals is not None and minerals >= 300 and (army is None or army < 8):
        notes.append("minerals floating while army is low")
    if workers is not None and workers < 18:
        notes.append("worker count still low")

    failures = [rec.get("failure_reason") for rec in recent if _to_text(rec.get("failure_reason"))]
    if failures:
        notes.append(f"recent failure: {_shorten(failures[-1], 60)}")
    return "; ".join(notes[:3]) if notes else "stable; follow build order priorities"


def next_action_guidance(facts: Dict[str, int]) -> str:
    guidance = []
    supply_left = facts.get("supply_left")
    minerals = facts.get("minerals")
    army = facts.get("army")

    if supply_left is not None and supply_left <= 1:
        guidance.append("build supply depot before more units")
    if minerals is not None and minerals >= 150:
        guidance.append("build or use barracks for marine production")
    else:
        guidance.append("avoid unaffordable buildings")
    if army is not None and army < 8:
        guidance.append("avoid attack until army >= 8")
    elif army is not None and army >= 8:
        guidance.append("consider attack only if army is available")
    return "; ".join(guidance[:3])


def summarize_action(action: Any) -> str:
    text = _to_text(action)
    if not text:
        return ""

    lower = text.lower()
    tags = []
    if any(token in lower for token in ("attack", "launch offense", "attacking instruction")):
        tags.append("attack")
    if "commandcentertrain_scv" in lower or re.search(r"\bscv\b", lower):
        tags.append("worker")
    if "marine" in lower:
        tags.append("marine")
    if any(token in lower for token in ("supplydepot", "supply depot")):
        tags.append("supply")
    if any(token in lower for token in ("barracks", "rax")):
        tags.append("barracks")
    if any(token in lower for token in ("command center", "expansion")) or (
        "commandcenter" in lower and "commandcentertrain_scv" not in lower
    ):
        tags.append("expand")

    head = _shorten(" ".join(text.split()), 120)
    if tags:
        return f"{','.join(tags)}: {head}"
    return head


class StrategicMemoryV2(BaseMemory):
    """SC2 strategy memory with compact state, salient events, and cross-episode notes."""

    def __init__(
        self,
        max_action_chars: int = None,
        max_episodic_items: int = None,
        max_strategy_notes: int = None,
    ):
        self._data = None
        self._episodes = None
        self._strategy_notes = []
        self.keys = None
        self.batch_size = 0
        self.max_action_chars = max_action_chars or int(os.getenv("AGENT_MEMORY_MAX_ACTION_CHARS", "160"))
        self.max_episodic_items = max_episodic_items or int(os.getenv("AGENT_MEMORY_MAX_EPISODIC_ITEMS", "3"))
        self.max_strategy_notes = max_strategy_notes or int(os.getenv("AGENT_MEMORY_MAX_STRATEGY_NOTES", "3"))

    def __len__(self):
        return len(self._data) if self._data is not None else 0

    def __getitem__(self, idx: int):
        return self._data[idx]

    def reset(self, batch_size: int):
        self._data = [[] for _ in range(batch_size)]
        self._episodes = [[] for _ in range(batch_size)]
        self.batch_size = batch_size
        self.keys = None

    def store(self, record: Dict[str, List[Any]]):
        if self.keys is None:
            self.keys = list(record.keys())
        assert self.keys == list(record.keys())

        for env_idx in range(self.batch_size):
            rec = {key: record[key][env_idx] for key in self.keys}
            self._data[env_idx].append(rec)
            self._store_episode_if_salient(env_idx, len(self._data[env_idx]), rec)
            self._store_strategy_note_if_terminal(env_idx, len(self._data[env_idx]), rec)

    def fetch(
        self,
        history_length: int,
        current_obs: List[Any],
        action_key: str = "action",
    ) -> Tuple[List[str], List[int]]:
        memory_contexts, valid_lengths = [], []
        for env_idx in range(self.batch_size):
            recent = self._data[env_idx][-history_length:] if history_length > 0 else self._data[env_idx][-2:]
            valid_lengths.append(len(recent))
            memory_contexts.append(self._format_context(env_idx, current_obs[env_idx], recent, action_key))
        return memory_contexts, valid_lengths

    def _format_context(self, env_idx: int, current_obs: Any, recent: List[Dict[str, Any]], action_key: str) -> str:
        facts = extract_sc2_facts(current_obs)
        lines = ["Strategic memory:"]
        lines.append(f"- Situation: {format_sc2_facts(facts)}")
        lines.append(f"- Tactical diagnosis: {tactical_diagnosis(facts, recent)}")
        lines.append(f"- Next action guidance: {next_action_guidance(facts)}")

        episodes = self._episodes[env_idx] if self._episodes else []
        if episodes:
            lines.append("- Recent outcomes: " + " | ".join(episodes[-self.max_episodic_items:]))

        if self._strategy_notes:
            lines.append("- Strategy notes: " + " | ".join(self._strategy_notes[-self.max_strategy_notes:]))

        return "\n".join(lines)

    def _store_episode_if_salient(self, env_idx: int, step_num: int, rec: Dict[str, Any]):
        event = self._make_episode_event(step_num, rec)
        if not event:
            return
        episodes = self._episodes[env_idx]
        if episodes and episodes[-1] == event:
            return
        episodes.append(event)
        if len(episodes) > self.max_episodic_items:
            del episodes[: len(episodes) - self.max_episodic_items]

    def _make_episode_event(self, step_num: int, rec: Dict[str, Any]) -> str:
        action = _to_text(rec.get("action"))
        reward = _as_float(rec.get("reward"))
        done = _as_bool(rec.get("done"))
        won = _as_float(rec.get("won"))
        valid = _as_bool(rec.get("is_action_valid"))
        failure_reason = _to_text(rec.get("failure_reason"))

        tags = []
        if not valid:
            tags.append(f"failed: {failure_reason or 'invalid action'}")
        if is_attack_action(action):
            tags.append("attack_order")
        if done and won > 0:
            tags.append("victory")
        elif done:
            tags.append("defeat")
        elif reward >= 15:
            tags.append(f"good_reward={reward:.1f}")
        elif reward <= -20:
            tags.append(f"bad_reward={reward:.1f}")

        if not tags:
            return ""
        summary = summarize_action(action)
        if summary:
            return f"step {step_num}: {', '.join(tags)}; {_shorten(summary, 90)}"
        return f"step {step_num}: {', '.join(tags)}"

    def _store_strategy_note_if_terminal(self, env_idx: int, step_num: int, rec: Dict[str, Any]):
        if not _as_bool(rec.get("done")):
            return

        won = _as_float(rec.get("won"))
        reward = _as_float(rec.get("reward"))
        obs = rec.get("text_obs")
        facts = format_sc2_facts(extract_sc2_facts(obs))
        action = summarize_action(rec.get("action"))

        if won > 0:
            note = f"win pattern: keep production; final {facts}; last={_shorten(action, 70)}"
        else:
            reason = infer_failure_reason(rec.get("action"), obs, bool(rec.get("is_action_valid")), rec.get("failure_reason"))
            detail = reason or "adjust timing and army before repeating"
            note = f"loss warning: {detail}; final {facts}; reward={reward:.1f}"

        if note and (not self._strategy_notes or self._strategy_notes[-1] != note):
            self._strategy_notes.append(note)
        if len(self._strategy_notes) > self.max_strategy_notes:
            del self._strategy_notes[: len(self._strategy_notes) - self.max_strategy_notes]
