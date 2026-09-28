import json
import os
import time
from typing import Any, Dict, List, Tuple

from .base import BaseMemory
from .strategic_memory_v2 import (
    extract_sc2_facts,
    format_sc2_facts,
    infer_failure_reason,
    is_attack_action,
    is_unit_train_action,
    next_action_guidance,
    summarize_action,
)


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
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _fact_tags(facts: Dict[str, int]) -> List[str]:
    tags = []
    supply_left = facts.get("supply_left")
    minerals = facts.get("minerals")
    army = facts.get("army")
    workers = facts.get("workers")
    enemy_units = facts.get("enemy_units")
    enemy_structures = facts.get("enemy_structures")

    if supply_left is not None:
        if supply_left <= 0:
            tags.append("supply_blocked")
        elif supply_left <= 1:
            tags.append("supply_low")
    if army is not None:
        if army < 8:
            tags.append("army_low")
        else:
            tags.append("army_attack_ready")
    if minerals is not None:
        if minerals < 50:
            tags.append("minerals_low")
        if minerals >= 150:
            tags.append("can_afford_barracks")
        if minerals >= 300:
            tags.append("minerals_floating")
    if workers is not None and workers < 18:
        tags.append("workers_low")
    if enemy_units is not None and enemy_units > 0:
        tags.append("enemy_units_seen")
    if enemy_structures is not None and enemy_structures > 0:
        tags.append("enemy_structures_seen")
    return tags


def _action_tags(action: Any) -> List[str]:
    lower = _to_text(action).lower()
    tags = []
    if is_attack_action(action):
        tags.append("attack")
    if is_unit_train_action(action):
        tags.append("production")
    if "scv" in lower or "commandcentertrain_scv" in lower:
        tags.append("worker")
    if "marine" in lower:
        tags.append("marine")
    if "supplydepot" in lower or "supply depot" in lower:
        tags.append("supply")
    if "barracks" in lower:
        tags.append("barracks")
    return tags


def _diagnostic_text(facts: Dict[str, int]) -> str:
    notes = []
    supply_left = facts.get("supply_left")
    minerals = facts.get("minerals")
    army = facts.get("army")
    workers = facts.get("workers")
    if supply_left is not None and supply_left <= 1:
        notes.append("supply blocked risk")
    if army is not None and army < 8:
        notes.append("army too small for reliable attack")
    if minerals is not None and minerals >= 300 and (army is None or army < 8):
        notes.append("minerals floating while army is low")
    if workers is not None and workers < 18:
        notes.append("worker count still low")
    return "; ".join(notes[:3]) if notes else "stable; follow build order priorities"


class SC2CompactAssetMemory(BaseMemory):
    """BIAgent-style compact memory for SC2 decisions.

    It stores only short reusable assets instead of raw observation history.
    """

    def __init__(
        self,
        max_prompt_items: int = None,
        max_assets: int = None,
        max_chain_actions: int = None,
        quality_threshold: float = None,
        match_threshold: float = None,
        log_path: str = "",
    ):
        self.max_prompt_items = int(max_prompt_items) if max_prompt_items is not None else int(os.getenv("SC2_ASSET_MEMORY_MAX_PROMPT_ITEMS", "3"))
        self.max_assets = int(max_assets) if max_assets is not None else int(os.getenv("SC2_ASSET_MEMORY_MAX_ASSETS", "64"))
        self.max_chain_actions = int(max_chain_actions) if max_chain_actions is not None else int(os.getenv("SC2_ASSET_MEMORY_MAX_CHAIN_ACTIONS", "5"))
        self.quality_threshold = float(quality_threshold) if quality_threshold is not None else float(os.getenv("SC2_ASSET_MEMORY_QUALITY_THRESHOLD", "0.65"))
        self.match_threshold = float(match_threshold) if match_threshold is not None else float(os.getenv("SC2_ASSET_MEMORY_MATCH_THRESHOLD", "0.35"))
        self.log_path = log_path or ""
        self._assets: List[Dict[str, Any]] = []
        self._recent_chains: List[List[str]] = []
        self._step_counts: List[int] = []
        self.keys = None
        self.batch_size = 0
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)

    def __len__(self):
        return len(self._assets)

    def __getitem__(self, idx: int):
        return self._assets[idx]

    def reset(self, batch_size: int):
        self.batch_size = batch_size
        self._recent_chains = [[] for _ in range(batch_size)]
        self._step_counts = [0 for _ in range(batch_size)]
        self.keys = None

    def store(self, record: Dict[str, List[Any]]):
        if self.keys is None:
            self.keys = list(record.keys())
        assert self.keys == list(record.keys())

        for env_idx in range(self.batch_size):
            rec = {key: record[key][env_idx] for key in self.keys}
            self._step_counts[env_idx] += 1
            step_num = self._step_counts[env_idx]
            self._remember_action(env_idx, rec.get("action"))

            for asset in self._make_assets(env_idx, step_num, rec):
                stored_asset = self._upsert_asset(asset)
                if stored_asset is not None:
                    self._write_asset_log(env_idx, step_num, stored_asset)

    def fetch(
        self,
        history_length: int,
        current_obs: List[Any],
        action_key: str = "action",
    ) -> Tuple[List[str], List[int]]:
        memory_contexts, valid_lengths = [], []
        for env_idx, obs in enumerate(current_obs):
            matches = self._retrieve_assets(obs)
            valid_lengths.append(len(matches))
            memory_contexts.append(self._format_context(obs, matches))
        return memory_contexts, valid_lengths

    def _remember_action(self, env_idx: int, action: Any) -> None:
        summary = summarize_action(action)
        if not summary:
            return
        chain = self._recent_chains[env_idx]
        if chain and chain[-1] == summary:
            return
        chain.append(summary)
        if len(chain) > self.max_chain_actions:
            del chain[: len(chain) - self.max_chain_actions]

    def _make_assets(self, env_idx: int, step_num: int, rec: Dict[str, Any]) -> List[Dict[str, Any]]:
        obs = rec.get("text_obs")
        action = rec.get("action")
        reward = _as_float(rec.get("reward"))
        done = _as_bool(rec.get("done"))
        won = _as_float(rec.get("won"))
        valid = _as_bool(rec.get("is_action_valid"))
        failure_reason = infer_failure_reason(action, obs, valid, rec.get("failure_reason"))
        facts = extract_sc2_facts(obs)
        tags = _dedup_keep_order(_fact_tags(facts) + _action_tags(action))
        action_summary = summarize_action(action)
        chain = self._chain_text(env_idx)

        assets = []

        if not valid:
            reason = failure_reason or "invalid action"
            assets.append(
                self._asset(
                    kind="action_skill",
                    polarity="avoid",
                    text=f"avoid {_shorten(action_summary or _to_text(action), 90)}; reason: {_shorten(reason, 70)}",
                    condition=format_sc2_facts(facts),
                    tags=tags + ["invalid_action"],
                    quality=0.9 if reason != "invalid action" else 0.75,
                    action_chain=chain,
                    failure_signals=[reason],
                    step_num=step_num,
                )
            )

        if is_attack_action(action) and facts.get("army", 99) < 8:
            assets.append(
                self._asset(
                    kind="action_skill",
                    polarity="avoid",
                    text="avoid attack before army >= 8",
                    condition=format_sc2_facts(facts),
                    tags=tags + ["attack_too_early"],
                    quality=0.85,
                    action_chain=chain,
                    failure_signals=["army too small for reliable attack"],
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
                    condition=format_sc2_facts(facts),
                    tags=tags + ["resource_rule"],
                    quality=0.8,
                    action_chain="",
                    failure_signals=[failure_reason],
                    step_num=step_num,
                )
            )

        if done and won > 0:
            assets.append(
                self._asset(
                    kind="action_skill",
                    polarity="do",
                    text=f"reuse winning chain: {_shorten(chain or action_summary, 120)}",
                    condition=format_sc2_facts(facts),
                    tags=tags + ["win_pattern"],
                    quality=0.95,
                    action_chain=chain,
                    failure_signals=[],
                    step_num=step_num,
                )
            )
        elif done:
            assets.append(
                self._asset(
                    kind="action_skill",
                    polarity="avoid",
                    text=f"avoid repeating losing chain: {_shorten(chain or action_summary, 120)}",
                    condition=format_sc2_facts(facts),
                    tags=tags + ["loss_pattern"],
                    quality=0.8,
                    action_chain=chain,
                    failure_signals=[failure_reason or "episode ended without win"],
                    step_num=step_num,
                )
            )
        elif reward >= 15:
            assets.append(
                self._asset(
                    kind="action_skill",
                    polarity="do",
                    text=f"positive chain: {_shorten(chain or action_summary, 120)}",
                    condition=format_sc2_facts(facts),
                    tags=tags + ["positive_reward"],
                    quality=0.75,
                    action_chain=chain,
                    failure_signals=[],
                    step_num=step_num,
                )
            )
        elif reward <= -20:
            assets.append(
                self._asset(
                    kind="action_skill",
                    polarity="avoid",
                    text=f"negative chain: {_shorten(chain or action_summary, 120)}",
                    condition=format_sc2_facts(facts),
                    tags=tags + ["negative_reward"],
                    quality=0.75,
                    action_chain=chain,
                    failure_signals=[failure_reason or f"reward={reward:.1f}"],
                    step_num=step_num,
                )
            )

        if facts.get("supply_left", 2) <= 1 or facts.get("army", 99) < 8 or facts.get("minerals", 0) >= 300:
            assets.append(
                self._asset(
                    kind="action_skill",
                    polarity="diagnostic",
                    text=_diagnostic_text(facts),
                    condition=format_sc2_facts(facts),
                    tags=tags + ["diagnostic_check"],
                    quality=0.7,
                    action_chain="",
                    failure_signals=[],
                    step_num=step_num,
                )
            )

        return [asset for asset in assets if asset["quality"] >= self.quality_threshold]

    def _asset(
        self,
        kind: str,
        polarity: str,
        text: str,
        condition: str,
        tags: List[str],
        quality: float,
        action_chain: str,
        failure_signals: List[str],
        step_num: int,
    ) -> Dict[str, Any]:
        return {
            "kind": kind,
            "polarity": polarity,
            "text": _shorten(text, 160),
            "condition": _shorten(condition, 120),
            "tags": _dedup_keep_order(tags),
            "quality": float(quality),
            "count": 1,
            "action_chain": _shorten(action_chain, 180),
            "failure_signals": [_shorten(signal, 80) for signal in failure_signals if signal],
            "last_step": step_num,
        }

    def _asset_key(self, asset: Dict[str, Any]) -> str:
        tag_text = ",".join(sorted(asset.get("tags", []))[:4])
        return f"{asset.get('kind')}:{asset.get('polarity')}:{asset.get('text')}:{tag_text}"

    def _upsert_asset(self, asset: Dict[str, Any]):
        key = self._asset_key(asset)
        for stored in self._assets:
            if stored.get("key") == key:
                stored["count"] = int(stored.get("count", 1)) + 1
                stored["quality"] = max(float(stored.get("quality", 0.0)), float(asset.get("quality", 0.0)))
                stored["last_step"] = asset.get("last_step")
                return stored

        asset["key"] = key
        self._assets.append(asset)
        if len(self._assets) > self.max_assets:
            self._assets.sort(key=lambda item: (float(item.get("quality", 0.0)), int(item.get("count", 1))))
            del self._assets[: len(self._assets) - self.max_assets]
        return asset

    def _retrieve_assets(self, obs: Any) -> List[Dict[str, Any]]:
        facts = extract_sc2_facts(obs)
        current_tags = set(_fact_tags(facts))
        scored = []
        for asset in self._assets:
            asset_tags = set(asset.get("tags", []))
            if not asset_tags:
                continue
            overlap = len(current_tags & asset_tags)
            if overlap <= 0:
                continue
            quality = float(asset.get("quality", 0.0))
            tag_score = overlap / max(1, min(len(asset_tags), 4))
            score = 0.55 * quality + 0.45 * tag_score
            if score >= self.match_threshold:
                scored.append((score, asset))
        scored.sort(key=lambda item: (item[0], float(item[1].get("quality", 0.0)), int(item[1].get("count", 1))), reverse=True)
        return [asset for _, asset in scored[: self.max_prompt_items]]

    def _format_context(self, obs: Any, matches: List[Dict[str, Any]]) -> str:
        facts = extract_sc2_facts(obs)
        lines = ["SC2 decision memory:"]
        lines.append(f"- Situation: {format_sc2_facts(facts)}")
        if matches:
            lines.append("- Matched assets: " + "; ".join(self._format_asset(asset) for asset in matches))
        lines.append(f"- Decision check: {self._decision_check(facts, matches)}")
        return "\n".join(lines)

    def _format_asset(self, asset: Dict[str, Any]) -> str:
        polarity = asset.get("polarity", "fact")
        text = asset.get("text", "")
        condition = asset.get("condition", "")
        if condition and len(condition) <= 40:
            return _shorten(f"{polarity}: {text} when {condition}", 120)
        return _shorten(f"{polarity}: {text}", 120)

    def _decision_check(self, facts: Dict[str, int], matches: List[Dict[str, Any]]) -> str:
        guidance = next_action_guidance(facts)
        avoid_items = [asset.get("text", "") for asset in matches if asset.get("polarity") == "avoid"]
        fact_items = [asset.get("text", "") for asset in matches if asset.get("polarity") == "fact"]
        pieces = [guidance]
        if avoid_items:
            pieces.append("matched risk: " + _shorten("; ".join(avoid_items[:2]), 120))
        if fact_items:
            pieces.append(_shorten("; ".join(fact_items[:1]), 100))
        return _shorten("; ".join(piece for piece in pieces if piece), 220)

    def _chain_text(self, env_idx: int) -> str:
        chain = self._recent_chains[env_idx] if env_idx < len(self._recent_chains) else []
        return " -> ".join(chain[-self.max_chain_actions :])

    def _write_asset_log(self, env_idx: int, step_num: int, asset: Dict[str, Any]) -> None:
        if not self.log_path:
            return
        row = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "step_index": step_num,
            "env_idx": env_idx,
            "asset": asset,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
