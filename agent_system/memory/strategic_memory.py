# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        return ""
    parts = []
    if "time_s" in facts:
        parts.append(f"time={facts['time_s'] // 60:02d}:{facts['time_s'] % 60:02d}")
    for key in ("minerals", "vespene", "army", "workers", "supply_left", "enemy_units", "enemy_structures"):
        if key in facts:
            parts.append(f"{key}={facts[key]}")
    return ", ".join(parts)


def extract_dcmrta_facts(obs: Any) -> Dict[str, Any]:
    text = _to_text(obs)
    facts: Dict[str, Any] = {}

    time_match = re.search(r"Time:\s*([0-9.]+)", text, flags=re.IGNORECASE)
    if time_match:
        try:
            facts["time"] = float(time_match.group(1))
        except ValueError:
            pass

    agent_match = re.search(r"Agent:\s*(\d+)", text, flags=re.IGNORECASE)
    if agent_match:
        facts["agent"] = int(agent_match.group(1))

    group_match = re.search(r"Group size at this location:\s*(\d+)", text, flags=re.IGNORECASE)
    if group_match:
        facts["group_size"] = int(group_match.group(1))

    location_match = re.search(r"Agent location:\s*\(([^)]+)\)", text, flags=re.IGNORECASE)
    if location_match:
        facts["agent_location"] = location_match.group(1)

    action_ids = re.findall(r"^\s*(\d+):\s*task\s+(\d+),\s*remaining=\[([^\]]*)\]", text, flags=re.MULTILINE)
    if action_ids:
        facts["candidate_count"] = len(action_ids)
        facts["candidate_ids"] = [int(action_id) for action_id, _, _ in action_ids[:12]]

    completed_match = re.search(r"Completed\s+(\d+)/(\d+)\s+tasks", text, flags=re.IGNORECASE)
    if completed_match:
        facts["completed"] = int(completed_match.group(1))
        facts["total"] = int(completed_match.group(2))

    return facts


def format_dcmrta_facts(facts: Dict[str, Any]) -> str:
    if not facts:
        return ""

    parts = []
    if "time" in facts:
        parts.append(f"time={facts['time']:.2f}")
    if "agent" in facts:
        parts.append(f"agent={facts['agent']}")
    if "group_size" in facts:
        parts.append(f"group_size={facts['group_size']}")
    if "agent_location" in facts:
        parts.append(f"agent_location=({facts['agent_location']})")
    if "candidate_count" in facts:
        ids = ",".join(str(x) for x in facts.get("candidate_ids", []))
        parts.append(f"candidate_tasks={facts['candidate_count']} ids=[{ids}]")
    if "completed" in facts and "total" in facts:
        parts.append(f"completed={facts['completed']}/{facts['total']}")
    return ", ".join(parts)


class StrategicMemory(BaseMemory):
    """
    Memory interface inspired by memory-enhanced agency:
    - working memory: compact current-state variables and recent actions
    - episodic memory: sparse salient events across the episode

    This is intentionally separate from memory.py. Existing SimpleMemory users
    are unaffected unless an environment explicitly imports StrategicMemory.
    """

    def __init__(
        self,
        max_obs_chars: int = None,
        max_action_chars: int = None,
        max_episodic_items: int = None,
    ):
        self._data = None
        self._episodes = None
        self.keys = None
        self.batch_size = 0
        self.max_obs_chars = max_obs_chars or int(os.getenv("AGENT_MEMORY_MAX_OBS_CHARS", "700"))
        self.max_action_chars = max_action_chars or int(os.getenv("AGENT_MEMORY_MAX_ACTION_CHARS", "240"))
        self.max_episodic_items = max_episodic_items or int(os.getenv("AGENT_MEMORY_MAX_EPISODIC_ITEMS", "4"))

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

    def fetch(
        self,
        history_length: int,
        obs_key: str = "text_obs",
        action_key: str = "action",
        domain: str = "generic",
    ) -> Tuple[List[str], List[int]]:
        memory_contexts, valid_lengths = [], []

        for env_idx in range(self.batch_size):
            recent = self._data[env_idx][-history_length:] if history_length > 0 else []
            valid_len = len(recent)
            start_idx = len(self._data[env_idx]) - valid_len

            sections = []
            strategic_context = self.fetch_strategic_context(env_idx, recent, obs_key, action_key, domain=domain)
            if strategic_context:
                sections.append(strategic_context)

            for j, rec in enumerate(recent):
                step_num = start_idx + j + 1
                sections.append(self._format_record(step_num, rec, obs_key, action_key))

            memory_contexts.append("\n".join(section for section in sections if section))
            valid_lengths.append(valid_len)

        return memory_contexts, valid_lengths

    def fetch_strategic_context(
        self,
        env_idx: int,
        recent: List[Dict[str, Any]] = None,
        obs_key: str = "text_obs",
        action_key: str = "action",
        domain: str = "generic",
    ) -> str:
        source = recent or self._data[env_idx][-1:]
        lines = []

        if source:
            if domain == "dcmrta":
                facts = format_dcmrta_facts(extract_dcmrta_facts(source[-1].get(obs_key)))
            else:
                facts = format_sc2_facts(extract_sc2_facts(source[-1].get(obs_key)))
            if facts:
                lines.append(f"Working memory: current state -> {facts}")

            actions = [
                _shorten(rec.get(action_key), 80)
                for rec in source[-3:]
                if _to_text(rec.get(action_key))
            ]
            if actions:
                lines.append("Working memory: recent actions -> " + " | ".join(actions))

        episodes = self._episodes[env_idx] if self._episodes else []
        if episodes:
            lines.append("Episodic memory: " + " | ".join(episodes[-self.max_episodic_items:]))

        return "\n".join(lines)

    def _format_record(self, step_num: int, rec: Dict[str, Any], obs_key: str, action_key: str) -> str:
        obs = _shorten(rec.get(obs_key), self.max_obs_chars)
        action = _shorten(rec.get(action_key), self.max_action_chars)
        return f"[Observation {step_num}: '{obs}', Action {step_num}: '{action}']"

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
        obs = rec.get("text_obs") or rec.get("obs") or rec.get("Observation")
        action = _to_text(rec.get("action"))
        dcmrta_facts = format_dcmrta_facts(extract_dcmrta_facts(obs))
        facts = dcmrta_facts or format_sc2_facts(extract_sc2_facts(obs))

        tags = []
        for key in ("done", "won", "success"):
            if key in rec:
                tags.append(f"{key}={rec[key]}")
        if rec.get("is_action_valid") is False or rec.get("action_is_valid") is False:
            tags.append("invalid_action")
        if "success_rate" in rec:
            tags.append(f"task_success_rate={rec['success_rate']}")
        if "finished_tasks" in rec and "total_tasks" in rec:
            tags.append(f"finished={rec['finished_tasks']}/{rec['total_tasks']}")
        if any(token in action.lower() for token in ("launch offense", "attack", "attacking instruction")):
            tags.append("attack_order")
        obs_text = _to_text(obs)
        if "Result.Victory" in obs_text or "game_result: Victory" in obs_text:
            tags.append("victory")
        if "Result.Defeat" in obs_text or "game_result: Defeat" in obs_text:
            tags.append("defeat")
        if "reward" in rec:
            tags.append(f"reward={rec['reward']}")

        if not tags:
            return ""

        parts = [f"Step {step_num}: {'; '.join(tags)}"]
        if facts:
            parts.append(facts)
        if action:
            parts.append(f"action={_shorten(action, 120)}")
        return "; ".join(parts)


class StrategicSearchMemory(StrategicMemory):
    def _format_record(self, step_num: int, rec: Dict[str, Any], obs_key: str, action_key: str) -> str:
        action = _shorten(rec.get(action_key), self.max_action_chars)
        obs = _shorten(rec.get(obs_key), self.max_obs_chars)
        return f"Step {step_num}: {action} {obs}\n"
