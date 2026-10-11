import json
import math
import os
import re
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional


class SC2LongLoopMemory:
	"""Cross-rollout SC2 memory updated from completed rollout records."""

	_RESOURCE_MEMORY_NAMES = (
		"Mineral collection rate",
		"Vespene collection rate",
		"Mineral utilization",
		"Vespene utilization",
		"Total supply growth rate",
	)

	def __init__(
		self,
		llm_generate: Optional[Callable[[str], Any]] = None,
		initial_resource_memory: Optional[List[Any]] = None,
		initial_army_memory: Optional[List[Any]] = None,
		max_items_per_section: int = 8,
		max_pending_rollouts: int = 64,
		memory_path: Optional[str] = None,
	):
		self._TELEMETRY_FIELDS = (
			"game_time_seconds",
			"minerals",
			"vespene",
			"cumulative_minerals_collected",
			"cumulative_vespene_collected",
			"cumulative_minerals_spent",
			"cumulative_vespene_spent",
			"supply_used",
			"supply_cap",
			"supply_army",
			"own_units",
			"own_structures",
			"visible_enemy_units",
			"visible_enemy_structures",
		)
		self.max_items_per_section = max(1, int(max_items_per_section))
		self.max_pending_rollouts = max(1, int(max_pending_rollouts))
		self.resource_memory = deepcopy(
			initial_resource_memory
			if initial_resource_memory is not None
			else [None] * len(self._RESOURCE_MEMORY_NAMES)
		)
		self.army_memory = deepcopy(initial_army_memory if initial_army_memory is not None else [])
		self.memory_path = os.fspath(memory_path) if memory_path else ""
		if self.memory_path and os.path.isfile(self.memory_path):
			with open(self.memory_path, "r", encoding="utf-8") as stream:
				restored = self._parse_model_output(stream.read())
			self.resource_memory = restored["resource_memory"]
			self.army_memory = restored["army_memory"]
		self.memory = {
			"resource_memory": self.resource_memory,
			"army_memory": self.army_memory,
		}
		self.llm_generate = llm_generate
		self._rollout_records: Dict[str, List[Dict[str, Any]]] = {}
		self._done_rollouts = set()
		self._completed_rollouts = set()
		self._pending_rollout_updates: List[str] = []
		self._rollout_generation = 0
		self.batch_size = 0
		self._previous_fetch_telemetry: List[Optional[Mapping[str, Any]]] = []

	def reset(self, batch_size: Optional[int] = None) -> None:
		"""Start a new collection batch without clearing learned memory."""
		self.batch_size = batch_size
		self._rollout_generation += 1
		pending = set(self._pending_rollout_updates)
		for rollout_id in list(self._rollout_records):
			if rollout_id not in pending:
				del self._rollout_records[rollout_id]
		self._done_rollouts.clear()
		self._completed_rollouts.clear()
		self._previous_fetch_telemetry = [None for _ in range(batch_size or 0)]

	def store(self, record: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
		"""Accumulate one step per environment and finalize slots when done."""
		if "done" not in record:
			raise ValueError("memory record must contain a 'done' batch field")
		batch_size = len(record["done"])
		if self.batch_size and batch_size != self.batch_size:
			raise ValueError("memory record batch size changed without reset()")
		if not self.batch_size:
			self.batch_size = batch_size

		rollout_ids = record.get("rollout_id")
		if rollout_ids is None:
			rollout_ids = [
				f"batch-{self._rollout_generation}:env-{index}"
				for index in range(batch_size)
			]
		elif len(rollout_ids) != batch_size:
			raise ValueError("rollout_id must contain one ID per environment slot")

		telemetry_batch = record.get("telemetry")
		won_batch = record.get("won")
		truncated_batch = record.get("truncated")
		for key, values in (
			("telemetry", telemetry_batch),
			("won", won_batch),
			("truncated", truncated_batch),
		):
			if values is not None and len(values) != batch_size:
				raise ValueError(f"record field {key!r} has a different batch size")

		updates = {}
		for index, rollout_id in enumerate(rollout_ids):
			rollout_id = str(rollout_id)
			if rollout_id in self._done_rollouts:
				continue
			if truncated_batch is not None and bool(truncated_batch[index]):
				self._rollout_records.pop(rollout_id, None)
				if rollout_id in self._pending_rollout_updates:
					self._pending_rollout_updates.remove(rollout_id)
				self._done_rollouts.add(rollout_id)
				continue

			transition = {}
			if telemetry_batch is not None and isinstance(telemetry_batch[index], Mapping):
				transition["telemetry"] = {
					key: deepcopy(telemetry_batch[index][key])
					for key in self._TELEMETRY_FIELDS
					if key in telemetry_batch[index]
				}

			is_done = bool(record["done"][index])
			if is_done:
				transition["won"] = bool(won_batch[index]) if won_batch is not None else False
			self._rollout_records.setdefault(rollout_id, []).append(transition)

			if is_done:
				self._done_rollouts.add(rollout_id)
				self._pending_rollout_updates.append(rollout_id)

		return updates

	def finalize_batch(
		self,
		llm_generate: Optional[Callable[[str], Any]] = None,
	) -> Dict[str, Dict[str, Any]]:
		"""Update completed rollouts after every episode in the collector batch ends."""
		updates = {}
		generate = llm_generate or self.llm_generate
		if generate is not None:
			for rollout_id in list(self._pending_rollout_updates):
				try:
					updates[rollout_id] = self.update_memory(rollout_id, llm_generate=generate)
				except Exception as exc:
					warnings.warn(
						f"Long-loop memory update failed for {rollout_id}; it will be retried "
						f"after the next rollout batch: {exc}",
						RuntimeWarning,
					)

		while len(self._pending_rollout_updates) > self.max_pending_rollouts:
			oldest = self._pending_rollout_updates.pop(0)
			self._rollout_records.pop(oldest, None)
			warnings.warn(
				f"Dropping oldest pending long-loop memory rollout {oldest} because "
				f"the pending limit ({self.max_pending_rollouts}) was reached",
				RuntimeWarning,
			)
		return updates

	def update_memory(
		self,
		rollout_id: str,
		llm_generate: Optional[Callable[[str], Any]] = None,
	) -> Dict[str, Any]:
		"""Use the rollout model callback to revise memory from a completed rollout."""
		rollout_id = str(rollout_id)
		if rollout_id in self._completed_rollouts:
			return deepcopy(self.memory)
		records = self._rollout_records.get(rollout_id)
		if not records:
			raise KeyError(f"no stored records for rollout {rollout_id!r}")

		generate = llm_generate or self.llm_generate
		if generate is None:
			raise RuntimeError("provide llm_generate to update memory")

		prompt = self._build_update_prompt(rollout_id, records)
		updated_memory = self._parse_model_output(generate(prompt))
		self._save_memory(updated_memory)
		self.resource_memory = deepcopy(
			updated_memory.get("resource_memory", self.resource_memory)
		)
		self.army_memory = deepcopy(
			updated_memory.get("army_memory", self.army_memory)
		)
		self.memory = {
			"resource_memory": self.resource_memory,
			"army_memory": self.army_memory,
		}
		del self._rollout_records[rollout_id]
		self._completed_rollouts.add(rollout_id)
		if rollout_id in self._pending_rollout_updates:
			self._pending_rollout_updates.remove(rollout_id)
		return deepcopy(self.memory)

	def _save_memory(self, memory: Mapping[str, Any]) -> None:
		if not self.memory_path:
			return

		parent = os.path.dirname(os.path.abspath(self.memory_path))
		os.makedirs(parent, exist_ok=True)
		temporary_path = None
		try:
			with tempfile.NamedTemporaryFile(
				mode="w",
				encoding="utf-8",
				dir=parent,
				prefix=".long_loop_memory_",
				suffix=".tmp",
				delete=False,
			) as stream:
				temporary_path = stream.name
				json.dump(memory, stream, ensure_ascii=False, indent=2)
				stream.write("\n")
			os.replace(temporary_path, self.memory_path)
		except Exception:
			if temporary_path and os.path.exists(temporary_path):
				os.unlink(temporary_path)
			raise

	def fetch(
		self,
		history_length: int,
		current_obs: Sequence[Any],
		action_key: str = "action",
		telemetry: Optional[Sequence[Optional[Mapping[str, Any]]]] = None,
	) -> tuple[List[str], List[int]]:
		"""Return metric warnings and counters relevant to each environment state."""
		if telemetry is None:
			telemetry = [None for _ in current_obs]
		if len(telemetry) != len(current_obs):
			raise ValueError("telemetry and current_obs must have the same batch size")
		if len(self._previous_fetch_telemetry) != len(current_obs):
			self._previous_fetch_telemetry = [None for _ in current_obs]

		def render(item: Any) -> str:
			if isinstance(item, str):
				return item
			return json.dumps(item, ensure_ascii=False, default=str)

		def normalize(text: Any) -> str:
			return re.sub(r"[^a-z0-9]+", "", str(text).lower())

		def is_too_low(value: Any) -> bool:
			text = render(value).lower()
			return any(marker in text for marker in ("too low", "below target", "insufficient"))

		def number(value: Any) -> Optional[float]:
			if isinstance(value, (int, float)) and math.isfinite(float(value)):
				return float(value)
			match = re.search(r"[-+]?\d+(?:\.\d+)?", render(value).replace(",", ""))
			if not match:
				return None
			parsed = float(match.group(0))
			return parsed if math.isfinite(parsed) else None

		def metric_values(
			point: Mapping[str, Any],
			previous: Optional[Mapping[str, Any]],
		) -> List[Optional[float]]:
			def read(key: str) -> Optional[float]:
				try:
					value = float(point.get(key))
				except (TypeError, ValueError):
					return None
				return value if math.isfinite(value) else None

			def cumulative_rate(key: str) -> Optional[float]:
				elapsed = read("game_time_seconds")
				total = read(key)
				if elapsed is None or elapsed <= 0 or total is None:
					return None
				return total / (elapsed / 60.0)

			def utilization(spent_key: str, collected_key: str) -> Optional[float]:
				spent = read(spent_key)
				collected = read(collected_key)
				if spent is None or collected is None or collected <= 0:
					return None
				return spent / collected * 100.0

			growth = None
			if previous is not None:
				try:
					previous_time = float(previous.get("game_time_seconds"))
					previous_supply = float(previous.get("supply_used"))
					current_time = float(point.get("game_time_seconds"))
					current_supply = float(point.get("supply_used"))
					elapsed_minutes = (current_time - previous_time) / 60.0
					if elapsed_minutes > 0:
						growth = (current_supply - previous_supply) / elapsed_minutes
				except (TypeError, ValueError, OverflowError):
					growth = None

			return [
				cumulative_rate("cumulative_minerals_collected"),
				cumulative_rate("cumulative_vespene_collected"),
				utilization("cumulative_minerals_spent", "cumulative_minerals_collected"),
				utilization("cumulative_vespene_spent", "cumulative_vespene_collected"),
				growth,
			]

		def metric_reminder(index: int, current: Optional[float]) -> Optional[str]:
			if current is None or index >= len(self.resource_memory):
				return None
			recommendation = self.resource_memory[index]
			target = number(recommendation)
			if target is None or target <= 0:
				return None
			low_tagged = is_too_low(recommendation)
			if current < target * 0.8:
				reason = "current value is below 80% of the recommendation"
			elif low_tagged and target * 0.9 <= current <= target * 1.1:
				reason = "the recommendation is already marked too low and current value is within 10% of it"
			else:
				return None
			return (
				f"Metric below target: {self._RESOURCE_MEMORY_NAMES[index]} is {current:.2f}; "
				f"recommended target is {target:.2f} ({reason}). Improve this metric."
			)

		memory_contexts = []
		valid_lengths = []
		for index, point in enumerate(telemetry):
			previous = self._previous_fetch_telemetry[index]
			values = metric_values(point, previous) if isinstance(point, Mapping) else [None] * len(self._RESOURCE_MEMORY_NAMES)
			resource_lines = [
				reminder
				for metric_index, value in enumerate(values)
				if (reminder := metric_reminder(metric_index, value)) is not None
			]

			enemy_units = point.get("visible_enemy_units", {}) if isinstance(point, Mapping) else {}
			if isinstance(enemy_units, Mapping):
				detected_units = [
					unit for unit, count in enemy_units.items()
					if isinstance(count, (int, float)) and count > 0
				]
			else:
				detected_units = list(enemy_units or [])
			army_lines = []
			for item in self.army_memory:
				if not isinstance(item, Mapping):
					continue
				opponent_units = item.get("opponent_units", [])
				if isinstance(opponent_units, str):
					opponent_units = [opponent_units]
				matched_units = [
					str(unit)
					for unit in opponent_units
					if normalize(unit) and normalize(unit) in {
						normalize(detected) for detected in detected_units
					}
				]
				if not matched_units:
					continue
				counters = item.get("counters", [])
				if isinstance(counters, str):
					counters = [counters]
				counter_text = ", ".join(str(counter) for counter in counters if str(counter).strip())
				fallback = str(item.get("fallback", "")).strip()
				rationale = str(item.get("rationale", "")).strip()
				line = (
					f"Opponent detected: {', '.join(matched_units)}. "
					f"Recommended counters: {counter_text or 'none specified'}. "
					f"Fallback: {fallback or 'produce or build a suitable counter'}"
				)
				if rationale:
					line += f" Rationale: {rationale}"
				army_lines.append(line)

			lines = [*resource_lines]
			lines.extend(f"Unit matchup response: {item}" for item in army_lines)
			context = "\n".join(["Long-term memory:", *lines]) if lines else ""
			memory_contexts.append(context)
			valid_lengths.append(len(resource_lines) + len(army_lines))
			if isinstance(point, Mapping):
				self._previous_fetch_telemetry[index] = deepcopy(point)

		return memory_contexts, valid_lengths

	def _build_update_prompt(
		self,
		rollout_id: str,
		records: List[Dict[str, Any]],
	) -> str:
		telemetry_points = [
			record["telemetry"]
			for record in records
			if isinstance(record.get("telemetry"), Mapping)
		]

		def read_number(point: Mapping[str, Any], key: str) -> Optional[float]:
			try:
				value = float(point.get(key))
			except (TypeError, ValueError):
				return None
			return value if math.isfinite(value) else None

		def per_game_minute(total_key: str) -> Optional[float]:
			if not telemetry_points:
				return None
			latest = telemetry_points[-1]
			elapsed_minutes = read_number(latest, "game_time_seconds")
			total = read_number(latest, total_key)
			if elapsed_minutes is None or elapsed_minutes <= 0 or total is None:
				return None
			return total / (elapsed_minutes / 60.0)

		def utilization_percent(spent_key: str, collected_key: str) -> Optional[float]:
			if not telemetry_points:
				return None
			latest = telemetry_points[-1]
			spent = read_number(latest, spent_key)
			collected = read_number(latest, collected_key)
			if spent is None or collected is None or collected <= 0:
				return None
			return spent / collected * 100.0

		def format_metric(value: Optional[float], unit: str = "") -> str:
			if value is None:
				return "unavailable (insufficient telemetry)"
			return f"{value:.2f}{(' ' + unit) if unit else ''}"

		total_population_growth_rate = None
		if len(telemetry_points) >= 2:
			first = telemetry_points[0]
			last = telemetry_points[-1]
			first_time = read_number(first, "game_time_seconds")
			last_time = read_number(last, "game_time_seconds")
			if first_time is not None and last_time is not None and last_time > first_time:
				elapsed_minutes = (last_time - first_time) / 60.0
				first_supply = read_number(first, "supply_used")
				last_supply = read_number(last, "supply_used")
				if first_supply is not None and last_supply is not None:
					total_population_growth_rate = (last_supply - first_supply) / elapsed_minutes

		measured_metrics = (
			"Measured rollout metrics (rates are per game minute):\n"
			f"- Mineral collection rate: {format_metric(per_game_minute('cumulative_minerals_collected'), 'minerals/min')}\n"
			f"- Vespene collection rate: {format_metric(per_game_minute('cumulative_vespene_collected'), 'vespene/min')}\n"
			f"- Mineral utilization (spent / collected): {format_metric(utilization_percent('cumulative_minerals_spent', 'cumulative_minerals_collected'), '%')}\n"
			f"- Vespene utilization (spent / collected): {format_metric(utilization_percent('cumulative_vespene_spent', 'cumulative_vespene_collected'), '%')}\n"
			f"- Total supply growth rate: {format_metric(total_population_growth_rate, 'supply/min')}"
		)

		last_record = records[-1]
		try:
			won = float(last_record.get("won", 0.0)) > 0
		except (TypeError, ValueError):
			won = False
		outcome_guidance = (
			"The game was won. Use this game's mineral and vespene collection rates, mineral and vespene utilization, "
			"supply growth rate, and both players' unit compositions at every model decision point "
			"to revise the recommended economic targets and unit matchup rules. Check whether any "
			"previously recommended target was too low, and retain matchup rules supported by "
			"the game evidence and sound reasoning. "
			if won
			else "The game was lost. First determine whether either collection rate, either resource utilization ratio, "
			"or either supply growth rate could have been improved substantially. If so, mark the corresponding "
			"recommended target as too low. Only if there is no substantial room for economic improvement, "
			"analyze both players' units using StarCraft II mechanics and record effective matchup responses. "
		)
		return (
			"Update the long-term StarCraft II strategy memory. Keep only reusable guidance supported by the rollout evidence. "
			"Write all memory values in clear, concise English.\n"
			f"{outcome_guidance}\n\n"
			"Return a JSON object containing exactly two keys: resource_memory and army_memory.\n"
			"resource_memory must contain five recommended values in this exact order:\n"
			"1. Mineral collection rate\n"
			"2. Vespene collection rate\n"
			"3. Mineral utilization (spent minerals / collected minerals)\n"
			"4. Vespene utilization (spent vespene / collected vespene)\n"
			"5. Total supply growth rate\n"
			"When a recommended target is too low, explicitly include the label 'too low' in that value.\n"
			"army_memory must be a list of JSON objects. Each object must contain: "
			"opponent_units (a list of exact enemy unit names as they appear in the visible-enemy telemetry), "
			"counters (a list of our counter units or structures), fallback (what to produce or build if counters are unavailable), "
			"and rationale (why the counters work). Only opponent_units identifies the enemy trigger; never place our counters there. "
			"Example: {\"opponent_units\":[\"Siege Tank\"],\"counters\":[\"Viking\"],"
			"\"fallback\":\"Build a Starport and produce Vikings\",\"rationale\":\"Vikings can attack the tank from the air\"}.\n\n"
			"Current resource memory (values are listed in the order defined above):\n"
			+ "\n".join(
				f"- {name}: {self.resource_memory[index] if index < len(self.resource_memory) else None}"
				for index, name in enumerate(self._RESOURCE_MEMORY_NAMES)
			)
			+ "\n\n"
			f"Current army matchup memory:\n{json.dumps(self.army_memory, ensure_ascii=False, default=str)}\n\n"
			f"Completed rollout ID: {rollout_id}\n"
			f"{measured_metrics}\n"
			f"Rollout records:\n{json.dumps(records, ensure_ascii=False, default=str)}"
		)

	def _parse_model_output(self, output: Any) -> Dict[str, Any]:
		if isinstance(output, Mapping):
			parsed = output
		elif isinstance(output, str):
			text = output.strip()
			if text.startswith("```"):
				text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
			try:
				parsed = json.loads(text)
			except json.JSONDecodeError as exc:
				raise ValueError("model output must be valid JSON") from exc
		else:
			raise ValueError("llm_generate must return a JSON string or mapping")

		if not isinstance(parsed, Mapping):
			raise ValueError("model output must be a JSON object")
		if "resource_memory" not in parsed or "army_memory" not in parsed:
			raise ValueError("model output must contain resource_memory and army_memory")

		resource_values = parsed["resource_memory"]
		if not isinstance(resource_values, Sequence) or isinstance(resource_values, (str, bytes)):
			raise ValueError("resource_memory must be a list of five ordered metric values")
		if len(resource_values) != len(self._RESOURCE_MEMORY_NAMES):
			raise ValueError(
				f"resource_memory must contain exactly {len(self._RESOURCE_MEMORY_NAMES)} values"
			)
		resources = []
		for index, value in enumerate(resource_values):
			if value is None:
				resources.append(None)
			elif isinstance(value, (str, int, float)):
				resources.append(str(value).strip())
			else:
				raise ValueError(
					f"resource_memory[{index}] ({self._RESOURCE_MEMORY_NAMES[index]}) must be a string, number, or null"
				)

		army_values = parsed["army_memory"]
		if not isinstance(army_values, Sequence) or isinstance(army_values, (str, bytes)):
			raise ValueError("army_memory must be a list of structured matchup objects")
		army = []
		for index, item in enumerate(army_values):
			if not isinstance(item, Mapping):
				raise ValueError(f"army_memory[{index}] must be a JSON object")
			opponent_units = item.get("opponent_units")
			counters = item.get("counters")
			fallback = item.get("fallback")
			rationale = item.get("rationale", "")
			if (
				not isinstance(opponent_units, list)
				or not opponent_units
				or not all(isinstance(unit, str) and unit.strip() for unit in opponent_units)
			):
				raise ValueError(f"army_memory[{index}].opponent_units must be a non-empty list of unit names")
			if not isinstance(counters, list) or not all(isinstance(counter, str) for counter in counters):
				raise ValueError(f"army_memory[{index}].counters must be a list of strings")
			if not isinstance(fallback, str) or not isinstance(rationale, str):
				raise ValueError(f"army_memory[{index}].fallback and rationale must be strings")
			army.append({
				"opponent_units": [unit.strip() for unit in opponent_units],
				"counters": [counter.strip() for counter in counters if counter.strip()],
				"fallback": fallback.strip(),
				"rationale": rationale.strip(),
			})

		return {
			"resource_memory": resources,
			"army_memory": army[-self.max_items_per_section:],
		}
