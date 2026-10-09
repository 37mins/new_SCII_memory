import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional


class SC2LongLoopMemory:
	"""Cross-rollout SC2 memory updated from completed rollout records."""

	def __init__(
		self,
		llm_generate: Optional[Callable[[str], Any]] = None,
		initial_resource_memory: Optional[List[Any]] = None,
		initial_army_memory: Optional[List[Any]] = None,
		max_items_per_section: int = 8,
	):
		self.max_items_per_section = max(1, int(max_items_per_section))
		self.resource_memory = deepcopy(
			initial_resource_memory if initial_resource_memory is not None else [None, None, None]
		)
		self.army_memory = deepcopy(initial_army_memory if initial_army_memory is not None else [])
		self.memory = {
			"resource_memory": self.resource_memory,
			"army_memory": self.army_memory,
		}
		self.llm_generate = llm_generate
		self._rollout_records: Dict[str, List[Dict[str, Any]]] = {}
		self._done_rollouts = set()
		self._completed_rollouts = set()
		self._rollout_generation = 0
		self.batch_size = 0

	def reset(self, batch_size: Optional[int] = None) -> None:
		"""Start a new collection batch without clearing learned memory."""
		self.batch_size = batch_size
		self._rollout_generation += 1

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

		updates = {}
		for key, values in record.items():
			if key == "rollout_id":
				continue
			if isinstance(values, (str, bytes, Mapping)) or not hasattr(values, "__len__"):
				if batch_size != 1:
					raise ValueError(f"record field {key!r} must have one value per rollout")
			elif len(values) != batch_size:
				raise ValueError(f"record field {key!r} has a different batch size")

		for index, rollout_id in enumerate(rollout_ids):
			rollout_id = str(rollout_id)
			if rollout_id in self._done_rollouts:
				continue

			transition = {}
			for key, values in record.items():
				if key == "rollout_id":
					continue
				value = values[index]
				transition[key] = deepcopy(value)

			self._rollout_records.setdefault(rollout_id, []).append(transition)
			if bool(transition["done"]):
				self._done_rollouts.add(rollout_id)
				if self.llm_generate is not None:
					updates[rollout_id] = self.update_memory(rollout_id)

		return updates

	def update_memory(
		self,
		rollout_id: str,
		llm_generate: Optional[Callable[[str], Any]] = None,
	) -> Dict[str, Any]:
		"""Use the rollout model callback to revise memory from one completed rollout."""
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
		return deepcopy(self.memory)

	def fetch(
		self,
		history_length: int,
		current_obs: Sequence[Any],
		action_key: str = "action",
	) -> tuple[List[str], List[int]]:
		"""Return the shared long-loop memory in the environment fetch format."""
		resource_names = (
			"Resource gathering rate",
			"Resource spending rate",
			"Supply growth rate",
		)
		resource_lines = [
			f"Recommended {resource_names[index]}: {value}"
			for index, value in enumerate(self.resource_memory[:len(resource_names)])
			if value is not None
		]
		army_lines = [
			item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
			for item in self.army_memory
		]
		lines = ["Long-term memory:", *resource_lines]
		lines.extend(f"Unit matchup response: {item}" for item in army_lines)
		context = "\n".join(lines)
		memory_size = len(resource_lines) + len(army_lines)
		return [context for _ in current_obs], [memory_size for _ in current_obs]

	def _build_update_prompt(
		self,
		rollout_id: str,
		records: List[Dict[str, Any]],
	) -> str:
		last_record = records[-1]
		try:
			won = float(last_record.get("won", 0.0)) > 0
		except (TypeError, ValueError):
			won = False
		outcome_guidance = (
			"The game was won. Use this game's resource gathering rate, resource spending rate, "
			"supply growth rate, and both players' unit compositions at every model decision point "
			"to revise the recommended economic targets and unit matchup rules. Check whether any "
			"previously recommended target was too low, and retain matchup rules supported by "
			"the game evidence and sound reasoning. "
			if won
			else "The game was lost. First determine whether resource gathering rate, resource spending rate, "
			"or supply growth rate could have been improved substantially. If so, mark the corresponding "
			"recommended target as too low. Only if there is no substantial room for economic improvement, "
			"analyze both players' units using StarCraft II mechanics and record effective matchup responses. "
		)
		return (
			"Update the long-term StarCraft II strategy memory. Keep only reusable guidance supported by the rollout evidence. "
			"Write all memory values in clear, concise English.\n"
			f"{outcome_guidance}\n"
			"Return a JSON object containing exactly two keys: resource_memory and army_memory.\n"
			"resource_memory must contain three recommended values in this exact order:\n"
			"1. Resource gathering rate\n"
			"2. Resource spending rate\n"
			"3. Supply growth rate\n"
			"When a recommended target is too low, explicitly include the label 'too low' in that value.\n"
			"army_memory must be a list of matchup responses. Each entry should identify the opponent's units, "
			"our suitable counter units or structures, and what to produce or build if none of those counters are available. "
			"Use this general format: 'When the opponent fields [unit(s)], we should field [counter A], [counter B], "
			"or [counter C]. If none are available, prioritize producing or building [fallback].'\n\n"
			"Current resource memory (values are listed in the order defined above):\n"
			f"- Resource gathering rate: {self.resource_memory[0] if len(self.resource_memory) > 0 else None}\n"
			f"- Resource spending rate: {self.resource_memory[1] if len(self.resource_memory) > 1 else None}\n"
			f"- Supply growth rate: {self.resource_memory[2] if len(self.resource_memory) > 2 else None}\n\n"
			f"Current army matchup memory:\n{json.dumps(self.army_memory, ensure_ascii=False, default=str)}\n\n"
			f"Completed rollout ID: {rollout_id}\n"
			f"Rollout records:\n{json.dumps(records, ensure_ascii=False, default=str)}"
		)

	@staticmethod
	def _parse_model_output(output: Any) -> Mapping[str, Any]:
		if isinstance(output, Mapping):
			return output
		if not isinstance(output, str):
			raise ValueError("llm_generate must return a JSON string or mapping")
		text = output.strip()
		if text.startswith("```"):
			text = text.strip("`")
			if text.lower().startswith("json"):
				text = text[4:].strip()
		parsed = json.loads(text)
		if not isinstance(parsed, Mapping):
			raise ValueError("model output must be a JSON object")
		return parsed
