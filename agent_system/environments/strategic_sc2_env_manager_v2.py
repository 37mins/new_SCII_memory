import json
import os
import re
import time
from collections import Counter
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import OmegaConf

from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory.sc2_compact_asset_memory import SC2CompactAssetMemory
from agent_system.memory.sc2_tool_asset_memory import (
    SC2ToolAssetMemory,
    compact_tool_result_for_prompt,
    extract_legal_sc2_actions,
    extract_tool_sc2_facts,
    is_sc2_memory_tool_call,
    parse_sc2_memory_tool_call,
)
from agent_system.memory.strategic_memory_v2 import StrategicMemoryV2, infer_failure_reason


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _normalize_memory_mode(value: Any) -> str:
    mode = str(value or "natural_tool_call_asset").lower().strip()
    aliases = {
        "natural": "natural_tool_call_asset",
        "natural_tool": "natural_tool_call_asset",
        "natural_tool_call": "natural_tool_call_asset",
        "agiagent": "natural_tool_call_asset",
        "agiagent_tool": "natural_tool_call_asset",
        "agiagent_tool_asset": "natural_tool_call_asset",
        "tool": "tool_call_asset",
        "tool_call": "tool_call_asset",
        "tool_asset": "tool_call_asset",
        "tool_call_assets": "tool_call_asset",
        "biagent_tool": "tool_call_asset",
        "biagent_tool_asset": "tool_call_asset",
        "compact": "compact_asset",
        "asset": "compact_asset",
        "biagent": "compact_asset",
        "biagent_asset": "compact_asset",
        "biagent_assets": "compact_asset",
        "strategic": "strategic_v2",
        "v2": "strategic_v2",
        "none": "off",
        "false": "off",
        "0": "off",
    }
    return aliases.get(mode, mode)


class StrategicSC2EnvironmentManagerV2(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config, is_train: bool = True):
        self.sc2_memory_mode = _normalize_memory_mode(config.env.get("sc2_memory_mode", "natural_tool_call_asset"))
        self.sc2_tool_memory_phase = str(config.env.get("sc2_tool_memory_phase", "query_frozen") or "query_frozen").lower().strip()
        if self.sc2_memory_mode in {"tool_call_asset", "natural_tool_call_asset"}:
            asset_log_path = str(config.env.get("sc2_asset_memory_log_path", "") or "")
            if not is_train:
                asset_log_path = ""
            freeze_retrieval = _as_bool(config.env.get("sc2_tool_freeze_retrieval_after_init", self.sc2_tool_memory_phase != "online"))
            self.memory = SC2ToolAssetMemory(
                max_prompt_items=int(config.env.get("sc2_asset_memory_max_prompt_items", 3) or 3),
                log_path=asset_log_path,
                online_update=_as_bool(config.env.get("sc2_tool_asset_online_update", False)),
                compact_assets_path=str(config.env.get("sc2_tool_compact_assets_path", "") or ""),
                write_compact_assets_path=str(config.env.get("sc2_tool_write_compact_assets_path", "") or ""),
                evidence_audit_path=str(config.env.get("sc2_tool_evidence_audit_path", "") or ""),
                asset_links_path=str(config.env.get("sc2_tool_asset_links_path", "") or ""),
                freeze_retrieval_after_init=freeze_retrieval,
                dynamic_quality_threshold=float(config.env.get("sc2_tool_dynamic_quality_threshold", 0.8) or 0.8),
                reflection_enabled=_as_bool(config.env.get("sc2_tool_reflection_enabled", False)),
                reflection_model=str(config.env.get("sc2_tool_reflection_model", "") or ""),
                reflection_api_key=str(config.env.get("sc2_tool_reflection_api_key", "") or ""),
                reflection_api_base=str(config.env.get("sc2_tool_reflection_api_base", "") or ""),
                reflection_log_path=str(config.env.get("sc2_tool_reflection_log_path", "") or ""),
                reflection_scene_log_path=str(config.env.get("sc2_tool_reflection_scene_log_path", "") or ""),
                reflection_min_quality=float(config.env.get("sc2_tool_reflection_min_quality", 0.65) or 0.65),
                reflection_fallback_to_rules=_as_bool(config.env.get("sc2_tool_reflection_fallback_to_rules", True)),
                curiosity_write_threshold=float(config.env.get("sc2_tool_curiosity_write_threshold", 0.45) or 0.45),
                curiosity_reflect_threshold=float(config.env.get("sc2_tool_curiosity_reflect_threshold", 0.65) or 0.65),
                curiosity_event_window=int(config.env.get("sc2_tool_curiosity_event_window", 12) or 12),
                curiosity_max_reflections_per_episode=int(config.env.get("sc2_tool_curiosity_max_reflections_per_episode", 8) or 8),
                training_model_reflection=_as_bool(config.env.get("sc2_tool_training_model_reflection", False)),
                training_reflection_max_items_per_step=int(config.env.get("sc2_tool_training_reflection_max_items_per_step", 8) or 8),
                attack_ready_min_army=int(
                    config.env.get(
                        "sc2_tool_attack_ready_min_army",
                        config.env.get("sc2_tool_attack_query_min_army", 8),
                    )
                    or 8
                ),
            )
        elif self.sc2_memory_mode == "compact_asset":
            asset_log_path = str(config.env.get("sc2_asset_memory_log_path", "") or "")
            if not is_train:
                asset_log_path = ""
            self.memory = SC2CompactAssetMemory(
                max_prompt_items=int(config.env.get("sc2_asset_memory_max_prompt_items", 3) or 3),
                log_path=asset_log_path,
            )
        elif self.sc2_memory_mode == "strategic_v2":
            self.memory = StrategicMemoryV2()
        elif self.sc2_memory_mode == "off":
            self.memory = None
        else:
            raise ValueError(
                "env.sc2_memory_mode should be one of natural_tool_call_asset, tool_call_asset, compact_asset, strategic_v2, off; "
                f"got {self.sc2_memory_mode}"
            )
        self.use_tool_call_memory = self.sc2_memory_mode in {"tool_call_asset", "natural_tool_call_asset"}
        self.use_natural_tool_call_memory = self.sc2_memory_mode == "natural_tool_call_asset"
        self.sc2_tool_query_train = _as_bool(config.env.get("sc2_tool_query_train", False))
        self.sc2_tool_query_policy = str(config.env.get("sc2_tool_query_policy", "on_demand") or "on_demand").lower().strip()
        self.sc2_tool_startup_query_steps = int(config.env.get("sc2_tool_startup_query_steps", 1) or 1)
        self.sc2_tool_query_cooldown = int(config.env.get("sc2_tool_query_cooldown", 4) or 4)
        self.sc2_tool_query_stall_steps = int(config.env.get("sc2_tool_query_stall_steps", 5) or 5)
        self.sc2_tool_attack_query_min_army = int(config.env.get("sc2_tool_attack_query_min_army", 8) or 8)
        self.sc2_tool_max_queries_per_episode = int(config.env.get("sc2_tool_max_queries_per_episode", 3) or 3)
        self.sc2_tool_verified_candidate_prior = _as_bool(
            config.env.get("sc2_tool_verified_candidate_prior", False)
        )
        self.sc2_tool_verified_candidate_override = _as_bool(
            config.env.get("sc2_tool_verified_candidate_override", False)
        )
        self.sc2_tool_verified_candidate_min_score = float(
            config.env.get("sc2_tool_verified_candidate_min_score", 0.75) or 0.75
        )
        self.sc2_tool_verified_sequence_position = _as_bool(
            config.env.get("sc2_tool_verified_sequence_position", False)
        )
        self.sc2_tool_verified_intervention_only = _as_bool(
            config.env.get("sc2_tool_verified_intervention_only", False)
        )
        self.sc2_tool_training_reflection_interval = int(config.env.get("sc2_tool_training_reflection_interval", 10) or 10)
        self.pre_text_obs = None
        self.is_train = is_train
        self._step_index = 0
        self._episode_reset_index = -1
        self._last_action_valids = []
        self._last_failure_reasons = []
        self._no_progress_counts = []
        self._last_query_step_by_env = []
        self._query_count_by_env = []
        self._attack_query_done_by_env = []
        self._episode_step_by_env = []
        self._terminal_by_env = []
        self._memory_tool_step_required = False
        self._last_query_reasons = []
        self._last_prompts = []
        self._last_memory_contexts = []
        self._last_tool_prompts = []
        self._last_tool_responses = []
        self._last_parsed_tool_calls = []
        self._last_tool_results = []
        self._last_tool_result_prompts = []
        self._pending_tool_call_counts = []
        self._forbid_thinking_output = _as_bool(config.env.get("forbid_thinking_output", False))
        self._strict_legacy_plan_contract = _as_bool(config.env.get("strict_legacy_plan_contract", False))
        self._enable_prompt_response_log = _as_bool(config.env.get("enable_prompt_response_log", False))
        self._prompt_response_log_train_only = _as_bool(config.env.get("prompt_response_log_train_only", True))
        self._prompt_response_log_path = str(config.env.get("prompt_response_log_path", "") or "")
        self._prompt_response_log_max_prompt_chars = int(config.env.get("prompt_response_log_max_prompt_chars", 0) or 0)
        self._tool_call_log_path = str(config.env.get("sc2_tool_call_log_path", "") or "")
        if self._enable_prompt_response_log and (self.is_train or not self._prompt_response_log_train_only):
            if self._prompt_response_log_path:
                os.makedirs(os.path.dirname(self._prompt_response_log_path) or ".", exist_ok=True)
        if self.use_tool_call_memory and self._tool_call_log_path and self.is_train:
            os.makedirs(os.path.dirname(self._tool_call_log_path) or ".", exist_ok=True)
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        text_obs, infos = self.envs.reset()
        self._episode_reset_index += 1
        if self.memory is not None:
            self.memory.reset(batch_size=len(text_obs))
        self._last_action_valids = [True for _ in text_obs]
        self._last_failure_reasons = ["" for _ in text_obs]
        self._no_progress_counts = [0 for _ in text_obs]
        self._last_query_step_by_env = [-999 for _ in text_obs]
        self._query_count_by_env = [0 for _ in text_obs]
        self._attack_query_done_by_env = [False for _ in text_obs]
        self._episode_step_by_env = [0 for _ in text_obs]
        self._terminal_by_env = [False for _ in text_obs]
        self.pre_text_obs = text_obs
        observations = {
            "text": self.build_text_obs(text_obs),
            "image": None,
            "anchor": text_obs.copy(),
        }
        return observations, infos

    def step(self, text_actions: List[str]):
        # A sampled protocol violation is an invalid action, not a batch failure.
        # Preserve the sampled response for audit/PPO; never salvage its suffix.
        raw_text_actions = list(text_actions)
        thinking_invalids = [bool(self._forbid_thinking_output) and
                             ("<think>" in str(response) or "</think>" in str(response))
                             for response in raw_text_actions]
        text_actions = self._apply_verified_candidate_override(text_actions)
        text_actions, final_tool_invalids = self._sanitize_final_actions(text_actions)
        text_actions = ["" if thinking_invalids[i] else response
                        for i, response in enumerate(text_actions)]
        semantic_valids = []
        semantic_evaluables = []
        semantic_reasons = []
        for index, action in enumerate(text_actions):
            prompt = self._last_prompts[index] if index < len(self._last_prompts) else ""
            semantic_evaluables.append(self._semantic_action_evaluable(prompt))
            valid, reason = self._validate_current_bo_response(prompt, action)
            if thinking_invalids[index]:
                valid, reason = False, "thinking output forbidden; entire action rejected"
            semantic_valids.append(valid)
            semantic_reasons.append(reason)
        actions, valids = self.projection_f(text_actions)
        # Reuse the full-response validator; parser repair must not earn its bonus.
        validated_actions = [
            {"text": action, "plan_format_valid": bool(
                semantic_evaluables[i] and semantic_valids[i]
                and valids[i] and not final_tool_invalids[i]
            )}
            for i, action in enumerate(actions)
        ]
        text_obs, rewards, terminateds, truncateds, infos = self.envs.step(validated_actions)
        from agent_system.sc2_episode import apply_external_truncation, interaction_limit_reached
        cutoff_indices = []
        for i, info in enumerate(infos):
            terminateds[i], truncateds[i], infos[i] = apply_external_truncation(
                bool(terminateds[i]),
                bool(truncateds[i]),
                info,
                not self._terminal_by_env[i]
                and interaction_limit_reached(
                    self.config.env.max_steps,
                    self._episode_step_by_env[i] + 1,
                ),
            )
            if self._terminal_by_env[i] or not (terminateds[i] or truncateds[i]):
                continue
            if truncateds[i]:
                cutoff_indices.append(i)
            with open(os.environ['SC2_EPISODE_OUTCOME_LOG'], 'a') as stream:
                stream.write(json.dumps({
                    'reset_index': self._episode_reset_index,
                    'env_index': i,
                    'decision_count': self._episode_step_by_env[i] + 1,
                    'step_index': self._step_index,
                    'policy_decision_count': (infos[i]['last_consumed_policy_action_id'] or 0)
                    if 'last_consumed_policy_action_id' in infos[i] else None,
                    **{key: infos[i][key] for key in (
                        'terminated', 'truncated', 'game_result',
                        'termination_reason', 'outcome_reward',
                    )},
                }) + '\n')
        if cutoff_indices:
            self.envs.truncate(cutoff_indices)
        dones = [bool(terminated or truncated) for terminated, truncated in zip(terminateds, truncateds)]

        action_valids = []
        projection_valids = []
        executor_valids = []
        won_values = []
        attack_counts = []
        failure_reasons = []
        for i, info in enumerate(infos):
            env_valid = bool(info.get("action_is_valid", True))
            is_valid = (
                bool(valids[i])
                and env_valid
                and not final_tool_invalids[i]
                and semantic_valids[i]
            )
            if final_tool_invalids[i]:
                explicit_reason = "tool call emitted during final action stage"
            elif not semantic_valids[i]:
                explicit_reason = semantic_reasons[i]
            else:
                explicit_reason = self._extract_failure_reason(info)
            failure_reason = explicit_reason if final_tool_invalids[i] or thinking_invalids[i] else infer_failure_reason(actions[i], self.pre_text_obs[i], is_valid, explicit_reason)
            action_valids.append(is_valid)
            projection_valids.append(bool(valids[i]))
            executor_valids.append(env_valid)
            won_values.append(info.get("won", 0.0))
            attack_counts.append(info.get("attack_ct", 0.0))
            failure_reasons.append(failure_reason)
            info["is_action_valid"] = to_numpy(is_valid)
            info["semantic_action_valid"] = to_numpy(bool(semantic_valids[i]))
            info["semantic_action_evaluable"] = to_numpy(bool(semantic_evaluables[i]))
            info["projection_action_valid"] = to_numpy(bool(valids[i]))
            info["executor_action_valid"] = to_numpy(env_valid)
            info["failure_reason"] = failure_reason
            if self.use_tool_call_memory and i < len(self._pending_tool_call_counts):
                info["tool_calling"] = to_numpy(self._pending_tool_call_counts[i])

        self._last_action_valids = action_valids
        self._last_failure_reasons = failure_reasons
        if not self._no_progress_counts or len(self._no_progress_counts) != len(rewards):
            self._no_progress_counts = [0 for _ in rewards]
        if not self._terminal_by_env or len(self._terminal_by_env) != len(rewards):
            self._terminal_by_env = [False for _ in rewards]
        for i, reward in enumerate(rewards):
            if bool(dones[i]) or float(reward) > 0:
                self._no_progress_counts[i] = 0
            else:
                self._no_progress_counts[i] += 1
            if bool(dones[i]):
                self._terminal_by_env[i] = True
            elif not self._terminal_by_env[i] and i < len(self._episode_step_by_env):
                self._episode_step_by_env[i] += 1

        memory_record = {
            "text_obs": self.pre_text_obs,
            "next_text_obs": text_obs,
            "action": actions,
            "reward": rewards,
            "done": dones,
            "won": won_values,
            "attack_ct": attack_counts,
            "is_action_valid": action_valids,
            "semantic_action_valid": semantic_valids,
            "semantic_action_evaluable": semantic_evaluables,
            "projection_action_valid": projection_valids,
            "executor_action_valid": executor_valids,
            "failure_reason": failure_reasons,
        }
        if self.memory is not None:
            self.memory.store(memory_record)
        self._write_prompt_response_log(
            text_actions=[raw_text_actions[i] if thinking_invalids[i] else response
                          for i, response in enumerate(text_actions)],
            projected_actions=actions,
            rewards=rewards,
            dones=dones,
            won_values=won_values,
            action_valids=action_valids,
            semantic_valids=semantic_valids,
            semantic_evaluables=semantic_evaluables,
            projection_valids=projection_valids,
            executor_valids=executor_valids,
            failure_reasons=failure_reasons,
            policy_infos=infos,
        )
        self._write_tool_call_log(
            final_responses=text_actions,
            projected_actions=actions,
            rewards=rewards,
            dones=dones,
            won_values=won_values,
            action_valids=action_valids,
            failure_reasons=failure_reasons,
        )
        self._step_index += 1

        if self.is_train and any(truncateds):
            raise RuntimeError(
                'Incomplete SC2 trajectory: refusing a PPO update from truncated games; '
                'see episode_outcomes.jsonl'
            )

        self.pre_text_obs = text_obs
        next_observations = {
            "text": self.build_text_obs(text_obs),
            "image": None,
            "anchor": text_obs.copy(),
        }
        return next_observations, to_numpy(rewards), to_numpy(dones), infos

    def build_text_obs(self, text_obs: List[str]) -> List[str]:
        if self.use_natural_tool_call_memory:
            if self.sc2_tool_memory_phase in {"write_only", "collect_only", "phase1", "learn_only"}:
                memory_contexts = ["" for _ in text_obs]
                prompts = list(text_obs)
                self._mark_memory_router_skipped(len(text_obs))
            else:
                context_ready = [
                    (
                        not self.sc2_tool_verified_candidate_prior
                        or self._verified_prior_context_ready(obs)
                    )
                    for obs in text_obs
                ]
                memory_contexts = [
                    json.dumps(
                        {
                            "mode": "natural_tool_call_asset",
                            "tool_policy": self.sc2_tool_query_policy,
                            "tool_offer_eligible": context_ready[i],
                        },
                        ensure_ascii=False,
                    )
                    for i in range(len(text_obs))
                ]
                prompts = [
                    self._build_natural_tool_prompt(obs)
                    if context_ready[i]
                    else str(obs)
                    for i, obs in enumerate(text_obs)
                ]
                self._last_tool_prompts = [
                    prompts[i] if context_ready[i] else ""
                    for i in range(len(text_obs))
                ]
                self._last_tool_responses = ["" for _ in text_obs]
                self._last_parsed_tool_calls = [
                    {"pending": True}
                    if context_ready[i]
                    else {
                        "skipped": True,
                        "reason": (
                            "natural memory tools require current stage "
                            "and legal BO candidates"
                        ),
                    }
                    for i in range(len(text_obs))
                ]
                self._last_tool_results = [
                    {}
                    if context_ready[i]
                    else {
                        "status": "success",
                        "memory_source": "sc2_tool_assets",
                        "tool_name": "router",
                        "message": (
                            "natural memory tools not offered without "
                            "current stage and legal BO candidates"
                        ),
                        "no_confident_match": True,
                        "guard_rejected": True,
                    }
                    for i in range(len(text_obs))
                ]
                self._last_tool_result_prompts = list(self._last_tool_results)
                self._pending_tool_call_counts = [0 for _ in text_obs]
            self._last_query_reasons = [[] for _ in text_obs]
            self._memory_tool_step_required = False
        elif self.use_tool_call_memory:
            query_reasons = [self._memory_query_reasons(i, obs) for i, obs in enumerate(text_obs)]
            self._last_query_reasons = query_reasons
            self._memory_tool_step_required = any(query_reasons)
            memory_contexts = [json.dumps({"query_reasons": reasons}, ensure_ascii=False) for reasons in query_reasons]
            if self._memory_tool_step_required:
                prompts = [
                    self._build_tool_prompt(obs, query_reasons[i]) if query_reasons[i] else str(obs)
                    for i, obs in enumerate(text_obs)
                ]
                self._last_tool_prompts = [
                    prompts[i] if query_reasons[i] else ""
                    for i in range(len(text_obs))
                ]
            else:
                prompts = list(text_obs)
                self._mark_memory_router_skipped(len(text_obs))
        elif self.memory is None:
            memory_contexts = ["" for _ in text_obs]
            prompts = list(text_obs)
        else:
            memory_contexts, _ = self.memory.fetch(
                self.config.env.history_length,
                current_obs=text_obs,
                action_key="action",
            )
            prompts = [
                f"{memory_contexts[i]}\nCurrent observation:\n{text_obs[i]}"
                for i in range(len(text_obs))
            ]
        self._last_memory_contexts = memory_contexts
        self._last_prompts = prompts
        return prompts

    def needs_memory_tool_query(self) -> bool:
        return bool(self.use_tool_call_memory and not self.use_natural_tool_call_memory and self._memory_tool_step_required)

    def needs_training_model_reflection(self, force: bool = False) -> bool:
        if not self.is_train or not self.use_tool_call_memory:
            return False
        if self.memory is None or not hasattr(self.memory, "has_pending_training_reflections"):
            return False
        if not self.memory.has_pending_training_reflections():
            return False
        if force:
            return True
        interval = max(1, int(self.sc2_tool_training_reflection_interval))
        return self._step_index > 0 and self._step_index % interval == 0

    def build_training_model_reflection_obs(self) -> Dict[str, Any]:
        if self.memory is None or not hasattr(self.memory, "build_training_reflection_prompts"):
            prompts = ["" for _ in range(len(self.pre_text_obs or []))]
        else:
            prompts = self.memory.build_training_reflection_prompts()
        return {
            "text": prompts,
            "image": None,
            "anchor": self.pre_text_obs.copy() if self.pre_text_obs is not None else prompts.copy(),
        }

    def apply_training_model_reflections(self, reflection_outputs: List[str]) -> Dict[str, int]:
        if self.memory is None or not hasattr(self.memory, "apply_training_reflection_outputs"):
            return {"parsed": 0, "stored": 0, "failed": 0}
        stats = self.memory.apply_training_reflection_outputs(reflection_outputs)
        if self._tool_call_log_path and self.is_train:
            with open(self._tool_call_log_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "step_index": self._step_index,
                            "event": "training_model_reflection",
                            "stats": stats,
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )
        return stats

    def run_memory_tools(self, tool_outputs: List[str]) -> Dict[str, Any]:
        if not self.use_tool_call_memory:
            return {
                "text": list(self._last_prompts),
                "image": None,
                "anchor": self.pre_text_obs.copy() if self.pre_text_obs is not None else None,
            }

        final_prompts = []
        parsed_calls = []
        tool_results = []
        prompt_results = []
        tool_counts = []
        self._last_tool_responses = [str(output) for output in tool_outputs]

        for env_idx, tool_output in enumerate(tool_outputs):
            current_obs = self.pre_text_obs[env_idx]
            query_reasons = self._last_query_reasons[env_idx] if env_idx < len(self._last_query_reasons) else []
            query_count = self._query_count_by_env[env_idx] if env_idx < len(self._query_count_by_env) else 0
            is_terminal = self._terminal_by_env[env_idx] if env_idx < len(self._terminal_by_env) else False
            budget_exhausted = (
                self.sc2_tool_max_queries_per_episode >= 0
                and int(query_count) >= self.sc2_tool_max_queries_per_episode
            )
            if is_terminal or not query_reasons or budget_exhausted:
                skip_reason = (
                    "environment already terminal"
                    if is_terminal
                    else "memory query budget exhausted for this episode"
                    if budget_exhausted
                    else "memory router skipped retrieval for this environment"
                )
                result = {
                    "status": "success",
                    "memory_source": "sc2_tool_assets",
                    "tool_name": "router",
                    "message": skip_reason,
                    "no_confident_match": True,
                    "router_reasons": query_reasons,
                    "guard_rejected": True,
                }
                compact_result = compact_tool_result_for_prompt(result)
                parsed_calls.append({"skipped": True, "reason": skip_reason})
                tool_results.append(result)
                prompt_results.append(compact_result)
                final_prompts.append(self._build_final_prompt(current_obs, compact_result))
                tool_counts.append(0)
                self._last_tool_responses[env_idx] = ""
                continue
            parsed, error = parse_sc2_memory_tool_call(tool_output)
            if parsed is None:
                result = self.memory.no_confident(error or "invalid tool call", "query_skill")
                tool_counts.append(0)
            else:
                result = self.memory.run_tool(parsed, env_idx=env_idx, current_observation=current_obs)
                result = self._guard_redundant_skill_result(result, current_obs)
                tool_counts.append(1)
            prior = self._select_verified_candidate_prior(
                result,
                current_obs,
                query_reasons,
            )
            if prior:
                result = dict(result)
                result["verified_candidate_prior"] = prior
            compact_result = compact_tool_result_for_prompt(result)
            if prior:
                compact_result["verified_candidate_prior"] = prior
            parsed_calls.append(parsed or {"error": error})
            tool_results.append(result)
            prompt_results.append(compact_result)
            final_prompts.append(self._build_final_prompt(current_obs, compact_result))

        self._last_parsed_tool_calls = parsed_calls
        self._last_tool_results = tool_results
        self._last_tool_result_prompts = prompt_results
        self._pending_tool_call_counts = tool_counts
        for env_idx, count in enumerate(tool_counts):
            if count > 0 and env_idx < len(self._last_query_step_by_env):
                self._last_query_step_by_env[env_idx] = self._step_index
                if env_idx < len(self._query_count_by_env):
                    self._query_count_by_env[env_idx] += count
                if (
                    env_idx < len(self._attack_query_done_by_env)
                    and env_idx < len(self._last_query_reasons)
                    and "attack_timing_check" in self._last_query_reasons[env_idx]
                ):
                    self._attack_query_done_by_env[env_idx] = True
        self._last_memory_contexts = [json.dumps(result, ensure_ascii=False, default=str) for result in prompt_results]
        self._last_prompts = final_prompts
        return {
            "text": final_prompts,
            "image": None,
            "anchor": self.pre_text_obs.copy(),
        }

    def run_natural_memory_tools(self, first_outputs: List[str]) -> Optional[Dict[str, Any]]:
        """Execute only the memory tool calls the model naturally selected.

        In AGIAgent-style mode the first model turn can either be a final SC2
        action or a memory tool call. A second model turn is created only when
        at least one sample selected a tool.
        """
        if not self.use_natural_tool_call_memory:
            return None

        parsed_calls = []
        tool_results = []
        prompt_results = []
        tool_counts = []
        final_prompts = []
        any_tool_call = False
        self._last_tool_responses = [str(output) for output in first_outputs]

        for env_idx, first_output in enumerate(first_outputs):
            current_obs = self.pre_text_obs[env_idx]
            if not is_sc2_memory_tool_call(first_output):
                parsed_calls.append({"direct_action": True})
                direct_result = {
                    "status": "success",
                    "memory_source": "sc2_tool_assets",
                    "tool_name": "none",
                    "message": "model selected direct SC2 action without memory retrieval",
                    "no_confident_match": True,
                    "direct_action": True,
                }
                tool_results.append(direct_result)
                prompt_results.append(direct_result)
                final_prompts.append(self._build_direct_action_repeat_prompt(current_obs, first_output))
                tool_counts.append(0)
                continue

            any_tool_call = True
            parsed, error = parse_sc2_memory_tool_call(first_output)
            if parsed is None:
                result = self.memory.no_confident(error or "invalid tool call", "query_skill")
                tool_counts.append(0)
                prior = {}
            else:
                allowed, reject_reason = self._natural_tool_call_allowed(env_idx, parsed)
                if allowed:
                    result = self.memory.run_tool(parsed, env_idx=env_idx, current_observation=current_obs)
                    result = self._guard_redundant_skill_result(result, current_obs)
                    tool_counts.append(1)
                    prior = self._select_verified_candidate_prior(
                        result,
                        current_obs,
                        ["model_requested_memory"],
                    )
                    if prior:
                        result = dict(result)
                        result["verified_candidate_prior"] = prior
                else:
                    result = self.memory.no_confident(reject_reason, parsed.get("name", "query_skill"))
                    result["guard_rejected"] = True
                    tool_counts.append(0)
                    prior = {}
            compact_result = compact_tool_result_for_prompt(result)
            if prior:
                compact_result["verified_candidate_prior"] = prior
            parsed_calls.append(parsed or {"error": error})
            tool_results.append(result)
            prompt_results.append(compact_result)
            final_prompts.append(self._build_final_prompt(current_obs, compact_result))

        if not any_tool_call:
            self._last_parsed_tool_calls = parsed_calls
            self._last_tool_results = tool_results
            self._last_tool_result_prompts = prompt_results
            self._pending_tool_call_counts = tool_counts
            self._last_memory_contexts = [json.dumps(result, ensure_ascii=False, default=str) for result in prompt_results]
            return None

        self._last_parsed_tool_calls = parsed_calls
        self._last_tool_results = tool_results
        self._last_tool_result_prompts = prompt_results
        self._pending_tool_call_counts = tool_counts
        for env_idx, count in enumerate(tool_counts):
            if count <= 0:
                continue
            if env_idx < len(self._last_query_step_by_env):
                self._last_query_step_by_env[env_idx] = self._step_index
            if env_idx < len(self._query_count_by_env):
                self._query_count_by_env[env_idx] += count
        self._last_memory_contexts = [json.dumps(result, ensure_ascii=False, default=str) for result in prompt_results]
        self._last_prompts = final_prompts
        return {
            "text": final_prompts,
            "image": None,
            "anchor": self.pre_text_obs.copy(),
        }

    def _memory_query_reasons(self, env_idx: int, text_obs: Any) -> List[str]:
        if self.sc2_tool_memory_phase in {"write_only", "collect_only", "phase1", "learn_only"}:
            return []
        if self.sc2_tool_query_policy in {"never", "off", "0", "false"}:
            return []
        if env_idx < len(self._terminal_by_env) and self._terminal_by_env[env_idx]:
            return []

        last_query_step = self._last_query_step_by_env[env_idx] if env_idx < len(self._last_query_step_by_env) else -999
        query_count = self._query_count_by_env[env_idx] if env_idx < len(self._query_count_by_env) else 0
        episode_step = self._episode_step_by_env[env_idx] if env_idx < len(self._episode_step_by_env) else self._step_index
        if self.sc2_tool_max_queries_per_episode >= 0 and int(query_count) >= self.sc2_tool_max_queries_per_episode:
            return []
        intervention_only = bool(
            getattr(self, "sc2_tool_verified_intervention_only", False)
        )
        if (
            self.sc2_tool_verified_candidate_prior
            and not self._verified_prior_context_ready(text_obs)
        ):
            return []
        if (
            self.sc2_tool_query_policy in {"always", "force", "1", "true"}
            and not intervention_only
        ):
            return ["policy=always"]

        reasons = []
        if (
            not intervention_only
            and int(last_query_step) < 0
            and int(episode_step) < self.sc2_tool_startup_query_steps
        ):
            reasons.append("startup")
            return reasons

        in_cooldown = int(last_query_step) >= 0 and (self._step_index - int(last_query_step)) < self.sc2_tool_query_cooldown
        if in_cooldown:
            return []

        facts = extract_tool_sc2_facts(text_obs)
        supply_left = facts.get("supply_left")
        army = facts.get("army")
        if env_idx < len(self._last_action_valids) and not self._last_action_valids[env_idx]:
            reasons.append("previous_invalid_action")
        failure_reason = self._last_failure_reasons[env_idx] if env_idx < len(self._last_failure_reasons) else ""
        lower_failure = str(failure_reason or "").lower()
        if "insufficient" in lower_failure or "supply" in lower_failure:
            reasons.append("previous_resource_or_supply_failure")
        if supply_left is not None and supply_left <= 0:
            reasons.append("supply_blocked")
        attack_query_done = self._attack_query_done_by_env[env_idx] if env_idx < len(self._attack_query_done_by_env) else False
        if army is not None and army >= self.sc2_tool_attack_query_min_army and not attack_query_done:
            reasons.append("attack_timing_check")
        if env_idx < len(self._no_progress_counts) and self._no_progress_counts[env_idx] >= self.sc2_tool_query_stall_steps:
            reasons.append("no_progress_stall")
        if intervention_only:
            intervention_reasons = {
                "model_requested_memory",
                "previous_invalid_action",
                "previous_resource_or_supply_failure",
                "supply_blocked",
                "no_progress_stall",
            }
            reasons = [
                reason for reason in reasons if reason in intervention_reasons
            ]
        return reasons[:3]

    def _mark_memory_router_skipped(self, batch_size: int) -> None:
        skipped = [
            {
                "status": "success",
                "memory_source": "sc2_tool_assets",
                "tool_name": "router",
                "message": "memory router skipped retrieval",
                "no_confident_match": True,
                "router_reasons": self._last_query_reasons[i] if i < len(self._last_query_reasons) else [],
            }
            for i in range(batch_size)
        ]
        self._last_tool_prompts = ["" for _ in range(batch_size)]
        self._last_tool_responses = ["" for _ in range(batch_size)]
        self._last_parsed_tool_calls = [{"skipped": True} for _ in range(batch_size)]
        self._last_tool_results = skipped
        self._last_tool_result_prompts = skipped
        self._pending_tool_call_counts = [0 for _ in range(batch_size)]

    def _bo_candidate_context(self, text_obs: Any) -> Tuple[dict, Counter]:
        text = str(text_obs or "")
        marker = "The remaining BO list in this phase:"
        if marker not in text:
            return {}, Counter()
        remaining = text.split(marker, 1)[1]
        indexed = {}
        allowed_counts = Counter()
        for line in remaining.splitlines():
            numbered = re.match(r"^\s*(\d+)\.\s*(.+?)\s*$", line)
            legacy = re.match(r"^\s*[-*]\s+(.+?)\s*$", line)
            match = numbered or legacy
            if not match:
                if allowed_counts:
                    break
                continue
            raw_value = numbered.group(2) if numbered else legacy.group(1)
            count_match = re.match(r"(.+?)\s+x\s*(\d+)\s*$", raw_value, re.IGNORECASE)
            item_text = count_match.group(1) if count_match else raw_value
            count = int(count_match.group(2)) if count_match else 1
            actions = extract_legal_sc2_actions(item_text, max_actions=4)
            if not actions:
                actions = [item_text.strip()]
            canonical = str(actions[0]).strip().lower()
            if numbered:
                indexed[numbered.group(1)] = canonical
            allowed_counts[canonical] += max(1, count)
        return indexed, allowed_counts

    @staticmethod
    def _strict_legacy_bo_candidate_counts(text_obs: Any) -> Counter:
        text = str(text_obs or "")
        marker = "The remaining BO list in this phase:"
        if marker not in text:
            return Counter()
        remaining = text.split(marker, 1)[1]
        allowed_counts = Counter()
        for line in remaining.splitlines():
            numbered = re.match(r"^\s*(\d+)\.\s*(.+?)\s*$", line)
            legacy = re.match(r"^\s*[-*]\s+(.+?)\s*$", line)
            match = numbered or legacy
            if not match:
                if allowed_counts:
                    break
                continue
            raw_value = numbered.group(2) if numbered else legacy.group(1)
            count_match = re.match(r"(.+?)\s+x\s*(\d+)\s*$", raw_value, re.IGNORECASE)
            canonical = (count_match.group(1) if count_match else raw_value).strip()
            count = int(count_match.group(2)) if count_match else 1
            allowed_counts[canonical] += max(1, count)
        return allowed_counts

    def _current_bo_candidates(self, text_obs: Any) -> List[str]:
        _, allowed_counts = self._bo_candidate_context(text_obs)
        return list(allowed_counts)[:12]

    def _semantic_action_evaluable(self, prompt: Any) -> bool:
        _, allowed_counts = self._bo_candidate_context(prompt)
        return bool(allowed_counts)

    def _validate_current_bo_response(self, prompt: Any, response: Any) -> Tuple[bool, str]:
        text_prompt = str(prompt or "")
        marker = "The remaining BO list in this phase:"
        if marker not in text_prompt:
            # Terminal/repeated observations can have no BO context. Keep them
            # neutral for reward validity and mark them non-evaluable separately.
            return True, "no current BO candidate context"

        indexed, allowed_counts = self._bo_candidate_context(text_prompt)
        strict_legacy = self._strict_legacy_plan_contract
        original_segment = strict_legacy and os.getenv("SC2_ORIGINAL_BO_SEGMENT", "0") == "1"
        candidate_counts = (
            self._strict_legacy_bo_candidate_counts(text_prompt)
            if strict_legacy
            else allowed_counts
        )
        if not candidate_counts:
            return False, "current BO candidate list is empty"

        answer = str(response or "").split("</think>")[-1].strip()
        answer = re.sub(r"</?answer>", "", answer, flags=re.IGNORECASE).strip()
        selected_counts = Counter()
        bullet_count = 0
        for raw_line in answer.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.lower() == "here are the build orders with attack instructions:":
                continue
            bullet_count += 1
            if strict_legacy:
                if not line.startswith("- "):
                    return False, "canonical legacy response requires `- exactname`"
                canonical = line[2:].strip()
                count = 1
                if original_segment:
                    count_match = re.fullmatch(r"(.+?)\s*x\s*(\d+)", canonical)
                    if count_match:
                        canonical = count_match.group(1).strip()
                        count = int(count_match.group(2))
                        if count == 0:
                            return False, "BO repetition count must be positive"
                if canonical not in candidate_counts:
                    return False, f"action outside current canonical BO candidates: {canonical}"
            else:
                if not line.startswith(("-", "*")):
                    return False, "non-build-order text in final response"
                item = line[1:].strip()
                choice_match = re.match(
                    r"#?\s*(\d+)(?:\s*[.)])?(?:\s+x\s*(\d+))?\s*$",
                    item,
                    re.IGNORECASE,
                )
                if choice_match and indexed:
                    canonical = indexed.get(choice_match.group(1))
                    if canonical is None:
                        return False, "choice index outside current BO candidates"
                    count = int(choice_match.group(2) or 1)
                elif choice_match:
                    return False, "choice index is not valid in legacy-name protocol"
                else:
                    count_match = re.match(r"(.+?)\s+x\s*(\d+)\s*$", item, re.IGNORECASE)
                    item_text = count_match.group(1) if count_match else item
                    count = int(count_match.group(2)) if count_match else 1
                    actions = extract_legal_sc2_actions(item_text, max_actions=4)
                    canonical = str(actions[0] if actions else item_text).strip().lower()
                    if canonical not in allowed_counts:
                        return False, f"action outside current BO candidates: {canonical}"
            selected_counts[canonical] += max(1, count)

        if bullet_count == 0:
            return False, "no build-order bullet lines"
        try:
            bullet_max = max(1, min(int(os.getenv("SC2_PLAN_BULLET_MAX", "15")), 15))
        except ValueError:
            bullet_max = 15
        if not original_segment and bullet_count > bullet_max:
            return False, "selected bullet count exceeds response line limit"
        try:
            segment_max = max(1, min(int(os.getenv("SC2_PLAN_SEGMENT_MAX", "96")), 512))
        except ValueError:
            segment_max = 96
        if not original_segment and sum(selected_counts.values()) > segment_max:
            return False, "selected action count exceeds executable plan segment"
        for canonical, count in selected_counts.items():
            if count > candidate_counts[canonical]:
                return False, f"action count exceeds current BO remainder: {canonical}"
        return True, ""

    def _verified_prior_context_ready(self, text_obs: Any) -> bool:
        text = str(text_obs or "")
        has_stage = re.search(r"Current Stage:\s*[^\n]+", text, re.IGNORECASE)
        return bool(has_stage and self._current_bo_candidates(text))

    def _build_grounded_skill_query(self, text_obs: Any, reasons: List[str]) -> str:
        text = str(text_obs or "")
        facts = extract_tool_sc2_facts(text)
        stage_match = re.search(r"Current Stage:\s*([^\n]+)", text, re.IGNORECASE)
        stage = stage_match.group(1).strip() if stage_match else "unknown"
        stage = re.sub(r"\s*\([^)]*\)\s*$", "", stage)
        compact_fact_names = {
            "time_s": "t",
            "army": "army",
            "minerals": "min",
            "supply_left": "supply",
            "enemy_units": "enemy",
        }
        fact_text = ",".join(
            f"{short_name}{facts[key]}"
            for key, short_name in compact_fact_names.items()
            if key in facts
        )
        candidates = self._current_bo_candidates(text)
        candidate_text = "|".join(
            re.sub(
                r"\s+",
                " ",
                re.sub(r"\bwith\b", "", candidate, flags=re.IGNORECASE),
            ).strip()
            for candidate in candidates
        ) or "unknown"
        reason_text = ",".join(reasons) if reasons else "uncertain"
        return (
            f"trigger={reason_text}; stage={stage}; state={fact_text or 'unknown'}; "
            f"legal_candidates={candidate_text}; need=learned,executable,nonredundant"
        )[:260]

    def _guard_redundant_skill_result(
        self,
        result: Dict[str, Any],
        current_obs: Any,
    ) -> Dict[str, Any]:
        if not isinstance(result, dict) or result.get("tool_name") != "query_skill":
            return result
        if result.get("no_confident_match", False):
            return result

        candidates = {action.lower() for action in self._current_bo_candidates(current_obs)}
        ready_skills = [
            skill
            for skill in result.get("skills", []) or []
            if skill.get("match_status") == "ready" and skill.get("executable_now")
        ]
        if not ready_skills:
            return result

        learned_skills = [
            skill
            for skill in ready_skills
            if str(skill.get("created_by") or "") not in {"", "seed_static"}
        ]
        novel_actions = {
            action
            for skill in ready_skills
            for action in skill.get("next_grounded_actions", []) or []
            if str(action).lower() not in candidates
        }
        if learned_skills or novel_actions:
            return result

        guarded = dict(result)
        guarded["no_confident_match"] = True
        guarded["redundant_with_current_bo"] = True
        guarded["retrieval_guidance"] = (
            "Only redundant seed-static actions were retrieved; ignore memory for this decision."
        )
        return guarded

    def _skill_adoption_audit(
        self,
        tool_result: Dict[str, Any],
        final_response: Any,
    ) -> Dict[str, Any]:
        if not isinstance(tool_result, dict):
            tool_result = {}
        skills = tool_result.get("skills", []) or []
        retrieved_actions = []
        sources = []
        for skill in skills:
            source = str(skill.get("created_by") or "unknown")
            if source not in sources:
                sources.append(source)
            for action in skill.get("next_grounded_actions", []) or []:
                if str(action).lower() not in {item.lower() for item in retrieved_actions}:
                    retrieved_actions.append(str(action))
        final_actions = extract_legal_sc2_actions(final_response, max_actions=12)
        final_markers = {action.lower() for action in final_actions}
        prior = tool_result.get("verified_candidate_prior") or {}
        prior_action = str(prior.get("action") or "")
        model_actions = [
            str(action)
            for action in prior.get("model_final_actions_before_override", []) or []
        ]
        model_markers = {action.lower() for action in model_actions}
        prior_in_model = bool(
            prior_action and prior_action.lower() in model_markers
        )
        prior_in_final = bool(
            prior_action and prior_action.lower() in final_markers
        )
        override_applied = bool(prior.get("execution_override_applied", False))
        override_blocked_reason = str(
            prior.get("execution_override_blocked_reason") or ""
        )
        adopted_actions = [
            action for action in retrieved_actions if action.lower() in final_markers
        ]
        if prior_in_final:
            if prior_action.lower() not in {action.lower() for action in adopted_actions}:
                adopted_actions.append(prior_action)
        causal_adoption = False
        adoption_mode = "none"
        if not skills:
            status = "no_retrieval"
        elif tool_result.get("redundant_with_current_bo"):
            status = "redundant_guarded"
            adopted_actions = []
            adoption_mode = "redundant_guard"
        elif override_applied and prior_in_final and not prior_in_model:
            status = "override_injected"
            adoption_mode = "execution_override"
            causal_adoption = True
        elif override_blocked_reason:
            status = "override_blocked"
            adoption_mode = override_blocked_reason
        elif prior_in_model and prior_in_final:
            status = "model_selected_prior"
            adoption_mode = "model_selected_without_override"
        elif prior_in_final:
            status = "prompt_aligned"
            adoption_mode = "prompt_or_independent_selection"
        elif adopted_actions:
            status = "retrieval_overlap_noncausal"
            adoption_mode = "retrieval_overlap"
        else:
            status = "not_adopted"
        return {
            "retrieved_next_actions": retrieved_actions,
            "retrieved_skill_sources": sources,
            "final_actions": final_actions,
            "model_final_actions_before_override": model_actions,
            "adopted_actions": adopted_actions,
            "adoption_status": status,
            "adoption_mode": adoption_mode,
            "causal_adoption": causal_adoption,
            "intervention_eligible": bool(
                prior.get("intervention_eligible", False)
            ),
            "sequence_alignment": prior.get("sequence_alignment", {}),
            "verified_candidate_prior": prior,
        }

    @staticmethod
    def _candidate_family(action: Any) -> str:
        text = re.sub(r"\s+", " ", str(action or "")).strip().lower()
        if text.startswith("launch offense"):
            return "launch offense"
        return text

    def _select_verified_candidate_prior(
        self,
        tool_result: Dict[str, Any],
        current_obs: Any,
        query_reasons: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not getattr(self, "sc2_tool_verified_candidate_prior", False):
            return {}
        if not isinstance(tool_result, dict) or tool_result.get("no_confident_match"):
            return {}

        query_reasons = list(query_reasons or [])
        intervention_reasons = {
            "model_requested_memory",
            "previous_invalid_action",
            "previous_resource_or_supply_failure",
            "supply_blocked",
            "no_progress_stall",
        }
        intervention_eligible = any(
            reason in intervention_reasons for reason in query_reasons
        )
        if (
            getattr(self, "sc2_tool_verified_intervention_only", False)
            and not intervention_eligible
        ):
            return {}

        candidates = self._current_bo_candidates(current_obs)
        min_score = float(
            getattr(self, "sc2_tool_verified_candidate_min_score", 0.75)
        )
        ranked = []
        for skill_index, skill in enumerate(tool_result.get("skills", []) or []):
            if str(skill.get("created_by") or "") != "verified_win_transition":
                continue
            if skill.get("match_status") != "ready" or not skill.get("executable_now"):
                continue
            live_score = float(skill.get("live_match_score") or 0.0)
            if live_score < min_score:
                continue
            quality = float(skill.get("quality") or 0.0)
            sequence_actions = (
                skill.get("sequence_actions", []) or []
                if getattr(self, "sc2_tool_verified_sequence_position", False)
                else skill.get("next_grounded_actions", []) or []
            )
            if not sequence_actions:
                sequence_actions = skill.get("next_grounded_actions", []) or []
            for action_index, retrieved_action in enumerate(
                sequence_actions
            ):
                retrieved_family = self._candidate_family(retrieved_action)
                for candidate_index, candidate in enumerate(candidates):
                    if self._candidate_family(candidate) != retrieved_family:
                        continue
                    evidence_score = live_score * max(quality, 0.01)
                    ranked.append(
                        {
                            "rank_key": (
                                -candidate_index,
                                evidence_score,
                                -action_index,
                                -skill_index,
                            ),
                            "priority": evidence_score / (candidate_index + 1),
                            "candidate": candidate,
                            "candidate_index": candidate_index,
                            "retrieved_action": str(retrieved_action),
                            "chain_index": action_index,
                            "chain_length": len(sequence_actions),
                            "asset_id": str(skill.get("asset_id") or ""),
                            "live_match_score": live_score,
                            "quality": quality,
                        }
                    )
        if not ranked:
            return {}
        best = max(ranked, key=lambda item: item["rank_key"])
        return {
            "action": best["candidate"],
            "retrieved_action": best["retrieved_action"],
            "asset_id": best["asset_id"],
            "source": "verified_win_transition",
            "live_match_score": round(best["live_match_score"], 6),
            "quality": round(best["quality"], 6),
            "priority": round(best["priority"], 6),
            "query_reasons": query_reasons,
            "intervention_eligible": intervention_eligible,
            "candidate_order": candidates,
            "sequence_alignment": {
                "candidate_index": best["candidate_index"],
                "chain_index": best["chain_index"],
                "chain_length": best["chain_length"],
            },
            "mode": "sequence_position_intervention_canary",
        }

    def _apply_verified_candidate_override(
        self,
        text_actions: List[str],
    ) -> List[str]:
        if not getattr(self, "sc2_tool_verified_candidate_override", False):
            return text_actions
        overridden = []
        for env_idx, response in enumerate(text_actions):
            tool_result = (
                self._last_tool_results[env_idx]
                if env_idx < len(self._last_tool_results)
                else {}
            )
            prior = (
                tool_result.get("verified_candidate_prior") or {}
                if isinstance(tool_result, dict)
                else {}
            )
            prior_action = str(prior.get("action") or "")
            final_actions = extract_legal_sc2_actions(response, max_actions=16)
            if prior_action:
                prior["model_final_actions_before_override"] = final_actions
            if not prior_action or prior_action.lower() in {
                action.lower() for action in final_actions
            }:
                if prior_action:
                    prior["execution_override_applied"] = False
                    prior.pop("execution_override_blocked_reason", None)
                overridden.append(response)
                continue
            if (
                getattr(self, "sc2_tool_verified_intervention_only", False)
                and not prior.get("intervention_eligible", False)
            ):
                prior["execution_override_applied"] = False
                prior["execution_override_blocked_reason"] = (
                    "trigger_not_intervention_eligible"
                )
                overridden.append(response)
                continue
            final_families = {
                self._candidate_family(action) for action in final_actions
            }
            if (
                "launch offense" in final_families
                and self._candidate_family(prior_action) != "launch offense"
            ):
                prior["execution_override_applied"] = False
                prior["execution_override_blocked_reason"] = (
                    "preserve_model_offense"
                )
                overridden.append(response)
                continue
            prior["execution_override_applied"] = True
            prior.pop("execution_override_blocked_reason", None)
            overridden.append(f"- {prior_action}\n{response}")
        return overridden

    def _build_tool_prompt(self, text_obs: Any, reasons: List[str]) -> str:
        reason_text = ", ".join(reasons) if reasons else "router selected retrieval"
        query = self._build_grounded_skill_query(text_obs, reasons)
        tool_call = {
            "name": "query_skill",
            "arguments": {"query": query, "env_name": "SC2"},
        }
        return (
            f"{text_obs}\n\n"
            "Memory tool step:\n"
            f"Retrieval trigger: {reason_text}.\n"
            "Call exactly one memory retrieval tool before producing the final SC2 build order.\n"
            "Available tools: query_skill, query_graph_skill, query_world_model.\n"
            "Prefer query_skill for build-order/action decisions, query_world_model for rules, "
            "and query_graph_skill only when linked fact-skill context is needed.\n"
            "Output only one tool call in this exact format:\n"
            f"<tool_call>{json.dumps(tool_call, ensure_ascii=False, separators=(',', ':'))}</tool_call>\n"
            "Do not output the final build order in this tool step."
        )

    def _build_natural_tool_prompt(self, text_obs: Any) -> str:
        grounded_query = self._build_grounded_skill_query(
            text_obs,
            ["model_requested_memory"],
        )
        tool_call = {
            "name": "query_skill",
            "arguments": {
                "query": grounded_query,
                "env_name": "SC2",
            },
        }
        return (
            f"{text_obs}\n\n"
            "Optional memory tools:\n"
            "- query_skill: retrieve compact action skills for build-order or attack decisions.\n"
            "- query_world_model: retrieve compact SC2 rules, constraints, or failure boundaries.\n"
            "- query_graph_skill: retrieve linked fact-skill context when one tool result needs both rules and actions.\n"
            "Use a memory tool only when the current observation has a real decision uncertainty, repeated failure, "
            "missing precondition, or attack-timing question. Do not call tools every step.\n"
            "If you need memory, output exactly one tool call and no final action:\n"
            f"<tool_call>{json.dumps(tool_call, ensure_ascii=False, separators=(',', ':'))}</tool_call>\n"
            "The query must remain specific to the shown stage, state, and legal candidates. "
            "Literal placeholder queries are rejected.\n"
            "If memory is not needed, output only the next SC2 build order/action lines. "
            "Do not include explanations."
        )

    def _build_final_prompt(self, text_obs: Any, tool_result: Dict[str, Any]) -> str:
        result_text = json.dumps(tool_result, ensure_ascii=False, default=str)
        prior = tool_result.get("verified_candidate_prior") or {}
        prior_action = str(prior.get("action") or "")
        prior_instruction = ""
        if prior_action:
            prior_instruction = (
                "\nCausal adoption canary: a high-confidence verified winning transition "
                f"maps to the legal candidate `{prior_action}`. Put this exact candidate "
                "first in the final output unless the current observation makes it invalid. "
                "Do not replace it with a different candidate.\n"
            )
        return (
            "Memory tool result:\n"
            f"{result_text}\n\n"
            "Current observation:\n"
            f"{text_obs}\n\n"
            "Final action step:\n"
            "Use the tool result only if it matches the current observation. If no_confident_match=true, ignore memory.\n"
            f"{prior_instruction}"
            "Output only the next SC2 build order/action lines. Do not output JSON, explanations, or another tool call."
        )

    def _build_direct_action_repeat_prompt(self, text_obs: Any, first_action: Any) -> str:
        return (
            "You already selected a final SC2 action without using memory.\n"
            "Current observation:\n"
            f"{text_obs}\n\n"
            "Previously selected final action:\n"
            f"{first_action}\n\n"
            "Repeat the final SC2 build order/action lines only. Do not call a tool and do not explain."
        )

    def _natural_tool_call_allowed(self, env_idx: int, parsed: Dict[str, Any]) -> Tuple[bool, str]:
        if self.sc2_tool_memory_phase in {"write_only", "collect_only", "phase1", "learn_only"}:
            return False, "memory retrieval disabled during write-only collection phase"
        if self.sc2_tool_query_policy in {"never", "off", "0", "false"}:
            return False, "memory retrieval disabled by sc2_tool_query_policy"
        if env_idx < len(self._terminal_by_env) and self._terminal_by_env[env_idx]:
            return False, "environment already terminal"
        if (
            self.sc2_tool_verified_candidate_prior
            and not self._verified_prior_context_ready(self.pre_text_obs[env_idx])
        ):
            return False, "verified prior requires current stage and legal BO candidates"
        query_count = self._query_count_by_env[env_idx] if env_idx < len(self._query_count_by_env) else 0
        if self.sc2_tool_max_queries_per_episode >= 0 and int(query_count) >= self.sc2_tool_max_queries_per_episode:
            return False, "memory query budget exhausted for this episode"
        last_query_step = self._last_query_step_by_env[env_idx] if env_idx < len(self._last_query_step_by_env) else -999
        if int(last_query_step) >= 0 and (self._step_index - int(last_query_step)) < self.sc2_tool_query_cooldown:
            return False, "memory query cooldown active; act from current observation"
        query = str(parsed.get("arguments", {}).get("query") or "").strip()
        if not query:
            return False, "empty memory query"
        normalized_query = re.sub(r"\s+", " ", query).strip().lower()
        placeholder_queries = {
            "short observation-grounded question",
            "current sc2 state and immediate local goal",
            "query",
            "question",
            "...",
        }
        if normalized_query in placeholder_queries or "placeholder" in normalized_query:
            return False, "placeholder memory query rejected"
        return True, ""

    def _sanitize_final_actions(self, text_actions: List[str]) -> Tuple[List[str], List[bool]]:
        if not self.use_tool_call_memory:
            return text_actions, [False for _ in text_actions]
        sanitized = []
        invalids = []
        for action in text_actions:
            invalid = is_sc2_memory_tool_call(action)
            invalids.append(invalid)
            sanitized.append("" if invalid else action)
        return sanitized, invalids

    def _extract_failure_reason(self, info: Dict[str, Any]) -> str:
        for key in ("failure_reason", "action_fail_reason", "error", "message"):
            value = info.get(key)
            if value:
                return str(value)
        return ""

    def _should_log_prompt_response(self) -> bool:
        if not self._enable_prompt_response_log:
            return False
        if not self._prompt_response_log_path:
            return False
        if self._prompt_response_log_train_only and not self.is_train:
            return False
        return True

    def _truncate_prompt_for_log(self, prompt: Any) -> str:
        text = str(prompt)
        limit = self._prompt_response_log_max_prompt_chars
        if limit <= 0 or len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."

    def _write_prompt_response_log(
        self,
        text_actions: List[str],
        projected_actions: List[Any],
        rewards,
        dones,
        won_values: List[Any],
        action_valids: List[bool],
        semantic_valids: List[bool],
        semantic_evaluables: List[bool],
        projection_valids: List[bool],
        executor_valids: List[bool],
        failure_reasons: List[str],
        policy_infos: List[Dict],
    ) -> None:
        if not self._should_log_prompt_response():
            return
        with open(self._prompt_response_log_path, "a", encoding="utf-8") as f:
            for env_idx, llm_response in enumerate(text_actions):
                row = {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "step_index": self._step_index,
                    "env_idx": env_idx,
                    "prompt": self._truncate_prompt_for_log(self._last_prompts[env_idx] if env_idx < len(self._last_prompts) else ""),
                    "memory_context": self._last_memory_contexts[env_idx] if env_idx < len(self._last_memory_contexts) else "",
                    "llm_response": str(llm_response),
                    "projected_action": projected_actions[env_idx] if env_idx < len(projected_actions) else None,
                    "reward": float(rewards[env_idx]),
                    "done": bool(dones[env_idx]),
                    "won": float(won_values[env_idx]),
                    "is_action_valid": bool(action_valids[env_idx]),
                    "semantic_action_valid": bool(semantic_valids[env_idx]),
                    "semantic_action_evaluable": bool(semantic_evaluables[env_idx]),
                    "projection_action_valid": bool(projection_valids[env_idx]),
                    "executor_action_valid": bool(executor_valids[env_idx]),
                    "failure_reason": failure_reasons[env_idx],
                    **{key: policy_infos[env_idx].get(key) for key in (
                        "policy_action_applied", "policy_action_id", "last_consumed_policy_action_id",
                    )},
                }
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def _should_log_tool_call(self) -> bool:
        # Evaluation needs the same audit trail as training so a zero call
        # count can be distinguished from a router or parser failure.
        return bool(self.use_tool_call_memory and self._tool_call_log_path)

    def _write_tool_call_log(
        self,
        final_responses: List[str],
        projected_actions: List[Any],
        rewards,
        dones,
        won_values: List[Any],
        action_valids: List[bool],
        failure_reasons: List[str],
    ) -> None:
        if not self._should_log_tool_call():
            return
        with open(self._tool_call_log_path, "a", encoding="utf-8") as f:
            for env_idx, final_response in enumerate(final_responses):
                tool_result = self._last_tool_results[env_idx] if env_idx < len(self._last_tool_results) else {}
                row = {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "step_index": self._step_index,
                    "env_idx": env_idx,
                    "tool_prompt": self._last_tool_prompts[env_idx] if env_idx < len(self._last_tool_prompts) else "",
                    "tool_response": self._last_tool_responses[env_idx] if env_idx < len(self._last_tool_responses) else "",
                    "parsed_tool_call": self._last_parsed_tool_calls[env_idx] if env_idx < len(self._last_parsed_tool_calls) else {},
                    "tool_result": tool_result,
                    "final_prompt": self._last_prompts[env_idx] if env_idx < len(self._last_prompts) else "",
                    "final_response": str(final_response),
                    "projected_action": projected_actions[env_idx] if env_idx < len(projected_actions) else None,
                    "reward": float(rewards[env_idx]),
                    "done": bool(dones[env_idx]),
                    "won": float(won_values[env_idx]),
                    "is_action_valid": bool(action_valids[env_idx]),
                    "failure_reason": failure_reasons[env_idx],
                    "no_confident_match": bool(tool_result.get("no_confident_match", True)) if isinstance(tool_result, dict) else True,
                    "retrieved_skill_count": int(tool_result.get("skills_count", tool_result.get("graph_skills_count", 0)) or 0) if isinstance(tool_result, dict) else 0,
                }
                row.update(self._skill_adoption_audit(tool_result, final_response))
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item["active_masks"]:
                info = total_infos[batch_idx][i]
                won_value = float(info.get("won", 0.0))
                success["success_rate"].append(won_value)
                success["sc2_attack_ct"].append(float(info.get("attack_ct", 0.0)))
                success["sc2_action_valid"].append(float(info.get("is_action_valid", 0.0)))
                success["sc2_semantic_action_valid"].append(float(info.get("semantic_action_valid", 0.0)))
                success["sc2_semantic_action_evaluable"].append(float(info.get("semantic_action_evaluable", 0.0)))
                success["sc2_projection_action_valid"].append(float(info.get("projection_action_valid", 0.0)))
                success["sc2_executor_action_valid"].append(float(info.get("executor_action_valid", 0.0)))
                return


def make_envs(config):
    if "sc2" not in config.env.env_name.lower():
        raise ValueError("strategic_sc2_env_manager_v2 only supports SC2 envs")
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")

    from agent_system.environments.env_package.sc2_online_restored import build_sc2_envs, sc2_projection

    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    val_group_n = int(config.env.get("val_rollout_n", 1))
    if val_group_n <= 0:
        val_group_n = 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    envs_raw = build_sc2_envs(
        config=config.env,
        seed=config.env.seed,
        env_num=config.data.train_batch_size,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=True,
    )
    val_envs_raw = build_sc2_envs(
        config=config.env,
        seed=config.env.seed + 1000,
        env_num=config.data.val_batch_size,
        group_n=val_group_n,
        resources_per_worker=resources_per_worker,
        is_train=False,
    )

    projection_f = partial(sc2_projection)
    envs = StrategicSC2EnvironmentManagerV2(envs_raw, projection_f, config, is_train=True)
    val_envs = StrategicSC2EnvironmentManagerV2(val_envs_raw, projection_f, config, is_train=False)
    return envs, val_envs
