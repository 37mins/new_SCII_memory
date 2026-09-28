"""Offline characterization of frozen Planner interfaces, not a game replay."""

import contextlib
import io
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_r6_format_reward import planner_scope, strict_validator
from test_r6_natural_termination import ROOT, method


MANAGER = ROOT / "agent_system/environments/strategic_sc2_env_manager_v2.py"
CONTROLS = dict(SC2_PLANNER_PROTOCOL="legacy", SC2_PLAN_BULLET_MAX="6",
                SC2_PLAN_SEGMENT_MAX="6", SC2_MAX_FORCED_ATTACKS="0")


def probe(answer, observation="We have 1 factory and 1 refinery."):
    scope = planner_scope()
    cls = scope["SC2Planner"]
    planner = cls.__new__(cls)
    planner.ct = 0
    captured = {}

    def reply(query, **kwargs):
        captured["query"] = query
        return answer

    planner.get_action = reply
    strategy = dict(phase_summary="offline fixture", timing="03:00-06:00",
                    instruction="Build the listed units.",
                    attack_instruction="No attacks in this phase.")
    with patch.dict(os.environ, CONTROLS), contextlib.redirect_stdout(io.StringIO()):
        plan = planner.generate_plan(observation,
            dict(marine=8, supplydepot=3, siegetank=2), strategy, 0)
        query = captured["query"]
        counts_method = method(MANAGER, "StrategicSC2EnvironmentManagerV2",
            "_strict_legacy_bo_candidate_counts", Any=object,
            Counter=scope["Counter"], re=scope["re"])
        counts = counts_method.__func__(query)
        parsed = scope["parse_build_order"](answer,
            choice_map={str(i + 1): name for i, name in enumerate(counts)},
            strict_names_only=True)
        valid, reason = strict_validator()(query, answer)
    return dict(query=query, candidates=dict(counts), raw_answer=answer,
                valid=valid, reason=reason, parsed_actions=parsed,
                executable_plan=plan)


class PlannerInterfaceTest(unittest.TestCase):
    def test_generated_one_shot_matches_current_candidates(self):
        item = probe("- marine")
        self.assertEqual(set(item["candidates"]), {"marine", "supplydepot", "siegetank"})
        self.assertIn("## Optional Operations/Units:", item["query"])
        self.assertIn("copied verbatim", item["query"])
        example = item["query"].split("<answer>", 1)[1].split("</answer>", 1)[0]
        with patch.dict(os.environ, CONTROLS):
            self.assertTrue(strict_validator()(item["query"], example)[0])

    def test_alias_and_count_rejection_agrees_with_strict_parser(self):
        for answer in ("- tank", "- depot", "- marine x 2"):
            with self.subTest(answer=answer):
                item = probe(answer)
                self.assertFalse(item["valid"])
                self.assertEqual(item["parsed_actions"], [])
                self.assertTrue(item["executable_plan"])  # Existing rule fallback.
        for name in ("marine", "supplydepot", "siegetank"):
            item = probe("- " + name)
            self.assertTrue(item["valid"])
            self.assertEqual(item["parsed_actions"], [name])

    def test_invalid_suffix_and_overlong_answer_can_keep_valid_prefix(self):
        for answer in ("- marine\n- tank", "- marine\n" * 7):
            with self.subTest(answer=answer):
                item = probe(answer)
                self.assertFalse(item["valid"])
                self.assertTrue(item["parsed_actions"])
                self.assertLessEqual(len(item["parsed_actions"]), 6)
                self.assertTrue(item["executable_plan"])

    def test_log_truncation_does_not_truncate_forwarded_query(self):
        # Long observation is synthetic; it makes the logging limit observable.
        query = probe("- marine", observation="We have 1 factory and 1 refinery. " * 150)["query"]
        build = method(MANAGER, "StrategicSC2EnvironmentManagerV2", "build_text_obs", List=list)
        truncate = method(MANAGER, "StrategicSC2EnvironmentManagerV2", "_truncate_prompt_for_log", Any=object)
        manager = SimpleNamespace(use_natural_tool_call_memory=False,
            use_tool_call_memory=False, memory=None, _prompt_response_log_max_prompt_chars=3000)
        forwarded = build(manager, [query])
        logged = truncate(manager, query)
        self.assertEqual(forwarded, [query])
        self.assertEqual(manager._last_prompts, [query])
        self.assertLessEqual(len(logged), 3000)
        self.assertNotIn("The remaining BO list in this phase:", logged)
        self.assertIn("The remaining BO list in this phase:", forwarded[0])
        manager._prompt_response_log_max_prompt_chars = 0
        self.assertEqual(truncate(manager, query), query)
        self.assertIn("<answer>", truncate(manager, query))
        self.assertIn("Additional constraint:", truncate(manager, query))

    def test_training_records_the_full_query(self):
        launcher = (ROOT / "scripts/train_sc2.sh").read_text()
        self.assertIn("+env.prompt_response_log_max_prompt_chars=0", launcher)


if __name__ == "__main__":
    unittest.main()
