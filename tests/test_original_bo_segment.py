"""Whole-answer validator -> real Planner -> Commander -> Executor queue, CPU only."""
import ast
from collections import Counter, defaultdict
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_r6_format_reward import planner_scope, strict_validator
from test_r6_natural_termination import ARENA, ROOT, method
from test_valid_bo_passthrough import CONTROLS, OBS, STRATEGY

BASELINE = "0d7ba09ddbdf94d85d67b553bf7978f402cf2052"


@contextlib.contextmanager
def frozen_source(sha):
    if sha is None:
        yield
        return
    paths = [ARENA / "agents/planner.py", ARENA / "agents/commander.py",
             ROOT / "agent_system/environments/strategic_sc2_env_manager_v2.py"]
    texts = {p: subprocess.check_output(["git", "show", f"{sha}:{p.relative_to(ROOT)}"],
                                        cwd=ROOT, text=True) for p in paths}
    read = Path.read_text
    with patch.object(Path, "read_text", lambda p, *a, **k: texts[p] if p in texts else read(p, *a, **k)):
        yield


def pipeline(answer, enabled, obs=OBS, todo=None, strategy=None, phase=2, sha=None, attack_position="after_build"):
    todo = dict(marine=42) if todo is None else copy.deepcopy(todo)
    strategy = STRATEGY if strategy is None else strategy
    with frozen_source(sha):
        scope = planner_scope()
        validate = strict_validator()
        path = ARENA / "agents/commander.py"
        methods = {name: method(path, "SC2Commander", name, os=os,
                    normalize_bo_name=lambda s: str(s).strip().lower(),
                    read_int_env=lambda k, d: int(os.getenv(k, d)))
                   for name in ("renew_bo", "ensure_opening_marine", "renew_waiting_bo", "get_intent")}
    planner = scope["SC2Planner"].__new__(scope["SC2Planner"])
    planner.ct = 0
    planner.correct_format = planner.exceed = False
    planner.last_consumed_policy_action_id = None
    sent, trace = [], {}
    planner._send = sent.append

    def receive():
        trace["valid"], trace["reason"] = validate(sent[-1]["obs"], answer)
        return dict(msg="ACTION", value=answer, policy_action_id=1, plan_format_valid=trace["valid"])

    planner._recv = receive
    parse = scope["parse_build_order"]

    def tracked_parse(text, **kwargs):
        trace["after_line_truncation"] = text
        trace["parsed"] = parse(text, **kwargs)
        return trace["parsed"]

    scope["parse_build_order"] = tracked_parse
    commander = SimpleNamespace(planner=planner, waiting_bo=todo, current_bo=[],
        strategy=strategy, strategy_renew_ct=phase, has_attack=False, attack_ct=0,
        bo_done=defaultdict(int), should_renew_strategy_by_time=lambda obs: False,
        ct_waiting_bo=lambda: len(todo),
        bo2action=json.loads((ARENA / "knowledge/data_BO2Action_1.json").read_text()))
    for name, fn in methods.items():
        setattr(commander, name, fn.__get__(commander))
    guard = commander.ensure_opening_marine

    def tracked_guard(obs):
        trace["planner_queue"] = list(commander.current_bo)
        guard(obs)

    commander.ensure_opening_marine = tracked_guard
    env = dict(CONTROLS, SC2_ORIGINAL_BO_SEGMENT=str(int(enabled)),
               SC2_PRESERVE_VALID_BO="1", SC2_ENABLE_AIR_OPTIONS="1",
               SC2_ATTACK_ORDER_POSITION=attack_position)
    console = io.StringIO()
    with patch.dict(os.environ, env), contextlib.redirect_stdout(console):
        queue = commander.renew_bo(obs)
    trace.update(raw_answer=answer, query=sent[0]["obs"], executor_queue=queue,
                 line_limit_exceeded=planner.exceed, remaining_todo=commander.waiting_bo,
                 rule_added=list((Counter(x["action"] for x in queue) - Counter(trace["planner_queue"])).elements()),
                 console=console.getvalue(), scope="CPU queue boundary; no SC2 engine command submitted")
    return trace


class OriginalBoSegmentTest(unittest.TestCase):
    def test_disabled_matches_frozen_baseline_including_query_and_fallback(self):
        for answer in ("- marine", "- marine\n" * 7, "- marine x 29", "- tank", "- marine\n- tank"):
            with self.subTest(answer=answer):
                self.assertEqual(pipeline(answer, False), pipeline(answer, False, sha=BASELINE))

    def test_seven_fifteen_and_over_fifteen_lines_preserved(self):
        for n in (7, 15, 29):
            with self.subTest(n=n):
                answer = "\n".join(["- marine"] * n)
                r = pipeline(answer, True)
                self.assertTrue(r["valid"])
                self.assertFalse(r["line_limit_exceeded"])
                self.assertEqual(r["after_line_truncation"], answer)
                self.assertEqual([x["action"] for x in r["executor_queue"]], ["marine"] * n)

    def test_counts_expand_in_order_with_no_hidden_15_96_or_512_limit(self):
        for n in (18, 29, 97, 513):
            with self.subTest(n=n):
                r = pipeline(f"- supplydepot x 2\n- marine x {n}\n- barracks", True,
                             todo=dict(marine=n, supplydepot=2, barracks=1))
                expected = ["supplydepot"] * 2 + ["marine"] * n + ["barracks"]
                self.assertTrue(r["valid"])
                self.assertEqual(r["parsed"], expected)
                self.assertEqual([x["action"] for x in r["executor_queue"]], expected)

    def test_archive_count_spacing_and_mixed_repeated_bullets(self):
        r = pipeline("- marine x2\n- marine\n- marinex3", True)
        self.assertTrue(r["valid"])
        self.assertEqual(r["parsed"], ["marine"] * 6)

    def test_invalid_answers_keep_old_queue_even_with_valid_count_prefix(self):
        for answer, todo in (("- marine x 99", dict(marine=42)),
                             ("- marine\n" * 29, dict(marine=8)),
                             ("- marine x 10\n- barrack", dict(marine=42, barracks=2)),
                             ("- Marine x 2", dict(marine=42)),
                             ("- siege tank", dict(siegetank=9)),
                             ("- #1 x 8", dict(marine=42)),
                             ("- factory", dict(marine=42))):
            with self.subTest(answer=answer):
                a, b = pipeline(answer, False, todo=todo), pipeline(answer, True, todo=todo)
                self.assertFalse(b["valid"])
                for key in ("after_line_truncation", "parsed", "planner_queue", "executor_queue", "remaining_todo"):
                    self.assertEqual(a[key], b[key], key)

    def test_zero_and_noninteger_counts_rejected(self):
        for count in ("0", "-1", "1.5"):
            self.assertFalse(pipeline(f"- marine x {count}", True)["valid"])

    def test_projection_and_memory_off_do_not_rewrite_count_answers(self):
        scope = {}
        path = ROOT / "agent_system/environments/env_package/sc2_online_restored/projection.py"
        exec(compile(ast.parse(path.read_text()), str(path), "exec"), scope)
        sanitize = method(ROOT / "agent_system/environments/strategic_sc2_env_manager_v2.py",
                          "StrategicSC2EnvironmentManagerV2", "_sanitize_final_actions", List=list, Tuple=tuple)
        answers = ["- marine x 29", "- marine\n" * 15]
        clean, invalids = sanitize(SimpleNamespace(use_tool_call_memory=False), answers)
        self.assertEqual((clean, invalids), (answers, [False, False]))
        self.assertEqual(scope["sc2_projection"](clean), (answers, [True, True]))

    def test_prerequisite_candidates_unchanged(self):
        todo = dict(siegetank=9, starport=1, vikingfighter=8)
        obs = "It is now 14:32 in the game. We have 1 barracks and 1 refinery."
        a = pipeline("- factory", False, obs=obs, todo=todo)
        b = pipeline("- factory x 1", True, obs=obs, todo=todo)
        marker = "The remaining BO list in this phase:"
        self.assertEqual(a["query"].split(marker)[1].split("Use only exact")[0],
                         b["query"].split(marker)[1].split("Use only exact")[0])
        self.assertTrue(a["valid"] and b["valid"])
        self.assertEqual(a["executor_queue"], b["executor_queue"])

    def test_opening_guard_preserves_model_actions_and_one_rule_marine(self):
        answer = "- supplydepot x 3\n- barracks x 4"
        todo = dict(supplydepot=3, barracks=4, marine=8)
        r = pipeline(answer, True, todo=todo, phase=1)
        self.assertTrue(r["valid"])
        self.assertEqual(r["planner_queue"], ["supplydepot"] * 3 + ["barracks"] * 4)
        self.assertEqual([x["action"] for x in r["executor_queue"]], r["planner_queue"] + ["marine"])
        self.assertEqual(r["rule_added"], ["marine"])
        self.assertEqual(r["remaining_todo"], dict(marine=7))

    def test_opening_guard_old_trigger_position_and_number_unchanged(self):
        r = pipeline("- barracks\n- supplydepot x 7", True,
                     todo=dict(barracks=1, supplydepot=7, marine=8), phase=1)
        self.assertEqual([x["action"] for x in r["executor_queue"]], ["barracks", "marine"] + ["supplydepot"] * 7)
        self.assertEqual(r["rule_added"], ["marine"])
        r = pipeline("- supplydepot x 7", True, todo=dict(supplydepot=7, marine=8),
                     obs="It is now 00:00 in the game.", phase=1)
        self.assertEqual(r["rule_added"], [])

    def test_invalid_opening_answer_still_uses_old_guard(self):
        todo = dict(supplydepot=8, barracks=4, marine=8)
        a = pipeline("- barracks x 99", False, todo=todo, phase=1)
        b = pipeline("- barracks x 99", True, todo=todo, phase=1)
        self.assertFalse(b["valid"])
        self.assertEqual(a["executor_queue"], b["executor_queue"])

    def test_attack_length_not_truncated_but_attack_position_frozen(self):
        for position in ("before_build", "after_build"):
            with self.subTest(position=position):
                r = pipeline("- Launch OFFENSE with 8 marine\n- marine x 20", True,
                             obs=OBS + " Supply army: 20", attack_position=position)
                self.assertTrue(r["valid"])
                attack = ["Attacking Instruction: Launch OFFENSE with 8 marine"]
                expected = attack + ["marine"] * 20 if position == "before_build" else ["marine"] * 20 + attack
                self.assertEqual([x["action"] for x in r["executor_queue"]], expected)

    def test_prompt_only_protocol_text_changes_not_context(self):
        a, b = pipeline("- marine", False), pipeline("- marine", True)
        self.assertIn("approximately 15 entries", b["query"])
        self.assertIn("- marine x 2", b["query"])
        self.assertNotIn("output at most 6", b["query"])
        self.assertNotIn("Do not use `x N`", b["query"])
        for text in (a["query"], b["query"]):
            self.assertIn("## Optional Operations/Units:", text)
        start, end = "Now the phased strategic instructions are:", "Use only exact canonical names"
        self.assertEqual(a["query"].split(start)[1].split(end)[0], b["query"].split(start)[1].split(end)[0])

    def test_paired_eval_launcher_only_changes_protocol_and_gpu_selection(self):
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        entry = ROOT / "experiments/records/20260909-bo-segment-protocol-v1/run_eval.sh"
        configs = []
        with tempfile.TemporaryDirectory() as tmp:
            for arm in ("A", "B"):
                folder = Path(tmp) / arm
                (folder / "model").mkdir(parents=True)
                (folder / "sc2").mkdir()
                env = {k: v for k, v in os.environ.items() if not k.startswith(
                    ("SC2_", "SPB_", "MODEL_", "CONFIG_", "MAX_", "BO_", "CUDA_VISIBLE_DEVICES"))}
                env.update(SOURCE_DIR=str(ROOT), SOURCE_SHA=sha, PYTHON_BIN=sys.executable,
                           DRY_RUN="1", VERIFY_ASSETS="0", MODEL_PATH=str(folder / "model"),
                           SC2PATH=str(folder / "sc2"), RUN_ID="segment-" + arm,
                           OUTPUT_DIR=str(folder / "run"), RAY_TMPDIR=str(folder / "ray"))
                run = subprocess.run(["bash", str(entry), arm], env=env, text=True, capture_output=True)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                cfg = json.loads((folder / "run/effective_config.json").read_text())["overrides"]
                manifest = json.loads((folder / "run/manifest.json").read_text())
                self.assertEqual(manifest["mode"], "evaluation")
                self.assertEqual(manifest["world_size"], 4)
                self.assertEqual(manifest["gpu_selection"], "0,1,2,3" if arm == "A" else "4,5,6,7")
                for key, value in {"trainer.val_only": "True", "trainer.resume_mode": "disable",
                    "trainer.save_freq": "-1", "env.rollout.n": "4", "+env.val_rollout_n": "4",
                    "actor_rollout_ref.rollout.val_kwargs.n": "4", "env.history_length": "0",
                    "data.max_response_length": "96",
                    "+ray_init.runtime_env.env_vars.SC2_FIXED_BO_PREFIX_LIMIT": "'-1'",
                    "+ray_init.runtime_env.env_vars.SC2_PLAN_SEGMENT_MAX": "'6'",
                    "+ray_init.runtime_env.env_vars.SC2_PLAN_BULLET_MAX": "'6'",
                    "+actor_rollout_ref.model.override_config.swa_dropout": "0.0"}.items():
                    self.assertEqual(cfg[key], value, key)
                flag = cfg.pop("+ray_init.runtime_env.env_vars.SC2_ORIGINAL_BO_SEGMENT")
                self.assertEqual(flag, "'0'" if arm == "A" else "'1'")
                configs.append({k: v for k, v in cfg.items() if str(folder) not in v and k != "trainer.experiment_name"})
        self.assertEqual(configs[0], configs[1])


class ResponseBudgetLaunchTest(unittest.TestCase):
    def test_b96_and_b192_only_change_response_budget(self):
        entry = ROOT / "experiments/records/20260909-bo-segment-protocol-v1/run_eval.sh"
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        configs = []
        with tempfile.TemporaryDirectory() as tmp:
            for budget, gpus in ((96, 4), (192, 4), (192, 8)):
                folder = Path(tmp) / f"{budget}-{gpus}"
                (folder / "model").mkdir(parents=True)
                (folder / "sc2").mkdir()
                env = {k: v for k, v in os.environ.items() if not k.startswith(
                    ("SC2_", "SPB_", "MODEL_", "CONFIG_", "MAX_", "BO_", "CUDA_VISIBLE_DEVICES"))}
                env.update(SOURCE_DIR=str(ROOT), SOURCE_SHA=sha, PYTHON_BIN=sys.executable,
                           DRY_RUN="1", VERIFY_ASSETS="0", MODEL_PATH=str(folder / "model"),
                           SC2PATH=str(folder / "sc2"), RUN_ID="budget-" + str(budget),
                           OUTPUT_DIR=str(folder / "run"), RAY_TMPDIR=str(folder / "ray"))
                if budget == 192:
                    env["SC2_EVAL_RESPONSE_LENGTH"] = "192"
                if gpus == 8:
                    env.update(SC2_EVAL_N_GPUS="8", CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7")
                run = subprocess.run(["bash", str(entry), "B"], env=env, text=True, capture_output=True)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                cfg = json.loads((folder / "run/effective_config.json").read_text())["overrides"]
                manifest = json.loads((folder / "run/manifest.json").read_text())
                self.assertEqual(manifest["mode"], "evaluation")
                self.assertEqual(manifest["world_size"], gpus)
                self.assertEqual(manifest["gpu_selection"], "4,5,6,7" if gpus == 4 else "0,1,2,3,4,5,6,7")
                self.assertEqual(cfg.pop("data.max_response_length"), str(budget))
                self.assertEqual(cfg.pop("trainer.n_gpus_per_node"), str(gpus))
                self.assertEqual(cfg.pop("env.rollout.n"), str(gpus))
                self.assertEqual(cfg["+env.val_rollout_n"], "4")
                self.assertEqual(cfg["actor_rollout_ref.rollout.val_kwargs.n"], "4")
                self.assertEqual(cfg["+ray_init.runtime_env.env_vars.SC2_ORIGINAL_BO_SEGMENT"], "'1'")
                self.assertEqual(cfg["trainer.val_only"], "True")
                self.assertEqual(cfg["trainer.resume_mode"], "disable")
                self.assertEqual(cfg["trainer.save_freq"], "-1")
                configs.append({k: v for k, v in cfg.items()
                                if str(folder) not in v and k != "trainer.experiment_name"})
        self.assertEqual(configs[0], configs[1])
        self.assertEqual(configs[1], configs[2])

    def test_existing_inference_padding_keeps_four_real_rank_outputs(self):
        tree = ast.parse((ROOT / "verl/protocol.py").read_text())
        names = {"pad_dataproto_to_divisor", "unpad_dataproto"}
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        class Rows(list):
            def __getitem__(self, key):
                result = super().__getitem__(key)
                return Rows(result) if isinstance(key, slice) else result
            @classmethod
            def concat(cls, parts):
                return cls(item for part in parts for item in part)
        scope = {"DataProto": Rows}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "frozen_protocol", "exec"), scope)
        padded, size = scope["pad_dataproto_to_divisor"](Rows(range(4)), 8)
        self.assertEqual(padded, [0, 1, 2, 3, 0, 1, 2, 3])
        self.assertEqual(size, 4)
        actual = scope["unpad_dataproto"](Rows(f"rank{rank}" for rank in range(8)), size)
        self.assertEqual(actual, ["rank0", "rank1", "rank2", "rank3"])


if __name__ == "__main__":
    unittest.main()
