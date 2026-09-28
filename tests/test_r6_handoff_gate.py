import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/records/20260905-r6-format-reward-v1/check_control.py"
spec = importlib.util.spec_from_file_location("format_gate", SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class HandoffGateTest(unittest.TestCase):
    def fixture(self, root):
        outcomes, rows, audit = [], [], []
        for i in range(8):
            applied = i < 4
            outcomes.append(dict(env_index=i, terminated=True, truncated=False,
                                 game_result="Victory", policy_decision_count=int(applied)))
            rows.append(dict(env_idx=i, done=True, reward=20, policy_action_applied=applied,
                             policy_action_id=1, last_consumed_policy_action_id=1 if applied else None))
            audit.append(dict(environment_reward=20, environment_interactions=1,
                              policy_decisions=int(applied), excluded_terminal_rows=int(not applied)))
        result = dict(status="completed", process_exit_code=0, completed_steps=1, target_steps=1,
                      checkpoint_complete=True, metrics_finite=True, episode_outcomes=dict(outcomes=outcomes))
        metrics = [dict(step=1, **{"actor/grad_norm": 100., "actor/ppo_kl": .1, "actor/kl_loss": .1})]
        for name, value in ((".done", result), ("result.json", result), ("metrics.json", metrics),
                            ("manifest.json", dict(source_commit=gate.CONTROL_SHA, dirty=False, formal=True))):
            (root / name).write_text(json.dumps(value))
        (root / "prompt_response.jsonl").write_text("\n".join(map(json.dumps, rows)))
        (root / "console.log").write_text("SC2 policy credit: " + json.dumps(audit))
        return result, metrics

    def test_pending_and_valid_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(gate.check_control(root)["status"], "pending")
            self.fixture(root)
            self.assertEqual(gate.check_control(root)["policy_decisions"], 4)

    def test_rejects_failure_empty_metrics_and_wrong_consumption(self):
        for defect in ("failed", "empty", "nan", "consumption"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                result, metrics = self.fixture(root)
                if defect == "failed":
                    result["status"] = "failed"
                    (root / "result.json").write_text(json.dumps(result))
                elif defect in ("empty", "nan"):
                    metrics = [] if defect == "empty" else [{**metrics[0], "actor/grad_norm": float("nan")}]
                    (root / "metrics.json").write_text(json.dumps(metrics))
                else:
                    text = (root / "prompt_response.jsonl").read_text().replace('"last_consumed_policy_action_id": 1',
                                                                             '"last_consumed_policy_action_id": 2')
                    (root / "prompt_response.jsonl").write_text(text)
                with self.assertRaises(ValueError):
                    gate.check_control(root)
