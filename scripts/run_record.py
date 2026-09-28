#!/usr/bin/env python3
"""Record an exact launch command and audit completion; never select or repair a run."""
import argparse
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from verify_assets import REPO, digest

ARENA = REPO / "agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy"
METRIC = re.compile(r"([\w/]+):(-?(?:\d+(?:\.\d*)?|nan|inf)(?:e[+-]?\d+)?)", re.I)


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def start(args):
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    overrides = dict(item.split("=", 1) for item in command if "=" in item)
    training = overrides.get('trainer.val_only') != 'True'
    resume_path = overrides.get('trainer.resume_from_path')
    start_step = 0
    if training and overrides.get('trainer.resume_mode') == 'resume_path':
        match = re.fullmatch(r'global_step_(\d+)', Path(resume_path).name)
        if match is None:
            raise ValueError('Resume path must identify an exact global-step checkpoint')
        start_step = int(match[1])
    epochs = int(overrides['trainer.total_epochs'])
    goal = int(overrides.get('trainer.total_training_steps', epochs))
    dirty = bool(subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=normal"], text=True
    ).strip())
    versions = {}
    for package in ("torch", "transformers", "ray", "gymnasium", "flash-attn", "flash-linear-attention", "moba"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    source_files = [
        args.config, REPO / "configs/system-v1.env", REPO / "scripts/train_sc2.sh",
        ARENA / "agents/planner.py", ARENA / "players/script_player.py",
        ARENA / "players/llm_player.py",
        REPO / "agent_system/sc2_episode.py",
        REPO / "agent_system/environments/env_package/sc2_online_restored/envSC2.py",
        REPO / "agent_system/environments/strategic_sc2_env_manager_v2.py",
    ]
    record = {
        "schema_version": 1, "status": "dry_run" if args.dry_run else "running",
        "started_at": now(), "source_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "dirty": dirty, "formal": not dirty and not args.dry_run,
        "model_id": args.model_id, "model_revision": args.model_revision,
        "model_source": str(args.model_source.resolve()),
        "model_manifest_sha256": digest(args.model_manifest),
        "bo_profile": args.bo_profile,
        "episode_api": "gymnasium-v0.26",
        "bo_path": str(args.bo.resolve()), "bo_sha256": digest(args.bo),
        "mode": "evaluation" if overrides.get("trainer.val_only") == "True" else "training",
        "source_files_sha256": {str(p.resolve().relative_to(REPO)): digest(p) for p in source_files},
        "target_steps": min(start_step + epochs, goal) if training else epochs,
        "sc2_signal_max_batches": int(overrides.get('trainer.sc2_signal_max_batches', 0)),
        "start_step": start_step,
        "sc2_checkpoint_resume": overrides.get('+trainer.sc2_checkpoint_resume') == 'True',
        "training_goal_global_steps": int(overrides.get("trainer.total_training_steps", overrides["trainer.total_epochs"])),
        "resume_from_path": overrides.get("trainer.resume_from_path"),
        "host": platform.node(), "python": platform.python_version(),
        "gpu_selection": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "world_size": int(overrides["trainer.n_gpus_per_node"]) * int(overrides["trainer.nnodes"]),
        "container_image_id": os.environ.get("CONTAINER_IMAGE_ID"),
        "versions": versions,
    }
    write_json(args.output / "manifest.json", record)
    write_json(args.output / "effective_config.json", {"command": command, "overrides": overrides})


def metric_rows(log):
    rows = {}
    if not log.exists():
        return rows
    with log.open(errors="replace") as stream:
        for line in stream:
            if "actor/grad_norm:" not in line or "episode/success_rate:" not in line:
                continue
            match = re.search(r"\bstep:(\d+)\s+-", line)
            if match:
                rows[int(match[1])] = {k: float(v) for k, v in METRIC.findall(line)}
    return rows


def episode_outcomes(output):
    path = output / "episode_outcomes.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    identities = [(row["reset_index"], row["env_index"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate trajectory outcome records")
    natural = [row for row in rows if row["terminated"] and not row["truncated"]]
    wins = sum(row["game_result"] == "Victory" for row in natural)
    return {
        "trajectories": len(rows),
        "natural_games": len(natural),
        "wins": wins,
        "defeats": sum(row["game_result"] == "Defeat" for row in natural),
        "ties": sum(row["game_result"] == "Tie" for row in natural),
        "truncated": sum(row["truncated"] for row in rows),
        "natural_win_rate": wins / len(natural) if natural else None,
        "outcomes": rows,
    }


def finish(output: Path, exit_code: int) -> dict:
    manifest = output / "manifest.json"
    record = json.loads(manifest.read_text()) if manifest.exists() else {}
    if record.get("mode") == "evaluation":
        return finish_evaluation(output, exit_code)
    rows = metric_rows(output / "console.log")
    finite = all(math.isfinite(v) for row in rows.values() for v in row.values())
    target = record.get("target_steps")
    start_step = record.get('start_step', 0)
    signal_budget = record.get('sc2_signal_max_batches', 0)
    signal_path = output / 'checkpoints' / 'signal_sampling_summary.json'
    signal = json.loads(signal_path.read_text()) if signal_budget and signal_path.exists() else None
    actual_step = max(rows, default=start_step)
    checkpoint = output / "checkpoints" / f"global_step_{actual_step if signal_budget else target}"
    world_size = record.get("world_size", 8)
    expected = [checkpoint / "data.pt"] + [
        checkpoint / "actor" / f"{kind}_world_size_{world_size}_rank_{rank}.pt"
        for kind in ("model", "optim", "extra_state") for rank in range(world_size)
    ]
    if record.get('sc2_checkpoint_resume'):
        expected.append(checkpoint / 'sc2_state.pt')
    checkpoint_complete = all(p.is_file() and p.stat().st_size > 0 for p in expected)
    steps_complete = target is not None and target > start_step and sorted(rows) == list(range(start_step + 1, target + 1))
    signal_entries = []
    if signal_budget:
        ledger = output / 'checkpoints' / 'signal_sampling.jsonl'
        signal_entries = [json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
        accepted = [s for s in signal_entries if s['action'] == 'accept']
        skipped_signal = [s for s in signal_entries if s['action'] == 'skip_zero_signal']
        steps_complete = bool(signal and rows) and (
            signal['sampled_batches'] == len(signal_entries) == signal_budget
            and signal['start_global_step'] == start_step
            and signal['last_global_step'] == actual_step
            and signal['completed_global_steps'] == len(accepted) == len(rows)
            and signal['skipped_batches'] == len(skipped_signal)
            and len(accepted) + len(skipped_signal) == len(signal_entries)
            and all(not s['has_task_signal'] for s in skipped_signal)
            and sorted(rows) == list(range(start_step + 1, actual_step + 1))
            and [s['pending_global_step'] for s in accepted] == sorted(rows)
            and [s['collection_index'] for s in signal_entries] == list(range(1, signal_budget + 1)))
    completed = exit_code == 0 and steps_complete and finite and checkpoint_complete
    outcomes = episode_outcomes(output) if record.get("episode_api") == "gymnasium-v0.26" else None
    empty_log = output / 'empty_policy_batches.jsonl'
    empty_batches = [json.loads(line) for line in empty_log.read_text().splitlines()] if empty_log.exists() else []
    if outcomes is not None:
        config = json.loads((output / "effective_config.json").read_text())["overrides"]
        expected_games = (target - start_step) * int(config["data.train_batch_size"]) * int(config["env.rollout.n"])
        if signal_budget:
            expected_games = len(signal_entries) * int(config['env.rollout.n'])
            expected_ids = {(s['reset_index'], i) for s in signal_entries for i in range(int(config['env.rollout.n']))}
            actual_ids = {(s['reset_index'], s['env_index']) for s in outcomes['outcomes']}
            completed = completed and expected_ids == actual_ids and len(expected_ids) == expected_games
            completed = completed and signal is not None and signal['sampled_games'] == expected_games
        skipped = [(row['reset_index'], ep['env_index']) for row in empty_batches
                   if row['action'] == 'skip' for ep in row['episodes']]
        outcome_by_id = {(row['reset_index'], row['env_index']): row for row in outcomes['outcomes']}
        skip_valid = len(skipped) == len(set(skipped)) and all(
            key in outcome_by_id and outcome_by_id[key]['policy_decision_count'] == 0
            and outcome_by_id[key]['terminated'] and not outcome_by_id[key]['truncated']
            for key in skipped)
        completed = completed and skip_valid and all(row['action'] == 'skip' for row in empty_batches)
        expected_games += len(skipped)
        completed = completed and outcomes["natural_games"] == expected_games and outcomes["truncated"] == 0
    result = {
        "status": "completed" if completed else "failed", "ended_at": now(),
        "process_exit_code": exit_code, "completed_steps": len(rows),
        "start_step": start_step, "last_global_step": max(rows, default=start_step),
        "target_steps": target, "metrics_finite": finite,
        "checkpoint_complete": checkpoint_complete,
        "checkpoint_files": {
            str(p.relative_to(output)): {
                "bytes": p.stat().st_size,
                "sha256": digest(p) if p.name.startswith("model_world_size") else None,
            }
            for p in expected if p.is_file()
        },
        "console_sha256": digest(output / "console.log") if (output / "console.log").exists() else None,
    }
    if outcomes is not None:
        result["episode_outcomes"] = outcomes
    result['empty_policy_batches'] = empty_batches
    if signal_budget:
        result['signal_sampling'] = signal
        result['update_target_reached'] = actual_step >= target
        result['learning_validated'] = False
        result['completion_basis'] = 'bounded sampling and actual updates; not the nominal update target'
    # Keep non-finite evidence valid JSON rather than silently dropping the affected row.
    write_json(output / "metrics.json", [
        {"step": step, **{k: v if math.isfinite(v) else str(v) for k, v in row.items()}}
        for step, row in sorted(rows.items())
    ])
    write_json(output / "result.json", result)
    write_json(output / ".done", result)
    return result


def finish_evaluation(output: Path, exit_code: int) -> dict:
    log = output / "console.log"
    lines = log.read_text(errors="replace").splitlines() if log.exists() else []
    metrics = {}
    for line in lines:
        if re.search(r"\bstep:\d+\s+-", line) and "val/" in line:
            metrics.update({k: float(v) for k, v in METRIC.findall(line) if k.startswith("val/")})
    updates = len(metric_rows(log))
    replay_files = list((output / "sc2-logs").rglob("*.SC2Replay"))
    prompt_log = output / "prompt_response.jsonl"
    plans = [json.loads(line) for line in prompt_log.read_text().splitlines()] if prompt_log.exists() else []
    actions = [p for p in plans if p.get("semantic_action_evaluable", False)]
    finite = bool(metrics) and all(math.isfinite(v) for v in metrics.values())
    completed = exit_code == 0 and finite and updates == 0 and len(replay_files) == 1
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    outcomes = episode_outcomes(output) if manifest.get("episode_api") == "gymnasium-v0.26" else None
    if outcomes is not None:
        config = json.loads((output / "effective_config.json").read_text())["overrides"]
        expected_games = int(config.get("data.val_batch_size", "1")) * int(config.get("+env.val_rollout_n", "1"))
        completed = (exit_code == 0 and finite and updates == 0
                     and outcomes["natural_games"] == expected_games and outcomes["truncated"] == 0)
    result = {
        "status": "completed" if completed else "failed", "ended_at": now(),
        "mode": "evaluation", "process_exit_code": exit_code, "training_updates": updates,
        "metrics": {k: v if math.isfinite(v) else str(v) for k, v in metrics.items()},
        "metrics_finite": finite, "response_record_count": len(plans),
        "evaluable_action_count": len(actions),
        "valid_evaluable_action_count": sum(bool(p.get("is_action_valid")) for p in actions),
        "replays": {str(p.relative_to(output)): digest(p) for p in replay_files},
        "console_sha256": digest(log) if log.exists() else None,
    }
    if outcomes is not None:
        result["episode_outcomes"] = outcomes
        result["natural_match_completed"] = outcomes["natural_games"] == expected_games
        result["termination_evidence"] = (
            outcomes["outcomes"][0]["termination_reason"] if outcomes["outcomes"] else None
        )
    write_json(output / "result.json", result)
    write_json(output / ".done", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    launch = sub.add_parser("start")
    launch.add_argument("--output", type=Path, required=True)
    launch.add_argument("--config", type=Path, required=True)
    launch.add_argument("--bo", type=Path, required=True)
    launch.add_argument("--bo-profile", required=True)
    launch.add_argument("--model-source", type=Path, required=True)
    launch.add_argument("--model-manifest", type=Path, required=True)
    launch.add_argument("--model-id", required=True)
    launch.add_argument("--model-revision", required=True)
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("command", nargs=argparse.REMAINDER)
    end = sub.add_parser("finish")
    end.add_argument("--output", type=Path, required=True)
    end.add_argument("--exit-code", type=int, required=True)
    args = parser.parse_args()
    if args.action == "start":
        start(args)
    else:
        result = finish(args.output, args.exit_code)
        print(json.dumps(result))
        raise SystemExit(0 if result["status"] == "completed" else 1)
