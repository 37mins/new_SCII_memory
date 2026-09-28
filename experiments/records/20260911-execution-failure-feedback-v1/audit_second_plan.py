"""Read retained evaluation artifacts; no model, SC2, or runtime mutation.

Usage: existing-decoder/bin/python -B audit_second_plan.py SNAPSHOT > report.json
SNAPSHOT has original/A4/B4 metadata and A4/B4/env{1,3,7} replays/traces/logs.
Reuses the existing s2protocol-75689 tracker decoder. Final-action trace entries
are pre-dispatch intents, not proof of SC2 acceptance; births come from replay.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import runpy
import sys

from mpyq import MPQArchive
from s2protocol.versions import build, latest

SHA = "937ae3cfcbf7340208a8f86c0abf680d733c0484"
SEEDS = [42613, 52620, 17089, 4018390961, 3236598124, 3297512220, 773351930, 1098341402]
MARKER = "Execution feedback for the previous segment (not owned units/buildings):\n"
FPS = 22.4


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def feedback(prompt):
    return json.JSONDecoder().raw_decode(prompt.split(MARKER, 1)[1])[0] if MARKER in prompt else None


def main(root):
    prior = Path(__file__).parents[1] / "20260908-garrison-wave-ownership-v1/decode_failed_seed_replays.py"
    decode = runpy.run_path(str(prior))["decode"]
    reports, streams, second_queries = {}, {}, {}
    for arm in ("original", "A4", "B4"):
        folder = root / arm
        result = json.loads((folder / "result.json").read_text())
        manifest = json.loads((folder / "manifest.json").read_text())
        load = json.loads((folder / "checkpoint_evaluation.json").read_text())
        assert manifest["source_commit"] == SHA and not manifest["dirty"]
        assert result == json.loads((folder / ".done").read_text())
        assert result["process_exit_code"] == result["training_updates"] == load["optimizer_updates"] == 0
        assert load["loaded_global_step"] == (0 if arm == "original" else 4)
        assert load["game_seeds"] == SEEDS and load["val_only"] and load["feedback"] == "1"
        assert result["console_sha256"] == digest(folder / "console.log")
        outcomes = {o["env_index"]: o for o in rows(folder / "episode_outcomes.jsonl")}
        assert len(outcomes) == 8 and all(o["terminated"] and not o["truncated"] for o in outcomes.values())
        responses = rows(folder / "prompt_response.jsonl")
        mapping = {r["env"]: r["game_pid"] for r in rows(folder / "game_worker_mapping.jsonl")}
        report = dict(result=result, input_hashes={p.name: digest(p) for p in folder.iterdir()
                      if p.is_file() and p.suffix in (".json", ".jsonl")}, episodes=[])
        for env, seed in enumerate(SEEDS):
            out = outcomes[env]
            active = [(n, r) for n, r in enumerate(responses, 1)
                      if r["env_idx"] == env and r["step_index"] <= out["step_index"]]
            policy = [(n, r) for n, r in active if r["policy_action_applied"]]
            assert len(policy) == out["policy_decision_count"]
            decisions = []
            for n, r in policy:
                mm, ss = map(int, re.search(r"It is now (\d+):(\d+)", r["prompt"]).groups())
                decisions.append(dict(raw_line=n, game_second=60*mm+ss,
                    prompt_sha256=hashlib.sha256(r["prompt"].encode()).hexdigest(),
                    answer=r["llm_response"], valid=r["is_action_valid"], feedback=feedback(r["prompt"])))
            ep = dict(env=env, seed=seed, outcome=out, decisions=decisions,
                      reward=sum(float(r["reward"]) for _, r in active),
                      valid=sum(bool(r["is_action_valid"]) for _, r in policy))
            second_queries[arm, env] = policy[1][1]["prompt"]
            if arm != "original" and env in (1, 3, 7):
                game = folder / f"env{env}"
                replay = game / "replay.SC2Replay"
                replay_key = next(k for k in result["replays"] if f"game-{mapping[env]}/" in k)
                assert digest(replay) == result["replays"][replay_key]
                ledger = decode(replay)
                archive = MPQArchive(str(replay))
                header = latest().decode_replay_header(archive.header["user_data_header"]["content"])
                assert header["m_version"]["m_baseBuild"] == 75689
                protocol = build(75689)
                events = list(protocol.decode_replay_tracker_events(archive.read_file("replay.tracker.events")))
                details = protocol.decode_replay_details(archive.read_file("replay.details"))
                assert details["m_playerList"][0]["m_result"] == (1 if out["game_result"] == "Victory" else 2)
                streams[arm, env] = [e for e in events if e["_gameloop"] / FPS <= decisions[1]["game_second"]]
                born = [dict(second=e["_gameloop"]/FPS, unit=e["m_unitTypeName"].decode(),
                             owner=e["m_controlPlayerId"], kind=e["_event"].split(".")[-1]) for e in events
                        if e["_event"].endswith(("SUnitBornEvent", "SUnitInitEvent"))]
                trace = json.loads((game / "trace.json").read_text())
                intents, failures = [], []
                for iteration, record in trace.items():
                    # Frozen step size is four loops. Verify against recorded game seconds.
                    sec = int(iteration)*4/FPS
                    if "time_seconds" in record:
                        assert abs(record["time_seconds"] - sec) < 1.01
                    for key, value in record.items():
                        if key.startswith("llm_decision_final_action_"):
                            intents.append(dict(second=sec, trace_key=f"{iteration}/{key}", actions=value))
                    if "execution_failure_feedback" in record:
                        failures.append(dict(second=sec, iteration=iteration, **record["execution_failure_feedback"]))
                ep["execution"] = dict(game_pid=mapping[env], artifact_hashes={p.name:digest(p) for p in game.iterdir()},
                    seconds=ledger["seconds"], own_born=ledger["own_born"], own_deaths=ledger["own_deaths"],
                    failures=failures, recorded_pre_dispatch_intents=intents, windows=[])
                boundaries = sorted(set([0, decisions[1]["game_second"], 300, 360, 420, 480, 540, 600,
                                         720, 900, 1200, int(ledger["seconds"])+1]))
                for lo, hi in zip(boundaries, boundaries[1:]):
                    if lo > ledger["seconds"]: break
                    deaths = [d for d in ledger["deaths"] if lo <= d["second"] < hi]
                    bs = [b for b in born if lo <= b["second"] < hi and b["owner"] == 1]
                    ep["execution"]["windows"].append(dict(seconds=[lo,hi],
                        own_born=dict(Counter(b["unit"] for b in bs if b["kind"]=="SUnitBornEvent")),
                        own_started=dict(Counter(b["unit"] for b in bs if b["kind"]=="SUnitInitEvent")),
                        own_deaths=dict(Counter(d["birth_type"] for d in deaths if d["owner"]==1)),
                        enemy_deaths=dict(Counter(d["birth_type"] for d in deaths if d["owner"]==2)),
                        marine_killers=dict(Counter(str(d["killer"]) for d in deaths if d["owner"]==1 and d["birth_type"]=="Marine"))))
                ep["execution"]["marine_birth_seconds"] = [b["second"] for b in born if b["owner"]==1 and b["unit"]=="Marine"]
                ep["execution"]["losses"] = [d for d in ledger["deaths"] if d["owner"]==1]
                game_log = next(game.glob("default_player_*.log"))
                ep["execution"]["attack_events"] = []
                for line_number, line in enumerate(game_log.read_text().splitlines(), 1):
                    if any(s in line for s in ("已创建总攻波次", "集结完毕", "已全灭")):
                        match = re.search(r"\[INFO\]: \((\d+)\) (.*)", line)
                        ep["execution"]["attack_events"].append(dict(line=line_number,
                            second=int(match[1])*4/FPS, text=match[2]))
            report["episodes"].append(ep)
        reward = sum(ep["reward"] for ep in report["episodes"])/8
        assert abs(reward-result["metrics"]["val/text/test_score"]) < .001
        assert sum(e["valid"] for e in report["episodes"]) == result["valid_evaluable_action_count"]
        report["recomputed_mean_reward"] = reward
        reports[arm] = report
    equal = [second_queries["A4",e] == second_queries["B4",e] for e in range(8)]
    assert all(equal)
    prefix = {str(e): streams["A4",e] == streams["B4",e] for e in (1,3,7)}
    print(json.dumps(dict(runtime_sha=SHA, scope="read-only existing artifacts; no GPU or gameplay",
        second_prompts_equal=equal, tracker_prefix_equal_through_second_query_integer_second=prefix,
        reports=reports), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
