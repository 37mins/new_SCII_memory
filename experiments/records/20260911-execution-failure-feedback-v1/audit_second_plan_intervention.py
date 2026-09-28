"""Read the retained eight-game second-answer A/B; no GPU, SC2 or model load.

Usage: existing-decoder/bin/python -B audit_second_plan_intervention.py SNAPSHOT [--expected-sha SHA]
Prints JSON; reuses the existing tracker decoder for births/deaths. Trace final
actions are pre-dispatch intents, not proof of command acceptance or unit birth.
"""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import re
import runpy

from mpyq import MPQArchive
from s2protocol.versions import build, latest

SHA = "a0e5425b2b9ff09db28d0c017f7c6dc7c0d1ba58"
FPS = 22.4


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stream_digest(events):
    return hashlib.sha256(json.dumps(events, sort_keys=True, default=str).encode()).hexdigest()


def main(root, expected_sha=SHA):
    def read(name):
        return json.loads((root / name).read_text())

    def rows(name):
        return [json.loads(line) for line in (root / name).read_text().splitlines()]

    decoder = Path(__file__).parents[1] / "20260908-garrison-wave-ownership-v1/decode_failed_seed_replays.py"
    decode = runpy.run_path(str(decoder))["decode"]
    manifest, result, load = read("manifest.json"), read("result.json"), read("checkpoint_evaluation.json")
    assert manifest["source_commit"] == expected_sha and not manifest["dirty"]
    assert result == read(".done") and result["process_exit_code"] == result["training_updates"] == 0
    assert load["loaded_global_step"] == 4 and load["optimizer_updates"] == 0 and load["val_only"]
    assert load["game_seeds"] == rows("evaluation_seeds.jsonl")[0]["seeds"] == [1098341402]*8
    outcomes = {r["env_index"]: r for r in rows("episode_outcomes.jsonl")}
    mapping = {r["env"]: r["game_pid"] for r in rows("game_worker_mapping.jsonl")}
    responses, interventions = rows("prompt_response.jsonl"), rows("interventions.jsonl")
    coverage = read("intervention_coverage.json")
    assert coverage["all_reached"] and len(interventions) == 16
    reports, trackers, command_prefixes = [], [], []
    for env in range(8):
        outcome = outcomes[env]
        assert outcome["terminated"] and not outcome["truncated"]
        active = [r for r in responses if r["env_idx"] == env and r["step_index"] <= outcome["step_index"]]
        policy = [r for r in active if r["policy_action_applied"]]
        assert len(policy) == outcome["policy_decision_count"]
        replay_key = next(k for k in result["replays"] if f"game-{mapping[env]}/" in k)
        game = (root / replay_key).parent
        assert digest(game / "replay.SC2Replay") == result["replays"][replay_key]
        ledger = decode(game / "replay.SC2Replay")
        archive = MPQArchive(str(game / "replay.SC2Replay"))
        header = latest().decode_replay_header(archive.header["user_data_header"]["content"])
        assert header["m_version"]["m_baseBuild"] == 75689
        protocol = build(75689)
        tracker = list(protocol.decode_replay_tracker_events(archive.read_file("replay.tracker.events")))
        commands = list(protocol.decode_replay_game_events(archive.read_file("replay.game.events")))
        details = protocol.decode_replay_details(archive.read_file("replay.details"))
        assert details["m_playerList"][0]["m_result"] == (1 if outcome["game_result"] == "Victory" else 2)
        trace = read(str((game / "trace.json").relative_to(root)))
        log_path = next(game.glob("default_player_*.log"))
        queues, attacks = [], []
        for line_no, line in enumerate(log_path.read_text().splitlines(), 1):
            match = re.search(r"\[INFO\]: \((\d+)\) (.*)", line)
            if not match:
                continue
            iteration, text = int(match[1]), match[2]
            if text.startswith("refresh build order: "):
                queues.append(dict(iteration=iteration, second=iteration*4/FPS,
                    line=line_no, actions=text.split(": ", 1)[1].split(", ")))
            if any(s in text for s in ("已创建总攻波次", "集结完毕", "已全灭")):
                attacks.append(dict(second=iteration*4/FPS, line=line_no, text=text))
        assert len(queues) == len(policy)
        second_loop = queues[1]["iteration"]*4
        assert queues[1]["iteration"] == 1462
        prefix = [e for e in tracker if e["_gameloop"] < second_loop]
        command_prefix = [e for e in commands if e["_gameloop"] < second_loop]
        trackers.append(tracker)
        command_prefixes.append(command_prefix)
        decisions = []
        for call, r in enumerate(policy):
            mm, ss = map(int, re.search(r"It is now (\d+):(\d+)", r["prompt"]).groups())
            if call < 2:
                iv = next(x for x in interventions if x["env"] == env and x["call_index"] == call)
                assert r["llm_response"] == iv["forwarded"]
                assert hashlib.sha256(r["prompt"].encode()).hexdigest() == iv["prompt_sha256"]
            decisions.append(dict(call=call+1, second=mm*60+ss, artificial=call<2,
                answer=r["llm_response"], valid=r["is_action_valid"], queue=queues[call]))
        births = [dict(second=e["_gameloop"]/FPS, unit=e["m_unitTypeName"].decode(),
            kind=e["_event"].split(".")[-1], x=e["m_x"], y=e["m_y"]) for e in tracker
            if e["_event"].endswith(("SUnitBornEvent", "SUnitInitEvent")) and e["m_controlPlayerId"] == 1]
        marine_births = [b["second"] for b in births if b["unit"] == "Marine"]
        marine_deaths = [d for d in ledger["deaths"] if d["owner"] == 1 and d["birth_type"] == "Marine"]
        maps, intents, failures = [], [], []
        for iteration, record in trace.items():
            sec = int(iteration)*4/FPS
            if "time_seconds" in record:
                assert abs(record["time_seconds"]-sec) < 1.01
            if "execution_failure_feedback" in record:
                failures.append(dict(second=sec, trace_key=iteration, **record["execution_failure_feedback"]))
            for key, value in record.items():
                item = dict(second=sec, trace_key=f"{iteration}/{key}", value=value)
                if key.startswith("llm_decision_bo_map_"):
                    maps.append(item)
                if key.startswith("llm_decision_final_action_"):
                    intents.append(item)
        windows = []
        for lo, hi in [(261,300),(300,319.64),(300,360),(360,420),(420,480),(480,600)]:
            deaths = [d for d in marine_deaths if lo <= d["second"] < hi]
            windows.append(dict(seconds=[lo,hi], marine_born=sum(lo <= b < hi for b in marine_births),
                marine_dead=len(deaths), killers=dict(Counter(d["killer"] for d in deaths))))
        reports.append(dict(env=env, arm=coverage["arms"][env], outcome=outcome["game_result"],
            seconds=ledger["seconds"], reward=sum(r["reward"] for r in active), game_path=str(game),
            online_choices=len(policy)-2, online_valid=sum(r["is_action_valid"] for r in policy[2:]),
            hashes={p.name:digest(p) for p in game.iterdir() if p.is_file()},
            tracker_prefix_count=len(prefix), tracker_prefix_sha256=stream_digest(prefix),
            game_events_prefix_count=len(command_prefix), game_events_prefix_sha256=stream_digest(command_prefix),
            decisions=decisions, own_born=ledger["own_born"], own_deaths=ledger["own_deaths"],
            marine_birth_seconds=marine_births, marine_deaths=marine_deaths,
            worker_deaths=[d for d in ledger["deaths"] if d["owner"] == 1 and d["birth_type"] == "SCV"],
            construction_starts=[b for b in births if b["kind"] == "SUnitInitEvent"],
            windows=windows, gaps=sorted([dict(seconds=b-a,start=a,end=b) for a,b in zip(marine_births,marine_births[1:])
                if a > second_loop/FPS],key=lambda x:x["seconds"],reverse=True)[:3],
            # A winning episode can end without another birth; do not report
            # its earlier inter-birth gaps as proof that production resumed.
            no_birth_tail_seconds=ledger["seconds"]-marine_births[-1],
            failures=failures, second_segment_mappings=[x for x in maps if queues[1]["second"] <= x["second"] < queues[2]["second"]],
            second_segment_intents=[x for x in intents if queues[1]["second"] <= x["second"] < queues[2]["second"]],
            attacks=attacks))
    prefixes = [r["tracker_prefix_sha256"] for r in reports]
    assert len(set(prefixes)) == 1
    first_diff = next((i,a,b) for i,(a,b) in enumerate(zip(trackers[0],trackers[1])) if a != b)
    assert abs(sum(r["reward"] for r in reports)/8-result["metrics"]["val/text/test_score"]) < .001
    prefix_differences = []
    for env, events in enumerate(command_prefixes):
        assert len(events) == len(command_prefixes[0])
        for index, (control, event) in enumerate(zip(command_prefixes[0], events)):
            if control != event:
                prefix_differences.append(dict(env=env, index=index, control=control, event=event))
    # Both observed runs differ only in a wall-clock sync value: playing time
    # in the whole-answer run, loading time in Marine-only env6. Retain all
    # differences and reject any changed game command or other event field.
    assert all(d["control"]["_event"].endswith(("SSetSyncPlayingTimeEvent", "SSetSyncLoadingTimeEvent")) and
        {k:v for k,v in d["control"].items() if k != "m_syncTime"} ==
        {k:v for k,v in d["event"].items() if k != "m_syncTime"} for d in prefix_differences)
    # Keep representative exact chains plus all-eight summaries; raw hashes
    # point to every complete trace, avoiding eight copies of expanded queues.
    for report in reports:
        for decision in report["decisions"]:
            queue = decision["queue"]
            queue["counts"] = dict(Counter(queue["actions"]))
            if report["env"] not in (0,1) or decision["call"] != 2:
                del queue["actions"]
        if report["env"] not in (0,1):
            for key in ("marine_deaths", "worker_deaths", "second_segment_intents", "second_segment_mappings"):
                del report[key]
        else:
            report["second_segment_intents"] = [x for x in report["second_segment_intents"]
                if x["second"] >= 299]
    print(json.dumps(dict(runtime_sha=expected_sha, snapshot=str(root), read_only=True,
        metadata_hashes={p.name:digest(p) for p in root.iterdir() if p.is_file()},
        tracker_prefix_all_equal=True, prefix_exclusive_loop=1462*4,
        game_events_prefix_all_equal=all(e == command_prefixes[0] for e in command_prefixes),
        game_events_prefix_differences=prefix_differences,
        game_events_prefix_equal_except_sync_wallclock=True,
        first_tracker_difference_env0_env1=dict(index=first_diff[0],control=first_diff[1],replacement=first_diff[2]),
        episodes=reports),ensure_ascii=False,default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--expected-sha", default=SHA)
    args = parser.parse_args()
    main(args.snapshot, args.expected_sha)
