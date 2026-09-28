"""Read three retained L6 replays; no game, model or training changes.

Usage: decoder/bin/python -B audit_l6_binary_replays.py SNAPSHOT_ROOT
SNAPSHOT_ROOT contains step20/step27/step28/replay.SC2Replay.
Reuses the existing tracker ledger; outputs a compact JSON report to stdout.
"""
from collections import Counter
import json
from pathlib import Path
import runpy
import sys

from mpyq import MPQArchive
from s2protocol.versions import build, latest


EXPECTED = {
    20: ("0a2605d695966618c7c77ec28e862f6726a4ca2c77ab75e397b9bbf633d83e16", 2),
    27: ("a0dd22d6de5fe64cd3ab392de8ed2059c0a9aba0bae4b6d83b4b306a8ced83e4", 1),
    28: ("5bc0577c16052a8b9d8b2fd23ec6f86f93ddb6f0c8759c0a2bc4b3b0b8dbd4e2", 2),
}


def main(snapshot_root):
    prior = Path(__file__).parents[1] / "20260908-garrison-wave-ownership-v1/decode_failed_seed_replays.py"
    decode = runpy.run_path(str(prior))["decode"]
    reports = []
    for step, (sha, outcome) in EXPECTED.items():
        path = snapshot_root / f"step{step}/replay.SC2Replay"
        ledger = decode(path)
        assert ledger["sha256"] == sha
        archive = MPQArchive(str(path))
        header = latest().decode_replay_header(archive.header["user_data_header"]["content"])
        assert header["m_version"]["m_baseBuild"] == 75689
        protocol = build(75689)
        details = protocol.decode_replay_details(archive.read_file("replay.details"))
        assert details["m_playerList"][0]["m_result"] == outcome
        starts = []
        for event in protocol.decode_replay_tracker_events(archive.read_file("replay.tracker.events")):
            if event["_gameloop"] > 1:
                break
            if event.get("m_unitTypeName") == b"CommandCenter":
                starts.append({"player": event["m_controlPlayerId"], "xy": [event["m_x"], event["m_y"]]})
        own = [d for d in ledger["deaths"] if d["owner"] == 1]
        snapshots = [
            {"second": s["second"], "players": {
                player: {unit: n for unit, n in counts.items() if not unit.startswith("Beacon")}
                for player, counts in s["players"].items()
            }}
            for s in ledger["snapshots"] if s["second"] in (300, 600, 720)
        ]
        reports.append({
            "step": step, "replay_sha256": sha, "seconds": ledger["seconds"],
            "result": "Victory" if outcome == 1 else "Defeat", "starts": starts,
            "own_born": {k: v for k, v in ledger["own_born"].items() if not k.startswith("Beacon")},
            "own_deaths": ledger["own_deaths"],
            "marine_killers": dict(Counter(str(d["killer"]) for d in own if d["birth_type"] == "Marine")),
            "worker_killers": dict(Counter(str(d["killer"]) for d in own if d["birth_type"] == "SCV")),
            "own_upgrades": [u for u in ledger["upgrades"] if u["player"] == 1],
            "snapshots": snapshots,
        })
    print(json.dumps({"runtime_sha": "6a1657f7e045dcf998495a3d0adfb331572bf60d",
                      "base_build": 75689, "loops_per_second": 22.4,
                      "scope": "One retained replay per selected batch; not all eight trajectories.",
                      "reports": reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
