"""Audit recorded SC2 entities and Marine occupancy; no engine simulation."""
import hashlib
import json
from pathlib import Path
import re
import sys


def audit(run):
    assert (run / ".done").read_text().strip() == "EXIT_CODE=0"
    result = json.loads((run / "result.json").read_text())
    assert result["planner_answers_sent"] == 0
    trace_path = run / "boundary_trace.json"
    if not trace_path.exists():
        traces = list((run / "sc2-logs").rglob("trace.json"))
        assert len(traces) == 1, traces
        trace_path = traces[0]
    trace = json.loads(trace_path.read_text())
    first_building, first_complete = {}, {}
    commands, skipped, occupancy = [], [], []
    names = ("BarracksReactor", "FactoryTechLab", "SiegeTank")
    for frame, row in sorted(trace.items(), key=lambda item: int(item[0])):
        obs = row.get("obs", {})
        for field in ("Own structures", "Own units"):
            for block in re.split(r"\n(?=\[\d+\])", obs.get(field, "")):
                header = re.match(r"^\[(\d+)\](BarracksReactor|FactoryTechLab|SiegeTank(?:Sieged)?)(?:\(building (\d+)%\))?\n", block)
                if header:
                    name = "SiegeTank" if header[2] == "SiegeTankSieged" else header[2]
                    key = (name, int(header[1]))
                    item = dict(frame=int(frame), game_seconds=int(frame)*4/22.4,
                        type=name, id=key[1], raw=block)
                    (first_complete if header[3] is None else first_building).setdefault(key, item)
        for key, value in row.items():
            if key.startswith("bo_skipped_"):
                skipped.append(dict(frame=int(frame), bo=int(key.split("_")[-1]), **value))
            if not key.startswith("llm_decision_final_action_"):
                continue
            bo = int(key.split("_")[-2])
            for action in value:
                if action["action"] in ("BUILD_REACTOR_BARRACKS", "BUILD_TECHLAB_FACTORY", "FACTORYTRAIN_SIEGETANK"):
                    commands.append(dict(frame=int(frame), bo=bo, action=action))
                if action["action"] == "BARRACKSTRAIN_MARINE":
                    for unit in action["units"]:
                        blocks = [b for b in re.split(r"\n(?=\[\d+\])", obs.get("Own structures", ""))
                            if b.startswith(f"[{unit}]Barracks\n")]
                        assert len(blocks) == 1, (frame, unit)
                        production = re.search(r"^Production list: (.+)$", blocks[0], re.M)
                        count = len(production[1].split(", ")) if production else 0
                        capacity = 2 if re.search(r"add-on: \[\d+\]BarracksReactor", blocks[0]) else 1
                        occupancy.append(dict(frame=int(frame), bo=bo, unit=unit,
                            orders_before=count, slots=capacity, overflow=count >= capacity))
    cleanup = [json.loads(line) for line in (run / "sc2-logs/cleanup.jsonl").read_text().splitlines()]
    boundary_path = run / "first_planner.json"
    boundary = json.loads(boundary_path.read_text()) if boundary_path.exists() else None
    if boundary:
        boundary.pop("request")
    artifacts = [".done", "source_sha.txt", "started_at.txt", "ended_at.txt", "effective_config.json",
        "driver.log", "result.json", "sc2-logs/cleanup.jsonl", str(trace_path.relative_to(run))]
    artifacts += [p.name for p in (run / "first_planner.json", run / "terminal.json") if p.exists()]
    return dict(run_id=run.name, source_sha=(run / "source_sha.txt").read_text().strip(),
        started_at=(run / "started_at.txt").read_text().strip(), ended_at=(run / "ended_at.txt").read_text().strip(),
        exit_code=0, result=result, boundary=boundary,
        first_observed_building=list(first_building.values()), first_observed_complete=list(first_complete.values()),
        distinct_complete_observed={name: sum(k[0] == name for k in first_complete) for name in names},
        commands=commands, skipped=skipped, marine_commands=len(occupancy),
        marine_slot_overflows=[o for o in occupancy if o["overflow"]],
        max_marine_producer_orders_before=max((o["orders_before"] for o in occupancy), default=None),
        cleanup=cleanup, replays=[str(p.relative_to(run)) for p in run.rglob("*.SC2Replay")],
        artifact_hashes={name: hashlib.sha256((run/name).read_bytes()).hexdigest() for name in artifacts},
        limitation="Entity observations, not exact birth/done times; display IDs persist within this game. No model or PPO. A Planner boundary is not a win/loss. Occupancy is pre-command observation, not engine acceptance.")


if __name__ == "__main__":
    print(json.dumps(audit(Path(sys.argv[1])), ensure_ascii=False, indent=2))
