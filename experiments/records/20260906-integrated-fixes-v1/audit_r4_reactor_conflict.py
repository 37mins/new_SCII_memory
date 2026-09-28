"""Read-only extraction of Reactor waits and producer occupancy from the frozen g2 trace."""
import hashlib
import json
from pathlib import Path
import re
import sys


def audit(run):
    assert (run / "source_sha.txt").read_text().strip() == "836460a8533b6f0429f9e9d401ac9715adc55f5b"
    assert (run / ".done").read_text().strip() == "EXIT_CODE=0"
    trace = json.loads((run / "boundary_trace.json").read_text())
    config = json.loads((run / "effective_config.json").read_text())
    assert config["env"]["SC2_LOCAL_EXECUTOR_FALLBACK"] == "1"
    commands, waits, snapshots = [], [], []
    for frame, row in sorted(trace.items(), key=lambda pair: int(pair[0])):
        for key, value in row.items():
            if key.startswith("llm_decision_final_action_"):
                bo = int(key.split("_")[-2])
                if 91 <= bo <= 114:
                    commands.append(dict(frame=int(frame), bo=bo, actions=value))
    for bo, start, end in ((110, 2112, 2332), (114, 2336, 2556)):
        skipped = trace[str(end)][f"bo_skipped_{bo}"]
        assert skipped == dict(reason="timeout", action="reactor barracks")
        interval = [r for f, r in trace.items() if start <= int(f) <= end]
        waits.append(dict(bo=bo, start_frame=start, end_frame=end,
            game_seconds=(end-start)*4/22.4, result=skipped,
            full_observations_in_wait=sum("obs" in r for r in interval)))
    for frame in (2095, 2109, 2111, 2333, 2334, 2557):
        obs = trace[str(frame)]["obs"]
        buildings = []
        for block in re.split(r"\n(?=\[\d+\])", obs["Own structures"]):
            header = re.match(r"^\[(\d+)\]Barracks\n", block)
            if not header:
                continue
            position = re.search(r"Position: \((\d+), (\d+)\)", block)
            production = re.search(r"^Production list: (.+)$", block, re.M)
            addon = re.search(r"add-on: \[(\d+)\](Barracks\w+)", block)
            buildings.append(dict(id=int(header[1]),
                position=[int(position[1]), int(position[2])],
                marine_orders=0 if not production else production[1].split(", ").count("Marine"),
                addon_id=int(addon[1]) if addon else 0,
                addon_type=addon[2] if addon else None, raw=block))
        allowed = []
        for line in obs["Structure abilities"].splitlines():
            match = re.match(r"Barracks\[([\d, ]+)\]: (.+)", line)
            if match and "BARRACKSTRAIN_MARINE" in match[2].split(", "):
                allowed.extend(int(x) for x in match[1].split(", "))
        selected = [c for c in commands if c["frame"] == frame]
        snapshots.append(dict(frame=frame, barracks=buildings,
            marine_capable_ids=allowed, commands=selected))
    return dict(run_id=run.name, source_sha=(run / "source_sha.txt").read_text().strip(),
        waits=waits, commands=commands, snapshots=snapshots,
        limitations="Observation snapshots, not a full per-retry unit/order trace or engine ActionResult. Positions are integer-rendered. No game simulation or new gameplay result.",
        artifact_hashes={name: hashlib.sha256((run / name).read_bytes()).hexdigest()
            for name in ("boundary_trace.json", "driver.log", "effective_config.json", ".done")})


if __name__ == "__main__":
    print(json.dumps(audit(Path(sys.argv[1])), ensure_ascii=False, indent=2))
