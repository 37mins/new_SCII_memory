#!/usr/bin/env python3
"""Convert SC2Replay tracker events into get_obs-style text observations.

The replay tracker supplies authoritative unit and resource events, but it does
not preserve each player's fog-of-war. Enemy counts in the generated text are
therefore omniscient replay counts rather than live-game visible counts.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import sc2reader


WORKER_NAMES = {"SCV", "drone", "probe"}
BASE_TYPES = {
    "CommandCenter", "OrbitalCommand", "PlanetaryFortress", "Nexus",
    "Hatchery", "Lair", "Hive",
}
ADDON_TYPES = {
    "BarracksReactor", "BarracksTechLab", "FactoryReactor", "FactoryTechLab",
    "StarportReactor", "StarportTechLab",
}
RACE_NAMES = {"Terr": "Terran", "Terran": "Terran", "Prot": "Protoss", "Protoss": "Protoss", "Zerg": "Zerg"}
IGNORED_UNIT_PREFIXES = ("Beacon", "Mineral", "Vespene", "Destructible", "XelNaga", "Critter")
IGNORED_UPGRADE_PREFIXES = ("RewardDance", "Spray")


def normalize_type(unit_type: str) -> str:
    """Match the normalization used by LLMPlayer.regular_unit_name()."""
    name = str(unit_type or "Unknown")
    lower = name.lower()
    if name == "SCV":
        return "SCV"
    if "flying" in lower:
        lower = lower.replace("flying", "")
    elif "lowered" in lower:
        lower = lower.replace("lowered", "")
    elif "sieged" in lower:
        lower = lower.replace("sieged", "")
    elif "burrowed" in lower:
        lower = lower.replace("burrowed", "")
    elif lower == "liberatorag":
        return "liberator"
    elif lower == "thorap":
        return "thor"
    elif lower in {"vikingassault", "vikingfighter"}:
        return "viking"
    elif "reactor" in lower:
        return f"{lower.replace('reactor', '')} with reactor"
    elif "techlab" in lower:
        return f"{lower.replace('techlab', '')} with techlab"
    return lower


def format_counts(counts: Counter, bases_first: bool) -> str:
    if not counts:
        return "no units currently" if not bases_first else "no structure currently"
    parts: List[str] = []
    if bases_first:
        for name, count in sorted(counts.items()):
            if name in {"commandcenter", "orbitalcommand", "planetaryfortress", "nexus", "hatchery", "lair", "hive"}:
                parts.append(f"{count} bases in the form of {name}")
        for name, count in sorted(counts.items()):
            if name not in {"commandcenter", "orbitalcommand", "planetaryfortress", "nexus", "hatchery", "lair", "hive"}:
                parts.append(f"{count} {name}")
    else:
        parts = [f"{count} {name}" for name, count in sorted(counts.items())]
    return ", ".join(parts)


def game_time(seconds: int) -> str:
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def event_type(event: Any) -> str:
    return type(event).__name__


def is_trackable_unit(unit_type: str) -> bool:
    return bool(unit_type) and not str(unit_type).startswith(IGNORED_UNIT_PREFIXES)


def building_types(events: Iterable[Any]) -> Set[str]:
    """Tracker construction starts are UnitInit events; include base/add-on morphs."""
    result = set(BASE_TYPES) | set(ADDON_TYPES)
    result.update(
        str(event.unit_type_name)
        for event in events
        if event_type(event) == "UnitInitEvent" and is_trackable_unit(getattr(event, "unit_type_name", ""))
    )
    return result


def unit_counts(units: Dict[int, Dict[str, Any]], pid: int, structures: bool, structure_types: Set[str]) -> Counter:
    counts: Counter = Counter()
    for unit in units.values():
        if unit["pid"] != pid:
            continue
        is_structure = unit["type"] in structure_types or unit["type"] in BASE_TYPES or unit["type"] in ADDON_TYPES
        if is_structure == structures:
            counts[normalize_type(unit["type"])] += 1
    return counts


def text_obs(
    pid: int,
    players: Dict[int, Dict[str, str]],
    units: Dict[int, Dict[str, Any]],
    stats: Dict[int, Dict[str, Any]],
    upgrades: Dict[int, List[str]],
    structure_types: Set[str],
    seconds: int,
    map_size: str,
) -> str:
    own_units = unit_counts(units, pid, structures=False, structure_types=structure_types)
    own_structures = unit_counts(units, pid, structures=True, structure_types=structure_types)
    enemy_units: Counter = Counter()
    enemy_structures: Counter = Counter()
    for other_pid in players:
        if other_pid == pid:
            continue
        enemy_units.update(unit_counts(units, other_pid, structures=False, structure_types=structure_types))
        enemy_structures.update(unit_counts(units, other_pid, structures=True, structure_types=structure_types))

    snapshot = stats.get(pid, {})
    food_used = int(round(float(snapshot.get("food_used", 0))))
    food_made = int(round(float(snapshot.get("food_made", 0))))
    worker_supply = sum(own_units.get(worker, 0) for worker in WORKER_NAMES)
    army_supply = max(0, food_used - worker_supply)
    supply_left = max(0, food_made - food_used)
    race = players[pid]["race"]
    research = ", ".join(upgrades.get(pid, [])) or "no research currently"

    return (
        f"It is now {game_time(seconds)} in the game, and we are playing as {race} with "
        f"{int(snapshot.get('minerals_current', 0))} minerals and {int(snapshot.get('vespene_current', 0))} vespene gas, "
        f"while our current supply is {food_used}, of which {army_supply} is allocated to the army, "
        f"{worker_supply} to workers, and {supply_left} remains unused on a {map_size} sized map. \n"
        f"In terms of our military forces, we have {format_counts(own_units, bases_first=False)}. "
        f"In terms of the structures, we have {format_counts(own_structures, bases_first=True)}. \n"
        f"In terms of research, we have researched {research}.\n"
        f"As for the visible enemy presence, there are {format_counts(enemy_units, bases_first=False)}. "
        f"As for the visible enemy structures, there are {format_counts(enemy_structures, bases_first=True)}."
    )


def load_players(replay: Any) -> Dict[int, Dict[str, str]]:
    return {
        int(player.pid): {
            "race": RACE_NAMES.get(str(player.play_race), str(player.play_race)),
            "result": str(player.result) if player.result else "Undecided",
            "name": str(player.name),
        }
        for player in replay.players
    }


def write_replay(replay_path: Path, output_dir: Path, interval_seconds: int, player_ids: Set[int]) -> List[Path]:
    replay = sc2reader.load_replay(str(replay_path), load_map=False, load_level=4)
    players = load_players(replay)
    selected_players = sorted(player_ids or set(players))
    unknown = set(selected_players) - set(players)
    if unknown:
        raise ValueError(f"{replay_path.name}: unknown player ids {sorted(unknown)}")

    events = sorted(replay.tracker_events, key=lambda event: event.frame)
    structure_types = building_types(events)
    units: Dict[int, Dict[str, Any]] = {}
    stats: Dict[int, Dict[str, Any]] = {}
    upgrades: Dict[int, List[str]] = defaultdict(list)
    outputs: Dict[int, List[Dict[str, Any]]] = {pid: [] for pid in selected_players}
    max_frame = max((event.frame for event in events), default=0)
    first_stats_frame = min(
        (event.frame for event in events if event_type(event) == "PlayerStatsEvent"),
        default=0,
    )
    sample_frames = range(first_stats_frame, max_frame + 1, max(1, interval_seconds) * 16)
    event_index = 0

    for sample_frame in sample_frames:
        while event_index < len(events) and events[event_index].frame <= sample_frame:
            event = events[event_index]
            kind = event_type(event)
            unit_id = getattr(event, "unit_id", None)
            if kind in {"UnitBornEvent", "UnitInitEvent"} and unit_id is not None and is_trackable_unit(event.unit_type_name):
                pid = int(getattr(event, "upkeep_pid", 0) or getattr(event, "control_pid", 0) or 0)
                if pid in players:
                    units[unit_id] = {"pid": pid, "type": str(event.unit_type_name)}
            elif kind == "UnitTypeChangeEvent" and unit_id in units:
                if is_trackable_unit(event.unit_type_name):
                    units[unit_id]["type"] = str(event.unit_type_name)
                else:
                    units.pop(unit_id, None)
            elif kind == "UnitDiedEvent" and unit_id is not None:
                units.pop(unit_id, None)
            elif kind == "PlayerStatsEvent":
                stats[int(event.pid)] = dict(event.__dict__)
            elif kind == "UpgradeCompleteEvent" and int(event.pid) in players:
                upgrade = str(event.upgrade_type_name)
                if not upgrade.startswith(IGNORED_UPGRADE_PREFIXES) and upgrade not in upgrades[int(event.pid)]:
                    upgrades[int(event.pid)].append(upgrade)
            event_index += 1

        seconds = sample_frame // 16
        for pid in selected_players:
            outputs[pid].append({
                "replay": replay_path.name,
                "player_id": pid,
                "player_name": players[pid]["name"],
                "race": players[pid]["race"],
                "result": players[pid]["result"],
                "frame": sample_frame,
                "seconds": seconds,
                "text_obs": text_obs(pid, players, units, stats, upgrades, structure_types, seconds, "128x128"),
                "observation_source": "replay tracker events; enemy counts are omniscient, not fog-of-war visibility",
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for pid, records in outputs.items():
        path = output_dir / f"{replay_path.stem}.p{pid}.text_obs.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        paths.append(path)
    return paths


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replays", nargs="*", type=Path, help="Replay files; defaults to data/replays/*.SC2Replay")
    parser.add_argument("--output-dir", type=Path, default=project_root / "data/replays/text_obs")
    parser.add_argument("--interval-seconds", type=int, default=10)
    parser.add_argument("--players", type=int, nargs="*", default=[], help="Optional player ids to export")
    args = parser.parse_args()

    replay_paths = args.replays or sorted((project_root / "data/replays").glob("*.SC2Replay"))
    if not replay_paths:
        raise SystemExit("No .SC2Replay files found.")
    for replay_path in replay_paths:
        for output_path in write_replay(replay_path, args.output_dir, args.interval_seconds, set(args.players)):
            print(output_path)


if __name__ == "__main__":
    main()