"""One-off node43 C1 cache cleanup: fixed allowlist, audit first, no process stops."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from datetime import datetime, timezone

MANIFEST_SHA = "d443210112b3a85de4cb82dfa1b9b624e9995c19ef2b4e71753e646057370740"
LEAVES = {"hf", "triton", "torchinductor_root"}


def now():
    return datetime.now(timezone.utc).isoformat()


def load_targets(manifest):
    raw = manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA:
        raise ValueError("Not the approved inventory")
    data = json.loads(raw)
    targets = [Path(item["path"]) for item in data["cache"]]
    excluded = set(data["excluded_active_temp_directories"])
    if len(targets) != 84 or len(set(targets)) != 84:
        raise ValueError("Unexpected target count")
    for path in targets:
        if (path.parent.parent != Path("/dev/shm") or path.name not in LEAVES
                or str(path.parent) in excluded):
            raise ValueError(f"Out of scope: {path}")
    return targets, excluded


def snapshot(root):
    """Metadata only; reject links, mounts and non-cache assets, never read contents."""
    if root.resolve() != root:
        raise ValueError(f"Symlink target: {root}")
    root_stat = root.lstat()
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError(f"Not a directory: {root}")
    rows = []
    pending = [root]
    while pending:
        path = pending.pop()
        info = path.lstat()
        if info.st_dev != root_stat.st_dev:
            raise ValueError(f"Mounted subtree: {path}")
        if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
            raise ValueError(f"Not a regular cache file/directory: {path}")
        if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
            raise ValueError(f"Hard-linked file: {path}")
        if (path.suffix.lower() in {".safetensors", ".sc2replay", ".log"}
                or path.name.startswith(("model_world_size_", "optim_world_size_"))
                or path.name in {"secrets", "checkpoints", ".env"}):
            raise ValueError(f"Unexpected asset: {path}")
        rows.append({"path": str(path.relative_to(root)), "device": info.st_dev,
                     "inode": info.st_ino, "mode": info.st_mode, "size": info.st_size,
                     "mtime_ns": info.st_mtime_ns, "blocks": info.st_blocks})
        if stat.S_ISDIR(info.st_mode):
            pending.extend(path.iterdir())
    return sorted(rows, key=lambda row: row["path"])


def references(parents):
    """Report only matching paths/PIDs, never complete process arguments."""
    found = set()
    def check(pid, source, value):
        for parent in parents:
            if parent in value:
                found.add((pid, source, parent))
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            for name in ("cwd", "exe"):
                try:
                    check(int(proc.name), name, os.readlink(proc / name))
                except FileNotFoundError:
                    pass
            for name in ("cmdline", "maps"):
                check(int(proc.name), name, (proc / name).read_text(errors="replace"))
            for fd in (proc / "fd").iterdir():
                try:
                    check(int(proc.name), "fd", os.readlink(fd))
                except FileNotFoundError:
                    pass
        except (FileNotFoundError, ProcessLookupError):
            pass
    return [{"pid": pid, "source": source, "parent": parent}
            for pid, source, parent in sorted(found)]


def free_bytes():
    info = os.statvfs("/dev/shm")
    return info.f_bavail * info.f_frsize


def write_new(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2)


def main():
    mode, manifest_name, audit_name = sys.argv[1:]
    if mode not in {"audit", "apply"}:
        raise ValueError("Expected audit or apply")
    targets, excluded = load_targets(Path(manifest_name))
    parents = {str(path.parent) for path in targets}
    refs = references(parents | excluded)
    blocked = {row["parent"] for row in refs}
    audit_path = Path(audit_name)
    if mode == "audit":
        data = {"node_alias": "sc2-a100-43", "time": now(), "manifest_sha256": MANIFEST_SHA,
                "references": refs, "free_bytes": free_bytes(), "targets": [], "skipped": []}
        for path in targets:
            if str(path.parent) in blocked or not path.exists():
                data["skipped"].append({"path": str(path), "reason": "referenced or absent"})
                continue
            rows = snapshot(path)
            data["targets"].append({"path": str(path), "entries": rows,
                                    "allocated_bytes": sum(row["blocks"] * 512 for row in rows)})
        write_new(audit_path, data)
        print(json.dumps({"audited": len(data["targets"]), "skipped": data["skipped"],
                          "allocated_bytes": sum(t["allocated_bytes"] for t in data["targets"]),
                          "references": refs, "audit": str(audit_path)}))
        return

    data = json.loads(audit_path.read_text())
    if data["manifest_sha256"] != MANIFEST_SHA or not shutil.rmtree.avoids_symlink_attacks:
        raise ValueError("Audit mismatch or unsafe platform")
    audited = [Path(item["path"]) for item in data["targets"]]
    if len(set(audited)) != len(audited) or not set(audited) <= set(targets):
        raise ValueError("Audit contains unapproved targets")
    # Validate the entire selected set before the first removal; recheck each at removal.
    for item in data["targets"]:
        path = Path(item["path"])
        if str(path.parent) not in blocked and snapshot(path) != item["entries"]:
            raise ValueError(f"Changed since audit: {path}")
    outside = {parent: set(os.listdir(parent)) - LEAVES for parent in parents}
    receipt = {"node_alias": "sc2-a100-43", "started_at": now(),
               "manifest_sha256": MANIFEST_SHA, "references_before": refs,
               "free_bytes_before": free_bytes(), "deleted": [], "skipped": [],
               "processes_stopped": 0, "backup": "None; only regenerable C1 caches"}
    journal = audit_path.with_suffix(".deleted.jsonl")
    with journal.open("x") as stream:
        for item in data["targets"]:
            path = Path(item["path"])
            if str(path.parent) in blocked:
                receipt["skipped"].append({"path": str(path), "reason": "active reference"})
                continue
            if snapshot(path) != item["entries"]:
                raise ValueError(f"Changed since audit: {path}")
            shutil.rmtree(path)
            row = {"path": str(path), "allocated_bytes": item["allocated_bytes"],
                   "files": sum(stat.S_ISREG(r["mode"]) for r in item["entries"]), "at": now()}
            receipt["deleted"].append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
    receipt.update({"finished_at": now(), "free_bytes_after": free_bytes(),
                    "references_after": references(parents | excluded),
                    "excluded_directories_preserved": all(Path(p).is_dir() for p in excluded),
                    "non_cache_siblings_preserved": all(
                        set(os.listdir(p)) - LEAVES == names for p, names in outside.items()),
                    "remaining_targets": [str(p) for p in targets if p.exists()]})
    receipt["allocated_bytes_removed"] = sum(row["allocated_bytes"] for row in receipt["deleted"])
    write_new(audit_path.with_suffix(".receipt.json"), receipt)
    print(json.dumps({key: value for key, value in receipt.items()
                      if key not in {"deleted", "references_before", "references_after"}}))


if __name__ == "__main__":
    main()
