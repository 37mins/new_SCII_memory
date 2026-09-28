#!/usr/bin/env python3
"""Prepare the verified r7 model-code patch without modifying the source snapshot."""
import argparse
import json
import shutil
from pathlib import Path

from verify_assets import REPO, digest, model_files


def patch_training_modes(source: str) -> str:
    # Exact eight-line patch used by the completed SpB r7, not a new optimization.
    for condition in ("q_len <= 64", "q_len // self.num_sparse_partition <= 64"):
        before = f"mode = 'fused_recurrent' if {condition} else self.mode"
        after = f"mode = self.mode if self.training else ('fused_recurrent' if {condition} else self.mode)"
        if source.count(before) != 2:
            raise ValueError(f"Unexpected SpB source: expected two occurrences of {before}")
        source = source.replace(before, after)
    return source


def prepare(source: Path, destination: Path, manifest: Path) -> dict:
    source, destination = source.resolve(), destination.resolve()
    if destination.is_relative_to(source):
        raise ValueError("Prepared model must be outside the original snapshot")
    files = list(model_files(source, manifest))
    # Verify before creating anything. Weight shards are shared read-only by convention.
    for path, expected in files:
        if digest(path) != expected:
            raise ValueError(f"Model snapshot mismatch: {path}")
    destination.mkdir(parents=True, exist_ok=False)
    patched = {}
    for path, _ in files:
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in (".bin", ".safetensors"):
            target.symlink_to(path)
        elif path.name in ("sse.py", "sse_swa.py"):
            target.write_text(patch_training_modes(path.read_text()))
            patched[path.name] = {"source_sha256": digest(path), "runtime_sha256": digest(target)}
        else:
            shutil.copy2(path, target)
    result = {
        "source": str(source), "runtime_model": str(destination),
        "manifest_sha256": digest(manifest), "patched_files": patched,
        "provenance": "20260818-qwen-canonical-plan-rl-v1-spb-generic-l5-full96-s0-r7",
    }
    (destination.parent / "model_preparation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.destination, REPO / "configs/spikingbrain2.sha256"), indent=2))
