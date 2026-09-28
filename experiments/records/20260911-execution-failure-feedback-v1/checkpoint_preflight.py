"""Small metadata checks only; actual distributed model loading is still required."""
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

import torch

actual = tuple(version(name) for name in ("torch", "transformers", "ray", "gymnasium"))
assert actual == ("2.7.1+cu128", "4.55.2", "2.48.0", "1.2.3"), actual
storage_root = Path(os.environ.get("FEEDBACK_STORAGE_ROOT", "/mnt/data0601"))
run = storage_root / "yangjinyue/new_SCII/experiments/20260910-spb-fsdp-layer-wrap-v1/runs/20260910-spb-fsdp-layer-wrap-v1-spb2-l7-g2-opening-quota-resume4to96-4gpu-s0-r1"
manifest = json.loads((run / "manifest.json").read_text())
assert manifest["source_commit"] == "90b3fdc2fb7fadc1230861306b20c27540d9b736"
assert manifest["world_size"] == 4
checkpoint = run / "checkpoints/global_step_24"
state = torch.load(checkpoint / "sc2_state.pt", map_location="cpu", weights_only=False)
assert state["global_step"] == 24
files = {}
for rank in range(4):
    for kind, expected in [("model", 5555629511), ("optim", 10332463827), ("extra_state", None)]:
        path = checkpoint / f"actor/{kind}_world_size_4_rank_{rank}.pt"
        size = path.stat().st_size
        assert size == expected if expected else size > 10000
        files[str(path.relative_to(checkpoint))] = size
assert (checkpoint / "data.pt").is_file()
assert json.loads((checkpoint / "actor/config.json").read_text())["swa_dropout"] == 0
proof = dict(checkpoint=str(checkpoint), source_sha=manifest["source_commit"], files=files,
             dependencies=actual, manifest_sha256=hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest(),
             size_and_metadata_preflight_only=True, actual_distributed_load_still_required=True)
with (Path(sys.argv[1]) / "checkpoint_preflight.json").open("x") as stream:
    json.dump(proof, stream, indent=2)
print("Checkpoint metadata verified: step24; actual load still required", flush=True)
