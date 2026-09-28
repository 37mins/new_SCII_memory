"""Read only ZIP metadata and 4-byte Adam step counters, never large tensors."""
import collections
import hashlib
import io
import json
from pathlib import Path
import pickle
import re
import struct
import zipfile

RUN = Path("/mnt/data0601/yangjinyue/new_SCII/experiments/20260906-integrated-fixes-v1/runs/20260906-integrated-fixes-v1-qwen-instruct2507-l6-8gpu-fp32-rl96-gate8-s0-r4-train8")


def tensor_metadata(storage, offset, size, stride, *unused):
    return {"storage": storage, "offset": offset, "size": size, "stride": stride}


class MetadataOnly(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return tensor_metadata
        if (module, name) == ("torch", "FloatStorage"):
            return "FloatStorage"
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        raise ValueError((module, name))

    def persistent_load(self, pid):
        assert pid[0] == "storage"
        return {"type": pid[1], "key": pid[2], "device": pid[3], "numel": pid[4]}


records = []
for step in (1, 2):
    for rank in range(8):
        path = RUN / f"checkpoints/global_step_{step}/actor/optim_world_size_8_rank_{rank}.pt"
        with zipfile.ZipFile(path) as z:
            entry = next(i for i in z.infolist() if i.filename.endswith("/data.pkl"))
            assert entry.file_size < 10000
            data = z.read(entry)
            meta = MetadataOnly(io.BytesIO(data)).load()
            counters, moments = [], []
            for state in meta["state"].values():
                st = state["step"]["storage"]
                assert st["type"] == "FloatStorage" and st["numel"] == 1
                name = entry.filename.removesuffix("data.pkl") + "data/" + st["key"]
                assert z.getinfo(name).file_size == 4
                counters.append(struct.unpack("<f", z.read(name))[0])
                moments.append({k: state[k] for k in ("exp_avg", "exp_avg_sq")})
            records.append({"global_step": step, "rank": rank, "path": str(path),
                "pickle_bytes": len(data), "pickle_sha256": hashlib.sha256(data).hexdigest(),
                "param_groups": meta["param_groups"], "adam_step_counters": counters,
                "moment_metadata_only": moments})
metrics = []
with (RUN / "console.log").open(errors="replace") as f:
    for line in f:
        if "actor/grad_norm" in line and "actor/lr" in line:
            clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
            metrics.append(clean.strip())
print(json.dumps({"optimizer_metadata": records, "console_metric_lines": metrics,
    "scope": "16 ZIP headers + 622-byte metadata and 4-byte counters each; moment/parameter tensor contents NOT read"}))
