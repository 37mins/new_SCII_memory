"""Read-only host samples; the pair controller owns this process's lifetime."""
import json
from pathlib import Path
import subprocess
import sys
import time

control = Path(sys.argv[1])
with (control / "host-memory.jsonl").open("x", buffering=1) as stream:
    while True:
        mem = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            mem[key] = int(value.split()[0])  # kB for memory sizes; hugepage counts unchanged.
        current = control / "current_arm.txt"
        top = subprocess.check_output(
            ["ps", "-eo", "pid,ppid,rss,comm", "--sort=-rss"], text=True).splitlines()[:21]
        row = {"utc": time.strftime("%FT%TZ", time.gmtime()),
               "arm": current.read_text().strip() if current.exists() else "preflight",
               "meminfo": mem, "top_rss_kb": top}
        stream.write(json.dumps(row) + "\n")
        time.sleep(20)
