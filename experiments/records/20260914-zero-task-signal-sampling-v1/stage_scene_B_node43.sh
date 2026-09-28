#!/usr/bin/env bash
# Approved September15: B-only retry on47, one2h budget including this staging.
set -euo pipefail
stage=/mnt/data0601/yangjinyue/new_SCII/staging/20260915-scene-B-r2
source_parent=/root/new-scii-execution-feedback-local-20260911/shared/yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs/20260911-execution-failure-feedback-v1-spb2-original-l6-g2-A-rl4-4gpu-s0-r2
test "${SC2_SCENE_B_RETRY_AUTHORIZED:-0}" = 1
timeout --kill-after=1s 5s stat /mnt/data0601/yangjinyue/new_SCII
mkdir -p "$(dirname "${stage}")"
mkdir "${stage}"
started=$(date +%s)
deadline=$((started + 7200))
echo "${started}" > "${stage}/started_unix.txt"
echo "${deadline}" > "${stage}/deadline_unix.txt"
date -u +%FT%TZ > "${stage}/started_at.txt"
finish() {
    rc=$?
    date -u +%FT%TZ > "${stage}/stage-ended_at.txt"
    echo "${rc}" > "${stage}/stage.done"
}
trap finish EXIT
bound() {
    remaining=$((deadline - $(date +%s) - 20))
    test "${remaining}" -gt 0
    timeout --signal=INT --kill-after=15s "${remaining}" "$@"
}
bound python3 - "${source_parent}" "${stage}" <<'PY'
import json,os,sys
from pathlib import Path
p,stage=map(Path,sys.argv[1:])
r=json.loads((p/'.done').read_text()); m=json.loads((p/'manifest.json').read_text())
assert r==json.loads((p/'result.json').read_text())
assert r['status']=='completed' and r['completed_steps']==4 and r['checkpoint_complete']
assert m['source_commit']=='361cb7ae7fd86fe8811d6fcb6946d0241bf93e11' and not m['dirty'] and m['world_size']==4
fs=os.statvfs(stage)
assert fs.f_bavail*fs.f_frsize > 150*1024**3
for rank in range(4):
    for kind,size in [('model',5555629511),('optim',10332463827)]:
        assert (p/f'checkpoints/global_step_4/actor/{kind}_world_size_4_rank_{rank}.pt').stat().st_size==size
PY
destination=${stage}/shared/yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs/$(basename "${source_parent}")
mkdir -p "${destination}/checkpoints"
bound rsync -a --stats "${source_parent}/checkpoints/global_step_4" "${destination}/checkpoints/"
bound rsync -a "${source_parent}/manifest.json" "${source_parent}/.done" "${source_parent}/result.json" "${source_parent}/training_seeds.jsonl" "${destination}/"
bound python3 - "${source_parent}/checkpoints/global_step_4" "${destination}/checkpoints/global_step_4" "${stage}/checkpoint-copy.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
src,dst,record=map(Path,sys.argv[1:])
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()
files=[]
for p in sorted(src.rglob('*')):
    if not p.is_file(): continue
    q=dst/p.relative_to(src)
    digest=sha(p)
    assert q.stat().st_size==p.stat().st_size and sha(q)==digest,str(p)
    files.append(dict(path=str(p.relative_to(src)),bytes=p.stat().st_size,sha256=digest))
with record.open('x') as f: json.dump(dict(source=str(src),destination=str(dst),files=files,verified=True),f,indent=2)
print('A4 copy verified',len(files),sum(p['bytes'] for p in files),flush=True)
PY
bound tar -C /root/new-scii-spb-level5-96-20260908 -cf "${stage}/python.tar" python include
bound sha256sum "${stage}/python.tar" > "${stage}/python.tar.sha256"
mkdir -p "${stage}/shared/yangjinyue/sc2-verl-agent-experiments"
ln -s /mnt/data0601/yangjinyue/sc2-verl-agent-experiments/assets "${stage}/shared/yangjinyue/sc2-verl-agent-experiments/assets"
echo 'Staging verified; no GPU or game started.'
