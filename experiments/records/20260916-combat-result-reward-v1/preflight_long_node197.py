"""CPU-only checks inside the existing node197 image; no CUDA/model/game creation."""
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess as sp
import sys

import torch

root = Path(sys.argv[1])
source = Path(__file__).resolve().parents[3]
record = Path(__file__).resolve().parent
versions = {s: metadata.version(s) for s in ('torch', 'transformers', 'ray', 'burnysc2', 'triton')}
assert versions == dict(torch='2.7.1+cu128', transformers='4.55.2', ray='2.48.0',
                       burnysc2='7.1.3', triton='3.3.1'), versions
assert Path('/usr/include/python3.12/Python.h').is_file()
assert not torch.cuda.is_initialized()
locations = {s: importlib.util.find_spec(s).origin for s in ('fla', 'moba', 'flash_attn', 'sc2')}
subprocess_env = dict(os.environ, DRY_RUN='1', VERIFY_ASSETS='0', PYTHON_BIN=sys.executable,
                     MODEL_PATH='/assets/spb', SC2PATH='/assets/StarCraftII',
                     PARENT_CHECKPOINT=str(root/'runs/unused/checkpoints/global_step_8'))
proof = dict(versions=versions, package_locations=locations, python=sys.version, stages={},
             cuda_initialized=False, model_loaded=False)
for stage in ('initial-eval', 'train8', 'step8-eval', 'train96'):
    output = root/'cpu-preflight'/stage
    env = dict(subprocess_env, RUN_ID='cpu-'+stage, OUTPUT_DIR=str(output))
    sp.run(['bash', str(record/'run_long_stage.sh'), stage], env=env, cwd=source, check=True, timeout=60)
    config = json.loads((output/'effective_config.json').read_text())
    result = sp.run(config['command']+['--cfg', 'job'], env=dict(env, PYTHONPATH=str(source)),
                    cwd=source, text=True, capture_output=True, timeout=120)
    (output/'hydra-cpu.log').write_text(result.stdout+result.stderr)
    proof['stages'][stage] = dict(hydra_exit=result.returncode, overrides=config['overrides'])
    if result.returncode:
        print(result.stderr[-4000:])
        raise SystemExit(result.returncode)
assert not torch.cuda.is_initialized()
with (root/'cpu-preflight.json').open('x') as stream:
    json.dump(proof, stream, indent=2)
print(json.dumps({k:v for k,v in proof.items() if k != 'stages'}, indent=2))
