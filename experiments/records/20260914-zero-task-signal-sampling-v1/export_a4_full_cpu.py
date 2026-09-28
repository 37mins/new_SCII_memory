"""Export existing four-rank A4 to full model/Adam state on CPU, once.

No forward/backward or optimizer update. Original checkpoint stays read-only.
PyTorch 2.7 FSDP APIs perform unflattening; no hand-written optimizer reslicing.
SpB's Triton imports require a CUDA context; no model computation runs on GPU.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP, FullStateDictConfig,
    ShardedStateDictConfig, StateDictType,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.tensor import DTensor
from transformers import AutoConfig, AutoModelForCausalLM


def digest_tensor(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(memoryview(value.numpy()).cast('B')).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    dist.init_process_group('gloo', timeout=datetime.timedelta(minutes=15))
    rank = dist.get_rank()
    try:
        assert dist.get_world_size() == 4
        assert torch.cuda.device_count() == 1
        torch.cuda.set_device(0)  # One explicitly selected, approved import-context GPU.
        if rank == 0:
            args.output.mkdir(exist_ok=False)
        dist.barrier()
        mesh = init_device_mesh('cpu', (4,), mesh_dim_names=('fsdp',))
        actor = args.checkpoint/'actor'
        saved_config = json.loads((actor/'config.json').read_text())
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True,
                                            local_files_only=True)
        config.update(saved_config)
        config._attn_implementation = 'eager'
        with torch.device('meta'):
            module = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, torch_dtype=torch.float32)
        module.to_empty(device='cpu')
        module.tie_weights()
        block_names = set(module._no_split_modules)
        blocks = {type(m) for m in module.modules() if type(m).__name__ in block_names}
        model = FSDP(module, device_id=torch.device('cpu'), use_orig_params=False,
                     auto_wrap_policy=ModuleWrapPolicy(blocks), device_mesh=mesh)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
        assert len(FSDP.fsdp_modules(model)) == 37, 'Expected A4 layer-wrapped layout'
        assert all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in model.parameters())
        print(f'rank={rank} CPU model built; loading A4', flush=True)
        shard = torch.load(actor/f'model_world_size_4_rank_{rank}.pt',
                           map_location='cpu', weights_only=False)
        # Checkpoint's logical CUDA mesh becomes a CPU mesh for conversion;
        # local tensor bytes, global shape/stride and placements stay unchanged.
        for name, value in shard.items():
            if isinstance(value, DTensor):
                shard[name] = DTensor.from_local(value.to_local(), device_mesh=mesh,
                    placements=value.placements, shape=value.shape, stride=value.stride(),
                    run_check=False)
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT,
                                 ShardedStateDictConfig(offload_to_cpu=True)):
            model.load_state_dict(shard)
        del shard
        state = torch.load(actor/f'optim_world_size_4_rank_{rank}.pt',
                           map_location='cpu', weights_only=False)
        optimizer.load_state_dict(state)
        for parameter, values in optimizer.state.items():
            assert values['exp_avg'].shape == parameter.shape
            assert values['exp_avg_sq'].shape == parameter.shape
        steps = sorted({int(s['step']) for s in optimizer.state.values()})
        assert len(steps) == 1 and steps[0] > 0
        print(f'rank={rank} model/Adam loaded; optimizer steps={steps}', flush=True)
        del state
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                 FullStateDictConfig(rank0_only=True)):
            full_model = model.state_dict()
        full_optim = FSDP.full_optim_state_dict(model, optimizer, rank0_only=True)
        if rank == 0:
            hashes = {'model': {k: digest_tensor(v) for k, v in full_model.items()},
                      'optim': {k: {f: digest_tensor(v) for f, v in s.items()
                                    if isinstance(v, torch.Tensor)}
                                for k, s in full_optim['state'].items()}}
            with (args.output/'model.pt').open('xb') as stream:
                torch.save(full_model, stream)
            with (args.output/'optimizer.pt').open('xb') as stream:
                torch.save(full_optim, stream)
            result = {'source_checkpoint': str(args.checkpoint), 'source_world_size': 4,
                      'torch': torch.__version__, 'optimizer_steps': steps,
                      'optimizer_param_groups': [{k:v for k,v in g.items() if k != 'params'}
                                                 for g in full_optim['param_groups']],
                      'tensor_sha256': hashes, 'gpu_context_for_triton_import': True,
                      'model_compute_device': 'cpu', 'model_forward_calls': 0,
                      'optimizer_updates_performed': 0,
                      'note': 'CPU export only; two-GPU loading/update not yet accepted.'}
            with (args.output/'export.json').open('x') as stream:
                json.dump(result, stream, indent=2)
            print('FULL_EXPORT_COMPLETE', flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
