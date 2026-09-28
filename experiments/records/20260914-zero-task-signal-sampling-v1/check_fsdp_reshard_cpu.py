"""Tiny CPU-only 4-to-2 FSDP/Adam restore test; never loads project weights.

Uses PyTorch 2.7 FULL_STATE_DICT and scatter_full_optim_state_dict. This is a
deployment compatibility test, not an SC2, GPU-memory, or learning acceptance.
"""
import argparse
import copy
import datetime
import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP, FullStateDictConfig,
    OptimStateKeyType, ShardedStateDictConfig, StateDictType,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.device_mesh import init_device_mesh


def network():
    torch.manual_seed(29)
    return nn.Sequential(nn.Linear(7, 11), nn.Tanh(), nn.Linear(11, 5))


def train_once(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    inputs = torch.arange(28, dtype=torch.float32).reshape(4, 7) / 28
    model(inputs).square().mean().backward()
    if isinstance(model, FSDP):
        model.clip_grad_norm_(1.0)
    else:
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()


def assert_tree_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected), 'State tensor changed during restoration'
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_tree_equal(a, b)
    else:
        assert actual == expected


def full_model(model):
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                             FullStateDictConfig(rank0_only=True)):
        return model.state_dict()


def worker(rank, world_size, root):
    root = Path(root)
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + str(root/f'gloo-{world_size}'),
                            rank=rank, world_size=world_size,
                            timeout=datetime.timedelta(seconds=60))
    try:
        model = FSDP(network(), device_id=torch.device('cpu'), use_orig_params=False,
                     auto_wrap_policy=ModuleWrapPolicy({nn.Linear}),
                     device_mesh=init_device_mesh('cpu', (world_size,)))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
        if world_size == 4:
            for _ in range(7):
                train_once(model, optimizer)
            # Exercise the legacy format: sharded model + native flat Adam state.
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT,
                                     ShardedStateDictConfig(offload_to_cpu=True)):
                path = root/f'legacy-rank-{rank}.pt'
                torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict()}, path)
                saved = torch.load(path, weights_only=False)
                model.load_state_dict(saved['model'])
                optimizer.load_state_dict(saved['optim'])
            full = full_model(model)
            optim = FSDP.full_optim_state_dict(model, optimizer, rank0_only=True)
            if rank == 0:
                torch.save({'model': full, 'optim': optim}, root/'full.pt')
        else:
            saved = torch.load(root/'full.pt', weights_only=False)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                model.load_state_dict(saved['model'])
            sharded = FSDP.scatter_full_optim_state_dict(
                saved['optim'] if rank == 0 else None, model, optim=optimizer)
            optimizer.load_state_dict(sharded)
            restored_model = full_model(model)
            restored_optim = FSDP.full_optim_state_dict(model, optimizer, rank0_only=True)
            if rank == 0:
                assert_tree_equal(restored_model, saved['model'])
                assert_tree_equal(restored_optim, saved['optim'])
            assert {int(s['step']) for s in optimizer.state.values()} == {7}
            reference = network()
            reference.load_state_dict(saved['model'])
            ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-6)
            ref_state = copy.deepcopy(FSDP.rekey_optim_state_dict(
                saved['optim'], OptimStateKeyType.PARAM_ID, reference))
            # Several unflattened parameters can share one flat-parameter step
            # tensor. A plain optimizer needs an independent counter per param.
            for state in ref_state['state'].values():
                for field, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[field] = value.clone()
            ref_optimizer.load_state_dict(ref_state)
            train_once(model, optimizer)
            train_once(reference, ref_optimizer)
            updated_model = full_model(model)
            updated_optim = FSDP.full_optim_state_dict(model, optimizer, rank0_only=True)
            assert {int(s['step']) for s in optimizer.state.values()} == {8}
            if rank == 0:
                for name, value in reference.state_dict().items():
                    torch.testing.assert_close(updated_model[name], value)
                expected = FSDP.rekey_optim_state_dict(
                    ref_optimizer.state_dict(), OptimStateKeyType.PARAM_NAME, reference)
                for name, state in expected['state'].items():
                    for field, value in state.items():
                        torch.testing.assert_close(updated_optim['state'][name][field], value)
                (root/'result.json').write_text(json.dumps({
                    'torch': torch.__version__, 'source_world_size': 4,
                    'target_world_size': 2, 'restore_model_and_adam_exact': True,
                    'adam_step_before': 7, 'adam_step_after': 8,
                    'one_update_matches_unsharded_reference': True,
                    'gpu_initialized': torch.cuda.is_initialized(),
                    'real_A4_or_gpu_acceptance': False,
                }, indent=2) + '\n')
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    mp.spawn(worker, args=(4, str(args.output.resolve())), nprocs=4)
    mp.spawn(worker, args=(2, str(args.output.resolve())), nprocs=2)
    print((args.output/'result.json').read_text(), flush=True)
