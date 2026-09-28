"""Bounded eight-rank Ray/NCCL startup check; no model, game or optimizer."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import socket

import ray


@ray.remote(num_gpus=1, num_cpus=1)
class Probe:
    def check(self, rank, address):
        import torch
        import torch.distributed as dist

        visible = os.environ['CUDA_VISIBLE_DEVICES']
        properties = torch.cuda.get_device_properties(0)
        dist.init_process_group('nccl', init_method=address, rank=rank, world_size=8,
                                timeout=timedelta(seconds=45))
        try:
            value = torch.tensor(float(rank), device='cuda')
            dist.all_reduce(value)
            return dict(rank=rank, pid=os.getpid(), visible_devices=visible,
                        ray_gpu_ids=ray.get_gpu_ids(), cuda_uuid=str(properties.uuid),
                        current_device=torch.cuda.current_device(), sum=value.item())
        finally:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--ray-tmp', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    ray.init(num_cpus=8, num_gpus=8, include_dashboard=False,
             object_store_memory=1024**3, _temp_dir=args.ray_tmp,
             runtime_env={'worker_process_setup_hook': 'gamma_ab.install'})
    try:
        workers = [Probe.remote() for _ in range(8)]
        rows = ray.get([worker.check.remote(i, f'tcp://127.0.0.1:{port}')
                        for i, worker in enumerate(workers)], timeout=90)
        passed = len({r['cuda_uuid'] for r in rows}) == 8 and all(r['sum'] == 28 for r in rows)
        (output/'results.json').write_text(json.dumps({'passed':passed,'ranks':rows},indent=2)+'\n')
        print(json.dumps({'passed':passed,'ranks':rows}), flush=True)
        if not passed:
            raise RuntimeError('GPU mapping or NCCL sum verification failed')
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
