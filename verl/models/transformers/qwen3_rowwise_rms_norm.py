"""Opt-in Qwen3 RMSNorm reduction using the installed FlashAttention rowwise kernel.

Preserve HF 4.55.2's FP32 normalization, input-dtype cast, then weight multiply.
FlashAttention 2.6.3, BSD-3-Clause, flash_attn/ops/triton/layer_norm.py.
"""
from types import MethodType

import torch


def rowwise_forward(self, hidden_states):
    from flash_attn.ops.triton.layer_norm import rms_norm_fn

    normalized = rms_norm_fn(hidden_states.float(),
        torch.ones_like(self.weight, dtype=torch.float32), None, eps=self.variance_epsilon)
    return self.weight * normalized.to(hidden_states.dtype)


def install_qwen3_rowwise_rms_norm(model):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

    count = 0
    for module in model.modules():
        if isinstance(module, Qwen3RMSNorm):
            module.forward = MethodType(rowwise_forward, module)
            count += 1
    if count == 0:
        raise ValueError("Expected native Qwen3RMSNorm modules")
    return count
