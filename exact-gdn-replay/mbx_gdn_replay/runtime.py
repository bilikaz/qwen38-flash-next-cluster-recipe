# SPDX-License-Identifier: Apache-2.0
"""Narrow, opt-in integration for the pinned v6 two-GB10 serving recipe.

The original cache layout, allocation, convolution, and align migration remain
authoritative. Only pure speculative native GDN decode uses replay. Intermediate
states that can be consumed by prefix caching are committed before migration.
"""

import os
from dataclasses import dataclass

import torch
from vllm.logger import init_logger

from . import commit, verify

# Use the configured vLLM handler instead of Python's unrelated root logger.
# This preserves the operator's activation evidence without changing log levels.
logger = init_logger("vllm.mbx_gdn_replay")
MAX_REQUESTS = 64
MAX_TOKENS = 384


class LayerReplay:
    def __init__(self, layer):
        self.layer = layer
        # FP32 actual decay, never log decay or lower-precision records.
        self.records = torch.empty(
            (MAX_TOKENS, 24, 257), dtype=torch.float32, device=layer.A_log.device
        )

    def supports(self, metadata):
        return (
            metadata is not None
            and metadata.num_prefills == 0
            and 0 < metadata.num_spec_decodes <= MAX_REQUESTS
            and 0 < metadata.num_actual_tokens <= MAX_TOKENS
            and self.layer.kv_cache[1].dtype == torch.float32
            and self.layer._can_use_fused_gdn_mtp_decode(metadata)
        )

    def run(self, mixed, a, b, gate, output, metadata):
        verify(
            mixed,
            a,
            b,
            self.layer.A_log,
            self.layer.dt_bias,
            metadata.spec_state_indices_tensor,
            metadata.spec_query_start_loc,
            metadata.num_accepted_tokens,
            self.layer.kv_cache[1],
            gate,
            self.layer.norm.weight,
            output,
            self.records,
            metadata.num_spec_decodes,
            self.layer.head_k_dim**-0.5,
            self.layer.layer_norm_epsilon,
        )


@dataclass
class ReplayStep:
    states: list
    records: list
    metadata: list
    width: int


class ExactReplayState:
    def __init__(self, config, model):
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            QwenGatedDeltaNetAttention,
        )

        self.layers = {}
        self.step = None
        self.logged = False
        self.block_size = config.cache_config.block_size
        if os.environ.get("MBX_EXACT_GDN_REPLAY", "0") != "1":
            logger.info("Exact GDN replay not requested; using the original path")
            return

        supported = (
            config.parallel_config.tensor_parallel_size == 2
            and config.parallel_config.pipeline_parallel_size == 1
            and config.cache_config.mamba_cache_mode == "align"
            and not config.cache_config.use_kda_recoverssm
            and config.num_speculative_tokens == 5
            and config.scheduler_config.max_num_seqs == MAX_REQUESTS
            and config.lora_config is None
            and self.block_size == 1632
            and torch.cuda.get_device_capability() == (12, 1)
            and config.model_config.dtype == torch.bfloat16
        )
        targets = [
            x for x in model.modules() if isinstance(x, QwenGatedDeltaNetAttention)
        ]
        supported = (
            supported
            and len(targets) == 36
            and all(
                x.num_k_heads // x.tp_size == 8
                and x.num_v_heads // x.tp_size == 24
                and x.head_k_dim == x.head_v_dim == 128
                and x.norm.activation == "sigmoid"
                and x.enable_fused_gdn_decode
                and x.gdn_decode_kernel == "cuda"
                and x.A_log.dtype == torch.float32
                for x in targets
            )
        )
        if not supported:
            # Selecting an unsupported configuration changes no established
            # feature. The optimization alone is unavailable and clearly logged.
            logger.warning(
                "Exact GDN replay does not support this configuration; "
                "retaining the original GDN implementation"
            )
            return
        for layer in targets:
            replay = LayerReplay(layer)
            self.layers[layer.prefix] = replay
            layer._mbx_exact_replay = replay
        logger.info(
            "Exact GDN replay enabled for %d layers; FP32 records %.3f MiB; "
            "original KV allocation, graphs and prefix policy retained",
            len(self.layers),
            len(self.layers) * MAX_TOKENS * 24 * 257 * 4 / 1024**2,
        )

    def record_step(self, metadata, for_capture):
        # Do not retain dummy/profile metadata or its potentially huge KV cache.
        self.step = None
        if for_capture or not self.layers or not isinstance(metadata, dict):
            return
        states, records, metas = [], [], []
        for name, replay in self.layers.items():
            meta = metadata.get(name)
            if not replay.supports(meta):
                continue
            state = replay.layer.kv_cache[1]
            states.append(state)
            records.append(replay.records)
            metas.append(meta)
        if not states:
            return
        widths = {m.spec_state_indices_tensor.shape[1] for m in metas}
        if len(widths) != 1:
            raise RuntimeError("Inconsistent GDN replay state-index widths")
        self.step = ReplayStep(states, records, metas, next(iter(widths)))
        if not self.logged:
            logger.info(
                "Exact GDN replay selected for %d layers; registered "
                "CUDA commit precedes native align postprocess",
                len(states),
            )
            self.logged = True

    def commit_step(self, sampled, mapping, computed):
        step, self.step = self.step, None
        if step is None or mapping.numel() == 0:
            return
        if isinstance(sampled, int) or computed is None:
            raise RuntimeError(
                "Speculative GDN replay requires sampled counts "
                "and post-step computed-token positions"
            )
        if any(m.num_spec_decodes < mapping.numel() for m in step.metadata):
            raise RuntimeError("GDN replay request metadata no longer matches")
        commit(
            step.states,
            step.records,
            [m.spec_state_indices_tensor for m in step.metadata],
            [m.spec_query_start_loc for m in step.metadata],
            [m.num_accepted_tokens for m in step.metadata],
            sampled,
            mapping,
            computed,
            step.width,
            self.block_size,
        )
