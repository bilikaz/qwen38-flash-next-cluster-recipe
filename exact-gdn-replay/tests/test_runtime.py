# SPDX-License-Identifier: Apache-2.0
"""CPU checks for dispatch, lifetime, and unchanged unsupported configurations."""

import copy
import sys
import weakref
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
import torch

from mbx_gdn_replay import runtime


class Metadata:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeLayer:
    def __init__(self, index):
        self.prefix = f"layer.{index}"
        self.num_k_heads, self.num_v_heads, self.tp_size = 16, 48, 2
        self.head_k_dim = self.head_v_dim = 128
        self.norm = NS(activation="sigmoid")
        self.enable_fused_gdn_decode = True
        self.gdn_decode_kernel = "cuda"
        self.A_log = torch.empty(24)
        self.kv_cache = [None, torch.empty(2, 24, 128, 128)]

    def _can_use_fused_gdn_mtp_decode(self, meta):
        return meta.num_decodes == 0 and meta.num_spec_decodes > 0


def configured(monkeypatch):
    module = ModuleType("fake_gdn_layer")
    module.QwenGatedDeltaNetAttention = FakeLayer
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn", module
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 1))
    monkeypatch.setenv("MBX_EXACT_GDN_REPLAY", "1")

    def small_replay(self, layer):
        self.layer = layer
        self.records = torch.empty(6, 24, 257)

    monkeypatch.setattr(runtime.LayerReplay, "__init__", small_replay)
    layers = [FakeLayer(i) for i in range(36)]
    model = NS(modules=lambda: iter(layers))
    config = NS(
        parallel_config=NS(tensor_parallel_size=2, pipeline_parallel_size=1),
        cache_config=NS(
            block_size=1632, mamba_cache_mode="align", use_kda_recoverssm=False
        ),
        num_speculative_tokens=5,
        scheduler_config=NS(max_num_seqs=64),
        lora_config=None,
        model_config=NS(dtype=torch.bfloat16),
    )
    return config, model, layers


@pytest.mark.parametrize(
    "change",
    [
        "tp",
        "pp",
        "block",
        "mode",
        "recoverssm",
        "drafts",
        "requests",
        "lora",
        "dtype",
        "layer",
    ],
)
def test_unsupported_configuration_keeps_original(monkeypatch, change):
    config, model, layers = configured(monkeypatch)
    if change == "tp":
        config.parallel_config.tensor_parallel_size = 1
    if change == "pp":
        config.parallel_config.pipeline_parallel_size = 2
    if change == "block":
        config.cache_config.block_size = 16
    if change == "mode":
        config.cache_config.mamba_cache_mode = "all"
    if change == "recoverssm":
        config.cache_config.use_kda_recoverssm = True
    if change == "drafts":
        config.num_speculative_tokens = 4
    if change == "requests":
        config.scheduler_config.max_num_seqs = 8
    if change == "lora":
        config.lora_config = "enabled"
    if change == "dtype":
        config.model_config.dtype = torch.float16
    if change == "layer":
        layers[-1].norm.activation = "silu"
    before = copy.deepcopy(config)
    state = runtime.ExactReplayState(config, model)
    assert config == before and not state.layers
    assert not any(hasattr(x, "_mbx_exact_replay") for x in layers)


def test_disabled_is_explicit_and_allocates_nothing(monkeypatch):
    config, model, layers = configured(monkeypatch)
    monkeypatch.setenv("MBX_EXACT_GDN_REPLAY", "0")
    assert not runtime.ExactReplayState(config, model).layers
    assert not any(hasattr(x, "_mbx_exact_replay") for x in layers)


def test_supported_layer_dispatch_and_profile_lifetime(monkeypatch):
    config, model, layers = configured(monkeypatch)
    state = runtime.ExactReplayState(config, model)
    assert len(state.layers) == 36
    meta = Metadata(
        num_prefills=0,
        num_spec_decodes=1,
        num_actual_tokens=6,
        num_decodes=0,
        spec_state_indices_tensor=torch.ones(1, 6, dtype=torch.int32),
        spec_query_start_loc=torch.tensor([0, 6], dtype=torch.int32),
        num_accepted_tokens=torch.ones(1, dtype=torch.int32),
    )
    metadata = {layer.prefix: meta for layer in layers}
    state.record_step(metadata, False)
    assert len(state.step.states) == 36 and len(state.step.metadata) == 36
    for_capture_ref = weakref.ref(meta)
    state.record_step(None, True)
    assert state.step is None
    del meta, metadata
    assert for_capture_ref() is None  # No cached pointers or profile metadata.

    replay = next(iter(state.layers.values()))
    meta = NS(num_prefills=1, num_spec_decodes=1, num_actual_tokens=6, num_decodes=0)
    assert not replay.supports(meta)
    meta.num_prefills, meta.num_decodes = 0, 1
    assert not replay.supports(meta)


def test_commit_uses_step_then_releases_it(monkeypatch):
    config, model, layers = configured(monkeypatch)
    state = runtime.ExactReplayState(config, model)
    meta = NS(
        num_spec_decodes=1,
        spec_state_indices_tensor=torch.ones(1, 6, dtype=torch.int32),
        spec_query_start_loc=torch.tensor([0, 6], dtype=torch.int32),
        num_accepted_tokens=torch.ones(1, dtype=torch.int32),
    )
    state.step = runtime.ReplayStep(
        [layers[0].kv_cache[1]], [torch.empty(1)], [meta], 6
    )
    calls = []
    monkeypatch.setattr(runtime, "commit", lambda *args: calls.append(args))
    sampled = torch.tensor([3], dtype=torch.int32)
    mapping = torch.tensor([5], dtype=torch.int64)
    computed = torch.zeros(64, dtype=torch.int32)
    state.commit_step(sampled, mapping, computed)
    assert state.step is None and len(calls) == 1
    assert calls[0][-2:] == (6, 1632)
    state.commit_step(sampled, mapping, computed)
    assert len(calls) == 1
