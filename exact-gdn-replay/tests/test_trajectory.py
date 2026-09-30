# SPDX-License-Identifier: Apache-2.0
"""Reordered recurrent trajectories through the installed native cache migration."""

import itertools

import pytest
import torch
import vllm._custom_ops  # noqa: F401 - register original native operator
from vllm.v1.worker.mamba_utils import (
    get_aligned_state_indices_multi_group_kernel,
    postprocess_mamba_fused_kernel,
    precopy_mamba_align_fused_kernel,
    preprocess_mamba_align_fused_kernel,
)

from mbx_gdn_replay import commit, verify

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def tensor(values, dtype=torch.int32):
    return torch.tensor(values, dtype=dtype, device="cuda")


def equal(x, y):
    return torch.equal(
        x.view(torch.int32 if x.dtype == torch.float32 else torch.int16),
        y.view(torch.int32 if y.dtype == torch.float32 else torch.int16),
    )


def original(args):
    torch.ops._C.fused_gdn_decode_post_conv_mtp(*args, 128**-0.5, 1e-6, "sigmoid")


@pytest.mark.parametrize("n", [1, 8])
@pytest.mark.parametrize("block", [16, 1632])
def test_native_cache_trajectory(n, block):
    torch.manual_seed(969)
    cols = 32 if block == 16 else 12
    slots = 1 + n * cols
    storage = torch.randn(slots, 24, 128, 128, device="cuda") * 0.1
    physical = (torch.randperm(n * cols, device="cuda", dtype=torch.int32) + 1).view(
        n, cols
    )
    table = torch.empty_like(physical)
    table_ptr = tensor([table.data_ptr()], torch.int64)
    computed = tensor([block - (i % 6 + 1) for i in range(n)])
    mapping = torch.arange(n, device="cuda", dtype=torch.int64)
    states = []
    for arm in ["base", "candidate"]:
        state = storage.clone()
        meta = [
            tensor([state.data_ptr()], torch.int64),
            tensor([state.stride(0) * 4], torch.int64),
            tensor([4]),
            tensor([24 * 128 * 128], torch.int64),
            tensor([0]),
            tensor([0]),
            tensor([0]),
            tensor([0], torch.int64),
        ]
        states.append(
            {
                "state": state,
                "meta": meta,
                "state_idx": (computed - 1) // block,
                "accepted": torch.ones(n, device="cuda", dtype=torch.int32),
                "src": tensor([-1] * n),
                "off": tensor([0] * n),
            }
        )
    records = torch.empty(n * 6, 24, 257, device="cuda")
    alog = torch.randn(24, device="cuda") * 0.1
    dt = torch.randn(24, device="cuda", dtype=torch.bfloat16) * 0.2
    norm = torch.randn(128, device="cuda", dtype=torch.bfloat16) * 0.1 + 1
    indices = torch.empty(1, n, 6, device="cuda", dtype=torch.int32)
    for step in range(64):
        mapping.copy_(torch.randperm(n, device="cuda", dtype=torch.int64))
        table.copy_(physical[mapping])
        lengths = [1 + (step + i) % 6 for i in range(n)]
        seq = tensor([0] + list(itertools.accumulate(lengths)))
        m = sum(lengths)
        after = computed[mapping] + tensor(lengths)
        for arm in states:
            preprocess_mamba_align_fused_kernel[(1,)](
                mapping,
                arm["state_idx"],
                computed,
                seq,
                arm["accepted"],
                arm["src"],
                arm["off"],
                n,
                BLOCK_SIZE=256,
                MAMBA_BLOCK_SIZE=block,
            )
            precopy_mamba_align_fused_kernel[(n, 1, 4)](
                arm["state_idx"],
                arm["src"],
                arm["off"],
                table_ptr,
                cols,
                *arm["meta"],
                mapping,
                n,
                COPY_BLOCK_SIZE=1024,
                CONV_STATE_DIM_FIRST=True,
                HAS_IDX_MAPPING=True,
                TEMPORAL_TILES=4,
            )
        assert torch.equal(
            states[0]["state_idx"], states[1]["state_idx"]
        ) and torch.equal(states[0]["accepted"], states[1]["accepted"])
        get_aligned_state_indices_multi_group_kernel[(1,)](
            table_ptr,
            after,
            indices,
            cols,
            1,
            indices.stride(0),
            indices.stride(1),
            indices.stride(2),
            n,
            CACHE_BLOCK_SIZE=block,
            NUM_GROUPS=1,
            BLOCK_GROUPS=1,
            NUM_STATE_SLOTS=6,
            BLOCK_STATE_SLOTS=8,
            BLOCK_ROWS=32,
            num_warps=1,
        )
        packed = torch.randn(m, 8192, device="cuda", dtype=torch.bfloat16) * 0.5
        ba = torch.randn(m, 48, device="cuda", dtype=torch.bfloat16) * 0.1
        args = []
        for arm in states:
            args.append(
                [
                    packed[:, :5120],
                    ba[:, 24:],
                    ba[:, :24],
                    alog,
                    dt,
                    indices[0],
                    seq,
                    arm["accepted"][mapping].contiguous(),
                    arm["state"],
                    packed[:, 5120:].view(m, 24, 128),
                    norm,
                    torch.empty(m, 24, 128, device="cuda", dtype=torch.bfloat16),
                ]
            )
        original(args[0])
        verify(*args[1], records[:m], n, 128**-0.5, 1e-6)
        torch.cuda.synchronize()
        assert equal(args[0][11], args[1][11]), ("trajectory output", n, block, step)
        sampled = tensor([1 + (step * 3 + i) % ln for i, ln in enumerate(lengths)])
        computed.index_add_(0, mapping, sampled)
        commit(
            [args[1][8]],
            [records],
            [args[1][5]],
            [args[1][6]],
            [args[1][7]],
            sampled,
            mapping,
            computed,
            6,
            block,
        )
        for arm in states:
            arm["accepted"][mapping] = sampled
            snapshot = arm["accepted"].clone()
            postprocess_mamba_fused_kernel[(n, 1, 4)](
                snapshot,
                arm["state_idx"],
                None,
                computed,
                None,
                table_ptr,
                cols,
                *arm["meta"],
                arm["accepted"],
                mapping,
                n,
                block_size=block,
                COPY_BLOCK_SIZE=1024,
                CONV_STATE_DIM_FIRST=True,
                HAS_IDX_MAPPING=True,
                PRECOMPUTED_NEW_COMPUTED=True,
                TEMPORAL_TILES=4,
            )
        torch.cuda.synchronize()
        assert torch.equal(states[0]["accepted"], states[1]["accepted"])
        for req in range(n):
            accepted = int(states[0]["accepted"][req])
            running = int(states[0]["state_idx"][req])
            slot = int(physical[req, running + accepted - 1])
            assert equal(states[0]["state"][slot], states[1]["state"][slot]), (
                "running checkpoint",
                n,
                block,
                step,
                req,
            )
            complete = int(computed[req]) // block
            for col in range(complete):
                slot = int(physical[req, col])
                assert equal(states[0]["state"][slot], states[1]["state"][slot]), (
                    "prefix checkpoint",
                    n,
                    block,
                    step,
                    req,
                    col,
                )
