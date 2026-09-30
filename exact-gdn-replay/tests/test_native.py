# SPDX-License-Identifier: Apache-2.0
"""Differential contract tests against the pinned, installed original kernel.

No checkpoint weights or private serving activations are needed. Run these on
an idle GB10 with the derived image. Mutated cache slots are checked bitwise,
including source/final/boundary aliasing and untouched slots.
"""

import gc
import itertools

import pytest
import torch
import vllm._custom_ops  # noqa: F401 - register the original native operator

from mbx_gdn_replay import commit, verify

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def clone(x):
    result = torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)
    return result.copy_(x)


def exact(x, y):
    integer = torch.int32 if x.dtype == torch.float32 else torch.int16
    return torch.equal(x.view(integer), y.view(integer))


def fixture(
    n,
    length,
    ragged=False,
    amplitude=1.0,
    dt_dtype=torch.bfloat16,
    norm_dtype=torch.bfloat16,
):
    lens = (
        [0 if i % 7 == 0 else max(0, length - i % 3) for i in range(n)]
        if ragged
        else [length] * n
    )
    tokens = sum(lens)
    packed = torch.randn(tokens, 8200, device="cuda", dtype=torch.bfloat16) * 0.5
    mixed = packed[:, 4:5124]
    gate = packed[:, 5124:8196].view(tokens, 24, 128)
    ba = torch.randn(tokens, 56, device="cuda", dtype=torch.bfloat16) * 0.1
    a, b = ba[:, 28:52], ba[:, 4:28]
    a_log = torch.randn(24, device="cuda") * 0.1
    dt = torch.randn(24, device="cuda", dtype=dt_dtype) * 0.2
    norm = torch.randn(128, device="cuda", dtype=norm_dtype) * 0.1 + 1
    slots = 1 + n * length
    raw = torch.randn(slots, 24 * 128 * 128 + 4, device="cuda") * (0.1 * amplitude)
    state = raw[:, : 24 * 128 * 128].view(slots, 24, 128, 128)
    indices = (torch.randperm(n * length, device="cuda", dtype=torch.int32) + 1).view(
        n, length
    )
    seqs = torch.tensor(
        [0, *itertools.accumulate(lens)], device="cuda", dtype=torch.int32
    )
    previous = torch.tensor(
        [i % length + 1 for i in range(n)], device="cuda", dtype=torch.int32
    )
    if ragged:
        previous[::5] = 0
        indices[1::5, :] = 0
    output = torch.empty(tokens, 24, 128, device="cuda", dtype=torch.bfloat16)
    records = torch.empty(tokens, 24, 257, device="cuda")
    args = [mixed, a, b, a_log, dt, indices, seqs, previous, state, gate, norm, output]
    return args, records, lens


def commit_inputs(bundles):
    return (
        [a[8] for a, _, _ in bundles],
        [records for _, records, _ in bundles],
        [a[5] for a, _, _ in bundles],
        [a[6] for a, _, _ in bundles],
        [a[7] for a, _, _ in bundles],
    )


def original(args):
    torch.ops._C.fused_gdn_decode_post_conv_mtp(*args, 128**-0.5, 1e-6, "sigmoid")


def replay(args, records, n):
    verify(*args, records, n, 128**-0.5, 1e-6)


@pytest.mark.parametrize(
    "n,length,layers,ragged",
    [
        (1, 2, 1, False),
        (1, 6, 1, False),
        (1, 8, 1, False),
        (4, 6, 3, False),
        (4, 8, 1, True),
        (8, 2, 1, False),
        (8, 6, 3, False),
        (8, 6, 3, True),
        (16, 6, 1, True),
        (64, 6, 1, False),
        (64, 6, 1, True),
    ],
)
def test_outputs_and_all_committed_slots(n, length, layers, ragged):
    torch.manual_seed(968)
    bundles = [fixture(n, length, ragged) for _ in range(layers)]
    initials = [clone(a[8]) for a, _, _ in bundles]
    references = []
    for args, _, _ in bundles:
        ref = list(args)
        ref[8], ref[11] = clone(args[8]), torch.empty_like(args[11])
        original(ref)
        references.append(ref)
    inputs = commit_inputs(bundles)
    mapping = torch.randperm(n + 4, device="cuda", dtype=torch.int64)[:n].contiguous()
    if n > 1:
        mapping[-1] = -1
    sampled = torch.ones(n, device="cuda", dtype=torch.int32)
    computed = torch.zeros(n + 4, device="cuda", dtype=torch.int32)
    states = [a[8] for a, _, _ in bundles]
    # Verification must never mutate the source checkpoint.
    for (args, records, _), initial, ref in zip(bundles, initials, references):
        replay(args, records, n)
        assert exact(args[8], initial)
        assert exact(args[11], ref[11])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for args, records, _ in bundles:
            replay(args, records, n)
        commit(*inputs, sampled, mapping, computed, length, 1632)
    slot_map = mapping.cpu().tolist()
    for step in range(length + 1):
        lens = bundles[0][2]
        counts = [min((i + step) % (length + 1), ln) for i, ln in enumerate(lens)]
        sampled.copy_(torch.tensor(counts, device="cuda", dtype=torch.int32))
        positions = [3264 + [0, 1, 2, 3, 5, 10, 1631][(i + step) % 7] for i in range(n)]
        for i, slot in enumerate(slot_map):
            if slot >= 0:
                computed[slot] = positions[i]
        for (args, _, _), initial in zip(bundles, initials):
            args[8].copy_(initial)
        graph.replay()
        torch.cuda.synchronize()
        for (args, _, _), initial, ref in zip(bundles, initials, references):
            expected = clone(initial)
            indices, previous = args[5].cpu().tolist(), args[7].cpu().tolist()
            for i, count in enumerate(counts):
                if slot_map[i] < 0 or count <= 0 or previous[i] <= 0:
                    continue
                if indices[i][previous[i] - 1] <= 0:
                    continue
                boundary = count - positions[i] % 1632
                if 0 < boundary < count and indices[i][boundary - 1] > 0:
                    expected[indices[i][boundary - 1]].copy_(
                        ref[8][indices[i][boundary - 1]]
                    )
                if indices[i][count - 1] > 0:
                    expected[indices[i][count - 1]].copy_(ref[8][indices[i][count - 1]])
            assert exact(args[11], ref[11])
            assert exact(args[8], expected)
            del expected
    del graph, bundles, initials, references, states, inputs
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("dt_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("norm_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", ["zero", "tiny", "near_tiny", "large", "inf", "nan"])
def test_rounding_edges(dt_dtype, norm_dtype, mode):
    torch.manual_seed(971)
    args, records, lens = fixture(2, 6, dt_dtype=dt_dtype, norm_dtype=norm_dtype)
    if mode == "zero":
        args[8].zero_()
        args[8][::2].neg_()
    elif mode == "tiny":
        args[8].mul_(1e-42)
    elif mode == "near_tiny":
        args[8].mul_(1e-37)
        args[0].mul_(1e-20)
    elif mode == "large":
        args[8].mul_(1e25)
        args[0].mul_(1e10)
    elif mode == "inf":
        args[8][:, ::3, ::7, ::13] = float("inf")
        args[8][:, 1::3, 1::7, 1::13] = -float("inf")
    else:
        args[8].view(torch.int32)[:, ::3, ::7, ::13] = 0x7FC01234
    initial = clone(args[8])
    ref = list(args)
    ref[8], ref[11] = clone(args[8]), torch.empty_like(args[11])
    original(ref)
    replay(args, records, 2)
    assert exact(args[11], ref[11]) and exact(args[8], initial)
    inputs = commit_inputs([(args, records, lens)])
    mapping = torch.arange(2, device="cuda", dtype=torch.int64)
    for count in range(1, 7):
        args[8].copy_(initial)
        sampled = torch.full((2,), count, device="cuda", dtype=torch.int32)
        boundary = (count + 1) // 2
        computed = torch.full(
            (2,), 3264 + count - boundary, device="cuda", dtype=torch.int32
        )
        commit(*inputs, sampled, mapping, computed, 6, 1632)
        expected = clone(initial)
        for i in range(2):
            if boundary < count:
                expected[args[5][i, boundary - 1]].copy_(
                    ref[8][args[5][i, boundary - 1]]
                )
            expected[args[5][i, count - 1]].copy_(ref[8][args[5][i, count - 1]])
        assert exact(args[8], expected)


@pytest.mark.parametrize(
    "fault",
    [
        "state_dtype",
        "norm_dtype",
        "indices_dtype",
        "records_shape",
        "output_shape",
        "requests",
    ],
)
def test_rejects_unsupported_tensor_contract(fault):
    args, records, _ = fixture(1, 6)
    n = 1
    if fault == "state_dtype":
        args[8] = args[8].to(torch.bfloat16)
    if fault == "norm_dtype":
        args[10] = args[10].to(torch.float16)
    if fault == "indices_dtype":
        args[5] = args[5].to(torch.int64)
    if fault == "records_shape":
        records = records[:, :, :256].contiguous()
    if fault == "output_shape":
        args[11] = args[11][:, :23].contiguous()
    if fault == "requests":
        n = 65
    with pytest.raises(RuntimeError):
        replay(args, records, n)


def test_operator_registration_contract():
    args, records, lens = fixture(1, 6)
    torch.library.opcheck(verify, (*args, records, 1, 128**-0.5, 1e-6))
    replay(args, records, 1)
    sampled = torch.tensor([3], device="cuda", dtype=torch.int32)
    mapping = torch.tensor([0], device="cuda", dtype=torch.int64)
    computed = torch.tensor([3265], device="cuda", dtype=torch.int32)
    torch.library.opcheck(
        commit,
        (*commit_inputs([(args, records, lens)]), sampled, mapping, computed, 6, 1632),
    )


def test_empty_verification_matches_original_rejection():
    args, records, _ = fixture(1, 6, ragged=True)
    assert args[0].shape[0] == 0
    with pytest.raises(RuntimeError, match="at least one token"):
        original(args)
    with pytest.raises(RuntimeError, match="verification tokens"):
        replay(args, records, 1)


def test_current_cuda_stream():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        args, records, lens = fixture(8, 6)
        ref = list(args)
        ref[8], ref[11] = clone(args[8]), torch.empty_like(args[11])
        original(ref)
        replay(args, records, 8)
        sampled = torch.full((8,), 3, device="cuda", dtype=torch.int32)
        mapping = torch.arange(8, device="cuda", dtype=torch.int64)
        computed = torch.full((8,), 3265, device="cuda", dtype=torch.int32)
        commit(
            *commit_inputs([(args, records, lens)]), sampled, mapping, computed, 6, 1632
        )
    stream.synchronize()
    assert exact(args[11], ref[11])
    for i in range(8):
        for position in [1, 2]:  # Boundary at 2, accepted final at 3.
            slot = args[5][i, position]
            assert exact(args[8][slot], ref[8][slot])
