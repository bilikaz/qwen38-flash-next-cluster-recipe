"""Registered CUDA operators built with the recipe's pinned Torch/CUDA stack."""

from pathlib import Path

import torch

if torch.__version__ != "2.13.0+cu130" or torch.version.cuda != "13.0":
    raise RuntimeError("mbx-gdn-replay requires the pinned v6 Torch 2.13.0+cu130 ABI")

torch.ops.load_library(str(Path(__file__).with_name("_C.so")))

verify = torch.ops.mbx_gdn_replay.verify
commit = torch.ops.mbx_gdn_replay.commit


@torch.library.register_fake("mbx_gdn_replay::verify")
def _verify_fake(
    mixed,
    a,
    b,
    a_log,
    dt,
    indices,
    seqs,
    previous,
    state,
    gate,
    norm,
    output,
    records,
    requests,
    scale,
    epsilon,
):
    # All outputs are preallocated mutations; no result tensor is returned.
    return None


@torch.library.register_fake("mbx_gdn_replay::commit")
def _commit_fake(
    states,
    records,
    indices,
    seqs,
    previous,
    sampled,
    mapping,
    computed,
    width,
    block_size,
):
    return None


__version__ = "0.1.0"
