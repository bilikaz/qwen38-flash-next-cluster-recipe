// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <torch/library.h>

#include <cmath>

#include "replay.h"

extern "C" int mbx_gdn_verify(const void*, const void*, const void*,
                              const void*, const void*, const void*,
                              const void*, const void*, void*, const void*,
                              const void*, void*, void*, int, int, int, int,
                              int, int, float, float, long long, long long,
                              long long, long long, long long,
                              unsigned long long);
extern "C" int mbx_gdn_commit_group(const void*, const void*, const void*,
                                    const void*, int, int, int, int, int,
                                    unsigned long long);

namespace {
using at::Tensor;

void device(const Tensor& x, const Tensor& reference) {
  TORCH_CHECK(x.is_cuda() && x.device() == reference.device(),
              "All replay tensors must be on the same CUDA device");
}

void contiguous(const Tensor& x, at::ScalarType dtype) {
  TORCH_CHECK(x.scalar_type() == dtype && x.is_contiguous(),
              "Unexpected replay dtype or noncontiguous tensor");
}

void state_layout(const Tensor& state) {
  TORCH_CHECK(state.scalar_type() == at::kFloat && state.dim() == 4 &&
                  state.size(1) == 24 && state.size(2) == 128 &&
                  state.size(3) == 128 && state.stride(1) == 16384 &&
                  state.stride(2) == 128 && state.stride(3) == 1 &&
                  state.stride(0) >= 393216 && state.stride(0) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(state.data_ptr()) % 16 == 0,
              "Replay requires aligned FP32 [slots,24,128,128] state");
}

void matrix(const Tensor& x, int64_t rows, int64_t columns) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 &&
                  x.size(0) >= rows && x.size(1) == columns &&
                  x.stride(1) == 1 && x.stride(0) >= columns,
              "Unexpected BF16 replay matrix layout");
}

void checked_launch(int code) {
  TORCH_CHECK(code == cudaSuccess, "Exact GDN replay launch failed: ",
              cudaGetErrorString(static_cast<cudaError_t>(code)));
}

void verify(const Tensor& mixed, const Tensor& a, const Tensor& b,
            const Tensor& a_log, const Tensor& dt, const Tensor& indices,
            const Tensor& seqs, const Tensor& previous, const Tensor& state,
            const Tensor& gate, const Tensor& norm, Tensor output,
            Tensor records, int64_t requests, double scale, double epsilon) {
  TORCH_CHECK(mixed.is_cuda(), "Replay requires CUDA");
  const c10::cuda::CUDAGuard guard(mixed.device());
  for (const auto& x : {a, b, a_log, dt, indices, seqs, previous, state, gate,
                        norm, output, records})
    device(x, mixed);
  TORCH_CHECK(mixed.dim() == 2 && requests > 0 && requests <= 64 &&
                  mixed.size(0) > 0 && mixed.size(0) <= 384,
              "Replay supports 1..64 requests and 1..384 verification tokens");
  const auto tokens = mixed.size(0);
  matrix(mixed, tokens, 5120);
  matrix(a, tokens, 24);
  matrix(b, tokens, 24);
  state_layout(state);
  contiguous(a_log, at::kFloat);
  TORCH_CHECK(a_log.numel() == 24 && dt.is_contiguous() && dt.numel() == 24,
              "Expected 24 decay parameters");
  TORCH_CHECK(dt.scalar_type() == at::kFloat || dt.scalar_type() == at::kHalf ||
                  dt.scalar_type() == at::kBFloat16,
              "Unsupported dt_bias dtype");
  TORCH_CHECK(norm.is_contiguous() && norm.numel() == 128 &&
                  (norm.scalar_type() == at::kFloat ||
                   norm.scalar_type() == at::kBFloat16),
              "Unsupported norm weight");
  contiguous(indices, at::kInt);
  contiguous(seqs, at::kInt);
  contiguous(previous, at::kInt);
  TORCH_CHECK(indices.dim() == 2 && indices.size(0) >= requests &&
                  indices.size(1) > 0 && indices.size(1) <= 8 &&
                  seqs.dim() == 1 && seqs.numel() >= requests + 1 &&
                  previous.dim() == 1 && previous.numel() >= requests,
              "Invalid replay metadata shape");
  TORCH_CHECK(gate.scalar_type() == at::kBFloat16 && gate.dim() == 3 &&
                  gate.size(0) >= tokens && gate.size(1) == 24 &&
                  gate.size(2) == 128 && gate.stride(0) >= 3072 &&
                  gate.stride(1) == 128 && gate.stride(2) == 1,
              "Expected row-strided BF16 [tokens,24,128] output gate");
  contiguous(output, at::kBFloat16);
  TORCH_CHECK(output.dim() == 3 && output.size(0) >= tokens &&
                  output.size(1) == 24 && output.size(2) == 128,
              "Invalid output shape");
  contiguous(records, at::kFloat);
  TORCH_CHECK(records.dim() == 3 && records.size(0) >= tokens &&
                  records.size(1) == 24 && records.size(2) == 257,
              "Expected FP32 [tokens,24,257] actual-decay records");
  TORCH_CHECK(std::isfinite(scale) && scale > 0 && std::isfinite(epsilon) &&
                  epsilon > 0,
              "Invalid normalization");
  const int dt_type = dt.scalar_type() == at::kFloat      ? 0
                      : dt.scalar_type() == at::kBFloat16 ? 1
                                                          : 2;
  checked_launch(mbx_gdn_verify(
      mixed.data_ptr(), a.data_ptr(), b.data_ptr(), a_log.data_ptr(),
      dt.data_ptr(), indices.data_ptr(), seqs.data_ptr(), previous.data_ptr(),
      state.data_ptr(), gate.data_ptr(), norm.data_ptr(), output.data_ptr(),
      records.data_ptr(), requests, 8, 24, indices.size(1), dt_type,
      norm.scalar_type() == at::kBFloat16, scale, epsilon, mixed.stride(0),
      a.stride(0), b.stride(0), gate.stride(0), state.stride(0),
      reinterpret_cast<uintptr_t>(c10::cuda::getCurrentCUDAStream().stream())));
}

// Every device allocation is an explicit tensor input. The small host table is
// copied by value into kernel arguments, avoiding a cached device pointer
// tensor whose contents would become stale when a PyTorch subsystem clones the
// inputs.
void commit(at::TensorList states, at::TensorList records,
            at::TensorList indices, at::TensorList seqs,
            at::TensorList previous, const Tensor& sampled,
            const Tensor& mapping, const Tensor& computed, int64_t width,
            int64_t block_size) {
  TORCH_CHECK(!states.empty() && states.size() <= 36 && states[0].is_cuda(),
              "Replay commit needs 1..36 CUDA state tensors");
  const c10::cuda::CUDAGuard guard(states[0].device());
  TORCH_CHECK(
      records.size() == states.size() && indices.size() == states.size() &&
          seqs.size() == states.size() && previous.size() == states.size(),
      "Replay layer lists have different lengths");
  GdnReplayTable table{};
  for (size_t i = 0; i < states.size(); ++i) {
    for (const auto& x :
         {states[i], records[i], indices[i], seqs[i], previous[i]})
      device(x, states[0]);
    state_layout(states[i]);
    contiguous(records[i], at::kFloat);
    contiguous(indices[i], at::kInt);
    contiguous(seqs[i], at::kInt);
    contiguous(previous[i], at::kInt);
    TORCH_CHECK(records[i].dim() == 3 && records[i].size(1) == 24 &&
                    records[i].size(2) == 257 && indices[i].dim() == 2 &&
                    indices[i].size(0) >= mapping.numel() &&
                    indices[i].size(1) == width && seqs[i].dim() == 1 &&
                    seqs[i].numel() >= mapping.numel() + 1 &&
                    previous[i].dim() == 1 &&
                    previous[i].numel() >= mapping.numel(),
                "Invalid per-layer replay inputs");
    table.rows[i][0] = reinterpret_cast<int64_t>(states[i].data_ptr());
    table.rows[i][1] = reinterpret_cast<int64_t>(records[i].data_ptr());
    table.rows[i][2] = reinterpret_cast<int64_t>(indices[i].data_ptr());
    table.rows[i][3] = reinterpret_cast<int64_t>(seqs[i].data_ptr());
    table.rows[i][4] = reinterpret_cast<int64_t>(previous[i].data_ptr());
    table.rows[i][5] = states[i].stride(0);
  }
  for (const auto& x : {sampled, mapping, computed}) device(x, states[0]);
  contiguous(sampled, at::kInt);
  contiguous(mapping, at::kLong);
  contiguous(computed, at::kInt);
  TORCH_CHECK(mapping.dim() == 1 && mapping.numel() > 0 &&
                  mapping.numel() <= 64 && sampled.dim() == 1 &&
                  sampled.numel() >= mapping.numel() && computed.dim() == 1 &&
                  computed.numel() > 0 && width > 0 && width <= 8 &&
                  block_size >= width,
              "Invalid commit metadata");
  checked_launch(mbx_gdn_commit_group(
      &table, sampled.data_ptr(), mapping.data_ptr(), computed.data_ptr(),
      mapping.numel(), states.size(), 24, width, block_size,
      reinterpret_cast<uintptr_t>(c10::cuda::getCurrentCUDAStream().stream())));
}
}  // namespace

TORCH_LIBRARY(mbx_gdn_replay, m) {
  m.def(
      "verify(Tensor mixed, Tensor a, Tensor b, Tensor a_log, Tensor dt, "
      "Tensor indices, Tensor seqs, Tensor previous, Tensor state, "
      "Tensor gate, Tensor norm, Tensor(a!) output, Tensor(b!) records, "
      "int requests, float scale, float epsilon) -> ()");
  m.def(
      "commit(Tensor(a!)[] states, Tensor[] records, Tensor[] indices, "
      "Tensor[] seqs, Tensor[] previous, Tensor sampled, "
      "Tensor mapping, Tensor computed, int width, int block_size) -> ()");
}
TORCH_LIBRARY_IMPL(mbx_gdn_replay, CUDA, m) {
  m.impl("verify", &verify);
  m.impl("commit", &commit);
}
