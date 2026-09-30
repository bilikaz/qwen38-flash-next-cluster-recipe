// SPDX-License-Identifier: Apache-2.0
// Commit accepted states and intermediate boundaries before native migration.
// Each CTA owns disjoint elements; an output slot may alias the source slot.
#include <cuda_runtime.h>

#include <cstdint>

#include "replay.h"

__device__ __forceinline__ float4 load_state4(const float* p) {
  return *reinterpret_cast<const float4*>(p);
}
__device__ __forceinline__ void store_state4(float* p, float4 v) {
  *reinterpret_cast<float4*>(p) = v;
}
__global__ void gdn_commit_group_exact_kernel(
    const __grid_constant__ GdnReplayTable tables, const int* sampled,
    const int64_t* idx_mapping, const int* new_computed, int requests, int HV,
    int width, int block_size) {
  const int request = blockIdx.x, head = blockIdx.y;
  const int layer = blockIdx.z / 16, tile = blockIdx.z % 16;
  const int linear = (tile * blockDim.x + threadIdx.x) * 4;
  if (request >= requests || linear >= 128 * 128) return;
  const int req_slot = idx_mapping[request];
  if (req_slot < 0) return;
  const int64_t* table = tables.rows[layer];
  float* state = reinterpret_cast<float*>(table[0]);
  const float* records = reinterpret_cast<const float*>(table[1]);
  const int* indices = reinterpret_cast<const int*>(table[2]);
  const int* seqs = reinterpret_cast<const int*>(table[3]);
  const int* previous = reinterpret_cast<const int*>(table[4]);
  const int64_t stride = table[5];
  const int n = sampled[request], old = previous[request];
  const int bos = seqs[request], length = seqs[request + 1] - bos;
  if (n <= 0 || n > length || n > width || old <= 0 || old > width ||
      length > 8)
    return;
  const int source = indices[request * width + old - 1];
  if (source <= 0) return;
  const int target = indices[request * width + n - 1];
  const int computed = new_computed[req_slot];
  const int boundary_count = n - (computed % block_size);
  const int v = linear / 128, k = linear % 128;
  const int64_t offset = head * 128 * 128 + linear;
  float4 h = load_state4(state + source * stride + offset);
  for (int t = 0; t < n; ++t) {
    const float* rec =
        records + (static_cast<int64_t>(bos + t) * HV + head) * 257;
    const float delta = rec[v], decay = rec[256];
    h.x = __fmaf_rn(rec[128 + k + 0], delta, __fmul_rn(h.x, decay));
    h.y = __fmaf_rn(rec[128 + k + 1], delta, __fmul_rn(h.y, decay));
    h.z = __fmaf_rn(rec[128 + k + 2], delta, __fmul_rn(h.z, decay));
    h.w = __fmaf_rn(rec[128 + k + 3], delta, __fmul_rn(h.w, decay));
    if (boundary_count == t + 1 && boundary_count < n) {
      const int boundary = indices[request * width + t];
      if (boundary > 0) store_state4(state + boundary * stride + offset, h);
    }
  }
  if (target > 0) store_state4(state + target * stride + offset, h);
}
extern "C" int mbx_gdn_commit_group(const void* tables, const void* sampled,
                                    const void* idx_mapping,
                                    const void* new_computed, int requests,
                                    int layers, int HV, int width,
                                    int block_size,
                                    unsigned long long raw_stream) {
  if (requests < 1 || layers < 1 || HV != 24 || width < 1 || width > 8 ||
      block_size < width)
    return (int)cudaErrorInvalidValue;
  gdn_commit_group_exact_kernel<<<dim3(requests, HV, layers * 16), 256, 0,
                                  reinterpret_cast<cudaStream_t>(raw_stream)>>>(
      *static_cast<const GdnReplayTable*>(tables),
      static_cast<const int*>(sampled),
      static_cast<const int64_t*>(idx_mapping),
      static_cast<const int*>(new_computed), requests, HV, width, block_size);
  return (int)cudaGetLastError();
}
