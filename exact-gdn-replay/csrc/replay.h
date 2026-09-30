// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

// Copied into CUDA launch parameters, never allocated as a GPU pointer tensor.
// 36 * 6 * 8 = 1728 bytes, within the kernel argument space on the pinned GPU.
struct GdnReplayTable {
  int64_t rows[36][6];
};
