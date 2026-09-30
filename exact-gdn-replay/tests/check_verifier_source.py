"""Check the verifier against the pinned vLLM kernel source supplied locally.

Usage: python tests/check_verifier_source.py /path/to/fused_gdn_decode_kernel.cu
No network fetch or mutable upstream branch is used by the build.
"""

import hashlib
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "source-manifest.json").read_text())
source = Path(sys.argv[1]).read_text()
assert (
    hashlib.sha256(source.encode()).hexdigest() == manifest["reference_kernel_sha256"]
)
source = source.split(
    "template <typename StateT, int ValueHeadsPerKeyHead, bool SigmoidGate>\nvoid launch_gdn_decode_post_conv_mtp"
)[0]
source = source.replace(
    '#include "../torch_utils.h"\n#include "../../cuda_compat.h"', ""
)
source = source.replace("gdn_decode_post_conv_mtp_kernel(", "gdn_verify_exact_kernel(")
source = source.replace(
    "GdnDecodeStrides strides) {",
    "GdnDecodeStrides strides, float* __restrict__ records) {",
)
start = source.index("      const int destination_slot =")
end = source.index("\n    }\n  }\n  __syncthreads();", start)
source = source[:start] + source[end:]
old = "            shared_beta[t];\n#pragma unroll"
new = "            shared_beta[t];\n        if (lane == 0) {\n          records[(static_cast<int64_t>(bos+t)*HV+value_head)*257+value] = delta;\n        }\n#pragma unroll"
assert source.count(old) == 1
source = source.replace(old, new)
old = "  __syncthreads();\n\n  const int k_base = lane * 4;"
new = "  __syncthreads();\n  if (warp < num_tokens) {\n    float* record = records + (static_cast<int64_t>(bos+warp)*HV+value_head)*257;\n#pragma unroll\n    for (int i=0; i<4; ++i) record[128+lane+i*32] = shared_k[warp][lane+i*32];\n    if (lane==0) record[256] = shared_decay[warp];\n  }\n\n  const int k_base = lane * 4;"
assert source.count(old) == 1
source = source.replace(old, new)
candidate = (root / "csrc/verify.cu").read_text().split("} // namespace")[0]
assert source.strip() == candidate.strip(), (
    "Verifier arithmetic drifted from the pinned transformation"
)
print("PASS: only full-state writes were replaced by actual-decay replay records")
