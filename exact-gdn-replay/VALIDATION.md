# Validation record

This record separates native correctness, serving capabilities and measured
performance. The tests use the pinned two-GB10 v6 stack described in the README.

## Build provenance

- Base digest: `sha256:861ac752164e0d723c5eff3f876586c6678c26ad4a516112c48745f6a101ff4d`.
- Packaged image: `sha256:8c689aca101820218f3995e93efa4c52c5dd1302765a13d065498ba9d9b65c44`.
- Installed native library SHA-256: `03cca6dafc1527f20ad8aea6f4c7efc399be51e0e1d7d26a9d11cc06b9a5eb11`.
- Torch 2.13.0+cu130, CUDA 13.0, Linux arm64, GB10 SM 12.1; extension target `sm_120f`.
- Both Sparks were **underclocked to approximately 2.2 GHz**, as configured by
  the owner. Observed SM clocks were 2184 MHz and 2190 MHz in both measured arms.
- Native source and Python integration hashes are recorded in `source-manifest.json`.

The final logging-only rebuild changed `runtime.py`'s logger to inherit the
configured vLLM handler. Its native library is byte-identical to the image used
for the captured-input and sanitizer checks below. No log-level or serving
configuration override was introduced.

## Native and integration checks

The installed suite contains **76 tests**, covering bitwise verifier outputs,
accepted states, boundary states, untouched slots, zero/rejected counts, masked
rows, request remapping, strided inputs and padded state storage, CUDA graph
replay, current-stream behavior, native cache migration trajectories, numerical
edge cases, dtype/layout rejection, runtime dispatch and metadata lifetime.
It also runs `torch.library.opcheck`, including schema, fake and AOT checks, and
tests source hashes/patch anchors against the actual image sources.

All 76 tests passed independently on both devices for the packaged kernels.
The final logging-only image passed all 76 tests again on each rank.

Private captured-input checks compared the packaged operators against both the
original native operator and the saved original serving outputs/states:

| Scope | Count | Result |
|---|---:|---|
| Real snapshots from two ranks and three GDN layers | 36 | All bitwise equal |
| Verified token outputs | 3132 | All bitwise equal |
| Accepted recurrent states | 522 | All bitwise equal |
| Intermediate prefix-boundary states | 12 | All bitwise equal |

These captures include request counts 1, 4, 16 and 64, plus deep generated
boundary crossings. Model weights and captured activations are not distributed.
The included synthetic tests provide a public reproduction path.

NVIDIA Compute Sanitizer, with graph/alias tests at N=1 and N=8:

| Tool | Result |
|---|---|
| memcheck | 0 errors |
| synccheck | 0 errors |
| initcheck, unfiltered | 0 errors |
| racecheck | 0 hazards, 0 errors, 0 warnings |

Do not apply a GDN-only kernel filter to initcheck: it must observe the fixture's
Torch initialization kernels too. The filtered diagnostic reported an
uninitialized synthetic index allocation in the unchanged reference kernel;
the same tests without that filter passed. This is not an error suppression.
Memcheck/synccheck/racecheck used the `kns=gdn_` filter. The exercised test IDs
are `test_outputs_and_all_committed_slots[1-6-1-False]` and
`test_outputs_and_all_committed_slots[8-6-3-True]` in `tests/test_native.py`.

An initial empty-input fixture exposed an API-contract difference: the original
native kernel rejects zero tokens. The wrapper now matches that rejection and
has a separate regression test. No numerical mismatch was observed.

## Serving checks

The first packaged staging run passed thinking, automatic tool calls, five
prompt-logprob probes, a measured peak of 64 running requests, and a total
262144-token request (262080 input plus 64 output). Cold and warm full-context
retrievals both returned the expected value. Cold cache hits were zero; the
warm request reused 259488 tokens. This proves an actual cache hit, not just an
enabled configuration flag.

The final image logged 36 enabled and selected layers on both ranks. Selection
was logged during startup warmup, before HTTP health; it is not a per-request
trace. Real HTTP validation followed. The mixed-prefill/decode and cancellation
suite passed all 17 cases, including three cancelled streams and a fresh
retrieval after slot reuse.

Four generated-prefix cases started at 1628, 1631, 3260 and 3263 input tokens,
then generated 1800 tokens. All follow-ups and fresh-cache controls retrieved
the expected value and had identical answer token sequences. As in the original
baseline, the first follow-up missed and the second hit 1632 or 3264 tokens.
The second hit follows recomputing prefill; it does not prove direct reuse of
the initial generation's speculative state. Native boundary-state exactness
is checked separately by the differential, trajectory and captured-input tests.

Final production validation used the original model/cache mounts on both nodes.
Both live configurations matched the saved originals except the derived image,
enable flag and descriptive/ownership labels; each node's recipe differed only
in its image value. The original containers remained stopped and intact.
All other recorded recipe files, file permissions and eight bundled patch files
per node were preserved. Installed native and Python package hashes matched the
tested public sources. Both ranks enabled and selected all 36 GDN layers.

On this final production pair, thinking/tool checks and five prompt-logprob
probes passed, a peak of 64 running requests was observed, and cold/warm requests
both reached 262144 total tokens with correct retrieval. Cold prefix hits were
zero and warm hits were 259488 tokens. TTFT was 86.283 seconds cold and 1.209
seconds warm. The 41 GB KV allocation per rank and 2,450,356-token KV pool remained
unchanged. These are capability/cache checks, not throughput-gain estimates.


## Repeated code workload at concurrency eight

Each run sent 16 requests, with eight in flight, approximately 1K prompt tokens
and exactly 1024 generated tokens per request. All paired payload hashes and
prompt/output token counts match. Each arm has zero prefix-hit tokens and zero
preemptions in these cases.

| Repeat | Original aggregate TPS | Packaged aggregate TPS | Change |
|---|---:|---:|---:|
| 0 | 309.865 | 350.059 | +12.97% |
| 1 | 310.370 | 343.312 | +10.61% |
| 2 | 318.148 | 339.372 | +6.67% |
| Mean | 312.794 | 344.248 | +10.06% |

Pooling total output tokens over total wall time gives 312.749 → 344.192 TPS
(+10.05%). The aggregate mean divided by eight is approximately 39.10 → 43.03
TPS per stream; this is a capacity share, not a measured latency percentile.
These three repeats are not a confidence interval.

After the final switch to the original production cache mounts, a separate C8
code confirmation measured **334.465 aggregate TPS**, versus 309.865 TPS for its
matched original case (**+7.94%**). Payload hashes and token counts matched, with
zero errors, prefix hits and preemptions. This one confirmation does not replace
the three-run staging mean; the observed run-to-run spread remains relevant.

## Mixed and thinking workloads at concurrency eight

Each mixed run sent 32 requests and each thinking run sent 16 requests, all
with exactly 1024 generated tokens per request. Thinking remained enabled for
the reasoning cases. All paired payload hashes and input/output counts match;
there were no request errors, prefix hits or preemptions in these short cases.

| Workload | Original TPS runs | Packaged TPS runs | Mean TPS change |
|---|---|---|---:|
| Mixed | 229.758 / 225.075 / 229.528 | 244.754 / 242.181 / 242.764 | 228.120 → 243.233, +6.62% |
| Thinking | 176.231 / 178.911 / 179.415 | 192.505 / 194.934 / 196.005 | 178.186 → 194.481, +9.15% |

Paired gains were 5.77–7.60% for mixed requests and 8.96–9.25% for thinking.
Pooled throughput changes were +6.63% and +9.15%, respectively. These remain
small repeat sets, not confidence intervals or guarantees for other workloads.
First-token latency was mixed; this is not a claim of universally faster TTFT.

The original arm ran before the candidate arm, rather than in randomized
alternating order. Both used the same launch configuration, checkpoint and
request protocol. The candidate's staged disk-cache directory was separate;
prefix-cache hits are measured from the live in-memory cache. No competing GPU
builds or tests overlapped timing. Mid-short-workload clocks were approximately
2184/2190 MHz in both arms, with GPU temperatures 66–68 °C. Dynamic batching and
draft acceptance can vary and are part of the end-to-end result.
These are measurements under the owner's approximately 2.2 GHz clock cap;
stock-clock throughput and the effect of changing that cap were not measured.

## Long context at concurrency eight

Each phase sent eight requests with exactly 1024 generated tokens each. Cold
and warm phases reused the same measured payloads in both arms. C8 is offered
client concurrency; cold prefill admission is dynamic and does not imply that
eight requests were decoding simultaneously throughout the cold phase.

| Input size | Phase | Original aggregate TPS | Packaged aggregate TPS | Change | Cache-hit tokens in each arm |
|---|---|---:|---:|---:|---:|
| 8K | cold | 137.647 | 145.199 | +5.49% | 0 |
| 8K | warm | 172.762 | 183.180 | +6.03% | 39,168 |
| 32K | cold | 67.869 | 68.708 | +1.24% | 0 |
| 32K | warm | 183.501 | 193.932 | +5.68% | 248,064 |
| 128K | cold | 21.716 | 21.777 | +0.28% | 0 |
| 128K | warm | 167.410 | 181.108 | +8.18% | 1,031,424 |
| 260K | cold | 11.040 | 11.022 | -0.16% | 0 |
| 260K | warm | 156.821 | 168.220 | +7.27% | 2,062,848 |

Actual input token ranges: 8K: 8136–8142; 32K: 32657–32663; 128K: 130858–130864; 260K: 260058–260064.

All paired payload hashes and input/output counts match. There were zero
request errors and zero preemptions. Each arm has 256 measured requests across
the 17 cases (512 requests total), excluding warmups.

Warm throughput improved 5.68–8.18% across these lengths. Cold 32K/128K/260K
end-to-end throughput was approximately flat (+1.24%, +0.28%, −0.16%): unchanged
prefill dominates those jobs. The 260K cold runs took 742.055 and 743.217 seconds.
These single matched long-context pairs do not establish small differences as
repeatable effects. The change does not accelerate prefill.

[Machine-readable measurements](measurements.json) include wall times, token
counts, TTFT/TPOT statistics, cache hits, draft acceptance and exact-output
sequence counts. Whole-server output identity is separate from native-state
bitwise correctness.

## Interpretation

Exactness is a narrow native FP32 arithmetic contract supported by direct
bitwise comparisons; finite tests cannot prove every possible input. Whole
distributed-server text generation can vary even in the unchanged baseline.
Fixed-output throughput benchmarks use request-local `ignore_eos` and are not
quality tests or production output limits. Cold admission, prefill, prefix reuse,
draft acceptance and decoding must be considered separately when reading TPS.
