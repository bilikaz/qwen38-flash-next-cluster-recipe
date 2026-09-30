# Exact native GDN replay for the v6 two-Spark recipe

This optional image build reduces repeated FP32 recurrent-state writes during
speculative **decode**. Prefill retains the existing implementation. It targets
this recipe's two GB10s, hibrid48, TP=2, PP=1, five draft tokens, 64 sequence
capacity, aligned 1632-token blocks, and native sigmoid GDN kernel.

It retains the checkpoint and its quantization, activation/state precision,
sampling, thinking, tool behavior, graph modes, prefix retention policy, and
41 GB KV allocation per rank. It adds approximately **325.3 MiB per GPU** of FP32
replay records. Unsupported configurations log that the optimization is
unavailable and retain the original implementation; they are not retuned.

The default image and settings in the root `recipe.yaml` are unchanged.

## Why it can be exact

The original verifier calculates and writes a full recurrent state after every
draft token. This version keeps the same verification arithmetic but records
the correction delta, normalized key, and **actual rounded decay multiplier**.
After rejection sampling, it reconstructs the accepted state in forward token
order, preserving each original FP32 multiply and FMA round point, including
the native compiler's flush-to-zero behavior. Any intermediate prefix-cache
boundary is written during that same pass, before the original state migration.

Each thread owns its state elements throughout the recurrence, including when
the final or boundary destination aliases the source. Unaccepted speculative
snapshots are not needed by the original postprocessor. Convolution state and
the original cache layout and allocation remain unchanged.

This is a narrow native arithmetic contract, not a claim that algebraically
equivalent floating-point formulas are bitwise equivalent. It must not be
applied to another state precision, Triton backend, activation, or log-decay
record format without separate validation.

## Build

Build on a Linux arm64 Spark using the same pinned base image as the runtime:

```bash
docker build --progress=plain \
  -t qwen38-flash-next-cluster-vllm:exact-gdn-replay-v1 \
  exact-gdn-replay
```

The Dockerfile pins the existing v6 digest. The extension is compiled at image
build time using its Torch/CUDA stack and installed as a wheel. There is no
runtime compilation, `ctypes` loader, external research directory, or downloaded
model modification. The two Python integration edits are SHA-256 guarded;
unexpected sources or applying the patch twice stops the build.

Use the **same built image on both nodes**. For example, transfer a `docker save`
archive with your normal SSH workflow and load it with `docker load`. Compare
`docker image inspect --format '{{.Id}}' <image>` on both nodes afterward. Do not
overwrite the existing v6 tag.

## Validate before switching the service

Run only when the serving pair is idle; these tests use additional GPU memory.
The image includes synthetic differential tests, so checkpoint data is not
needed for kernel testing:

```bash
docker run --rm --gpus all --network none --ipc=host \
  --entrypoint python3 qwen38-flash-next-cluster-vllm:exact-gdn-replay-v1 \
  -m pytest -q /opt/mbx/exact-gdn/tests
```

The tests cover native bitwise outputs and accepted/boundary states, unchanged
slots, rejection counts, request remapping and masked rows, strided inputs and
state padding, CUDA graph replay, subnormal and large values, signed zero,
infinities and NaN payloads, dtype/layout rejection, operator registration,
dispatch guards, profile metadata lifetime, and source-installation guards.

`tests/check_verifier_source.py` additionally checks the exact source
transformation against the pinned vLLM CUDA file listed in `source-manifest.json`.
The verifier's changes are limited to replacing full-state writes with records;
normalization, gates, reductions and output arithmetic remain aligned with that
reference. Preserve the supplied Apache-2.0 attribution when modifying it.

Kernel tests alone do not validate a deployment. Check eight-request mixed and
thinking workloads, cold and warm long contexts, cancellation/mixed batches,
real prefix hits and boundary continuations, 262144 total context, 64 sequence
capacity, and live effective configuration after switching. Retain raw prompt
identities and input/output counts when comparing throughput.

The included `benchmark.py` reproduces the fixed mixed/code/thinking request
protocol using only Python's standard library. For example, on an exclusively
reserved server:

```bash
python exact-gdn-replay/benchmark.py \
  --label original-r0 --out /tmp/original-r0 --concurrency 8 --requests 32 \
  --max-tokens 1024 --repeat 24 --case install-c8-mixed-r0
```

Run the candidate with the same arguments except `--label` and `--out`. Use
`--thinking` for reasoning, or `--workload code` for the code prompt. Three
distinct `--case` values provide repeat runs. The `--repeat` counts 202, 815,
3270 and 6500 create approximately 8K, 32K, 128K and 260K-token prompts; record
the actual reported token counts rather than treating these names as exact.
For a warm phase, repeat the exact same case and requests with `--warmups 0`
and a new output directory. Compare cache-hit counters across both arms.
The default fixed output budget uses `ignore_eos` to equalize work; it is not a
server limit or a quality score.

## Select the image and keep a rollback

First record and back up the current `recipe.yaml`, `cluster.env`, image ID,
container configuration, and any local patches. Wait for running and waiting
request counts to reach zero. Retain the old image and stopped original
containers under timestamped backup names; the root `run.sh` otherwise removes
containers with the standard name.

Change only `server.image` to your tested local image tag, on both nodes. Preserve
all other local recipe values, including locally selected tool parsers and
patches. The derived image enables `MBX_EXACT_GDN_REPLAY=1`; startup logs must show
36 enabled target layers and selection of the registered commit path. Selection
can be logged during startup warmup; that line alone is not proof of an HTTP
workload. Exercise real speculative requests, inspect the effective
KV/context/concurrency configuration and prove a cold-to-warm cache hit. A
healthy process alone is insufficient validation.

To roll back, wait for requests to drain, stop the new pair, restore the saved
recipe image value and original container names, then start the preserved
original pair using the established launch order. Verify health, effective
settings and cache behavior again. Keep model files, caches and the working
image; no deletion is required.

## Evidence and limitations

The packaged build's three matched C8 code runs measured **312.79 → 344.25
aggregate tokens/s on average (+10.06%)**, with approximately 1K input and 1024
output tokens per request. Individual paired gains were 6.67%, 10.61% and 12.97%;
these are repeat measurements, not a confidence interval. Payload hashes and
input/output counts matched, with no prefix hits or preemptions in either arm.
Both Sparks were **underclocked to approximately 2.2 GHz** by the owner; observed
SM clocks were 2184 and 2190 MHz in both arms. These are not stock-clock results.
A separate confirmation after installation on the original production cache
mounts measured 334.46 TPS versus its matched original case at 309.86 TPS
(+7.94%). See [the validation record](VALIDATION.md) for provenance and test scope.

The earlier research prototype measured 316.09 → 339.65 aggregate tokens/s
(+7.45%) in one C8 code pair. Repeated short-code runs at concurrency 64 improved
about 25%; that is not a promise of the same gain at eight requests. First-token
latency was mixed, and the batch-one verification-plus-commit kernel pair was
slower in profiling.

Both original research arms completed nine cold and warm full-context requests
with identical cache-hit counts, no preemptions and all retrievals correct.
Those capacity tests generated only 64 tokens and do not establish long-context
decode throughput. Finite correctness/quality tests do not prove losslessness
for every possible input; the native arithmetic contract and direct state
comparisons are the relevant evidence for the changed operation.

The packaged operator uses explicit tensor arguments and copies a small pointer
table into kernel launch parameters. It does not expose or cache a GPU tensor of
raw pointers. Packaged-build measurements are recorded separately from the
earlier `ctypes` prototype. Three matched C8 mixed runs averaged 228.12 → 243.23
TPS (+6.62%); three thinking runs averaged 178.19 → 194.48 TPS (+9.15%). These
fixed-output runs preserve thinking for the reasoning workload. At 8K–260K input,
matched warm C8 runs improved 5.68–8.18%; cold very-long-prompt totals were
approximately flat because unchanged prefill dominates. Detailed counts, cache
hits and limitations are in [VALIDATION.md](VALIDATION.md) and
[measurements.json](measurements.json).

## Relationship to upstream work

Compact GDN replay is already being developed in
[vLLM #59366](https://github.com/vllm-project/vllm/pull/59366) and
[#58863](https://github.com/vllm-project/vllm/pull/58863). At the inspected heads
`285e578f9e6d955e91abb2f08651db025fb26c72` and
`3388ba1a254d1f308c54f0bdec102866fe38d24b`, those PRs include full-graph support
and ordered launches to handle source/boundary aliasing. Those ideas and fixes
are not claimed as new here.

This recipe addition preserves the pinned native FP32 arithmetic through
actual-decay records and a forward commit. It is a deployment-specific build,
not a replacement for the broader upstream work or a general vLLM backend.

Thanks to the recipe maintainer for assembling and documenting this stack so
carefully. The readable patches, reproducible configuration, preserved
measurements and explanations of the Spark's constraints made it possible to
identify and test this improvement without guessing at the baseline.
