# Qwen3.8-Flash-Next on two DGX Sparks

### Two checkpoints: `hibrid48`, the default — and `hibrid48-uncensored`, made from it (abliterated, gated). Switch with one line in `recipe.yaml`

[`hibrid48`](https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid48) (the base model, default)
and [`hibrid48-uncensored`](https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored) (OrcaRouter's abliterated
body, no refusals, no guardrails — gated, research / private use). Same stack. To switch: in `recipe.yaml`
comment the active `model:` line and uncomment the other, then `./run.sh`. Details in [Which checkpoint](#which-checkpoint).

Two boxes, one model, RDMA. **106 tok/s single-stream (121 peak), 817 tok/s at 64 streams (883 peak), a 2.45M-token KV
pool, ~3,200 tok/s prefill all the way to 256k** — and it boots in about four minutes. Three commands.

## Measured performance (this exact stack, 2× DGX Spark, RDMA, K=5, `vm.compaction_proactiveness=0`)

**v4.1 (2026-09-27, FlashInfer GDN prefill)** — the kit as shipped, one complete run of `bench/full.py`: thinking on
×3, then the ladder on a code prompt (thinking off, 120 s windows), then one request per context size. The ladder uses a
code prompt from here on; the v4 table below used the pasture scene, whose SVG drafts better (acceptance 5.1 vs 4.8), so
the two tables are not comparable row for row — same engine speed, steps/s equal.

| concurrent requests | tok/s | peak | per-stream | acceptance |
|---|---|---|---|---|
| 1 | **99** | 109 | 99 | 4.73 |
| 2 | **159** | 173 | 79 | 4.87 |
| 4 | **233** | 248 | 58 | 4.93 |
| 8 | **342** | 367 | 43 | 4.93 |
| 16 | **458** | 495 | 29 | 4.90 |
| 24 | **547** | 585 | 23 | 4.91 |
| 32 | **627** | 662 | 20 | 4.91 |
| 48 | **721** | 752 | 15 | 4.81 |
| 64 | **793** | 834 | 12 | 4.80 |

Thinking on, one request: **81 tok/s** average, 120 peak.

Long context, one request. Cold = fresh prompt; hot = the same prompt again (prefix cache — an agent's next turn). Decode =
200 tokens right after the prompt: prose = a summary of it, code = a Python module.

| prompt tokens | cold time to first token | hot time to first token | prefill tok/s | decode prose | decode code |
|---|---|---|---|---|---|
| 1k | **0.43 s** | 0.43 s | 2,516 | 75.8 | 65.6 |
| 8k | 2.5 s | 0.63 s | 3,246 | 65.1 | 68.5 |
| 32k | 9.9 s | 0.71 s | 3,302 | 81.2 | 56.9 |
| 64k | 20 s | 0.78 s | 3,291 | 64.7 | 58.0 |
| 128k | 41 s | **1.0 s** | **3,227** | 64.6 | 67.4 |
| 192k | 62 s | 1.2 s | 3,155 | 67.6 | 48.0 |
| 256k | 85 s | 1.2 s | 3,086 | 57.3 | 55.2 |

**v4 ladder (2026-09-24, vLLM 0.30, image v6, `hibrid48`, 41G bf16 pin, 64 seats, Marlin MoE)** — this kit exactly as shipped,
same prompt and the same windows as the tables below, steady-state averages of 3–6 runs per rung. Thinking off.

| concurrent requests | **PEAK tok/s** (v3 → v4) | average tok/s | per-stream (v3 → v4) | acceptance |
|---|---|---|---|---|
| 1 | 107 → **121** | 106 | 92 → **106** | 5.05 |
| 2 | 160 → **180** | 173 | 75 → **86** | 5.14 |
| 4 | 237 → **265** | 252 | 58 → **63** | 5.10 |
| 8 | 349 → **388** | 364 | 42 → **46** | 5.13 |
| 16 | 474 → **508** | 492 | 28 → **31** | 5.11 |
| 24 | 559 → **614** | 575 | 22 → **24** | 5.14 |
| 32 | 589 → **708** | 652 | 18 → **20** | 5.15 |
| 48 | 674\* → **795** | 738 | 13.2\* → **15** | 5.14 |
| 64 | — → **883** | 817 | — → **13** | 5.10 |

\* v3 has no 48-stream row; the value is v2's (hibrid47, vendor pin, 28G bf16 pin). No earlier kit seated 64.

Reading it: **every rung is 8–16 % faster than v3.** Each engine step now drafts five tokens instead of four and keeps
5.1 of a possible 6 instead of 4.2 of 5 — the drafter's tokens are sampled from its own distribution and verified against the model's (see [What changed](#what-changed)), which
is what moved acceptance. Single-stream, v3's **peak** of 107 is now v4's **average**. The top rung is new: 817 tok/s at 64
streams, 883 at peak. With thinking on (a full request: reasoning, then the answer), one stream averages ~81 tok/s and the
code phase peaks at 121 (measured on the same image and flags before this ladder; v3: 79.5 over the same kind of request).
The pool is the limit at the top: 64 streams of this test filled it to 99 % after about three minutes, after which new
tokens wait for a request to finish. 48 streams peaked at 81 %.


**v3 ladder (2026-09-13, hibrid48 on vLLM 0.29, 28G bf16 pin, Marlin MoE)** — kept as the reference v4 is measured against; same prompt, same windows, same rules as the v2 table below (rungs 2–32 measured on the v3 kit boot; c=48 pending).

| concurrent requests | **PEAK tok/s** | average tok/s | per-stream | engine steps/s (v2 → v3) | acceptance |
|---|---|---|---|---|---|
| 1 | **107** | 92 | 92 | 17.7 → **22.1** | 4.3 |
| 2 | **160** | 150 | 75 | 15.1 → **18.2** | 4.15 |
| 4 | **237** | 232 | 58 | 11.8 → **13.7** | 4.22 |
| 8 | **349** | 338 | 42 | 8.8 → **10.1** | 4.17 |
| 16 | **474** | 454 | 28 | 6.2 → **6.8** | 4.16 |
| 24 | **559** | 529 | 22 | 4.9 → **5.2** | 4.20 |
| 32 | **589** | 563 | 18 | 4.0 → **4.2** | 4.18 |
| 48 | — | — | — | 3.2 → — | — |

Reading it: **two streams on v3 each get what one stream got on v2** — 75 tok/s per stream at c=2 against v2's 73 at c=1;
further down the gain narrows (v3 at c=4 = 58 per stream, v2 at c=2 = 63). The 4-bit head is a fixed
per-step saving, so it shows most where the step is cheapest — +25 % engine steps at c=1, +21 % at c=2, +15 % at c=8,
+5 % at c=32, where the routed experts dominate the step. Acceptance is unchanged (4.15–4.22 at every rung).

**v2 ladder (2026-09-06, hibrid47 on the vendor pin, 25G bf16 pin)** — kept as the reference the v3 numbers are measured against:

| concurrent requests | **PEAK tok/s** | average tok/s | per-stream | engine steps/s (v1 → v2) | acceptance |
|---|---|---|---|---|---|
| 1 | **80** | 73 | 73 | 16.4 → **17.7** | 4.1 |
| 2 | **133** | 126 | 63 | 13.8 → **15.1** | 4.15 |
| 4 | **209** | 198 | 50 | 10.7 → **11.8** | 4.2 |
| 8 | **309** | 294 | 37 | 8.0 → **8.8** | 4.16 |
| 16 | **451** | 417 | 26 | 5.8 → **6.2** | 4.2 |
| 24 | **514** | 488 | 20 | 4.4 → **4.9** | 4.18 |
| 32 | **579** | 533 | 17 | 3.7 → **4.0** | 4.19 |
| 48 | **674** | 635 | 13.2 | 3.0 → **3.2** | 4.18 |

Reading it: the old averages became the new floors — v1 averaged 68 tok/s single-stream, v2's seven runs never went
below 69. Thinking enabled at 32 streams: 320–340 tok/s (acceptance 2.5 on reasoning prose; the engine speed is the
same, the text decides how many tokens each step yields). Per-position draft acceptance on prose, single stream:
0.91 / 0.85 / 0.80 / 0.71. **Quality:** 32 boss-animals renders at 32 streams, thinking on — 26 good, 3 partial,
3 broken; v1 scored about half/half on the same scenes. Full 262,144-token context; the 28G-per-box KV pool holds
2.85M pooled tokens in fp8 (1.71M bf16). Each running request pins ~1.9 % of the pool at admission — the model's GDN
recurrent state, 36 layers × (2 + K) state blocks, which fp8 KV does not touch — so ~52 short requests is the hard
ceiling either way; what fp8 changes is how much context each seat can hold. The kit seats 32: ~89k tokens of pool per
seat and 17 tok/s per stream (rungs 1–32 were measured at a 25G pin, c=48 at this kit's 28G, all bf16). Numbers carry
their conditions on purpose — the tools that produced them (`bench/test.py`, `bench/summary.py`, `bench/accept.py`
in the myllmbox repo) are yours to rerun.

## Quality (measured on this serve, thinking on)

[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) run against the running two-Spark serve of
hibrid48 and hibrid48-uncensored (2026-09-13), **thinking on** — the mode the model is served in — at temperature 0.6, top-p 0.95, top-k 20 (the model
card recommends 1.0 / 0.95 / 20 for thinking; a card-faithful pass at 1.0 is a separate run), a 32k-token budget per answer, 16
requests in flight. Qwen publishes no numbers for these four tests — its card reports LiveCodeBench v6 91.9, GPQA Diamond 91.7,
IFBench 81.3, SWE-bench Pro 62.5 — so there is no official row to compare against; GPQA Diamond is the one overlap still to run. HumanEval is complete; the
other tasks use a fixed 200-question subset (seed 123123123), the same questions on every checkpoint we compare.

| task | questions | hibrid48 | hibrid48-uncensored |
|---|---|---|---|
| HumanEval pass@1 | 164 | **95.7** | **94.5** |
| GSM8K exact match | 200 | **98.0** | **97.5** |
| IFEval prompt-level strict / instruction-level strict | 200 | **91.5** / 93.4 | **94.5** / 96.2 |
| MMLU-Pro (14 subjects, sampled by size) | 200 | **84.9** | **82.9** |
| answers that ran into the 32k budget while thinking (count as wrong) | 763 | 11 | 6 |

Same 763 questions on both checkpoints, 91 and 75 minutes. The IFEval gain is the one difference larger than the two subsets'
sampling error; the other deltas are one to four questions each. The abliterated body also thinks shorter (median reasoning
−8 %, 90th percentile −27 %) and runs away half as often. The runaway rate is itself a number to compare between checkpoints. Subsets of 200 carry about ±3 points of sampling noise;
published leaderboard numbers use other prompts, few-shot counts and full sets, so treat them as a sanity band, not a column.
The runner (`bench/quality/` in the myllmbox repo: harness driver, answer extraction that reads a chat model's final answer,
and a side-by-side table tool) works against any OpenAI-compatible endpoint — rerun it and count. hibrid47 on the same
questions is the pending A/B; until then these are the model's numbers, not a measured cost of the 4-bit head.

**v4 check (vLLM 0.30, 2026-09-23):** GSM8K rerun on the v4 image, hibrid48-uncensored, 400 questions (seed 123123123),
thinking on, same sampling, 32 in flight — **97.75 % and 97.25 %** on two runs of the same questions, 0 answers truncated.
The table above is the v3 serve's; the v4 engine change did not move this one.

## What changed

**v4.1 (2026-09-27): FlashInfer GDN prefill.** `gdn-prefill-backend` back to 0.30's default: **+5 % prefill from 8k to
256k tokens** (3,300 vs 3,140 tok/s at 32k), the decode ladder unchanged at every rung from 1 to 64 streams. Measured as
two complete runs of the same kit, one per backend. Thanks to sethforprivacy for the finding.

**v4 (2026-09-24): vLLM 0.30, five draft tokens, a 2.45M-token pool and 64 seats.** The engine moved to upstream vLLM
0.30, and speculative decoding was retuned on it:

1. **Five draft tokens, sampled.** The drafter now proposes 5 tokens instead of 4 and *samples* them from its own
   distribution (`draft_sample_method: probabilistic`) instead of always taking its top guess; the model checks them with
   the exact probability-ratio test and block verification. Output quality cannot change — the text follows the model's
   own distribution either way — but more drafts survive: 5.1 accepted per step on code (v3: 4.2 of 5). +15 % single-stream,
   +16 % at 32 streams. Measured one knob at a time: the sampled draft is the gain; block verification is neutral on top.
2. **NCCL on four channels** (`NCCL_MAX_NCHANNELS=4`). NCCL builds 64 channels on this pair of boxes and splits every large
   message across all of them; the GB10 has no GPUDirect RDMA, so each piece is copied through host memory. Four channels
   made the per-layer all-reduce 3× faster at 32–64 streams: +10 % at 32 streams, +7 % at 64, nothing at one.
3. **Half the n-gram table per box.** On 0.30 splitting it across the two boxes costs nothing measurable (it did on 0.29),
   and the 13.4G it frees per box goes to the KV pool: 28G → 41G per box, 1.71M → **2,450,356 pooled tokens**, 9.35× the full
   262K window. 64 seats instead of 32.
4. **Boots in about four minutes instead of about seventeen:** 0.30 serves this model without torch.compile, so the
   per-boot model compilation is gone.

The image also carries the fused multi-step draft for this model's attention (proposed upstream as vllm-project/vllm#58449;
+1–3 % engine steps at 1–12 streams), and the kit reads each box's RoCE GID index at launch — the index moves after a
reboot or link flap, and a stale one fails NCCL with "unhandled system error". fp8 KV is not in the v4 image. v3 stays
available: `git checkout v3` (image `…-cluster-vllm:v5`, vLLM 0.29, full table per box, 28G pin, 32 seats, fp8 KV opt-in).

**v3 (2026-09-13): vLLM 0.29 and a 4-bit output head.** Two changes, one number: the serve moved from the vendor's SM121 vLLM
pin to upstream vLLM 0.29 (the model is upstream now), and the checkpoint's 1.18 GiB bf16 output head — read ~5.4 times per
decode step by speculative decoding, 27 % of the step — became 0.33 GiB of NVFP4 (`hibrid48`). Engine steps **17.7 → 22.0**
single-stream; same body, same table, same drafter. v2's single-stream **peak** was 80 tok/s — on v3, 80 is the **average** of a
whole thinking-on request, reasoning included (79.5 over 38k tokens), and the code phase peaks at 107. KV is bf16 by default (1.71M pooled tokens on
the 28G pin): fp8 KV still works (2.85M) but cost 0.3 accepted tokens per step on this stack, so it is one commented line
away, not the default. Quality, measured with lm-evaluation-harness against this serve with thinking on: HumanEval 95.7,
GSM8K 98.0, IFEval 91.5, MMLU-Pro 84.9. v2.1 stays available: `git checkout v2.1` (image `…-cluster-vllm:v4`, hibrid47, fp8 KV).

**v2.1 (2026-09-08): fp8 KV.** Same model, same speed, 1.66× the KV pool: **2.85M pooled tokens** on the same 28G pin
(bf16 held 1.71M), so every seat carries more context and one request can run the full 262K window 10.9 times over.
Upstream vLLM's fp8-KV port for this model's QSA attention (PR #54846) is image patch 04; the engine still steps at
17.3/s single-stream (74 tok/s writing code, 54 thinking, 39,487 tokens in 11 minutes in one request). v2 stays
available: `git checkout v2` (image `…-cluster-vllm:v3`, bf16 KV).

**v2 (2026-09-06)** serves [myllmbox/Qwen3.8-Flash-Next-hibrid47](https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid47):
the hibrid46 body with its 95 GB n-gram (PLE) table re-quantized to NVFP4 and held **resident on the GPU** — no CPU
offload worker, no per-step detour — split tensor-parallel across **two NVIDIA DGX Sparks (GB10, 119G unified memory
each)** over their ConnectX link, NCCL on RDMA. Against v1 (the int3 table in a CPU worker, same boxes, same tests):
**+7–11 % engine steps on every concurrency** and a table that draws 26 of 32 boss scenes where int3 drew half.
v1 stays available: `git checkout v1` in this repo (image `…-cluster-vllm:v2`, checkpoint hibrid46).

## Quick start

```bash
git clone https://github.com/myllmbox/qwen38-flash-next-cluster-recipe.git
cd qwen38-flash-next-cluster-recipe
./run.sh        # first run: sets the cluster up (asks for the 2nd box), downloads ~99G, syncs it, serves on :8000
```

**Hugging Face token.** Anonymous downloads are rate-limited, and gated models (license-agreement repos, e.g. uncensored variants) refuse anonymous access. `run.sh` looks for `HF_TOKEN`, then `~/.cache/huggingface/token` (`hf auth login`), and asks for one when the repo is gated — after you accepted its agreement on the model page. Nothing is stored by the kit.

`./stop.sh` stops both boxes. `./view.sh` shows live stats plus the RDMA proof. Requirements: two DGX Sparks
with docker + the NVIDIA container runtime, connected by their ConnectX ports (a direct cable or a switch),
ssh from the head to the worker (a password once — `setup.sh` installs a key). With the weights on both boxes, a boot
reaches healthy in about four minutes (weights ~97 s, profile + KV + graph capture ~67 s).

**What `run.sh` does the first time:** no `cluster.env` yet → it runs [`setup.sh`](setup.sh), which asks for the
worker's ssh address, probes both boxes (interfaces, RDMA devices, GPU, docker), **discovers the interconnect**
(the interface pair that actually reaches the other box, tried by bound pings — never the management LAN),
checks the firewall **without root** (a throwaway listener on one box, one connect from the other, over the
interconnect — if it passes, nothing to open and no password is ever asked; if a box blocks its peer, it shows the
single `ufw allow from <peer>` it would run and asks first), creates the model/cache dirs on the worker, and writes
`cluster.env`. Rerun `./setup.sh` after re-cabling.

**All model configuration lives in [`recipe.yaml`](recipe.yaml)** — image, weights repo, port, KV budget, every
vLLM flag. The cluster flags (`--nnodes 2`, ranks, rendezvous address, TP=2, `--headless` on the worker) and the
per-box NCCL/gloo interface pins are added by `run.sh` from `cluster.env`; you never write them.

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": "hello"}]
}'
```

## Which checkpoint

Two checkpoints run on this exact stack; `recipe.yaml` ships with the first active and the second commented out under it.
Switching is comment one line, uncomment the other, `./run.sh`:

| `model:` | what it is | speed on this kit |
|---|---|---|
| `myllmbox/Qwen3.8-Flash-Next-hibrid48` (default) | the base model, calibrated body, NVFP4 output head — the checkpoint the v3 and v4 ladders were measured on | v4: 106 tok/s at c=1, **817** tok/s at 64 streams |
| `myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored` | OrcaRouter's abliterated (refusal-removed) body with the same head — **no guardrails**; research, red-teaming, private use behind your own moderation | same head, same stack, same speed; quality table above (IFEval 94.5, HumanEval 94.5, GSM8K 97.5, MMLU-Pro 82.9) |

The uncensored repo is **gated**: open its Hugging Face page, accept the agreement, then `hf auth login` (or `export
HF_TOKEN=…`) before `./run.sh` — the kit checks both and tells you what is missing. Running both checkpoints at different
times? Set `served-model-name` to something distinct (e.g. `Qwen/Qwen3.8-Flash-Next-Uncensored`) so clients and logs can tell
them apart. Both weigh ~99 GB; the first download of the second one is a full download (different body), the 8 table shards are
shared bytes. hibrid47 / hibrid47-uncensored (bf16 head) load on this image too, at v2.1 speed.

## The RDMA part (why the numbers are what they are)

NCCL will happily run over TCP sockets on the same ConnectX cable and never say so — the env vars look right, the
serve works, and every step is ~2× slower. A container needs three things to actually open the RDMA device:
`--device /dev/infiniband`, `--cap-add IPC_LOCK`, `--ulimit memlock=-1:-1`. `run.sh` passes them; `view.sh`
proves it by sampling the HCA's port counter against the interface's TCP byte counter while the model decodes
(RDMA moving, TCP flat = good). If `/dev/infiniband` is missing on a box (`rdma-core` not installed, or the link
is not a ConnectX one), the kit still runs, over TCP, and says so.

## Memory on a Spark: what the kit does about it

Unified memory means the GPU driver and the page cache share one pool, and the driver wants pages that are
**free**, not just reclaimable. After a few model loads the checkpoint's shards sit in the page cache and free
memory drops to ~1 GB while the next load allocates — the driver can stall on a copy that never completes, and the
boot looks hung at 100 % CPU. The kit never asks for your password; it stabilises memory with what a user may do:

- **waits** after removing old containers until both boxes report ≥ 100 GB available (unified memory takes
  30–60 s to come back after a container dies; launching earlier gives a phantom "CUDA out of memory"),
- **evicts its own checkpoint files from the page cache** before launch (`dd iflag=nocache`, no privileges),
- the weights load with `fastsafetensors` (`load-format` in `recipe.yaml`), which booted cleanly on every v4 launch; with
  vLLM's default loader instead, the image **drops each shard from the cache as soon as it has been consumed**, so the
  cache never balloons during the load itself.

Nothing to do on your side.

One thing you *can* do, with root, and it is worth ~10 % on a serve that runs this close to the memory edge:

```
./tune-host.sh      # sets vm.compaction_proactiveness=0 on both boxes; shows the two commands, asks, then sudo prompts
```

`run.sh` checks the value on both boxes before every launch (reading needs no privilege) and prints a one-line
warning while it is not 0; it never applies it for you.

The kernel's background page compactor wakes on a low-free-memory box and migrates pages to build large
contiguous blocks. On a Spark the GPU's memory *is* those pages, so every migration first unmaps them from the
GPU: measured as a 4–5 s slowdown every ~37 s (the compactor's retry cycle), both GPUs idling together, no swap,
no clock change. A serving box allocates once at boot and gains nothing from the upkeep. The kit never runs
this for you (it needs root); it takes effect immediately, no restart.

## Tuning (recipe.yaml)

- **`kv-cache-memory`** (bytes, per box): 41G default with half the table on each box (~51G of weights per box) =
  2,450,356 pooled tokens. Leaves ~6G of host headroom on the head box after graph capture — check `free -g` on both
  boxes after the first boot and back off if either shows swap in use. Do **not** take vLLM's "fully utilize"
  suggestion: unified memory over-commit has needed a power cycle.
- **`MBX_PLE_REPLICATE: "0"`** (env): half the n-gram table per box, exchanged per gather — on vLLM 0.30 this measured the
  same speed as the full table per box at 1, 16 and 32 streams, and it is what pays for the 41G pin. `"1"` puts the full
  table on each box (14G more per box); then drop `kv-cache-memory` back to 28G.
- **`compilation-config`**: the model runner rounds every FULL-graph capture size up to a multiple of K+1 and drops the
  ones past the largest listed, so the list must reach `max-num-seqs × 6` = 384. Shorten it only together with
  `max-num-seqs`; a list that stops short leaves the upper rungs decoding without CUDA graphs.
- **`block-size: 1632`**: required by K=5. This model's attention keeps a small ring of recent keys sized to hold the
  draft tokens (8 slots at K=4, 12 at K=5), and the ring must divide the attention block; vLLM's automatic block (1616)
  does not divide by 12 and the boot stops with "QSA ring capacity 12 must divide the attention block size 1616". Remove
  the line if you go back to K=4.
- **`NCCL_MAX_NCHANNELS: "4"`** (env): see [What changed](#what-changed). More channels only add host copies here; 2 and 8
  measured the same as 4, 1 was slower at 32 streams.
- **`engram-config: {"cpu_offload": false}`**: vLLM 0.30 would otherwise keep the n-gram table in pinned host memory;
  this image serves it from the GPU, the layout every number here was measured on.
- **`gdn-prefill-backend: flashinfer`**: +5 % prefill over `triton`, same decode speed.
- **`load-format: fastsafetensors`**: the loader every v4 boot was measured with. Remove it for vLLM's default loader.
- **`gpu-memory-utilization`** 0.70: with the pin set it does not size the KV pool.
- **`max-num-seqs`**: 64 — ~38k tokens of pool per seat, 13 tok/s per stream, 817 tok/s aggregate. Each running request
  also pins part of the pool the moment it is admitted, regardless of length: the GDN recurrent state, 36 layers ×
  (2 + K) blocks, held for rollback of rejected draft tokens (7 blocks at K=5). At 64 streams of this test the pool filled
  to 99 % after about three minutes; at 48 it peaked at 81 %. Set 48 if your load is many long answers at once; a smaller
  K = more seats (2 + K).
- **fp8 KV** is not in the v4 image (the patch has not been re-ported to 0.30 — upstream PR #54846 is still open);
  `git checkout v3` for it.
- **`speculative-config`** K=5, `draft_sample_method: probabilistic`, `rejection_sample_method: block`: acceptance ~5.1
  on code, ~3.9 over a whole thinking-on request, cap 6.0. Measured one change at a time against K=5 with the defaults:
  the sampled draft +7 % single-stream (thinking off), +1.5 % thinking on, +9 % at 32 streams; block verification neutral
  alone and on top. K=4 (`{"method":"mtp","num_speculative_tokens":4}`, capture list to 320, no `block-size`) is the
  previous setting. A bf16 drafter was A/B'd and rejected (acceptance +0.01, −3 % steps, +3.4G).
- **`max-num-batched-tokens`**: also the image-input encoder budget — 8192 fits realistic multi-image requests.
- **`host: 127.0.0.1`**: the cluster runs on host networking, so the API would otherwise be on every interface.
  Set `0.0.0.0` to expose it on the LAN.
- Thinking is ON by default (model native); disable per request with
  `"chat_template_kwargs": {"enable_thinking": false}` for max speed on structured output.

## What's in the image

`myllmbox/qwen38-flash-next-cluster-vllm:v6` — upstream `vllm/vllm-openai:v0.30.0` plus **readable patches**, each an
anchored or sha256-checked edit that refuses to apply twice and fails the build if its target moved:

1. **the n-gram table as a GPU parameter** (`18-ple-nvfp4-v030.py`): when the checkpoint declares its table as NVFP4
   (hibrid47/48 do, in `config.json`), the table loads through 0.30's embedding plugin as a resident parameter — half the
   rows per box, or all of them with `MBX_PLE_REPLICATE=1` — gathered and dequantized inside the forward pass, inside the
   CUDA graphs. Stock 0.30 refuses the checkpoint without it.
2. **the 4-bit output head** (`11-lm-head-quant-config.py`): 0.30 still builds the model's and the drafter's output head
   without the checkpoint's quantization config, so stock 0.30 fails on hibrid48 with a shape mismatch.
3. **fused multi-step draft metadata** (`05-qsa-fused-draft-v2.py`): lets speculative decoding reuse one attention-metadata
   build across its draft steps on this model's QSA attention — the code proposed upstream as
   [vllm-project/vllm#58449](https://github.com/vllm-project/vllm/pull/58449), shipped as a sha256-checked overlay.
   +1–3 % engine steps at 1–12 streams, acceptance and GSM8K unchanged.
4. **loader page-cache drop** (`13`), QSA pre-indexer rope clamp (`03`), and two inert knobs (`16`, `17`), described in the
   patch files.

Digest: `sha256:861ac752164e0d723c5eff3f876586c6678c26ad4a516112c48745f6a101ff4d`
(v5, vLLM 0.29 + hibrid48, the v3 kit: `sha256:49b57ee9920b7132cd0b4d3e351c5ae96829c4094594981b8d9c711a56b65360`;
v4, vendor pin + hibrid47 + fp8 KV: `sha256:91423fc292d527935b2f0363cc614305b1c1a00dc56981953a723abe1b50ed2e`).

## The full box

This kit serves one model across two boxes, plain. The same model also runs under [myllmbox](https://myllmbox.com) with a
public HTTPS tunnel, dashboard, keepalive and multi-model management — same image, same weights.

## License

Weights: Qwen Community License 1.0 (permissive incl. commercial; >100M MAU/$20M-revenue products must display
the model name; Model-as-a-Service businesses need a separate Qwen license). Kit scripts and image patches: MIT.
