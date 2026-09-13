# Qwen3.8-Flash-Next on two DGX Sparks

### Two checkpoints: `hibrid48`, the default — and `hibrid48-uncensored`, made from it (abliterated, gated). Switch with one line in `recipe.yaml`

[`hibrid48`](https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid48) (the base model, default)
and [`hibrid48-uncensored`](https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored) (OrcaRouter's abliterated
body, no refusals, no guardrails — gated, research / private use). Same stack. To switch: in `recipe.yaml`
comment the active `model:` line and uncomment the other, then `./run.sh`. Details in [Which checkpoint](#which-checkpoint).

Two boxes, one model, RDMA. **22 engine steps/s**: 92 tok/s single-stream writing code, **107 tok/s peak**. Three commands.

## Quality (measured on this serve, thinking on)

[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) run against the running two-Spark serve of
hibrid48 (2026-09-13), **thinking on** — the mode the model is served in — at temperature 0.6, top-p 0.95, top-k 20 (the model
card recommends 1.0 / 0.95 / 20 for thinking; a card-faithful pass at 1.0 is a separate run), a 32k-token budget per answer, 16
requests in flight. Qwen publishes no numbers for these four tests — its card reports LiveCodeBench v6 91.9, GPQA Diamond 91.7,
IFBench 81.3, SWE-bench Pro 62.5 — so there is no official row to compare against; GPQA Diamond is the one overlap still to run. HumanEval is complete; the
other tasks use a fixed 200-question subset (seed 123123123), the same questions on every checkpoint we compare.

| task | questions | score |
|---|---|---|
| HumanEval pass@1 | 164 | **95.7** |
| GSM8K exact match | 200 | **98.0** |
| IFEval prompt-level strict / instruction-level strict | 200 | **91.5** / 93.4 |
| MMLU-Pro (14 subjects, sampled by size) | 200 | **84.9** |

763 requests in 91 minutes. 11 of them (1.4 %) ran into the 32k budget while still thinking and count as wrong above — that
runaway rate is itself a number to compare between checkpoints. Subsets of 200 carry about ±3 points of sampling noise;
published leaderboard numbers use other prompts, few-shot counts and full sets, so treat them as a sanity band, not a column.
The runner (`bench/quality/` in the myllmbox repo: harness driver, answer extraction that reads a chat model's final answer,
and a side-by-side table tool) works against any OpenAI-compatible endpoint — rerun it and count. hibrid47 on the same
questions is the pending A/B; until then these are the model's numbers, not a measured cost of the 4-bit head.

## Measured performance (this exact stack, 2× DGX Spark, RDMA, K=4, `vm.compaction_proactiveness=0`)

**v3 ladder (2026-09-13, hibrid48 on vLLM 0.29, 28G bf16 pin, Marlin MoE)** — same prompt, same windows, same rules as the v2 table below (rungs 2–32 measured on the v3 kit boot; c=48 pending).

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


## What changed

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
git clone https://github.com/bilikaz/qwen38-flash-next-cluster-recipe.git
cd qwen38-flash-next-cluster-recipe
./run.sh        # first run: sets the cluster up (asks for the 2nd box), downloads ~99G, syncs it, serves on :8000
```

**Hugging Face token.** Anonymous downloads are rate-limited, and gated models (license-agreement repos, e.g. uncensored variants) refuse anonymous access. `run.sh` looks for `HF_TOKEN`, then `~/.cache/huggingface/token` (`hf auth login`), and asks for one when the repo is gated — after you accepted its agreement on the model page. Nothing is stored by the kit.

`./stop.sh` stops both boxes. `./view.sh` shows live stats plus the RDMA proof. Requirements: two DGX Sparks
with docker + the NVIDIA container runtime, connected by their ConnectX ports (a direct cable or a switch),
ssh from the head to the worker (a password once — `setup.sh` installs a key). First boot reaches healthy in
~10 minutes after the weights are on both boxes; later boots are faster.

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
| `myllmbox/Qwen3.8-Flash-Next-hibrid48` (default) | the base model, calibrated body, NVFP4 output head — the checkpoint the v3 numbers were measured on | 22.0 steps/s, 92 tok/s at c=1, **107** peak |
| `myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored` | OrcaRouter's abliterated (refusal-removed) body with the same head — **no guardrails**; research, red-teaming, private use behind your own moderation | same head, same stack; its own table is not measured yet |

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
- the image's loader **drops each shard from the cache as soon as it has been consumed**, so the cache never
  balloons during the load itself.

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

- **`kv-cache-memory`** (bytes, per box): 28G default with the full table on each box (~65G of weights per box).
  Leaves ~5G of host headroom per box after graph capture — check `free -g` on both boxes after the first boot and
  back off to 25G if either shows swap in use. Do **not** take vLLM's "fully utilize" suggestion: unified memory
  over-commit has needed a power cycle.
- **`kv-cache-dtype: fp8`** (commented out in v3): upstream PR #54846 ported to 0.29 as image patch 12 (fp8_e4m3 on the QSA
  path), 1.66× the pooled tokens on the same pin (2.85M). Measured on this stack against bf16 in 5+5 runs: same steps/s,
  acceptance 4.23 → 3.92, i.e. ~7 % fewer tokens per second. Uncomment it when many long seats matter more than
  single-stream speed. The GDN state is a separate thing and stays as shipped.
- **`compilation-config`** (new in v3): 0.29's model runner rounds every FULL-graph capture size up to a multiple of K+1 and
  drops the ones past the largest listed, so the list must reach `max-num-seqs × 5`. Shorten it only together with
  `max-num-seqs`; a list that stops short leaves the upper rungs decoding without CUDA graphs.
- **`MBX_PLE_REPLICATE: "1"`** (env): the full table on each box, no per-step exchange — the layout the numbers were
  measured on. Unset it for half the table per box: 14G freed per box, so the pin can go to 40G (~80 seats) at
  engine steps within 1 % — measure before you publish numbers from it.
- **`gpu-memory-utilization`** 0.70: with the pin set it does not size the KV pool (verified: 65G of weights plus the
  pin boot at 0.70).
- **`max-num-seqs`**: 32 — ~89k tokens of pool per seat, 17 tok/s per stream. Each running request also pins ~1.9 % of
  the pool the moment it is admitted, regardless of length: the GDN recurrent state, 36 layers × (2 + K) blocks of one
  state each, held for rollback of rejected draft tokens. fp8 KV halves the attention bytes, not this, so ~52 short
  requests is the hard ceiling of the 28G pin; 48 seats ran at 13 tok/s each with ~3k tokens of pool left per seat, which
  is why the kit stops at 32. Fewer seats = longer holds before preemption; a smaller K = more seats (2 + K).
- **`speculative-config`** K=4: acceptance ~4.2 on code and prose, ~2.5 on long reasoning, cap 5.0. A bf16 drafter
  was A/B'd and rejected (acceptance +0.01, −3 % steps, +3.4G) — the drafter's precision is not what bounds acceptance.
- **`max-num-batched-tokens`**: also the image-input encoder budget — 8192 fits realistic multi-image requests.
- **`host: 127.0.0.1`**: the cluster runs on host networking, so the API would otherwise be on every interface.
  Set `0.0.0.0` to expose it on the LAN.
- Thinking is ON by default (model native); disable per request with
  `"chat_template_kwargs": {"enable_thinking": false}` for max speed on structured output.

## What's in the image

`myllmbox/qwen38-flash-next-cluster-vllm:v5` — upstream `vllm/vllm-openai:v0.29.0` (the model is upstream in 0.29) plus
**six readable patches**, each an anchored edit that refuses to apply twice and fails the build if its anchor moved:

1. **the n-gram table as a GPU parameter** (`01-ple-nvfp4-v029.py`): when the checkpoint declares its table as NVFP4
   (hibrid47/48 do, in `config.json`), the table loads as an ordinary resident parameter — half the rows per box, or all
   of them with `MBX_PLE_REPLICATE=1` — and is gathered and dequantized inside the model's forward pass, inside the CUDA graphs.
2. **the 4-bit output head** (`11-lm-head-quant-config.py`): two lines — 0.29 builds the model's and the drafter's output
   head without the checkpoint's quantization config, so stock 0.29 fails on hibrid48 with a shape mismatch. This is the
   one patch a stock vLLM needs to load these checkpoints; hibrid47 (bf16 head) does not need it.
3. **fp8 KV on the QSA path** (`12-qsa-fp8-kv-v029.py`): upstream PR #54846 as whole-file overlays of the three files it
   touches; the build asserts the sha256 of the 0.29 originals and of the overlays. Inert until `kv-cache-dtype: fp8`.
4. QSA pre-indexer rope clamp (`03`), 5. QSA fused draft (`05`), 6. b12x MoE pre-warm clamp (`09`): three small
   robustness fixes for the SM121 path, described in the patch files.

The Dockerfile, the patch scripts and the build ledger live in the myllmbox repo under
[`recipes/qwen38-flash-next-fast/`](https://github.com/bilikaz/myllmbox-runner/tree/main/recipes/qwen38-flash-next-fast)
— rebuild and diff it yourself. Digest: `sha256:49b57ee9920b7132cd0b4d3e351c5ae96829c4094594981b8d9c711a56b65360`
(v4, vendor pin + hibrid47 + fp8 KV: `sha256:91423fc292d527935b2f0363cc614305b1c1a00dc56981953a723abe1b50ed2e`).

## The full box

This kit serves one model across two boxes, plain. The same model runs under
[myllmbox](https://github.com/bilikaz/myllmbox-runner) with a public HTTPS tunnel, dashboard, keepalive and
multi-model management — same image, same weights, one `./run.sh qwen38-flash-next-fast`.

## License

Weights: Qwen Community License 1.0 (permissive incl. commercial; >100M MAU/$20M-revenue products must display
the model name; Model-as-a-Service businesses need a separate Qwen license). Kit scripts and image patches: MIT.
