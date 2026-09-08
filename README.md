# Qwen3.8-Flash-Next on two DGX Sparks

Two boxes, one model, RDMA. Peaks: **80 tok/s single-stream**, 674 tok/s at 48 streams (averages **73@c=1 and 635@c=48** on code). 26 of 32 boss-level render tests passed. Three commands.

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

## Measured performance (this exact stack, 2× DGX Spark, RDMA, K=4, `vm.compaction_proactiveness=0`)

Boot 2026-09-06 (fresh reboot), myllmbox "pasture" prompt, thinking disabled; each row is 3–7 independent runs of
steady 10-second engine windows (all streams running, zero prefill in the window); **peak** = the best steady window.
Measured on the layout this kit ships (full table per box) at a 25G KV pin with bf16 KV; the kit pins 28G. The fp8-KV
boot (v2.1) so far: single stream, thinking on, one 39,487-token request — 17.3 engine steps/s (16.8–17.7), 74 tok/s
on the code phase, 54 on thinking, 59 over the request; KV 1.9 % per running request, pool 2.85M tokens. The ladder
below is re-run on fp8 as the rungs get measured.

| concurrent requests | **PEAK tok/s** | average tok/s | per-stream | engine steps/s (v1 → v2) | acceptance |
|---|---|---|---|---|---|
| 1 | **80** | 73 | 73 | 16.4 → **17.7** | 4.1 (3.9–4.3) |
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
- **`kv-cache-dtype: fp8`** (v2.1): the vendor image refuses anything but bf16 KV on this model's QSA attention; image
  patch 04 is upstream PR #54846 ported (fp8_e4m3, with the PR's numerical tests run on the GB10), 1.66× the pooled
  tokens on the same pin. Set `bf16` to get the v2 pool back. The GDN state is a separate thing and stays as shipped.
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

`myllmbox/qwen38-flash-next-cluster-vllm:v4` — the single-Spark kit's image (`myllmbox/qwen38-flash-next-vllm:v1`,
vendor SM121 vLLM + the int3 n-gram-table loader patch) plus **four readable patches**:

1. **the n-gram table as a GPU parameter** (`03-ple-gpu-nvfp4.py`): when the checkpoint declares its table as NVFP4
   (hibrid47 does, in `config.json`), the table loads as an ordinary resident parameter — half the rows per box, or
   all of them with `MBX_PLE_REPLICATE=1` — and is gathered and dequantized inside the model's forward pass, inside
   the CUDA graphs. The converter that made the checkpoint (`make-hibrid47.py`) ships in the image at `/opt/mbx/`.
2. the safetensors loader drops each shard's pages from the page cache as soon as its tensors are consumed
   (`POSIX_FADV_DONTNEED`, gate `MBX_LOAD_DROP_CACHE=0` to disable) — see "Memory on a Spark" above.
3. the v1 path, kept as a fallback: upstream vLLM refuses its PLE CPU-offload worker when `nnodes != 1`; gated by
   `MBX_PLE_MULTINODE=1` (unset here), one full-table worker per box. Inert unless you serve an int3 checkpoint.
4. **fp8 KV on the QSA path** (`04-qsa-fp8-nvfp4-kv.py`, v2.1): upstream vLLM PR #54846 ported as whole-file overlays
   of the three files it touches (`docker/overlays/`; the patch script asserts the sha256 of the vendor originals and of the
   overlays at build time, so what ships is what is described); the PR's numerical tests ship in
   `docker/tests/` and pass on the GB10. Activated by `kv-cache-dtype: fp8` in `recipe.yaml`.

Each patch refuses to apply twice and fails the build if its anchor moved. The Dockerfile, the patch scripts, the
converter and the build ledger live in the myllmbox repo under
[`builds/qwen38-flash-next/cluster/`](https://github.com/bilikaz/myllmbox-runner/tree/main/builds/qwen38-flash-next/cluster)
— rebuild and diff it yourself. Digest: `sha256:91423fc292d527935b2f0363cc614305b1c1a00dc56981953a723abe1b50ed2e` (v3, bf16 KV: `sha256:ba28f473c766919afac75a898014d1dbe18a0929a7f55c0588512f27ee365513`).

## The full box

This kit serves one model across two boxes, plain. The same model runs under
[myllmbox](https://github.com/bilikaz/myllmbox-runner) with a public HTTPS tunnel, dashboard, keepalive and
multi-model management — same image, same weights, one `./run.sh qwen38-flash-next-cluster`.

## License

Weights: Qwen Community License 1.0 (permissive incl. commercial; >100M MAU/$20M-revenue products must display
the model name; Model-as-a-Service businesses need a separate Qwen license). Kit scripts and image patches: MIT.
