# Qwen3.8-Flash-Next on two DGX Sparks

**Two boxes, one model, RDMA. 667 tok/s of code at 64 streams (540 averaged over code and thinking), 68 tok/s single-stream. Three commands.**

Serves [myllmbox/Qwen3.8-Flash-Next-hibrid46](https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid46)
— the same 4.35-bit-effective build the [single-Spark kit](https://github.com/bilikaz/qwen38-flash-next-recipe)
serves — split tensor-parallel across **two NVIDIA DGX Sparks (GB10, 119G unified memory each)** over their
ConnectX link, NCCL on RDMA. What the second box buys, measured on this exact kit: **+35–40% engine speed at every
concurrency** (per-step weight traffic halves) and **2× the seats** at equal per-stream speed. Not 2× single-stream:
a Spark pair adds one interconnect round trip per layer, and the numbers below say what is left after paying it.

## Quick start

```bash
git clone https://github.com/bilikaz/qwen38-flash-next-cluster-recipe.git
cd qwen38-flash-next-cluster-recipe
./run.sh        # first run: sets the cluster up (asks for the 2nd box), downloads ~91G, syncs it, serves on :8000
```

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

## Measured performance (this exact kit, 2× DGX Spark, RDMA, K=4)

65 runs, 1,994 steady 10-second engine windows (all streams running, zero prefill in the window), the myllmbox
"pasture" prompt in both bands: **code** = thinking disabled, **thinking** = thinking enabled. AVERAGE is the
arithmetic mean of the two bands' averages (each band weighs the same), its range the extremes either band reported.
Engine steps/s and acceptance are pooled the same way — note the engine speed is identical in both bands; only
acceptance (tokens per step) differs, 4.2 on code vs 2.7–3.1 on thinking.

| concurrent requests | AVERAGE tok/s (min–max) | AVERAGE per-stream | code tok/s | code per-stream | thinking tok/s | thinking per-stream | engine steps/s | acceptance |
|---|---|---|---|---|---|---|---|---|
| 1 | **59** (34–77) | **59.2** | 68.4 | 68.4 | 50.1 | 50.1 | 16.4 (15.3–17.2) | 3.60 (2.19–4.62) |
| 2 | **99** (61–123) | **49.7** | 118.3 | 59.1 | 80.4 | 40.2 | 13.8 (12.6–14.4) | 3.59 (2.33–4.44) |
| 4 | **153** (103–195) | **38.2** | 184.7 | 46.2 | 121.3 | 30.3 | 10.7 (9.7–11.3) | 3.56 (2.53–4.48) |
| 8 | **224** (157–290) | **28.1** | 279.8 | 35.0 | 169.1 | 21.1 | 8.0 (7.3–8.7) | 3.47 (2.55–4.36) |
| 16 | **322** (229–416) | **20.1** | 395.7 | 24.7 | 247.8 | 15.5 | 5.8 (5.3–6.2) | 3.47 (2.55–4.48) |
| 24 | **366** (265–475) | **15.3** | 450.2 | 18.8 | 282.3 | 11.8 | 4.4 (4.2–4.6) | 3.47 (2.54–4.50) |
| 32 | **407** (293–522) | **12.7** | 501.7 | 15.7 | 312.7 | 9.8 | 3.7 (3.4–3.9) | 3.45 (2.56–4.37) |
| 48 | **493** (364–645) | **10.3** | 600.8 | 12.5 | 384.3 | 8.0 | 3.0 (2.7–3.2) | 3.45 (2.49–4.41) |
| 52 | **509** (375–661) | **9.8** | 620.2 | 11.9 | 398.3 | 7.7 | 2.8 (2.6–3.0) | 3.46 (2.58–4.41) |
| 64 | **540** (346–721) | **8.4** | 666.5 | 10.4 | 413.2 | 6.5 | 2.4 (2.1–2.7) | 3.45 (2.55–4.42) |

Reading it: single-stream 68 tok/s on code, 50 on thinking. At 16 seats every agent still gets 25 tok/s on code
(20 average). At 64 seats the cluster delivers 667 tok/s of code (540 average) at 10 tok/s each — the same
per-user speed one Spark gives at 32 seats, so **twice the seats at equal speed**. Single Spark, same checkpoint,
for reference: 44 tok/s at c=1, ~153 at c=8, 305 max at c=32 (code). Acceptance holds 3.45–3.60 average at every
rung with the same 2.2–4.6 range throughout — speculative decoding does not degrade under load. The engine has a
batch-size step between 52 and 56 sequences (56 and 60 cost a 64-sized step): 52 and 64 are the efficient seat
counts at the top. Full 262,144-token context; the 46G-per-box KV pool holds ~3.1M pooled tokens, ~1.15 % of it per
running request (the model's fixed GDN state), so ~85 seats is the pool's ceiling. Numbers carry their conditions
on purpose — the tools that produced them (`bench/test.py`, `bench/summary.py` in the myllmbox repo) are yours to
rerun.

## The RDMA part (why the numbers are what they are)

NCCL will happily run over TCP sockets on the same ConnectX cable and never say so — the env vars look right, the
serve works, and every step is ~2× slower. A container needs three things to actually open the RDMA device:
`--device /dev/infiniband`, `--cap-add IPC_LOCK`, `--ulimit memlock=-1:-1`. `run.sh` passes them; `view.sh`
proves it by sampling the HCA's port counter against the interface's TCP byte counter while the model decodes
(RDMA moving, TCP flat = good). If `/dev/infiniband` is missing on a box (`rdma-core` not installed, or the link
is not a ConnectX one), the kit still runs, over TCP, and says so.

## Tuning (recipe.yaml)

- **`kv-cache-memory`** (bytes, per box): 46G default. vLLM's own "fit" figure at util 0.70 is 39G; 46 leaves
  ~11G of host headroom per box. Do **not** take its "fully utilize" suggestion (~64G): that ignores the 18G
  n-gram table living in the CPU worker and the OS, and unified memory over-commit has needed a power cycle.
  KV must stay **bf16** on this model (the vendor's QSA guard refuses fp8).
- **`max-num-seqs`**: 64 measured, aggregate still climbing. KV usage is ~1.15% per running request
  regardless of length (the model's fixed GDN state), so the 46G pool tops out near 85 seats.
- **`speculative-config`** K=4: acceptance 4.2–4.6 with the cap at 5.0, ~8% step cost vs K=3, net faster at
  every concurrency. K=3 was 18 steps/s single-stream if you want to compare.
- **`max-num-batched-tokens`**: also the image-input encoder budget — 8192 fits realistic multi-image requests.
- **`host: 127.0.0.1`**: the cluster runs on host networking, so the API would otherwise be on every interface.
  Set `0.0.0.0` to expose it on the LAN.
- Thinking is ON by default (model native); disable per request with
  `"chat_template_kwargs": {"enable_thinking": false}` for max speed on structured output.

## What's in the image

`myllmbox/qwen38-flash-next-cluster-vllm:v1` (pushed 2026-09-05) — the single-Spark kit's image
(`myllmbox/qwen38-flash-next-vllm:v1`, vendor SM121 vLLM + the int3 n-gram-table loader patch) plus **one more
readable patch**: upstream vLLM refuses its PLE CPU-offload worker when `nnodes != 1`; the patch (gated by
`MBX_PLE_MULTINODE=1`) runs one full-table worker per box and lets each rank feed its own. Six anchored edits in
`gpu_worker.py`, `ple_offload/worker.py`, `ple_offload/connector.py`, each refusing to apply twice. The
Dockerfile, the patch script and the build ledger live in the myllmbox repo under
[`builds/qwen38-flash-next/cluster/`](https://github.com/bilikaz/myllmbox-runner/tree/main/builds/qwen38-flash-next/cluster)
— rebuild and diff it yourself. Digest: `sha256:c93988a847674d742fc4cc87ec2bd386c753727d65e8bbc4b739de9a9da68618`.

## The full box

This kit serves one model across two boxes, plain. The same model runs under
[myllmbox](https://github.com/bilikaz/myllmbox-runner) with a public HTTPS tunnel, dashboard, keepalive and
multi-model management — same image, same weights, one `./run.sh qwen38-flash-next-cluster`.

## License

Weights: Qwen Community License 1.0 (permissive incl. commercial; >100M MAU/$20M-revenue products must display
the model name; Model-as-a-Service businesses need a separate Qwen license). Kit scripts and image patches: MIT.
