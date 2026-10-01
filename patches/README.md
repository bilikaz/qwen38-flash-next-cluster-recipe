# Optional vLLM patches

Off by default. Turn one on by listing its name in `recipe.yaml`:

```yaml
server:
  patches: hermes-chat          # space-separated, applied in this order
```

At launch `run.sh` copies each file a patch touches out of the image, applies the patch, copies the result to the
worker and mounts it read-only over the image's file on both boxes. The image is never changed; an empty
`patches` = the stock image. A patch that does not fit the image stops `run.sh` before anything starts.

| patch | what it does |
|---|---|
| `hermes-chat` | Hermes agent: reads its `{"reasoning": {…}}` object (thinking on/off, effort) and makes an omitted temperature greedy. Contributed by [@yume-arasaki](https://github.com/yume-arasaki) (#2) |
| `gb10-skinny-gemm` | GB10 (SM12x) plans for vLLM's Qwen4Exp skinny decode GEMM: decode-sized BF16 projections run the CuTe-DSL kernel (240-255 GB/s) instead of cuBLAS's SM80 WMMA fallback (125-225 GB/s). 1-stream decode step 47.6 → 46.1 ms on hibrid48 (with five vLLM backports in the same measurement); `MBX_SKINNY_GEMM_SM12X=0` turns it off. Contributed by [@sethforprivacy](https://github.com/sethforprivacy) |

## Adding one

A unified diff with paths relative to the `vllm` package (`--- a/entrypoints/…`, `+++ b/entrypoints/…`), made against
the image in `recipe.yaml`. Lines before the first `---` are a free-text description.
