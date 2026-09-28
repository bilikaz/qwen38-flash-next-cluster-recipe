# Hermes on hibrid48: what broke, what was the quant, what we changed

Date: 2026-09-28. Serve under test: `myllmbox/Qwen3.8-Flash-Next-hibrid48` on two DGX Sparks, image `myllmbox/qwen38-flash-next-cluster-vllm:v6` (vLLM 0.30.0), MTP K=5 probabilistic drafts accepted as a block. This is not Mia's RadixArk NVFP4 recipe and the numbers below are not a chart against it.

## What the harnesses send

Mia's `start.sh` sets no temperature. The Yume runners send `"temperature": 0.0` on every request. Hermes, for a custom provider, does not. `_fixed_temperature_for_model` returns nothing for this model, and `chat_completions.py` only adds `temperature` when the caller passed one. Chippy and Cassia have no temperature key. The request leaves the field out.

Hermes does send reasoning, but not as `reasoning_effort`. A custom provider puts `{"reasoning": {"enabled": false, "effort": "none"}}` at the top of the JSON (`extra_body`). vLLM 0.30 only maps the field named `reasoning_effort` onto `enable_thinking`.

## What vLLM 0.30 does with that

This checkpoint's `generation_config.json` is `temperature: 1.0`, `top_k: 20`, `top_p: 0.95`, `do_sample: true`. RadixArk's file is the same. vLLM 0.30 applies it when the request omits temperature. The boot log says so, and names `--generation-config vllm` as the opt-out. That flag is not enough: the server default under it is still temperature 1.0. Mia's older image does not apply this file, and her draft is argmax (`use_local_argmax_reduction` when the draft vocab is limited, K=3). This recipe samples the draft.

The chat template does the other half. If thinking is on and `reasoning_effort` is missing, it defaults to `xhigh`. So a Hermes `medium` that never reached the template became the longest think. The trace ran until the cap, never emitted `</think>`, and decayed into `function86 to90 turn91`.

## What we ruled out

- The NVFP4 output head. Grafting RadixArk's bf16 `lm_head.weight` (`[248320, 2560]`, snapshot `7b719225`) did not stop the loop. The graft is on disk and is not served. Dequantizing the 4-bit head would not have restored the lost weights anyway.
- YaRN as the cause of the loop. The same loop appeared with YaRN off, native 262144, greedy MTP K=3, temperature 0, and a 76-token prompt, while the server was still secretly on `xhigh`.
- Watermarking. Forcing `watermarking: false` did not stop the loop and did not stop script-mixing.
- Context past the native window. Cassia's bad table was inside a long session, but Chippy reproduced five Chinese characters on an empty-context table with thinking forced off. 162k is under 262144. Static YaRN still rescales those positions. It was not required for the loop.

Greedy, thinking off, stays English. Prose at 600 tokens was 74.4 tok/s and count-to-200 at 400 tokens was 133.0 tok/s on 2026-09-27, next to the K=5 grid's 78.4 and 131.8. Those gains are real for a harness that sends temperature 0. They are not a description of Hermes.

## The patch

`patches/hermes-chat.patch`, applied by `run.sh` onto the image's `protocol.py` and bind-mounted on both ranks:

1. A top-level `reasoning` object sets `enable_thinking`, and `low` / `medium` / `xhigh` are forwarded into the template. `none` turns thinking off. An explicit `chat_template_kwargs.enable_thinking` still wins.
2. If the request omits `temperature`, the chat path uses 0. An explicit temperature still passes through.

After that, three table requests with temperature omitted and thinking off were English. One `medium` request with temperature omitted stopped after a short think and answered in English.

## What the patch does not do

It does not scrub a session that already contains the bad text. A greedy decode will quote `превосход` or `<|SPEC|>` if they are in the prompt. Chippy's profile is still `medium` and that chat was not cleared.

It does not make a sampled `medium` think fit inside a short `max_tokens`. At temperature 0.8 the think block used the whole 900–1200 token budget and the answer was empty. Hermes does not send temperature, so with this patch those turns are greedy and the think closes. A client that sends 0.8 still gets 0.8.

It does not re-measure his 1-in-6 Chinese rate on long tables after the temperature change. That probe was taken while omitted temperature still sampled.

FlashInfer GDN prefill (`gdn-prefill-backend: flashinfer`, his v4.1) is a separate change. He measured about +5% prefill from 8k to 256k and no decode change. It is not part of this patch.
