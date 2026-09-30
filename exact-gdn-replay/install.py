"""Apply the integration only inside a derived image with exact source guards."""

import hashlib
import importlib.util
import json
from pathlib import Path


def once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError("Integration anchor changed or already applied")
    return text.replace(old, new)


def integrate(root, manifest):
    originals = {}
    for relative, expected in manifest["integration_source_sha256"].items():
        path = root / relative
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f"Pinned v6 source mismatch: {relative}")
        originals[relative] = data.decode()

    layer = "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
    text = originals[layer]
    text = once(
        text,
        "        num_requests = attn_metadata.num_spec_decodes\n"
        "        ops.fused_gdn_decode_post_conv_mtp(\n",
        "        num_requests = attn_metadata.num_spec_decodes\n"
        '        replay = getattr(self, "_mbx_exact_replay", None)\n'
        "        if replay is not None and replay.supports(attn_metadata):\n"
        "            replay.run(mixed_qkv, a, b, output_gate, core_attn_out, attn_metadata)\n"
        "            return\n"
        "        ops.fused_gdn_decode_post_conv_mtp(\n",
    )
    changed = {layer: text}
    state = "v1/worker/gpu/model_states/mamba_hybrid.py"
    text = originals[state]
    text = once(
        text,
        "        super().__init__(vllm_config, model, encoder_cache, device)\n",
        "        super().__init__(vllm_config, model, encoder_cache, device)\n"
        "        from mbx_gdn_replay.runtime import ExactReplayState\n"
        "        self._mbx_exact_replay = ExactReplayState(vllm_config, model)\n",
    )
    text = once(
        text,
        "        return attn_metadata\n",
        "        self._mbx_exact_replay.record_step(attn_metadata, for_capture)\n"
        "        return attn_metadata\n",
    )
    text = once(
        text,
        "        # Chunked prefill does not sample a token, so num_sampled can be 0.\n",
        "        self._mbx_exact_replay.commit_step(num_sampled, idx_mapping, num_computed_tokens)\n"
        "        # Chunked prefill does not sample a token, so num_sampled can be 0.\n",
    )
    changed[state] = text
    # Validate every edit before writing either file. The original image remains
    # immutable; backups here also make the layer contents auditable.
    for relative, text in changed.items():
        compile(text, relative, "exec")
    for relative, text in changed.items():
        path = root / relative
        path.with_suffix(path.suffix + ".before-mbx-replay").write_text(
            originals[relative]
        )
        path.write_text(text)
    return {
        name: hashlib.sha256(text.encode()).hexdigest()
        for name, text in changed.items()
    }


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    spec = importlib.util.find_spec("vllm")
    root = Path(next(iter(spec.submodule_search_locations)))
    result = integrate(root, json.loads((here / "source-manifest.json").read_text()))
    (here / "installed-source-sha256.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result))
