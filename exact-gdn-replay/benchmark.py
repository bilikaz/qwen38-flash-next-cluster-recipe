# SPDX-License-Identifier: Apache-2.0
"""Fixed-workload HTTP benchmark; no server configuration changes.

Use an idle, exclusively reserved server. Compare identical payload hashes,
input/output token counts and cache state. Fixed output length is a throughput
workload, not a quality or losslessness evaluation.
"""

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import re
import statistics
import time
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--label", required=True)
p.add_argument("--out", required=True)
p.add_argument("--base-url", default="http://127.0.0.1:8000")
p.add_argument("--concurrency", type=int, default=1)
p.add_argument("--requests", type=int, default=3)
p.add_argument("--max-tokens", type=int, default=512)
p.add_argument("--repeat", type=int, default=0)
p.add_argument("--temperature", type=float, default=0)
p.add_argument("--thinking", action="store_true")
p.add_argument("--quality", action="store_true")
p.add_argument("--warmups", type=int, default=1)
p.add_argument("--case", default="code")
p.add_argument("--workload", choices=["mixed", "code"], default="mixed")
p.add_argument("--profile", action="store_true")
a = p.parse_args()
root = Path(a.out)
root.mkdir(parents=True, exist_ok=False)
MODEL = "Qwen/Qwen3.8-Flash-Next"
CASES = {
    "code": "Write a complete Python module implementing an LRU cache, a TTL cache, and a thread-safe bounded queue. Include type annotations, detailed docstrings, edge-case handling and usage examples. Explain the complexity after the implementation.",
    "reason": "A train leaves A at 09:00 travelling 60 km/h. A second leaves B at 09:30 travelling 90 km/h toward A. A and B are 240 km apart. Determine the meeting time and distances, then derive a general formula and verify it.",
    "long": "Summarize the engineering notes above, then write Python code implementing a bounded queue with tests. Discuss its invariants and edge cases.",
}
MIXED_CASES = [
    "Write a complete Python implementation of a persistent work queue using SQLite, leases, retries and crash recovery. Include detailed tests and explain all concurrency invariants.",
    "Create a complete TypeScript library for parsing and validating a small expression language. Include a lexer, parser, evaluator, error handling, extensive tests and worked examples.",
    "Write a rigorous tutorial on probability using ten fully worked problems about conditional probability, independence, Bayes theorem, expected values and variance. Explain each calculation in detail.",
    "Write a detailed technical comparison of B-trees, LSM trees and hash indexes. Discuss read and write paths, caching, compaction, concurrency, persistence and realistic workload examples.",
    "Design a relational database for a lending library. Provide the schema, constraints, indexes, sample data and fifteen SQL queries with detailed explanations of correctness and efficiency.",
    "Produce a long, carefully structured engineering guide to Unicode text processing. Cover code points, normalization, grapheme clusters, encoding errors, sorting, search and concrete Python examples.",
    "Explain Newtonian mechanics through ten fully worked examples involving motion, forces, energy, momentum and circular motion. Derive the formulas and verify units at each step.",
    "Write a detailed fictional expedition journal about mapping an unfamiliar island. Include distinct daily entries, natural observations, logistical decisions, setbacks and a coherent ending.",
    "Write a complete Rust implementation of a small key-value command-line database with serialization, atomic file replacement, error handling, unit tests and an explanation of the design.",
    "Explain how an HTTP request travels from a browser through DNS, TLS, a reverse proxy, an application server and a database. Include detailed timing examples, failure cases and debugging procedures.",
    "Create a complete Python program that generates and solves Sudoku puzzles, including constraint propagation, backtracking, uniqueness checking, tests and detailed commentary on complexity.",
    "Write a thorough guide to translating technical documentation between English and Spanish. Include a glossary, twenty paired examples, ambiguity analysis and a practical review checklist.",
    "Develop a detailed project plan for restoring a public botanical garden. Include a phased schedule, resource estimates, dependencies, accessibility considerations and measurable success criteria.",
    "Explain linear algebra through ten detailed worked examples involving matrices, rank, eigenvalues, projections, least squares and numerical conditioning. Include Python verification code.",
    "Write a complete Go HTTP service with a thread-safe in-memory cache, TTL expiration, graceful shutdown, metrics, request validation and a comprehensive collection of tests.",
    "Create a detailed course on reliable software testing. Cover unit, integration, property-based, concurrency and fault-injection tests with concrete code examples and explanations of their limitations.",
]
FILL = "Engineering note: producers append jobs to a bounded queue; consumers remove them in FIFO order. A condition variable protects shared state. Tests must cover timeout, cancellation, shutdown, empty and full queues.\n"


def request(path, data=None, timeout=900):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(
        a.base_url + path, data=body, headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(req, timeout=timeout)


def metrics():
    with request("/metrics", timeout=10) as r:
        return r.read().decode()


def values(text):
    out = {}
    for line in text.splitlines():
        if line.startswith("vllm:") and (
            "_total{" in line or "_sum{" in line or "_count{" in line
        ):
            key, val = line.rsplit(" ", 1)
            try:
                out[key] = float(val)
            except ValueError:
                pass
    return out


initial = metrics()
for name in ["num_requests_running", "num_requests_waiting"]:
    vals = re.findall(r"^vllm:" + name + r"\{[^\n]*\} ([^\n]+)", initial, re.M)
    if not vals or any(float(v) != 0 for v in vals):
        raise RuntimeError("Server is not verified idle: " + name)


def one(i, warmup=False):
    # Identical payloads across baseline/candidate labels. Unique early prefixes
    # distinguish cold requests; rerunning the same case can explicitly measure warm reuse.
    prompt = (
        f"Research workload {a.case}, record {i}.\n"
        + FILL * a.repeat
        + (
            MIXED_CASES[i % len(MIXED_CASES)]
            if a.workload == "mixed"
            else CASES["code"]
        )
    )
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": a.temperature,
        "top_p": 0.95,
        "top_k": 20,
        "seed": 8100 + i,
        "max_tokens": a.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": a.thinking},
        "ignore_eos": not a.quality,
        "return_token_ids": True,
    }
    if a.quality:
        payload.update(logprobs=True, top_logprobs=1)
    start = time.perf_counter()
    events = []
    ids = []
    text = []
    reasoning = []
    logs = []
    usage = None
    finish = None
    first = None
    with request("/v1/chat/completions", payload) as r:
        for raw in r:
            if not raw.startswith(b"data: "):
                continue
            data = raw[6:].strip()
            if data == b"[DONE]":
                break
            d = json.loads(data)
            now = time.perf_counter()
            if d.get("error"):
                raise RuntimeError(d["error"])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                tokens = c.get("token_ids") or []
                content = delta.get("content") or ""
                think = delta.get("reasoning") or delta.get("reasoning_content") or ""
                if tokens or content or think:
                    if first is None:
                        first = now
                    events.append([now - start, len(tokens)])
                ids.extend(tokens)
                text.append(content)
                reasoning.append(think)
                if c.get("logprobs"):
                    logs.append(c["logprobs"])
                if c.get("finish_reason"):
                    finish = c["finish_reason"]
    end = time.perf_counter()
    if usage is None:
        raise RuntimeError("Missing terminal usage")
    n = usage["completion_tokens"]
    if not a.quality and n != a.max_tokens:
        raise RuntimeError(f"Unequal fixed output workload: {n} != {a.max_tokens}")
    if len(ids) != n:
        raise RuntimeError(f"Missing token IDs: {len(ids)} != {n}")
    result = {
        "index": i,
        "warmup": warmup,
        "payload": payload,
        "payload_sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest(),
        "usage": usage,
        "latency_s": end - start,
        "ttft_s": None if first is None else first - start,
        "decode_tps": None
        if first is None or end == first
        else max(0, n - 1) / (end - first),
        "token_ids": ids,
        "text": "".join(text),
        "reasoning": "".join(reasoning),
        "logprobs": logs,
        "finish_reason": finish,
        "events": events,
    }
    (root / f"request-{i}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "index",
                    "warmup",
                    "usage",
                    "latency_s",
                    "ttft_s",
                    "decode_tps",
                ]
            }
        ),
        flush=True,
    )
    return result


for i in range(a.warmups):
    one(-1 - i, True)
before = metrics()
(root / "metrics-before.txt").write_text(before)
if a.profile:
    with request("/start_profile", {}, 30) as r:
        print("profile_start", r.status, flush=True)
start = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
    rows = list(pool.map(one, range(a.requests)))
wall = time.perf_counter() - start
if a.profile:
    with request("/stop_profile", {}, 120) as r:
        print("profile_stop", r.status, flush=True)
after = metrics()
(root / "metrics-after.txt").write_text(after)
b = values(before)
e = values(after)
delta = {k: v - b.get(k, 0) for k, v in e.items() if k in b}
summary = {
    "label": a.label,
    "args": vars(a),
    "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "wall_s": wall,
    "aggregate_tps": sum(x["usage"]["completion_tokens"] for x in rows) / wall,
    "median_ttft_s": statistics.median(x["ttft_s"] for x in rows),
    "median_decode_tps": statistics.median(x["decode_tps"] for x in rows),
    "metrics_delta": delta,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(
    json.dumps({k: v for k, v in summary.items() if k != "metrics_delta"}), flush=True
)
