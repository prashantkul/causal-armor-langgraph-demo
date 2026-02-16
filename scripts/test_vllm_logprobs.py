#!/usr/bin/env python3
"""Smoke test: verify vLLM returns logprobs for the Gemma model.

Run after starting the vLLM server (./scripts/start_vllm.sh):

    uv run python scripts/test_vllm_logprobs.py

Reads CAUSAL_ARMOR_PROXY_BASE_URL and CAUSAL_ARMOR_PROXY_MODEL from .env
(via causal-armor's dotenv loading) or falls back to defaults.
"""

from __future__ import annotations

import os
import sys

import httpx

import causal_armor  # noqa: F401 — triggers load_dotenv()

BASE_URL = os.environ.get("CAUSAL_ARMOR_PROXY_BASE_URL", "http://localhost:8000").rstrip("/")
MODEL = os.environ.get("CAUSAL_ARMOR_PROXY_MODEL", "google/gemma-3-12b-it")

PROMPT = "User: Book a flight to Paris\nAssistant: Sure, I'll book flight AA123 for you."


def main() -> None:
    print(f"vLLM server : {BASE_URL}")
    print(f"Model       : {MODEL}")
    print(f"Prompt      : {PROMPT!r}")
    print()

    # 1. Check /v1/models is reachable
    print("1) Checking /v1/models ...")
    try:
        resp = httpx.get(f"{BASE_URL}/v1/models", timeout=10)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"   FAIL — cannot reach vLLM: {exc}")
        sys.exit(1)

    models = [m["id"] for m in resp.json().get("data", [])]
    print(f"   OK — serving: {models}")

    if MODEL not in models:
        print(f"   WARN — requested model '{MODEL}' not in served models")

    # 2. Call /v1/completions with echo + logprobs
    print()
    print("2) Calling /v1/completions with echo=True, logprobs=1, max_tokens=0 ...")
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": 0,
        "echo": True,
        "logprobs": 1,
    }

    try:
        resp = httpx.post(f"{BASE_URL}/v1/completions", json=payload, timeout=30)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"   FAIL — completions request failed: {exc}")
        sys.exit(1)

    data = resp.json()

    # 3. Validate logprobs structure
    try:
        choice = data["choices"][0]
        logprobs_data = choice["logprobs"]
        token_logprobs = logprobs_data["token_logprobs"]
        tokens = logprobs_data["tokens"]
        text_offsets = logprobs_data["text_offset"]
    except (KeyError, IndexError) as exc:
        print(f"   FAIL — unexpected response structure: {exc}")
        print(f"   Response: {data}")
        sys.exit(1)

    n_tokens = len(tokens)
    n_valid = sum(1 for lp in token_logprobs if lp is not None)

    print(f"   OK — got {n_tokens} tokens, {n_valid} with valid logprobs")
    print()

    # 4. Show a sample of tokens + logprobs
    print("3) Sample tokens and logprobs:")
    print(f"   {'Token':<20} {'LogProb':>10}  {'Offset':>6}")
    print(f"   {'─' * 20} {'─' * 10}  {'─' * 6}")
    for i in range(min(15, n_tokens)):
        tok = repr(tokens[i])
        lp = token_logprobs[i]
        off = text_offsets[i]
        lp_str = f"{lp:.4f}" if lp is not None else "None"
        print(f"   {tok:<20} {lp_str:>10}  {off:>6}")

    if n_tokens > 15:
        print(f"   ... ({n_tokens - 15} more tokens)")

    print()
    print("All checks passed. vLLM is returning logprobs correctly.")


if __name__ == "__main__":
    main()
