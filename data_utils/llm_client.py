"""Shared client for the local vLLM OpenAI-compatible server.

Defaults to the Qwen3.6 server on port 9003 (the old Gemma endpoint on 9002 is dead).
Supports guided JSON decoding via response_format=json_schema and thread-pool fan-out.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

LLM_URL = os.environ.get("LLM_URL", "http://localhost:9003/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen/qwen3.6-35b-a3b")
# Thinking off by default: with guided JSON the reasoning phase can consume the whole
# max_tokens budget and leave content empty. Set LLM_THINKING=1 to re-enable.
LLM_THINKING = os.environ.get("LLM_THINKING", "0") == "1"


class LLMError(RuntimeError):
    pass


def chat(messages: list[dict], max_tokens: int = 2048, temperature: float = 0.7,
         retries: int = 3, timeout: int = 180, response_format: dict | None = None) -> str:
    """One chat completion; returns message.content (reasoning is separated
    into reasoning_content by the server's --reasoning-parser qwen3)."""
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": LLM_THINKING},
    }
    if response_format is not None:
        payload["response_format"] = response_format

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.post(LLM_URL, json=payload, timeout=timeout)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            if content is None:
                raise LLMError("empty content (all tokens went to reasoning?)")
            return content.strip()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(2 * (attempt + 1))
    raise LLMError(f"LLM call failed after {retries} attempts: {last_exc}")


def chat_json(messages: list[dict], schema: dict, schema_name: str = "output",
              max_tokens: int = 2048, temperature: float = 0.7, retries: int = 3) -> dict:
    """Chat completion with vLLM guided JSON decoding; returns the parsed object.

    Retries with a temperature bump when the output fails to parse.
    """
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": schema_name, "schema": schema},
    }
    last_exc: Exception | None = None
    for attempt in range(retries):
        text = chat(messages, max_tokens=max_tokens,
                    temperature=min(temperature + 0.1 * attempt, 1.0),
                    retries=2, response_format=response_format)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            last_exc = exc
    raise LLMError(f"guided JSON failed to parse after {retries} attempts: {last_exc}")


def map_concurrent(fn, items: list, workers: int = 16, desc: str = "llm"):
    """Run fn(item) across a thread pool. Yields (item, result, error) as they
    complete — exactly one of result/error is None."""
    from tqdm import tqdm
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): item for item in items}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            item = futures[fut]
            try:
                yield item, fut.result(), None
            except Exception as exc:  # noqa: BLE001 — surface per-item errors to caller
                yield item, None, exc
