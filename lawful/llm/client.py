"""llama.cpp server lifecycle + a cached, thread-safe chat client.

Every request is keyed by (model, endpoint, payload) in a SQLite cache, so experiments are resumable and
re-running an analysis never re-pays for inference. Decoding is greedy unless a seed/temperature is given.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import math
import sqlite3
import subprocess
import threading
import time

import requests
from tqdm import tqdm

from lawful.llm.registry import LLAMA_SERVER, MODELS
from lawful.paths import RESULTS


class Cache:
    def __init__(self, path=RESULTS / "llm_cache.sqlite"):
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
        self._lock = threading.Lock()

    def get(self, k):
        with self._lock:
            row = self._db.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, k, v):
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (k, json.dumps(v)))
            self._db.commit()


CACHE = Cache()


class Server:
    """Context manager that runs `llama-server` for one model with N parallel decoding slots."""

    def __init__(self, alias: str, port: int = 8090, parallel: int = 4, ctx_per_slot: int = 4096, threads: int = 4):
        self.spec, self.port = MODELS[alias], port
        self.cmd = [str(LLAMA_SERVER), "-m", str(self.spec.path), "--port", str(port), "-t", str(threads),
                    "-c", str(ctx_per_slot * parallel), "-np", str(parallel), "--jinja", "--no-webui"]
        self.proc = None

    def __enter__(self) -> LLM:
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(240):
            try:
                if requests.get(f"http://127.0.0.1:{self.port}/health", timeout=2).json().get("status") == "ok":
                    return LLM(self.spec.alias, f"http://127.0.0.1:{self.port}")
            except requests.RequestException:
                pass
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early for {self.spec.alias}")
            time.sleep(1)
        raise TimeoutError(f"llama-server did not become healthy for {self.spec.alias}")

    def __exit__(self, *exc):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()


class LLM:
    def __init__(self, alias: str, base_url: str | None = None, cache: Cache = CACHE):
        self.alias, self.base, self.cache = alias, base_url, cache
        self.session = requests.Session()

    def _post(self, payload: dict) -> dict:
        key = hashlib.sha256(json.dumps([self.alias, payload], sort_keys=True).encode()).hexdigest()
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        if self.base is None:
            raise LookupError(f"cache miss for {self.alias} and no server attached")
        r = self.session.post(f"{self.base}/v1/chat/completions", json=payload, timeout=900)
        r.raise_for_status()
        out = r.json()
        self.cache.put(key, out)
        return out

    def chat(self, messages: list[dict], max_tokens: int = 256, temperature: float = 0.0, seed: int = 0,
             schema: dict | None = None, stop: list[str] | None = None) -> str:
        payload = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "seed": seed}
        if schema:
            payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
        if stop:
            payload["stop"] = stop
        return self._post(payload)["choices"][0]["message"]["content"] or ""

    def p_yes(self, messages: list[dict]) -> float:
        """P(Yes) renormalized over {Yes, No} at the first answer token; NaN if neither is in the top-20."""
        out = self._post({"messages": messages, "max_tokens": 1, "temperature": 0.0, "logprobs": True,
                          "top_logprobs": 20})
        top = out["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        mass = {"yes": 0.0, "no": 0.0}
        for t in top:
            w = t["token"].strip().strip("*").lower()
            if w in mass:
                mass[w] += math.exp(t["logprob"])
        tot = mass["yes"] + mass["no"]
        return mass["yes"] / tot if tot > 0 else float("nan")


def pmap(fn, items, workers: int = 4, desc: str = ""):
    """Order-preserving parallel map (matches the server's parallel slots)."""
    items = list(items)
    out = [None] * len(items)
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, x): i for i, x in enumerate(items)}
        for f in tqdm(cf.as_completed(futs), total=len(items), desc=desc, leave=False, mininterval=5):
            out[futs[f]] = f.result()
    return out
