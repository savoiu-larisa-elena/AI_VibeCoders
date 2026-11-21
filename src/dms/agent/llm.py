"""LLM client. One function for the rest of the code: chat_json(system, user) -> dict.

Provider is chosen in .env (LLM_PROVIDER=groq | gemini | ollama). All three speak the
OpenAI-compatible API, so the same `openai` package talks to every one of them.
"""
from __future__ import annotations

import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from ..config import ROOT

load_dotenv(ROOT / ".env")

PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model": "openai/gpt-oss-20b",      # fast; "openai/gpt-oss-120b" = smarter, slower
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-flash-latest",
    },
    "ollama": {   # local + offline:  ollama pull llama3.2
        "base_url": "http://localhost:11434/v1",
        "key_env": None,
        "model": "llama3.2",
    },
}


class LLMError(Exception):
    pass


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None,
                 timeout: float = 4.0):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower()
        if self.provider not in PROVIDERS:
            raise ValueError(f"LLM_PROVIDER must be one of {list(PROVIDERS)}")
        p = PROVIDERS[self.provider]
        key = os.getenv(p["key_env"]) if p["key_env"] else "ollama"
        if not key:
            raise LLMError(f"{p['key_env']} is missing - add it to the .env file")
        self.model = model or os.getenv("LLM_MODEL") or p["model"]
        # max_retries=0: in a car, a late answer is useless - we fall back instead
        self.client = OpenAI(api_key=key, base_url=p["base_url"], timeout=timeout,
                             max_retries=0)

    def chat_json(self, system: str, user: str) -> tuple[dict, float]:
        """Returns (parsed JSON reply, latency in seconds). Raises LLMError on any failure."""
        kwargs = dict(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            max_tokens=400,
        )
        if self.model.startswith("openai/gpt-oss"):
            kwargs["reasoning_effort"] = "low"   # these models "think" first; keep it short
        t0 = time.monotonic()
        try:
            resp = self.client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            return json.loads(text), time.monotonic() - t0
        except json.JSONDecodeError as e:
            raise LLMError(f"reply was not valid JSON: {e}") from e
        except Exception as e:  # noqa: BLE001 - network, auth, rate limit, timeout...
            raise LLMError(f"{type(e).__name__}: {e}") from e
