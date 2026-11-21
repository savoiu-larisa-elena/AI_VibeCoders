"""LLM client with AUTOMATIC FAILOVER. One function for the rest of the code:
    chat_json(system, user) -> (dict, latency)

Free API tiers have limits (Groq free: 200,000 tokens per day per model). When the current
provider hits a limit or fails, the next one is tried automatically:

    1. LLM_PROVIDER's main model on every key   (groq: openai/gpt-oss-20b)
    2. Groq backup model on every Groq key       (openai/gpt-oss-120b - Groq limits are PER MODEL)
    3. the other providers that have a key       (gemini: gemini-flash-lite-latest)
    -> if all fail: LLMError, and the agent uses its pre-written message

Keys come from .env:  GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3, GEMINI_API_KEY
NOTE: Groq limits are per ORGANIZATION (account). A second key from the SAME account shares
the same limit; a key from another person's account adds real extra capacity.
All providers speak the OpenAI-compatible API, so the same `openai` package talks to all.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import openai
from dotenv import load_dotenv
from openai import OpenAI

from ..config import ROOT

load_dotenv(ROOT / ".env")

PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_envs": ["GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3"],
        "models": ["openai/gpt-oss-20b", "openai/gpt-oss-120b"],   # main, backup
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_envs": ["GEMINI_API_KEY"],
        "models": ["gemini-flash-lite-latest"],
    },
    "ollama": {   # local + offline:  ollama pull llama3.2  (only used if LLM_PROVIDER=ollama)
        "base_url": "http://localhost:11434/v1",
        "key_envs": [],
        "models": ["llama3.2"],
    },
}

PAUSE_MINUTE_LIMIT = 60        # seconds to skip a candidate after a per-minute limit
PAUSE_DAY_LIMIT = 30 * 60      # ... after a per-day limit (re-checked every 30 min)


class LLMError(Exception):
    pass


@dataclass
class Candidate:
    provider: str
    model: str
    key_name: str
    client: OpenAI
    paused_until: float = 0.0
    disabled: str = ""             # permanent problem (bad key, unknown model)
    calls: int = 0
    failures: int = field(default=0)

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}" + (f" ({self.key_name})" if self.key_name else "")

    def available(self, now: float) -> bool:
        return not self.disabled and now >= self.paused_until


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None,
                 timeout: float = 4.0):
        primary = (provider or os.getenv("LLM_PROVIDER", "groq")).lower()
        if primary not in PROVIDERS:
            raise ValueError(f"LLM_PROVIDER must be one of {list(PROVIDERS)}")
        main_model = model or os.getenv("LLM_MODEL")
        # max_retries=0: in a car, a late answer is useless - we fail over instead
        mk = lambda base, key: OpenAI(api_key=key, base_url=base, timeout=timeout,  # noqa: E731
                                      max_retries=0)
        self.candidates: list[Candidate] = []

        def add(name: str, models: list[str]):
            p = PROVIDERS[name]
            keys = [(k, os.getenv(k, "").strip()) for k in p["key_envs"]]
            keys = [(k, v) for k, v in keys if v] if p["key_envs"] else [("", "ollama")]
            for m in models:                       # model first, then keys: main model on
                for key_name, key in keys:         # every key before the backup model
                    self.candidates.append(Candidate(name, m, key_name, mk(p["base_url"], key)))

        p = PROVIDERS[primary]
        add(primary, [main_model or p["models"][0]] + [m for m in p["models"][1:] if m != main_model])
        for other in ("groq", "gemini"):
            if other != primary:
                add(other, PROVIDERS[other]["models"])
        if not self.candidates:
            raise LLMError("No LLM key found - add GROQ_API_KEY or GEMINI_API_KEY to the .env file")
        self.active = self.candidates[0]

    # what the dashboard shows: the provider/model that answered last
    @property
    def provider(self) -> str:
        return self.active.provider

    @property
    def model(self) -> str:
        return self.active.model

    def status(self) -> list[dict]:
        now = time.monotonic()
        return [{"name": c.label, "calls": c.calls,
                 "state": c.disabled or ("paused" if now < c.paused_until else "ready")}
                for c in self.candidates]

    def chat_json(self, system: str, user: str) -> tuple[dict, float]:
        """Returns (parsed JSON reply, latency in seconds). Fails over between candidates;
        raises LLMError only if every candidate failed (or is paused)."""
        t0 = time.monotonic()
        errors = []
        for c in self.candidates:
            now = time.monotonic()
            if not c.available(now):
                continue
            if now - t0 > 6.0:                     # don't keep the driver waiting forever
                errors.append("time budget used up")
                break
            try:
                data = self._call(c, system, user)
                c.calls += 1
                if c is not self.active:
                    print(f"[LLM] now using {c.label}", flush=True)
                self.active = c
                return data, time.monotonic() - t0
            except openai.RateLimitError as e:
                per_day = "per day" in str(e).lower()
                c.paused_until = time.monotonic() + (PAUSE_DAY_LIMIT if per_day else PAUSE_MINUTE_LIMIT)
                print(f"[LLM] {c.label}: {'DAILY' if per_day else 'per-minute'} limit reached "
                      f"-> trying the next one", flush=True)
                errors.append(f"{c.label}: rate limit")
            except (openai.AuthenticationError, openai.PermissionDeniedError,
                    openai.NotFoundError) as e:
                c.disabled = type(e).__name__
                print(f"[LLM] {c.label}: {type(e).__name__} -> disabled for this run", flush=True)
                errors.append(f"{c.label}: {type(e).__name__}")
            except (openai.APITimeoutError, openai.APIConnectionError) as e:
                c.failures += 1
                errors.append(f"{c.label}: {type(e).__name__}")
            except (json.JSONDecodeError, openai.BadRequestError) as e:
                c.failures += 1
                errors.append(f"{c.label}: {type(e).__name__}")
        raise LLMError("; ".join(errors) or "all LLM providers are paused (limits reached)")

    @staticmethod
    def _call(c: Candidate, system: str, user: str) -> dict:
        kwargs = dict(
            model=c.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            max_tokens=400,
        )
        if c.model.startswith("openai/gpt-oss"):
            kwargs["reasoning_effort"] = "low"   # these models "think" first; keep it short
        resp = c.client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        return json.loads(text)
