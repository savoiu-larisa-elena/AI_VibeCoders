"""Voice output for the assistant: Groq Orpheus TTS, offline fallback."""
from __future__ import annotations

import hashlib
import io
import os
import threading
import time
from pathlib import Path

import pygame
import requests
from dotenv import load_dotenv

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")          # project .env, no matter which folder the app starts from
TTS_URL = "https://api.groq.com/openai/v1/audio/speech"
MODEL = "canopylabs/orpheus-v1-english"
CACHE_DIR = ROOT / "voice_cache"

# Sentences the app says WITHOUT the LLM (instant gesture replies). Together with the
# agent's fallback messages they are pre-generated and saved to voice_cache/, so the
# natural voice still works when the LLM or the network is down.
RULE_PHRASES = [
    "Okay, I'll stay quiet for a minute unless it's urgent.",
    "Okay, warnings are back on.",
]


def fixed_phrases(fallbacks: dict) -> list[str]:
    """All fixed sentences: agent FALLBACKS texts + RULE_PHRASES (deduplicated)."""
    texts = [v[0] for v in fallbacks.values()] + RULE_PHRASES
    return list(dict.fromkeys(texts))


DEFAULT_VOICE = "troy"


class Speaker:
    def __init__(self, voice: str | None = None, cooldown=8.0):
        self.api_key = os.environ.get("GROQ_API_KEY", "")
        # voice: argument > TTS_VOICE in .env > default
        voice = voice or os.environ.get("TTS_VOICE", "").strip() or DEFAULT_VOICE
        self.voice, self.cooldown = voice, cooldown
        # Groq API state: a permanent error (terms not accepted, bad key) stops further
        # requests for this run; a rate limit (429) pauses them for a minute.
        self.api_disabled: str | None = None
        self.api_paused_until = 0.0
        self.cache: dict[str, bytes] = {}      # text -> wav bytes
        self.last_said: dict[str, float] = {}
        self.cur_priority = 0
        self.channel = None
        self.pending = None                    # (priority, text)
        self.persist: set[str] = set()         # only these are saved to voice_cache/ on disk
        self.cv = threading.Condition()
        CACHE_DIR.mkdir(exist_ok=True)
        pygame.mixer.init()
        print(f"[VOICE] key loaded: {bool(self.api_key)} | voice: {self.voice}", flush=True)
        threading.Thread(target=self._worker, daemon=True).start()

    def prewarm(self, phrases):
        """Genereaza din timp audio pentru frazele fixe (alerte critice).
        Fixed phrases are saved to disk; unique LLM sentences stay in memory only,
        so voice_cache/ doesn't fill up with one .wav per alert."""
        self.persist.update(phrases)
        threading.Thread(target=lambda: [self._tts(p) for p in phrases],
                         daemon=True).start()

    def say(self, text: str, priority: int = 1):
        """priority: 1=info, 2=warning, 3=critical. Apelabil din orice thread."""
        print(f"[VOICE] say prio={priority} {text!r}", flush=True)
        if not text:
            return
        now = time.monotonic()
        with self.cv:
            if priority < 3 and now - self.last_said.get(text, 0) < self.cooldown:
                print("[VOICE] dropped: cooldown", flush=True)
                return
            if self.pending and self.pending[0] > priority:
                print("[VOICE] dropped: higher priority pending", flush=True)
                return
            self.pending = (priority, text)
            if self.channel and self.channel.get_busy() and priority > self.cur_priority:
                self.channel.stop()            # alerta mai grava taie vorbirea curenta
            self.cv.notify()

    def is_speaking(self) -> bool:
        return self.pending is not None or bool(self.channel and self.channel.get_busy())

    def _disk_path(self, text):
        h = hashlib.sha1(f"{self.voice}|{text}".encode("utf-8")).hexdigest()[:16]
        return CACHE_DIR / f"{h}.wav"

    def _tts(self, text):
        if text in self.cache:
            return self.cache[text]
        path = self._disk_path(text)
        if path.exists():
            self.cache[text] = path.read_bytes()
            return self.cache[text]
        if not self.api_key:
            print("[VOICE] no GROQ_API_KEY found (check .env)", flush=True)
            return None
        if self.api_disabled or time.monotonic() < self.api_paused_until:
            return None                          # known problem: go straight to fallback voice
        for attempt in range(2):
            try:
                r = requests.post(
                    TTS_URL,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": MODEL, "input": text, "voice": self.voice,
                          "response_format": "wav"},
                    timeout=8,
                )
                if r.ok:
                    self.cache[text] = r.content
                    if text in self.persist:
                        path.write_bytes(r.content)
                    return r.content
                self._api_error(r)
                if r.status_code in (400, 401, 403, 429):   # retry nu ajuta
                    return None
            except Exception as e:
                print(f"[VOICE] TTS error (attempt {attempt + 1}): {e}", flush=True)
        return None

    def _api_error(self, r):
        """Explain the Groq error once, and stop wasting requests on permanent ones."""
        try:
            err = r.json().get("error", {})
        except ValueError:
            err = {}
        code, msg = err.get("code", ""), err.get("message", r.text[:200])
        if code == "model_terms_required":
            self.api_disabled = "terms"
            print("[VOICE] The Groq AI voice needs its terms accepted once (by the owner of the\n"
                  "        API key). Open https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english\n"
                  "        accept the terms, then restart. Using the offline voice until then.", flush=True)
        elif r.status_code in (401, 403):
            self.api_disabled = "auth"
            print(f"[VOICE] Groq rejected the API key ({r.status_code}): {msg}\n"
                  "        Using the offline voice for this run.", flush=True)
        elif r.status_code == 429:
            self.api_paused_until = time.monotonic() + 60
            print("[VOICE] Groq voice rate limit reached - offline voice for the next 60 s.", flush=True)
        elif r.status_code == 400 and "voice" in msg.lower():
            self.api_disabled = "voice"
            print(f"[VOICE] Voice '{self.voice}' not accepted: {msg}\n"
                  "        Set a valid TTS_VOICE in .env. Using the offline voice.", flush=True)
        else:
            print(f"[VOICE] TTS {r.status_code}: {msg}", flush=True)

    def _fallback(self, text):
        print("[VOICE] using offline fallback voice", flush=True)
        if pyttsx3 is None:
            return
        try:
            eng = pyttsx3.init()
            eng.say(text)
            eng.runAndWait()
        except Exception as e:
            print(f"[VOICE] fallback error: {e}", flush=True)

    def _worker(self):
        while True:
            with self.cv:
                while self.pending is None:
                    self.cv.wait()
                priority, text = self.pending
                self.pending = None
            self.cur_priority = priority
            self.last_said[text] = time.monotonic()
            audio = self._tts(text)
            if audio:
                self.channel = pygame.mixer.Sound(file=io.BytesIO(audio)).play()
                while self.channel and self.channel.get_busy():
                    time.sleep(0.05)
            else:
                self._fallback(text)
            self.cur_priority = 0