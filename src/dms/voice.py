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

load_dotenv()
TTS_URL = "https://api.groq.com/openai/v1/audio/speech"
MODEL = "canopylabs/orpheus-v1-english"
CACHE_DIR = Path(__file__).resolve().parents[2] / "voice_cache"


class Speaker:
    def __init__(self, voice="troy", cooldown=8.0):
        self.api_key = os.environ.get("GROQ_API_KEY", "")
        self.voice, self.cooldown = voice, cooldown
        self.cache: dict[str, bytes] = {}      # text -> wav bytes
        self.last_said: dict[str, float] = {}
        self.cur_priority = 0
        self.channel = None
        self.pending = None                    # (priority, text)
        self.cv = threading.Condition()
        CACHE_DIR.mkdir(exist_ok=True)
        pygame.mixer.init()
        print(f"[VOICE] key loaded: {bool(self.api_key)}", flush=True)
        threading.Thread(target=self._worker, daemon=True).start()

    def prewarm(self, phrases):
        """Genereaza din timp audio pentru frazele fixe (alerte critice)."""
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
                    path.write_bytes(r.content)
                    return r.content
                print(f"[VOICE] TTS {r.status_code}: {r.text[:300]}", flush=True)
                if r.status_code in (400, 401, 403, 429):   # retry nu ajuta
                    return None
            except Exception as e:
                print(f"[VOICE] TTS error (attempt {attempt + 1}): {e}", flush=True)
        return None

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