"""Voice output for the assistant: Groq Orpheus TTS, offline fallback."""
from __future__ import annotations

import io
import os
import threading
import time

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
        pygame.mixer.init()
        threading.Thread(target=self._worker, daemon=True).start()

    def prewarm(self, phrases):
        """Genereaza din timp audio pentru frazele fixe (alerte critice)."""
        threading.Thread(target=lambda: [self._tts(p) for p in phrases],
                         daemon=True).start()

    def say(self, text: str, priority: int = 1):
        """priority: 1=info, 2=warning, 3=critical. Apelabil din orice thread."""
        if not text:
            return
        now = time.monotonic()
        with self.cv:
            if priority < 3 and now - self.last_said.get(text, 0) < self.cooldown:
                return
            if self.pending and self.pending[0] > priority:
                return
            self.pending = (priority, text)
            if self.channel and self.channel.get_busy() and priority > self.cur_priority:
                self.channel.stop()            # alerta mai grava taie vorbirea curenta
            self.cv.notify()

    def is_speaking(self) -> bool:
        return self.pending is not None or bool(self.channel and self.channel.get_busy())

    def _tts(self, text):
        if text in self.cache:
            return self.cache[text]
        if not self.api_key:
            return None
        try:
            r = requests.post(
                TTS_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": MODEL, "input": text, "voice": self.voice,
                      "response_format": "wav"},
                timeout=5,
            )
            if not r.ok:
                print(f"[VOICE] TTS {r.status_code}: {r.text[:300]}")
                return None
            self.cache[text] = r.content
            return r.content
        except Exception as e:
            print(f"[VOICE] TTS error: {e}")
            return None

    def _fallback(self, text):
        if pyttsx3 is None:
            return
        try:
            eng = pyttsx3.init()
            eng.say(text)
            eng.runAndWait()
        except Exception as e:
            print(f"[VOICE] fallback error: {e}")

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