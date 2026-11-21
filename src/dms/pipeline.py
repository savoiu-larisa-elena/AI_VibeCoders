"""Runs the selected detectors on each frame and collects signals + events.

This is the piece the LLM agent and dashboard will plug into later:
    signals, events = pipeline.process(frame)
"""
from __future__ import annotations

import time

import cv2
import mediapipe as mp

from .config import Config
from .detectors import DETECTORS
from .detectors.base import DetectorOutput, FrameContext
from .events import Cooldown, Event
from .providers import PROVIDERS


class Pipeline:
    def __init__(self, detector_names: list[str], config: Config | None = None):
        self.config = config or Config()
        unknown = set(detector_names) - set(DETECTORS)
        if unknown:
            raise ValueError(f"Unknown detectors {unknown}. Available: {list(DETECTORS)}")
        self.detectors = [DETECTORS[n]()(self.config) for n in detector_names]
        needed = set().union(*(d.needs for d in self.detectors)) if self.detectors else set()
        self.providers = {name: PROVIDERS[name]() for name in needed}
        self.cooldown = Cooldown(self.config.event_cooldown_seconds)
        self._t0 = time.monotonic()
        self._last_ts_ms = -1
        self.outputs: dict[str, DetectorOutput] = {}

    @property
    def calibrating(self) -> bool:
        return time.monotonic() - self._t0 < self.config.calibration.seconds

    def recalibrate(self):
        self._t0 = time.monotonic()
        for d in self.detectors:
            d.recalibrate()

    def process(self, frame) -> tuple[dict, list[Event]]:
        now = time.monotonic()
        ts_ms = max(int(now * 1000), self._last_ts_ms + 1)  # MediaPipe needs increasing timestamps
        self._last_ts_ms = ts_ms

        ctx = FrameContext(frame=frame, ts=now, calibrating=self.calibrating)
        if self.providers:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            for name, provider in self.providers.items():
                setattr(ctx, name, provider.process(mp_image, ts_ms))

        signals: dict = {"calibrating": ctx.calibrating}
        events: list[Event] = []
        for d in self.detectors:
            out = d.process(ctx)
            self.outputs[d.name] = out
            signals[d.name] = out.signals
            # cooldown per event type; gestures per gesture name (thumbs-up shouldn't block palm)
            events += [e for e in out.events
                       if self.cooldown.ready(e.type + str(e.data.get("gesture", "")), now)]
        self._ctx = ctx
        return signals, events

    def draw(self, frame):
        for d in self.detectors:
            if d.name in self.outputs:
                d.draw(frame, self._ctx, self.outputs[d.name])

    def close(self):
        for d in self.detectors:
            d.close()
        for p in self.providers.values():
            p.close()
