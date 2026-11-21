"""Runs the selected detectors on each frame and collects signals + events.

    signals, events = pipeline.process(frame)

`pipeline.timing` holds how long each part of the last frame took (ms), so the
dashboard can show where the time goes if the FPS drops.
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
        names = list(dict.fromkeys(detector_names))
        # driver + passenger tracking comes for free whenever face detection runs
        if "occupants" not in names and any("face" in DETECTORS[n]().needs for n in names):
            names.append("occupants")
        self.detectors = [DETECTORS[n]()(self.config) for n in names]
        # run order matters: phone/drink boxes must exist before gestures checks them
        self.detectors.sort(key=lambda d: d.order)
        needed = set().union(*(d.needs for d in self.detectors)) if self.detectors else set()
        self.providers = {name: PROVIDERS[name]() for name in needed}
        self.cooldown = Cooldown(self.config.event_cooldown_seconds)
        # calibration counts only time with the driver's face visible, so starting the app
        # while nobody sits in front of the camera doesn't skip calibration
        self._calib_elapsed = 0.0
        self._last_now: float | None = None
        self._last_ts_ms = -1
        self._frame_i = 0
        self._last_results: dict = {}
        self.outputs: dict[str, DetectorOutput] = {}
        self.timing: dict[str, float] = {}

    @property
    def calibrating(self) -> bool:
        return self._calib_elapsed < self.config.calibration.seconds

    def recalibrate(self):
        self._calib_elapsed = 0.0
        for d in self.detectors:
            d.recalibrate()
        for p in self.providers.values():
            if hasattr(p, "reset"):
                p.reset()       # e.g. face provider re-locks onto the driver

    def process(self, frame) -> tuple[dict, list[Event]]:
        now = time.monotonic()
        ts_ms = max(int(now * 1000), self._last_ts_ms + 1)  # MediaPipe needs increasing timestamps
        self._last_ts_ms = ts_ms
        self._frame_i += 1
        timing: dict[str, float] = {}

        dt = now - self._last_now if self._last_now is not None else 0.0
        self._last_now = now
        ctx = FrameContext(frame=frame, ts=now, calibrating=self.calibrating)
        if self.providers:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            for name, provider in self.providers.items():
                t = time.perf_counter()
                every = getattr(provider, "every_n_frames", 1)
                # some models (hands) don't need every frame: reuse the last result in between
                if every > 1 and self._frame_i % every and name in self._last_results:
                    result = self._last_results[name]
                else:
                    result = provider.process(mp_image, ts_ms)
                    self._last_results[name] = result
                setattr(ctx, name, result)
                timing[f"model_{name}"] = (time.perf_counter() - t) * 1000
        if ctx.calibrating:
            face_visible = "face" not in self.providers or bool(ctx.face and ctx.face.face_landmarks)
            if face_visible:
                self._calib_elapsed += min(dt, 0.5)

        signals: dict = {"calibrating": ctx.calibrating}
        events: list[Event] = []
        t = time.perf_counter()
        for d in self.detectors:
            out = d.process(ctx)
            self.outputs[d.name] = out
            signals[d.name] = out.signals
            # cooldown per event type; gestures per gesture name (thumbs-up shouldn't block palm)
            events += [e for e in out.events
                       if self.cooldown.ready(e.type + str(e.data.get("gesture", "")), now)]
        timing["detectors"] = (time.perf_counter() - t) * 1000
        self.timing = timing
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
