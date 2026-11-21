"""Hands on the steering wheel - with a VIRTUAL wheel, so it can be demoed on a laptop.

A real car's driver camera sees the steering wheel; a laptop webcam doesn't. So we draw
the UPPER half of a wheel in the lower-LEFT corner - in front of the driver, who sits on
the left (camera on the roof between the seats). The demo driver holds their hands on
the drawn rim, as if gripping a real wheel.

    at least one hand on the wheel  -> OK (one hand is fine, e.g. while gesturing)
    no hand for 2 s                 -> warning  "hands_off_wheel"
    no hand for 5 s                 -> critical "hands_off_wheel"

It only starts checking ("armed") after hands have been seen on the wheel once, so it
doesn't complain before the demo starts. Recalibrating disarms it again.
Always on by default (config.wheel.always_on); hide it with:  --no-wheel
"""
from __future__ import annotations

import cv2
import numpy as np

from ..events import Event, Severity
from .base import Detector, DetectorOutput, FrameContext

PALM_POINTS = (0, 9)   # wrist + middle-finger knuckle = where the hand grips


class WheelDetector(Detector):
    name = "wheel"
    needs = {"hands"}
    order = 30

    def __init__(self, config):
        super().__init__(config)
        self.cfg = config.wheel
        self.recalibrate()

    def recalibrate(self, soft: bool = False):
        if soft:
            return
        self.armed = False
        self._off_since: float | None = None
        self._level = 0          # 0 = no alert yet, 1 = warned, 2 = critical sent
        self._on = 0

    def _on_wheel(self, lm) -> bool:
        """Hand grips the wheel: wrist or knuckle inside the (slightly enlarged) wheel."""
        cx, cy = self.cfg.center
        m = self.cfg.grip_margin
        rx, ry = self.cfg.size[0] / 2 * m, self.cfg.size[1] / 2 * m
        return any(((lm[i].x - cx) / rx) ** 2 + ((lm[i].y - cy) / ry) ** 2 <= 1.0
                   for i in PALM_POINTS)

    def process(self, ctx: FrameContext) -> DetectorOutput:
        out = DetectorOutput()
        hands = ctx.hands.hand_landmarks if ctx.hands else []
        self._on = sum(1 for lm in hands if self._on_wheel(lm))
        off_for = 0.0

        if self._on:
            self.armed = True
            self._off_since, self._level = None, 0
        elif self.armed and not (ctx.calibrating and not ctx.soft_calibrating):
            self._off_since = self._off_since or ctx.ts
            off_for = ctx.ts - self._off_since
            if off_for >= self.cfg.warn_seconds and self._level < 1:
                self._level = 1
                out.events.append(Event("hands_off_wheel", Severity.WARNING, self.name,
                                        data={"seconds": round(off_for, 1)}))
            if off_for >= self.cfg.critical_seconds and self._level < 2:
                self._level = 2
                out.events.append(Event("hands_off_wheel", Severity.CRITICAL, self.name,
                                        data={"seconds": round(off_for, 1)}))

        out.signals = {"hands_on_wheel": self._on, "armed": self.armed,
                       "off_seconds": round(off_for, 1)}
        if not self.armed:
            out.status = "wheel: put your hands on the wheel to start"
        elif self._on:
            out.status = f"wheel: {self._on} hand{'s' if self._on > 1 else ''} on"
        else:
            out.status = f"wheel: HANDS OFF {off_for:.1f}s"
        return out

    def draw(self, frame, ctx, out):
        h, w = frame.shape[:2]
        center = (int(self.cfg.center[0] * w), int(self.cfg.center[1] * h))
        axes = (int(self.cfg.size[0] * w / 2), int(self.cfg.size[1] * h / 2))
        if not self.armed:
            color = (150, 150, 150)
        elif self._on:
            color = (0, 200, 0)
        else:
            color = (0, 0, 255) if self._level >= 2 else (0, 165, 255)
        overlay = frame.copy()
        thick = max(8, h // 60)
        cv2.ellipse(overlay, center, axes, 0, 180, 360, color, thick)           # upper rim only
        hub = (center[0], center[1] - axes[1] // 4)
        for ang in (200, 340):                                                 # two spokes
            rad = np.deg2rad(ang)
            rim = (int(center[0] + axes[0] * np.cos(rad)), int(center[1] + axes[1] * np.sin(rad)))
            cv2.line(overlay, hub, rim, color, max(4, thick // 2))
        cv2.circle(overlay, hub, max(10, axes[1] // 6), color, -1)              # hub
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)                     # see-through
        label_y = max(20, center[1] - axes[1] - 12)
        cv2.putText(frame, "STEERING WHEEL", (max(5, center[0] - 72), label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
