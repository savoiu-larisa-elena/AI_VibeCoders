"""Hand gestures (MediaPipe built-ins): Thumb_Up, Thumb_Down, Open_Palm,
Closed_Fist, Pointing_Up, Victory, ILoveYou.

Later these become driver replies: thumbs-up = "I'm OK", palm = "dismiss".
"""
from __future__ import annotations

import cv2

from ..events import Event, Severity
from .base import Detector, DetectorOutput, FrameContext


class GestureDetector(Detector):
    name = "gestures"
    needs = {"hands"}

    def __init__(self, config):
        super().__init__(config)
        self.cfg = config.gesture
        self._current: str | None = None
        self._since = 0.0
        self._emitted = False

    def process(self, ctx: FrameContext) -> DetectorOutput:
        out = DetectorOutput()
        res = ctx.hands
        best, score = None, 0.0
        if res and res.gestures:
            for hand in res.gestures:
                top = hand[0]
                if top.category_name != "None" and top.score > max(score, self.cfg.min_score):
                    best, score = top.category_name, top.score

        if best != self._current:
            self._current, self._since, self._emitted = best, ctx.ts, False
        held = ctx.ts - self._since if best else 0.0

        if best and held >= self.cfg.hold_seconds and not self._emitted:
            self._emitted = True
            out.events.append(Event("gesture", Severity.INFO, self.name, confidence=score,
                                    data={"gesture": best}))

        out.signals = {"gesture": best, "gesture_score": round(score, 2),
                       "hands": len(res.hand_landmarks) if res else 0}
        out.status = f"gesture: {best or '-'}" + (f" {score:.0%}" if best else "")
        return out

    def draw(self, frame, ctx, out):
        if not ctx.hands:
            return
        h, w = frame.shape[:2]
        for hand in ctx.hands.hand_landmarks:
            for p in hand:
                cv2.circle(frame, (int(p.x * w), int(p.y * h)), 3, (255, 0, 255), -1)
