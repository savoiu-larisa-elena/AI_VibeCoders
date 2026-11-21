"""Drowsiness: eyes closed, PERCLOS, blinks, yawns.

EAR  = Eye Aspect Ratio  - drops toward 0 when the eye closes.
MAR  = Mouth Aspect Ratio - rises when the mouth opens wide (yawn).
PERCLOS = % of time eyes were closed in the last N seconds (the industry fatigue metric).

The detector calibrates on YOUR open-eye EAR during the first seconds,
because eye shape differs per person and with glasses.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from ..events import Event, Severity
from ..face_utils import PitchBaseline
from .base import Detector, DetectorOutput, FrameContext

# MediaPipe face mesh indices: [corner, top1, top2, corner, bottom2, bottom1]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 308, 13, 14]  # left corner, right corner, upper inner lip, lower inner lip


def _pts(landmarks, idx, w, h) -> np.ndarray:
    return np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in idx])


def eye_aspect_ratio(p: np.ndarray) -> float:
    vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    horizontal = 2.0 * np.linalg.norm(p[0] - p[3])
    return float(vertical / horizontal) if horizontal else 0.0


def mouth_aspect_ratio(p: np.ndarray) -> float:
    width = np.linalg.norm(p[0] - p[1])
    return float(np.linalg.norm(p[2] - p[3]) / width) if width else 0.0


class DrowsinessDetector(Detector):
    name = "drowsiness"
    needs = {"face"}

    def __init__(self, config):
        super().__init__(config)
        self.cfg = config.drowsiness
        self.pitch = PitchBaseline(sign=config.head_pose.pitch_sign)
        self.recalibrate()

    def recalibrate(self, soft: bool = False):
        """Full: forget everything (new driver). Soft: learn a fresh open-eye baseline but
        keep the old one until then, and keep PERCLOS/blink history (fatigue trend)."""
        self.pitch.reset(soft)
        self._calib_ears: list[float] = []
        self._needs_baseline = True
        if soft:
            return
        self.baseline_ear: float | None = None
        self._closed_since: float | None = None
        self._yawn_since: float | None = None
        self._was_closed = False
        self._blinks: deque[float] = deque()
        self._history: deque[tuple[float, bool]] = deque()  # (ts, closed)
        self._alerted_closed = False
        self._alerted_yawn = False

    def process(self, ctx: FrameContext) -> DetectorOutput:
        out = DetectorOutput()
        if not ctx.face or not ctx.face.face_landmarks:
            out.status = "drowsiness: no face"
            self._closed_since = None
            return out

        lm = ctx.face.face_landmarks[0]
        h, w = ctx.frame.shape[:2]
        ear = (eye_aspect_ratio(_pts(lm, LEFT_EYE, w, h)) +
               eye_aspect_ratio(_pts(lm, RIGHT_EYE, w, h))) / 2
        mar = mouth_aspect_ratio(_pts(lm, MOUTH, w, h))

        # --- calibration: learn this driver's normal open-eye EAR ---
        if ctx.calibrating:
            self._calib_ears.append(ear)
        elif self._needs_baseline and self._calib_ears:
            self.baseline_ear = float(np.median(self._calib_ears))
            self._needs_baseline, self._calib_ears = False, []

        threshold = (self.baseline_ear * self.cfg.ear_closed_ratio
                     if self.baseline_ear else self.cfg.ear_fallback_threshold)
        rel_pitch = self.pitch.relative(ctx.face, ctx.calibrating) or 0.0
        # head tilted down -> eyelids look closed from the camera's angle; don't trust EAR
        head_down = rel_pitch < self.cfg.head_down_ignore_pitch_deg
        closed = ear < threshold and not head_down

        # --- blinks (closed -> open transition) ---
        if self._was_closed and not closed:
            self._blinks.append(ctx.ts)
        self._was_closed = closed
        while self._blinks and ctx.ts - self._blinks[0] > 60:
            self._blinks.popleft()

        # --- PERCLOS over a sliding window ---
        self._history.append((ctx.ts, closed))
        while self._history and ctx.ts - self._history[0][0] > self.cfg.perclos_window_seconds:
            self._history.popleft()
        perclos = sum(c for _, c in self._history) / len(self._history)

        # --- continuous eye closure ---
        closed_for = 0.0
        if closed:
            self._closed_since = self._closed_since or ctx.ts
            closed_for = ctx.ts - self._closed_since
        else:
            self._closed_since = None
            self._alerted_closed = False

        if closed_for >= self.cfg.closed_alert_seconds and not self._alerted_closed:
            self._alerted_closed = True
            out.events.append(Event("eyes_closed", Severity.CRITICAL, self.name,
                                    data={"seconds": round(closed_for, 2), "ear": round(ear, 3)}))

        window_full = ctx.ts - self._history[0][0] > self.cfg.perclos_window_seconds * 0.9
        if window_full and perclos >= self.cfg.perclos_alert:
            out.events.append(Event("fatigue_perclos", Severity.WARNING, self.name,
                                    data={"perclos": round(perclos, 2)}))

        # --- yawn ---
        if mar > self.cfg.yawn_mar_threshold:
            self._yawn_since = self._yawn_since or ctx.ts
            if ctx.ts - self._yawn_since >= self.cfg.yawn_min_seconds and not self._alerted_yawn:
                self._alerted_yawn = True
                out.events.append(Event("yawn", Severity.WARNING, self.name,
                                        data={"mar": round(mar, 2)}))
        else:
            self._yawn_since = None
            self._alerted_yawn = False

        out.signals = {
            "ear": round(ear, 3),
            "ear_threshold": round(threshold, 3),
            "eyes_closed": closed,
            "head_down": head_down,
            "pitch": round(rel_pitch, 1),
            "closed_seconds": round(closed_for, 2),
            "perclos": round(perclos, 3),
            "blinks_per_min": len(self._blinks),
            "mar": round(mar, 3),
        }
        state = "HEAD DOWN (EAR ignored)" if head_down else ("CLOSED" if closed else "open")
        out.status = (f"EAR {ear:.2f} (thr {threshold:.2f}) {state} | "
                      f"PERCLOS {perclos:.0%} | blinks/min {len(self._blinks)} | MAR {mar:.2f}")
        return out

    def draw(self, frame, ctx, out):
        if not ctx.face or not ctx.face.face_landmarks:
            return
        lm = ctx.face.face_landmarks[0]
        h, w = frame.shape[:2]
        if out.signals.get("head_down"):
            color = (0, 255, 255)      # yellow: looking down
        elif out.signals.get("eyes_closed"):
            color = (0, 0, 255)        # red: closed
        else:
            color = (0, 255, 0)        # green: open
        for idx in (LEFT_EYE, RIGHT_EYE, MOUTH):
            for x, y in _pts(lm, idx, w, h).astype(int):
                cv2.circle(frame, (x, y), 2, color, -1)
