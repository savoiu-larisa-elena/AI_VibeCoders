"""Head pose -> where is the driver looking? (left / right / down / up / road)

Uses MediaPipe's facial transformation matrix (a 3D rotation of the head).
Angles are measured RELATIVE to the calibrated "looking at the road" pose,
so camera placement (laptop below the face, off to the side...) doesn't matter.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..events import Event, Severity
from ..face_utils import blendshapes, matrix_to_euler
from .base import Detector, DetectorOutput, FrameContext


class HeadPoseDetector(Detector):
    name = "head_pose"
    needs = {"face"}

    def __init__(self, config):
        super().__init__(config)
        self.cfg = config.head_pose
        self.recalibrate()

    def recalibrate(self, soft: bool = False):
        self._calib: list[tuple[float, float]] = []
        self._needs_baseline = True
        if soft:                       # keep the old "road" pose until the new one is learned
            return
        self.baseline: tuple[float, float] | None = None
        self._away_since: float | None = None
        self._alerted = False

    def process(self, ctx: FrameContext) -> DetectorOutput:
        out = DetectorOutput()
        if not ctx.face or not ctx.face.facial_transformation_matrixes:
            out.status = "head_pose: no face"
            return out

        pitch, yaw, roll = matrix_to_euler(ctx.face.facial_transformation_matrixes[0])

        if ctx.calibrating:
            self._calib.append((pitch, yaw))
        elif self._needs_baseline and self._calib:
            self.baseline = tuple(np.median(np.array(self._calib), axis=0))
            self._needs_baseline, self._calib = False, []

        bp, by = self.baseline or (0.0, 0.0)
        rel_pitch = (pitch - bp) * self.cfg.pitch_sign
        # verified: on a MIRRORED frame, +yaw = driver turned to THEIR left
        mirror_fix = 1 if self.config.camera.mirror else -1
        rel_yaw = (yaw - by) * self.cfg.yaw_sign * mirror_fix

        # eyes looking down (people often glance at a phone with the eyes, not the head).
        # Ignored while blinking: closed lids can fake a "look down" score.
        bs = blendshapes(ctx.face)
        gaze_down = (bs.get("eyeLookDownLeft", 0) + bs.get("eyeLookDownRight", 0)) / 2
        blink = (bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2
        eyes_down = gaze_down > self.cfg.gaze_down_threshold and blink < 0.5

        if rel_yaw > self.cfg.yaw_threshold_deg:
            direction = "left"
        elif rel_yaw < -self.cfg.yaw_threshold_deg:
            direction = "right"
        elif rel_pitch < -self.cfg.pitch_down_threshold_deg or eyes_down:
            direction = "down"
        elif rel_pitch > self.cfg.pitch_up_threshold_deg:
            direction = "up"
        else:
            direction = "road"

        away_for = 0.0
        # during a full calibration nothing is timed; during a soft refresh monitoring continues
        if direction != "road" and not (ctx.calibrating and not ctx.soft_calibrating):
            self._away_since = self._away_since or ctx.ts
            away_for = ctx.ts - self._away_since
        else:
            self._away_since = None
            self._alerted = False

        if away_for >= self.cfg.away_alert_seconds and not self._alerted:
            self._alerted = True
            sev = Severity.CRITICAL if direction == "down" else Severity.WARNING
            out.events.append(Event("distraction", sev, self.name,
                                    data={"direction": direction, "seconds": round(away_for, 2)}))

        out.signals = {
            "pitch": round(rel_pitch, 1), "yaw": round(rel_yaw, 1), "roll": round(roll, 1),
            "looking": direction, "away_seconds": round(away_for, 2),
            "gaze_down": round(gaze_down, 2),
        }
        out.status = (f"yaw {rel_yaw:+.0f}  pitch {rel_pitch:+.0f}  roll {roll:+.0f}  "
                      f"gazeDown {gaze_down:.2f} -> "
                      f"{direction.upper()}" + (f" ({away_for:.1f}s)" if away_for else ""))
        return out

    def draw(self, frame, ctx, out):
        if not ctx.face or not ctx.face.face_landmarks:
            return
        h, w = frame.shape[:2]
        nose = ctx.face.face_landmarks[0][1]
        start = (int(nose.x * w), int(nose.y * h))
        # arrow showing head direction (debug aid for checking yaw/pitch signs)
        dx = -out.signals.get("yaw", 0) * 3
        dy = -out.signals.get("pitch", 0) * 3
        end = (int(start[0] + dx), int(start[1] + dy))
        color = (0, 255, 0) if out.signals.get("looking") == "road" else (0, 165, 255)
        cv2.arrowedLine(frame, start, end, color, 3, tipLength=0.3)
