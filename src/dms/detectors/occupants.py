"""Who is in the car: the driver and any passengers.

Runs automatically whenever face detection is on. The driver is chosen by DriverSelector
(seat position learned at calibration); every other face is a passenger.
  * dashboard: driver present yes/no + number of passengers
  * LLM: knows a passenger is there, so at high fatigue it can ask the passenger to help
  * camera overlay: green DRIVER box, blue PASSENGER boxes with eye/mouth points
"""
from __future__ import annotations

import cv2

from ..events import Event, Severity
from ..face_utils import face_box
from .base import Detector, DetectorOutput, FrameContext

EYES_MOUTH = [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380, 78, 308, 13, 14]


class OccupantsDetector(Detector):
    name = "occupants"
    needs = {"face"}
    order = 5            # first: cheap, and other parts may read who is present
    HOLD_SECONDS = 2.0   # passenger count must be stable this long before it "changes"

    def __init__(self, config):
        super().__init__(config)
        self._stable = 0
        self._candidate = 0
        self._since = 0.0

    def process(self, ctx: FrameContext) -> DetectorOutput:
        out = DetectorOutput()
        face = ctx.face
        driver = bool(face and face.face_landmarks)
        passengers = len(getattr(face, "passengers", []) or []) if face else 0

        # debounce: count must stay the same for HOLD_SECONDS (faces flicker in/out)
        if passengers != self._candidate:
            self._candidate, self._since = passengers, ctx.ts
        elif passengers != self._stable and ctx.ts - self._since >= self.HOLD_SECONDS:
            before, self._stable = self._stable, passengers
            if passengers > before:
                out.events.append(Event("passenger_joined", Severity.INFO, self.name,
                                        data={"passengers": passengers}))
            else:
                out.events.append(Event("passenger_left", Severity.INFO, self.name,
                                        data={"passengers": passengers}))

        out.signals = {"driver_present": driver, "passengers": self._stable}
        out.status = (f"driver {'present' if driver else 'NOT VISIBLE'} | "
                      f"passengers: {self._stable}")
        return out

    def draw(self, frame, ctx, out):
        face = ctx.face
        if not face:
            return
        h, w = frame.shape[:2]
        if face.face_landmarks:
            self._box(frame, face.face_landmarks[0], w, h, "DRIVER", (0, 200, 0))
        for lms in getattr(face, "passengers", []) or []:
            self._box(frame, lms, w, h, "PASSENGER", (255, 170, 60))
            for i in EYES_MOUTH:
                cv2.circle(frame, (int(lms[i].x * w), int(lms[i].y * h)), 2, (255, 170, 60), -1)

    @staticmethod
    def _box(frame, lms, w, h, label, color):
        x1, y1, x2, y2 = face_box(lms, w, h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
