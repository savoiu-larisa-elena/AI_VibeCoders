"""Small helpers shared by the face-based detectors."""
from __future__ import annotations

import math

import numpy as np


def matrix_to_euler(m) -> tuple[float, float, float]:
    """(pitch, yaw, roll) in degrees from MediaPipe's 4x4 facial transformation matrix."""
    r = np.asarray(m)[:3, :3]
    sy = math.sqrt(r[2, 1] ** 2 + r[2, 2] ** 2)
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(-r[2, 0], sy))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return pitch, yaw, roll


def blendshapes(face_result) -> dict[str, float]:
    """{name: score} for the driver's face, e.g. 'eyeBlinkLeft', 'eyeLookDownRight', 'jawOpen'."""
    if not face_result or not face_result.face_blendshapes:
        return {}
    return {c.category_name: c.score for c in face_result.face_blendshapes[0]}


def face_box(landmarks, w: int, h: int) -> tuple[int, int, int, int]:
    xs = [p.x for p in landmarks]
    ys = [p.y for p in landmarks]
    return int(min(xs) * w), int(min(ys) * h), int(max(xs) * w), int(max(ys) * h)


def put_driver_first(face_result) -> None:
    """With several faces in view, the driver = the LARGEST face (closest to camera).
    Reorders the result in place so index 0 is always the driver."""
    if not face_result or len(face_result.face_landmarks) < 2:
        return

    def area(lms):
        xs = [p.x for p in lms]
        ys = [p.y for p in lms]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    order = sorted(range(len(face_result.face_landmarks)),
                   key=lambda i: area(face_result.face_landmarks[i]), reverse=True)
    face_result.face_landmarks = [face_result.face_landmarks[i] for i in order]
    if face_result.face_blendshapes:
        face_result.face_blendshapes = [face_result.face_blendshapes[i] for i in order]
    if face_result.facial_transformation_matrixes:
        face_result.facial_transformation_matrixes = [
            face_result.facial_transformation_matrixes[i] for i in order]


class PitchBaseline:
    """Learns the driver's neutral head pitch during calibration; returns pitch relative to it.
    Negative = head tilted DOWN (flip config.head_pose.pitch_sign if it comes out reversed)."""

    def __init__(self, sign: int = 1):
        self.sign = sign
        self.reset()

    def reset(self):
        self._samples: list[float] = []
        self.baseline: float | None = None

    def relative(self, face_result, calibrating: bool) -> float | None:
        if not face_result or not face_result.facial_transformation_matrixes:
            return None
        pitch = matrix_to_euler(face_result.facial_transformation_matrixes[0])[0]
        if calibrating:
            self._samples.append(pitch)
        elif self.baseline is None and self._samples:
            self.baseline = float(np.median(self._samples))
        return (pitch - (self.baseline or 0.0)) * self.sign
