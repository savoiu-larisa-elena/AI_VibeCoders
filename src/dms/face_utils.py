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


def driver_zone(face_result, w: int, h: int, scale: float = 1.6):
    """Area that belongs to the DRIVER: their face box widened by `scale` face-widths on
    each side, from well above the head (raised hand) down to the bottom (hands, lap, phone).
    Objects and hands outside it belong to someone else (passenger). None = no driver."""
    if not face_result or not face_result.face_landmarks:
        return None
    x1, y1, x2, y2 = face_box(face_result.face_landmarks[0], w, h)
    fw, fh = x2 - x1, y2 - y1
    return (max(0, int(x1 - scale * fw)), max(0, int(y1 - 1.5 * fh)),   # room for a raised hand
            min(w, int(x2 + scale * fw)), h)


def in_box(x: float, y: float, box) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _reorder(face_result, order: list[int]) -> None:
    """Keep only the faces in `order` (in that order), across all result lists."""
    face_result.face_landmarks = [face_result.face_landmarks[i] for i in order]
    if face_result.face_blendshapes:
        face_result.face_blendshapes = [face_result.face_blendshapes[i] for i in order]
    if face_result.facial_transformation_matrixes:
        face_result.facial_transformation_matrixes = [
            face_result.facial_transformation_matrixes[i] for i in order]


class DriverSelector:
    """Decides which face is the DRIVER when several people are in view.

    In a car the driver sits in a fixed place in the camera image. So:
      * first lock-on: the large face closest to the driver's seat (seat_x; left by default)
      * after that: always follow the face closest to the driver's remembered position
      * if no face is near that position (driver turned away, passenger leaning in),
        report "no driver face" instead of switching to the passenger
      * if the driver is lost for `lost_reset_s`, re-lock - but only onto a face near the
        middle of the image (where the driver sits), never onto a passenger at the side
      * recalibrating (reset) re-locks immediately
    Result is changed in place:
      face_result.face_landmarks / blendshapes / matrices -> ONLY the driver (or empty)
      face_result.passengers                              -> landmarks of every other face
    so all driver detectors keep working on index 0 and never see a passenger.
    """

    def __init__(self, max_jump: float = 1.2, lost_reset_s: float = 10.0,
                 relock_center: float = 0.25, seat_x: float = 0.5):
        self.max_jump = max_jump          # in face-widths from the remembered position
        self.lost_reset_s = lost_reset_s
        self.relock_center = relock_center  # auto re-lock only if |face x - seat_x| < this
        # where the driver's seat is horizontally in the (mirrored) image: 0 = left edge,
        # 0.5 = middle. With the camera on the roof between the seats, the driver is on
        # the left and the passenger on the right (config.camera.driver_seat_x).
        self.seat_x = seat_x
        self.reset()

    def reset(self):
        self.anchor: tuple[float, float, float] | None = None   # (cx, cy, width), normalized
        self.last_seen = 0.0

    @staticmethod
    def _geom(lms):
        xs = [p.x for p in lms]
        ys = [p.y for p in lms]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, w, h

    def select(self, face_result, now: float) -> None:
        if face_result is None:
            return
        face_result.passengers = []
        if not face_result.face_landmarks:
            return
        geoms = [self._geom(l) for l in face_result.face_landmarks]
        idx = self._pick(geoms, now)
        others = [i for i in range(len(geoms)) if i != idx]
        face_result.passengers = [face_result.face_landmarks[i] for i in others]
        _reorder(face_result, [idx] if idx is not None else [])

    def _pick(self, geoms, now: float) -> int | None:
        """Index of the driver's face, or None if nobody sits in the driver's position."""
        candidates = list(range(len(geoms)))
        if self.anchor and now - self.last_seen > self.lost_reset_s:
            # driver gone for a long time: re-lock, but only onto a face in the middle
            candidates = [i for i in candidates if abs(geoms[i][0] - self.seat_x) < self.relock_center]
            if not candidates:
                return None
            self.anchor = None

        if self.anchor is None:
            # the person in the driver's seat: big AND close to the driver's seat position
            def score(g):
                cx, _, w, h = g
                return (w * h) * max(0.05, 1.0 - abs(cx - self.seat_x) * 1.6)
            idx = max(candidates, key=lambda i: score(geoms[i]))
        else:
            ax, ay, aw = self.anchor
            dist = [((g[0] - ax) ** 2 + (g[1] - ay) ** 2) ** 0.5 / max(aw, 1e-3) for g in geoms]
            idx = min(range(len(geoms)), key=lambda i: dist[i])
            if dist[idx] > self.max_jump:
                return None                          # nobody in the driver's seat right now

        cx, cy, w, _ = geoms[idx]
        if self.anchor is None:
            self.anchor = (cx, cy, w)
        else:                                        # follow the driver's small movements
            ax, ay, aw = self.anchor
            self.anchor = (0.7 * ax + 0.3 * cx, 0.7 * ay + 0.3 * cy, 0.7 * aw + 0.3 * w)
        self.last_seen = now
        return idx


class PitchBaseline:
    """Learns the driver's neutral head pitch during calibration; returns pitch relative to it.
    Negative = head tilted DOWN (flip config.head_pose.pitch_sign if it comes out reversed)."""

    def __init__(self, sign: int = 1):
        self.sign = sign
        self.reset()

    def reset(self, soft: bool = False):
        """soft=True keeps using the old baseline until the new one is learned."""
        self._samples: list[float] = []
        self._needs_baseline = True
        if not soft:
            self.baseline: float | None = None

    def relative(self, face_result, calibrating: bool) -> float | None:
        if not face_result or not face_result.facial_transformation_matrixes:
            return None
        pitch = matrix_to_euler(face_result.facial_transformation_matrixes[0])[0]
        if calibrating:
            self._samples.append(pitch)
        elif self._needs_baseline and self._samples:
            self.baseline = float(np.median(self._samples))
            self._needs_baseline, self._samples = False, []
        return (pitch - (self.baseline or 0.0)) * self.sign
