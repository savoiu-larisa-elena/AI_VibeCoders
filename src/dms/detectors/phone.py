"""Phone / drink detection with YOLO (pretrained on COCO, no training needed).

Robustness tricks (from live testing):
  * cup/bottle/wine glass merged into one "drink" label -> no cup<->bottle flicker
  * stricter confidence for "phone" (a soda can scored ~40% phone)
  * agnostic NMS -> one object gets ONE label, never "phone" and "bottle" on the same box
  * temporal voting -> phone must appear in 3 of the last 5 YOLO runs, drink in 2 of 5
  * driver zone -> objects far from the driver's face (passenger's can) are ignored
  * flat-hand check -> fingers held together (straight) are not a phone
  * call pose -> a driver hand held at the ear for 2 s = phone call, even if the phone
    itself is hidden behind the hand (YOLO often misses a phone at the ear)
  * YOLO runs in its own background thread -> the video never waits for it
"""
from __future__ import annotations

import threading
import time
from collections import deque

import cv2

from ..config import MODELS_DIR
from ..events import Event, Severity
from ..face_utils import driver_zone, face_box
from .base import Detector, DetectorOutput, FrameContext


def flat_hand_over(box, hands, w: int, h: int, min_straight: int) -> bool:
    """True if the box is mostly covered by a FLAT hand (fingers straight).
    Fingers held together look like a phone to YOLO; a hand really holding a phone has
    its fingers curled around it, so a flat hand on top of the 'phone' = it's the hand."""
    from .gestures import hand_box, straight_fingers   # local import: no import cycle
    x1, y1, x2, y2 = box
    area = max(1.0, (x2 - x1) * (y2 - y1))
    for lm in hands:
        hx1, hy1, hx2, hy2 = hand_box(lm, w, h, pad=0.0)
        iw = max(0.0, min(x2, hx2) - max(x1, hx1))
        ih = max(0.0, min(y2, hy2) - max(y1, hy1))
        if iw * ih / area >= 0.4 and straight_fingers(lm, w, h) >= min_straight:
            return True
    return False


EARS = (234, 454)                       # face-mesh points at the left/right side of the head
HAND_POINTS = (0, 4, 5, 8, 9, 12, 13, 17)  # wrist, knuckles, fingertips


def hand_at_ear(face_result, hands, w: int, h: int, max_dist: float) -> bool:
    """True if a hand is held next to one of the driver's ears (phone-call pose)."""
    if not face_result or not face_result.face_landmarks or not hands:
        return False
    lm = face_result.face_landmarks[0]
    x1, _, x2, _ = face_box(lm, w, h)
    limit = max_dist * max(1, x2 - x1)
    ears = [(lm[i].x * w, lm[i].y * h) for i in EARS]
    for hand in hands:
        for i in HAND_POINTS:
            hx, hy = hand[i].x * w, hand[i].y * h
            if any(((hx - ex) ** 2 + (hy - ey) ** 2) ** 0.5 < limit for ex, ey in ears):
                return True
    return False


class PhoneDetector(Detector):
    name = "phone"
    needs = {"face", "hands"}   # face: driver zone; hands: reject "flat hand = phone" mistakes
    order = 10                  # before gestures, which uses the object boxes

    def __init__(self, config):
        super().__init__(config)
        self.cfg = config.phone
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("Phone detection needs:  pip install ultralytics") from e
        local = MODELS_DIR / self.cfg.model
        self.model = YOLO(str(local) if local.exists() else self.cfg.model)
        self.labels = sorted(set(self.cfg.classes.values()))
        self._boxes: list[dict] = []           # latest YOLO result, for drawing + gestures
        self._zone: tuple[int, int, int, int] | None = None
        self._votes: deque[dict[str, float]] = deque(maxlen=self.cfg.vote_window)
        self._seen_since: dict[str, float] = {}
        self._alerted: set[str] = set()
        self._ear_since: float | None = None   # hand at the ear since (phone-call pose)
        self.call_pose = False
        # background worker: takes the newest frame, runs YOLO, publishes the result
        self._lock = threading.Lock()
        self._job: tuple | None = None         # (frame, zone) waiting to be processed
        self._result: tuple | None = None      # (found, boxes, zone) not yet consumed
        self._wake = threading.Event()
        self._stop = threading.Event()
        self.last_run_ms = 0.0
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    # ---------- background thread ----------
    def _loop(self):
        while not self._stop.is_set():
            if not self._wake.wait(0.2):
                continue
            with self._lock:
                job, self._job = self._job, None
                self._wake.clear()
            if job is None:
                continue
            frame, zone = job
            t0 = time.perf_counter()
            try:
                found, boxes = self._run_yolo(frame, zone)
            except Exception:   # noqa: BLE001 - never kill the thread on one bad frame
                continue
            self.last_run_ms = (time.perf_counter() - t0) * 1000
            with self._lock:
                self._result = (found, boxes, zone)
            # cap the check rate so YOLO doesn't steal CPU from the face/hand models
            rest = 1.0 / self.cfg.max_checks_per_second - (time.perf_counter() - t0)
            if rest > 0:
                self._stop.wait(rest)

    def _run_yolo(self, frame, zone):
        result = self.model(frame, conf=min(self.cfg.min_conf.values()),
                            classes=list(self.cfg.classes), agnostic_nms=True,
                            verbose=False)[0]
        found: dict[str, float] = {}
        boxes = []
        for b in result.boxes:
            label = self.cfg.classes[int(b.cls)]
            conf = float(b.conf)
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            status = "ok"
            if conf < self.cfg.min_conf.get(label, 0.5):
                status = "low"
            elif zone and not (zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]):
                status = "outside"   # e.g. passenger's object
            boxes.append({"label": label, "conf": conf, "box": (x1, y1, x2, y2), "status": status})
            if status == "ok":
                found[label] = max(found.get(label, 0), conf)
        return found, boxes

    # ---------- main thread ----------
    def _driver_zone(self, ctx: FrameContext):
        if not self.cfg.driver_zone_only:
            return None
        h, w = ctx.frame.shape[:2]
        return driver_zone(ctx.face, w, h, self.cfg.driver_zone_scale)

    def process(self, ctx: FrameContext) -> DetectorOutput:
        out = DetectorOutput()
        zone = self._driver_zone(ctx)
        with self._lock:
            if self._job is None:              # worker idle -> give it the newest frame
                self._job = (ctx.frame.copy(), zone)
                self._wake.set()
            new, self._result = self._result, None
        if new:                                # a YOLO run finished since last frame
            found, self._boxes, self._zone = new
            hands = ctx.hands.hand_landmarks if ctx.hands else []
            if hands:
                h, w = ctx.frame.shape[:2]
                for b in self._boxes:
                    if (b["label"] == "phone" and b["status"] == "ok" and
                            flat_hand_over(b["box"], hands, w, h,
                                           self.cfg.flat_hand_straight_fingers)):
                        b["status"] = "flat hand"          # drawn grey, not counted
                found = {}
                for b in self._boxes:
                    if b["status"] == "ok":
                        found[b["label"]] = max(found.get(b["label"], 0), b["conf"])
            self._votes.append(found)
        # share what's in the driver's hands with the gesture detector (flat-hand boxes too:
        # a hand holding a phone with spread fingers must not count as an open palm)
        ctx.objects = [b for b in self._boxes if b["status"] in ("ok", "low", "flat hand")]

        # phone-call pose: driver's hand at the ear for call_pose_seconds
        h, w = ctx.frame.shape[:2]
        hands = ctx.hands.hand_landmarks if ctx.hands else []
        at_ear = hand_at_ear(ctx.face, hands, w, h, self.cfg.call_pose_distance)
        if at_ear:
            self._ear_since = self._ear_since or ctx.ts
        else:
            self._ear_since = None
        self.call_pose = bool(at_ear and ctx.ts - self._ear_since >= self.cfg.call_pose_seconds)

        # a label is "confirmed" only if it appeared in enough recent YOLO runs
        confirmed: dict[str, float] = {}
        for label in self.labels:
            hits = [v[label] for v in self._votes if label in v]
            if len(hits) >= self.cfg.votes_needed.get(label, 3):
                confirmed[label] = sum(hits) / len(hits)
        via_pose = self.call_pose and "phone" not in confirmed
        if via_pose:
            confirmed["phone"] = 0.6          # hand at the ear long enough: treat as a call
        # a hand on the phone (or at the ear) is busy - gestures ignore the open palm
        ctx.phone_busy = "phone" in confirmed or at_ear

        for label in self.labels:
            if label in confirmed:
                self._seen_since.setdefault(label, ctx.ts)
                held = ctx.ts - self._seen_since[label]
                if held >= self.cfg.hold_seconds and label not in self._alerted:
                    self._alerted.add(label)
                    sev = Severity.CRITICAL if label == "phone" else Severity.INFO
                    data = {"seconds": round(held, 2)}
                    if label == "phone" and via_pose:
                        data["pose"] = "hand at ear (phone call)"
                    out.events.append(Event(label, sev, self.name,
                                            confidence=round(confirmed[label], 2), data=data))
            else:
                self._seen_since.pop(label, None)
                self._alerted.discard(label)

        out.signals = {f"{label}_visible": label in confirmed for label in self.labels}
        out.signals["call_pose"] = self.call_pose
        out.status = "objects: " + (", ".join(f"{k} {v:.0%}" for k, v in confirmed.items())
                                    or "none")
        if at_ear:
            out.status += (" | PHONE CALL POSE" if self.call_pose
                           else f" | hand at ear {ctx.ts - self._ear_since:.1f}s")
        return out

    def draw(self, frame, ctx, out):
        if self._zone:
            x1, y1, x2, y2 = self._zone
            cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)
            cv2.putText(frame, "driver zone", (x1 + 4, y1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
        for b in self._boxes:
            x1, y1, x2, y2 = b["box"]
            if b["status"] == "ok":
                color = (0, 0, 255) if b["label"] == "phone" else (255, 200, 0)
                text = f"{b['label']} {b['conf']:.0%}"
            else:   # grey = seen but ignored (too unsure / outside driver zone)
                color = (130, 130, 130)
                text = f"{b['label']} {b['conf']:.0%} ({b['status']})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    def close(self):
        self._stop.set()
        self._wake.set()
        self._worker.join(timeout=2)
