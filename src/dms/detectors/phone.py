"""Phone / drink detection with YOLO (pretrained on COCO, no training needed).

Robustness tricks (from live testing):
  * cup/bottle/wine glass merged into one "drink" label -> no cup<->bottle flicker
  * stricter confidence for "phone" (a soda can scored ~40% phone)
  * agnostic NMS -> one object gets ONE label, never "phone" and "bottle" on the same box
  * temporal voting -> a label must appear in 3 of the last 5 YOLO runs
  * driver zone -> objects far from the driver's face (passenger's can) are ignored
"""
from __future__ import annotations

from collections import deque

import cv2

from ..config import MODELS_DIR
from ..events import Event, Severity
from ..face_utils import face_box
from .base import Detector, DetectorOutput, FrameContext


class PhoneDetector(Detector):
    name = "phone"
    needs = {"face"}   # only for the driver zone; works without a face too

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
        self._frame_i = 0
        self._boxes: list[dict] = []           # last YOLO run, for drawing
        self._zone: tuple[int, int, int, int] | None = None
        self._votes: deque[dict[str, float]] = deque(maxlen=self.cfg.vote_window)
        self._seen_since: dict[str, float] = {}
        self._alerted: set[str] = set()

    def _driver_zone(self, ctx: FrameContext):
        if not self.cfg.driver_zone_only or not ctx.face or not ctx.face.face_landmarks:
            return None
        h, w = ctx.frame.shape[:2]
        x1, y1, x2, y2 = face_box(ctx.face.face_landmarks[0], w, h)
        fw, fh = x2 - x1, y2 - y1
        s = self.cfg.driver_zone_scale
        return (max(0, int(x1 - s * fw)), max(0, int(y1 - 0.5 * fh)),
                min(w, int(x2 + s * fw)), h)   # down to the bottom: hands / lap

    def _run_yolo(self, ctx: FrameContext) -> dict[str, float]:
        result = self.model(ctx.frame, conf=min(self.cfg.min_conf.values()),
                            classes=list(self.cfg.classes), agnostic_nms=True,
                            verbose=False)[0]
        self._zone = self._driver_zone(ctx)
        found: dict[str, float] = {}
        self._boxes = []
        for b in result.boxes:
            label = self.cfg.classes[int(b.cls)]
            conf = float(b.conf)
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            status = "ok"
            if conf < self.cfg.min_conf.get(label, 0.5):
                status = "low"
            elif self._zone and not (self._zone[0] <= cx <= self._zone[2]
                                     and self._zone[1] <= cy <= self._zone[3]):
                status = "outside"   # e.g. passenger's object
            self._boxes.append({"label": label, "conf": conf, "box": (x1, y1, x2, y2),
                                "status": status})
            if status == "ok":
                found[label] = max(found.get(label, 0), conf)
        return found

    def process(self, ctx: FrameContext) -> DetectorOutput:
        out = DetectorOutput()
        self._frame_i += 1
        if self._frame_i % self.cfg.every_n_frames == 0:
            self._votes.append(self._run_yolo(ctx))

        # a label is "confirmed" only if it appeared in enough recent YOLO runs
        confirmed: dict[str, float] = {}
        for label in self.labels:
            hits = [v[label] for v in self._votes if label in v]
            if len(hits) >= self.cfg.votes_needed:
                confirmed[label] = sum(hits) / len(hits)

        for label in self.labels:
            if label in confirmed:
                self._seen_since.setdefault(label, ctx.ts)
                held = ctx.ts - self._seen_since[label]
                if held >= self.cfg.hold_seconds and label not in self._alerted:
                    self._alerted.add(label)
                    sev = Severity.CRITICAL if label == "phone" else Severity.INFO
                    out.events.append(Event(label, sev, self.name,
                                            confidence=round(confirmed[label], 2),
                                            data={"seconds": round(held, 2)}))
            else:
                self._seen_since.pop(label, None)
                self._alerted.discard(label)

        out.signals = {f"{label}_visible": label in confirmed for label in self.labels}
        out.status = "objects: " + (", ".join(f"{k} {v:.0%}" for k, v in confirmed.items())
                                    or "none")
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
