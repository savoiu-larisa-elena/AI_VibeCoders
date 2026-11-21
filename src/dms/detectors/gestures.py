"""Hand gestures = the driver's way to talk back to the assistant without speaking.

    Thumb_Up    "I'm OK / got it"       -> acknowledges the last alert, lowers escalation;
                                           while warnings are muted: UNMUTES them
    Open_Palm   "quiet for a minute"    -> mutes WARNING messages for 60 s (critical still speak);
                                           must be held 1.8 s so it isn't triggered by accident
    Thumb_Down  "I need a break"        -> assistant suggests a rest stop
    Victory     "how am I doing?"       -> assistant gives a short status report of the drive
    Closed_Fist  ignored on purpose: hands gripping the steering wheel look like fists

A hand that holds a drink or phone (box from the phone detector overlaps the hand) is
ignored, and an open palm needs 4 STRAIGHT fingers - a cup grip is spread but curved.

Accuracy (from live testing: fist was read as thumbs-up, palm was missed):
  1. MediaPipe's gesture model gives a first guess.
  2. A geometric check on the 21 hand landmarks (which fingers are extended, where the
     thumb points) confirms or corrects it - e.g. thumb tucked in = fist, not thumbs-up.
  3. Majority vote over the last frames + a hold time removes flicker.
"""
from __future__ import annotations

from collections import Counter, deque

import cv2
import numpy as np

from ..events import Event, Severity
from .base import Detector, DetectorOutput, FrameContext

MEANINGS = {
    "Thumb_Up": "I'm OK",
    "Open_Palm": "mute 60s",
    "Thumb_Down": "need a break",
    "Victory": "status report",
    "Closed_Fist": "ignored",
    "Pointing_Up": "no action",
    "ILoveYou": "no action",
}

# hand landmark indices
WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_MCP = 5
FINGERS = [(6, 8), (10, 12), (14, 16), (18, 20)]   # (PIP, TIP) for index, middle, ring, pinky
FINGER_CHAINS = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]  # MCP..TIP
MIDDLE_MCP = 9
STRAIGHT_RATIO = 0.85   # finger straightness needed for an open palm (1.0 = perfectly straight)


def straight_fingers(lm, w: int, h: int) -> int:
    """How many of the 4 fingers are STRAIGHT (not curved).
    Straightness = MCP->tip distance / length along the finger joints.
    Fingers wrapped around a cup are spread but curved (~0.6-0.8), an open palm is ~0.95."""
    p = np.array([(q.x * w, q.y * h) for q in lm])
    n = 0
    for chain in FINGER_CHAINS:
        along = sum(float(np.linalg.norm(p[a] - p[b])) for a, b in zip(chain, chain[1:]))
        direct = float(np.linalg.norm(p[chain[0]] - p[chain[-1]]))
        if along > 0 and direct / along >= STRAIGHT_RATIO:
            n += 1
    return n


def hand_box(lm, w: int, h: int, pad: float = 0.15) -> tuple[float, float, float, float]:
    xs = [q.x * w for q in lm]
    ys = [q.y * h for q in lm]
    bw, bh = max(xs) - min(xs), max(ys) - min(ys)
    return min(xs) - pad * bw, min(ys) - pad * bh, max(xs) + pad * bw, max(ys) + pad * bh


def holds_object(hbox, objects) -> str | None:
    """Returns the label of an object the hand is holding (object box overlaps the hand)."""
    hx1, hy1, hx2, hy2 = hbox
    for o in objects:
        ox1, oy1, ox2, oy2 = o["box"]
        iw = max(0.0, min(hx2, ox2) - max(hx1, ox1))
        ih = max(0.0, min(hy2, oy2) - max(hy1, oy1))
        area = max(1.0, (ox2 - ox1) * (oy2 - oy1))
        if iw * ih / area > 0.25:      # a quarter of the object lies inside the hand area
            return o["label"]
    return None


def classify_geometry(lm, w: int, h: int) -> str | None:
    """Rule-based gesture from landmark positions. Returns None when unsure."""
    p = np.array([(q.x * w, q.y * h) for q in lm])
    dist = lambda a, b: float(np.linalg.norm(p[a] - p[b]))  # noqa: E731
    palm = dist(WRIST, MIDDLE_MCP) or 1.0

    extended = [dist(WRIST, tip) > 1.1 * dist(WRIST, pip) for pip, tip in FINGERS]
    curled = [dist(WRIST, tip) < dist(WRIST, pip) for pip, tip in FINGERS]
    n_ext = sum(extended)

    thumb_out = (dist(THUMB_TIP, INDEX_MCP) > 0.6 * palm
                 and dist(THUMB_TIP, WRIST) > dist(THUMB_IP, WRIST))
    # tucked = thumb tip lies on the curled fingers (classic fist). Between "out" and
    # "tucked" we return None and let the ML model decide with a stricter threshold.
    thumb_tucked = min(dist(THUMB_TIP, INDEX_MCP), dist(THUMB_TIP, 10)) < 0.45 * palm
    thumb_dy = p[THUMB_TIP][1] - p[THUMB_MCP][1]       # image y grows downward
    thumb_up = thumb_out and thumb_dy < -0.3 * palm
    thumb_down = thumb_out and thumb_dy > 0.3 * palm

    if n_ext >= 4:
        # spread AND straight; fingers curled around a cup are spread but curved
        return "Open_Palm" if straight_fingers(lm, w, h) >= 4 else None
    if extended[0] and extended[1] and curled[2] and curled[3]:
        return "Victory"
    if extended[0] and curled[1] and curled[2] and curled[3]:
        return "Pointing_Up"
    if sum(curled) >= 3:
        if thumb_up:
            return "Thumb_Up"
        if thumb_down:
            return "Thumb_Down"
        if thumb_tucked:
            return "Closed_Fist"
    return None


class GestureDetector(Detector):
    name = "gestures"
    needs = {"hands"}

    def __init__(self, config):
        super().__init__(config)
        self.cfg = config.gesture
        self._votes: deque[str | None] = deque(maxlen=self.cfg.vote_window)
        self._current: str | None = None
        self._since = 0.0
        self._emitted = False
        self._debug = ""

    def _frame_label(self, ctx: FrameContext) -> tuple[str | None, float]:
        res = ctx.hands
        if not res or not res.hand_landmarks:
            self._debug = ""
            return None, 0.0
        h, w = ctx.frame.shape[:2]
        best, best_score, debug = None, 0.0, []
        for i, lm in enumerate(res.hand_landmarks):
            # a hand holding a drink/phone is busy, not gesturing
            held = holds_object(hand_box(lm, w, h), ctx.objects)
            if held:
                debug.append(f"holding {held} -> ignored")
                continue
            top = res.gestures[i][0] if i < len(res.gestures) and res.gestures[i] else None
            model_label = top.category_name if top and top.category_name != "None" else None
            model_score = top.score if top else 0.0
            geo = classify_geometry(lm, w, h)
            if geo:
                # geometry is the referee: it agrees -> keep model confidence; disagrees -> trust geometry
                label, score = geo, (model_score if model_label == geo else 0.7)
            elif model_label and model_score >= self.cfg.min_score.get(model_label, 0.75):
                label, score = model_label, model_score
                # the model alone also confuses a cup grip with an open palm
                if label == "Open_Palm" and straight_fingers(lm, w, h) < 4:
                    label, score = None, 0.0
            else:
                label, score = None, 0.0
            debug.append(f"model={model_label or '-'} geo={geo or '-'}")
            if label and score > best_score:
                best, best_score = label, score
        self._debug = " | ".join(debug)
        return best, best_score

    def process(self, ctx: FrameContext) -> DetectorOutput:
        out = DetectorOutput()
        label, score = self._frame_label(ctx)
        self._votes.append(label)

        # majority vote over recent frames
        stable = None
        if self._votes:
            top, n = Counter(self._votes).most_common(1)[0]
            if top and n >= self.cfg.votes_needed:
                stable = top

        if stable != self._current:
            self._current, self._since, self._emitted = stable, ctx.ts, False
        held = ctx.ts - self._since if stable else 0.0

        if (stable and stable in self.cfg.active
                and held >= self.cfg.hold_seconds.get(stable, 1.0) and not self._emitted):
            self._emitted = True
            out.events.append(Event("gesture", Severity.INFO, self.name,
                                    confidence=round(score, 2),
                                    data={"gesture": stable, "meaning": MEANINGS.get(stable, "")}))

        out.signals = {"gesture": stable, "gesture_score": round(score, 2),
                       "hands": len(ctx.hands.hand_landmarks) if ctx.hands else 0}
        meaning = f" -> {MEANINGS.get(stable, '')}" if stable else ""
        out.status = f"gesture: {stable or '-'}{meaning}   [{self._debug}]" if self._debug \
            else "gesture: -"
        return out

    def draw(self, frame, ctx, out):
        if not ctx.hands:
            return
        h, w = frame.shape[:2]
        color = (0, 255, 0) if out.signals.get("gesture") else (255, 0, 255)
        for hand in ctx.hands.hand_landmarks:
            for p in hand:
                cv2.circle(frame, (int(p.x * w), int(p.y * h)), 3, color, -1)
