"""Hand gestures = the driver's way to talk back to the assistant without speaking.

    Thumb_Up    "I'm OK / got it"       -> acknowledges the last alert, lowers escalation
    Open_Palm   "quiet for a minute"    -> mutes WARNING messages for 60 s (critical still speak)
    Thumb_Down  "I need a break"        -> assistant suggests a rest stop
    Victory     "how am I doing?"       -> assistant gives a short status report of the drive
    Closed_Fist  ignored on purpose: hands gripping the steering wheel look like fists

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
MIDDLE_MCP = 9


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
        return "Open_Palm"
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
            top = res.gestures[i][0] if i < len(res.gestures) and res.gestures[i] else None
            model_label = top.category_name if top and top.category_name != "None" else None
            model_score = top.score if top else 0.0
            geo = classify_geometry(lm, w, h)
            if geo:
                # geometry is the referee: it agrees -> keep model confidence; disagrees -> trust geometry
                label, score = geo, (model_score if model_label == geo else 0.7)
            elif model_label and model_score >= self.cfg.min_score.get(model_label, 0.75):
                label, score = model_label, model_score
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

        if (stable and stable in self.cfg.active and held >= self.cfg.hold_seconds
                and not self._emitted):
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
