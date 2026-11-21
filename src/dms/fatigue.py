"""Step 2: turns drowsiness EVENTS into a stable fatigue LEVEL (0 ok, 1 tired, 2 very tired).

    ft = FatigueTracker()
    level, announce = ft.update(events, now)   # call every frame with the frame's events

Why a score and not "if eyes_closed -> hotel": one yawn means nothing, three yawns plus a long
eye closure means a lot. Every event adds points, points fade with time (half-life), and the
level has HYSTERESIS (enter high, leave low) so it doesn't flicker between levels.
`announce` is True only when it's worth speaking: the level went UP, or a reminder is due.
"""
from __future__ import annotations

import math
import time

# event type -> points. EDIT THE KEYS to match the event names your drowsiness detector emits.
WEIGHTS = {
    "yawn": 1.0,
    "fatigue_perclos": 2.0,  # eyes closed too large a share of the last minute
    "eyes_closed": 3.0,    # eyes closed >= closed_alert_seconds (micro-sleep)
}
SEVERITY_BONUS = {"info": 0.0, "warning": 0.0, "critical": 0.5}
# The detector re-emits fatigue_perclos every few seconds while PERCLOS stays high, so count it
# at most once per this many seconds (otherwise one bad minute would max out the score).
MIN_GAP_S = {"fatigue_perclos": 30.0}
MAX_SCORE = 12.0


class FatigueTracker:
    def __init__(self, weights: dict | None = None, half_life_s: float = 120.0,
                 enter: tuple[float, float] = (2.5, 6.0),    # score to ENTER level 1 / level 2
                 leave: tuple[float, float] = (1.5, 4.0),    # score to LEAVE level 1 / level 2
                 reminder_s: tuple[float, float] = (180.0, 60.0)):  # repeat advice at level 1 / 2
        self.w = weights or WEIGHTS
        self.decay = math.log(2) / half_life_s
        self.enter, self.leave, self.reminder_s = enter, leave, reminder_s
        self.score = 0.0
        self.level = 0
        self._t = None
        self._announced_at = -1e9
        self._last_counted: dict[str, float] = {}

    def reset(self):
        self.score, self.level, self._t, self._announced_at = 0.0, 0, None, -1e9
        self._last_counted = {}

    def update(self, events, now: float | None = None) -> tuple[int, bool]:
        now = time.monotonic() if now is None else now
        if self._t is not None:
            self.score *= math.exp(-self.decay * max(0.0, now - self._t))
        self._t = now
        for e in events:
            pts = self.w.get(e.type)
            if pts and now - self._last_counted.get(e.type, -1e9) >= MIN_GAP_S.get(e.type, 0.0):
                self._last_counted[e.type] = now
                bonus = SEVERITY_BONUS.get(getattr(e.severity, "value", e.severity), 0.0)
                self.score = min(MAX_SCORE, self.score + (pts + bonus) * getattr(e, "confidence", 1.0))

        old = self.level
        if self.level < 2 and self.score >= self.enter[1]:
            self.level = 2
        elif self.level < 1 and self.score >= self.enter[0]:
            self.level = 1
        elif self.level == 2 and self.score < self.leave[1]:
            self.level = 1 if self.score >= self.leave[0] else 0
        elif self.level == 1 and self.score < self.leave[0]:
            self.level = 0

        announce = False
        if self.level > old:
            announce = True
        elif self.level > 0 and now - self._announced_at >= self.reminder_s[self.level - 1]:
            announce = True
        if announce:
            self._announced_at = now
        return self.level, announce


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class E:                     # stand-in for dms.events.Event
        type: str
        severity: str = "warning"
        confidence: float = 1.0

    ft = FatigueTracker()
    timeline = [(0, [E("yawn")]), (20, [E("yawn")]), (40, [E("yawn")]),
                (60, []), (80, [E("eyes_closed", "critical")]), (100, [E("eyes_closed", "critical")]),
                (110, [E("fatigue_perclos")]), (115, [E("fatigue_perclos")]), (120, [E("fatigue_perclos")]), (130, []), (200, []), (400, []), (700, []), (900, [])]
    for t, evs in timeline:
        lvl, ann = ft.update(evs, float(t))
        print(f"t={t:4}s events={[e.type for e in evs]!s:22} score={ft.score:4.1f} level={lvl} announce={ann}")
