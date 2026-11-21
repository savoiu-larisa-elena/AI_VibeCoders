"""Turns the per-frame detector output into a compact driver-state summary for the LLM.

The LLM never sees video - only this summary (a few hundred characters of JSON).
"""
from __future__ import annotations

import time
from collections import Counter, deque

from ..events import Event


class DriverContext:
    def __init__(self, history_seconds: float = 300.0):
        self.started = time.time()
        self.history_seconds = history_seconds
        self.events: deque[Event] = deque()
        self.signals: dict = {}
        self.attention = 100.0       # smoothed 0-100
        self.totals: Counter = Counter()   # whole-session event counts (for status reports)

    # ---- called every frame ----
    def update(self, signals: dict, events: list[Event]) -> None:
        self.signals = signals
        now = time.time()
        self.events.extend(events)
        self.totals.update(e.type for e in events if e.type != "gesture")
        while self.events and now - self.events[0].ts > self.history_seconds:
            self.events.popleft()
        self.attention = 0.9 * self.attention + 0.1 * self._instant_attention()

    def _instant_attention(self) -> float:
        """Simple, explainable score: start at 100, subtract for each risky signal."""
        d = self.signals.get("drowsiness", {})
        h = self.signals.get("head_pose", {})
        p = self.signals.get("phone", {})
        score = 100.0
        if d.get("eyes_closed"):
            score -= 40 + min(40, d.get("closed_seconds", 0) * 30)
        score -= min(30, d.get("perclos", 0) * 100)
        if h.get("looking") in ("left", "right", "up"):
            score -= 20 + min(30, h.get("away_seconds", 0) * 10)
        elif h.get("looking") == "down":
            score -= 30 + min(40, h.get("away_seconds", 0) * 15)
        if p.get("phone_visible"):
            score -= 40
        score -= 5 * self.count("yawn", 300)
        return max(0.0, min(100.0, score))

    def count(self, event_type: str, seconds: float) -> int:
        now = time.time()
        return sum(1 for e in self.events if e.type == event_type and now - e.ts <= seconds)

    # ---- called when we ask the LLM ----
    def summary(self, trigger: Event) -> dict:
        now = time.time()
        d = self.signals.get("drowsiness", {})
        h = self.signals.get("head_pose", {})
        p = self.signals.get("phone", {})
        recent = [
            f"{e.type}{_fmt(e.data)} ({int(now - e.ts)}s ago)"
            for e in list(self.events)[-8:] if e is not trigger and e.type != "gesture"
        ]
        return {
            "trigger": {"type": trigger.type, "severity": trigger.severity.value, **trigger.data},
            "driver_state": {
                "attention_score": round(self.attention),
                "eyes": "closed" if d.get("eyes_closed") else ("head_down" if d.get("head_down") else "open"),
                "perclos_60s": d.get("perclos"),
                "blinks_per_min": d.get("blinks_per_min"),
                "looking": h.get("looking"),
                "phone_in_hand": p.get("phone_visible", False),
                "drinking": p.get("drink_visible", False),
            },
            "counts_last_5min": {
                t: self.count(t, 300)
                for t in ("eyes_closed", "yawn", "distraction", "phone", "fatigue_perclos")
                if self.count(t, 300)
            },
            "counts_session": dict(self.totals),
            "recent_events": recent,
            "session_minutes": round((now - self.started) / 60, 1),
        }


def _fmt(data: dict) -> str:
    keep = {k: v for k, v in data.items() if k in ("direction", "seconds", "gesture")}
    return f" {keep}" if keep else ""
