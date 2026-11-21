"""Step 3 glue: simulated trip + fatigue tracking + stop advice, shared with the agent.

    copilot = TripCopilot(speed_factor=20)
    context.trip_provider = copilot.for_llm      # DriverContext asks it when building the LLM summary
    ...every frame:  copilot.update(events)

Code decides WHERE to stop (Advisor); the LLM only decides how to SAY it.
A stop is offered when the fatigue level rises or a reminder is due (FatigueTracker.announce),
or when the driver asks for a break (thumbs down). Otherwise the LLM gets no place to mention,
so it doesn't nag with the same hotel on every yawn.
"""
from __future__ import annotations

import threading
import time
from dataclasses import asdict

from .fatigue import FatigueTracker
from .trip import Advisor, Trip

FATIGUE_TYPES = {"eyes_closed", "yawn", "fatigue_perclos"}


class TripCopilot:
    def __init__(self, trip: Trip | None = None, speed_factor: float = 1.0, pending_s: float = 20.0):
        self.trip = trip or Trip(speed_factor=speed_factor)
        self.tracker = FatigueTracker()
        self.advisor = Advisor(self.trip)
        self.pending_s = pending_s
        self._pending_until = 0.0
        self._last_t: float | None = None
        self._lock = threading.RLock()   # camera thread + web endpoints both touch the trip

    # ---- every frame ----
    def update(self, events, now: float | None = None) -> None:
        with self._lock:
            self._update(events, now)

    def _update(self, events, now: float | None) -> None:
        now = time.monotonic() if now is None else now
        dt = 0.0 if self._last_t is None else now - self._last_t
        self._last_t = now
        self.trip.advance(dt)
        if any(e.type == "recalibration" and e.data.get("type") == "full" for e in events):
            self.tracker.reset()                   # new driver -> new fatigue history
            self._pending_until = 0.0
        before = self.tracker.level
        level, announce = self.tracker.update([e for e in events if e.type in FATIGUE_TYPES], now)
        if level != before:
            print(f"[TRIP] fatigue level {before} -> {level} (score {self.tracker.score:.1f}) "
                  f"at km {self.trip.km:.0f}")
        if announce:
            self._pending_until = now + self.pending_s

    # ---- called by DriverContext.summary(trigger) ----
    def for_llm(self, trigger) -> dict | None:
        with self._lock:
            return self._for_llm(trigger)

    def _for_llm(self, trigger) -> dict | None:
        wants_break = trigger.type == "gesture" and trigger.data.get("gesture") == "Thumb_Down"
        if trigger.type not in FATIGUE_TYPES and not wants_break:
            return None
        level = self.tracker.level
        if wants_break:
            level = max(level, 1)                  # "I need a break" always gets a pit stop
        if level == 0:
            return None                            # a lone yawn is not worth a place suggestion
        now = time.monotonic()
        out = {"fatigue_level": level, "km_driven": round(self.trip.km)}
        if wants_break or now < self._pending_until:
            rec = self.advisor.recommend(level)
            if rec:
                self._pending_until = 0.0          # offered once; reminders come from the tracker
                out["recommendation"] = _slim(rec)
                if rec.get("target"):
                    out["_stop"] = {"kind": rec["kind"], "urgent": rec["urgent"], **rec["target"]}
        return out

    # ---- for the dashboard map (step 4) ----
    def state(self) -> dict:
        """Live trip state for /state: car position, fatigue, and the CURRENT stop advice
        (full target with coordinates - the browser draws it on the map)."""
        with self._lock:
            lat, lon = self.trip.position
            level = self.tracker.level
            rec = self.advisor.recommend(level) if level > 0 else None
            return {"lat": round(lat, 5), "lon": round(lon, 5), "km": round(self.trip.km, 1),
                    "route_km": round(self.trip.route.length_km, 1),
                    "speed_factor": self.trip.speed_factor,
                    "fatigue_level": level, "fatigue_score": round(self.tracker.score, 1),
                    "recommendation": rec}

    def map_data(self) -> dict:
        """Static data for the map, fetched once: route polyline (+ cumulative km) and all POIs."""
        r = self.trip.route
        step = max(1, len(r.points) // 1500)        # the real road has ~3600 points: thin it out
        idx = list(range(0, len(r.points), step))
        if idx[-1] != len(r.points) - 1:
            idx.append(len(r.points) - 1)
        return {"route": [[round(r.points[i][0], 5), round(r.points[i][1], 5)] for i in idx],
                "cum": [round(r.cum[i], 2) for i in idx], "length": round(r.length_km, 1),
                "name": getattr(self.trip, "route_name", ""),
                "source": getattr(self.trip, "data_source", ""),
                "pois": [asdict(p) for p in self.trip.pois]}

    # ---- demo controls (dashboard buttons) ----
    def force_level(self, level: int) -> None:
        """Jump the fatigue score to a level so the demo doesn't depend on real tiredness."""
        with self._lock:
            level = max(0, min(2, level))
            if level == 0:
                self.tracker.reset()
                self._pending_until = 0.0
                return
            self.tracker.score = (self.tracker.enter[0], 8.0)[level - 1] + 0.2
            self.tracker.level = level
            self._pending_until = time.monotonic() + self.pending_s   # next LLM message names a place

    def jump(self, km: float) -> None:
        with self._lock:
            self.trip.km = max(0.0, min(self.trip.route.length_km, km))

    def set_speed(self, factor: float) -> None:
        with self._lock:
            self.trip.speed_factor = max(0.0, min(200.0, factor))


def _slim(rec: dict) -> dict:
    """What the LLM sees: names and distances only (no coordinates)."""
    def one(t):
        return {"name": t["name"], "type": t["kind"], "distance_km": t["distance_km"],
                "distance_text": t.get("distance_text"), "eta_min": t["eta_min"], "note": t["note"]}
    return {"kind": rec["kind"], "urgent": rec["urgent"],
            "target": one(rec["target"]) if rec.get("target") else None,
            "alternatives": [{"name": a["name"], "distance_km": a["distance_km"]}
                             for a in rec.get("alternatives", [])]}
