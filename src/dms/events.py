"""The shared data contract between team members.

Detectors produce two things every frame:
  * signals - continuous numbers for the dashboard (EAR, yaw, ...)
  * events  - discrete things that happened ("eyes closed for 1.2s")

Only EVENTS go to the LLM later. Signals go to the dashboard.
Agree on this format early so vision / LLM / dashboard can be built in parallel.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum


class Severity(str, Enum):
    INFO = "info"          # e.g. gesture recognized
    WARNING = "warning"    # e.g. looking away 2s
    CRITICAL = "critical"  # e.g. eyes closed 1s+


@dataclass
class Event:
    type: str                      # "eyes_closed", "distraction", "phone", "yawn", "gesture"...
    severity: Severity
    source: str                    # which detector produced it
    confidence: float = 1.0
    data: dict = field(default_factory=dict)   # extra numbers: duration, direction...
    ts: float = field(default_factory=time.time)

    def to_json(self) -> str:
        d = asdict(self)
        d["severity"] = self.severity.value
        return json.dumps(d)


class Cooldown:
    """Stops the same event firing on every frame."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self._last: dict[str, float] = {}

    def ready(self, key: str, now: float) -> bool:
        if now - self._last.get(key, -1e9) >= self.seconds:
            self._last[key] = now
            return True
        return False
