"""Every detector follows this interface, so they can be added one at a time."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..events import Event


@dataclass
class FrameContext:
    frame: np.ndarray            # BGR image (what OpenCV gives you)
    ts: float                    # seconds (time.monotonic)
    calibrating: bool            # True during the first N seconds
    face: object | None = None   # FaceLandmarkerResult, if a detector needs "face"
    hands: object | None = None  # GestureRecognizerResult, if a detector needs "hands"
    # objects seen by the phone detector: [{"label", "conf", "box": (x1,y1,x2,y2)}]
    # (gestures uses it: a hand holding a drink/phone is not making a gesture)
    objects: list = field(default_factory=list)


@dataclass
class DetectorOutput:
    signals: dict = field(default_factory=dict)   # -> dashboard
    events: list[Event] = field(default_factory=list)  # -> LLM later
    status: str = ""     # one short line drawn on the debug window


class Detector:
    name: str = "base"
    needs: set[str] = set()   # which providers to run: {"face"}, {"hands"}, or empty
    order: int = 50           # lower runs first within a frame

    def __init__(self, config):
        self.config = config

    def process(self, ctx: FrameContext) -> DetectorOutput:
        raise NotImplementedError

    def draw(self, frame: np.ndarray, ctx: FrameContext, out: DetectorOutput) -> None:
        """Optional: draw debug info (landmarks, boxes) on the frame."""

    def recalibrate(self) -> None:
        """Optional: reset baselines (pressed 'c')."""

    def close(self) -> None:
        pass
