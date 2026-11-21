"""Checks the pipeline loads and runs WITHOUT a camera (synthetic frames).

    python tools/smoke_test.py
    python tools/smoke_test.py --detectors drowsiness,head_pose,phone,gestures
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dms.pipeline import Pipeline  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--detectors", default="drowsiness,head_pose,gestures")
args = ap.parse_args()

pipe = Pipeline(args.detectors.split(","))
frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
for _ in range(10):
    signals, events = pipe.process(frame.copy())
    pipe.draw(frame)
pipe.close()
print("signals:", signals)
print("OK - pipeline runs:", args.detectors)
