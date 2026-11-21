"""Shared MediaPipe models.

Several detectors need the same face landmarks (drowsiness AND head pose).
Providers run each heavy model ONCE per frame and share the result.
"""
from __future__ import annotations

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from .config import MODELS_DIR
from .face_utils import DriverSelector


def _model_path(name: str) -> str:
    path = MODELS_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Model {path} missing. Run:  python scripts/download_models.py"
        )
    return str(path)


class FaceProvider:
    """478 face landmarks + 52 blendshapes + head transformation matrix."""

    def __init__(self):
        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=_model_path("face_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=3,   # passengers may be in view; DriverSelector picks the driver
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self.driver = DriverSelector()

    def process(self, mp_image: mp.Image, ts_ms: int):
        result = self._landmarker.detect_for_video(mp_image, ts_ms)
        self.driver.select(result, ts_ms / 1000.0)
        return result

    def reset(self):
        """Recalibrate: forget the driver's position and lock on again."""
        self.driver.reset()

    def close(self):
        self._landmarker.close()


class HandProvider:
    """Hand landmarks + built-in gestures (Thumb_Up, Open_Palm, Victory, ...)."""

    # Hand tracking is the heaviest MediaPipe model when hands are visible. Gestures must be
    # held 0.6 s anyway, so running it on every 2nd frame loses nothing and saves CPU.
    every_n_frames = 2

    def __init__(self):
        options = vision.GestureRecognizerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=_model_path("gesture_recognizer.task")),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=4,   # driver + passenger hands; gestures only use the driver's
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)

    def process(self, mp_image: mp.Image, ts_ms: int):
        return self._recognizer.recognize_for_video(mp_image, ts_ms)

    def close(self):
        self._recognizer.close()


PROVIDERS = {"face": FaceProvider, "hands": HandProvider}
