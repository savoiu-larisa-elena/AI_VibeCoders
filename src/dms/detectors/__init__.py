"""Registry of detectors. To add a new one: write the class, add it here."""
from .drowsiness import DrowsinessDetector
from .gestures import GestureDetector
from .head_pose import HeadPoseDetector
from .occupants import OccupantsDetector
from .wheel import WheelDetector


def _phone():
    # imported lazily so the project runs even if ultralytics isn't installed yet
    from .phone import PhoneDetector
    return PhoneDetector


DETECTORS = {
    "drowsiness": lambda: DrowsinessDetector,
    "head_pose": lambda: HeadPoseDetector,
    "phone": _phone,
    "gestures": lambda: GestureDetector,
    "occupants": lambda: OccupantsDetector,   # added automatically when face detection runs
    "wheel": lambda: WheelDetector,           # optional: virtual steering wheel (hands on wheel)
}
