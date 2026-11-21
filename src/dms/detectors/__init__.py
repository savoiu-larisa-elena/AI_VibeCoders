"""Registry of detectors. To add a new one: write the class, add it here."""
from .drowsiness import DrowsinessDetector
from .gestures import GestureDetector
from .head_pose import HeadPoseDetector


def _phone():
    # imported lazily so the project runs even if ultralytics isn't installed yet
    from .phone import PhoneDetector
    return PhoneDetector


DETECTORS = {
    "drowsiness": lambda: DrowsinessDetector,
    "head_pose": lambda: HeadPoseDetector,
    "phone": _phone,
    "gestures": lambda: GestureDetector,
}
