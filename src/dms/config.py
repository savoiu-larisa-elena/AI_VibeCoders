"""All tunable thresholds in one place.

Tune these live during testing. When you add driver profiles later
(rookie vs experienced), a profile simply overrides some of these values.
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"


@dataclass
class CameraConfig:
    index: int = 0              # try 1, 2... if you have several cameras
    width: int = 1280
    height: int = 720
    mirror: bool = True         # selfie view on screen


@dataclass
class CalibrationConfig:
    seconds: float = 5.0        # look straight at the road, eyes open, during this time


@dataclass
class DrowsinessConfig:
    ear_closed_ratio: float = 0.70    # eye "closed" if EAR < baseline * ratio
    ear_fallback_threshold: float = 0.20  # used before calibration finishes
    closed_alert_seconds: float = 1.0     # eyes closed this long -> event
    perclos_window_seconds: float = 60.0
    perclos_alert: float = 0.25           # >25% of the window with eyes closed -> event
    yawn_mar_threshold: float = 0.55      # mouth aspect ratio
    yawn_min_seconds: float = 1.2
    # When the head tilts down, eyelids LOOK closed to the camera even when open.
    # Below this relative pitch, low EAR counts as "looking down", not "eyes closed".
    # (Head-down is still caught by head_pose as a critical distraction.)
    head_down_ignore_pitch_deg: float = -10.0


@dataclass
class HeadPoseConfig:
    yaw_threshold_deg: float = 25.0     # left/right beyond this = looking away
    pitch_down_threshold_deg: float = 13.0
    gaze_down_threshold: float = 0.55   # eyes looking down (blendshape) without moving head
    pitch_up_threshold_deg: float = 20.0
    away_alert_seconds: float = 2.0     # looking away this long -> event
    yaw_sign: int = 1                   # flip to -1 if LEFT/RIGHT come out swapped
    pitch_sign: int = 1                 # flip to -1 if UP/DOWN come out swapped


@dataclass
class PhoneConfig:
    model: str = "yolo11n.pt"   # "yolo11s.pt" = more accurate, ~2.5x slower
    every_n_frames: int = 3     # YOLO is the heaviest model; don't run it on every frame
    hold_seconds: float = 1.0   # object confirmed this long -> event
    # COCO class id -> our label. Cup / bottle / wine glass are merged into "drink":
    # we don't care WHICH drink, and merging removes cup<->bottle flicker.
    classes: dict = field(default_factory=lambda: {
        67: "phone", 39: "drink", 40: "drink", 41: "drink"})
    # per-label minimum confidence: phone is the one that triggers critical alerts,
    # so it needs to be surer (a soda can scored ~40% "phone")
    min_conf: dict = field(default_factory=lambda: {"phone": 0.55, "drink": 0.35})
    # temporal voting: a label counts only if seen in >= votes_needed of the last vote_window runs
    vote_window: int = 5
    votes_needed: int = 3
    # ignore objects far from the driver's face (e.g. a passenger holding a can)
    driver_zone_only: bool = True
    driver_zone_scale: float = 1.6   # zone = face box widened by this many face-widths each side


@dataclass
class GestureConfig:
    min_score: float = 0.6
    hold_seconds: float = 0.5   # gesture must be held this long to count (avoids flicker)


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    drowsiness: DrowsinessConfig = field(default_factory=DrowsinessConfig)
    head_pose: HeadPoseConfig = field(default_factory=HeadPoseConfig)
    phone: PhoneConfig = field(default_factory=PhoneConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    event_cooldown_seconds: float = 5.0   # same event type won't repeat faster than this
