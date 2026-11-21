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
    # Where the DRIVER sits horizontally in the (mirrored) image: 0 = left edge, 1 = right.
    # Camera on the roof between the seats, aimed at the driver: driver on the left,
    # passenger on the right. A single person anywhere is still taken as the driver.
    driver_seat_x: float = 0.35


@dataclass
class CalibrationConfig:
    seconds: float = 5.0        # look straight at the road, eyes open, during this time
    # Driver change: the driver's seat is empty this long, then a face appears
    # -> FULL recalibration (new person = new eye shape, new seat position).
    absent_recalibrate_seconds: float = 15.0
    # Periodic BASELINE REFRESH (lighting changes, driver shifted in the seat).
    # Only while the driver is alert - never learn a tired face as "normal" - and it keeps
    # the fatigue history. 0 = off.
    periodic_minutes: float = 10.0
    periodic_quiet_seconds: float = 60.0   # no warning/critical alert for this long before


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
    # yolo11s: clearly better at cups/bottles than yolo11n (~70 ms vs ~30 ms per run).
    # It runs in a background thread, so it doesn't slow the video down.
    model: str = "yolo11s.pt"
    max_checks_per_second: float = 5.0   # plenty for objects; leaves CPU for the face/hand models
    hold_seconds: float = 1.0   # object confirmed this long -> event
    # COCO class id -> our label. Cup / bottle / wine glass are merged into "drink":
    # we don't care WHICH drink, and merging removes cup<->bottle flicker.
    classes: dict = field(default_factory=lambda: {
        67: "phone", 39: "drink", 40: "drink", 41: "drink"})
    # per-label minimum confidence: phone is the one that triggers critical alerts,
    # so it needs to be surer (a soda can scored ~40% "phone")
    min_conf: dict = field(default_factory=lambda: {"phone": 0.55, "drink": 0.25})
    # temporal voting: a label counts only if seen in >= votes_needed of the last vote_window runs.
    # Drinks are often half-hidden by the hand and only recognized at some angles -> easier rule.
    vote_window: int = 5
    votes_needed: dict = field(default_factory=lambda: {"phone": 3, "drink": 2})
    # ignore objects far from the driver's face (e.g. a passenger holding a can)
    # Fingers held together look like a phone to YOLO. A hand HOLDING a phone has curled
    # fingers; if the "phone" overlaps a FLAT hand (this many straight fingers) it's the hand.
    flat_hand_straight_fingers: int = 3
    driver_zone_only: bool = True
    driver_zone_scale: float = 1.6   # zone = face box widened by this many face-widths each side
    # Phone at the ear is mostly hidden by the hand, so YOLO often misses it. A driver hand
    # held at the ear this long counts as a phone call even without seeing the phone.
    call_pose_seconds: float = 2.0
    call_pose_distance: float = 0.6  # hand within this many face-widths of an ear


@dataclass
class GestureConfig:
    # model-only confidence needed when the geometric check is unsure
    min_score: dict = field(default_factory=lambda: {
        "Thumb_Up": 0.80, "Thumb_Down": 0.80, "Open_Palm": 0.75, "Victory": 0.70})
    # open palm needs 4 fingers at least this straight (1.0 = perfectly straight). A hand
    # holding a phone with spread fingers is spread but less straight.
    open_palm_straightness: float = 0.90
    vote_window: int = 10       # look at the last 10 frames...
    votes_needed: int = 6       # ...the same gesture must win at least 6 of them
    # ...then be held this long to count. Open palm (mute) is the riskiest to trigger by
    # accident - hands open naturally - so it needs the longest hold.
    hold_seconds: dict = field(default_factory=lambda: {
        "Open_Palm": 1.8, "Thumb_Up": 1.0, "Thumb_Down": 1.0, "Victory": 1.0})
    # gestures that trigger something (fist is ignored: hands on the wheel look like fists)
    active: tuple = ("Thumb_Up", "Open_Palm", "Thumb_Down", "Victory")


@dataclass
class WheelConfig:
    """Virtual steering wheel: the UPPER half of a wheel in the lower-LEFT corner of the
    image, in front of the driver (camera on the roof, driver on the left). A laptop camera
    can't see a real wheel, so the demo driver holds their hands on the drawn rim.
    Coordinates are fractions of the image size."""
    always_on: bool = True            # added automatically to every run (--no-wheel turns it off)
    center: tuple = (0.24, 1.0)       # x, y of the wheel centre (bottom edge -> only top half shows)
    size: tuple = (0.36, 0.50)        # full width, height of the wheel ellipse
    grip_margin: float = 1.25         # hands slightly outside the drawn rim still count
    warn_seconds: float = 2.0         # no hand on the wheel this long -> warning
    critical_seconds: float = 5.0     # ... this long -> critical


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    drowsiness: DrowsinessConfig = field(default_factory=DrowsinessConfig)
    head_pose: HeadPoseConfig = field(default_factory=HeadPoseConfig)
    phone: PhoneConfig = field(default_factory=PhoneConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    wheel: WheelConfig = field(default_factory=WheelConfig)
    event_cooldown_seconds: float = 5.0   # same event type won't repeat faster than this
