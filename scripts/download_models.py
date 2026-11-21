"""Downloads the MediaPipe model files into models/ (run once per laptop)."""
import sys
import urllib.request
from pathlib import Path

MODELS = {
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "gesture_recognizer.task":
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
}

out_dir = Path(__file__).resolve().parents[1] / "models"
out_dir.mkdir(exist_ok=True)
for name, url in MODELS.items():
    dest = out_dir / name
    if dest.exists():
        print(f"ok        {name}")
        continue
    print(f"download  {name} ...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"{dest.stat().st_size / 1e6:.1f} MB")
    except Exception as e:  # noqa: BLE001
        dest.unlink(missing_ok=True)
        sys.exit(f"FAILED: {e}")
yolo = out_dir / "yolo11n.pt"
if yolo.exists():
    print("ok        yolo11n.pt")
else:
    try:
        import os

        from ultralytics import YOLO
        os.chdir(out_dir)          # ultralytics downloads into the current folder
        YOLO("yolo11n.pt")
        print("download  yolo11n.pt ... done")
    except ImportError:
        print("skip      yolo11n.pt (pip install ultralytics first, needed for phone detection)")
