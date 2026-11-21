# AI_VibeCoders – AI Driver Monitoring

Codelap Hackathon 2026 · Porsche · AI Drive challenge

A live driver-monitoring assistant: **computer vision** reads the driver's state from a webcam,
an **LLM** turns it into in-car alerts and voice guidance, and a **live dashboard** shows everything.

> Status: **Phase 1 – computer vision.** LLM, voice and dashboard come next.

## Architecture

```
Webcam ──▶ Providers (run each model once per frame)
             ├─ FaceProvider   (MediaPipe Face Landmarker: 478 points, blendshapes, head matrix)
             └─ HandProvider   (MediaPipe Gesture Recognizer)
                    ▼
           Detectors (each one independent, testable alone)
             ├─ drowsiness   eyes closed · PERCLOS · blinks · yawns
             ├─ head_pose    looking left / right / down / up
             ├─ phone        YOLO11: phone · cup · bottle
             └─ gestures     thumbs up · open palm · ...
                    ▼
           signals (continuous → dashboard)   +   events (discrete → LLM)
                    ▼
           logs/session_*.jsonl     [next: LLM agent → voice → dashboard]
```

| Folder | What's inside |
|---|---|
| `src/dms/config.py` | **All thresholds** – tune here |
| `src/dms/events.py` | The `Event` format shared by every part of the system |
| `src/dms/providers.py` | Shared MediaPipe models |
| `src/dms/detectors/` | One file per detection feature |
| `src/dms/pipeline.py` | Runs the selected detectors, returns `(signals, events)` |
| `run.py` | Live webcam app with debug overlay |
| `tools/` | Camera test, smoke test |
| `presentation/` | Final `AI_VibeCoders.pptx` goes here |

## Setup (each team member, once)

Requires **Python 3.12 (64-bit)**.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
$env:PYTHONUTF8=1                 # avoids emoji/encoding crashes in Windows consoles
pip install -r requirements.txt
python scripts/download_models.py
python tools/smoke_test.py        # should end with "OK - pipeline runs"
```

## Run

```powershell
python tools/test_camera.py                    # 1. camera works?
python run.py --detectors drowsiness           # 2. test ONE detector at a time
python run.py --detectors head_pose
python run.py --detectors phone
python run.py --detectors gestures
python run.py --detectors drowsiness,head_pose,phone,gestures   # 3. all together
```

**Keys:** `q` quit · `c` recalibrate.
The first 5 seconds are **calibration**: look straight ahead with eyes open.
Every event is printed in the terminal and saved to `logs/session_<time>.jsonl`.

## Detection logic

| Detector | Signal | Event fires when |
|---|---|---|
| drowsiness | EAR (eye aspect ratio) vs. your calibrated baseline | eyes closed ≥ 1.0 s · PERCLOS ≥ 25 % over 60 s · mouth open ≥ 1.2 s (yawn). EAR is ignored while the head is tilted down (lids look closed from below). |
| head_pose | yaw / pitch relative to calibrated "road" pose + eye-gaze-down blendshape | looking away ≥ 2 s (down = critical) |
| phone | YOLO11n: phone + "drink" (cup/bottle/glass merged) | confirmed in 3 of last 5 runs, inside the driver zone, ≥ 1 s |
| gestures | MediaPipe gesture classes | gesture held ≥ 0.5 s |

**Robustness:** with several people in view, the **largest face = driver**. Objects outside the
driver zone (grey box) are ignored, e.g. a passenger holding a can.

## Tech stack

Python 3.12 · OpenCV · MediaPipe Tasks · Ultralytics YOLO11 · *(next: LLM API, TTS, FastAPI + WebSocket dashboard)*
