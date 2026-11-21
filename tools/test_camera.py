"""Step 1: check every team laptop's camera works. Shows index, resolution, FPS.

    python tools/test_camera.py            # camera 0
    python tools/test_camera.py --camera 1
"""
import argparse
import sys
import time

import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--camera", type=int, default=0)
args = ap.parse_args()

backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
cap = cv2.VideoCapture(args.camera, backend)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    sys.exit(f"Camera {args.camera} not available. Try --camera 1, and close Teams/Zoom.")

frames, t0 = 0, time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frames += 1
    fps = frames / (time.time() - t0)
    h, w = frame.shape[:2]
    cv2.putText(frame, f"camera {args.camera}  {w}x{h}  {fps:.0f} FPS  (q to quit)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Camera test", cv2.flip(frame, 1))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
