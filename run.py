"""Live driver monitoring from the webcam.

Test detectors ONE AT A TIME:
    python run.py --detectors drowsiness
    python run.py --detectors head_pose
    python run.py --detectors phone
    python run.py --detectors gestures
Then combine:
    python run.py --detectors drowsiness,head_pose,phone,gestures

Add the LLM assistant (needs a key in .env):
    python run.py --detectors drowsiness,head_pose,phone,gestures --llm

Keys:  q = quit   c = recalibrate (look straight at the road, eyes open)
Every event (and assistant reply) is printed and saved to logs/session_<time>.jsonl
"""
from __future__ import annotations

import argparse
import sys
import textwrap
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dms.config import LOGS_DIR, Config  # noqa: E402
from dms.events import Severity  # noqa: E402
from dms.pipeline import Pipeline  # noqa: E402

COLORS = {Severity.INFO: (255, 255, 0), Severity.WARNING: (0, 165, 255),
          Severity.CRITICAL: (0, 0, 255)}
PRIORITY_COLORS = {"info": (255, 255, 0), "warning": (0, 165, 255), "critical": (0, 0, 255)}
REPLY_SHOW_SECONDS = 8


def open_camera(cfg) -> cv2.VideoCapture:
    # CAP_DSHOW opens much faster on Windows
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(cfg.index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {cfg.index}. Try --camera 1")
    return cap


def put(frame, text, y, color=(255, 255, 255), scale=0.6):
    """Draw ONE line of text on a dark semi-transparent strip (readable on any background)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, 1)
    x1, y1, x2, y2 = 6, y - th - 6, min(frame.shape[1], 14 + tw), y + base + 2
    roi = frame[max(0, y1):y2, x1:x2]
    roi[:] = (roi * 0.4).astype(roi.dtype)      # darken behind the text
    cv2.putText(frame, text, (10, y), font, scale, color, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", default="drowsiness,head_pose",
                    help="comma-separated: drowsiness,head_pose,phone,gestures")
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--llm", action="store_true", help="enable the LLM assistant (key in .env)")
    args = ap.parse_args()

    config = Config()
    if args.camera is not None:
        config.camera.index = args.camera

    pipeline = Pipeline([n.strip() for n in args.detectors.split(",") if n.strip()], config)
    cap = open_camera(config.camera)

    LOGS_DIR.mkdir(exist_ok=True)
    log_path = LOGS_DIR / f"session_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    log = log_path.open("w", encoding="utf-8")
    log_lock = threading.Lock()   # the LLM thread also writes to the log

    def write_log(line: str):
        with log_lock:
            if not log.closed:           # a late LLM reply may arrive after quitting
                log.write(line + "\n"); log.flush()

    # ---- optional LLM assistant ----
    context = agent = None
    latest_reply = {"reply": None, "shown_at": 0.0}
    if args.llm:
        from dms.agent import Agent, DriverContext, LLMClient

        def on_reply(r):   # runs in the agent's background thread
            tag = {"llm": f"LLM {r.latency:.2f}s", "rule": "instant rule"}.get(
                r.source, f"FALLBACK: {r.error}")
            print(f"[ASSISTANT] level {r.level} {r.priority.upper():8} action={r.action:15} "
                  f"({tag})\n            \"{r.speak}\"\n            dashboard: {r.dashboard_note}")
            write_log(r.to_json())
            latest_reply.update(reply=r, shown_at=time.monotonic())

        llm = LLMClient()
        context = DriverContext()
        agent = Agent(context, on_reply, llm)
        print(f"LLM assistant ON: provider={llm.provider} model={llm.model}")

    print(f"Logging events to {log_path}\nCalibrating for {config.calibration.seconds}s "
          "- look at the road with eyes open...")

    recent: list[tuple[float, str, Severity]] = []
    fps, last = 0.0, time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame failed"); break
            if config.camera.mirror:
                frame = cv2.flip(frame, 1)

            signals, events = pipeline.process(frame)
            for e in events:
                print(f"[EVENT] {e.severity.value.upper():8} {e.type:16} {e.data}")
                write_log(e.to_json())
                recent.append((time.monotonic(), f"{e.type} {e.data}", e.severity))
            if agent and not signals["calibrating"]:
                context.update(signals, events)
                for e in events:
                    agent.handle(e)      # returns immediately; LLM runs in background

            # ---- debug overlay ----
            pipeline.draw(frame)
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6); last = now
            put(frame, f"FPS {fps:.0f}" + (f"   |   attention {context.attention:.0f}/100"
                                           if context else ""), 25)
            if signals["calibrating"]:
                put(frame, "CALIBRATING - look at the road, eyes open", 55, (0, 255, 255), 0.8)
            y = 85
            for name, out in pipeline.outputs.items():
                put(frame, out.status, y); y += 25
            recent = [r for r in recent if now - r[0] < 4]
            for _, text, sev in recent[-4:]:
                put(frame, text, y, COLORS[sev], 0.7); y += 28

            # ---- latest assistant message, bottom of the screen ----
            r = latest_reply["reply"]
            if r and now - latest_reply["shown_at"] < REPLY_SHOW_SECONDS:
                lines = textwrap.wrap(f"ASSISTANT: {r.speak}", 60)
                yb = frame.shape[0] - 20 - 32 * (len(lines) - 1)
                for line in lines:
                    put(frame, line, yb, PRIORITY_COLORS.get(r.priority, (255, 255, 255)), 0.8)
                    yb += 32

            cv2.imshow("Driver Monitor (q=quit, c=recalibrate)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                pipeline.recalibrate(); print("Recalibrating...")
    finally:
        cap.release(); cv2.destroyAllWindows(); pipeline.close()
        with log_lock:
            log.close()


if __name__ == "__main__":
    main()
