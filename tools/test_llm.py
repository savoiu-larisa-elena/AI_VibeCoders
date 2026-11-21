"""Test the LLM part WITHOUT a camera: sends fake driver situations, prints the replies.

    python tools/test_llm.py
Checks: API key works, latency, reply format, escalation, fallback.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dms.agent import Agent, DriverContext, LLMClient  # noqa: E402
from dms.events import Event, Severity  # noqa: E402

llm = LLMClient()
print(f"provider={llm.provider}  model={llm.model}\n")

done = []
ctx = DriverContext()
agent = Agent(ctx, on_reply=done.append, llm=llm)

scenarios = [
    ("Yawning",          {"drowsiness": {"perclos": 0.12, "blinks_per_min": 22}, "head_pose": {"looking": "road"}},
     Event("yawn", Severity.WARNING, "drowsiness", data={"mar": 0.9})),
    ("Looking down 2.5s", {"drowsiness": {"head_down": True}, "head_pose": {"looking": "down", "away_seconds": 2.5}},
     Event("distraction", Severity.CRITICAL, "head_pose", data={"direction": "down", "seconds": 2.5})),
    ("Phone in hand",    {"phone": {"phone_visible": True}, "head_pose": {"looking": "down"}},
     Event("phone", Severity.CRITICAL, "phone", data={"seconds": 1.2})),
    ("Eyes closed",      {"drowsiness": {"eyes_closed": True, "closed_seconds": 1.3, "perclos": 0.28}},
     Event("eyes_closed", Severity.CRITICAL, "drowsiness", data={"seconds": 1.3})),
    ("Eyes closed AGAIN", {"drowsiness": {"eyes_closed": True, "closed_seconds": 1.6, "perclos": 0.31}},
     Event("eyes_closed", Severity.CRITICAL, "drowsiness", data={"seconds": 1.6})),
    ("Thumbs up (I'm OK)", {"head_pose": {"looking": "road"}},
     Event("gesture", Severity.INFO, "gestures", data={"gesture": "Thumb_Up"})),
    ("Open palm (mute)",  {"head_pose": {"looking": "road"}},
     Event("gesture", Severity.INFO, "gestures", data={"gesture": "Open_Palm"})),
    ("Yawn while muted",  {"drowsiness": {"perclos": 0.1}},
     Event("yawn", Severity.WARNING, "drowsiness", data={"mar": 0.8})),
    ("Thumbs down (break)", {"head_pose": {"looking": "road"}},
     Event("gesture", Severity.INFO, "gestures", data={"gesture": "Thumb_Down"})),
    ("Victory (status)",  {"head_pose": {"looking": "road"}},
     Event("gesture", Severity.INFO, "gestures", data={"gesture": "Victory"})),
]

for name, signals, ev in scenarios:
    ctx.update(signals, [ev])
    agent._last_call = 0          # test only: skip the cooldown between scenarios
    agent.handle(ev)
    t0 = time.time()
    while not done and time.time() - t0 < 10:
        time.sleep(0.05)
    if not done:
        print(f"--- {name}: no reply (dropped by rules)\n")
        continue
    r = done.pop()
    tag = {"llm": f"LLM {r.latency:.2f}s", "rule": "instant rule"}.get(
        r.source, f"FALLBACK ({r.error})")
    print(f"--- {name}  [level {r.level}, {r.priority}, action={r.action}]  {tag}")
    print(f"    \"{r.speak}\"")
    print(f"    dashboard: {r.dashboard_note}\n")
