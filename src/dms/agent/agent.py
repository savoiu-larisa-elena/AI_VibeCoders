"""The in-car assistant: decides WHEN to call the LLM, and handles the reply.

Split of responsibilities (say this to the jury):
  * code decides WHEN (rules, cooldowns) and HOW SERIOUS (escalation level) -> predictable, safe
  * the LLM decides WHAT TO SAY, using the full context              -> natural, situational
  * if the LLM is slow or offline, a pre-written fallback is used     -> never silent
"""
from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field

from ..events import Event, Severity
from .context import DriverContext
from .llm import LLMClient, LLMError

SYSTEM_PROMPT = """You are the in-car safety assistant of a driver-monitoring system.
A camera watches the driver; you receive a JSON summary of what it detected.
Your job: write ONE short message that is spoken to the driver right now.

Rules:
- Max 25 words. Spoken aloud while driving: simple, direct, no lists, no emojis.
- Match the urgency to "escalation_level" (1 = gentle, 2 = firm, 3 = urgent, 4 = insist on stopping).
- Address the TRIGGER first. Mention other context only if it is the same kind of problem
  (e.g. repeated yawns + eyes closing = fatigue pattern).
- Pick the action that fits the trigger:
    phone -> put_phone_away;  distraction -> eyes_on_road;  hands_off_wheel -> hands_on_wheel;
    yawn / fatigue_perclos -> suggest_break;  eyes_closed -> eyes_on_road (level 1-2) or pull_over (level 3-4).
  Only use pull_over for fatigue at level 3-4.
- If "passengers" > 0 and the trigger is fatigue at escalation_level 3-4, you may also ask
  the passenger to help (keep the driver talking, help find a place to stop).
- Only use numbers that appear in the JSON. Never invent times, counts or scores.
  "session_minutes" is the drive time (e.g. 0.4 = less than a minute).
- Never repeat the exact wording from "previous_messages".
- Never blame or lecture. Be calm, caring and concrete (what to do now).
- If the trigger is a gesture, follow its "meaning" (priority "info"):
    acknowledge   -> short thank-you, max 8 words, action "none".
    need_break    -> supportive, suggest stopping at the next safe place, action "suggest_break".
    status_report -> max 35 words: drive time, attention score, notable alerts this session,
                     one practical tip. Action "none".

Reply ONLY with JSON:
{"speak": "<message>",
 "priority": "info" | "warning" | "critical",
 "action": "none" | "eyes_on_road" | "put_phone_away" | "hands_on_wheel" | "suggest_break" | "pull_over" | "check_in",
 "dashboard_note": "<max 12 words, technical summary for the dashboard>"}"""

ACTIONS = {"none", "eyes_on_road", "put_phone_away", "hands_on_wheel", "suggest_break",
           "pull_over", "check_in"}

# used when the LLM is slow / offline / returns garbage
FALLBACKS = {
    "eyes_closed": ("Wake up! Your eyes were closed. Keep your eyes on the road.", "critical", "eyes_on_road"),
    "fatigue_perclos": ("You're showing signs of fatigue. Plan a break soon.", "warning", "suggest_break"),
    "yawn": ("You seem tired. Consider a short break.", "warning", "suggest_break"),
    "distraction": ("Eyes on the road, please.", "warning", "eyes_on_road"),
    "phone": ("Please put the phone away while driving.", "critical", "put_phone_away"),
    "hands_off_wheel": ("Please put your hands back on the steering wheel.", "warning", "hands_on_wheel"),
    "acknowledge": ("Got it, thanks.", "info", "none"),
    "need_break": ("Understood. Let's stop at the next safe place for a break.", "info", "suggest_break"),
    "status_report": ("Status report is unavailable right now. Keep up the focus.", "info", "none"),
}


NAMES = {"eyes_closed": "eye closure", "yawn": "yawn", "distraction": "distraction",
         "phone": "phone alert", "fatigue_perclos": "fatigue warning",
         "hands_off_wheel": "hands-off-wheel alert"}


def local_status_report(payload: dict) -> str:
    """Status report built from the real numbers - used when the LLM is unavailable,
    so the driver still gets an answer instead of 'not available'."""
    st = payload.get("driver_state", {})
    minutes = payload.get("session_minutes", 0) or 0
    drive = f"{round(minutes)} minutes" if minutes >= 1 else "less than a minute"
    counts = payload.get("counts_session", {}) or {}
    parts = [f"{n} {NAMES.get(k, k.replace('_', ' '))}{'s' if n > 1 else ''}"
             for k, n in sorted(counts.items(), key=lambda kv: -kv[1])[:3]]
    alerts = ("So far: " + ", ".join(parts) + ".") if parts else "No alerts so far."
    return (f"Status: driving for {drive}, attention {st.get('attention_score', 'unknown')} "
            f"out of 100. {alerts} Keep your eyes on the road.")


@dataclass
class Reply:
    speak: str
    priority: str
    action: str
    dashboard_note: str
    trigger: str
    level: int
    source: str            # "llm" or "fallback"
    latency: float = 0.0
    error: str = ""
    ts: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps({"type": "assistant_reply", **self.__dict__})


class Agent:
    # which events go to the LLM, and the minimum gap between LLM calls for that severity
    MIN_GAP = {Severity.CRITICAL: 3.0, Severity.WARNING: 12.0}
    # gesture -> meaning sent to the LLM (Open_Palm is handled locally: instant mute)
    GESTURE_MEANING = {"Thumb_Up": "acknowledge", "Thumb_Down": "need_break",
                       "Victory": "status_report"}
    SNOOZE_SECONDS = 60

    def __init__(self, context: DriverContext, on_reply, llm: LLMClient | None = None):
        self.context = context
        self.on_reply = on_reply           # callback(Reply), called from the worker thread
        self.llm = llm or LLMClient()
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._last_call = 0.0
        self._last_alert_ts = 0.0
        self._alerts: list[tuple[float, str]] = []   # (timestamp, category) of alerts
        self._previous: list[str] = []     # last messages, so the LLM doesn't repeat itself
        self._busy = threading.Event()
        self._facts_enabled = lambda: False
        self.snoozed_until = 0.0
        threading.Thread(target=self._worker, daemon=True).start()

    def request_fun_fact(self, reason: str) -> bool:
        """Low-priority request: never displace pending safety guidance."""
        now = time.time()
        if (now < self.snoozed_until or self._busy.is_set() or not self._queue.empty()
                or now - self._last_alert_ts < 12 or now - self._last_call < 8):
            return False
        ev = Event("fun_fact", Severity.INFO, "engagement", data={"reason": reason})
        self._submit(ev, 1, now)
        return True

    # ---- escalation: decided by CODE, not by the LLM ----
    # Alerts only escalate within the same category: a yawn followed by a phone
    # is two different problems, not one getting worse.
    CATEGORY = {"eyes_closed": "fatigue", "yawn": "fatigue", "fatigue_perclos": "fatigue",
                "distraction": "distraction", "phone": "distraction",
                "hands_off_wheel": "distraction"}
    MAX_LEVEL = {"fatigue": 4, "distraction": 3}   # only fatigue escalates to "pull over"

    def _level(self, ev: Event) -> int:
        now = time.time()
        cat = self.CATEGORY.get(ev.type, "other")
        recent = [t for t, c in self._alerts if c == cat and now - t < 300]
        level = 1 + len(recent)                 # each recent same-kind alert raises the level
        if ev.severity == Severity.CRITICAL:
            level += 1
        if ev.type == "eyes_closed" and self.context.count("eyes_closed", 120) >= 2:
            level = 4                           # repeated microsleeps -> insist on stopping
        return max(1, min(self.MAX_LEVEL.get(cat, 2), level))

    def acknowledge(self):
        """Driver confirmed (thumbs up): calm down escalation by forgetting the oldest alert."""
        if self._alerts:
            self._alerts.pop(0)

    # ---- called from the main loop for every new event ----
    def handle(self, ev: Event) -> None:
        now = time.time()
        if ev.type == "gesture":
            self._handle_gesture(ev, now)
            return
        if ev.severity not in self.MIN_GAP:
            return                              # info events (drink...) are dashboard-only
        if ev.severity == Severity.WARNING and now < self.snoozed_until:
            return                              # driver muted warnings (open palm)
        if now - self._last_call < self.MIN_GAP[ev.severity] or self._busy.is_set():
            if ev.severity != Severity.CRITICAL:
                return                          # drop: we just spoke / are speaking
        level = self._level(ev)
        self._alerts.append((now, self.CATEGORY.get(ev.type, "other")))
        self._last_alert_ts = now
        self._submit(ev, level, now)

    def _handle_gesture(self, ev: Event, now: float) -> None:
        g = ev.data.get("gesture")
        if g == "Open_Palm":
            # instant, no LLM needed: mute warnings (critical alerts still get through)
            self.snoozed_until = now + self.SNOOZE_SECONDS
            self.on_reply(Reply("Okay, I'll stay quiet for a minute unless it's urgent.", "info",
                                "none", f"Warnings muted {self.SNOOZE_SECONDS}s by driver",
                                "gesture", 1, "rule"))
            return
        if g == "Thumb_Up" and now < self.snoozed_until:
            self.unmute()               # thumbs up while muted = "warnings back on"
            return
        meaning = self.GESTURE_MEANING.get(g)
        if not meaning:
            return
        if meaning == "acknowledge":
            self.acknowledge()
        self._submit(ev, level=1, now=now, meaning=meaning)

    def unmute(self) -> bool:
        """End the open-palm mute early (thumbs up, or the dashboard's UNMUTE button)."""
        if time.time() >= self.snoozed_until:
            return False
        self.snoozed_until = 0.0
        self.on_reply(Reply("Okay, warnings are back on.", "info", "none",
                            "Warnings unmuted by driver", "gesture", 1, "rule"))
        return True

    def _submit(self, ev: Event, level: int, now: float, meaning: str | None = None) -> None:
        payload = self.context.summary(ev)
        if meaning:
            payload["trigger"]["meaning"] = meaning
        payload["escalation_level"] = level
        payload["previous_messages"] = self._previous[-3:]
        self._last_call = now
        try:
            self._queue.get_nowait()            # keep only the newest request
        except queue.Empty:
            pass
        self._queue.put((ev, level, payload))

    # ---- background thread: the camera loop never waits for the network ----
    def _worker(self):
        while True:
            ev, level, payload = self._queue.get()
            self._busy.set()
            try:
                reply = self._ask(ev, level, payload)
            finally:
                self._busy.clear()
            if ev.type == "fun_fact" and (time.time() < self.snoozed_until or
                    time.time() - self._last_alert_ts < 12 or not self._facts_enabled()):
                continue
            self._previous.append(reply.speak)
            self.on_reply(reply)

    def _ask(self, ev: Event, level: int, payload: dict) -> Reply:
        try:
            prompt = SYSTEM_PROMPT
            if ev.type == "fun_fact":
                if ev.data.get("reason") == "driver_checkin":
                    prompt = """You are a friendly in-car driving assistant.
The camera has detected no visible driver motion for at least 10 seconds.
Ask ONE short, calm, caring check-in question such as 'Hey, are you okay?'
or 'Everything alright?'. Do not imply the driver is unconscious or asleep.
Do not ask them to look away from the road or make a gesture.
Avoid repeating previous_messages. Maximum 12 words.
Return ONLY JSON with keys speak, priority, action, dashboard_note.
Set priority to info and action to none."""
                else:
                    prompt = """You are a friendly in-car assistant. Return ONLY JSON with keys
speak, priority, action, dashboard_note. Say ONE interesting, accurate, family-friendly
random fun fact in at most 22 words. Do not ask the driver to look away, move hands,
turn their head, or perform any action. Avoid repeating previous_messages.
Set priority to info and action to none. Keep it brief and pleasant."""
            data, latency = self.llm.chat_json(prompt, json.dumps(payload))
            speak = str(data.get("speak", "")).strip()
            if not speak:
                raise LLMError("empty 'speak'")
            action = data.get("action") if data.get("action") in ACTIONS else "none"
            if ev.type == "fun_fact":
                action = "none"
            priority = data.get("priority") if data.get("priority") in ("info", "warning", "critical") \
                else ev.severity.value
            if ev.type == "fun_fact":
                priority = "info"
            return Reply(speak, priority, action, str(data.get("dashboard_note", ""))[:120],
                         ev.type, level, "llm", latency)
        except LLMError as e:
            key = payload["trigger"].get("meaning", ev.type)
            text, priority, action = FALLBACKS.get(key, ("Please stay focused on the road.",
                                                             ev.severity.value, "eyes_on_road"))
            if ev.type == "fun_fact":
                text = ("Hey, are you doing okay?" if ev.data.get("reason") == "driver_checkin"
                        else "Fun fact: Octopuses have three hearts.")
                priority, action = "info", "none"
            if key == "status_report":
                text = local_status_report(payload)   # real numbers, no LLM needed
            return Reply(text, priority, action, f"fallback: {ev.type}", ev.type, level,
                         "fallback", error=str(e)[:200])
