"""Opt-in engagement prompts. Landmark-based motion is heuristic, not a safety detector."""
import time


class MotionTracker:
    def __init__(self):
        self.last_motion = time.monotonic()
        self.previous_face = None
        self.previous_hands = None
        self.previous_closed = None
        self.still_prompted = False

    def update(self, pipeline, signals, now):
        # The pipeline caches MediaPipe results; no additional inference needed.
        face_result = pipeline._last_results.get("face")
        hands_result = pipeline._last_results.get("hands")
        faces = getattr(face_result, "face_landmarks", None) or []
        hands = getattr(hands_result, "hand_landmarks", None) or []
        if not faces:
            self.last_motion = now
            self.previous_face = self.previous_hands = self.previous_closed = None
            self.still_prompted = False
            return False

        # Stable landmark coordinates: eyes, nose, mouth, chin and face edges.
        face = faces[0]
        face_points = [(face[i].x, face[i].y) for i in (1, 4, 33, 133, 263, 362, 61, 291, 152)]
        hand_points = [[(hand[i].x, hand[i].y) for i in (0, 4, 8, 12, 16, 20)] for hand in hands]
        closed = (signals.get("drowsiness") or {}).get("eyes_closed")
        moved = False
        if self.previous_face is not None:
            displacement = sum(((x-a)**2+(y-b)**2)**0.5 for (x,y),(a,b) in zip(face_points,self.previous_face))/len(face_points)
            moved |= displacement > 0.009
        if self.previous_hands is not None:
            if len(hand_points) != len(self.previous_hands):
                moved = True
            else:
                for hand, old in zip(hand_points, self.previous_hands):
                    displacement = sum(((x-a)**2+(y-b)**2)**0.5 for (x,y),(a,b) in zip(hand,old))/len(hand)
                    moved |= displacement > 0.012
        if closed is not None and self.previous_closed is not None and closed != self.previous_closed:
            moved = True  # blink/eye closure transition
        self.previous_face, self.previous_hands, self.previous_closed = face_points, hand_points, closed
        if moved:
            self.last_motion = now
            self.still_prompted = False
        return True


class FunFactScheduler:
    """One engagement prompt per 5-second inactivity interval.

    When both options are enabled, alternate the prompt type. A rejected
    request (busy LLM, recent safety alert) is retried without flooding.
    """
    INTERVAL = 5.0

    def __init__(self):
        self.motion = MotionTracker()
        self.last_prompt = None
        self.next_attempt = 0.0
        self.next_kind = "no_motion"

    def tick(self, pipeline, signals, now, enabled_still, enabled_periodic, muted, agent):
        visible = self.motion.update(pipeline, signals, now)
        if not visible or signals.get("calibrating"):
            self.last_prompt = None
            return

        if muted:
            # Mute suspends prompts and starts a fresh inactivity period.
            self.motion.last_motion = now
            self.motion.still_prompted = False
            self.last_prompt = None
            self.next_attempt = 0.0
            return

        if not enabled_still and not enabled_periodic:
            self.last_prompt = None
            return

        # Any detected movement restarts the full 5-second countdown.
        if now - self.motion.last_motion < self.INTERVAL:
            self.last_prompt = None
            return

        if agent is None or now < self.next_attempt:
            return

        # First prompt at 5 seconds, then at most once per 5 seconds.
        if self.last_prompt is not None and now - self.last_prompt < self.INTERVAL:
            return

        if enabled_still and enabled_periodic:
            kind = self.next_kind
        elif enabled_still:
            kind = "no_motion"
        else:
            kind = "driver_checkin"

        if agent.request_fun_fact(kind):
            self.last_prompt = now
            self.next_attempt = 0.0
            if enabled_still and enabled_periodic:
                self.next_kind = ("driver_checkin" if kind == "no_motion" else "no_motion")
        else:
            # The LLM may be occupied with a more important safety event.
            self.next_attempt = now + 1.0
