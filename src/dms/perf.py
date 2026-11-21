"""Performance helpers.

boost_process(): on Windows 11 laptops with Intel hybrid CPUs (P-cores + E-cores), Windows
"power throttles" a process whose window is not in the foreground - e.g. the terminal
running the dashboard while you look at the browser. It then runs on the slow efficiency
cores: in our tests every model became 3-5x slower (hands 29 -> 7 ms, YOLO 377 -> 67 ms)
and the dashboard dropped to ~5-10 FPS. This opts the process out of that throttling and
raises its priority slightly. Does nothing on other operating systems.

FpsCounter: real frames per second over the last second (robust to the camera sometimes
handing over two frames back-to-back, which made a smoothed 1/dt counter spike).
"""
from __future__ import annotations

import sys
import time
from collections import deque


def boost_process() -> str:
    if sys.platform != "win32":
        return "not windows - nothing to do"
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # declare types: the process handle is pointer-sized (64-bit) - without this ctypes
        # truncates it to a 32-bit int and the calls fail
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                              wintypes.DWORD]
        k32.SetProcessInformation.restype = wintypes.BOOL
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.SetPriorityClass.restype = wintypes.BOOL

        class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG),
                        ("StateMask", wintypes.ULONG)]

        # ProcessPowerThrottling = 4, EXECUTION_SPEED = 1, StateMask 0 = never throttle
        state = PROCESS_POWER_THROTTLING_STATE(1, 1, 0)
        handle = k32.GetCurrentProcess()
        ok_throttle = k32.SetProcessInformation(handle, 4, ctypes.byref(state), ctypes.sizeof(state))
        ok_priority = k32.SetPriorityClass(handle, 0x8000)   # ABOVE_NORMAL_PRIORITY_CLASS
        return (f"power throttling off: {bool(ok_throttle)}, "
                f"priority above normal: {bool(ok_priority)}")
    except Exception as e:  # noqa: BLE001 - never block startup because of this
        return f"could not change process settings: {e}"


class FpsCounter:
    def __init__(self, window_s: float = 1.0):
        self.window_s = window_s
        self._ticks: deque[float] = deque()

    def tick(self) -> float:
        now = time.monotonic()
        self._ticks.append(now)
        while self._ticks and now - self._ticks[0] > self.window_s:
            self._ticks.popleft()
        if len(self._ticks) < 2:
            return 0.0
        span = self._ticks[-1] - self._ticks[0]
        return (len(self._ticks) - 1) / span if span > 0 else 0.0
