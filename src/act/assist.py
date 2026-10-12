"""Opt-in action assists applied between the policy and the machine (scripts/control_loop.py).

  GripLatch  --grip-latch: once the policy clamps the thumb past a threshold, hold a full clamp for a while
"""
import math
import time

import numpy as np


class GripLatch:
    """Grip latch on one channel (the thumb; + = clamp in). Applied to the action a controller's tick() returned,
    AFTER the policy and BEFORE sending, for every backend.

      trigger    status "ok" (the policy's own action, deadman held), no latch active, refractory over, and the
                 policy's thumb >= threshold. Never on an open command (thumb < 0): only a clamp latches.
      latched    sent thumb = +1.0 (full clamp) for hold_s seconds; every other channel is the policy's, unchanged.
      hold over  thumb back to the policy; no new latch for refractory_s (counted from the end of the hold).
      cancel     any tick that is not the policy acting -- status other than "ok" (deadman, STOP: ...,
                 starved), allowed False, or an all-zero (neutral) action -- ends the latch IMMEDIATELY
                 and the action goes out exactly as the controller returned it (the neutral it would have been).
                 A cancel does not start the refractory period: the policy must clamp past the threshold again.
    The input action is never modified (a copy is returned). clock: injectable for tests."""

    COLUMNS = ["latch_active", "latch_remaining_s", "latch_policy_thumb"]

    def __init__(self, index, threshold=0.3, hold_s=3.5, refractory_s=5.0, clock=time.monotonic):
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"grip latch threshold must be in (0, 1], got {threshold}")
        if not hold_s > 0 or not refractory_s >= 0:
            raise ValueError(f"grip latch hold must be > 0 s and refractory >= 0 s, got {hold_s}, {refractory_s}")
        self.index, self.threshold, self.hold_s, self.refractory_s, self.clock = (
            int(index), float(threshold), float(hold_s), float(refractory_s), clock)
        self.until, self.refractory_until = None, -math.inf     # hold end (None = not latched), next allowed trigger
        self.fired = self.cancelled = 0
        self.policy_thumb = None

    @property
    def active(self):
        return self.until is not None

    def remaining(self):
        return 0.0 if self.until is None else max(0.0, self.until - self.clock())

    def apply(self, action, status, allowed):
        """-> the action to send (a new array)."""
        now = self.clock()
        a = np.array(action, dtype=float, copy=True)
        self.policy_thumb = float(a[self.index])
        if status != "ok" or not allowed or not np.any(a):
            if self.until is not None:
                self.until = None
                self.cancelled += 1
            return a
        if self.until is not None and now >= self.until:
            self.until, self.refractory_until = None, self.until + self.refractory_s
        if self.until is None and now >= self.refractory_until and a[self.index] >= self.threshold:
            self.until = now + self.hold_s
            self.fired += 1
        if self.until is not None:
            a[self.index] = 1.0
        return a

    def tick_columns(self):
        """latch_active, latch_remaining_s, latch_policy_thumb (the policy's thumb before the latch) for the tick log."""
        return [int(self.active), round(self.remaining(), 3),
                "" if self.policy_thumb is None else round(self.policy_thumb, 4)]
