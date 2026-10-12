"""Action execution for remote policies.

  ChunkedController  ask ahead, drop stale steps, watchdog (every backend)
"""
import threading
import time

import numpy as np


class ChunkedController:
    """Action queue keyed by absolute control step. A chunk computed from the observation at step k
    maps its i-th action to step k+i; anything for steps already executed is dropped as stale."""

    def __init__(self, policy, safe_action_fn, request_at=5, max_use=25, stop_after_s=1.0, fps=10):
        self.policy, self.safe_action_fn = policy, safe_action_fn
        self.request_at, self.max_use, self.stop_after, self.fps = request_at, max_use, stop_after_s, fps
        self.queue, self.lock = {}, threading.Lock()
        self.step, self.in_flight = 0, False
        self.last_ok = self.last_action = time.monotonic()
        self.log, self.starved, self.errors = [], 0, 0
        self.planned = None                        # the policy's action for the last tick (None = queue empty)

    def reset(self):
        with self.lock:
            self.queue.clear()
        self.policy.reset()

    def remaining(self):
        with self.lock:
            return sum(1 for k in self.queue if k > self.step)

    def _request(self, obs, prompt, k_obs):
        t0 = time.monotonic()
        try:
            chunk = self.policy.infer(obs, prompt)[: self.max_use]
            k_now = self.step
            first = max(k_obs, k_now)              # steps before k_now have already been executed
            stale = first - k_obs
            with self.lock:
                for k in list(self.queue):
                    if k >= first:
                        del self.queue[k]          # newer information replaces the old plan
                for i, a in enumerate(chunk):
                    if k_obs + i >= first:
                        self.queue[k_obs + i] = a
            self.last_ok = time.monotonic()
            self.log.append({"t": round(t0, 3), "k_obs": k_obs, "rtt_ms": round(1e3 * (time.monotonic() - t0), 1),
                             "stale_steps": stale, "chunk_len": len(chunk), "ok": 1,
                             "payload_bytes": getattr(self.policy, "last_payload_bytes", 0),
                             "server_ms": getattr(self.policy, "last_server_ms", None)})
        except Exception as e:  # network errors must never kill the control loop
            self.errors += 1
            self.log.append({"t": round(t0, 3), "k_obs": k_obs, "rtt_ms": round(1e3 * (time.monotonic() - t0), 1),
                             "stale_steps": None, "chunk_len": 0, "ok": 0, "error": repr(e)[:120]})
        finally:
            self.in_flight = False

    def tick(self, obs, prompt, allowed=True):
        """Returns (action, status). Call once per control step with the fresh observation."""
        if not self.in_flight and self.remaining() <= self.request_at:
            self.in_flight = True
            snap = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
            threading.Thread(target=self._request, args=(snap, prompt, self.step), daemon=True).start()
        with self.lock:
            a = self.queue.pop(self.step, None)
            for k in [k for k in self.queue if k < self.step]:
                del self.queue[k]
        self.step += 1
        self.planned = a
        if not allowed:
            return self.safe_action_fn(obs), "deadman"
        now = time.monotonic()
        if a is None:
            self.starved += 1
            if now - self.last_action > self.stop_after:
                return self.safe_action_fn(obs), "STOP: no chunk"   # watchdog: link lost / server stuck
            return self.safe_action_fn(obs), "starved"
        self.last_action = now
        return a, "ok"

