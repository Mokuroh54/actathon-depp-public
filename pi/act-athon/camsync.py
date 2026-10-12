"""camsync -- one clock and one frame-selection rule for every camera. Standard library only.

Used by camview.py (network relay: maps a sender's capture times onto this Pi's clock), dashboard.py (recorder) and
runner/act/cameras.py (policy runner), so training and inference pick frames the SAME way.

Clock: every frame carries X-Capture-Time, epoch s on THIS Pi's wall clock (time.time()):
  - Pi cameras (USB / OAK): when the frame reached camview.
  - network cameras whose sender stamps frames and serves GET /api/time (worldcam/worldcam.py on a laptop): the
    sender's capture time, mapped onto the Pi clock by ClockSync (NTP-style over HTTP, error <= round trip / 2).
    Senders without /api/time keep camview's arrival time (network delay not removed).

Selection (select()), one step at a time, over each camera's recent frames (FrameRing):
  aligned  t_ref = the OLDEST of the cameras' newest capture times: the camera that lags the most (normally the
           network world camera) sets the reference, and every camera gets its frame captured nearest t_ref. All
           images of a step show the same instant; they are as old as the laggiest camera.
  latest   every camera's newest frame (the pre-sync behaviour): freshest images, but the world camera's image is
           its network delay older than the onboard ones.
A camera whose newest frame is older than max_lag is left out of t_ref (it would drag every camera back) and gets its
newest frame; the callers' own freshness checks flag it. Record and run with the SAME mode (the episode meta.json
"sync" block says which).
"""
import collections
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

SYNC_MODES = ("aligned", "latest")
DEFAULT_MODE = "aligned"
MAX_LAG_S = 0.5             # a camera this far behind is not waited for (= the converter's / QC's --max-age)
RING_KEEP_S = 1.5           # FrameRing history (must exceed the largest lag between cameras)
RING_MAX = 90               # ... and at most this many frames (~3 s at 30 fps)


# ---------------------------------------------------------------- clock
def base_url(url):
    """http://host:port/stream/world -> http://host:port"""
    u = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((u.scheme, u.netloc, "", "", ""))


def probe(base, timeout=1.0):
    """One /api/time probe -> (offset = remote - local, rtt), both s."""
    t0 = time.time()
    with urllib.request.urlopen(base + "/api/time", timeout=timeout) as r:
        tr = float(json.loads(r.read())["t"])
    t1 = time.time()
    return tr - (t0 + t1) / 2, t1 - t0


class ClockSync:
    """remote clock - local clock for one sender, kept up to date by a daemon thread.
    Every EVERY_S a burst of BURST probes of <base>/api/time; the probe with the smallest round trip is that burst's
    sample (offset, error = rtt / 2). The estimate is the lowest-error sample of the last WINDOW bursts (~1 min: clock
    drift over that is a few ms), unless the newest sample disagrees with it by more than both errors + STEP_TOL_S
    (the sender's clock was stepped, e.g. by NTP): then the history restarts from the newest sample.
    to_local(t) maps a sender timestamp onto this machine's clock; None until the first burst succeeded or when the
    sender has no /api/time (`status` says why)."""

    EVERY_S, BURST, WINDOW, STEP_TOL_S, FRESH_S = 5.0, 8, 12, 0.005, 30.0

    def __init__(self, base, name="clock"):
        self.base = base.rstrip("/")
        self.samples = collections.deque(maxlen=self.WINDOW)
        self.est = None             # (offset, err, t_sample)
        self.status = "measuring"
        self.steps = 0              # sender clock steps seen
        self.stop_ev = threading.Event()
        threading.Thread(target=self._run, daemon=True, name="clocksync-" + name).start()

    def _burst(self):
        best = None
        for _ in range(self.BURST):
            try:
                off, rtt = probe(self.base)
            except urllib.error.HTTPError as ex:
                if ex.code == 404:
                    raise
                continue
            except (OSError, ValueError, KeyError):
                continue
            if best is None or rtt < best[1]:
                best = (off, rtt)
            self.stop_ev.wait(0.02)
        return best

    def _run(self):
        while not self.stop_ev.is_set():
            try:
                b = self._burst()
            except urllib.error.HTTPError:
                self.status = "sender has no /api/time: arrival times used"
                self.est = None
                self.stop_ev.wait(60.0)
                continue
            now = time.time()
            if b is None:
                self.status = "no /api/time answer from %s" % self.base
            else:
                s = (b[0], b[1] / 2, now)
                cur = self.est
                if cur is not None and abs(s[0] - cur[0]) > s[1] + cur[1] + self.STEP_TOL_S:
                    self.samples.clear()
                    self.steps += 1
                self.samples.append(s)
                self.est = min(self.samples, key=lambda x: x[1])
                self.status = "ok"
            if self.est is not None and now - self.est[2] > self.FRESH_S + self.WINDOW * self.EVERY_S:
                self.est = None
            self.stop_ev.wait(self.EVERY_S)

    def to_local(self, t_remote):
        e = self.est
        return None if e is None else t_remote - e[0]

    def info(self):
        e = self.est
        return {"status": self.status, "offset_ms": None if e is None else round(e[0] * 1e3, 2),
                "err_ms": None if e is None else round(e[1] * 1e3, 2),
                "sample_age_s": None if e is None else round(time.time() - e[2], 1), "steps": self.steps}

    def stop(self):
        self.stop_ev.set()


# ---------------------------------------------------------------- frames
class FrameRing:
    """Recent frames of one camera, oldest first: (t_capture on this machine's wall clock, item). Thread-safe."""

    def __init__(self, keep_s=RING_KEEP_S, maxlen=RING_MAX):
        self.keep_s = keep_s
        self.buf = collections.deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def push(self, t, item):
        with self.lock:
            if self.buf and t < self.buf[-1][0]:     # never let the order break (a clock re-estimate moved back)
                t = self.buf[-1][0]
            self.buf.append((t, item))
            while self.buf and t - self.buf[0][0] > self.keep_s:
                self.buf.popleft()

    def newest(self):
        with self.lock:
            return self.buf[-1] if self.buf else None

    def nearest(self, t):
        """Frame captured nearest t (ties -> the older one)."""
        with self.lock:
            best = None
            for x in reversed(self.buf):
                if best is not None and abs(x[0] - t) > abs(best[0] - t):
                    break                              # capture times only grow: moving away from t now
                if best is None or abs(x[0] - t) <= abs(best[0] - t):
                    best = x
            return best

    def clear(self):
        with self.lock:
            self.buf.clear()


def select(rings, mode=DEFAULT_MODE, max_lag=MAX_LAG_S, now=None):
    """rings: {name: FrameRing}. -> ({name: (t_capture, item) or None}, info) where info = {"mode", "t_ref" (wall s,
    None if no camera has a frame), "anchor" (the camera that set t_ref), "skew" (s, newest - oldest chosen capture
    time), "left_out" ([cameras older than max_lag])}."""
    if mode not in SYNC_MODES:
        raise ValueError("sync mode %r: one of %s" % (mode, ", ".join(SYNC_MODES)))
    now = time.time() if now is None else now
    newest = {n: r.newest() for n, r in rings.items()}
    live = {n: f for n, f in newest.items() if f is not None and now - f[0] <= max_lag}
    left_out = sorted(n for n, f in newest.items() if f is not None and n not in live)
    if mode == "latest" or not live:
        chosen, anchor = newest, None
        t_ref = max((f[0] for f in live.values()), default=None)
    else:
        anchor = min(live, key=lambda n: live[n][0])
        t_ref = live[anchor][0]
        chosen = {n: (rings[n].nearest(t_ref) if n in live else newest[n]) for n in rings}
    ts = [chosen[n][0] for n in live if chosen.get(n) is not None]
    return chosen, {"mode": mode, "t_ref": t_ref, "anchor": anchor,
                    "skew": (max(ts) - min(ts)) if len(ts) >= 2 else 0.0, "left_out": left_out}
