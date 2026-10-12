"""Camera helpers: threaded capture that always returns the newest frame, and --cam parsing.
Every camera also keeps `ring` (camsync.FrameRing of (capture time, RingFrame) for genuinely NEW frames only) and
frame_of(item), so RigMachine can pick one observation's frames with camsync.select -- the rule the dashboard
recorder used for the training episodes (--sync, docs/runner_sync.md)."""
import hashlib
import threading
import time
import urllib.request
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np

from act import camsync

CAMSYNC_SHA1 = "4b9ee4300557f6e1675c3aea899f984e4a8df21d"   # the Pi's ~/act-athon/camsync.py that act/camsync.py copies
CAMSYNC_MARK = b"# ---- verbatim ----\n"

# One ring entry: data = JPEG bytes (MjpegCamera) or BGR frame (ThreadedCamera), arrival = time.monotonic() when it
# arrived here (the freshness basis, same as `stamp`), seq = camview's X-Frame-Seq (None if not sent / device camera).
RingFrame = namedtuple("RingFrame", "data arrival seq")


def camsync_status(recorder_copy=None):
    """-> (ok, message). act/camsync.py must be the Pi's ~/act-athon/camsync.py byte for byte below its header: checks
    the sha1 it was copied at and, when present (on the Pi: <runner>/../camsync.py), the recorder's own copy."""
    ours = Path(camsync.__file__).read_bytes()
    body = ours.split(CAMSYNC_MARK, 1)[1] if CAMSYNC_MARK in ours else ours
    if hashlib.sha1(body).hexdigest() != CAMSYNC_SHA1:
        return False, f"{camsync.__file__} was edited (sha1 != {CAMSYNC_SHA1}): re-copy the Pi's camsync.py"
    rec = Path(recorder_copy) if recorder_copy else Path(camsync.__file__).resolve().parents[2] / "camsync.py"
    if not rec.is_file():
        return True, f"camsync {CAMSYNC_SHA1[:10]} (no recorder copy at {rec} to compare)"
    if rec.read_bytes() != body:
        return False, (f"camsync differs from the recorder's {rec} (sha1 {hashlib.sha1(rec.read_bytes()).hexdigest()}"
                       f" vs {CAMSYNC_SHA1}): the runner would pick frames by another rule than the recorder")
    return True, f"camsync {CAMSYNC_SHA1[:10]} = recorder's {rec}"


class ThreadedCamera:
    """Reads a camera continuously in a background thread and keeps only the newest frame,
    so get_observation() never blocks on the camera or returns a stale buffered frame."""

    def __init__(self, src, hw, fps=30, timeout_s=3.0):
        self.src = src
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"camera {src!r} failed to open")
        h, w = hw
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)   # ignored by streams/files, fine
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.frame, self.stamp = None, 0.0
        self.ring = camsync.FrameRing()                 # (time.time() at read, RingFrame(BGR frame, ...))
        self.lock, self.running = threading.Lock(), True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        t0 = time.monotonic()
        while self.frame is None:
            if time.monotonic() - t0 > timeout_s:
                self.release()
                raise RuntimeError(f"camera {src!r} opened but produced no frames")
            time.sleep(0.01)

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                now = time.monotonic()
                with self.lock:
                    self.frame, self.stamp = frame, now
                self.ring.push(time.time(), RingFrame(frame, now, None))
            else:
                time.sleep(0.005)

    def check_fresh(self, max_age_s=0.5):
        """Raises if the camera stopped delivering (no frame for max_age_s)."""
        with self.lock:
            stamp = self.stamp
        if time.monotonic() - stamp > max_age_s:
            raise RuntimeError(f"camera {self.src!r} stale: last frame {time.monotonic() - stamp:.2f}s ago")

    def read(self, max_age_s=0.5):
        """Newest BGR frame; raises if the camera stopped delivering."""
        with self.lock:
            frame = self.frame
        self.check_fresh(max_age_s)
        return frame

    @staticmethod
    def frame_of(item):
        """A RingFrame from `ring` -> its BGR frame."""
        return item.data

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


class MjpegCamera:
    """Reads a camview.py stream (multipart MJPEG with Content-Length, http://<pi>:8100/stream/<id>) with the standard
    library in a background thread, keeps the newest JPEG and decodes it on read(). Same interface as ThreadedCamera.
    Used instead of cv2.VideoCapture for these URLs: FFmpeg buffers HTTP streams, so its frames can lag. The JPEG bytes
    are camview's, decoded with cv2.imdecode, the same as the recorder -> converter path.

    Freshness is the SAFETY check here: read() must raise when the camera stops producing new images. A part that
    arrives is not necessarily a new image -- older camview re-sends its last JPEG every ~2 s when the camera freezes --
    so only a genuinely NEW part refreshes `stamp` (time.monotonic() at its arrival):
      * part has X-Frame-Seq: new iff its seq differs from the previous part's (lower = camview restarted = new);
      * no X-Frame-Seq: new iff its bytes differ from the previous part's bytes.
    The very first part ever received has nothing to compare against (a frozen camera's old frame looks the same), so it
    only primes the comparison: the camera goes live on the first part that is new relative to it. The constructor
    waits for that. Mid-run reconnects keep the previous part, so they need no re-priming.
    Every genuinely new frame (never a repeat) also goes into `ring` (camsync.FrameRing) as (X-Capture-Time, clamped to
    arrival wall time like the recorder / arrival wall time if camview sent none, RingFrame(JPEG, arrival monotonic,
    seq)): RigMachine picks from it with camsync.select. `capture_time` = the newest accepted frame's X-Capture-Time.
    The freshness basis (check_fresh / read) stays local monotonic arrival time. JPEGs are decoded only in frame_of(),
    i.e. only the frames an observation actually uses."""

    def __init__(self, url, timeout_s=3.0):
        self.src = url
        self.jpeg, self.stamp, self.cache = None, 0.0, (None, None)
        self.seq, self.capture_time = None, None      # of the newest accepted frame (None if camview sends no headers)
        self.ring = camsync.FrameRing()                # genuinely new frames only (bounded: RING_KEEP_S / RING_MAX)
        self.part_stamp, self.repeats = 0.0, 0        # arrival of ANY part / repeats since the last new frame
        self._prev = None                              # (seq, bytes) of the previous part, kept across reconnects
        self.lock, self.running = threading.Lock(), True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        t0 = time.monotonic()
        while self.jpeg is None:
            if time.monotonic() - t0 > timeout_s:
                self.running = False       # no join: the daemon reader may sit in readline() until the next part /
                if self._prev is None:     # socket timeout; joining would delay this raise by up to release()'s 1 s
                    raise RuntimeError(f"camera stream {url!r} produced no frames in {timeout_s}s (is camview running?)")
                raise RuntimeError(f"camera stream {url!r} not live after {timeout_s}s: got a frame but no second new "
                                   f"frame ({self.repeats} repeats of it) -> camera frozen at camview?")
            time.sleep(0.01)

    def _loop(self):
        while self.running:
            try:
                with urllib.request.urlopen(self.src, timeout=5.0) as r:
                    headers = {}
                    while self.running:
                        line = r.readline(65536)
                        if not line:
                            break
                        line = line.strip()
                        if not line:                     # blank line: end of a part's header block
                            n = headers.pop("content-length", None)
                            if n is not None and n.isdigit():
                                data = r.read(int(n))
                                if len(data) == int(n):
                                    self._on_part(headers, data)
                            headers = {}
                        elif line.startswith(b"--"):     # boundary: a new part starts
                            headers = {}
                        elif b":" in line:               # part header, any case / order
                            k, _, v = line.partition(b":")
                            headers[k.strip().lower().decode("latin-1")] = v.strip().decode("latin-1")
            except Exception:
                pass
            if self.running:
                time.sleep(0.5)                          # camview restarting / stream gone: reconnect

    def _on_part(self, headers, data):
        try:
            seq = int(headers["x-frame-seq"])
        except (KeyError, ValueError):
            seq = None
        try:
            capture_time = float(headers["x-capture-time"])
        except (KeyError, ValueError):
            capture_time = None
        prev = self._prev
        if prev is None:                                 # first part ever: prime the comparison, not live yet
            self._prev = (seq, data)
            with self.lock:
                self.part_stamp = time.monotonic()
            return
        if seq is not None:
            is_new = prev[0] != seq
        else:
            # Exact bytes compare, not a hash: we already hold the previous part, `==` short-circuits on length
            # (almost always different for real frames) and is a memcmp otherwise -- cheaper than hashing ~100 KB
            # (blake2b ~1 GB/s on a Pi 5 => ~0.1 ms/part) and has no collision question. A live sensor's JPEGs never
            # repeat byte-for-byte (noise), so an identical part is a re-send of a frozen camera's last frame.
            is_new = prev[1] != data
        self._prev = (seq, data)
        now = time.monotonic()
        with self.lock:
            self.part_stamp = now
            if is_new:
                self.jpeg, self.stamp, self.seq, self.capture_time, self.repeats = data, now, seq, capture_time, 0
            else:
                self.repeats += 1
        if is_new:     # repeats never enter the ring: a frozen camera's re-sent frame must not look like a new one
            wall = time.time()                           # same clamp as the recorder (dashboard.py CamReader)
            self.ring.push(wall if capture_time is None else min(capture_time, wall), RingFrame(data, now, seq))

    def check_fresh(self, max_age_s=0.5):
        """Raises if the stream stopped delivering NEW frames (repeats of an old one don't count). No decoding.
        Returns the newest accepted JPEG."""
        with self.lock:
            jpeg, stamp, part_stamp, repeats = self.jpeg, self.stamp, self.part_stamp, self.repeats
        now = time.monotonic()
        if jpeg is None:
            raise RuntimeError(f"camera {self.src!r} not live: waiting for a second new frame "
                               f"({'no data yet' if not part_stamp else f'{repeats} repeats of the first'})")
        if now - stamp > max_age_s:
            if repeats and now - part_stamp <= max(max_age_s, 3.0):   # camview re-sends every ~2 s when frozen
                why = (f"stream alive but only repeating the old frame ({repeats} repeats, last "
                       f"{now - part_stamp:.2f}s ago) -> camera frozen/disconnected at camview")
            else:
                why = f"no data from the stream for {now - part_stamp:.2f}s -> stream disconnected / camview down"
            raise RuntimeError(f"camera {self.src!r} stale: last new frame {now - stamp:.2f}s ago; {why}")
        return jpeg

    def read(self, max_age_s=0.5):
        """Newest BGR frame; raises if the stream stopped delivering NEW frames (repeats of an old one don't count)."""
        return self._decode(self.check_fresh(max_age_s))

    def frame_of(self, item):
        """A RingFrame from `ring` -> its BGR frame (decoded here; the last decode is cached)."""
        return self._decode(item.data)

    def _decode(self, jpeg):
        if self.cache[0] is not jpeg:
            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"camera {self.src!r}: undecodable JPEG")
            self.cache = (jpeg, frame)
        return self.cache[1]

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)


def open_camera(src, hw):
    """camview stream URL (.../stream/<id>) -> MjpegCamera; device index / other URL -> ThreadedCamera."""
    if isinstance(src, str) and src.startswith("http") and "/stream/" in src:
        return MjpegCamera(src)
    return ThreadedCamera(src, hw)


def parse_cameras(specs):
    """["front=0", "wrist=1", "top=http://..."] -> {"front": 0, ...}; empty -> {}."""
    if not specs:
        return {}
    cams = {}
    for spec in specs:
        name, _, src = spec.partition("=")
        cams[name] = int(src) if src.isdigit() else src
    return cams


def tile(obs, names, hw):
    """Side-by-side RGB view of the named cameras; a blank grey panel if there are none."""
    if not names:
        return np.full((*hw, 3), 60, np.uint8)
    return np.hstack([obs[n] for n in names])
