"""camview -- serve every camera on the Pi as MJPEG over HTTP, so you can watch them from a laptop browser.

Three kinds of source. The first two are auto-discovered and hot-plugged (unplug/replug or add a new one -> picked up
within a few seconds, no restart needed):
  - USB webcams (v4l2): grabbed via ffmpeg, one subprocess per camera.
  - Luxonis OAK camera (depthai): color, left, right, colorized stereo depth (the OAK_STREAMS subset; default color
    only) -- one pipeline per device, every stream rotated 180 deg (OAK_ROTATE_180: the OAK is mounted upside down).
  - Network cameras (stream id "net-<name>"), listed in remote_cams.json next to this file and/or with
    --remote NAME=URL (the file is re-read when it changes). HTTP MJPEG is passed through unchanged; RTSP and other
    URLs are decoded + JPEG-encoded by ffmpeg on the Pi (CPU cost); "mode": "poll" fetches snapshot JPEG URLs.
    Reconnects forever with backoff; a stream with no new frame for REMOTE_STALE_S says so in its `error`.
    See docs/remote-camera.md.

/stream/<id> sends a part only for a NEW frame (never re-sends an old one while a camera stalls); each part carries
X-Frame-Seq (strictly increasing per stream), X-Arrival-Time (epoch s when the frame arrived at camview) and
X-Capture-Time (epoch s on this Pi's clock when the frame was taken: the arrival time for Pi cameras; for a network
camera whose sender stamps X-Capture-Time and serves /api/time, like worldcam/worldcam.py, the sender's capture time
mapped onto the Pi clock by camsync.ClockSync, so the network delay is not counted as "fresh").

Run ON the Pi from the act-athon repo root:
    .venv/bin/python3 camview.py       # venv has depthai + opencv-python-headless -> all cameras
    python3 camview.py                 # system python: USB webcams only, Luxonis silently skipped
then open http://<pi>:8100 on your laptop.

Running camview (it owns the cameras: a second or leftover camview makes them "busy"):
  - From the laptop use `ssh -t excavator 'cd act-athon && .venv/bin/python3 camview.py'` (-t allocates a tty, so
    Ctrl-C reaches camview and closing the ssh session sends it SIGHUP). Without -t, Ctrl-C only kills the local ssh
    client and camview keeps running on the Pi as an orphan, still holding the webcams. Or run it from a shell on the Pi.
  - Ctrl-C (SIGINT), `kill <pid>` (SIGTERM) and a dropped session (SIGHUP) all shut down cleanly: HTTP server closed,
    ffmpeg children terminated (killed after 2 s), OAK devices closed; after SHUTDOWN_TIMEOUT_S camview exits hard.
    A second Ctrl-C / SIGTERM exits at once. Its ffmpeg children also die if camview itself is SIGKILLed (Linux).
  - Find / stop a leftover one: `pgrep -af camview.py`, then `kill <pid>`. A camera held by another process shows
    "busy: held by PID ..." in its `error` (camview retries it every few seconds, so killing the holder is enough).

Stdlib only for the HTTP/USB/network path; depthai + opencv + numpy (in .venv) only needed for the Luxonis path,
and their absence degrades gracefully instead of crashing the whole server.
"""
import argparse
import base64
import collections
import ctypes
import http.client
import json
import os
import re
import select
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import camsync

PORT = 8100
BOUNDARY = "camviewframe"
SCAN_PERIOD_S = 3.0

try:
    import depthai as dai
    import cv2
    import numpy as np
    HAVE_DEPTHAI = True
except ImportError:
    HAVE_DEPTHAI = False

USB_FPS = 15               # USB webcams (C270 MJPG 640x480 supports 30/24/20/15/10/7.5/5)
USB_RES = (640, 480)
OAK_FPS = 15               # every OAK output (color, left, right, stereo)
OAK_FALLBACK_FPS = 15      # OAK when frames are Pi-encoded (raw NV12 over USB): keep <= OAK_FPS
COLOR_RES = (640, 480)     # OAK color: ISP-scaled on the device; 4:3 = the sensor's native aspect (full field of view).
                           # Every camera is captured 640x480; the square crop happens in conversion + deployment.
MONO_RES = (640, 480)      # OAK left / right (also the stereo input)
DEPTH_RANGE_MM = (500, 4000)  # stereo depth reported outside this band is dropped (see StereoDepth's thresholdFilter)
JPEG_QUALITY = 80
POLL_S = 0.005             # OAK reader threads: sleep between empty tryGet()s
ENCODER_FIRST_FRAME_S = 10.0   # OAK encoder mode: no JPEG from the device this long after start -> Pi-encode fallback
OAK_ROTATE_180 = True      # the OAK is mounted upside down: rotate every OAK stream 180 deg (color on the sensor,
                           # left/right with an on-device ImageManip, depth and the Pi-encode fallback on the Pi).
                           # Episodes recorded with a different setting must not be mixed.
OAK_STREAMS = ("color",)   # OAK streams to run, any of "color", "left", "right", "depth". Only what is listed is built:
                           # without left/right/depth the mono sensors and StereoDepth are not started (less power).
OAK_ALL_STREAMS = ("color", "left", "right", "depth")
STREAM_FIRST_MAX_AGE_S = 1.0   # /stream/<id>: a new client gets the current frame only if it is younger than this
                               # (otherwise it waits for the next new one: a stalled camera sends nothing at all)

# Network cameras (RemoteFeed, docs/remote-camera.md)
REMOTE_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "remote_cams.json")
REMOTE_STALE_S = 1.0           # no new frame this long -> `error` says stale (the stream stays listed)
REMOTE_CONNECT_TIMEOUT_S = 3.0
REMOTE_READ_TIMEOUT_S = 2.0    # HTTP: no byte this long -> drop the connection and reconnect
REMOTE_BACKOFF_S = (0.5, 5.0)  # reconnect delay: doubles from the first to the second value; reset by a frame
REMOTE_FFMPEG_FIRST_FRAME_S = 10.0  # ffmpeg path: no JPEG this long after start (RTSP setup + first keyframe) ...
REMOTE_FFMPEG_NO_FRAME_S = 5.0      # ... or this long between JPEGs -> kill ffmpeg and reconnect
REMOTE_POLL_FPS = 10.0         # "poll" mode default request rate (the per-camera "fps" overrides it)
REMOTE_FPS_WINDOW_S = 2.0      # fps / mbit_s in /api/cameras are measured over this window
REMOTE_GAP_WINDOW_S = 60.0     # net.last_gap_s_max_60s / stale_60s / reconnects_60s window
REMOTE_MAX_JPEG = 4_000_000    # a part / buffer bigger than this is dropped as garbage
REMOTE_NAME_RE = re.compile(r"^[A-Za-z0-9-]+$")

# Shutdown (SIGINT / SIGTERM / SIGHUP) and child processes
SHUTDOWN_TIMEOUT_S = 5.0   # clean shutdown not done this long after the signal -> kill children, os._exit
CHILD_TERM_S = 2.0         # ffmpeg children: SIGTERM, then SIGKILL after this long
USB_BUSY_RETRY_S = 5.0     # USB camera busy (held by another process): retry this often
USB_BUSY_SCAN_S = 10.0     # ... and look up the holder in /proc at most this often per camera


# ---------------------------------------------------------------- shared frame registry
class Stream:
    """One video source. A feed thread pushes JPEG bytes in; any number of HTTP clients tail them."""

    def __init__(self, sid, label, group):
        self.id, self.label, self.group = sid, label, group
        self.cond = threading.Condition()
        self.jpeg = None
        self.seq = 0
        self.t_last = 0.0  # arrival of the newest frame (time.time())
        self.t_cap = 0.0   # its capture time on this Pi's clock (= t_last unless the feed knows better)
        self.error = None
        self.mode = None  # how frames are produced (USB: ffmpeg capture mode; OAK: oak_mode(), e.g. "oak mjpeg+rot")
        self.fps = 0.0    # measured push rate (EMA), shown in /api/cameras
        self.stats = None  # optional callable(now) -> dict merged into this stream's /api/cameras entry (RemoteFeed)

    def push(self, jpeg_bytes, t_cap=None):
        with self.cond:
            now = time.time()
            if self.t_last:
                dt = now - self.t_last
                if dt > 0:
                    self.fps = (1.0 / dt) if not self.fps else 0.9 * self.fps + 0.1 / dt
            self.jpeg = jpeg_bytes
            self.seq += 1
            self.t_last = now
            self.t_cap = now if t_cap is None else min(t_cap, now)
            self.error = None
            self.cond.notify_all()

    def wait_for_frame(self, after_seq, timeout=2.0):
        with self.cond:
            if self.seq == after_seq:
                self.cond.wait(timeout)
            return self.jpeg, self.seq

    def wait_for_new(self, after_seq, timeout=2.0):
        """-> (jpeg, seq, capture time, arrival time) of the newest frame; seq == after_seq means nothing new within
        timeout."""
        with self.cond:
            if self.seq == after_seq:
                self.cond.wait(timeout)
            return self.jpeg, self.seq, self.t_cap, self.t_last


REGISTRY = {}
REG_LOCK = threading.Lock()


def put_stream(sid, label, group):
    with REG_LOCK:
        s = REGISTRY.get(sid)
        if s is None:
            s = Stream(sid, label, group)
            REGISTRY[sid] = s
        return s


def drop_stream(sid):
    with REG_LOCK:
        REGISTRY.pop(sid, None)


def list_streams():
    with REG_LOCK:
        out = []
        now = time.time()
        for s in REGISTRY.values():
            age = (now - s.t_last) if s.t_last else None
            d = {"id": s.id, "label": s.label, "group": s.group, "error": s.error, "mode": s.mode,
                 "fps": round(s.fps, 1), "age_s": round(age, 2) if age is not None else None}
            if s.stats is not None:
                try:
                    d.update(s.stats(now))
                except Exception as ex:   # stats are diagnostics: never let them break the camera list
                    d["error"] = d["error"] or "stats failed: %s" % ex
            out.append(d)
        return sorted(out, key=lambda x: (x["group"], x["label"]))


FEEDS, FEEDS_LOCK = [], threading.Lock()
SHUTDOWN = threading.Event()   # set once by shutdown_all_feeds(): scanners stop, no new feed / ffmpeg starts


def register_feed(f):
    with FEEDS_LOCK:
        FEEDS.append(f)


def unregister_feed(f):
    with FEEDS_LOCK:
        if f in FEEDS:
            FEEDS.remove(f)


def log(msg):
    """print that never raises: after SIGHUP (ssh session gone) stdout may be a dead tty / broken pipe."""
    try:
        print(msg, flush=True)
    except Exception:
        pass


# ---------------------------------------------------------------- child processes (ffmpeg)
# Every ffmpeg camview starts goes through spawn_child(): it is tracked in CHILDREN (so shutdown can terminate / kill
# it even when the feed thread reading it is stuck) and, on Linux, gets PR_SET_PDEATHSIG = SIGKILL, so it dies with
# camview however camview dies (SIGKILL, OOM, crash): an ffmpeg on a stalled camera never writes, so it would never see
# a broken pipe and would keep /dev/videoN open forever.
# Caveat: the death signal fires when the THREAD that forked the child exits, not when the process does. Children are
# only spawned from long-lived feed threads (UsbFeed._run, RemoteFeed._run -> _session_ffmpeg), which also do every
# restart / reconnect / busy retry themselves and reap their child before returning, so a live ffmpeg's forking
# thread lives exactly as long as it is needed. Don't spawn from short-lived helper threads.
PR_SET_PDEATHSIG = 1
CHILDREN, CHILD_LOCK = set(), threading.Lock()


def _load_prctl():
    """libc prctl() via ctypes on Linux, else None (macOS dev machines: children just aren't death-signalled)."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        fn = ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError):
        return None
    fn.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    fn.restype = ctypes.c_int
    return fn


_PRCTL = _load_prctl()


def child_preexec():
    """-> preexec_fn for Popen that sets PR_SET_PDEATHSIG = SIGKILL in the child, or None without prctl. The child
    exits at once if camview already died between fork and prctl (it would otherwise miss the signal). Runs between
    fork and exec, so it only calls the pre-resolved prctl + getppid."""
    if _PRCTL is None:
        return None
    parent, sig = os.getpid(), int(signal.SIGKILL)
    prctl = _PRCTL

    def preexec():
        prctl(PR_SET_PDEATHSIG, sig, 0, 0, 0)
        if os.getppid() != parent:
            os._exit(1)
    return preexec


def spawn_child(args, **kw):
    """subprocess.Popen(args, **kw) with the death signal, tracked in CHILDREN. -> None when shutting down."""
    with CHILD_LOCK:
        if SHUTDOWN.is_set():
            return None
        proc = subprocess.Popen(args, preexec_fn=child_preexec(), **kw)
        CHILDREN.add(proc)
    return proc


def reap_child(proc, grace=CHILD_TERM_S):
    """SIGTERM, wait up to grace, SIGKILL; untrack. Safe on an already-exited process."""
    try:
        proc.terminate()
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            proc.wait(timeout=1.0)
        except Exception:
            pass
    except Exception:
        pass
    with CHILD_LOCK:
        CHILDREN.discard(proc)


def reap_all_children(grace=CHILD_TERM_S):
    """Shutdown: SIGTERM every tracked child at once, SIGKILL whatever is still alive after grace."""
    with CHILD_LOCK:
        procs = list(CHILDREN)
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    t_end = time.monotonic() + grace
    alive = procs
    while alive and time.monotonic() < t_end:
        time.sleep(0.05)
        alive = [p for p in alive if p.poll() is None]
    for p in alive:
        try:
            p.kill()
            p.wait(timeout=0.5)
        except Exception:
            pass
    with CHILD_LOCK:
        CHILDREN.difference_update(procs)
    return len(procs), len(alive)


def kill_children_now():
    """Hard-exit path (second signal, watchdog): SIGKILL every tracked child, no waiting and no locks (may run in a
    signal handler that interrupted a thread holding CHILD_LOCK)."""
    try:
        procs = list(CHILDREN)
    except RuntimeError:
        procs = []
    for p in procs:
        try:
            p.kill()
        except Exception:
            pass


def shutdown_all_feeds(timeout=SHUTDOWN_TIMEOUT_S - 1.0):
    """Stop every feed: no new feeds / ffmpegs from here on, stop flags + sockets shut, ffmpeg children terminated
    (SIGKILL after CHILD_TERM_S), then wait up to `timeout` s in total for the feed threads to finish (an OAK feed
    thread joins its reader threads, then closes the pipeline and the device). Returns the feeds still running."""
    t_end = time.monotonic() + timeout
    with FEEDS_LOCK:
        SHUTDOWN.set()   # under FEEDS_LOCK: a feed registered after this snapshot sees SHUTDOWN in its thread
        feeds = list(FEEDS)
    for f in feeds:
        try:
            f.stop()
        except Exception:
            pass
    n, killed = reap_all_children(min(CHILD_TERM_S, max(0.0, t_end - time.monotonic())))
    if n:
        log("camview: %d ffmpeg child(ren) stopped%s" % (n, " (%d needed SIGKILL)" % killed if killed else ""))
    me = threading.current_thread()
    for f in feeds:
        t = getattr(f, "thread", None)
        if t is not None and t is not me:
            t.join(max(0.0, t_end - time.monotonic()))
    left = [f for f in feeds if getattr(f, "thread", None) is not None and f.thread.is_alive()]
    if left:
        log("camview: %d feed(s) still stopping after %.1f s: %s"
            % (len(left), timeout, ", ".join(type(f).__name__ for f in left)))
    return left


# ---------------------------------------------------------------- busy device: who holds it
def device_holders(dev, proc_root="/proc"):
    """PIDs of the processes that have `dev` open (a /proc/<pid>/fd/* symlink to it), among those whose fd dir camview
    may read (its own user's; root's and other users' are invisible without sudo)."""
    targets = {dev, os.path.realpath(dev)}
    try:
        pids = sorted((p for p in os.listdir(proc_root) if p.isdigit()), key=int)
    except OSError:
        return []
    out = []
    for pid in pids:
        fd_dir = os.path.join(proc_root, pid, "fd")
        try:
            fds = os.listdir(fd_dir)
        except OSError:   # another user's process (EACCES) or it just exited
            continue
        for fd in fds:
            try:
                if os.readlink(os.path.join(fd_dir, fd)) in targets:
                    out.append(int(pid))
                    break
            except OSError:
                continue
    return out


def proc_cmd(pid, proc_root="/proc", maxlen=60):
    """Short command line of pid: argv[0] as its basename (".venv/bin/python3 camview.py" -> "python3 camview.py")."""
    try:
        with open(os.path.join(proc_root, str(pid), "cmdline"), "rb") as f:
            argv = f.read().decode("utf-8", "replace").rstrip("\0").split("\0")
        cmd = " ".join([os.path.basename(argv[0])] + argv[1:]).strip()
    except OSError:
        cmd = ""
    if not cmd:
        try:
            with open(os.path.join(proc_root, str(pid), "comm")) as f:
                cmd = "[%s]" % f.read().strip()
        except OSError:
            cmd = "?"
    return cmd if len(cmd) <= maxlen else cmd[:maxlen - 1] + "…"


def proc_ppid(pid, proc_root="/proc"):
    try:
        with open(os.path.join(proc_root, str(pid), "status")) as f:
            for line in f:
                if line.startswith("PPid:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def busy_message(dev, proc_root="/proc"):
    """`error` text for a USB camera node that is busy: who holds it, if camview can see it."""
    holders = device_holders(dev, proc_root)
    if not holders:
        return ("%s busy: held by another user's process (not visible to camview; on the Pi: sudo fuser -v %s)"
                % (dev, dev))
    parts = []
    for pid in holders[:3]:
        s = "PID %d (%s)" % (pid, proc_cmd(pid, proc_root))
        ppid = proc_ppid(pid, proc_root)
        if ppid == 1:
            s += " orphaned"
        elif ppid:
            s += " child of PID %d (%s)" % (ppid, proc_cmd(ppid, proc_root, 40))
        parts.append(s)
    return "%s busy: held by %s" % (dev, ", ".join(parts))


# ---------------------------------------------------------------- USB webcams (v4l2 + ffmpeg)
SYSFS_V4L = "/sys/class/video4linux"


def discover_usb_cameras():
    """Group /dev/videoN nodes by physical USB device (a camera's capture + metadata nodes share one).
    Excludes the Pi's own CSI/ISP/HEVC pipelines, which never sit behind a /usb sysfs path."""
    groups = {}
    if not os.path.isdir(SYSFS_V4L):
        return {}
    for entry in sorted(os.listdir(SYSFS_V4L)):
        if not entry.startswith("video"):
            continue
        sys_dir = os.path.join(SYSFS_V4L, entry)
        try:
            real = os.path.realpath(os.path.join(sys_dir, "device"))
        except OSError:
            continue
        if "/usb" not in real:
            continue
        try:
            with open(os.path.join(sys_dir, "name")) as f:
                name = f.read().strip()
        except OSError:
            name = entry
        groups.setdefault(real, {"name": name, "nodes": []})["nodes"].append("/dev/" + entry)
    out = {}
    for real, g in groups.items():
        cam_id = "usb-" + re.sub(r"[^A-Za-z0-9]+", "-", real.rsplit("/", 1)[-1]).strip("-")
        out[cam_id] = {"label": g["name"], "nodes": sorted(g["nodes"])}
    return out


def jpeg_end(buf, start):
    """Index just past the EOI of the JPEG whose SOI is at buf[start], or -1 if it isn't all in buf yet.
    Walks the marker segments up to SOS by their lengths, because an APPn segment (e.g. an EXIF thumbnail)
    can itself contain FFD9. From SOS on a plain search is exact: entropy-coded data byte-stuffs every
    0xFF (FF00) and RSTn markers are FFD0-FFD7, so FFD9 there can only be the real EOI."""
    p, n = start + 2, len(buf)
    while True:
        while p + 1 < n and buf[p] == 0xFF and buf[p + 1] == 0xFF:  # fill bytes before a marker
            p += 1
        if p + 1 >= n:
            return -1
        if buf[p] != 0xFF:  # not a marker where one must be: malformed, fall back to a plain search
            break
        m = buf[p + 1]
        if m == 0xD9:
            return p + 2
        if m == 0x01 or 0xD0 <= m <= 0xD7:  # standalone markers, no length field
            p += 2
            continue
        if p + 3 >= n:
            return -1
        seg_len = (buf[p + 2] << 8) | buf[p + 3]
        if seg_len < 2:  # malformed length, fall back
            break
        p += 2 + seg_len
        if m == 0xDA:  # SOS header skipped; entropy-coded data follows
            break
    end = buf.find(b"\xff\xd9", p)
    return end + 2 if end != -1 else -1


def split_jpegs(buf):
    """Concatenated JPEGs (e.g. ffmpeg `-f mjpeg` output) -> (complete JPEGs, remaining bytes). Bytes before an SOI
    are dropped; a buffer over 2 MB without a complete JPEG in it is dropped as runaway."""
    frames = []
    while True:
        start = buf.find(b"\xff\xd8")
        end = jpeg_end(buf, start) if start != -1 else -1
        if start == -1 or end == -1:
            if len(buf) > 2_000_000:  # runaway buffer, no valid JPEG boundary in it: drop
                buf = b""
            return frames, buf
        frames.append(buf[start:end])
        buf = buf[end:]


class UsbFeed:
    """ffmpeg against one physical USB camera; tries its video nodes in order (a metadata-only sibling
    node fails immediately, so this also picks the right one without needing a v4l2 capability query)."""

    def __init__(self, cam_id, label, nodes):
        self.cam_id, self.label, self.nodes = cam_id, label, nodes
        self.stream = put_stream(cam_id, label, "USB webcam")
        self.stop_flag = False
        self.proc = None
        register_feed(self)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    # Per node: ask the camera for MJPG and pass its JPEGs straight through (no encode on the Pi); only if
    # that yields no frame (camera without MJPG) fall back to raw YUYV + ffmpeg's own JPEG encode.
    MODES = (("mjpeg passthrough", ["-input_format", "mjpeg"], ["-c:v", "copy"]),
             ("yuyv422 + encode", ["-input_format", "yuyv422"], ["-q:v", "6"]))

    _busy_t = None     # busy holder lookup: monotonic time of the last /proc scan, and its message
    _busy_msg = None

    def _stopping(self):
        return self.stop_flag or SHUTDOWN.is_set()

    def _run(self):
        # All attempts, restarts and busy retries run in this one long-lived thread (PR_SET_PDEATHSIG is per thread).
        while True:
            busy_dev = None
            for dev in self.nodes:
                for mode, in_args, out_args in self.MODES:
                    if self._stopping():
                        return
                    args = (["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "v4l2"] + in_args +
                            ["-video_size", "%dx%d" % USB_RES, "-framerate", str(USB_FPS), "-i", dev] + out_args +
                            ["-f", "mjpeg", "-"])
                    try:
                        proc = spawn_child(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
                    except FileNotFoundError:
                        self.stream.error = "ffmpeg not installed"
                        return
                    if proc is None:   # shutting down
                        return
                    self.proc = proc
                    errs, drainer = self._drain(proc)
                    self.stream.mode = mode
                    buf, got_frame = b"", False
                    while not self.stop_flag:
                        chunk = proc.stdout.read(4096)
                        if not chunk:
                            break
                        frames, buf = split_jpegs(buf + chunk)
                        for jpeg in frames:
                            self.stream.push(jpeg)
                            got_frame = True
                    reap_child(proc)
                    self.proc = None
                    if got_frame or self._stopping():
                        return
                    if drainer is not None:
                        drainer.join(0.5)
                    if busy_dev is None and any("resource busy" in e.lower() for e in errs):
                        busy_dev = dev
            self.stream.mode = None
            if busy_dev is None:
                self.stream.error = "no capture node on this device produced frames"
                return
            # Busy (another process has the camera, e.g. a leftover camview's ffmpeg): say who, retry until it's free.
            now = time.monotonic()
            if self._busy_t is None or now - self._busy_t >= USB_BUSY_SCAN_S:
                self._busy_t = now
                try:
                    self._busy_msg = busy_message(busy_dev)
                except Exception as ex:   # diagnostics only
                    self._busy_msg = "%s busy (holder lookup failed: %s)" % (busy_dev, ex)
                log("%s: %s" % (self.cam_id, self._busy_msg))
            self.stream.error = "%s; retrying every %g s" % (self._busy_msg, USB_BUSY_RETRY_S)
            t_retry = time.monotonic() + USB_BUSY_RETRY_S
            while time.monotonic() < t_retry:
                if self._stopping():
                    return
                time.sleep(0.2)

    @staticmethod
    def _drain(proc):
        """Keep ffmpeg's stderr moving (a full pipe would block it); -> (last lines, drain thread or None)."""
        errs = collections.deque(maxlen=5)
        err = getattr(proc, "stderr", None)
        if err is None:
            return errs, None

        def drain():
            try:
                for line in iter(err.readline, b""):
                    errs.append(line.decode("utf-8", "replace").strip())
            except (OSError, ValueError):
                pass
        t = threading.Thread(target=drain, daemon=True)
        t.start()
        return errs, t

    def stop(self):
        self.stop_flag = True
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        drop_stream(self.cam_id)
        unregister_feed(self)


def usb_scanner():
    active = {}
    while not SHUTDOWN.is_set():
        found = discover_usb_cameras()
        for cam_id, info in found.items():
            if cam_id not in active:
                active[cam_id] = UsbFeed(cam_id, info["label"], info["nodes"])
        for cam_id in list(active):
            if cam_id not in found:
                active.pop(cam_id).stop()
        SHUTDOWN.wait(SCAN_PERIOD_S)


# ---------------------------------------------------------------- Luxonis OAK (depthai)
# Device ids whose on-device MJPEG encoder failed (build raised, pipeline died before the first JPEG, or no JPEG within
# ENCODER_FIRST_FRAME_S): their next pipeline (the scanner restarts it within SCAN_PERIOD_S) encodes on the Pi instead.
# Kept until camview restarts.
OAK_NO_ENCODER = set()


class EncoderFailed(Exception):
    pass


def oak_streams():
    """OAK_STREAMS in canonical order; unknown names are dropped (luxonis_scanner warns about them)."""
    return tuple(n for n in OAK_ALL_STREAMS if n in OAK_STREAMS)


def oak_mode(on_device):
    """/api/cameras `mode` of an OAK stream: where it is encoded, plus "+rot" when OAK_ROTATE_180 (rotated on the
    device in "oak mjpeg+rot", on the Pi in "pi encode+rot")."""
    return ("oak mjpeg" if on_device else "pi encode") + ("+rot" if OAK_ROTATE_180 else "")


class LuxonisFeed:
    """One OAK device: the OAK_STREAMS subset of color, left, right and colorized stereo depth, each its own Stream.
    Encoder mode ("oak mjpeg"): color / left / right are JPEG-encoded ON the OAK (VideoEncoder, MJPEG) and the Pi pushes
    the bytes unchanged; only JPEGs cross the USB link. Fallback ("pi encode"): raw NV12 frames, cv2.imencode on the Pi,
    at OAK_FALLBACK_FPS. Depth is always colormapped + encoded on the Pi. One reader thread per output (tryGet + short
    sleep), so every stream runs at its own device rate.
    OAK_ROTATE_180: encoder mode rotates on the device (color: the Camera's sensor orientation; left/right: an ImageManip
    between the camera and its encoder, while StereoDepth keeps the unrotated mono frames its calibration expects);
    the fallback and depth rotate on the Pi."""

    def __init__(self, mxid, device_info, on_done):
        self.mxid, self.on_done = mxid, on_done
        self.stop_flag = False
        group = "Luxonis OAK %s" % mxid[-6:]
        self.streams = {name: put_stream("lux-%s-%s" % (mxid, name), name, group) for name in oak_streams()}
        register_feed(self)
        self.thread = threading.Thread(target=self._run, args=(device_info,), daemon=True)
        self.thread.start()

    def _fail(self, msg):
        for s in self.streams.values():
            s.error = msg

    def _build(self, pipeline, use_encoder):
        """-> {stream name: (queue, handler)}. Raises EncoderFailed if the encoder nodes can't be built."""
        fps = OAK_FPS if use_encoder else OAK_FALLBACK_FPS
        want = set(self.streams)
        rotate_on_device = OAK_ROTATE_180 and use_encoder
        outs = {}   # stream name -> unrotated camera output
        if "color" in want:
            color = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            if rotate_on_device:   # on the sensor: every output of this camera comes out rotated, at no extra cost
                try:
                    color.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
                except Exception as ex:
                    raise EncoderFailed("color setImageOrientation: %s: %s" % (type(ex).__name__, ex))
            outs["color"] = color.requestOutput(COLOR_RES, dai.ImgFrame.Type.NV12, fps=fps)

        # Mono sensors only run when something needs them: their own stream, or depth (which needs both).
        mono = {}
        for name, socket in (("left", dai.CameraBoardSocket.CAM_B), ("right", dai.CameraBoardSocket.CAM_C)):
            if name in want or "depth" in want:
                cam = pipeline.create(dai.node.Camera).build(socket)
                mono[name] = cam.requestOutput(MONO_RES, dai.ImgFrame.Type.NV12, fps=fps)
                if name in want:
                    outs[name] = mono[name]

        max_disp = None
        stereo = None
        if "depth" in want:
            # StereoDepth gets the UNROTATED mono frames: rectification uses the calibration in sensor orientation.
            stereo = pipeline.create(dai.node.StereoDepth).build(left=mono["left"], right=mono["right"])
            stereo.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)
            stereo.setRectification(True)
            stereo.setExtendedDisparity(True)
            stereo.setLeftRightCheck(True)
            stereo.initialConfig.postProcessing.thresholdFilter.minRange = DEPTH_RANGE_MM[0]
            stereo.initialConfig.postProcessing.thresholdFilter.maxRange = DEPTH_RANGE_MM[1]
            max_disp = stereo.initialConfig.getMaxDisparity()

        readers = {}
        if use_encoder:
            for name, out in outs.items():
                src = out
                if rotate_on_device and name != "color":   # mono: raw -> ImageManip(180) -> encoder
                    try:
                        src = self._rotated(pipeline, out)
                    except Exception as ex:
                        raise EncoderFailed("%s ImageManip rotation: %s: %s" % (name, type(ex).__name__, ex))
                try:
                    enc = pipeline.create(dai.node.VideoEncoder).build(
                        src, frameRate=fps, profile=dai.VideoEncoderProperties.Profile.MJPEG)
                except Exception as ex:
                    raise EncoderFailed("VideoEncoder build: %s: %s" % (type(ex).__name__, ex))
                try:
                    enc.setQuality(JPEG_QUALITY)
                except Exception as ex:   # quality is cosmetic: keep the device default
                    print("OAK %s: %s encoder setQuality failed (%s), using the default" % (self.mxid, name, ex),
                          flush=True)
                try:
                    readers[name] = (enc.out.createOutputQueue(maxSize=1, blocking=False), self._push_encoded)
                except Exception as ex:
                    raise EncoderFailed("VideoEncoder queue: %s: %s" % (type(ex).__name__, ex))
        else:
            for name, out in outs.items():
                readers[name] = (out.createOutputQueue(maxSize=1, blocking=False), self._push_cv)
        if stereo is not None:
            q_depth = stereo.disparity.createOutputQueue(maxSize=1, blocking=False)
            readers["depth"] = (q_depth, lambda name, msg: self._push_depth(msg, max_disp))
        return readers

    @staticmethod
    def _rotated(pipeline, out):
        """out -> on-device ImageManip rotating 180 deg -> NV12 output of the same size (the VideoEncoder's input).
        The default max output frame size (1 MiB) covers 640x480 NV12 (460800 bytes)."""
        manip = pipeline.create(dai.node.ImageManip)
        manip.initialConfig.addRotateDeg(180)
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)
        out.link(manip.inputImage)
        return manip.out

    def _ir_off(self, device):
        """Power: make sure the IR dot projector and flood light (OAK-D Pro models only) are off. They are off by
        default; this only guards against a previous process having left them on. Failures just print."""
        try:
            if not device.getIrDrivers():
                return
            ok_dots = device.setIrLaserDotProjectorIntensity(0.0)
            ok_flood = device.setIrFloodLightIntensity(0.0)
            print("OAK %s: IR emitters off (dot projector %s, flood light %s)"
                  % (self.mxid, "ok" if ok_dots else "failed", "ok" if ok_flood else "failed"), flush=True)
        except Exception as ex:
            print("OAK %s: could not turn the IR emitters off (%s: %s)" % (self.mxid, type(ex).__name__, ex),
                  flush=True)

    def _run(self, device_info):
        use_encoder = self.mxid not in OAK_NO_ENCODER
        try:
            if SHUTDOWN.is_set() or self.stop_flag:   # created while camview was shutting down: don't open the device
                return
            with dai.Device(device_info) as device:
                self._ir_off(device)
                with dai.Pipeline(device) as pipeline:
                    readers = self._build(pipeline, use_encoder)
                    for name, s in self.streams.items():
                        s.mode = oak_mode(use_encoder and name != "depth")
                    pipeline.start()
                    # the first-JPEG watchdog only applies when something is encoded on the device (not depth-only)
                    self._pump(pipeline, readers, use_encoder and any(n != "depth" for n in readers))
        except EncoderFailed as ex:
            OAK_NO_ENCODER.add(self.mxid)
            print("OAK %s: on-device MJPEG/rotation failed (%s): restarting with Pi-side encoding%s"
                  % (self.mxid, ex, " and rotation" if OAK_ROTATE_180 else ""), flush=True)
            self._fail("encoder failed, restarting with Pi encoding: %s" % ex)
        except Exception as ex:
            self._fail(str(ex))
        finally:
            for s in self.streams.values():
                drop_stream(s.id)
            unregister_feed(self)
            self.on_done(self.mxid)

    def _pump(self, pipeline, readers, use_encoder):
        """Start one reader thread per output; return when stopped, the pipeline ends, or a reader fails (the
        error is re-raised here, after every reader has stopped, so the device is closed only once they're done)."""
        halt = threading.Event()
        errors = []
        got_jpeg = threading.Event()

        def reader(name, q, handler):
            try:
                while not halt.is_set() and not self.stop_flag:
                    msg = q.tryGet()
                    if msg is None:
                        time.sleep(POLL_S)
                        continue
                    handler(name, msg)
                    if name != "depth":
                        got_jpeg.set()
            except Exception as ex:
                errors.append(ex)
                halt.set()

        threads = [threading.Thread(target=reader, args=(n, q, h), daemon=True, name="oak-%s-%s" % (self.mxid[-6:], n))
                   for n, (q, h) in readers.items()]
        for t in threads:
            t.start()
        t0 = time.monotonic()
        try:
            while not halt.is_set() and not self.stop_flag:
                if not pipeline.isRunning():
                    errors.append(RuntimeError("pipeline stopped"))
                    break
                if use_encoder and not got_jpeg.is_set() and time.monotonic() - t0 > ENCODER_FIRST_FRAME_S:
                    errors.append(EncoderFailed("no JPEG from the device within %.0f s" % ENCODER_FIRST_FRAME_S))
                    break
                halt.wait(0.2)
        finally:
            halt.set()
            for t in threads:
                t.join(timeout=2.0)
        if errors:
            ex = errors[0]
            if use_encoder and not got_jpeg.is_set() and not isinstance(ex, EncoderFailed) and not self.stop_flag:
                # died before the first encoded frame: most likely the encoder (resource / format), not an unplug
                raise EncoderFailed("pipeline failed before the first JPEG: %s: %s" % (type(ex).__name__, ex))
            raise ex

    def _push_encoded(self, name, msg):
        data = bytes(msg.getData())   # EncodedFrame (MJPEG): one complete JPEG per message
        if data[:2] != b"\xff\xd8":
            raise EncoderFailed("%s: encoder output is not a JPEG (%d bytes, starts %r)" % (name, len(data), data[:4]))
        self.streams[name].push(data)

    def _push_cv(self, name, msg):
        frame = msg.getCvFrame()
        if OAK_ROTATE_180:   # fallback frames come unrotated from the device
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        ok, jpeg = cv2.imencode(".jpg", frame, (cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY))
        if ok:
            self.streams[name].push(jpeg.tobytes())

    def _push_depth(self, msg, max_disp):
        norm = (msg.getFrame().astype(np.float32) * (255.0 / max_disp)).astype(np.uint8)
        if OAK_ROTATE_180:   # disparity is computed on the unrotated mono frames
            norm = cv2.rotate(norm, cv2.ROTATE_180)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        ok, jpeg = cv2.imencode(".jpg", colored, (cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY))
        if ok:
            self.streams["depth"].push(jpeg.tobytes())

    def stop(self):
        self.stop_flag = True


def luxonis_scanner():
    if not HAVE_DEPTHAI:
        print("depthai/opencv not importable: Luxonis cameras skipped (USB webcams still work). "
              "Run with .venv/bin/python3 to get them.", flush=True)
        return
    unknown = [n for n in OAK_STREAMS if n not in OAK_ALL_STREAMS]
    if unknown:
        print("OAK_STREAMS: ignoring unknown stream name(s) %s (allowed: %s)" % (unknown, ", ".join(OAK_ALL_STREAMS)),
              flush=True)
    if not oak_streams():
        print("OAK_STREAMS is empty: Luxonis cameras skipped.", flush=True)
        return
    active, lock = {}, threading.Lock()

    def remove(mxid):
        with lock:
            active.pop(mxid, None)

    while not SHUTDOWN.is_set():
        try:
            found = dai.Device.getAllAvailableDevices()
        except Exception:
            found = []
        with lock:
            for info in found:
                mxid = info.getDeviceId()
                if mxid not in active:
                    active[mxid] = LuxonisFeed(mxid, info, remove)
        SHUTDOWN.wait(SCAN_PERIOD_S)


# ---------------------------------------------------------------- network cameras (HTTP MJPEG / RTSP / snapshots)
REMOTE_MODES = ("auto", "mjpeg", "ffmpeg", "poll")
REMOTE_KEYS = ("url", "label", "mode", "rotate", "width", "height", "fps")
SOI, EOI = b"\xff\xd8", b"\xff\xd9"


def redact_url(url):
    """url with any user:password replaced by ***, for logs and /api/cameras."""
    try:
        u = urllib.parse.urlsplit(url)
        if u.username is None and u.password is None:
            return url
        host = u.hostname or ""
        if ":" in host:
            host = "[%s]" % host
        netloc = "***@" + host + (":%d" % u.port if u.port else "")
        return urllib.parse.urlunsplit((u.scheme, netloc, u.path, u.query, u.fragment))
    except ValueError:
        return "<unparseable url>"


def parse_remote_spec(name, raw):
    """One remote_cams.json entry (a URL string or an object with REMOTE_KEYS) -> normalized spec dict, with "mode"
    resolved to "mjpeg", "ffmpeg" or "poll". Raises ValueError with a readable reason."""
    if not isinstance(name, str) or not REMOTE_NAME_RE.match(name):
        raise ValueError("camera name %r: use letters, digits and '-' only" % (name,))
    spec = {"url": raw} if isinstance(raw, str) else raw
    if not isinstance(spec, dict):
        raise ValueError("%s: expected a URL string or an object" % name)
    unknown = sorted(set(spec) - set(REMOTE_KEYS))
    if unknown:
        raise ValueError("%s: unknown key(s) %s (allowed: %s)" % (name, unknown, ", ".join(REMOTE_KEYS)))
    url = spec.get("url")
    if not isinstance(url, str) or "://" not in url:
        raise ValueError("%s: \"url\" must be a URL like http://192.168.1.50:8080/video" % name)
    u = urllib.parse.urlsplit(url)
    scheme = u.scheme.lower()
    try:
        u.port
    except ValueError:
        raise ValueError("%s: bad port in %s" % (name, redact_url(url)))
    if not u.hostname:
        raise ValueError("%s: no host in %s" % (name, redact_url(url)))
    mode = spec.get("mode", "auto")
    if mode not in REMOTE_MODES:
        raise ValueError("%s: \"mode\" must be one of %s" % (name, ", ".join(REMOTE_MODES)))
    rotate = spec.get("rotate", 0)
    if isinstance(rotate, bool) or rotate not in (0, 90, 180, 270):
        raise ValueError("%s: \"rotate\" must be 0, 90, 180 or 270" % name)
    nums = {}
    for k in ("width", "height", "fps"):
        v = spec.get(k)
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0
                              or (k != "fps" and int(v) != v)):
            raise ValueError("%s: \"%s\" must be a positive %s" % (name, k, "number" if k == "fps" else "integer"))
        nums[k] = (int(v) if k != "fps" else float(v)) if v is not None else None
    is_http = scheme in ("http", "https")
    if mode == "auto":
        if is_http and u.path.lower().endswith((".jpg", ".jpeg")):
            mode = "poll"
        else:
            mode = "mjpeg" if is_http else "ffmpeg"
    if mode in ("mjpeg", "poll") and not is_http:
        raise ValueError("%s: mode %s needs an http:// or https:// URL" % (name, mode))
    if mode == "poll" and rotate:
        raise ValueError("%s: \"rotate\" is not supported with mode poll" % name)
    if mode == "mjpeg" and rotate:
        mode = "ffmpeg"   # rotating needs decode + re-encode: only the ffmpeg path does that (Pi CPU)
    label = spec.get("label") or name
    if not isinstance(label, str):
        raise ValueError("%s: \"label\" must be a string" % name)
    return {"name": name, "url": url, "label": label, "mode": mode, "rotate": rotate,
            "width": nums["width"], "height": nums["height"], "fps": nums["fps"]}


def ffmpeg_remote_args(spec):
    """ffmpeg command: decode the network stream, optionally rotate (then scale to width x height, then cap to fps),
    JPEG-encode at -q:v 6 and write concatenated JPEGs to stdout (split like the USB feeds)."""
    url = spec["url"]
    timeout_us = str(int(REMOTE_CONNECT_TIMEOUT_S * 1e6))
    args = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]
    if urllib.parse.urlsplit(url).scheme.lower() in ("rtsp", "rtsps"):
        args += ["-rtsp_transport", "tcp", "-timeout", timeout_us]   # RTSP over TCP: no UDP loss / NAT trouble
    else:
        args += ["-rw_timeout", timeout_us]
    # no "-fflags nobuffer": with it ffmpeg can open the filter graph before the stream's pixel format is known and the
    # MJPEG encoder then fails to open ("Invalid argument"; seen with H.264 over HTTP)
    args += ["-flags", "low_delay", "-i", url, "-an", "-sn", "-dn"]
    vf = []
    if spec.get("rotate") == 90:
        vf.append("transpose=clock")
    elif spec.get("rotate") == 180:
        vf.append("hflip,vflip")
    elif spec.get("rotate") == 270:
        vf.append("transpose=cclock")
    if spec.get("width") or spec.get("height"):
        vf.append("scale=%d:%d" % (spec.get("width") or -2, spec.get("height") or -2))
    if spec.get("fps"):
        vf.append("fps=%g" % spec["fps"])
    if vf:
        args += ["-vf", ",".join(vf)]
    return args + ["-q:v", "6", "-f", "mjpeg", "-"]


class MjpegPartParser:
    """Body of an HTTP multipart/x-mixed-replace response -> complete JPEGs, bytes passed through unchanged.
    Per part: the Content-Length body if the part has one and it is a whole JPEG (SOI..EOI); otherwise the JPEG found
    by its markers (jpeg_end) from the part's SOI. Garbage between parts is skipped (resync on the next boundary line);
    a part cut short by the next boundary is dropped. No boundary (or a declared one that never appears within
    REMOTE_MAX_JPEG bytes) -> plain split_jpegs on the byte stream. feed() -> [(jpeg, the part's X-Capture-Time as a
    float, or None)]: the sender's clock, not this Pi's."""

    def __init__(self, boundary):
        tok = (boundary or "").strip().strip('"').lstrip("-")   # servers disagree on leading dashes: match any
        self.delim = ("--" + tok).encode("latin-1") if tok else None
        self.buf = b""
        self.bad = 0   # parts / bytes skipped as unusable

    def feed(self, data):
        self.buf += data
        out = []
        while True:
            if self.delim is None:
                frames, self.buf = split_jpegs(self.buf)
                return out + [(f, None) for f in frames]
            got = self._part()
            if got is None:
                return out
            out.extend(got)

    def _skip(self, pos):
        self.bad += 1
        self.buf = self.buf[pos:]
        return []

    def _part(self):
        """Consume one part from self.buf -> [(jpeg, capture time)] or [] (consumed, nothing usable), or None (need more
        bytes)."""
        buf, d = self.buf, self.delim
        i = buf.find(d)
        if i == -1:
            if len(buf) > REMOTE_MAX_JPEG:
                print("MJPEG boundary %r never seen: splitting on JPEG markers instead" % d, flush=True)
                self.delim = None
                return []
            if len(buf) > 2 * len(d):   # keep only a tail that could hold the start of a boundary
                self.buf = buf[len(buf) - len(d):]
            return None
        line_end = buf.find(b"\n", i + len(d))
        if line_end == -1:
            return None if len(buf) - i < 1024 else self._skip(i + len(d))
        hs = line_end + 1
        if buf[hs:hs + 2] == b"\r\n":
            hdr, body = b"", hs + 2
        elif buf[hs:hs + 1] == b"\n":
            hdr, body = b"", hs + 1
        else:
            ends = [(p + n) for p, n in ((buf.find(b"\r\n\r\n", hs), 4), (buf.find(b"\n\n", hs), 2)) if p != -1]
            soi = buf.find(SOI, hs, hs + 8192)
            if soi != -1 and (not ends or soi < min(ends)):   # JPEG right after the boundary line, no blank line
                hdr, body = buf[hs:soi], soi
            elif ends:
                body = min(ends)
                hdr = buf[hs:body]
            else:
                return None if len(buf) - hs < 8192 else self._skip(hs)
        clen = t_cap = None
        for line in hdr.decode("latin-1").splitlines():
            k, _, v = line.partition(":")
            k = k.strip().lower()
            if k == "content-length":
                try:
                    clen = int(v.strip())
                except ValueError:
                    pass
            elif k == "x-capture-time":
                try:
                    t_cap = float(v.strip())
                except ValueError:
                    pass
        if clen is not None and 0 < clen <= REMOTE_MAX_JPEG:
            if len(buf) < body + clen:
                return None
            part = buf[body:body + clen]
            if part[:2] == SOI and part[-2:] == EOI:
                self.buf = buf[body + clen:]
                return [(part, t_cap)]
            # Content-Length wrong or padded: fall back to the JPEG markers below
        soi = buf.find(SOI, body)
        nxt = buf.find(d, body)
        if soi == -1 or (nxt != -1 and nxt < soi):   # a part without a JPEG
            if nxt != -1:
                return self._skip(nxt)
            return None if len(buf) - body <= REMOTE_MAX_JPEG else self._skip(len(buf))
        end = jpeg_end(buf, soi)
        if end == -1:
            if nxt != -1:                              # cut short by the next part
                return self._skip(nxt)
            return None if len(buf) - soi <= REMOTE_MAX_JPEG else self._skip(soi + 2)
        if nxt != -1 and nxt < end:                    # truncated JPEG; its "end" is in a later part
            return self._skip(nxt)
        self.buf = buf[end:]
        return [(buf[soi:end], t_cap)]


def describe_exc(ex):
    if isinstance(ex, socket.timeout):
        return "timed out (no data for %.0f s)" % REMOTE_READ_TIMEOUT_S
    if isinstance(ex, ConnectionRefusedError):
        return "connection refused"
    return "%s: %s" % (type(ex).__name__, ex)


class RemoteFeed:
    """One network camera -> Stream "net-<name>" (group "Network camera"). A thread connects, reads, and on any failure
    (timeout, refused, closed, HTTP error, ffmpeg exit) reconnects after a backoff of REMOTE_BACKOFF_S, forever; the
    stream stays registered meanwhile, with the reason in `error`. Only genuinely new frames are pushed: a JPEG that is
    byte-identical to the previous one (a sender repeating its last frame, an unchanged snapshot) is dropped, so seq,
    age_s and X-Capture-Time follow real frames. /api/cameras gets windowed fps, a stale `error` after REMOTE_STALE_S
    without a new frame, and a `net` block (see stats()).
    Sender clock: once a part arrives with X-Capture-Time (worldcam.py stamps every frame), a camsync.ClockSync on the
    sender's /api/time maps those stamps onto this Pi's clock and they become the stream's X-Capture-Time; until the
    first clock estimate (~0.5 s), or for a sender without /api/time, the arrival time is used. `net.clock` and
    `net.latency_ms` (sender capture -> arrival here) show how that goes."""

    GROUP = "Network camera"
    MODE_NAMES = {"mjpeg": "net mjpeg passthrough", "poll": "net snapshot poll", "ffmpeg": "net ffmpeg decode+encode"}

    def __init__(self, spec):
        self.spec = spec
        self.name, self.sid = spec["name"], "net-" + spec["name"]
        self.stop_ev = threading.Event()
        self.lock = threading.Lock()   # stats deques / counters (never held while pushing)
        self.sock = self.proc = None
        self.connected = False
        self.reconnects = 0
        self.t_reconnects = collections.deque()   # reconnect times within REMOTE_GAP_WINDOW_S
        self.arrivals = collections.deque()       # (t, nbytes) within REMOTE_FPS_WINDOW_S
        self.gaps = collections.deque()           # (t, gap between consecutive new frames) within REMOTE_GAP_WINDOW_S
        self.t_prev = None
        self.last = None
        self.frames = self.dups = self.bad = 0
        self.last_error = None
        self._session_frames = 0
        self.clock = None                         # camsync.ClockSync, made on the first part with X-Capture-Time
        self.lat = collections.deque()            # (t, sender capture -> arrival here) within REMOTE_FPS_WINDOW_S * 5
        self.stream = put_stream(self.sid, spec["label"], self.GROUP)
        self.stream.mode = self.MODE_NAMES[spec["mode"]] + ("+rot%d" % spec["rotate"] if spec["rotate"] else "")
        self.stream.error = "connecting"
        self.stream.stats = self.stats
        register_feed(self)
        self.thread = threading.Thread(target=self._run, daemon=True, name="net-" + self.name)
        self.thread.start()

    # -- frames + stats
    def _prune(self, now):
        while self.arrivals and now - self.arrivals[0][0] > REMOTE_FPS_WINDOW_S:
            self.arrivals.popleft()
        while self.gaps and now - self.gaps[0][0] > REMOTE_GAP_WINDOW_S:
            self.gaps.popleft()
        while self.t_reconnects and now - self.t_reconnects[0] > REMOTE_GAP_WINDOW_S:
            self.t_reconnects.popleft()
        while self.lat and now - self.lat[0][0] > 5 * REMOTE_FPS_WINDOW_S:
            self.lat.popleft()

    def _offer(self, jpeg, t_remote=None):
        """Push jpeg if it is a new JPEG (t_remote: the sender's X-Capture-Time, its clock). -> True if pushed."""
        if jpeg[:2] != SOI:
            self.bad += 1
            return False
        if jpeg == self.last:
            self.dups += 1
            return False
        if self.stop_ev.is_set():
            return False
        now = time.time()
        t_cap = None
        if t_remote is not None:
            if self.clock is None:
                self.clock = camsync.ClockSync(camsync.base_url(self.spec["url"]), self.sid)
            t_cap = self.clock.to_local(t_remote)
        with self.lock:
            if t_cap is not None:
                self.lat.append((now, now - t_cap))
            if self.t_prev is not None:
                self.gaps.append((now, now - self.t_prev))
            self.t_prev = now
            self.arrivals.append((now, len(jpeg)))
            self.frames += 1
            self._prune(now)
        self.last = jpeg
        self.connected = True
        self.stream.push(jpeg, t_cap)
        if self._session_frames == 0:
            print("net-%s: receiving frames" % self.name, flush=True)
        self._session_frames += 1
        return True

    def stats(self, now):
        """/api/cameras fields for this stream: fps over REMOTE_FPS_WINDOW_S (decays to 0 when frames stop), `error`
        (connection problem, or stale after REMOTE_STALE_S), and `net`."""
        with self.lock:
            self._prune(now)
            arr = list(self.arrivals)
            gap_max = max((g for _, g in self.gaps), default=None)
            stale_n = sum(1 for _, g in self.gaps if g > REMOTE_STALE_S)
            lat = sorted(x for _, x in self.lat)
            rec_60 = len(self.t_reconnects)
            t_prev = self.t_prev
        open_gap = (now - t_prev) if t_prev is not None else None
        if open_gap is not None:
            gap_max = max(gap_max or 0.0, open_gap)
        fps = 0.0
        if len(arr) >= 2:
            span = (arr[-1][0] - arr[0][0]) if now - arr[-1][0] <= REMOTE_STALE_S else (now - arr[0][0])
            fps = (len(arr) - 1) / span if span > 0 else 0.0
        mbit = sum(n for _, n in arr) * 8 / REMOTE_FPS_WINDOW_S / 1e6
        stale = open_gap is None or open_gap > REMOTE_STALE_S
        error = self.stream.error
        if error is None and stale and open_gap is not None:
            error = "stale: no new frame for %.1f s" % open_gap
        return {"fps": round(fps, 1), "error": error,
                "net": {"url": redact_url(self.spec["url"]), "connected": self.connected, "stale": stale,
                        "reconnects": self.reconnects, "reconnects_60s": rec_60,
                        "last_gap_s_max_60s": round(gap_max, 2) if gap_max is not None else None,
                        "stale_60s": stale_n, "mbit_s": round(mbit, 2), "frames": self.frames,
                        "dup_dropped": self.dups, "bad_parts": self.bad, "last_error": self.last_error,
                        "clock": self.clock.info() if self.clock is not None else None,
                        "latency_ms": {"p50": round(lat[len(lat) // 2] * 1e3, 1),
                                       "p95": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))] * 1e3, 1),
                                       "n": len(lat)} if lat else None}}

    # -- connection loop
    def _run(self):
        session = {"mjpeg": self._session_mjpeg, "poll": self._session_poll, "ffmpeg": self._session_ffmpeg}
        session = session[self.spec["mode"]]
        backoff, first = REMOTE_BACKOFF_S[0], True
        while not self.stop_ev.is_set() and not SHUTDOWN.is_set():
            if not first:
                with self.lock:
                    self.reconnects += 1
                    self.t_reconnects.append(time.time())
            first = False
            n0, self._session_frames = self.frames, 0
            try:
                why = session()
            except Exception as ex:
                why = describe_exc(ex)
            finally:
                self._close()
                self.connected = False
            if self.stop_ev.is_set():
                break
            if self.frames > n0:
                backoff = REMOTE_BACKOFF_S[0]
            if why != self.last_error or self.frames > n0:
                print("net-%s: %s; reconnecting (backoff %.1f..%.1f s)" % (self.name, why, *REMOTE_BACKOFF_S),
                      flush=True)
            self.last_error = why
            self.stream.error = "%s; reconnecting in %.1f s" % (why, backoff)
            self.stop_ev.wait(backoff)
            backoff = min(REMOTE_BACKOFF_S[1], backoff * 2)

    def _close(self):
        sock, self.sock = self.sock, None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)   # the response object may still hold a reference to it
            except Exception:
                pass
            try:
                sock.close()
            except Exception:
                pass
        proc, self.proc = self.proc, None
        if proc is not None:
            reap_child(proc)

    def _http(self):
        """-> (HTTPConnection, path, headers); user:password in the URL becomes HTTP basic auth."""
        u = urllib.parse.urlsplit(self.spec["url"])
        cls = http.client.HTTPSConnection if u.scheme.lower() == "https" else http.client.HTTPConnection
        conn = cls(u.hostname, u.port, timeout=REMOTE_CONNECT_TIMEOUT_S)
        path = (u.path or "/") + ("?" + u.query if u.query else "")
        headers = {"User-Agent": "camview", "Accept": "*/*"}
        if u.username is not None:
            cred = "%s:%s" % (urllib.parse.unquote(u.username), urllib.parse.unquote(u.password or ""))
            headers["Authorization"] = "Basic " + base64.b64encode(cred.encode()).decode()
        return conn, path, headers

    def _session_mjpeg(self):
        conn, path, headers = self._http()
        try:
            try:
                conn.connect()
            except socket.timeout:
                return "connect timed out (%.0f s)" % REMOTE_CONNECT_TIMEOUT_S
            conn.request("GET", path, headers=headers)
            self.sock = conn.sock    # kept: getresponse() hands the socket to the response on a streaming reply
            resp = conn.getresponse()
            if resp.status != 200:
                return "HTTP %d %s" % (resp.status, resp.reason)
            self.sock.settimeout(REMOTE_READ_TIMEOUT_S)
            ctype = resp.getheader("Content-Type", "") or ""
            if "multipart" not in ctype.lower() and ctype.lower().startswith("image/"):
                return "the URL serves a single image (%s): set \"mode\": \"poll\"" % ctype
            m = re.search(r'boundary="?([^";,\s]+)', ctype, re.I)
            parser = MjpegPartParser(m.group(1) if (m and "multipart" in ctype.lower()) else None)
            self.connected = True
            self.stream.error = "connected, waiting for the first frame"
            while not self.stop_ev.is_set():
                data = resp.read1(65536)
                if not data:
                    return "the camera closed the stream"
                for jpeg, t_remote in parser.feed(data):
                    self._offer(jpeg, t_remote)
                self.bad += parser.bad
                parser.bad = 0
            return "stopped"
        finally:
            conn.close()

    def _session_poll(self):
        conn, path, headers = self._http()
        period = 1.0 / (self.spec["fps"] or REMOTE_POLL_FPS)
        try:
            while not self.stop_ev.is_set():
                t0 = time.monotonic()
                if conn.sock is None:   # first request, or the server closed the last connection
                    try:
                        conn.connect()
                    except socket.timeout:
                        return "connect timed out (%.0f s)" % REMOTE_CONNECT_TIMEOUT_S
                conn.request("GET", path, headers=headers)
                self.sock = conn.sock
                conn.sock.settimeout(REMOTE_READ_TIMEOUT_S)
                resp = conn.getresponse()
                data = resp.read()
                if resp.status != 200:
                    return "HTTP %d %s" % (resp.status, resp.reason)
                if data[:2] != SOI:
                    self.bad += 1
                    return "snapshot is not a JPEG (Content-Type %s, %d bytes)" % (resp.getheader("Content-Type"),
                                                                                   len(data))
                self.connected = True
                if self.stream.error:
                    self.stream.error = "connected, waiting for a new snapshot"
                self._offer(data)
                self.stop_ev.wait(max(0.0, period - (time.monotonic() - t0)))
            return "stopped"
        finally:
            conn.close()

    def _session_ffmpeg(self):
        url = self.spec["url"]
        try:   # from this feed's own long-lived thread (PR_SET_PDEATHSIG, see spawn_child)
            proc = spawn_child(ffmpeg_remote_args(self.spec), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, bufsize=0)
        except FileNotFoundError:
            return "ffmpeg not installed"
        if proc is None:
            return "stopped"
        self.proc = proc
        errs = collections.deque(maxlen=3)

        def drain():   # keep ffmpeg's stderr moving (a full pipe would block it); remember the last lines
            for line in iter(proc.stderr.readline, b""):
                errs.append(line.decode("utf-8", "replace").strip().replace(url, redact_url(url)))
        drainer = threading.Thread(target=drain, daemon=True)
        drainer.start()

        def tail():
            drainer.join(0.5)
            return (": " + " | ".join(errs)) if errs else ""
        self.stream.error = "ffmpeg started, waiting for the first frame"
        fd, buf, got = proc.stdout.fileno(), b"", False
        t_frame = time.monotonic()
        while not self.stop_ev.is_set():
            limit = REMOTE_FFMPEG_NO_FRAME_S if got else REMOTE_FFMPEG_FIRST_FRAME_S
            if time.monotonic() - t_frame > limit:
                return "no new frame from ffmpeg for %.0f s%s" % (limit, tail())
            ready, _, _ = select.select([fd], [], [], 0.5)
            if not ready:
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                try:
                    code = proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    code = None
                return "ffmpeg exited (code %s)%s" % (code, tail())
            frames, buf = split_jpegs(buf + chunk)
            for jpeg in frames:
                if self._offer(jpeg):
                    got, t_frame = True, time.monotonic()
        return "stopped"

    def stop(self):
        self.stop_ev.set()
        if self.clock is not None:
            self.clock.stop()
        sock, proc = self.sock, self.proc
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)   # unblocks a pending read in the feed thread
            except Exception:
                pass
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
        with REG_LOCK:
            if REGISTRY.get(self.sid) is self.stream:
                REGISTRY.pop(self.sid)
        unregister_feed(self)


def load_remote_file(path):
    """remote_cams.json -> {name: raw entry} ({} if the file doesn't exist; keys starting with "_" are comments).
    Raises ValueError for unreadable JSON or a wrong top-level shape."""
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except ValueError as ex:
        raise ValueError("not valid JSON: %s" % ex)
    if not isinstance(data, dict):
        raise ValueError("the top level must be an object {\"name\": {\"url\": ...}, ...}")
    return {k: v for k, v in data.items() if not str(k).startswith("_")}


def parse_remote_cli(items):
    """--remote NAME=URL values -> {name: url}. Raises ValueError."""
    out = {}
    for item in items or ():
        name, sep, url = item.partition("=")
        if not sep:
            raise ValueError("--remote %r: expected NAME=URL" % item)
        out[name.strip()] = url.strip()
    return out


def remote_specs(file_raw, cli_raw):
    """File entries overridden by --remote ones -> ({name: spec}, [errors])."""
    merged = dict(file_raw)
    merged.update(cli_raw)
    specs, errors = {}, []
    for name, raw in merged.items():
        try:
            specs[name] = parse_remote_spec(name, raw)
        except ValueError as ex:
            errors.append(str(ex))
    return specs, errors


def remote_scanner(path, cli_raw):
    """Start / stop / restart RemoteFeeds to match the config file (re-read whenever it changes) + --remote flags.
    A broken file keeps the previous file entries running."""
    active, file_raw, sig = {}, {}, "unread"
    while not SHUTDOWN.is_set():
        try:
            st = os.stat(path)
            cur = (st.st_mtime_ns, st.st_size)
        except OSError:
            cur = None
        if cur != sig:
            sig = cur
            try:
                file_raw = load_remote_file(path)
            except (OSError, ValueError) as ex:
                print("%s: %s -- keeping the current network cameras" % (path, ex), flush=True)
            specs, errors = remote_specs(file_raw, cli_raw)
            for e in errors:
                print("network camera skipped: %s" % e, flush=True)
            for name in list(active):
                if name not in specs or specs[name] != active[name].spec:
                    active.pop(name).stop()
                    print("network camera net-%s stopped" % name, flush=True)
            for name, spec in sorted(specs.items()):
                if name not in active:
                    active[name] = RemoteFeed(spec)
                    print("network camera net-%s: %s (%s)" % (name, redact_url(spec["url"]), active[name].stream.mode),
                          flush=True)
        SHUTDOWN.wait(SCAN_PERIOD_S)


# ---------------------------------------------------------------- page + HTTP server
PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>camview</title><style>
:root{color-scheme:dark}body{margin:0;background:#0e0f12;color:#e8e8ea;font:16px -apple-system,system-ui,sans-serif;padding:16px}
h1{font-size:18px;margin:4px 0 10px}#st{color:#9a9ca3;font-size:13px;margin-bottom:12px}
#grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px}
.tile{background:#16181d;border-radius:10px;overflow:hidden}
.hd{padding:6px 10px;font-size:13px;color:#c9cbd1;background:#1a1c22}
.tile img{display:block;width:100%;aspect-ratio:4/3;object-fit:contain;background:#000}
.meta{padding:5px 10px;font-size:12px;color:#9a9ca3;font-variant-numeric:tabular-nums}
.tile.stale .meta{color:#e0a030}
</style></head><body>
<h1>camview</h1>
<div id="st">connecting&hellip;</div>
<div id="grid"></div>
<script>
const $=id=>document.getElementById(id);
const tiles=new Map();
function tileFor(cam){
 let t=tiles.get(cam.id);
 if(!t){const div=document.createElement("div");div.className="tile";
  div.innerHTML=`<div class="hd">${cam.group} &middot; ${cam.label}</div><img loading="lazy"><div class="meta"></div>`;
  const img=div.querySelector("img");img.src="/stream/"+encodeURIComponent(cam.id);
  $("grid").appendChild(div);t={el:div,meta:div.querySelector(".meta")};tiles.set(cam.id,t);}
 t.meta.textContent=cam.error?("error: "+cam.error):(cam.age_s==null?"waiting for first frame…":"age "+cam.age_s.toFixed(2)+" s · "+(cam.fps||0).toFixed(1)+" fps"+(cam.mode?" · "+cam.mode:""))+(cam.net?" · reconnects "+cam.net.reconnects+" · max gap 60 s "+(cam.net.last_gap_s_max_60s==null?"–":cam.net.last_gap_s_max_60s.toFixed(1)+" s"):"");
 t.el.classList.toggle("stale",!!cam.error||(cam.age_s!=null&&cam.age_s>2));
 return t;}
async function poll(){
 try{const cams=await(await fetch("/api/cameras")).json();
  const seen=new Set();
  for(const c of cams){tileFor(c);seen.add(c.id);}
  for(const [id,t] of tiles){if(!seen.has(id)){t.el.remove();tiles.delete(id);}}
  $("st").textContent=cams.length?(cams.length+" camera(s)"):"no cameras detected — plug one in";}
 catch(e){$("st").textContent="camview server unreachable";}
 setTimeout(poll,1500);}
poll();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/":
            b = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif self.path == "/api/cameras":
            b = json.dumps(list_streams()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif self.path.startswith("/stream/"):
            self._stream(self.path[len("/stream/"):])
        else:
            self.send_response(404)
            self.end_headers()

    def _stream(self, sid):
        with REG_LOCK:
            s = REGISTRY.get(sid)
        if s is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=%s" % BOUNDARY)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        # A part goes out only for a frame this client hasn't had: a stalled camera sends NOTHING (no keepalive
        # re-sends, which made a frozen camera look fresh to readers that age frames by arrival). A new client gets the
        # current frame only if it is younger than STREAM_FIRST_MAX_AGE_S. X-Frame-Seq (strictly increasing) and
        # X-Capture-Time / X-Arrival-Time (module doc) let readers check freshness themselves.
        sent = 0
        with s.cond:
            if s.jpeg is not None and time.time() - s.t_last > STREAM_FIRST_MAX_AGE_S:
                sent = s.seq
        try:
            while True:
                with REG_LOCK:
                    if REGISTRY.get(sid) is not s:
                        break
                jpeg, seq, t_cap, t_arr = s.wait_for_new(sent, timeout=2.0)
                if jpeg is None or seq == sent:
                    continue
                self.wfile.write(b"--" + BOUNDARY.encode() + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                  str(len(jpeg)).encode() + b"\r\nX-Frame-Seq: " + str(seq).encode() +
                                  b"\r\nX-Capture-Time: " + ("%.6f" % t_cap).encode() +
                                  b"\r\nX-Arrival-Time: " + ("%.6f" % t_arr).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                sent = seq
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--remote", action="append", metavar="NAME=URL",
                    help="network camera, stream id net-NAME (repeatable; overrides the same name in --remote-config)")
    ap.add_argument("--remote-config", default=REMOTE_CONFIG, metavar="PATH",
                    help="JSON {name: {url, label, mode, rotate, width, height, fps}}, re-read when it changes "
                         "(default: remote_cams.json next to camview.py; a missing file means none)")
    args = ap.parse_args()
    try:
        cli_remote = parse_remote_cli(args.remote)
        _, errors = remote_specs({}, cli_remote)
    except ValueError as ex:
        errors = [str(ex)]
    if errors:
        ap.error("; ".join(errors))
    server = None
    try:
        install_signal_handlers()
        threading.Thread(target=usb_scanner, daemon=True).start()
        threading.Thread(target=luxonis_scanner, daemon=True).start()
        threading.Thread(target=remote_scanner, args=(args.remote_config, cli_remote), daemon=True).start()
        print("camview on http://0.0.0.0:%d  (ctrl-c to quit)" % args.port, flush=True)
        server = ThreadingHTTPServer(("0.0.0.0", args.port), H)
        server.serve_forever()
    except StopCamview as ex:
        log("camview: %s, shutting down" % ex)
    finally:
        begin_shutdown()
        if server is not None:
            server.server_close()   # listening socket closed: the port is free again
        shutdown_all_feeds()
        log("camview: stopped")


# ---------------------------------------------------------------- signals
class StopCamview(BaseException):
    """Raised in the main thread by the SIGINT / SIGTERM / SIGHUP handler: unwinds serve_forever() (like Ctrl-C's
    KeyboardInterrupt) into main()'s cleanup. BaseException so no `except Exception` on the way swallows it."""


_STOPPING = []   # the first signal's name, once shutdown has begun (a list: set from the handler without a lock)
_WATCHDOG = []


def hard_exit(why, code=1):
    kill_children_now()
    log("camview: %s: exiting now" % why)
    os._exit(code)


def _arm_alarm():
    """Kernel-side backstop: SIGALRM (default action: terminate the process) 2 s after the watchdog would have fired,
    in case the main thread hangs in C code holding the GIL (e.g. a depthai call) so no Python thread can run."""
    if hasattr(signal, "alarm"):
        signal.alarm(int(SHUTDOWN_TIMEOUT_S) + 2)


def begin_shutdown():
    """Arm the shutdown bound (once, main thread): a watchdog thread hard-exits after SHUTDOWN_TIMEOUT_S."""
    if _WATCHDOG:
        return
    _WATCHDOG.append(True)
    _arm_alarm()

    def watchdog():
        time.sleep(SHUTDOWN_TIMEOUT_S)
        hard_exit("shutdown took longer than %.0f s" % SHUTDOWN_TIMEOUT_S)
    threading.Thread(target=watchdog, daemon=True, name="shutdown-watchdog").start()


def _on_signal(signum, frame):
    # Runs in the main thread between bytecodes, possibly while it holds a lock: no thread starts or locking here
    # (the watchdog thread is started by begin_shutdown() in main()'s finally, a moment later).
    name = signal.Signals(signum).name
    if _STOPPING:
        if signum == signal.SIGHUP:   # a hangup often arrives twice (tty + the shell forwarding it): not a 2nd request
            return
        hard_exit("second signal (%s) during shutdown" % name)
    _STOPPING.append(name)
    _arm_alarm()
    raise StopCamview(name)


def install_signal_handlers():
    """SIGINT (Ctrl-C), SIGTERM (kill / pkill) and SIGHUP (ssh session gone) all run the same clean shutdown. A second
    SIGINT / SIGTERM while it runs exits at once (children SIGKILLed first). A signal that camview inherited as ignored
    stays ignored (`nohup camview.py &` must survive the hangup)."""
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        if signal.getsignal(sig) != signal.SIG_IGN:
            signal.signal(sig, _on_signal)


if __name__ == "__main__":
    main()
