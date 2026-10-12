"""Excavator dashboard: camera grid + gate status + keyboard / gamepad teleop + episode recording, one web page.
Standard library only. Same path as calib.py: rig -> 127.0.0.1:8096 -> gate.

Run ON the Pi from the act-athon repo root (it starts, watches and stops camview.py, the camera server on port 8100,
itself: --camview auto, see "camview supervisor" below):
    python3 dashboard.py            (optional: --cam front=<camview stream id> to seed the camera assignment)
then open http://<pi>:8080 on a laptop and pick the recorded cameras on their tiles ("Record as"). Someone at the kill switch, nobody within 3 m of the arm. Docs: docs/dashboard.md.

Teleoperate is OFF when the page loads: until you switch it on the dashboard sends NOTHING to the gate (it only reads
/state). With it on, the key map is (rig signs in brackets):
    hold SPACE   deadman: nothing moves unless it is held      (gamepad: hold L1 or R1)
    A / D        swing left / right        [swing -/+]          left stick X
    W / S        arm out / in              [arm +/-]            left stick Y (push forward = out)
    I / K        boom down / up            [boom +/-]           right stick Y (push forward = down)
    J / L        bucket curl / dump        [bucket -/+]         right stick X (left = curl)
    U / O        thumb open / clamp        [thumb -/+]          LT open / RT clamp
    full scale (-1 / 0 / +1 per key); gamepad: deadzone 0.12 + expo 0.5 as in src/act/teleop.py
    ESC or X or the STOP button: neutral + relay open (always, even with Teleoperate off)
Commands repeat every 100 ms while held. Release the deadman / all inputs, blur the window or hide the tab -> neutral.
Page stops sending (tab closed, network drop) -> this server neutrals within 0.3 s; the gate itself after 0.5 s.
Switching Teleoperate off sends one neutral (if this session wrote anything) and then nothing.
After a STOP nothing is written until someone switches Teleoperate on again; a request that was sent before the STOP but
arrives after it is refused. Ctrl-C (or SIGTERM / SIGHUP) on the server sends STOP.

Policy (button next to Teleoperate): a policy runner ON THIS PI drives the arm, a human holds the deadman here.
The runner (scripts/control_loop.py --machine dashboard, in the venv) reads /state and the cameras itself (read only)
and posts every action to POST /api/policy/act (accepted from this Pi only). This server is still the only writer:
an action reaches the gate only while Policy is on AND the deadman (SPACE, gamepad L1 / R1) was held within the last
0.3 s; otherwise it writes one neutral (if the policy was driving) and nothing after. Releasing the deadman neutrals
at once; the runner stopping (crash, Ctrl-C) neutrals within 0.3 s (watchdog). Policy, Teleoperate and recording
exclude each other; STOP switches Policy off like Teleoperate.
The Policy runner panel (shown in Policy mode) starts / stops that runner for you: backend, URL, prompt and training
dataset (a folder in runner/datasets/ with meta/info.json + crops.json) in, and it launches runner/control_loop.py with
the venv's python (--runner-python), --cam taken from the "Record as" assignment (the same names as in training).
Backends: "pi07 (PI gateway)" = --policy pi --gateway-url <the hosted checkpoint's gateway_url>, PI_API_KEY from
runner/.env; "pi0.5 (Modal URL)" = --policy http --url <modal_jobs/serve_pi05.py's URL> (the fallback),
PI05_SERVE_TOKEN from runner/.env. Stop sends the runner Ctrl-C: it exits through its normal shutdown, which asks
this server for STOP. Starting the runner moves nothing by itself.

Cameras come from camview.py, the only process that opens them (V4L2 devices can't be opened twice). The page's
tiles load camview's MJPEG streams directly (http://<same host>:8100/stream/<id>); every camview stream gets a tile
(hide some with --exclude PATTERN). Each tile has a "Record as" selector: off / front / wrist / side / world (SLOTS below; a
name is on at most one stream). Assigned streams are the ones recorded (the name is the dataset key), and this server
reads each of them over loopback with one persistent reader (newest JPEG + arrival time, reconnect with backoff);
unassigned tiles are browser-only views. Assignments bind to the camview stream id, are locked while an episode is
open, and persist in --assign-file (default cam_assign.json next to this script); --cam NAME=ID seeds / overrides
them at startup. An assigned stream that camview no longer lists shows as "assigned but missing".

camview supervisor (--camview auto, the default): this server runs camview.py as its child (--camview-python, default
.venv/bin/python3 next to this script: the venv with depthai), output in logs/camview.log. A camview.py of this user
already on the port is taken over (SIGTERM, then ours starts); anything else on the port is used as is ("external",
never signalled). The child is restarted (backoff 1 .. 30 s) when it exits or its /api/cameras stops answering for 10 s,
but never while an episode is recording or the policy runner / Policy is on. Quitting the dashboard stops it; on Linux
it also gets SIGTERM if the dashboard dies (PR_SET_PDEATHSIG). --camview external = use what is on the port, never
start / stop anything; --camview off = no supervisor at all.

Recording (Record panel; Enter = start / save, Backspace = discard when no text field has focus) records a HUMAN
OPERATOR IN THE CAB driving with the machine's own joysticks; keyboard / gamepad driving is never recorded. While an
episode is open this dashboard sends NOTHING to the gate: Start is refused while Teleoperate is on, and Teleoperate
can't be switched on until the episode ends (STOP / Esc / X still send /estop at any time, and then discard the episode).
A server-side loop at --rec-fps on the Pi clock writes one step per tick to
~/act-athon/episodes/<YYYYmmdd_HHMMSS>_<slug>.partial/ (steps.jsonl + <cam>/<step>.jpg, camview's JPEG bytes
unchanged) and renames the folder without `.partial` on save. Each step: the action derived from the gate's raw
`controls` (the cab input) through CAB_MAP + CAB_DEADBAND, the raw `controls` dict itself (so a wrong map can be fixed
at conversion), the gate /state from the one shared status poller, and one frame per named camera with its arrival
and capture time, picked by --sync (camsync.py; default aligned: all frames of a step captured at the same instant,
which the world camera's network delay would otherwise skew; latest: each camera's newest frame). The mode is in
meta.json "sync"; the Policy runner is started with the same one. Auto-saves after --episode-s. Needs at least one assigned camera; Start asks for confirmation
(confirm_new_camera_set) when the camera set differs from the last saved episode's. The Cab monitor (page section, GET /api/cab)
shows the raw `controls` fields live, to check the map. Convert on the laptop with scripts/convert_pi_episodes.py.
"""
import argparse
import ctypes
import fnmatch
import glob
import hashlib
import json
import math
import os
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camsync
import rig
try:                         # optional: the Record panel's QC card (docs/episode-qc.md); teleop / recording never need it
    import episode_qc
    EPISODE_QC_ERROR = None
except ImportError as _e:
    episode_qc = None
    EPISODE_QC_ERROR = "episode_qc.py not importable (%s): QC card disabled" % _e

VERSION = "2026-09-26 camview+cabrec+camassign+policy+runner+pi05+camsync+camsup"
try:
    with open(os.path.abspath(__file__), "rb") as _f:
        SOURCE_SHA1 = hashlib.sha1(_f.read()).hexdigest()[:12]
except OSError:
    SOURCE_SHA1 = "?"

PORT = 8080
CAMVIEW_URL = "http://127.0.0.1:8100"
WATCHDOG_S = 0.3             # page stopped sending -> neutral (same as calib.py)
BACKOFF_MAX_S = 10.0         # camview stream failed: reconnect after 1, 2, 4, ... s (reset once it ran for STABLE_S)
STABLE_S = 10.0
NO_FRAME_S = 5.0             # connected but no frame for this long -> reconnect
MAX_JPEG = 8 << 20           # a "frame" bigger than this is garbage: resync
# dataset channel order (src/act/machine.py CHANNELS) and rig signs
CHANNELS = ("swing", "boom", "arm", "bucket", "thumb")
SIGNS = {"swing": "+ = house right", "boom": "+ = down", "arm": "+ = out", "bucket": "+ = dump/open",
         "thumb": "+ = clamp in"}
# Cab joysticks -> recorded action: which raw /state `controls` field each rig channel comes from, and its sign
# relative to the rig signs above. Entries: channel -> (raw field, sign) or (raw field, sign, rest offset);
# action = clip(sign * (raw - offset), -1, 1), and 0 while |raw - offset| <= deadband. `sign` is a multiplier.
# VERIFIED on the machine 2026-09-26 (an operator in the cab calling out each move, relay open, `sent` = 0
# throughout): the cab joysticks are `controls.*` on the same -1..1 scale as the rig, rest offsets within +-0.04
# (deadband 0.08 covers them), and every channel maps with sign +1:
#   swing <- body (- = house left), boom <- boom (+ = down), arm <- arm (+ = extend / out, - = curl / in),
#   bucket <- bucket (+ = out / dump), thumb <- thumb (+ = clamp / curl, - = open / extend).
#   `controls.swing` and `controls.blade` only move +-0.2 together: NOT house swing, ignore them.
# Override without editing code: --cab-map map.json, e.g. {"arm": ["arm", -1.0], "deadband": 0.1} (listed channels
# replace these entries, the rest are kept; "deadband" is a number or {channel: number}). The converter
# (scripts/convert_pi_episodes.py) can re-derive the action from the stored raw `controls` with a corrected map.
CAB_MAP = {"swing": ("body", +1.0), "boom": ("boom", +1.0), "arm": ("arm", +1.0), "bucket": ("bucket", +1.0),
           "thumb": ("thumb", +1.0)}
CAB_DEADBAND = 0.08
CAB_STATUS = {ch: "verified" for ch in CAB_MAP}   # built-in entries; a --cab-map entry is reported as "--cab-map"
# Recorded camera names ("Record as" on a tile) = dataset image keys (observation.images.<name>). Each maps to one
# model image slot, and that mapping MUST be identical in training and inference (AGENTS.md):
#   pi0.5:  front -> base_0_rgb, wrist -> left_wrist_0_rgb, side -> right_wrist_0_rgb
#   pi07 training profile: front -> base camera, wrist -> wrist camera
#   world: static third-person camera on a laptop (worldcam/worldcam.py), relayed by camview as net-world. pi0.5 has
#   no 4th slot (record world instead of side there); pi07 matches cameras by name.
# Recorded in meta.json ("slot" per camera, "camera_slots").
SLOTS = {"front": "base_0_rgb", "wrist": "left_wrist_0_rgb", "side": "right_wrist_0_rgb", "world": "third_person_rgb"}
ASSIGN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cam_assign.json")
TRACKS = ("left_track", "right_track")      # the policy can't drive the tracks: steps where they move are counted
TRACK_WARN = 0.1

LOCK = threading.Lock()      # every gate write / stop happens under this lock, so nothing can be written after a STOP
BOOT = uuid.uuid4().hex[:8]  # epochs from a previous server run can never match
# on: Teleoperate switched on; epoch: id of the current teleop session (None when off); n: highest request seq seen
# in this epoch (older writes arriving late are dropped); wrote: this session has written to the gate
TELE = {"on": False, "epoch": None, "count": 0, "n": -1, "wrote": False}
LAST = {"t": 0.0, "cmd": {}}                # watchdog: last non-neutral write
# Policy mode: on / epoch / n as TELE (the page's deadman heartbeats); hold_t: last "deadman held" from the page;
# wrote: this session has written to the gate; runner_t: last action posted by the policy runner
POL = {"on": False, "epoch": None, "count": 0, "n": -1, "hold_t": 0.0, "wrote": False, "runner_t": 0.0}
# owner: id of the episode being recorded from the cab (None = no recording). Set under LOCK together with the
# "Teleoperate is off" check, so teleop can't be switched on between that check and the episode starting.
REC_GUARD = {"owner": None}
GONE = {"t": 0.0, "n": 0, "slow_ms": 0.0}    # replies whose client had already hung up (see Handler._send)
LAST_STOP = {"sid": None, "ct": None}       # last STOP from a page (session id + browser clock), as calib.py
STATUS = {"snap": {"gate": None, "error": "not polled yet", "t": 0.0}}   # newest /state (one poller for everyone)
CAMS = {}                                   # assigned name -> CamReader (only assigned cameras are read / recorded)
# CAM_LOCK serialises assignment changes against Start (which takes REC_GUARD under it): no change while recording.
CAM_LOCK = threading.Lock()
CFG = {"camview_url": CAMVIEW_URL, "camview_public": None, "exclude": [], "no_auto": False, "assign_file": ASSIGN_FILE,
       "rec_fps": 15.0, "episode_s": 90.0, "episodes_dir": os.path.expanduser("~/act-athon/episodes"),
       "poll_hz": 10.0, "cab_map": None, "cab_deadband": None, "cab_map_file": None,
       "cab_status": dict(CAB_STATUS), "sync": camsync.DEFAULT_MODE}
RUNNING = threading.Event()
RUNNING.set()
# Policy runner (runner/control_loop.py --machine dashboard) started from the page: one at a time
HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = {"dir": os.path.join(HERE, "runner"), "python": os.path.join(HERE, ".venv", "bin", "python")}
RUN = {"proc": None, "since": None, "exit": None, "cfg": None, "lines": deque(maxlen=60)}
RUN_LOCK = threading.Lock()
# runner backends: page value -> (label, control_loop.py policy args before the URL)
RUNNER_BACKENDS = {"pi07": ("pi07 (PI gateway)", ["--policy", "pi", "--gateway-url"]),
                   "pi05": ("pi0.5 (Modal URL)", ["--policy", "http", "--url"])}


# ---------------------------------------------------------------- cab map (same code in scripts/convert_pi_episodes.py)
def cab_map_merge(obj, base_map, base_deadband):
    """Map JSON ({channel: [field, sign(, offset)], "deadband": number | {channel: number}}; listed entries replace
    base's) -> (map {ch: [field, sign, offset]}, deadband {ch: float}). Raises ValueError on a malformed entry."""
    if not isinstance(obj, dict):
        raise ValueError("cab map must be a JSON object")
    cmap = {ch: [str(e[0]), float(e[1]), float(e[2]) if len(e) > 2 else 0.0] for ch, e in base_map.items()}
    db = dict(base_deadband) if isinstance(base_deadband, dict) else {ch: float(base_deadband) for ch in CHANNELS}
    for k, v in obj.items():
        if k == "deadband":
            vals = v if isinstance(v, dict) else {ch: v for ch in CHANNELS}
            for ch, x in vals.items():
                if ch not in CHANNELS or isinstance(x, bool) or not isinstance(x, (int, float)) or x < 0:
                    raise ValueError("bad deadband entry %r: %r" % (ch, x))
                db[ch] = float(x)
        elif k in CHANNELS:
            if (not isinstance(v, (list, tuple)) or len(v) not in (2, 3) or not isinstance(v[0], str)
                    or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in v[1:])):
                raise ValueError("bad map entry %r: %r (want [raw field, sign] or [raw field, sign, offset])" % (k, v))
            cmap[k] = [v[0], float(v[1]), float(v[2]) if len(v) > 2 else 0.0]
        else:
            raise ValueError("unknown key %r (channels: %s, or deadband)" % (k, ", ".join(CHANNELS)))
    missing = [ch for ch in CHANNELS if ch not in cmap or ch not in db]
    if missing:
        raise ValueError("cab map has no entry for %s" % missing)
    return {ch: cmap[ch] for ch in CHANNELS}, {ch: db[ch] for ch in CHANNELS}


def cab_action(controls, cmap, deadband):
    """Raw /state `controls` -> {channel: -1..1} (rig signs). A missing / non-numeric field gives 0."""
    out = {}
    for ch in CHANNELS:
        field, sign, off = cmap[ch]
        v = controls.get(field) if isinstance(controls, dict) else None
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            out[ch] = 0.0
            continue
        x = float(v) - off
        out[ch] = 0.0 if abs(x) <= deadband[ch] else max(-1.0, min(1.0, sign * x))
    return out


def numeric_fields(controls):
    """All numeric fields of `controls` (bools as 0 / 1), for the Cab monitor."""
    return {k: (int(v) if isinstance(v, bool) else v) for k, v in sorted((controls or {}).items())
            if isinstance(v, (int, float)) and (isinstance(v, bool) or math.isfinite(v))}


def cab_info():
    """Read-only Cab monitor: newest raw `controls` from the one poller, and the action the map derives from it."""
    snap = STATUS["snap"]
    g = snap.get("gate")
    c = g["controls"] if g is not None else None
    return {"ok": g is not None, "error": snap.get("error"),
            "poll_age": round(time.monotonic() - snap["t"], 3) if snap.get("t") else None,
            "controls": numeric_fields(c) if c is not None else None,
            "other": {k: v for k, v in (c or {}).items() if not isinstance(v, (int, float))},   # e.g. fresh
            "action": cab_action(c, CFG["cab_map"], CFG["cab_deadband"]) if c is not None else None,
            "missing": [ch for ch in CHANNELS if c is not None and CFG["cab_map"][ch][0] not in c],
            "tracks": tracks_moving(c), "map": CFG["cab_map"], "deadband": CFG["cab_deadband"],
            "map_file": CFG["cab_map_file"], "status": CFG["cab_status"]}


def tracks_moving(controls):
    """The track fields beyond TRACK_WARN (the operator is driving the tracks: the arm policy can't)."""
    out = []
    for k in TRACKS:
        v = (controls or {}).get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and abs(v) > TRACK_WARN:
            out.append(k)
    return out


# ---------------------------------------------------------------- gate
def parse_state(s):
    """/state -> what the page shows and the recorder stores. `sent` = what the gate is outputting (verified on
    hardware: it tracks our commands while `controls.*` stays 0); `controls` = the live inputs the gate reads (CAN +
    ADC + buttons: the cab joysticks, see CAB_MAP), kept whole."""
    c = s.get("controls") if isinstance(s.get("controls"), dict) else {}
    sent = s.get("sent") if isinstance(s.get("sent"), dict) else {}
    last = s.get("last_cmd") if isinstance(s.get("last_cmd"), dict) else {}
    return {"live": bool(s.get("armed")), "veto": s.get("veto") or "", "throttle": c.get("throttle"),
            "rabbit": c.get("rabbit_button"), "age": c.get("age"), "has_sent": isinstance(s.get("sent"), dict),
            "sent": {k: float(sent.get(rig._TO_API.get(k, k)) or 0.0) for k in CHANNELS},
            "last_cmd": {k: float(last.get(rig._TO_API.get(k, k)) or 0.0) for k in CHANNELS},
            "controls": c}


def poller():
    """The only /state reader: the page and the recorder both use its newest result (read-only, never writes)."""
    while RUNNING.is_set():
        t0 = time.monotonic()
        try:
            snap = {"gate": parse_state(rig._http("/state")), "error": None}
        except Exception as ex:
            snap = {"gate": None, "error": str(ex) or type(ex).__name__}
        snap["t"] = time.monotonic()
        STATUS["snap"] = snap
        time.sleep(max(0.01, 1.0 / CFG["poll_hz"] - (time.monotonic() - t0)))


def _neutral_mark():
    """Under LOCK: nothing is commanded any more (the caller writes neutral / stop if appropriate)."""
    LAST["cmd"] = {}


def teleop_on(b):
    with LOCK:
        if (b.get("sid") is not None and b.get("sid") == LAST_STOP["sid"] and b.get("ct") is not None
                and LAST_STOP["ct"] is not None and float(b["ct"]) <= LAST_STOP["ct"]):
            raise RuntimeError("STOP was pressed after this request was sent")
        if REC_GUARD["owner"] is not None:
            raise RuntimeError("recording from the cab: Teleoperate stays off until the episode is saved or discarded")
        if POL["on"]:
            raise RuntimeError("Inference is on: switch it off first")
        TELE["count"] += 1
        TELE.update(on=True, epoch="%s-%d" % (BOOT, TELE["count"]), n=-1, wrote=False)
        _neutral_mark()
        return {"on": True, "epoch": TELE["epoch"]}      # nothing is written: the deadman + a key does that


def teleop_off(b):
    with LOCK:
        if not TELE["on"] or b.get("epoch") != TELE["epoch"]:
            return {"on": TELE["on"], "skipped": "not the current session"}
        wrote = TELE["wrote"]
        TELE.update(on=False, epoch=None, n=-1, wrote=False)
        _neutral_mark()
        res = rig.write() if wrote else {}       # one neutral, then nothing; never energise a relay we never used
        return {"on": False, "neutral_sent": wrote, "gate": res}


def check_session(b):
    """Under LOCK. Raises PermissionError if this write belongs to an ended session; returns False if it is older than
    one already handled (reordered on the network: drop it silently)."""
    if not TELE["on"] or b.get("epoch") != TELE["epoch"]:
        raise PermissionError("teleop is off (STOP, switched off, or taken over by another page)")
    n = int(b.get("n", -1))
    if n <= TELE["n"]:
        return False
    TELE["n"] = n
    return True


def clamp_cmd(v):
    """-1..1 joystick value; anything non-numeric or non-finite (NaN, inf) is 0 (neutral). NOTE: max(-1, min(1, nan))
    is 1.0 in Python, so a plain clamp would turn a NaN from a model into full-scale motion."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(-1.0, min(1.0, v)) if math.isfinite(v) else 0.0


def gate_write(b):
    cmd = {k: clamp_cmd(v) for k, v in (b.get("cmd") or {}).items() if k in rig.KEYS}
    with LOCK:
        if not check_session(b):
            return {"skipped": "stale"}
        TELE["wrote"] = True
        LAST["t"] = time.time()
        LAST["cmd"] = cmd
        return rig.write(**cmd)


def gate_neutral(b):
    with LOCK:
        if not check_session(b):
            return {"skipped": "stale"}
        _neutral_mark()
        if not TELE["wrote"]:
            return {"skipped": "nothing written yet"}
        return rig.write()


def gate_stop(b):
    with LOCK:
        TELE.update(on=False, epoch=None, n=-1, wrote=False)
        POL.update(on=False, epoch=None, n=-1, hold_t=0.0, wrote=False)
        _neutral_mark()
        if b.get("sid") is not None and b.get("ct") is not None:
            LAST_STOP.update(sid=b["sid"], ct=float(b["ct"]))
        return rig.stop()


def policy_on(b):
    with LOCK:
        if (b.get("sid") is not None and b.get("sid") == LAST_STOP["sid"] and b.get("ct") is not None
                and LAST_STOP["ct"] is not None and float(b["ct"]) <= LAST_STOP["ct"]):
            raise RuntimeError("STOP was pressed after this request was sent")
        if REC_GUARD["owner"] is not None:
            raise RuntimeError("recording from the cab: Inference stays off until the episode is saved or discarded")
        if TELE["on"]:
            raise RuntimeError("Teleoperate is on: switch it off first")
        POL["count"] += 1
        POL.update(on=True, epoch="%s-p%d" % (BOOT, POL["count"]), n=-1, hold_t=0.0, wrote=False)
        _neutral_mark()
        return {"on": True, "epoch": POL["epoch"]}      # nothing is written: deadman + runner do that


def policy_off(b):
    with LOCK:
        if not POL["on"] or b.get("epoch") != POL["epoch"]:
            return {"on": POL["on"], "skipped": "not the current session"}
        POL.update(on=False, epoch=None, n=-1, hold_t=0.0, wrote=False)
        _neutral_mark()
        # Stop inference also gives up control: neutral + relay open (/estop), so the cab can drive at once
        # (it can't while the relay is closed). The runner keeps running; its actions are refused while Policy is off.
        return {"on": False, "relay_opened": True, "gate": rig.stop()}


def policy_held():
    """Under LOCK: Policy is on and the page held the deadman within the last WATCHDOG_S."""
    return POL["on"] and time.time() - POL["hold_t"] <= WATCHDOG_S


def policy_hold(b):
    """Page, every 100 ms while Policy is on and the deadman is (or just stopped being) held: {epoch, n, held}."""
    with LOCK:
        if not POL["on"] or b.get("epoch") != POL["epoch"]:
            raise PermissionError("policy is off (STOP, switched off, or taken over by another page)")
        n = int(b.get("n", -1))
        if n <= POL["n"]:
            return {"skipped": "stale"}
        POL["n"] = n
        if b.get("held"):
            POL["hold_t"] = time.time()
            return {"held": True}
        POL["hold_t"] = 0.0
        if LAST["cmd"]:                          # released while the policy was driving: neutral now
            _neutral_mark()
            return {"held": False, "gate": rig.write()}
        return {"held": False}


def policy_act(b):
    """Policy runner on this Pi, every control tick: {cmd: {channel: -1..1}} -> {allowed, on}. Written to the gate
    only while policy_held(); otherwise one neutral if this session was moving, then nothing."""
    cmd = {k: clamp_cmd(v) for k, v in (b.get("cmd") or {}).items() if k in rig.KEYS}
    with LOCK:
        POL["runner_t"] = time.time()
        if policy_held():
            POL["wrote"] = True
            LAST["t"] = time.time()
            LAST["cmd"] = cmd
            return {"allowed": True, "on": True, "gate": rig.write(**cmd)}
        res = {}
        if POL["on"] and LAST["cmd"]:
            _neutral_mark()
            res = rig.write()
        return {"allowed": False, "on": POL["on"], "gate": res}


def policy_info():
    with LOCK:
        age = time.time() - POL["runner_t"] if POL["runner_t"] else None
        return {"on": POL["on"], "epoch": POL["epoch"], "held": policy_held(),
                "runner_age": round(age, 2) if age is not None else None}


def runner_datasets():
    base = os.path.join(RUNNER["dir"], "datasets")
    try:
        return sorted(d for d in os.listdir(base) if os.path.isfile(os.path.join(base, d, "meta", "info.json")))
    except OSError:
        return []


DS_CAMS = {}                                  # dataset -> ((info.json mtime), camera names): runner_info runs at 5 Hz


def dataset_cams(ds):
    """Camera names a runner dataset was trained on (observation.images.<name> in meta/info.json), or None."""
    p = os.path.join(RUNNER["dir"], "datasets", ds, "meta", "info.json")
    try:
        mt = os.stat(p).st_mtime_ns
        hit = DS_CAMS.get(ds)
        if hit is not None and hit[0] == mt:
            return hit[1]
        with open(p) as f:
            names = sorted(k.split(".", 2)[2] for k in json.load(f)["features"] if k.startswith("observation.images."))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    DS_CAMS[ds] = (mt, names)
    return names


DS_ZERO = {}                                  # dataset -> (meta dir mtime, has meta/zero_state.json)


def dataset_zero_state(ds):
    """A "zero-state" copy (meta/zero_state.json exists): its model must run with control_loop.py --zero-state.
    Cached by the meta folder's mtime (creating / deleting the marker changes it)."""
    meta = os.path.join(RUNNER["dir"], "datasets", ds, "meta")
    try:
        mt = os.stat(meta).st_mtime_ns
    except OSError:
        return False
    hit = DS_ZERO.get(ds)
    if hit is not None and hit[0] == mt:
        return hit[1]
    zero = os.path.isfile(os.path.join(meta, "zero_state.json"))
    DS_ZERO[ds] = (mt, zero)
    return zero


def cam_mapping_now():
    """Assigned name -> the stream it is bound to now, and whether it delivers (for the runner panel)."""
    out = {}
    for n, c in list(CAMS.items()):
        f, st = getattr(c, "frame", None), getattr(c, "status", None)
        ok = c.id is not None and st == "ok" and f is not None and time.monotonic() - f[2] <= 1.0
        out[n] = {"id": c.id, "pattern": getattr(c, "pattern", None), "label": getattr(c, "label", None), "ok": ok,
                  "status": "missing" if c.id is None else st if st != "ok" or ok else "stalled",
                  "msg": getattr(c, "msg", None)}
    return out


def _runner_output(proc):
    """Runner stdout+stderr -> RUN["lines"] (its status line is redrawn with \r: every redraw is a line)."""
    buf = b""
    while True:
        chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") else proc.stdout.read(1)
        if not chunk:
            break
        buf += chunk
        parts = re.split(rb"[\r\n]", buf)
        buf = parts.pop()
        for ln in parts:
            if ln.strip():
                RUN["lines"].append(ln.decode("utf-8", "replace")[:1000])
    if buf.strip():
        RUN["lines"].append(buf.decode("utf-8", "replace")[:1000])
    code = proc.wait()
    with RUN_LOCK:
        if RUN["proc"] is proc:
            RUN.update(proc=None, exit=code)
    RUN["lines"].append("[runner exited with code %s]" % code)


# The runner sends PI_API_KEY (pi07) or PI05_SERVE_TOKEN (pi05) to the URL it is given, so only trusted hosts are
# accepted: PI's gateway and Modal web endpoints, over TLS. Extra hosts (exact names) can be allowed with
# RUNNER_EXTRA_HOSTS=host1,host2 in the dashboard's environment.
RUNNER_HOST_SUFFIXES = {"pi07": (".pi-fleet.com",), "pi05": (".modal.run",)}
RUNNER_SCHEMES = {"pi07": ("wss", "https"), "pi05": ("https",)}


def runner_url_ok(backend, url):
    """-> (ok, reason). Rejects anything that isn't TLS to a pinned host: the runner would send the API key there."""
    if not url or len(url) > 2000 or re.search(r"\s", url) or url.startswith("-"):
        return False, "URL: paste the checkpoint's gateway_url / the Modal URL (one line, no spaces)"
    u = urllib.parse.urlsplit(url)
    host = (u.hostname or "").lower()
    extra = {h.strip().lower() for h in os.environ.get("RUNNER_EXTRA_HOSTS", "").split(",") if h.strip()}
    if u.scheme not in RUNNER_SCHEMES[backend]:
        return False, "URL must start with %s:// (the API key is sent to it)" % " or ".join(RUNNER_SCHEMES[backend])
    if u.username or u.password:
        return False, "URL must not contain user:password@"
    if not (host in extra or any(host.endswith(sfx) for sfx in RUNNER_HOST_SUFFIXES[backend])):
        return False, ("URL host %r is not allowed for %s (allowed: *%s%s): the runner would send the API key there"
                       % (host, backend, ", *".join(RUNNER_HOST_SUFFIXES[backend]),
                          (", " + ", ".join(sorted(extra))) if extra else ""))
    return True, ""


def _num(v, name, lo, hi, cast=float):
    try:
        x = cast(v)
    except (TypeError, ValueError):
        raise ValueError("%s: a number" % name)
    if not (lo <= x <= hi):
        raise ValueError("%s: %s .. %s" % (name, lo, hi))
    return x


def runner_options(b, backend, zero):
    """The Inference panel's runner options -> (control_loop.py flags, what was chosen). Checked with the runner's own
    rules (scripts/control_loop.py refuses the same combinations), so a bad choice fails here, not in the runner log."""
    chunk = _num(b.get("chunk") or 25, "chunk", 1, 50, int)
    req_at = _num(b.get("request_at") or 8, "request at", 1, 49, int)
    if req_at >= chunk:
        raise ValueError("request at must be smaller than the chunk (%d)" % chunk)
    subtasks = bool(b.get("subtasks"))
    if subtasks and backend != "pi07":
        raise ValueError("subtasks: pi07 only (the phase goes out as PI's raw_text; the Modal servers take no subtask)")
    sar = b.get("stop_after_release")
    sar = None if sar in (None, "") else _num(sar, "stop after release", 0.0, 60.0)
    if sar is not None and not subtasks:
        raise ValueError("stop after release needs subtasks (the release comes from the phase tracker)")
    grip = bool(b.get("grip_latch"))
    grip_thr = _num(b.get("grip_latch_threshold") or 0.3, "grip latch threshold", 0.01, 1.0)
    sfd = None if zero else bool(b.get("state_from_drove"))      # zero-state: the policy sees zeros, never passed
    args = ((["--chunk", str(chunk)] if backend == "pi07" else [])  # execute_chunk_size: pi-sdk only
            + ["--request-at", str(req_at)]
            + (["--subtasks"] if subtasks else [])
            + (["--stop-after-release", "%g" % sar] if sar is not None else [])
            + (["--grip-latch", "--grip-latch-threshold", "%g" % grip_thr] if grip else [])
            + ([] if sfd is None else ["--state-from-drove" if sfd else "--no-state-from-drove"]))
    opts = {"chunk": chunk, "request_at": req_at, "subtasks": subtasks, "stop_after_release": sar,
            "grip_latch": grip, "grip_latch_threshold": grip_thr, "state_from_drove": sfd}
    return args, opts


def runner_start(b):
    backend = str(b.get("backend") or "pi07")            # absent = pi07 (the page before the backend selector)
    if backend not in RUNNER_BACKENDS:
        raise ValueError("backend: one of %s" % ", ".join(sorted(RUNNER_BACKENDS)))
    url = str(b.get("gateway_url" if backend == "pi07" else "url") or "").strip()
    prompt = str(b.get("prompt") or "").strip()
    ds = str(b.get("dataset") or "")
    ok, why = runner_url_ok(backend, url)
    if not ok:
        raise ValueError(why)
    if not prompt or len(prompt) > 200:
        raise ValueError("prompt: the exact training task string (1-200 characters)")
    if ds not in runner_datasets():
        raise ValueError("dataset: pick one of runner/datasets/ (%s)" % (", ".join(runner_datasets()) or "none there"))
    need = dataset_cams(ds)
    zero = dataset_zero_state(ds)
    sync = CFG["sync"]      # camsync frame selection: the runner uses the dashboard's own --sync (see runner panel line)
    if need is None:
        raise ValueError("dataset %s: meta/info.json unreadable" % ds)
    if not os.path.isfile(RUNNER["python"]):
        raise ValueError("no python at %s (--runner-python)" % RUNNER["python"])
    # default 25 / 8: with ~4 ticks of PI round trip, 15 / 10 replaced every chunk after ~5 steps (only steps ~4-8 ran)
    opt_args, opts = runner_options(b, backend, zero)
    dagger = bool(b.get("dagger"))          # DAgger (docs/dagger.md): the runner records policy steps + cab takeovers
    cam_url = CFG["camview_url"].rstrip("/") + "/stream/"
    # CAM_LOCK from reading the assignment until the runner is registered in RUN: cams_assign (which refuses while
    # the runner runs) can't slip in between, so the runner always gets the assignment the page shows
    with CAM_LOCK:
        bound = {n: c.id for n, c in CAMS.items() if c.id}
        labels = {n: getattr(c, "label", None) for n, c in CAMS.items() if c.id}
        missing = [n for n in need if n not in bound]
        if missing:
            raise ValueError("the dataset was trained on cameras %s but %s %s not assigned / not found: set \"Record "
                             "as\" on their tiles" % (need, missing, "is" if len(missing) == 1 else "are"))
        args = ([RUNNER["python"], "-u", "control_loop.py", "--machine", "dashboard",
                 "--dashboard-url", "http://127.0.0.1:%d" % CFG["port"]] + RUNNER_BACKENDS[backend][1] + [url,
                 "--dataset", os.path.join("datasets", ds), "--prompt", prompt, "--sync", sync]
                + (["--zero-state"] if zero else [])     # zero-state dataset: control_loop.py refuses a mismatch
                + opt_args
                + (["--record-dagger", "--episodes-dir", CFG["episodes_dir"],   # same folder as the Record tab
                    "--dagger-split-file", DAGGER_SPLIT] if dagger else [])
                + [x for n in need for x in ("--cam", "%s=%s%s" % (n, cam_url, bound[n]))])
        with RUN_LOCK:
            if RUN["proc"] is not None:
                raise RuntimeError("the runner is already running (pid %d): stop it first" % RUN["proc"].pid)
            try:                                 # a split left from an earlier run would end this run's first episode
                os.remove(os.path.join(RUNNER["dir"], DAGGER_SPLIT))
            except OSError:
                pass
            env = dict(os.environ, PYTHONPATH=RUNNER["dir"], PYTHONUNBUFFERED="1")
            proc = subprocess.Popen(args, cwd=RUNNER["dir"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    env=env, start_new_session=True)  # own process group: Ctrl-C here doesn't reach it
            RUN.update(proc=proc, since=time.time(), exit=None,
                       cfg={"backend": backend, "url": url, "prompt": prompt, "dataset": ds,
                            "cams": {n: bound[n] for n in need}, "cam_labels": {n: labels.get(n) for n in need},
                            "zero_state": zero, "sync": sync,
                            "dagger": dagger, "argv": args[1:], **opts})
            RUN["lines"].clear()
    threading.Thread(target=_runner_output, args=(proc,), daemon=True).start()
    last = {}
    try:
        with open(os.path.join(RUNNER["dir"], "last.json")) as f:
            last = json.load(f)
    except (OSError, ValueError):
        pass
    last = last if isinstance(last, dict) else {}
    last.update({"backend": backend, "gateway_url" if backend == "pi07" else "url": url, "prompt": prompt, "dataset": ds,
                 "zero_state": zero, "dagger": dagger, "opts": opts})
    try:
        with open(os.path.join(RUNNER["dir"], "last.json"), "w") as f:
            json.dump(last, f)                       # both backends' URLs are kept
    except OSError:
        pass
    return {"started": True, "pid": proc.pid}


def runner_stop(wait_s=0.0):
    """Ctrl-C the runner (it quits through its finally: dashboard STOP); escalate if it doesn't exit in 5 s."""
    with RUN_LOCK:
        proc = RUN["proc"]
    if proc is None:
        return {"stopped": False, "why": "not running"}

    def _escalate():
        for sig, t in ((signal.SIGINT, 5.0), (signal.SIGTERM, 3.0), (signal.SIGKILL, 1.0)):
            try:
                proc.send_signal(sig)
                proc.wait(t)
                return
            except subprocess.TimeoutExpired:
                continue
            except OSError:
                return
    th = threading.Thread(target=_escalate, daemon=True)
    th.start()
    if wait_s:
        th.join(wait_s)
    return {"stopped": True, "pid": proc.pid}


DAGGER_SPLIT = "dagger_next"               # in the runner dir: touching it saves the DAgger episode, next one opens


DAGGER_SCAN = {"t": 0.0, "found": []}     # runner_info is polled: scan /proc at most once a second


def dagger_runners():
    """Every control_loop.py --record-dagger process of this user, started from the panel or by hand over ssh:
    [{"pid", "split": absolute path of its --dagger-split-file (relative paths resolve against its cwd)}].
    Linux /proc only; elsewhere (and for processes we can't read) nothing is found."""
    now = time.monotonic()
    if now - DAGGER_SCAN["t"] < 1.0:
        return DAGGER_SCAN["found"]
    found = []
    for pid in (os.listdir("/proc") if os.path.isdir("/proc") else []):
        if not pid.isdigit():
            continue
        try:
            with open("/proc/%s/cmdline" % pid, "rb") as f:
                argv = [a.decode("utf-8", "replace") for a in f.read().split(b"\0") if a]
            if not any(os.path.basename(a) == "control_loop.py" for a in argv) or "--record-dagger" not in argv:
                continue
            split = "dagger_next"                                   # control_loop.py's default
            for k, a in enumerate(argv):
                if a == "--dagger-split-file" and k + 1 < len(argv):
                    split = argv[k + 1]
                elif a.startswith("--dagger-split-file="):
                    split = a.split("=", 1)[1]
            cwd = os.readlink("/proc/%s/cwd" % pid)
            found.append({"pid": int(pid), "split": os.path.join(cwd, os.path.expanduser(split))})
        except OSError:
            continue
    DAGGER_SCAN.update(t=now, found=found)
    return found


def runner_dagger_split():
    """Save the current DAgger episode and start a new one at the next engagement: touch the --dagger-split-file of
    every running DAgger runner (the panel's, and one started by hand over ssh); the runner consumes the file."""
    paths = {r["split"] for r in dagger_runners()}
    with RUN_LOCK:
        if RUN["proc"] is not None and (RUN["cfg"] or {}).get("dagger"):
            paths.add(os.path.join(RUNNER["dir"], DAGGER_SPLIT))
    if not paths:
        raise RuntimeError("no DAgger run: start a runner with --record-dagger (Record DAgger episodes on)")
    for path in sorted(paths):
        with open(path, "a"):
            pass
    return {"ok": True, "split_files": sorted(paths)}


def runner_info():
    with RUN_LOCK:
        proc, since, code, cfg = RUN["proc"], RUN["since"], RUN["exit"], RUN["cfg"]
    last = {}
    try:
        with open(os.path.join(RUNNER["dir"], "last.json")) as f:
            last = json.load(f)
    except (OSError, ValueError):
        pass
    return {"running": proc is not None, "pid": proc.pid if proc else None,
            "age": round(time.time() - since, 1) if since else None, "exit": code, "cfg": cfg,
            "lines": list(RUN["lines"])[-12:], "datasets": runner_datasets(), "last": last,
            "backends": {k: v[0] for k, v in RUNNER_BACKENDS.items()},
            "dataset_cams": {d: dataset_cams(d) for d in runner_datasets()},
            "dataset_zero_state": {d: dataset_zero_state(d) for d in runner_datasets()}, "mapping_now": cam_mapping_now(),
            "sync": CFG["sync"], "dagger_pids": [r["pid"] for r in dagger_runners()]}


def watchdog():
    """Page stopped sending (tab closed, network drop): neutral within WATCHDOG_S."""
    while RUNNING.is_set():
        time.sleep(0.02)
        with LOCK:
            if LAST["cmd"] and time.time() - LAST["t"] > WATCHDOG_S:
                _neutral_mark()
                if TELE["on"] or POL["on"]:
                    try:
                        rig.write()
                    except Exception:
                        pass


# ---------------------------------------------------------------- cameras (clients of camview.py, the Pi's camera server)
# camview.py (teammate's, port 8100) is the only process that opens cameras: V4L2 devices can't be opened twice.
# Routes used: GET /api/cameras -> [{id, label, group, error, age_s}], GET /stream/<id> -> multipart MJPEG
# (boundary camviewframe, Content-Type + Content-Length per part). The browser loads /stream/<id> straight from
# camview; this server reads only the --cam streams (to record them), over loopback. Every part carries X-Frame-Seq
# and X-Capture-Time. When a camera stalls, camview sends nothing (zero parts): the reader's socket times out after
# NO_FRAME_S and it reconnects, and the recorded frame age keeps growing. A part whose bytes equal the last frame (an
# older camview re-sent its last JPEG every 2 s on a stall) is still not counted as a new frame.
CAMVIEW = {"ok": False, "error": "not contacted yet", "t": 0.0, "list": []}


class MjpegSplitter:
    """Byte stream of concatenated JPEGs -> whole JPEGs (fallback for multipart parts without Content-Length).
    Walks the marker segments up to SOS, then scans the entropy-coded data for EOI (FF D9; FF 00 stuffing and RSTn are
    skipped), so an FF D9 inside an APP segment can't end a frame early. State is kept between feed() calls: markers
    may be split across reads. Anything between frames (multipart headers, garbage) is skipped."""

    def __init__(self):
        self.buf = bytearray()
        self.i = 0               # parse position inside the frame at buf[0:]
        self.scan = False        # inside entropy-coded data

    def _reset(self, drop):
        del self.buf[:drop]
        self.i, self.scan = 0, False

    def feed(self, data):
        self.buf += data
        out = []
        while True:
            if self.i == 0:                          # look for SOI
                k = self.buf.find(b"\xff\xd8")
                if k < 0:
                    keep = 1 if self.buf[-1:] == b"\xff" else 0
                    del self.buf[:len(self.buf) - keep]
                    return out
                del self.buf[:k]
                self.i = 2
            end = self._walk()
            if end is None:
                if len(self.buf) > MAX_JPEG:         # no end in sight: drop this SOI, resync
                    self._reset(2)
                    continue
                return out
            if end < 0:                              # malformed: drop this SOI, look for the next one
                self._reset(2)
                continue
            out.append(bytes(self.buf[:end]))
            self._reset(end)

    def _walk(self):
        """Advance self.i. Returns the frame length at EOI, None if more data is needed, -1 if malformed."""
        b, n = self.buf, len(self.buf)
        while True:
            i = self.i
            if self.scan:
                j = b.find(b"\xff", i)
                if j < 0:
                    self.i = n
                    return None
                if j + 1 >= n:
                    self.i = j
                    return None
                m = b[j + 1]
                if m == 0x00 or 0xD0 <= m <= 0xD7:
                    self.i = j + 2
                elif m == 0xFF:                      # fill byte
                    self.i = j + 1
                elif m == 0xD9:
                    return j + 2
                else:                                # a new segment (e.g. progressive / multi-scan)
                    self.scan = False
                    self.i = j
                continue
            if i + 1 >= n:
                return None
            if b[i] != 0xFF:
                return -1
            m = b[i + 1]
            if m == 0xFF:
                self.i = i + 1
                continue
            if m == 0xD9:
                return i + 2
            if 0xD0 <= m <= 0xD7 or m == 0x01:
                self.i = i + 2
                continue
            if m == 0xD8:
                return -1
            if i + 4 > n:
                return None
            ln = (b[i + 2] << 8) | b[i + 3]
            if ln < 2:
                return -1
            if i + 2 + ln > n:                       # segment continues past the data we have
                return None
            self.i = i + 2 + ln
            if m == 0xDA:
                self.scan = True


def jpeg_wh(b):
    """(width, height) from a JPEG's SOF segment, or None."""
    i, n = 2, len(b)
    while i + 9 <= n and b[i] == 0xFF:
        m = b[i + 1]
        if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            return (b[i + 7] << 8) | b[i + 8], (b[i + 5] << 8) | b[i + 6]
        if m == 0xDA:
            return None
        i += 2 + ((b[i + 2] << 8) | b[i + 3])
    return None


def camview_addr():
    u = urllib.parse.urlsplit(CFG["camview_url"])
    return u.hostname or "127.0.0.1", u.port or 80, u.path.rstrip("/")


def _match(pat, entry):
    return fnmatch.fnmatchcase(entry.get("id") or "", pat) or fnmatch.fnmatchcase(entry.get("label") or "", pat)


def camview_scanner():
    """GET /api/cameras every 2 s: which streams exist (hot-plug) -> bind each assigned name to its stream id."""
    while RUNNING.is_set():
        try:
            with urllib.request.urlopen(CFG["camview_url"].rstrip("/") + "/api/cameras", timeout=2.0) as r:
                lst = [c for c in json.loads(r.read()) if isinstance(c, dict) and c.get("id")]
            CAMVIEW.update(ok=True, error=None, list=lst, t=time.monotonic())
            for cam in list(CAMS.values()):
                cam.bind(lst)
        except Exception as ex:
            CAMVIEW.update(ok=False, error="camview not reachable at %s (%s)" % (CFG["camview_url"], ex),
                           t=time.monotonic())
        time.sleep(2.0)


class CamReader:
    """One assigned camera: a persistent reader of camview's /stream/<id> for the stream `pattern` names (exact id, or an
    fnmatch pattern on id / label). `frame` = (jpeg bytes, seq, t_monotonic, t_wall, t_capture) of the newest frame,
    replaced as a whole (the recorder reads the tuple without a lock): t_monotonic / t_wall = arrival here, t_capture =
    camview's X-Capture-Time (Pi wall clock; for the world camera the laptop's capture time, see camsync.py), or the
    arrival wall time if camview sent none. `ring` (camsync.FrameRing) keeps the last ~1.5 s of frames by t_capture
    for the recorder's --sync selection. Reconnects with backoff. stop() ends it (the camera was unassigned or moved)."""

    def __init__(self, name, pattern):
        self.name, self.pattern = name, pattern
        self.alive = True
        self.id = self.label = self.group = None
        self.upstream_error = None
        self.status, self.msg = "not_found", "waiting for camview's camera list"
        self.frame, self.seq = None, 0
        self.ring = camsync.FrameRing()
        self.w = self.h = 0
        self.cap_fps = self.jpeg_avg = 0.0
        self.restarts = 0
        self.sock = None
        if CAMVIEW["ok"]:
            self.bind(CAMVIEW["list"])
        threading.Thread(target=self._run, name="cam-" + name, daemon=True).start()

    def bind(self, lst):
        if not self.alive:
            return
        exact = [c for c in lst if c["id"] == self.pattern]
        hits = exact or sorted((c for c in lst if _match(self.pattern, c)), key=lambda c: c["id"])
        if not hits:
            if self.id is not None or self.status != "not_found":
                self.id = None
                self._set("not_found", "camview has no stream matching %r" % self.pattern)
            return
        c = hits[0]
        self.id, self.label, self.group, self.upstream_error = c["id"], c.get("label"), c.get("group"), c.get("error")
        if len(hits) > 1:
            self.msg = "%d streams match %r, using %s" % (len(hits), self.pattern, c["id"])

    def info(self):
        f = self.frame
        return {"key": self.name, "name": self.name, "id": self.id, "pattern": self.pattern, "label": self.label,
                "group": self.group, "status": self.status, "msg": self.msg, "rec": True, "w": self.w, "h": self.h,
                "cap_fps": round(self.cap_fps, 1), "jpeg_kb": round(self.jpeg_avg / 1024, 1), "n_frames": self.seq,
                "restarts": self.restarts, "upstream_error": self.upstream_error,
                "frame_age": None if f is None else round(time.monotonic() - f[2], 3),
                "cap_lag": None if f is None else round(time.time() - f[4], 3),
                "kbit_s": round(self.jpeg_avg * 8 * self.cap_fps / 1000)}

    def _set(self, status, msg=""):
        self.status, self.msg = status, msg
        if status != "ok":
            self.cap_fps = 0.0

    def _run(self):
        backoff = 1.0
        while RUNNING.is_set() and self.alive:
            cid = self.id
            if cid is None:
                time.sleep(0.25)
                continue
            t0 = time.monotonic()
            why = self._read(cid)
            if not RUNNING.is_set() or not self.alive:
                return
            if self.id != cid:                       # rebound to another stream / gone: no backoff
                continue
            self.restarts += 1
            self._set("error", why + (" (camview: %s)" % self.upstream_error if self.upstream_error else ""))
            backoff = 1.0 if time.monotonic() - t0 > STABLE_S else min(BACKOFF_MAX_S, backoff * 2)
            time.sleep(backoff)

    @staticmethod
    def _part(f):
        """Next multipart part: (JPEG bytes, X-Capture-Time or None), or None if the part has no Content-Length (caller
        falls back to the splitter, which finds this part's JPEG in the bytes that follow)."""
        while True:
            line = f.readline(4096)
            if not line:
                raise EOFError
            if line.startswith(b"--"):
                break
        hdr = {}
        while True:
            line = f.readline(4096)
            if not line:
                raise EOFError
            if not line.strip():
                break
            k, _, v = line.decode("latin-1").partition(":")
            hdr[k.strip().lower()] = v.strip()
        n = int(hdr.get("content-length", "-1"))
        if n < 0:
            return None
        if n > MAX_JPEG:
            raise ValueError("part of %d bytes" % n)
        data = f.read(n)
        if len(data) < n:
            raise EOFError
        try:
            t_cap = float(hdr["x-capture-time"])
        except (KeyError, ValueError):
            t_cap = None
        return data, t_cap

    def _read(self, cid):
        """Read /stream/<cid> until it fails or the binding changes; returns why it stopped."""
        host, port, prefix = camview_addr()
        self._set("starting", "connecting to camview /stream/%s" % cid)
        try:
            sock = socket.create_connection((host, port), timeout=NO_FRAME_S)
        except OSError as ex:
            return "cannot connect to camview at %s:%d: %s" % (host, port, ex)
        self.sock = sock
        f = None
        try:
            sock.sendall(("GET %s/stream/%s HTTP/1.0\r\nHost: %s:%d\r\n\r\n" % (prefix, cid, host, port)).encode())
            f = sock.makefile("rb")
            status = f.readline(1024).split()
            if len(status) < 2 or status[1] != b"200":
                return "camview answered %s for /stream/%s" % (b" ".join(status[1:]).decode("latin-1") or "?", cid)
            ctype = ""
            while True:
                line = f.readline(4096)
                if not line:
                    return "camview closed the connection"
                if not line.strip():
                    break
                k, _, v = line.decode("latin-1").partition(":")
                if k.strip().lower() == "content-type":
                    ctype = v.strip()
            if "multipart" not in ctype:
                return "not an MJPEG stream (%s)" % ctype
            self._set("starting", "waiting for the first frame")
            self.ring.clear()
            sp, t_win, n_win = None, time.monotonic(), 0
            while RUNNING.is_set() and self.alive and self.id == cid:
                t_cap = None
                if sp is None:
                    part = self._part(f)
                    if part is None:
                        sp = MjpegSplitter()
                        continue
                    jpeg, t_cap = part
                    frames = [jpeg] if jpeg[:2] == b"\xff\xd8" else []
                else:
                    data = f.read1(1 << 16)
                    if not data:
                        raise EOFError
                    frames = sp.feed(data)
                prev = self.frame
                if frames and prev is not None and frames[-1] == prev[0]:
                    frames = []          # same bytes as the last frame (an older camview re-sent it on a stall): not new
                if not frames:
                    if prev is not None and self.status == "ok" and time.monotonic() - prev[2] > NO_FRAME_S:
                        # parts keep arriving but all repeat the last frame (the current camview sends
                        # nothing on a stall: that ends in the socket timeout below instead)
                        self._set("error", "stalled: no new frame for %.0f s (camview re-sends the last one)"
                                  % NO_FRAME_S)
                    continue
                t, wall = time.monotonic(), time.time()
                n_win += len(frames)
                jpeg = frames[-1]                    # several at once: only the newest is kept
                self.jpeg_avg = len(jpeg) if not self.jpeg_avg else 0.95 * self.jpeg_avg + 0.05 * len(jpeg)
                wh = jpeg_wh(jpeg)
                if wh:
                    self.w, self.h = wh
                self.seq += len(frames)
                self.frame = (jpeg, self.seq, t, wall, wall if t_cap is None else min(t_cap, wall))
                self.ring.push(self.frame[4], self.frame)
                if self.status != "ok":
                    self._set("ok", "")
                if t - t_win >= 1.0:
                    self.cap_fps, t_win, n_win = n_win / (t - t_win), t, 0
            return "stopped"
        except socket.timeout:
            return "no frame from camview for %.0f s" % NO_FRAME_S
        except EOFError:
            return "camview closed the stream (camera unplugged, or camview restarted?)"
        except (OSError, ValueError) as ex:
            return "camview stream error: %s" % ex
        finally:
            self.sock = None
            for x in (f, sock):
                try:
                    if x is not None:
                        x.close()
                except OSError:
                    pass

    def stop(self):
        self.alive = False
        self.id = None
        self.close()

    def close(self):
        s = self.sock
        if s is not None:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def cams_info():
    """Tiles for the page, in camview's order: every camview stream (minus --exclude; only assigned ones with
    --no-auto), with its reader's state if it is assigned, then one tile per assigned name whose stream camview doesn't
    list any more ("assigned but missing")."""
    readers = dict(CAMS)
    by_id = {r.id: r for r in readers.values() if r.id}
    tiles = []
    for e in CAMVIEW["list"]:
        r = by_id.get(e["id"])
        if r is not None:
            tiles.append(dict(r.info(), key=e["id"], stream_id=e["id"], missing=False))
        elif not CFG["no_auto"] and not any(_match(p, e) for p in CFG["exclude"]):
            tiles.append({"key": e["id"], "stream_id": e["id"], "name": None, "id": e["id"], "label": e.get("label"),
                          "group": e.get("group"), "status": "error" if e.get("error") else "ok",
                          "msg": e.get("error") or "", "rec": False, "missing": False, "age_s": e.get("age_s")})
    for n in SLOTS:
        r = readers.get(n)
        if r is not None and not r.id:
            tiles.append(dict(r.info(), key="missing:" + n, stream_id=r.pattern, missing=True))
    host, port, prefix = camview_addr()
    return {"camview": {"url": CFG["camview_url"], "public": CFG["camview_public"], "port": port, "prefix": prefix,
                        "ok": CAMVIEW["ok"], "error": CAMVIEW["error"], "sup": sup_info()},
            "user": os.environ.get("USER", "?"), "cams": tiles, "slots": SLOTS,
            "assign": {n: r.pattern for n, r in readers.items()}, "assign_file": CFG["assign_file"],
            "locked": REC_GUARD["owner"] is not None or RUN["proc"] is not None,
            "lock_why": ("recording" if REC_GUARD["owner"] is not None
                         else "policy runner running" if RUN["proc"] is not None else None),
            "last_set": LAST_SET["set"], "last_set_episode": LAST_SET["episode"]}


# ---------------------------------------------------------------- camview supervisor (--camview auto | external | off)
# In auto mode one long-lived thread (camview_supervisor) owns camview.py as a child process: it starts it, restarts it
# when it exits or stops answering, and main()'s finally stops it. Only a camview.py of this user listening on the
# camview port is ever signalled (takeover at startup); anything else there is used as "external" and never touched.
# PR_SET_PDEATHSIG fires when the THREAD that forked the child exits, not the process: so every spawn happens on the
# supervisor thread, which lives as long as the process (never from a request thread, which ends after its request).
CAMVIEW_SUP_LOG_MAX = 5 << 20      # logs/camview.log above this at a spawn -> moved to camview.log.1 (and truncated in
                                   # place while running: the child writes with O_APPEND)
CAMVIEW_STALE_S = 10.0             # managed child's /api/cameras silent this long -> restart it
CAMVIEW_TERM_S = 6.0               # SIGTERM -> this long -> SIGKILL
CAMVIEW_BACKOFF_MAX_S = 30.0
REQ_WHY = "restart requested on the page"
CAMVIEW_HEALTHY_S = 60.0           # healthy this long -> restart backoff back to 1 s
PR_SET_PDEATHSIG = 1
_PRCTL = None
if sys.platform.startswith("linux"):   # resolved here, in the parent: the child only calls it between fork and exec
    try:
        _PRCTL = ctypes.CDLL(None, use_errno=True).prctl
        _PRCTL.argtypes = (ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong)
        _PRCTL.restype = ctypes.c_int
    except (OSError, AttributeError):
        _PRCTL = None
SUP_LOCK = threading.Lock()        # spawn vs shutdown: nothing is started after shutdown began
SUP_WAKE = threading.Event()       # restart button / shutdown: wake the supervisor now
SUP = {"mode": "off", "state": "off", "proc": None, "since": None, "spawn_t": 0.0, "ok_t": 0.0, "healthy_t": None,
       "restarts": 0, "last_error": None, "note": None, "replaced": [], "external": None, "busy": None,
       "restart_at": None, "fail_t": None, "backoff": 1.0, "restart_req": False, "shutdown": False,
       "python": None, "script": os.path.join(HERE, "camview.py"), "log": os.path.join(HERE, "logs", "camview.log")}


def _sup_log(text):
    """One line into logs/camview.log (prefixed, next to camview's own output) and on the dashboard's stdout."""
    line = "[dashboard %s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), text)
    print("camview supervisor: " + text, flush=True)
    try:
        os.makedirs(os.path.dirname(SUP["log"]), exist_ok=True)
        with open(SUP["log"], "ab") as f:
            f.write((line + "\n").encode("utf-8", "replace"))
    except OSError:
        pass


def _sup_error(text):
    SUP["last_error"] = "%s %s" % (time.strftime("%H:%M:%S"), text)
    _sup_log(text)


def log_tail(path, n=20, max_bytes=16384):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read()
    except OSError:
        return []
    lines = data.decode("utf-8", "replace").replace("\r", "\n").split("\n")
    if size > max_bytes:
        lines = lines[1:]                        # first one is probably cut
    return [ln[:500] for ln in lines if ln.strip()][-n:]


def camview_port():
    return camview_addr()[:2]


def port_in_use(host, port):
    """Something accepts TCP connections on host:port (camview's port)."""
    try:
        socket.create_connection((host, port), timeout=0.5).close()
        return True
    except OSError:
        return False


def camview_probe():
    """camview's /api/cameras answers with a JSON list."""
    try:
        with urllib.request.urlopen(CFG["camview_url"].rstrip("/") + "/api/cameras", timeout=2.0) as r:
            return isinstance(json.loads(r.read()), list)
    except Exception:
        return False


# Process identification, pluggable per platform: /proc on Linux (the Pi); lsof + ps elsewhere (macOS, tests only).
def _listeners_proc(port):
    inodes = set()
    for fn in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(fn) as f:
                next(f, None)
                for line in f:
                    p = line.split()
                    if len(p) > 9 and p[3] == "0A" and int(p[1].rsplit(":", 1)[1], 16) == port:
                        inodes.add("socket:[%s]" % p[9])
        except (OSError, ValueError):
            continue
    pids = set()
    if not inodes:
        return pids
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            for fd in os.listdir("/proc/%s/fd" % d):        # only this user's processes are readable (and root's not)
                try:
                    if os.readlink("/proc/%s/fd/%s" % (d, fd)) in inodes:
                        pids.add(int(d))
                        break
                except OSError:
                    continue
        except OSError:
            continue
    return pids


def _listeners_lsof(port):
    lsof = shutil.which("lsof") or ("/usr/sbin/lsof" if os.path.exists("/usr/sbin/lsof") else None)
    if lsof is None:
        return set()
    try:
        out = subprocess.run([lsof, "-nP", "-t", "-iTCP:%d" % port, "-sTCP:LISTEN"], capture_output=True, text=True,
                             timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    return {int(x) for x in out.split() if x.isdigit()}


def _ident_proc(pid):
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as f:
            argv = [a.decode("utf-8", "replace") for a in f.read().split(b"\0") if a]
        return os.stat("/proc/%d" % pid).st_uid, argv
    except OSError:
        return None


def _ident_ps(pid):
    try:
        out = subprocess.run(["ps", "-o", "uid=", "-o", "command=", "-p", str(pid)], capture_output=True, text=True,
                             timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    uid, _, cmd = out.partition(" ")
    if not uid.strip().isdigit():
        return None
    return int(uid), cmd.split()


HAVE_PROC = os.path.isdir("/proc/self/fd") and os.path.exists("/proc/net/tcp")
PROC_ID = {"listeners": _listeners_proc if HAVE_PROC else _listeners_lsof,
           "ident": _ident_proc if HAVE_PROC else _ident_ps}


def is_user_camview(ident):
    """(uid, argv) is a camview.py run by this user (argv of a zombie is empty: not a camview any more)."""
    return (ident is not None and ident[0] == os.getuid()
            and any(os.path.basename(a) == "camview.py" for a in ident[1]))


def camview_busy():
    """Why camview must not be restarted now (None = it may)."""
    if REC_GUARD["owner"] is not None or REC["id"] is not None:
        return "an episode is recording"
    with RUN_LOCK:
        running = RUN["proc"] is not None
    if running:
        return "the policy runner is running"
    if POL["on"]:
        return "Inference is on"
    return None


def _stop_proc(proc, why):
    """SIGTERM our managed child, SIGKILL after CAMVIEW_TERM_S. Returns its exit code."""
    if proc.poll() is None:
        _sup_log("stopping camview pid %d (%s): SIGTERM" % (proc.pid, why))
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(CAMVIEW_TERM_S)
        except subprocess.TimeoutExpired:
            _sup_log("camview pid %d still running %g s after SIGTERM: SIGKILL" % (proc.pid, CAMVIEW_TERM_S))
            proc.kill()
            try:
                proc.wait(2.0)
            except subprocess.TimeoutExpired:
                pass
        except OSError:
            pass
    return proc.poll()


def _take_over(pid, host, port):
    """A camview.py of this user (not our child) holds the port: SIGTERM, wait <= CAMVIEW_TERM_S for it to exit and the
    port to be free, then SIGKILL (only if it is still that camview). True when the port is free."""
    _sup_log("port %d is held by a camview.py of this user started outside the dashboard (pid %d): taking it over "
             "(SIGTERM)" % (port, pid))
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as ex:
        _sup_error("can't signal camview pid %d: %s" % (pid, ex))
        return False
    t_end = time.monotonic() + CAMVIEW_TERM_S
    while time.monotonic() < t_end:
        if not is_user_camview(PROC_ID["ident"](pid)) and not port_in_use(host, port):
            SUP["replaced"].append(pid)
            _sup_log("camview pid %d exited, port %d free" % (pid, port))
            return True
        time.sleep(0.1)
    if is_user_camview(PROC_ID["ident"](pid)):  # same pid still that camview (not a reused pid)
        _sup_log("camview pid %d still running %g s after SIGTERM: SIGKILL" % (pid, CAMVIEW_TERM_S))
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    t_end = time.monotonic() + 3.0
    while time.monotonic() < t_end:
        if not port_in_use(host, port):
            SUP["replaced"].append(pid)
            return True
        time.sleep(0.1)
    _sup_error("port %d still in use after taking over camview pid %d" % (port, pid))
    return False


def _spawn():
    """Start camview.py as our child. Called on the supervisor thread only (PR_SET_PDEATHSIG, see above)."""
    host, port = camview_port()
    script = SUP["script"]
    py = SUP["python"]
    args = [py, "-u", os.path.basename(script), "--port", str(port)]
    log = SUP["log"]
    try:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        if os.path.getsize(log) > CAMVIEW_SUP_LOG_MAX:
            os.replace(log, log + ".1")
    except OSError:
        pass
    parent = os.getpid()
    sig = int(signal.SIGTERM)
    prctl = _PRCTL

    def _preexec():                             # in the child, between fork and exec: nothing but the prctl call
        prctl(PR_SET_PDEATHSIG, sig, 0, 0, 0)
        if os.getppid() != parent:              # the dashboard died before the prctl: nobody will signal us
            os.kill(os.getpid(), sig)
    with SUP_LOCK:
        if SUP["shutdown"]:
            return None
        _sup_log("starting camview: %s (cwd %s)" % (" ".join(args), os.path.dirname(script)))
        try:
            with open(log, "ab") as lf:
                proc = subprocess.Popen(args, cwd=os.path.dirname(script), stdin=subprocess.DEVNULL, stdout=lf,
                                        stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUNBUFFERED="1"),
                                        preexec_fn=_preexec if prctl is not None else None,
                                        start_new_session=False)   # same process group: Ctrl-C reaches it too
        except OSError as ex:
            SUP["state"] = "error"
            _sup_error("can't start camview (%s): %s" % (" ".join(args), ex))
            return None
        now = time.monotonic()
        SUP.update(proc=proc, since=time.time(), spawn_t=now, ok_t=0.0, healthy_t=None, state="starting",
                   external=None)
    _sup_log("camview started: pid %d" % proc.pid)
    return proc


def _claim_and_start(why):
    """Make the camview port ours and start camview on it. Port held by a camview.py of this user -> take it over;
    held by anything else -> use it as external (never signalled). Returns True when a child was started."""
    host, port = camview_port()
    listeners = sorted(PROC_ID["listeners"](port))
    if listeners or port_in_use(host, port):
        ids = {pid: PROC_ID["ident"](pid) for pid in listeners}
        ours = [pid for pid in listeners if is_user_camview(ids[pid])]
        if not listeners or len(ours) != len(listeners):
            other = [pid for pid in listeners if pid not in ours]
            desc = ", ".join("pid %d (%s)" % (p, " ".join(ids[p][1])[:200] if ids[p] else "unknown")
                             for p in other) or "a process this user can't identify"
            if SUP["state"] != "external" or SUP["external"] != desc:
                _sup_log("port %d is held by %s, not a camview.py of this user: using it as external camview "
                         "(not managed, never signalled)" % (port, desc))
            SUP.update(state="external", external=desc, proc=None, since=None)
            return False
        for pid in ours:
            if not _take_over(pid, host, port):
                SUP["state"] = "error"
                return False
    if _spawn() is None:
        return False
    if why != "startup":
        SUP["restarts"] += 1
    return True


def camview_supervisor():
    """--camview auto: the long-lived thread that owns camview (see the section comment)."""
    _claim_and_start("startup")
    while not SUP["shutdown"]:
        ra = SUP["restart_at"]                           # a pending backoff shortens the wait (1 s backoff = 1 s)
        SUP_WAKE.wait(2.0 if ra is None else max(0.05, min(2.0, ra - time.monotonic())))
        SUP_WAKE.clear()
        if SUP["shutdown"]:
            break
        now = time.monotonic()
        proc = SUP["proc"]
        req, SUP["restart_req"] = SUP["restart_req"], False
        if proc is not None:
            ok = camview_probe()
            if ok:
                SUP["ok_t"] = now
                if SUP["healthy_t"] is None:
                    SUP["healthy_t"] = now
                if SUP["state"] == "starting":
                    SUP["state"] = "managed"
                    _sup_log("camview pid %d answers on /api/cameras" % proc.pid)
                if now - SUP["healthy_t"] >= CAMVIEW_HEALTHY_S:
                    SUP["backoff"] = 1.0
            else:
                SUP["healthy_t"] = None
            code = proc.poll()
            if code is not None:
                why = "camview exited with code %s" % code
            elif now - max(SUP["ok_t"], SUP["spawn_t"]) > CAMVIEW_STALE_S:
                why = "camview /api/cameras has not answered for %.0f s" % (now - max(SUP["ok_t"], SUP["spawn_t"]))
            elif req:
                why = REQ_WHY
            else:
                SUP.update(busy=None, restart_at=None, fail_t=None)
                continue
        elif SUP["state"] == "external":
            host, port = camview_port()
            if port_in_use(host, port):                 # external camview still there: leave it alone
                continue
            why, code = "external camview on port %d went away" % port, None
        else:
            why, code = "camview is not running", None
        if SUP["fail_t"] is None:                       # a new failure: log it once, start its backoff clock
            SUP["fail_t"] = now
            tail = log_tail(SUP["log"], 3) if code is not None else []
            (_sup_log if why == REQ_WHY else _sup_error)(why + (": " + tail[-1] if tail else ""))
        if code is not None:
            SUP["state"] = "exited"
        busy = camview_busy()
        if busy:
            if SUP["busy"] is None:
                _sup_log("%s, but %s: not restarting camview until that ends" % (why, busy))
            SUP["busy"] = "%s, but %s: camview is restarted when that ends" % (why, busy)
            continue
        SUP["busy"] = None
        if not req:
            if SUP["restart_at"] is None:               # backoff counted from the failure (not from the end of a
                SUP["restart_at"] = SUP["fail_t"] + SUP["backoff"]    # recording that held the restart back)
                SUP["backoff"] = min(CAMVIEW_BACKOFF_MAX_S, SUP["backoff"] * 2)
            if now < SUP["restart_at"]:
                continue
        SUP.update(restart_at=None, fail_t=None, state="restarting")   # not "managed" while the old one stops
        if proc is not None:
            _stop_proc(proc, why)
            SUP.update(proc=None)
        _claim_and_start(why)
    while SUP["proc"] is not None and SUP["proc"].poll() is None:   # camview_shutdown() is stopping it: this thread
        time.sleep(0.2)                                             # ending now would SIGTERM it early (pdeathsig)


def camview_restart(b):
    """POST /api/camview/restart: the supervisor restarts its camview now. Refused (409) while recording / runner /
    Policy, and when camview isn't ours to restart."""
    if SUP["mode"] != "auto" or SUP["state"] == "external":
        raise PermissionError("camview is not managed by this dashboard (%s): restart it where it runs"
                              % ("--camview " + SUP["mode"] if SUP["mode"] != "auto" else "external"))
    busy = camview_busy()
    if busy:
        raise PermissionError("%s: camview can't be restarted now (every camera stream would drop)" % busy)
    SUP["restart_req"] = True
    SUP_WAKE.set()
    return {"restarting": True, "pid": SUP["proc"].pid if SUP["proc"] is not None else None}


def camview_shutdown():
    """main()'s finally: stop our managed camview (SIGTERM, SIGKILL after CAMVIEW_TERM_S). Never touches an external
    one. Nothing is started after this (SUP_LOCK)."""
    with SUP_LOCK:
        SUP["shutdown"] = True
        proc = SUP["proc"]
    SUP_WAKE.set()
    if SUP["mode"] != "auto" or proc is None:
        return None
    code = _stop_proc(proc, "dashboard quitting")
    SUP.update(state="stopped", proc=None)
    _sup_log("camview pid %d stopped (exit code %s)" % (proc.pid, code))
    return code


def sup_info(tail=False):
    proc = SUP["proc"]
    now = time.monotonic()
    out = {"mode": SUP["mode"], "state": SUP["state"], "pid": proc.pid if proc is not None else None,
           "uptime": round(time.time() - SUP["since"]) if proc is not None and proc.poll() is None and SUP["since"]
           else None,
           "restarts": SUP["restarts"], "last_error": SUP["last_error"], "note": SUP["note"],
           "busy": SUP["busy"], "external": SUP["external"], "replaced": SUP["replaced"][-5:],
           "restart_in": round(max(0.0, SUP["restart_at"] - now), 1) if SUP["restart_at"] else None,
           "python": SUP["python"], "script": SUP["script"], "log": SUP["log"],
           "api_ok": CAMVIEW["ok"], "can_restart": SUP["mode"] == "auto" and SUP["state"] != "external"
           and camview_busy() is None}
    if tail:
        out["tail"] = log_tail(SUP["log"], 20)
    return out


# ---------------------------------------------------------------- camera assignment ("Record as" on a tile)
def load_assign(path):
    """cam_assign.json -> {name: camview stream id (or --cam pattern)}. Raises OSError / ValueError."""
    with open(path) as f:
        obj = json.load(f)
    a = obj.get("assign") if isinstance(obj, dict) else None
    if not isinstance(a, dict):
        raise ValueError("no \"assign\" object")
    out = {}
    for n in SLOTS:                              # a stream under two names: the first slot keeps it
        v = a.get(n)
        if isinstance(v, str) and v and v not in out.values():
            out[n] = v
    return out


def save_assign(assign):
    """Atomic write of --assign-file (tmp + fsync + rename): a crash leaves the old file or the new one."""
    path = CFG["assign_file"]
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = "%s.tmp%d" % (path, os.getpid())
    with open(tmp, "w") as f:
        json.dump({"format": "act-athon-cam-assign/1", "assign": assign, "slots": SLOTS,
                   "saved": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def apply_assign(assign):
    """Make CAMS match {name: stream id}: stop the readers of removed / moved names, start the new ones. Under CAM_LOCK
    (or at startup), never while recording."""
    for n in list(CAMS):
        if assign.get(n) != CAMS[n].pattern:
            CAMS.pop(n).stop()
    for n in SLOTS:
        if n in assign and n not in CAMS:
            CAMS[n] = CamReader(n, assign[n])


def cams_assign(b):
    """POST /api/cams/assign {stream_id, name | null}: record this stream as `name` (off with null). A name is on one
    stream only: giving it to another stream takes it off the old one (reported in `changes`)."""
    sid, name = b.get("stream_id"), b.get("name")
    if not isinstance(sid, str) or not sid:
        raise ValueError("stream_id missing")
    if name is not None and name not in SLOTS:
        raise ValueError("name must be one of %s, or null (off)" % ", ".join(SLOTS))
    with CAM_LOCK:
        if REC_GUARD["owner"] is not None:
            raise PermissionError("an episode is being recorded: camera assignments are locked until it is saved or "
                                  "discarded")
        with RUN_LOCK:                           # CAM_LOCK -> RUN_LOCK, the same order as runner_start
            running = RUN["proc"] is not None
        if running:
            raise PermissionError("the policy runner is running: camera assignments are locked until you stop it")
        old = {n: r.pattern for n, r in CAMS.items()}
        bound = {n: r.id for n, r in CAMS.items()}
        new = {n: p for n, p in old.items() if n != name and p != sid and bound.get(n) != sid}
        if name is not None:
            if not any(e["id"] == sid for e in CAMVIEW["list"]):
                raise ValueError("camview has no stream %r (unplugged?)" % sid)
            new[name] = sid
        changes = [{"name": n, "was": old.get(n), "now": new.get(n)} for n in SLOTS if old.get(n) != new.get(n)]
        if changes:
            save_assign(new)                     # file first: if it can't be written nothing changes
            apply_assign(new)
    return {"changes": changes, "assign": new}


# ---------------------------------------------------------------- camera-set guard (one dataset = one camera set)
LAST_SET = {"set": None, "episode": None}    # {name: [camview id, width, height]} of the newest saved cab episode


class CamSetChanged(Exception):
    """Start with a camera set other than the last saved episode's, without confirm_new_camera_set (-> 409)."""

    def __init__(self, was, now, episode):
        super().__init__("Camera set changed (was: %s, now: %s). Episodes with different camera sets must go into "
                         "separate datasets. Start anyway?" % (fmt_set(was), fmt_set(now)))
        self.info = {"was": was, "now": now, "episode": episode}


def fmt_set(cs):
    return ", ".join("%s=%s %sx%s" % (n, v[0], v[1], v[2]) for n, v in sorted(cs.items())) or "none"


def meta_set(cameras):
    return {n: [v.get("camview_id"), v.get("width"), v.get("height")] for n, v in cameras.items()}


def load_last_set():
    """Newest saved cab episode in --episodes-dir (folder names start with the start time) -> LAST_SET."""
    for p in sorted(glob.glob(os.path.join(CFG["episodes_dir"], "*", "meta.json")), reverse=True):
        if ".partial" in p:
            continue
        try:
            with open(p) as f:
                meta = json.load(f)
        except (OSError, ValueError):
            continue
        if meta.get("action_source") == "cab" and isinstance(meta.get("cameras"), dict):
            LAST_SET.update(set=meta_set(meta["cameras"]), episode=os.path.basename(os.path.dirname(p)))
            return


# ---------------------------------------------------------------- recording
# REC is guarded by REC_LOCK (never held during disk I/O or gate calls). The tick thread builds each step in memory and
# hands it to the writer thread through WQ; the writer does every file operation, in order.
REC_LOCK = threading.Lock()
REC = {"id": None}
REC_LAST = {}                               # result of the last finished episode (for the page)
RECENT = []                                 # recent task prompts, newest first
WQ = queue.Queue(maxsize=450)               # ~30 s of steps at 15 fps before steps are dropped
WRITER = {"err": None}
MIN_FREE_MB = 300


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40] or "episode"


def load_recent():
    metas = []
    for p in glob.glob(os.path.join(CFG["episodes_dir"], "*", "meta.json")):
        if ".partial" in p:
            continue
        try:
            with open(p) as f:
                metas.append((os.path.getmtime(p), json.load(f).get("task", "")))
        except (OSError, ValueError):
            pass
    for _, t in sorted(metas, reverse=True):
        if t and t not in RECENT:
            RECENT.append(t)
    del RECENT[8:]


# ---------------------------------------------------------------- episode QC (episode_qc.py; read only, never the gate)
# GET /api/episodes/qc?name=<saved episode folder> or ?recent=N. A name is accepted only if it is an entry of the
# episodes folder's own listing (saved: not .partial, has meta.json); paths are built from that listing entry, never
# from the request. Reports are cached by (name, mtimes of the episode and of its predecessor). QC_LOCK only guards
# the cache and one computation at a time (~40 ms per episode); it is never taken together with LOCK or REC_LOCK.
QC_LOCK = threading.Lock()
QC_CACHE = {}
QC_RECENT_MAX = 20


def qc_saved_names():
    """Saved episode folder names in --episodes-dir, oldest first (names start with the start time)."""
    base = CFG["episodes_dir"]
    try:
        entries = os.listdir(base)
    except OSError:
        return []
    return sorted(n for n in entries if not n.endswith(".partial") and not n.startswith(".")
                  and os.path.isfile(os.path.join(base, n, "meta.json")))


def _qc_mtimes(d):
    out = []
    for p in (d, os.path.join(d, "meta.json"), os.path.join(d, "steps.jsonl")):
        try:
            out.append(os.stat(p).st_mtime_ns)
        except OSError:
            out.append(None)
    return tuple(out)


def qc_report(names, k):
    """QC report (+ one line per item) for names[k]; the previous saved episode is names[k - 1]."""
    base = CFG["episodes_dir"]
    path = os.path.join(base, names[k])
    prev = os.path.join(base, names[k - 1]) if k > 0 else None
    key = (_qc_mtimes(path), names[k - 1] if prev else None, _qc_mtimes(prev) if prev else None)
    with QC_LOCK:
        hit = QC_CACHE.get(names[k])
        if hit is not None and hit[0] == key:
            return hit[1]
        rep = episode_qc.qc_episode(path, prev)
        rep["lines"] = episode_qc.item_lines(rep)
        QC_CACHE[names[k]] = (key, rep)
        return rep


def qc_api(query):
    """-> (http code, json-able). query: urllib.parse.parse_qs of the request's query string."""
    if episode_qc is None:
        return 503, {"error": EPISODE_QC_ERROR}
    names = qc_saved_names()
    with QC_LOCK:                                # forget deleted / renamed episodes
        for gone in set(QC_CACHE) - set(names):
            del QC_CACHE[gone]
    try:
        if "name" in query:
            name = query["name"][0]
            if name not in names:                # also rejects "..", paths, .partial and not-yet-renamed episodes
                return 404, {"error": "no saved episode %r in the episodes folder" % name[:120]}
            return 200, qc_report(names, names.index(name))
        if "recent" in query:
            try:
                n = int(query["recent"][0])
            except ValueError:
                return 400, {"error": "recent must be a number"}
            n = max(0, min(n, QC_RECENT_MAX))
            return 200, [qc_report(names, k) for k in range(len(names) - 1, max(-1, len(names) - 1 - n), -1)]
    except Exception as e:                       # a QC bug must not take the request thread's answer with it
        return 500, {"error": "QC failed: %s: %s" % (type(e).__name__, e)}
    return 400, {"error": "give name=<episode> or recent=N (N <= %d)" % QC_RECENT_MAX}


def rec_info():
    now = time.monotonic()
    with REC_LOCK:
        r = dict(REC)
    info = {"active": r["id"] is not None, "fps": CFG["rec_fps"], "episode_s": CFG["episode_s"], "source": "cab",
            "recent": RECENT[:8], "last": REC_LAST or None, "writer_err": WRITER["err"], "backlog": WQ.qsize(),
            "qc": episode_qc is not None}
    try:
        info["free_mb"] = shutil.disk_usage(CFG["episodes_dir"] if os.path.isdir(CFG["episodes_dir"])
                                            else os.path.expanduser("~")).free // (1 << 20)
    except OSError:
        info["free_mb"] = None
    if r["id"] is not None:
        info.update(id=r["id"], task=r["task"], name=r["name"], n=r["n"],
                    elapsed=round(now - r["t0"], 2) if r["t0"] else 0.0,
                    missing=r["missing"], late=r["late"], dropped=r["dropped"], state_missing=r["state_missing"],
                    track_steps=r["track_steps"], tracks_now=r["tracks_now"], sync_last=r["sync_last"])
    info["sync"] = CFG["sync"]
    return info


def rec_start(b):
    task = " ".join(str(b.get("task") or "").split())[:200]
    if not task:
        raise ValueError("type a task prompt first")
    g = STATUS["snap"].get("gate")
    if g is None:
        raise RuntimeError("gate /state not readable (%s): the cab input can't be recorded" % STATUS["snap"].get("error"))
    missing = sorted({CFG["cab_map"][ch][0] for ch in CHANNELS} - set(g["controls"]))
    if missing:
        raise RuntimeError("the cab map reads %s, which the gate's /state controls doesn't have (it has: %s); fix it "
                           "with --cab-map" % (missing, ", ".join(sorted(g["controls"]))))
    base = CFG["episodes_dir"]
    os.makedirs(base, exist_ok=True)
    if shutil.disk_usage(base).free < MIN_FREE_MB << 20:
        raise RuntimeError("less than %d MB free in %s" % (MIN_FREE_MB, base))
    with REC_LOCK:
        if REC["id"] is not None:
            raise RuntimeError("already recording %s" % REC["name"])
    token = uuid.uuid4().hex[:10]
    with CAM_LOCK:                               # from here until the episode ends the assignments can't change
        with LOCK:                               # ... and nothing is written to the gate
            if TELE["on"] or POL["on"]:
                raise RuntimeError("%s is on: switch it off first (a cab recording sends nothing to the gate)"
                                   % ("Teleoperate" if TELE["on"] else "Policy"))
            if REC_GUARD["owner"] is not None:
                raise RuntimeError("another page is starting a recording")
            REC_GUARD["owner"] = token
        readers = dict(CAMS)
    try:
        if not readers:
            ids = ", ".join(e["id"] for e in CAMVIEW["list"]) or "none listed"
            raise RuntimeError("no camera assigned: set \"Record as\" (%s) on at least one camera tile (the name "
                               "becomes the dataset key). camview streams: %s" % (" / ".join(SLOTS), ids))
        for c in readers.values():
            f = c.frame
            if c.id is None:
                raise RuntimeError("camera %s is assigned to %s, which camview doesn't list (unplugged?): plug it "
                                   "back in or set that tile to off" % (c.name, c.pattern))
            if c.status != "ok" or f is None or time.monotonic() - f[2] > 1.0:
                raise RuntimeError("camera %s (%s) is not delivering frames: %s %s" % (c.name, c.id, c.status, c.msg))
        now_set = {n: [c.id, c.w, c.h] for n, c in readers.items()}
        was = LAST_SET["set"]
        changed = was is not None and was != now_set
        if changed and b.get("confirm_new_camera_set") is not True:
            raise CamSetChanged(was, now_set, LAST_SET["episode"])
        return _rec_open(task, [readers[n] for n in SLOTS if n in readers], token, now_set, was if changed else None)
    except BaseException:
        if REC_GUARD["owner"] == token:
            REC_GUARD["owner"] = None
        raise


def _rec_open(task, cams, token, cam_set, prev_set):
    base = CFG["episodes_dir"]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name, k = "%s_%s" % (stamp, slug(task)), 1
    while os.path.exists(os.path.join(base, name)) or os.path.exists(os.path.join(base, name + ".partial")):
        k += 1
        name = "%s_%s_%d" % (stamp, slug(task), k)
    path = os.path.join(base, name + ".partial")
    os.makedirs(path)
    for c in cams:
        os.makedirs(os.path.join(path, c.name))
    g = STATUS["snap"].get("gate") or {}
    meta = {"format": "act-athon-pi-episode/1", "task": task, "fps": CFG["rec_fps"],
            "episode_s": CFG["episode_s"], "channels": list(CHANNELS), "signs": SIGNS,
            "action_source": "cab",
            "action": "cab joysticks: cab_action(state.controls, cab_map, cab_deadband) at each tick, -1..1 rig "
                      "signs; this dashboard sent nothing to the gate during the episode",
            "cab_map": CFG["cab_map"], "cab_deadband": CFG["cab_deadband"], "cab_map_file": CFG["cab_map_file"],
            "cab_map_status": CFG["cab_status"],
            "state": "gate /state: controls (raw, all fields), sent (= gate output), last_cmd, armed, veto, "
                     "throttle, rabbit_button, age",
            "cameras": {c.name: {"camview_id": c.id, "cam_arg": c.pattern, "label": c.label, "group": c.group,
                                 "width": c.w, "height": c.h, "fps_at_start": round(c.cap_fps, 1),
                                 "slot": SLOTS[c.name]}
                        for c in cams},
            "camera_slots": SLOTS, "camera_set": cam_set,
            "sync": {"mode": CFG["sync"], "max_lag_s": camsync.MAX_LAG_S,
                     "rule": "camsync.select: aligned = every camera's frame captured nearest t_ref, the oldest of "
                             "the cameras' newest capture times; latest = every camera's newest frame",
                     "clock": "cams.<name>.cap = camview X-Capture-Time on the Pi wall clock (network cameras: the "
                              "sender's capture time mapped by camview's ClockSync); run the policy with the same "
                              "--sync"},
            "camera_set_changed_from": prev_set,     # set when Start was confirmed with a camera set other than the
                                                     # last saved episode's (a new dataset starts here)
            "camview_url": CFG["camview_url"],
            "throttle_at_start": g.get("throttle"), "rabbit_at_start": g.get("rabbit"),
            "armed_at_start": g.get("live"), "dashboard_version": VERSION, "dashboard_sha1": SOURCE_SHA1,
            "host": socket.gethostname(), "started_wall": time.time(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    with REC_LOCK:
        busy = REC["id"] is not None             # another page started one meanwhile
        if not busy:
            REC.clear()
            REC.update(id=token, name=name, path=path, task=task, meta=meta,
                       cams=[c.name for c in cams], readers={c.name: c for c in cams}, n=0, t0=0.0, missing={c.name: 0 for c in cams}, late=0,
                       repeated={c.name: 0 for c in cams}, last_seq={c.name: None for c in cams},
                       dropped=0, state_missing=0, track_steps=0, tracks_now=[], sync_last=None, skews=[])
            if task in RECENT:
                RECENT.remove(task)
            RECENT.insert(0, task)
            del RECENT[8:]
    if busy:
        shutil.rmtree(path, ignore_errors=True)
        raise RuntimeError("another page started recording")
    return {"rec": rec_info()}


def _rec_end(how, reason):
    """Under REC_LOCK: end the current episode. Returns (done event, writer item) for the caller to queue."""
    r = dict(REC)
    REC.clear()
    REC["id"] = None
    if REC_GUARD["owner"] == r["id"]:            # plain assignment, no LOCK needed: it only re-allows Teleoperate
        REC_GUARD["owner"] = None
    meta = dict(r["meta"])
    meta.update(n_steps=r["n"], duration_s=round(r["n"] / CFG["rec_fps"], 3), late_ticks=r["late"],
                dropped_steps=r["dropped"], frames_missing=r["missing"], state_missing=r["state_missing"],
                frames_repeated=r["repeated"], sync_skew_ms=_skew_summary(r["skews"]),
                track_steps=r["track_steps"], track_warn=TRACK_WARN, end=reason, ended_wall=time.time())
    final = os.path.join(os.path.dirname(r["path"]), r["name"])
    done = threading.Event()
    summary = {"name": r["name"], "task": r["task"], "n": r["n"], "how": how, "reason": reason,
               "track_steps": r["track_steps"],
               "dir": final if how == "save" else r["path"] if how == "abort" else None}
    REC_LAST.clear()
    REC_LAST.update(summary)
    if how == "save":                            # the camera-set guard compares the next Start with this episode
        LAST_SET.update(set=meta["camera_set"], episode=r["name"])
    return done, ("close", how, r["path"], final, meta, done)


def rec_end(b, how):
    with REC_LOCK:
        if REC["id"] is None or b.get("id") != REC["id"]:
            raise PermissionError("not recording (or not this episode: another page ended it)")
        if how == "save" and REC["n"] == 0:
            how = "discard"
        done, item = _rec_end(how, "page")
    WQ.put(item)                                 # blocking: the writer is draining the queue ahead of it
    done.wait(30.0)
    return {"rec": rec_info(), "done": done.is_set()}


def rec_discard_on_stop():
    """STOP / Esc / X on the page during a recording = Discard (only for the page's stop: /api/stop discard_rec).
    Called after the gate stop, so it never delays it; doesn't wait for the writer (the page sees it in rec_info)."""
    with REC_LOCK:
        if REC["id"] is None:
            return False
        _, item = _rec_end("discard", "STOP")
    WQ.put(item)
    return True


def _skew_summary(skews):
    if not skews:
        return None
    xs = sorted(skews)
    return {"p50": round(xs[len(xs) // 2] * 1e3, 1), "p95": round(xs[min(len(xs) - 1, int(0.95 * len(xs)))] * 1e3, 1),
            "max": round(xs[-1] * 1e3, 1), "n": len(xs)}


def _step(i, now):
    """Under REC_LOCK: one step (json-able dict) + the frame files to write. Frames are picked by camsync.select
    (--sync): step.sync = {t_ref, anchor, skew} and every camera's `cap` (capture, Pi wall clock) / `lag` (cap -> this
    step); `age` stays arrival-based (now - arrival here) as before."""
    snap = STATUS["snap"]
    g = snap.get("gate")
    state = {"ok": g is not None, "poll_age": round(now - snap["t"], 4) if snap.get("t") else None,
             "error": snap.get("error")}
    if g is not None:
        state.update(controls=g["controls"], sent=g["sent"], last_cmd=g["last_cmd"], armed=g["live"],
                     veto=g["veto"], throttle=g["throttle"], rabbit_button=g["rabbit"], age=g["age"])
        action = cab_action(g["controls"], REC["meta"]["cab_map"], REC["meta"]["cab_deadband"])
        REC["tracks_now"] = tracks_moving(g["controls"])
        if REC["tracks_now"]:
            REC["track_steps"] += 1
    else:
        REC["state_missing"] += 1
        action = {k: 0.0 for k in CHANNELS}
    wall = time.time()
    live = {n: c.ring for n, c in REC["readers"].items() if c.status == "ok"}
    chosen, sel = camsync.select(live, CFG["sync"], camsync.MAX_LAG_S, wall)
    REC["sync_last"] = {"anchor": sel["anchor"], "skew_ms": round(sel["skew"] * 1e3, 1), "left_out": sel["left_out"]}
    if len(live) >= 2 and not sel["left_out"]:
        REC["skews"].append(sel["skew"])
    step = {"i": i, "t": round(now - REC["t0"], 4), "mono": round(now, 4), "wall": round(wall, 4),
            "teleop": TELE["on"], "action": action, "state": state, "cams": {},
            "sync": {"t_ref": None if sel["t_ref"] is None else round(sel["t_ref"], 4), "anchor": sel["anchor"],
                     "skew": round(sel["skew"], 4), "left_out": sel["left_out"]}}
    files = []
    for name in REC["cams"]:
        cam = REC["readers"].get(name)
        f = chosen[name][1] if chosen.get(name) is not None else None
        if f is None or cam.status != "ok":
            REC["missing"][name] += 1
            step["cams"][name] = {"file": None, "status": None if cam is None else cam.status}
            continue
        # camera rate ~= record rate, so tick jitter makes some steps reuse the previous step's frame: counted
        # (meta frames_repeated, converter output), not an error. A stalled camera shows up as a growing `age`.
        if f[1] == REC["last_seq"][name]:
            REC["repeated"][name] += 1
        REC["last_seq"][name] = f[1]
        rel = "%s/%06d.jpg" % (name, i)
        step["cams"][name] = {"file": rel, "seq": f[1], "t": round(f[2] - REC["t0"], 4), "mono": round(f[2], 4),
                              "wall": round(f[3], 4), "age": round(now - f[2], 4), "cap": round(f[4], 4),
                              "lag": round(wall - f[4], 4), "bytes": len(f[0])}
        files.append((rel, f[0]))
    return step, files


def recorder():
    """REC tick on the Pi clock: one step per 1/--rec-fps while an episode is open. Late ticks are skipped (counted),
    never bunched up."""
    period = 1.0 / CFG["rec_fps"]
    t_next = time.monotonic()
    while RUNNING.is_set():
        now = time.monotonic()
        if now < t_next:
            time.sleep(min(t_next - now, 0.05))
            continue
        late = now - t_next > 0.5 * period
        t_next = (now if late else t_next) + period
        item = None
        with REC_LOCK:
            if REC["id"] is None:
                continue
            if REC["n"] == 0:
                REC["t0"] = now
            elif late:
                REC["late"] += 1
            step, files = _step(REC["n"], now)
            try:
                WQ.put_nowait(("step", REC["path"], step, files))
                REC["n"] += 1
            except queue.Full:
                REC["dropped"] += 1
            if REC["n"] >= round(CFG["episode_s"] * CFG["rec_fps"]):
                _, item = _rec_end("save", "auto (--episode-s %g)" % CFG["episode_s"])
        if item is not None:
            WQ.put(item)


def writer():
    """Every file operation of the recording, in order, off the tick and request threads."""
    fh, fh_path = None, None
    while True:
        item = WQ.get()
        try:
            if item[0] == "step":
                _, path, step, files = item
                for rel, data in files:
                    with open(os.path.join(path, rel), "wb") as f:
                        f.write(data)
                if fh_path != path:
                    if fh is not None:
                        fh.close()
                    fh, fh_path = open(os.path.join(path, "steps.jsonl"), "a"), path
                fh.write(json.dumps(step, separators=(",", ":")) + "\n")
                fh.flush()
            elif item[0] == "close":
                _, how, path, final, meta, done = item
                if fh is not None:
                    fh.close()
                    fh, fh_path = None, None
                try:
                    if how == "discard":
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        with open(os.path.join(path, "meta.json"), "w") as f:
                            json.dump(meta, f, indent=1)
                        if how == "save":
                            os.rename(path, final)
                finally:
                    done.set()
            WRITER["err"] = None
        except Exception as ex:
            WRITER["err"] = "%s: %s" % (type(ex).__name__, ex)


# ---------------------------------------------------------------- page
PAGE = r"""<!doctype html><html class="dark"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Excavator dashboard</title><style>
:root{color-scheme:dark;--background:240 10% 3.9%;--foreground:0 0% 98%;--card:240 10% 3.9%;--muted:240 3.7% 15.9%;
 --muted-foreground:240 5% 64.9%;--secondary:240 3.7% 15.9%;--border:240 3.7% 15.9%;--input:240 3.7% 15.9%;--primary:0 0% 98%;
 --primary-foreground:240 5.9% 10%;--destructive:9 70% 46%;--destructive-foreground:0 0% 98%;--ok:142 71% 45%;--warn:38 92% 50%;
 --info:213 94% 68%;--radius:.5rem;--shadow-1:0 1px 2px 0 rgb(0 0 0/.4)}
*{box-sizing:border-box;border-color:hsl(var(--border))}
html,body{margin:0;background:hsl(var(--background));color:hsl(var(--foreground))}
body{font:14px/1.45 Inter,system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}
h2{font-size:20px;font-weight:500;letter-spacing:-.01em;margin:0}
.mono{font-family:"JetBrains Mono",ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums}
.muted{color:hsl(var(--muted-foreground))}.xs{font-size:12px}
.eyebrow{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:hsl(var(--muted-foreground))}
.page{min-height:100vh;display:flex;align-items:center;justify-content:center;padding:8px}
.shell{width:100%;min-height:95vh;display:flex;flex-direction:column;gap:16px;padding:8px}
@media(min-width:1024px){.shell{flex-direction:row;height:95vh}.page{padding:16px}}
.card{background:hsl(var(--card));border:1px solid hsl(var(--border));border-radius:var(--radius);padding:16px;flex:1;display:flex;flex-direction:column;min-width:0;min-height:0}
.head{display:flex;align-items:center;gap:16px;margin-bottom:16px;flex-wrap:wrap}
.logo{width:28px;height:28px;border-radius:6px;background:hsl(var(--primary));color:hsl(var(--primary-foreground));display:grid;place-items:center}
.vr{width:1px;height:24px;background:hsl(var(--border))}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;white-space:nowrap;border-radius:calc(var(--radius) - 2px);
 font:500 14px/1 inherit;font-family:inherit;height:40px;padding:0 16px;border:0;cursor:pointer;transition:background-color .15s;color:inherit}
.btn svg{width:16px;height:16px;flex-shrink:0}
.btn:focus-visible{outline:2px solid hsl(var(--primary));outline-offset:2px}
.btn-destructive{background:hsl(var(--destructive));color:hsl(var(--destructive-foreground))}.btn-destructive:hover{background:hsl(var(--destructive)/.9)}
.btn-secondary{background:hsl(var(--secondary));color:hsl(var(--foreground))}.btn-secondary:hover{background:hsl(var(--secondary)/.8)}
.btn-ghost{background:transparent}.btn-ghost:hover{background:hsl(var(--muted))}
.btn-on{background:hsl(var(--ok)/.18);color:hsl(var(--ok));box-shadow:inset 0 0 0 1px hsl(var(--ok)/.5)}
.pill{display:flex;align-items:center;gap:4px;border-radius:999px;border:1px solid hsl(var(--border));background:hsl(var(--card));padding:2px}
.pill .btn{height:28px;border-radius:999px;padding:0 10px;font-size:13px}
.pill .btn svg{width:14px;height:14px}
#stop{margin-left:auto;height:40px;font-weight:600}
#stop.idle{background:transparent;color:hsl(var(--destructive));box-shadow:inset 0 0 0 1px hsl(var(--destructive)/.5)}#stop.idle:hover{background:hsl(var(--destructive)/.12)}
.status{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}
.chip{display:inline-flex;align-items:center;gap:6px;height:26px;padding:0 10px;border-radius:999px;border:1px solid hsl(var(--border));font-size:12px;background:hsl(var(--background))}
.dot{width:8px;height:8px;border-radius:999px;background:hsl(var(--muted-foreground))}
.chip.live{border-color:hsl(var(--destructive)/.6);background:hsl(var(--destructive)/.15);color:hsl(9 90% 70%)}.chip.live .dot{background:hsl(var(--destructive));animation:p 1s infinite}
.chip.ok .dot{background:hsl(var(--ok))}.chip.warn{border-color:hsl(var(--warn)/.5);color:hsl(var(--warn))}.chip.warn .dot{background:hsl(var(--warn))}
.chip.bad{border-color:hsl(var(--destructive)/.6);color:hsl(9 90% 70%)}.chip.bad .dot{background:hsl(var(--destructive))}
@keyframes p{50%{opacity:.35}}
.viewer{flex:1;background:hsl(var(--background));border:1px solid hsl(var(--border));border-radius:calc(var(--radius) - 2px);display:flex;justify-content:center;overflow:auto;min-height:50vh}
@media(min-width:1024px){.viewer{min-height:0}}
.tab{background:transparent;color:hsl(var(--muted-foreground))}.tab.sel{background:hsl(var(--secondary));color:hsl(var(--foreground))}
.tab.live{color:hsl(var(--ok))}.tab.sel.live{color:hsl(var(--ok))}
.tabpage,.tabbody{display:flex;flex-direction:column;gap:16px}
.modebtn{height:44px;font-size:15px;font-weight:600}
.inner{width:100%;max-width:560px;padding:24px;display:flex;flex-direction:column;gap:16px}
.banner{border-radius:var(--radius);border:1px solid hsl(var(--border));padding:12px 14px;font-weight:600;font-size:14px;display:flex;align-items:center;gap:8px}
.banner.off{color:hsl(var(--muted-foreground))}.banner.ready{border-color:hsl(var(--warn)/.4);background:hsl(var(--warn)/.1);color:hsl(var(--warn))}
.banner.drive{border-color:hsl(var(--ok)/.5);background:hsl(var(--ok)/.12);color:hsl(var(--ok))}
.banner.stopped{border-color:hsl(var(--destructive)/.5);background:hsl(var(--destructive)/.12);color:hsl(9 90% 70%)}
.rows{display:flex;flex-direction:column}
.row{display:grid;grid-template-columns:64px 1fr 56px 56px;align-items:center;gap:12px;padding:6px 0;border-bottom:1px solid hsl(var(--border)/.5)}
.bar{position:relative;height:10px;border-radius:999px;background:hsl(var(--muted))}
.bar .mid{position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:hsl(var(--muted-foreground)/.6)}
.bar .fill{position:absolute;top:0;bottom:0;border-radius:999px;background:hsl(var(--foreground))}
.bar .fill.cmd{top:3px;bottom:3px;background:hsl(var(--info));opacity:.9}
.num{text-align:right;font-size:13px}.num.cmd{color:hsl(var(--info))}
.keys{display:grid;grid-template-columns:auto 1fr auto;gap:2px 12px;font-size:12px}
kbd{font-family:"JetBrains Mono",ui-monospace,monospace;font-size:11px;border:1px solid hsl(var(--border));border-bottom-width:2px;border-radius:4px;padding:0 5px;background:hsl(var(--muted)/.5)}
.side{display:flex;flex-direction:column;gap:12px;min-height:0}
@media(min-width:1024px){.side{width:clamp(384px,44vw,900px)}}
.side-head{display:flex;align-items:center;justify-content:space-between;gap:8px}
.switch{position:relative;width:44px;height:24px;border-radius:999px;border:2px solid transparent;background:hsl(var(--input));cursor:pointer;padding:0}
.switch[aria-checked=true]{background:hsl(var(--primary))}
.switch::after{content:"";position:absolute;top:0;left:0;width:20px;height:20px;border-radius:999px;background:hsl(var(--background));box-shadow:0 2px 6px rgb(0 0 0/.4);transition:transform .15s}
.switch[aria-checked=true]::after{transform:translateX(20px)}
.grid{display:grid;gap:12px;align-content:start;overflow-y:auto;min-height:0;grid-template-columns:1fr}
.grid.two{grid-template-columns:1fr 1fr}
.tile{min-width:0}
.media{position:relative;aspect-ratio:4/3;background:hsl(var(--muted));border-radius:calc(var(--radius) - 2px);overflow:hidden;display:grid;place-items:center}
.media img{width:100%;height:100%;object-fit:contain;display:block;background:#000}
.sq{position:absolute;top:0;bottom:0;left:12.5%;width:75%;box-sizing:border-box;border:2px dashed hsl(var(--warn)/.9);pointer-events:none}
.sq span{position:absolute;left:0;right:0;bottom:4px;text-align:center;font-size:11px;color:#fff;text-shadow:0 0 3px #000}
.empty{display:flex;flex-direction:column;align-items:center;gap:8px;padding:16px;text-align:center;color:hsl(var(--muted-foreground));font-size:13px}
.empty svg{width:32px;height:32px}
.empty code{font-size:11px;word-break:break-all;color:hsl(var(--foreground)/.8)}
.cap{display:flex;align-items:center;gap:4px 8px;margin-top:4px;font-size:12px;color:hsl(var(--muted-foreground));flex-wrap:wrap}
.cap .t{flex:1 1 100%;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.cap .btn{height:24px;padding:0 8px;font-size:12px}
.recas{display:inline-flex;align-items:center;gap:4px;white-space:nowrap}
.recas select{height:24px;border-radius:calc(var(--radius) - 2px);border:1px solid hsl(var(--input));background:hsl(var(--background));color:inherit;font:12px inherit;font-family:inherit;padding:0 4px}
.recas select:disabled{opacity:.5}
.tile.rec .media{box-shadow:0 0 0 2px hsl(var(--destructive)/.55)}
.tile.gone .media{opacity:.35;box-shadow:0 0 0 2px hsl(var(--warn)/.7)}
.tile.gone .cap .t{color:hsl(var(--warn))}
.note{font-size:12px;color:hsl(var(--warn));margin-top:2px}
#camMsg{min-height:16px;font-size:12px;color:hsl(var(--warn))}
#msg{min-height:18px;font-size:12px;color:hsl(var(--warn))}#msg:empty{display:none}
.recbox{border:1px solid hsl(var(--border));border-radius:var(--radius);padding:12px;display:flex;flex-direction:column;gap:8px}
.recbox.on{border-color:hsl(var(--destructive)/.6);background:hsl(var(--destructive)/.06)}
.recrow{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.input{flex:1;min-width:0;height:36px;border-radius:calc(var(--radius) - 2px);border:1px solid hsl(var(--input));background:transparent;color:inherit;font:inherit;padding:0 10px}
.input.tall{flex:none;height:auto;min-height:56px;padding:8px 10px;resize:vertical;line-height:1.4;overflow-wrap:anywhere}
.input:focus{outline:2px solid hsl(var(--primary)/.6);outline-offset:1px}
.recent{display:flex;flex-wrap:wrap;gap:6px}
.recent button{height:24px;border-radius:999px;border:1px solid hsl(var(--border));background:transparent;color:hsl(var(--muted-foreground));font:12px inherit;font-family:inherit;padding:0 10px;cursor:pointer;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.recent button:hover{color:hsl(var(--foreground));background:hsl(var(--muted))}
.recbox .btn{height:34px;padding:0 12px}
.qcbox{border-top:1px solid hsl(var(--border));padding-top:8px;display:flex;flex-direction:column;gap:6px}
.qv{display:inline-block;min-width:34px;text-align:center;border-radius:999px;border:1px solid hsl(var(--border));font-size:11px;font-weight:600;padding:0 6px;margin-right:4px}
.qv.ok{color:hsl(var(--ok));border-color:hsl(var(--ok)/.5)}.qv.warn{color:hsl(var(--warn));border-color:hsl(var(--warn)/.5)}
.qv.bad{color:hsl(var(--destructive));border-color:hsl(var(--destructive)/.6);background:hsl(var(--destructive)/.08)}.qv.na{color:hsl(var(--muted-foreground))}
.qcl{margin:2px 0;overflow-wrap:anywhere}.qcl .qn{color:hsl(var(--warn))}.qcep{margin:4px 0 0 4px}.qcep summary{font-weight:400}
.btn:disabled{opacity:.4;cursor:default}
.rdot{width:10px;height:10px;border-radius:999px;background:hsl(var(--destructive))}.recbox.on .rdot{animation:p 1s infinite}
#recTime{margin-left:auto;font-size:13px}
.cabbox{border:1px solid hsl(var(--border));border-radius:var(--radius);padding:12px;display:flex;flex-direction:column;gap:6px}
.cabbox.on{border-color:hsl(var(--destructive)/.6)}
.row.cab{grid-template-columns:64px 1fr 96px 56px}
.row.raw{grid-template-columns:110px 1fr 64px 56px;padding:3px 0}
.row.raw.hot .mono{color:hsl(var(--warn))}
.bar .fill.cabf{background:hsl(var(--ok))}
details summary{cursor:pointer;font-weight:500}
.unver{color:hsl(var(--warn))}
.camsup{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.camsup .btn{height:26px;padding:0 10px;font-size:12px}
.camsup details{flex:1 1 100%}
.camsup pre{max-height:180px;overflow:auto;white-space:pre-wrap;margin:4px 0 0}
.runmap .bad{color:hsl(9 90% 70%);font-weight:600}
.runopts{display:flex;flex-wrap:wrap;align-items:center;gap:6px 14px}.runopts .brk{flex-basis:100%;height:0}
.runopts label{display:inline-flex;align-items:center;gap:6px}.input.num{flex:none;width:64px;height:26px;padding:0 6px}
</style></head><body>
<div class="page"><div class="shell">
 <div class="card">
  <div class="head">
   <div class="logo" aria-hidden="true"><svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 20h10M5 20v-4h6l2-5 6-3 2 3-4 3"/><circle cx="19" cy="14" r="1"/></svg></div>
   <div class="vr"></div>
   <h2>Excavator</h2>
   <div class="pill" role="tablist" aria-label="pages">
    <button class="btn tab" id="tabTele" role="tab" title="Teleoperate page: drive with keys / gamepad">
     <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="6" x2="10" y1="11" y2="11"/><line x1="8" x2="8" y1="9" y2="13"/><line x1="15" x2="15.01" y1="12" y2="12"/><line x1="18" x2="18.01" y1="10" y2="10"/><path d="M17.32 5H6.68a4 4 0 0 0-3.978 3.59c-.006.052-.01.101-.017.152C2.604 9.416 2 14.456 2 16a3 3 0 0 0 3 3c1 0 1.5-.5 2-1l1.414-1.414A2 2 0 0 1 9.828 16h4.344a2 2 0 0 1 1.414.586L17 18c.5.5 1 1 2 1a3 3 0 0 0 3-3c0-1.545-.604-6.584-.685-7.258-.007-.05-.011-.1-.017-.151A4 4 0 0 0 17.32 5z"/></svg>
     <span id="tabTeleTxt">Teleoperate</span></button>
    <button class="btn tab" id="tabRec" role="tab" title="Record page: record a demonstration driven from the cab">
     <span class="rdot"></span><span id="tabRecTxt">Record</span></button>
    <button class="btn tab" id="tabInf" role="tab" title="Inference page: the runner on the Pi drives while you hold SPACE / L1 / R1">
     <svg viewBox="0 0 24 24" fill="currentColor" stroke="none"><path d="M7 4.5v15a1 1 0 0 0 1.5.86l12.5-7.5a1 1 0 0 0 0-1.72L8.5 3.64A1 1 0 0 0 7 4.5z"/></svg>
     <span id="tabInfTxt">Inference</span></button>
   </div>
   <button class="btn btn-destructive idle" id="stop" title="neutral + relay open">
    <svg viewBox="0 0 24 24" fill="currentColor"><rect x="5" y="5" width="14" height="14" rx="2"/></svg>STOP · Esc</button>
  </div>
  <div class="status" id="status"><span class="chip"><span class="dot"></span>connecting…</span></div>
  <div class="viewer" id="viewer"><div class="inner" id="main">
   <div class="banner off" id="banner">Off: nothing is sent to the machine</div>
   <div class="tabpage" id="pageTele">
    <button class="btn btn-secondary modebtn" id="tele" title="Teleoperate: off = nothing is sent to the machine"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="6" x2="10" y1="11" y2="11"/><line x1="8" x2="8" y1="9" y2="13"/><line x1="15" x2="15.01" y1="12" y2="12"/><line x1="18" x2="18.01" y1="10" y2="10"/><path d="M17.32 5H6.68a4 4 0 0 0-3.978 3.59c-.006.052-.01.101-.017.152C2.604 9.416 2 14.456 2 16a3 3 0 0 0 3 3c1 0 1.5-.5 2-1l1.414-1.414A2 2 0 0 1 9.828 16h4.344a2 2 0 0 1 1.414.586L17 18c.5.5 1 1 2 1a3 3 0 0 0 3-3c0-1.545-.604-6.584-.685-7.258-.007-.05-.011-.1-.017-.151A4 4 0 0 0 17.32 5z"/></svg><span id="teleTxt">Start teleoperating</span></button>
   </div>
   <div class="tabpage" id="pageInf">
    <button class="btn btn-secondary modebtn" id="pol" title="Start: the runner on the Pi drives while you hold SPACE / L1 / R1. Stop: inference off and the relay opens (the cab can drive at once)"><svg viewBox="0 0 24 24" fill="currentColor" stroke="none"><path d="M7 4.5v15a1 1 0 0 0 1.5.86l12.5-7.5a1 1 0 0 0 0-1.72L8.5 3.64A1 1 0 0 0 7 4.5z"/></svg><span id="polTxt">Start inference</span></button>
   <div class="recbox" id="runbox">
    <div class="recrow"><span style="font-weight:500">Inference runner</span>
     <span class="xs muted mono" id="runStat" style="margin-left:auto"></span></div>
    <div class="recrow"><span class="xs muted">backend</span><select class="input" id="runBackend">
     <option value="pi07">pi07 (PI gateway)</option><option value="pi05">pi0.5 (Modal URL)</option></select></div>
    <textarea class="input tall" id="runUrl" rows="2" autocomplete="off" spellcheck="false" placeholder="gateway URL (the checkpoint's gateway_url)"></textarea>
    <textarea class="input tall" id="runPrompt" rows="2" maxlength="200" autocomplete="off" spellcheck="false" placeholder="prompt: the exact training task string"></textarea>
    <div class="recrow"><span class="xs muted">training dataset</span><select class="input" id="runDs"></select></div>
    <label class="xs" id="runDaggerRow" title="The runner records every tick as an episode (Record tab's folder): the policy's steps, and your cab corrections after a STOP (docs/dagger.md)"><input type="checkbox" id="runDagger"> Record DAgger episodes (policy steps + cab takeovers)</label>
    <div class="runopts xs" id="runOpts">
     <label title="--subtasks (pi07): send the cone phase (reach / carry / release) as PI's raw_text, like the training labels"><input type="checkbox" id="optSub"> Subtasks</label>
     <label title="--stop-after-release: end the run this long after the phase enters release (subtasks only; empty = off)">stop after release <input class="input num" id="optSar" type="number" min="0" max="60" step="0.5" placeholder="off"> s</label>
     <span class="brk"></span>
     <label title="--grip-latch: once the policy clamps the thumb past the threshold, send a full clamp for 3.5 s (cancelled by deadman release / STOP)"><input type="checkbox" id="optGrip"> Grip latch</label>
     <label title="--grip-latch-threshold">threshold <input class="input num" id="optGripThr" type="number" min="0.01" max="1" step="0.05" value="0.3"></label>
     <span class="brk"></span>
     <label title="--chunk (pi07: actions per request)">chunk <input class="input num" id="optChunk" type="number" min="1" max="50" step="1" value="25"></label>
     <label title="--request-at: ask for the next chunk when this many actions remain">request at <input class="input num" id="optReq" type="number" min="1" max="49" step="1" value="8"></label>
     <span class="brk"></span>
     <label id="optSfdRow" title="--state-from-drove: the policy's state is the command that drove the machine on the previous tick (the training datasets' definition), not the gate's sent readback"><input type="checkbox" id="optSfd" checked> state = the command that drove (real-state datasets)</label>
     <span class="muted" id="optZero"></span>
    </div>
    <div class="xs mono runmap" id="runMap"></div>
    <details id="runCmdBox"><summary class="xs">command</summary><pre class="xs mono" id="runCmd" style="white-space:pre-wrap;margin:4px 0 0;overflow-wrap:anywhere"></pre></details>
    <div class="recrow">
     <button class="btn btn-secondary" id="runStart">Start runner</button>
     <button class="btn btn-ghost" id="runStop">Stop runner</button>
     <button class="btn btn-secondary" id="runSplit" style="display:none" title="Save the current DAgger episode; the next one opens when the policy or the cab drives again">Save episode · start new</button>
    </div>
    <div class="xs muted" id="runDaggerHelp" style="display:none">DAgger: to take over, press <b>Stop inference</b> (or STOP: both open
     the relay, the cab can't drive while it is closed), then drive from the cab: recorded as "human". Start inference and hold SPACE to hand
     back. Episodes are saved on Stop runner, a prompt change or "Save episode · start new".</div>
    <pre class="xs mono" id="runLog" style="max-height:160px;overflow:auto;white-space:pre-wrap;margin:0"></pre>
    <div class="xs muted">The runner asks the hosted model (pi07 at PI, or the pi0.5 fallback on Modal) for actions;
     nothing moves unless you hold SPACE / L1 / R1 here. Stop runner = STOP (relay open).</div>
   </div>
   </div>
   <div class="tabpage" id="pageRec">
   <div class="recbox" id="recbox">
    <div class="recrow"><span class="rdot"></span><span style="font-weight:500">Record episode · cab joysticks</span>
     <span class="xs muted" id="recCfg"></span><span class="mono" id="recTime"></span></div>
    <input class="input" id="task" maxlength="200" autocomplete="off" spellcheck="false" placeholder="task prompt (Enter here = done typing)">
    <div class="recent" id="recent"></div>
    <div class="recrow">
     <button class="btn btn-secondary" id="recStart"><span class="rdot"></span>Start · Enter</button>
     <button class="btn btn-secondary" id="recSave">Save · Enter</button>
     <button class="btn btn-ghost" id="recDiscard">Discard · Backspace</button>
    </div>
    <div class="xs muted mono" id="recStat"></div>
    <div class="banner stopped" id="recWarn" style="display:none"></div>
    <div class="qcbox" id="qcbox" style="display:none">
     <div class="recrow"><span style="font-weight:500">Last episode QC</span><span id="qcChip"></span>
      <span class="xs muted mono" id="qcName"></span></div>
     <div class="xs mono" id="qcLines"></div>
     <details id="qcRecent"><summary class="xs">Recent episodes · QC (last 10)</summary><div id="qcList" class="xs"></div></details>
    </div>
    <div class="xs muted">Records the operator in the cab. While recording this page sends nothing to the machine:
     Teleoperate must be off to start and stays off until you save or discard. STOP / Esc still works and discards the episode.</div>
   </div>
   <div class="cabbox" id="cabbox">
    <div style="display:flex;align-items:center;gap:8px">
     <span style="font-weight:500">Cab joysticks → recorded action</span>
     <span class="xs muted" id="cabNote" style="margin-left:auto"></span></div>
    <div class="rows" id="cabrows"></div>
    <details id="cabmon"><summary class="xs">Cab monitor · raw <span class="mono">/state controls</span> (read only)</summary>
     <div class="xs muted" style="margin:6px 0">Move one cab joystick at a time and watch which field moves and which way.
      Highlighted = beyond the deadband. <span class="mono">GET /api/cab</span> returns the same.</div>
     <div class="rows" id="cabraw"></div><div class="xs muted mono" id="cabother"></div>
    </details>
   </div>
   </div>
   <div class="tabbody" id="driveBox">
    <div>
     <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
      <svg id="act" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="muted"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
      <span style="font-weight:500">Gate output</span><span class="xs muted" style="margin-left:auto">white = <span class="mono">sent</span> (gate) · <span style="color:hsl(var(--info))">blue</span> = this page</span>
     </div>
     <div class="rows" id="rows"></div>
    </div>
    <div class="xs muted" id="pad">gamepad: none (press a button on it to connect)</div>
    <div id="keysBox">
     <div class="eyebrow" style="margin-bottom:6px">Keys · rig signs</div>
     <div class="keys" id="keys"></div>
    </div>
   </div>
   <div id="msg"></div>
  </div></div>
 </div>
 <section class="side" aria-label="Cameras">
  <div class="side-head"><span style="font-weight:500">Cameras <span class="muted" id="ncam">(0)</span></span>
   <span class="xs muted" id="bw"></span>
   <button class="switch" id="camsw" role="switch" aria-checked="true" aria-label="camera streams on/off"></button></div>
  <div class="camsup"><span id="supChip"></span>
   <button class="btn btn-secondary" id="supRestart" title="Restart camview (the dashboard's camera server): every stream drops for a few seconds">Restart camview</button>
   <span class="xs muted" id="supNote"></span>
   <details id="supLog"><summary class="xs">camview log (last 20 lines)</summary><pre class="xs mono" id="supTail"></pre></details></div>
  <div id="camMsg"></div>
  <div class="grid" id="grid"></div>
 </section>
</div></div>
<script>
// channel: [key for -1, key for +1, label -, label +]
const MAP={swing:["KeyA","KeyD","left","right"],arm:["KeyS","KeyW","in","out"],boom:["KeyK","KeyI","up","down"],
           bucket:["KeyJ","KeyL","curl","dump"],thumb:["KeyU","KeyO","open","clamp"]};
const PAD={swing:"left stick X",arm:"left stick Y",boom:"right stick Y",bucket:"right stick X",thumb:"LT / RT"};
const DRIVE=new Set(["Space"]);for(const m of Object.values(MAP)){DRIVE.add(m[0]);DRIVE.add(m[1]);}
const $=id=>document.getElementById(id);
const SID=Math.random().toString(36).slice(2);          // this page instance (its performance.now clock)
const now=()=>performance.now()/1000;
const down=new Set();
let on=false,epoch=null,seq=0,moving=false,timer=null,onAt=0,stopped=false,wantOn=false,lastCmd={},halts=0,gateLive=false;
let pol=false,pepoch=null,pseq=0,ptimer=null,pheld=false,polAt=0,wantPol=false,polInfo=null;
let runInfo=null,runFilled=false,runBusy=false,runDsKey="",runUrls={pi07:"",pi05:""},runBk="pi07";
const ICON_OFF='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.66 6H14a2 2 0 0 1 2 2v2.5l5.248-3.062A.5.5 0 0 1 22 7.87v8.196"/><path d="M16 16a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h2"/><path d="m2 2 20 20"/></svg>';
const ICON_PAUSE='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="14" y="4" width="4" height="16" rx="1"/><rect x="6" y="4" width="4" height="16" rx="1"/></svg>';
function esc(s){return String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}
function post(p,b){return fetch(p,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b||{})}).then(r=>r.json().then(j=>Object.assign({status:r.status},j))).catch(()=>({error:"network"}));}
function msg(t){$("msg").textContent=t||"";}
// ---- rows + key help
for(const [k,[n,p,ln,lp]] of Object.entries(MAP)){const r=document.createElement("div");r.className="row";
 r.innerHTML=`<span class="mono xs">${k}</span><div class="bar"><div class="mid"></div><div class="fill" id="f_${k}"></div><div class="fill cmd" id="g_${k}"></div></div>`+
  `<span class="num mono cmd" id="c_${k}">0.00</span><span class="num mono" id="v_${k}">0.00</span>`;$("rows").appendChild(r);
 $("keys").insertAdjacentHTML("beforeend",`<span class="mono">${k}</span><span><kbd>${n.slice(3)}</kbd> ${ln} · <kbd>${p.slice(3)}</kbd> ${lp}</span><span class="muted">${PAD[k]}</span>`);}
$("keys").insertAdjacentHTML("beforeend",`<span class="mono">deadman</span><span>hold <kbd>Space</kbd></span><span class="muted">hold L1 / R1</span><span class="mono">stop</span><span><kbd>Esc</kbd> or <kbd>X</kbd></span><span></span>`);
for(const k of Object.keys(MAP)){const r=document.createElement("div");r.className="row cab";
 r.innerHTML=`<span class="mono xs">${k}</span><div class="bar"><div class="mid"></div><div class="fill cabf" id="cf_${k}"></div></div>`+
  `<span class="mono xs muted" id="cm_${k}"></span><span class="num mono" id="cv_${k}">0.00</span>`;$("cabrows").appendChild(r);}
function setBar(id,v){setBarE($(id),v);}
function setBarE(e,v){v=Math.max(-1,Math.min(1,+v||0));e.style.left=v>=0?"50%":(50+50*v)+"%";e.style.width=Math.abs(50*v)+"%";}
// ---- gamepad (standard mapping only: other layouts could put sticks on unexpected axes)
function shape(x){const dz=0.12;if(Math.abs(x)<dz)return 0;const y=Math.sign(x)*(Math.abs(x)-dz)/(1-dz);return 0.5*y+0.5*y*y*y;}
function pad(){const gs=navigator.getGamepads?navigator.getGamepads():[];for(const g of gs)if(g&&g.connected)return g;return null;}
function padRead(){const g=pad();if(!g)return {g:null,dm:false,c:{}};if(g.mapping!=="standard")return {g,bad:true,dm:false,c:{}};
 const bv=i=>g.buttons[i]?(g.buttons[i].value||(g.buttons[i].pressed?1:0)):0,a=i=>+g.axes[i]||0;
 const dm=!!((g.buttons[4]&&g.buttons[4].pressed)||(g.buttons[5]&&g.buttons[5].pressed));
 return {g,dm,c:{swing:shape(a(0)),arm:-shape(a(1)),boom:-shape(a(3)),bucket:shape(a(2)),thumb:shape(bv(7)-bv(6))}};}
function focused(){return document.hasFocus()&&!document.hidden;}
// ---- command
function cmd(){const c={};if(!on||!focused())return c;const kd=down.has("Space"),p=padRead();
 for(const [k,[n,pp]] of Object.entries(MAP)){let v=0;if(kd)v+=(down.has(pp)?1:0)-(down.has(n)?1:0);if(p.dm)v+=p.c[k]||0;
  v=Math.max(-1,Math.min(1,v));if(Math.abs(v)>=0.005)c[k]=Math.round(v*1000)/1000;}return c;}
function deadman(){return down.has("Space")||padRead().dm;}
// ---- tabs: the header buttons only pick a page; each page has its own on / off control
const PAGES={tele:"pageTele",rec:"pageRec",inf:"pageInf"};let tab="tele";
try{const t=localStorage.getItem("tab");if(PAGES[t])tab=t;}catch(e){}
function setTab(t){if(t===tab)return;
 if(rec&&rec.active){msg("recording: save or discard the episode first");return;}
 if(on)teleOff();if(pol)polOff();                               // leaving a page switches its mode off (one neutral)
 tab=t;try{localStorage.setItem("tab",t);}catch(e){}msg("");tabLayout();}
function tabLayout(){const recOn=!!(rec&&rec.active);
 if(recOn&&tab!=="rec")tab="rec";                                // a recording (from any page) is always in view
 for(const k in PAGES)$(PAGES[k]).style.display=k===tab?"":"none";
 if(tab!=="rec"){const pg=$(PAGES[tab]);if($("driveBox").parentNode!==pg)pg.appendChild($("driveBox"));}
 $("driveBox").style.display=tab==="rec"?"none":"";$("keysBox").style.display=tab==="inf"?"none":"";
 const live={tele:on,rec:recOn,inf:pol};
 for(const [k,id] of [["tele","tabTele"],["rec","tabRec"],["inf","tabInf"]]){const b=$(id);
  b.classList.toggle("sel",k===tab);b.classList.toggle("live",!!live[k]);b.setAttribute("aria-selected",k===tab);}
 $("tabRecTxt").textContent=recOn?"Recording":"Record";
 $("stop").classList.toggle("idle",!on&&!pol&&!recOn&&!gateLive);}   // STOP glows while anything is live or the relay is closed
$("tabTele").onclick=()=>{$("tabTele").blur();setTab("tele");};
$("tabRec").onclick=()=>{$("tabRec").blur();setTab("rec");};
$("tabInf").onclick=()=>{$("tabInf").blur();setTab("inf");};
function show(c){tabLayout();for(const k in MAP){$("c_"+k).textContent=(c[k]||0).toFixed(2);setBar("g_"+k,c[k]||0);}
 const b=$("banner");
 $("pol").className="btn modebtn "+(pol?"btn-on":"btn-secondary");$("polTxt").textContent=pol?"Stop inference":"Start inference";
 if(pol){const ra=polInfo&&polInfo.runner_age,run=ra!=null&&ra<0.5;
  if(!focused()){b.className="banner ready";b.textContent="Inference on · window not focused: click the page (deadman released)";}
  else if(!run){b.className="banner ready";b.textContent="Inference on · runner not sending"+(ra==null?"":` (last ${ra.toFixed(1)} s ago)`)+": start control_loop.py --machine dashboard on the Pi";}
  else if(deadman()){b.className="banner drive";b.textContent="Deadman held: the model is driving (release to stop)";}
  else{b.className="banner ready";b.textContent="Inference on: hold SPACE (or L1 / R1) to let the model drive";}
  $("tele").className="btn btn-secondary modebtn";$("teleTxt").textContent="Start teleoperating";return;}
 if(stopped&&!on){b.className="banner stopped";b.textContent="STOPPED: relay open. Switch Teleoperate on to drive again.";}
 else if(!on){b.className="banner off";b.textContent="Off: nothing is sent to the machine";}
 else if(!focused()){b.className="banner ready";b.textContent="Window not focused: click the page to drive (sending neutral)";}
 else if(deadman()){b.className="banner drive";b.textContent="Deadman held: driving";}
 else{b.className="banner ready";b.textContent="Teleoperate on: hold SPACE (or L1 / R1) + keys / sticks to move";}
 $("tele").className="btn modebtn "+(on?"btn-on":"btn-secondary");$("teleTxt").textContent=on?"Stop teleoperating":"Start teleoperating";}
function tick(){const c=cmd();show(c);if(!on)return;
 if(Object.keys(c).length){moving=true;lastCmd=c;post("/api/write",{epoch,n:++seq,cmd:c}).then(r=>{if(r.status===409)forceOff(r.error);});}
 else if(moving){moving=false;lastCmd={};post("/api/neutral",{epoch,n:++seq}).then(r=>{if(r.status===409)forceOff(r.error);});}}
function stopLoop(){clearInterval(timer);timer=null;moving=false;lastCmd={};}
function forceOff(why){if(!on)return;stopLoop();on=false;epoch=null;down.clear();msg("teleop switched off: "+(why||"server"));show({});}
async function teleOn(){if(on||wantOn)return;wantOn=true;msg("");const from=tab,h0=halts;
 const r=await post("/api/teleop",{on:true,sid:SID,ct:now()});wantOn=false;
 if(r.epoch&&(tab!==from||halts!==h0)){post("/api/teleop",{on:false,epoch:r.epoch,n:1});show({});return;}  // left the page / STOP meanwhile
 if(r.epoch){on=true;stopped=false;epoch=r.epoch;seq=0;onAt=now();down.clear();timer=setInterval(tick,100);}
 else msg("could not switch on: "+(r.error||"?"));show({});}
function teleOff(){if(!on)return;const e=epoch,n=++seq;stopLoop();on=false;epoch=null;down.clear();
 post("/api/teleop",{on:false,epoch:e,n});show({});}
function polTick(){if(!pol)return;const h=focused()&&deadman();
 if(h||pheld)post("/api/policy/hold",{epoch:pepoch,n:++pseq,held:h}).then(r=>{if(r.status===409)polForceOff(r.error);});
 pheld=h;show({});}
function polStop(){clearInterval(ptimer);ptimer=null;pheld=false;}
function polForceOff(why){if(!pol)return;polStop();pol=false;pepoch=null;down.clear();msg("inference switched off: "+(why||"server"));show({});}
async function polOn(){if(pol||wantPol)return;if(on){msg("switch Teleoperate off first");return;}wantPol=true;msg("");
 const from=tab,h0=halts;
 const r=await post("/api/policy",{on:true,sid:SID,ct:now()});wantPol=false;
 if(r.epoch&&(tab!==from||halts!==h0)){post("/api/policy",{on:false,epoch:r.epoch,n:1});show({});return;}  // left the page / STOP meanwhile
 if(r.epoch){pol=true;stopped=false;pepoch=r.epoch;pseq=0;polAt=now();down.clear();ptimer=setInterval(polTick,100);}
 else msg("could not switch Inference on: "+(r.error||"?"));show({});}
function polOff(){if(!pol)return;const e=pepoch,n=++pseq;polStop();pol=false;pepoch=null;down.clear();
 post("/api/policy",{on:false,epoch:e,n});show({});}
$("pol").onclick=()=>{$("pol").blur();pol?polOff():polOn();};
function runShow(r){runInfo=r;if(!r)return;
 const k=(r.datasets||[]).map(d=>d+((r.dataset_zero_state||{})[d]?"*":"")).join("|");if(k!==runDsKey){runDsKey=k;const sel=$("runDs"),cur=sel.value;sel.innerHTML="";
  for(const d of r.datasets||[]){const o=document.createElement("option");o.value=d;o.textContent=d+((r.dataset_zero_state||{})[d]?" · zero-state":"");sel.appendChild(o);}
  if(cur)sel.value=cur;}
 if(!runFilled&&r.last){runFilled=true;runUrls={pi07:r.last.gateway_url||"",pi05:r.last.url||""};
  runBk=r.last.backend==="pi05"?"pi05":"pi07";$("runBackend").value=runBk;$("runUrl").value=runUrls[runBk];runBkShow();
  if(r.last.prompt)$("runPrompt").value=r.last.prompt;if(r.last.dataset)$("runDs").value=r.last.dataset;$("runDagger").checked=!!r.last.dagger;
  const o=r.last.opts;if(o){$("optSub").checked=!!o.subtasks;$("optSar").value=o.stop_after_release==null?"":o.stop_after_release;
   $("optGrip").checked=!!o.grip_latch;$("optGripThr").value=o.grip_latch_threshold||0.3;$("optChunk").value=o.chunk||25;
   $("optReq").value=o.request_at||8;if(o.state_from_drove!=null)$("optSfd").checked=!!o.state_from_drove;}}
 $("runStat").textContent=r.running?`running · pid ${r.pid} · ${Math.round(r.age||0)} s`:(r.exit!=null?`stopped (exit ${r.exit})`:"not running");
 const ssh=(r.dagger_pids||[]).filter(p=>!(r.running&&p===r.pid));if(ssh.length)$("runStat").textContent+=` · DAgger runner pid ${ssh.join(", ")} (started outside this panel)`;
 $("runStart").disabled=runBusy||r.running;$("runStop").disabled=runBusy||!r.running;
 $("runUrl").disabled=$("runPrompt").disabled=$("runDs").disabled=$("runBackend").disabled=$("runDagger").disabled=r.running;
 for(const e of $("runOpts").querySelectorAll("input"))e.disabled=r.running;
 const zs=!!(r.dataset_zero_state||{})[$("runDs").value];$("optSfdRow").style.display=zs?"none":"";
 $("optZero").textContent=zs?"zero-state dataset: --zero-state is set automatically":"";
 if(runBk!=="pi07"){$("optSub").checked=false;$("optSub").disabled=true;}$("optChunk").disabled=r.running||runBk!=="pi07";
 $("runCmdBox").style.display=r.running&&r.cfg&&r.cfg.argv?"":"none";
 const cmd=r.running&&r.cfg&&r.cfg.argv?r.cfg.argv.map(a=>/^[\w.\/:=,+-]+$/.test(a)?a:JSON.stringify(a)).join(" "):"";if($("runCmd").textContent!==cmd)$("runCmd").textContent=cmd;
 const ext=(r.dagger_pids||[]).filter(p=>!(r.running&&p===r.pid));   // DAgger runners started by hand (ssh)
 const dg=!!(r.running&&r.cfg&&r.cfg.dagger)||ext.length>0;
 $("runSplit").title=ext.length?`Save the current DAgger episode (runner pid ${ext.join(", ")}, started outside this panel)`:"Save the current DAgger episode; the next one opens when the policy or the cab drives again";$("runSplit").style.display=dg?"":"none";$("runSplit").disabled=runBusy;
 $("runDaggerHelp").style.display=dg||(!r.running&&$("runDagger").checked)?"":"none";
 const log=$("runLog"),t=(r.lines||[]).join("\n");if(log.textContent!==t){log.textContent=t;log.scrollTop=log.scrollHeight;}
 runMapShow(r);}
// camera mapping the runner uses: while running the one it was started with, before that the current "Record as"
// assignment for the cameras the selected dataset was trained on (missing / stalled in red)
function runMapShow(r){let h="";
 if(r.running&&r.cfg&&r.cfg.cams){const lb=r.cfg.cam_labels||{};
  h="runner cameras: "+Object.entries(r.cfg.cams).map(([n,id])=>`${esc(n)} ← ${esc(id)}${lb[n]?" ("+esc(lb[n])+")":""}`).join(" · ")+(r.cfg.zero_state?" · zero-state (--zero-state)":"")+(r.cfg.sync?` · sync ${esc(r.cfg.sync)}`:"");}
 else{const need=(r.dataset_cams||{})[$("runDs").value];
  if(!$("runDs").value)h="";else if(!need)h=`<span class="bad">dataset ${esc($("runDs").value)}: meta/info.json unreadable</span>`;
  else{const m=r.mapping_now||{};h="cameras: "+(need.length?need.map(n=>{const c=m[n];
   if(!c)return `<span class="bad">${esc(n)} ← not assigned (set Record as on its tile)</span>`;
   const t=`${esc(n)} ← ${esc(c.id||c.pattern)}${c.label?" ("+esc(c.label)+")":""}`;
   return c.ok?t:`<span class="bad">${t} · ${esc(c.status)}</span>`;}).join(" · "):"none in this dataset")+
   ((r.dataset_zero_state||{})[$("runDs").value]?" · zero-state (--zero-state)":"")+(r.sync?` · sync ${esc(r.sync)}`:"");}}
 const e=$("runMap");if(e.innerHTML!==h)e.innerHTML=h;}
function runBkShow(){$("runUrl").placeholder=runBk==="pi05"?"Modal URL (modal deploy modal_jobs/serve_pi05.py prints it)":"gateway URL (the checkpoint's gateway_url)";}
$("runBackend").onchange=()=>{runUrls[runBk]=$("runUrl").value;runBk=$("runBackend").value==="pi05"?"pi05":"pi07";
 $("runUrl").value=runUrls[runBk]||"";runBkShow();$("runBackend").blur();};
async function runOp(op){if(runBusy)return;const b={op};
 if(op==="start"){const u=$("runUrl").value.replace(/\s+/g,"");b.backend=runBk;if(runBk==="pi05")b.url=u;else b.gateway_url=u;
  b.prompt=$("runPrompt").value.replace(/\s+/g," ").trim();b.dataset=$("runDs").value;b.dagger=$("runDagger").checked;
  Object.assign(b,{subtasks:$("optSub").checked,stop_after_release:$("optSar").value,grip_latch:$("optGrip").checked,
   grip_latch_threshold:$("optGripThr").value,chunk:$("optChunk").value,request_at:$("optReq").value,
   state_from_drove:$("optSfd").checked});
  if(!u||!b.prompt){msg("runner: paste the "+(runBk==="pi05"?"Modal URL":"gateway URL")+" and the prompt first");return;}}
 runBusy=true;runShow(runInfo);const r=await post("/api/runner",b);runBusy=false;
 if(r.error)msg("runner: "+r.error);else msg(op==="start"?"runner started: "+(runBk==="pi05"?"a cold Modal container loads the model":"connecting to PI")+" (can take minutes)"+(b.dagger?" · recording DAgger episodes":""):op==="dagger_split"?"DAgger: episode saved; the next one opens when the policy or the cab drives":"runner stopping");
 runShow(await (await fetch("/api/runner")).json());}
$("runStart").onclick=()=>{$("runStart").blur();runOp("start");};
$("runDs").onchange=()=>{$("runDs").blur();if(runInfo)runMapShow(runInfo);};
$("runStop").onclick=()=>{$("runStop").blur();runOp("stop");};
$("runSplit").onclick=()=>{$("runSplit").blur();runOp("dagger_split");};
$("runDagger").onchange=()=>{$("runDagger").blur();runShow(runInfo);};
$("runDs").addEventListener("change",()=>runShow(runInfo));
for(const e of $("runOpts").querySelectorAll("input[type=checkbox]"))e.addEventListener("change",()=>e.blur());
function halt(){halts++;polStop();pol=false;pepoch=null;stopLoop();on=false;epoch=null;stopped=true;down.clear();post("/api/stop",{sid:SID,ct:now(),discard_rec:true}).then(r=>{if(r.error)msg("STOP error: "+r.error+" (use the kill switch)");});show({});}
$("tele").onclick=()=>{$("tele").blur();if(pol){msg("switch Inference off first");return;}on?teleOff():teleOn();};
$("stop").onclick=()=>{$("stop").blur();halt();};
// ---- keys (while a text field has focus only Esc (STOP) and Enter (leave the field) are handled)
function typing(e){const t=e.target;return !!t&&(t.tagName==="INPUT"||t.tagName==="TEXTAREA"||t.tagName==="SELECT"||t.isContentEditable);}
window.addEventListener("keydown",e=>{if(e.code==="Escape"){e.preventDefault();if(typing(e))e.target.blur();halt();return;}
 if(typing(e)){if(e.code==="Enter"||e.code==="NumpadEnter"){e.preventDefault();e.target.blur();}return;}
 if(e.code==="KeyX"){e.preventDefault();halt();return;}
 if(e.code==="Enter"||e.code==="NumpadEnter"){e.preventDefault();if(!e.repeat&&tab==="rec")recOp(rec&&rec.active?"save":"start");return;}
 if(e.code==="Backspace"){e.preventDefault();if(!e.repeat&&tab==="rec"&&rec&&rec.active)recOp("discard");return;}
 if(!(on||pol)||!DRIVE.has(e.code))return;e.preventDefault();if(e.repeat)return;down.add(e.code);pol?polTick():tick();});
window.addEventListener("keyup",e=>{if(down.delete(e.code)){if(pol)polTick();else if(on)tick();}});
function release(){down.clear();if(pol)polTick();else if(on)tick();else show({});}
window.addEventListener("blur",release);window.addEventListener("focus",()=>show(cmd()));
document.addEventListener("visibilitychange",()=>{if(document.hidden)release();camsVisible();});
window.addEventListener("gamepadconnected",padInfo);window.addEventListener("gamepaddisconnected",padInfo);
function padInfo(){const p=padRead();$("pad").textContent=!p.g?"gamepad: none (press a button on it to connect)":
 p.bad?`gamepad: ${p.g.id} — non-standard mapping, ignored`:`gamepad: ${p.g.id}${p.dm?" · deadman held":""}`;}
setInterval(()=>{padInfo();if(!on&&!pol)show({});},250);
// ---- status
function chip(cls,t){return `<span class="chip ${cls}"><span class="dot"></span>${t}</span>`;}
async function poll(){const t0=now();try{const j=await (await fetch("/api/state")).json();const rtt=Math.round((now()-t0)*1000);
 if(on&&t0>onAt&&j.teleop&&j.teleop.epoch!==epoch)forceOff("STOP or Teleoperate from another page");
 polInfo=j.policy||null;if(pol&&t0>polAt&&polInfo&&polInfo.epoch!==pepoch)polForceOff("STOP or Inference from another page");
 let h="";const g=j.gate;
 if(!g){gateLive=false;h=chip("bad","gate unreachable: "+esc(j.gate_error));}
 else{gateLive=!!g.live;h=(g.live?chip("live","LIVE: relay closed"):"")+                 // relay open (normal) shows nothing
  (g.veto?chip("warn","veto: "+esc(g.veto)):"")+          // veto / stale gate data: shown only when there is one
  (g.age==null||g.age>0.5?chip("warn","gate data <span class=mono>"+(g.age==null?"?":(+g.age).toFixed(2))+" s</span> old"):"")+(g.has_sent?"":chip("warn","no 'sent' in /state"));
  for(const k in MAP){$("v_"+k).textContent=(+g.sent[k]).toFixed(2);setBar("f_"+k,g.sent[k]);}}
 if(polInfo&&polInfo.on){const ra=polInfo.runner_age;
  h=chip(polInfo.held?"live":"warn",polInfo.held?"INFERENCE driving":"INFERENCE: deadman released")+
   chip(ra!=null&&ra<0.5?"ok":"warn","runner "+(ra==null?"not connected":`<span class=mono>${ra.toFixed(1)} s</span>`))+h;}
 if(j.rec&&j.rec.active)h=chip("live","REC cab joysticks <span class=mono>"+(+j.rec.elapsed).toFixed(1)+" s</span>")+h;
 if(j.cab)cabShow(j.cab);
 h+=chip(rtt>250?"warn":"ok","page ⇄ Pi <span class=mono>"+rtt+" ms</span>");$("status").innerHTML=h;$("act").style.color=g?"hsl(var(--ok))":"";tabLayout();
 if(j.rec)recShow(j.rec);if(j.runner!==undefined)runShow(j.runner);}
 catch(e){$("status").innerHTML=chip("bad","dashboard server unreachable");$("act").style.color="";}setTimeout(poll,200);}
// ---- cameras
let camsOn=true,camInfo=null;const tiles={};
try{camsOn=localStorage.getItem("camsOn")!=="0";}catch(e){}
function save(){try{localStorage.setItem("camsOn",camsOn?"1":"0");}catch(e){}}
// camview serves the streams; the browser loads them straight from it (a different port, so video never queues
// behind teleop requests). camview matches the path exactly: no query string on /stream/<id>.
function camviewBase(){const v=camInfo.camview;return v.public?v.public.replace(/\/$/,""):`${location.protocol}//${location.hostname}:${v.port}${v.prefix||""}`;}
function streamUrl(c){return `${camviewBase()}/stream/${c.id}`;}
function kill(t){if(t.img){t.img.onerror=null;t.img.src="data:,";t.img.remove();t.img=null;}if(t.sq){t.sq.remove();t.sq=null;}}
function emptyHtml(icon,text,code){return `<div class="empty">${icon}<span>${esc(text)}</span>${code?`<code>${esc(code)}</code>`:""}</div>`;}
function locked(){return !!((camInfo&&camInfo.locked)||(rec&&rec.active)||(runInfo&&runInfo.running));}
function lockWhy(){return (rec&&rec.active)||(camInfo&&camInfo.lock_why==="recording")?"Locked while an episode is being recorded":"Locked while the inference runner is running (stop it to change cameras)";}
function camMsg(t){$("camMsg").textContent=t||"";}
function camLabel(sid){const c=camInfo&&camInfo.cams.find(x=>x.stream_id===sid);return c&&c.label?`${sid} (${c.label})`:sid;}
function fmtChanges(ch){return ch.map(c=>c.was&&c.now?`${c.name} moved from ${camLabel(c.was)} to ${camLabel(c.now)}`:
 c.now?`${c.name} = ${camLabel(c.now)}`:`${c.name} off (was ${camLabel(c.was)})`).join(" · ");}
async function assign(t,sel){const v=sel.value||null;sel.blur();t.busy=true;
 const r=await post("/api/cams/assign",{stream_id:t.c.stream_id,name:v});t.busy=false;
 if(r.error)camMsg("Record as: "+r.error);else camMsg(r.changes&&r.changes.length?fmtChanges(r.changes):"");
 refreshCams();}
function renderTile(c){const k=c.key;let t=tiles[k];
 if(!t){t=tiles[k]={el:document.createElement("div"),img:null,err:0,id:null,busy:false};t.el.className="tile";
  t.el.innerHTML=`<div class="media"></div><div class="cap"><span class="t"></span><label class="recas" title="Record this stream under a name (dataset key / model image slot). A name is on one stream only.">Record as <select></select></label></div><div class="note"></div>`;
  const sel=t.el.querySelector("select");sel.onchange=()=>assign(t,sel);
  $("grid").appendChild(t.el);}
 if(t.img&&t.id!==c.id)kill(t);                                 // rebound to another camview stream
 t.c=c;const m=t.el.querySelector(".media"),live=!!c.id&&camInfo.camview.ok&&c.status!=="not_found";
 const gone=c.rec&&(c.missing||c.status!=="ok");
 t.el.classList.toggle("rec",!!c.rec&&!gone);t.el.classList.toggle("gone",!!gone);
 const want=live&&camsOn&&!document.hidden&&Date.now()>t.err;
 if(want&&!t.img){m.innerHTML="";const img=document.createElement("img");img.alt=k;
  img.onerror=()=>{kill(t);t.err=Date.now()+3000;renderTile(t.c);};img.src=streamUrl(c);t.id=c.id;m.appendChild(img);t.img=img;}
 else if(!want){kill(t);let html;
  if(c.missing)html=emptyHtml(ICON_OFF,`Assigned as ${c.name} but missing`,`camview doesn't list ${c.stream_id}: plug it back in, or set Record as to off`);
  else if(!camInfo.camview.ok)html=emptyHtml(ICON_OFF,"camview not reachable",camInfo.camview.error);
  else if(c.status==="not_found")html=emptyHtml(ICON_OFF,"Not in camview",c.msg);
  else if(!camsOn)html=emptyHtml(ICON_PAUSE,"Streams off","");
  else html=emptyHtml(ICON_OFF,Date.now()<=t.err?"Stream error, retrying…":"Hidden","");
  if(m.dataset.h!==html){m.innerHTML=html;m.dataset.h=html;}}
 if(t.img)m.dataset.h="";
 // resolution: the server's reader for assigned streams, else what the browser decoded
 const w=c.rec&&c.w?c.w:(t.img&&t.img.naturalWidth)||0,h=c.rec&&c.h?c.h:(t.img&&t.img.naturalHeight)||0,is43=!!w&&w*3===h*4;
 if(t.img&&c.rec&&!t.sq){const sq=document.createElement("div");sq.className="sq";   // default training crop: centered 480x480 of 640x480
  sq.title="Default training crop: the centered 480x480 square of the 640x480 frame. The crop can be shifted in conversion (--crop); keep everything that matters inside the square.";
  sq.innerHTML="<span>480×480 crop · can be shifted in conversion</span>";m.appendChild(sq);t.sq=sq;}
 if(t.sq&&(!c.rec||!t.img)){t.sq.remove();t.sq=null;}
 if(t.sq)t.sq.style.display=is43?"":"none";                      // outline geometry assumes a 4:3 frame
 const note=t.el.querySelector(".note");note.textContent=w&&!is43?`non-4:3 (${w}×${h}): set --crop in conversion`:"";
 note.title=w&&!is43?"The default crop is the centered square, which may be the wrong part of this frame (a side-by-side stereo camera like the ZED 2i at 1344×376 needs one eye, e.g. --crop NAME=148,0,376 for the left eye). See docs/dashboard.md.":"";
 const sel=t.el.querySelector("select"),lk=locked(),asg=camInfo.assign||{};
 const opts=[["","off"]].concat(Object.keys(camInfo.slots||{}).map(n=>[n,n+(asg[n]&&asg[n]!==c.stream_id&&n!==c.name?` (now ${asg[n]})`:"")]));
 const okey=JSON.stringify(opts)+c.missing;
 if(sel.dataset.k!==okey){sel.dataset.k=okey;sel.innerHTML="";for(const [v,txt] of opts){const o=document.createElement("option");o.value=v;o.textContent=txt;
   o.disabled=!!c.missing&&v!==""&&v!==c.name;if(v)o.title=`${v} → ${camInfo.slots[v]}`;sel.appendChild(o);}}
 if(document.activeElement!==sel&&!t.busy)sel.value=c.name||"";
 sel.disabled=lk||t.busy;sel.title=lk?lockWhy():`${c.name||"off"}${c.name?" → "+camInfo.slots[c.name]:""}`;
 let info;const res=w?` · ${w}×${h}`:"";
 if(c.missing)info=" · assigned but missing (not in camview)";
 else if(c.rec)info=c.status==="ok"?` · REC${res} · ${c.cap_fps} fps · ${c.jpeg_kb} KB · ~${(c.kbit_s/1000).toFixed(1)} Mbit/s`+(c.cap_lag!=null?` · lag ${Math.round(c.cap_lag*1000)} ms`:""):` · assigned but ${c.status==="error"&&/stalled/.test(c.msg||"")?"stalled":"not delivering"}: ${c.status} ${c.msg}`;
 else info=" · view only"+res+(c.msg?` · ${c.msg}`:c.age_s!=null?` · age ${(+c.age_s).toFixed(2)} s`:"");
 const cap=t.el.querySelector(".t");cap.textContent=`${c.name?c.name+" = ":""}${c.stream_id||c.id} · ${c.group||""} ${c.label||""}${info}`;
 cap.title=c.rec?`${c.name} = ${c.pattern} → ${camInfo.slots[c.name]} · frames ${c.n_frames} · reconnects ${c.restarts} · ${c.msg||""}`:(c.id||"");}
async function refreshCams(){try{camInfo=await (await fetch("/api/cams")).json();const cs=camInfo.cams;
 $("ncam").textContent=`(${cs.length})`;$("grid").className="grid"+(cs.length>1?" two":"");
 for(const k of Object.keys(tiles))if(!cs.some(c=>c.key===k)){kill(tiles[k]);tiles[k].el.remove();delete tiles[k];}
 if(!cs.length){$("grid").innerHTML=camInfo.camview.ok?emptyHtml(ICON_OFF,"camview lists no cameras","plug one in: camview picks it up within a few seconds"):
   supEmpty(camInfo.camview);}
 else{if(!Object.keys(tiles).length||$("grid").querySelector(":scope>.empty"))$("grid").innerHTML="";cs.forEach(renderTile);}
 const tot=cs.filter(c=>c.rec&&tiles[c.key]&&tiles[c.key].img).reduce((a,c)=>a+(c.kbit_s||0),0);
 const na=Object.keys(camInfo.assign||{}).length;
 $("bw").textContent=(na?`recording ${Object.entries(camInfo.assign).map(([n,s])=>n).join(" + ")}`:"no camera assigned: set Record as on a tile")+(tot?` · ≈ ${(tot/1000).toFixed(1)} Mbit/s`:"")+(locked()?(rec&&rec.active||camInfo.lock_why==="recording"?" · locked (recording)":" · locked (runner running)"):"");
 supShow(camInfo.camview.sup);}
 catch(e){}}
async function loadCams(){await refreshCams();setTimeout(loadCams,1000);}
function camsVisible(){if(camInfo)camInfo.cams.forEach(renderTile);}
$("camsw").setAttribute("aria-checked",camsOn);
$("camsw").onclick=e=>{e.currentTarget.blur();camsOn=!camsOn;$("camsw").setAttribute("aria-checked",camsOn);save();camsVisible();};
// ---- camview supervisor (the dashboard starts / restarts / stops camview itself: --camview auto)
let supInfo=null,supBusy=false,supLogT=null;
function fmtUp(s){return s==null?"":s<120?`${s} s`:s<7200?`${Math.round(s/60)} min`:`${(s/3600).toFixed(1)} h`;}
function supShow(s){supInfo=s;if(!s)return;let cls,t;
 if(s.mode==="off"){cls="";t="camview: off (not managed)";}
 else if(s.mode==="external"||s.state==="external"){cls=s.api_ok?"ok":"bad";t="camview: external (not managed)"+(s.api_ok?"":" · not answering");}
 else if(s.busy){cls="bad";t="camview: "+esc(s.state)+" · restart held back";}
 else if(s.state==="managed"&&s.api_ok){cls="ok";t=`camview: managed · pid <span class=mono>${s.pid}</span> · up ${fmtUp(s.uptime)}`;}
 else{cls="warn";t=`camview: ${esc(s.state)}`+(s.pid&&s.uptime!=null?` · pid <span class=mono>${s.pid}</span>`:"")+(s.restart_in!=null?` · restart in ${s.restart_in} s`:"");}
 if(s.mode==="auto"&&s.state!=="external")t+=` · restarts <span class=mono>${s.restarts}</span>`;
 const h=chip(cls,t);if($("supChip").innerHTML!==h)$("supChip").innerHTML=h;
 $("supChip").title=(s.external?"port held by "+s.external+"\n":"")+(s.last_error?"last error: "+s.last_error+"\n":"")+(s.python?"python "+s.python+"\n":"")+"log "+s.log;
 const n=[s.busy?"⚠ "+s.busy:"",s.last_error?"last error: "+s.last_error:"",s.note||"",
  s.replaced&&s.replaced.length?"took over camview pid "+s.replaced.join(", "):""].filter(x=>x).join(" · ");
 $("supNote").textContent=n;$("supNote").style.color=s.busy?"hsl(9 90% 70%)":"";
 const healthy=s.api_ok&&!s.busy&&(s.state==="managed"||s.state==="external"||s.mode==="external");   // shown only when camview needs attention
 $("supChip").style.display=healthy?"none":"";$("supNote").style.display=healthy?"none":"";
 const managed=s.mode==="auto"&&s.state!=="external";$("supRestart").style.display=managed?"":"none";
 $("supRestart").disabled=supBusy||!s.can_restart;
 $("supRestart").title=s.can_restart?"Restart camview (the dashboard's camera server): every stream drops for a few seconds":
  "Not now: refused while an episode is recording, the inference runner is running or Inference is on";
 $("supLog").style.display=s.mode==="off"?"none":"";}
function supEmpty(v){const s=v.sup||{};
 if(s.mode==="auto"&&s.state!=="external")return emptyHtml(ICON_OFF,s.busy?"camview down: "+s.busy:"camview is starting (the dashboard runs it)…",s.last_error||v.error);
 if(s.mode==="off")return emptyHtml(ICON_OFF,"camview not reachable (dashboard runs with --camview off): start it by hand","cd act-athon && .venv/bin/python3 camview.py");
 return emptyHtml(ICON_OFF,"external camview not reachable",v.error);}
async function supTail(){clearTimeout(supLogT);if(!$("supLog").open)return;
 try{const j=await (await fetch("/api/camview")).json();const e=$("supTail"),t=(j.tail||[]).join("\n")||"(empty)";
  if(e.textContent!==t){e.textContent=t;e.scrollTop=e.scrollHeight;}supShow(j);}catch(e){}
 supLogT=setTimeout(supTail,2000);}
$("supLog").addEventListener("toggle",supTail);
$("supLog").querySelector("summary").addEventListener("click",e=>e.currentTarget.blur());
$("supRestart").onclick=async e=>{e.currentTarget.blur();if(supBusy)return;
 if(!window.confirm("Restart camview? Every camera stream drops for a few seconds."))return;
 supBusy=true;supShow(supInfo);const r=await post("/api/camview/restart",{});supBusy=false;
 camMsg(r.error?"camview restart: "+r.error:"camview restarting…");supShow(supInfo);refreshCams();};
// ---- recording (server-side loop on the Pi; this page only starts / saves / discards and shows progress)
let rec=null,recBusy=false,recentKey="";
function recShow(r){rec=r;const a=!!r.active;$("recbox").classList.toggle("on",a);$("cabbox").classList.toggle("on",a);
 $("tele").disabled=a&&!on;$("tele").title=a?"Recording from the cab: Teleoperate stays off until the episode is saved or discarded":"Teleoperate: off = nothing is sent to the machine";
 $("recStart").disabled=a||recBusy;$("recSave").disabled=!a||recBusy;$("recDiscard").disabled=!a||recBusy;
 $("recCfg").textContent=`${r.fps} steps/s · auto-save ${r.episode_s} s · sync ${r.sync}`+(r.free_mb!=null?` · ${(r.free_mb/1024).toFixed(1)} GB free`:"");
 $("recTime").textContent=a?`${(+r.elapsed).toFixed(1)} / ${r.episode_s} s`:"";
 let s="";
 if(a){const miss=Object.entries(r.missing||{}).filter(([k,v])=>v).map(([k,v])=>`${k} ${v}`);
  s=`${r.n} steps · ${r.name}`+(r.track_steps?` · tracks moved in ${r.track_steps} steps`:"")+(miss.length?` · frames missing: ${miss.join(", ")}`:"")+(r.state_missing?` · gate state missing ${r.state_missing}`:"")+
   (r.late?` · late ${r.late}`:"")+(r.dropped?` · dropped ${r.dropped}`:"")+(r.backlog>20?` · disk backlog ${r.backlog}`:"")+
   (r.sync_last?` · skew ${r.sync_last.skew_ms} ms`+(r.sync_last.anchor?` (${r.sync_last.anchor} sets t)`:"")+(r.sync_last.left_out.length?` · waited out: ${r.sync_last.left_out.join(", ")}`:""):"");}
 else if(r.last)s=r.last.how==="save"?`saved ${r.last.name} · ${r.last.n} steps (${r.last.reason})`:
   r.last.how==="discard"?`discarded (${r.last.n} steps)`:`left unsaved: ${r.last.dir}`;
 if(!a&&r.last&&r.last.how==="save"&&r.last.track_steps)s+=` · tracks moved in ${r.last.track_steps} steps`;
 const w=$("recWarn"),tn=a&&r.tracks_now&&r.tracks_now.length;
 w.style.display=a&&(tn||r.track_steps)?"":"none";
 if(a)w.textContent=(tn?`TRACKS MOVING (${r.tracks_now.join(", ")}): the policy can't drive the tracks. `:"")+
  (r.track_steps?`Tracks moved in ${r.track_steps} of ${r.n} steps so far (still recording; counted in meta.json).`:"");
 if(r.writer_err)s+=` · WRITE ERROR: ${r.writer_err}`;
 $("recStat").textContent=s;
 const key=JSON.stringify(r.recent||[]);if(key!==recentKey){recentKey=key;$("recent").innerHTML="";
  for(const t of r.recent||[]){const b=document.createElement("button");b.textContent=t;b.title=t;
   b.onclick=()=>{$("task").value=t;b.blur();};$("recent").appendChild(b);}}
 qcShow(r);tabLayout();}
async function recOp(op){if(recBusy)return;const b={op};
 if(op==="start"){if(on||pol){msg("record: switch Teleoperate / Inference off first (a cab recording sends nothing to the machine)");return;}
  b.task=$("task").value.trim();if(!b.task){msg("record: type a task prompt first");$("task").focus();return;}}
 else{if(!rec||!rec.active)return;b.id=rec.id;}
 recBusy=true;if(rec)recShow(rec);let r=await post("/api/rec",b);
 if(op==="start"&&r.status===409&&r.confirm){          // camera set differs from the last saved episode's
  if(window.confirm(r.error)){b.confirm_new_camera_set=true;r=await post("/api/rec",b);}
  else r={error:"not started: camera set changed (cancelled)"};}
 recBusy=false;
 if(r.error)msg("record: "+r.error);else msg("");if(r.rec)recShow(r.rec);else if(rec)recShow(rec);}
// ---- episode QC (read only, docs/episode-qc.md): fetched when a newly saved episode shows up in rec.last and when
// "Recent episodes" is opened; never on the 200 ms poll itself
let qcKey="",qcTry=0;
function qcChip(v){v=v||"n/a";return `<span class="qv ${v==="n/a"?"na":esc(v)}">${esc(v.toUpperCase())}</span>`;}
function qcWhy(r){return (r.lines||[]).filter(l=>l.verdict==="warn"||l.verdict==="bad").map(l=>`${l.item}: ${l.verdict}`)
 .concat(r.problems||[]).join("\n");}
async function qcLoad(name){let res,j;try{res=await fetch("/api/episodes/qc?name="+encodeURIComponent(name));j=await res.json();}catch(e){j=null;}
 if(name!==qcKey)return;
 if((!res||res.status===404)&&++qcTry<=10){setTimeout(()=>{if(name===qcKey)qcLoad(name);},1000);return;}  // folder still being renamed
 if(!res){$("qcLines").textContent="QC request failed";return;}
 if(!res.ok){$("qcChip").innerHTML="";$("qcLines").textContent=(j&&j.error)||`QC failed (${res.status})`;return;}
 $("qcChip").innerHTML=qcChip(j.verdict);$("qcChip").title=qcWhy(j);$("qcLines").textContent="";}
async function qcRecent(){$("qcList").textContent="checking…";let res,j;
 try{res=await fetch("/api/episodes/qc?recent=10");j=await res.json();}catch(e){$("qcList").textContent="QC request failed";return;}
 if(!res.ok){$("qcList").textContent=j.error||`QC failed (${res.status})`;return;}
 $("qcList").innerHTML=j.length?"":"no saved episodes";
 for(const r of j){const d=document.createElement("div");d.className="qcep";d.title=qcWhy(r);
  d.innerHTML=`${qcChip(r.verdict)}<span class="mono">${esc(r.episode)}</span> <span class="muted">${esc(r.task)}</span>`;
  $("qcList").appendChild(d);}}
function qcShow(r){$("qcbox").style.display=r.qc?"":"none";if(!r.qc)return;
 const l=r.last,name=l&&l.how==="save"?l.name:"";
 if(!name){if(!qcKey)$("qcLines").textContent="no episode saved since the dashboard started · open Recent episodes below";return;}
 if(name===qcKey)return;
 qcKey=name;qcTry=0;$("qcName").textContent=name;$("qcChip").innerHTML="";$("qcChip").title="";$("qcLines").textContent="checking…";qcLoad(name);
 if($("qcRecent").open)qcRecent();}
$("qcRecent").addEventListener("toggle",()=>{if($("qcRecent").open)qcRecent();});
$("qcRecent").querySelector("summary").addEventListener("click",e=>e.currentTarget.blur());
// ---- cab joysticks (read only: the map + deadband are the server's; this only displays them)
let rawKey="",rawEls={};
function fmtMap(e){return `${e[1]<0?"−":"+"}${Math.abs(e[1])!==1?Math.abs(e[1]):""}${e[0]}${e[2]?(e[2]>0?"−":"+")+Math.abs(e[2]).toFixed(3):""}`;}
function cabShow(c){const used={};for(const k in c.map)used[c.map[k][0]]=k;
 $("cabNote").innerHTML=(c.ok?"":"gate unreachable · ")+(c.missing&&c.missing.length?`<span class="unver">map field missing for ${esc(c.missing.join(", "))}</span> · `:"")+
  ((p=>p.length?`<span class="unver">${esc(p.join(", "))}: ${esc(c.status[p[0]])}</span> · `:"")(Object.keys(c.status).filter(k=>c.status[k]!=="verified"&&c.status[k]!=="--cab-map")))+
  (c.map_file?`map ${esc(c.map_file)}`:"built-in map")+(c.tracks&&c.tracks.length?` · <span class="unver">tracks moving</span>`:"")+` · deadband ${(d=>d.length===1?d[0]:"per channel")([...new Set(Object.values(c.deadband))])}`;
 for(const k in MAP){const v=c.action?c.action[k]:0;$("cv_"+k).textContent=(+v).toFixed(2);setBar("cf_"+k,v);
  $("cm_"+k).textContent=fmtMap(c.map[k])+(c.status[k]==="verified"?"":" ?");$("cm_"+k).classList.toggle("unver",c.status[k]!=="verified");
  $("cm_"+k).title=`${c.status[k]} · action = clip(sign·(raw − offset)), 0 inside the deadband`;}
 const f=c.controls||{},keys=Object.keys(f),key=keys.join(",");
 if(key!==rawKey){rawKey=key;rawEls={};$("cabraw").innerHTML="";for(const n of keys){const r=document.createElement("div");r.className="row raw";
  r.innerHTML=`<span class="mono xs">${esc(n)}</span><div class="bar"><div class="mid"></div><div class="fill"></div></div><span class="xs muted">${used[n]?"→ "+used[n]:""}</span><span class="num mono"></span>`;
  rawEls[n]={r,b:r.querySelector(".fill"),v:r.querySelector(".num")};$("cabraw").appendChild(r);}}
 for(const n of keys){const v=+f[n],k=used[n],off=k?c.map[k][2]:0,db=k?c.deadband[k]:Math.min(...Object.values(c.deadband));
  const e=rawEls[n],clock=n==="age"||n==="t";e.v.textContent=v.toFixed(3);setBarE(e.b,clock?0:v);e.r.classList.toggle("hot",Math.abs(v-off)>db&&!clock);}
 $("cabother").textContent=Object.keys(c.other||{}).length?JSON.stringify(c.other):(c.ok?"":"gate: "+(c.error||"?"));}
$("recStart").onclick=e=>{e.currentTarget.blur();recOp("start");};
$("recSave").onclick=e=>{e.currentTarget.blur();recOp("save");};
$("recDiscard").onclick=e=>{e.currentTarget.blur();recOp("discard");};
show({});poll();loadCams();padInfo();
</script></body></html>"""


# ---------------------------------------------------------------- server
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"     # keep-alive: no new TCP handshake per teleop POST on the lossy link

    def log_message(self, *a):
        pass

    def _send(self, code, ctype, b):
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        except (BrokenPipeError, ConnectionResetError):
            # The client hung up before the reply (a page that reloaded, or the runner past its 1 s timeout): nothing
            # to send it. Counted and printed at most once every 5 s, with how long the request had taken.
            self.close_connection = True
            ms = (time.monotonic() - getattr(self, "_t0", time.monotonic())) * 1e3
            GONE["n"] += 1
            GONE["slow_ms"] = max(GONE["slow_ms"], ms)
            if time.monotonic() - GONE["t"] >= 5.0:
                print("client gone before the reply: %d since last note (latest %s %s after %.0f ms, slowest %.0f ms)"
                      % (GONE["n"], self.command, self.path.split("?", 1)[0], ms, GONE["slow_ms"]), flush=True)
                GONE.update(t=time.monotonic(), n=0, slow_ms=0.0)

    def _j(self, obj, code=200):
        self._send(code, "application/json", json.dumps(obj).encode())

    def do_GET(self):
        self._t0 = time.monotonic()
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode())
        elif path == "/api/state":             # newest result of the one read-only /state poller
            with LOCK:
                tele = {"on": TELE["on"], "epoch": TELE["epoch"]}
            snap = STATUS["snap"]
            self._j({"gate": snap["gate"], "gate_error": snap["error"], "teleop": tele, "policy": policy_info(),
                     "runner": runner_info(),
                     "rec": rec_info(),
                     "cab": cab_info(), "poll_age": round(time.monotonic() - snap["t"], 3) if snap["t"] else None})
        elif path == "/api/cab":               # Cab monitor: raw controls + derived action (read only)
            self._j(cab_info())
        elif path == "/api/cams":
            self._j(cams_info())
        elif path == "/api/runner":
            self._j(runner_info())
        elif path == "/api/camview":           # supervisor status + the tail of logs/camview.log
            self._j(sup_info(tail=True))
        elif path == "/api/episodes/qc":       # episode quality report (read only)
            code, obj = qc_api(urllib.parse.parse_qs(self.path.partition("?")[2]))
            self._j(obj, code)
        else:
            self._j({"error": "not found"}, 404)

    def _same_origin_json(self):
        """CSRF guard: a cross-site page can POST text/plain but not application/json without a CORS preflight
        (which this server never answers), and a browser always sends Origin on cross-site POSTs."""
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            return False
        origin = self.headers.get("Origin")
        if origin and origin != "null":
            return urllib.parse.urlsplit(origin).netloc.lower() == (self.headers.get("Host") or "").lower()
        return origin != "null"

    def do_POST(self):
        self._t0 = time.monotonic()
        ln = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(ln) if ln else b"{}"
        try:
            if self.path not in ("/api/stop", "/api/neutral") and not self._same_origin_json():
                self._j({"error": "rejected: POST must be application/json from this dashboard's own page"}, 403)
                return
            b = json.loads(body or b"{}")
            if self.path == "/api/teleop":
                self._j(teleop_on(b) if b.get("on") else teleop_off(b))
            elif self.path == "/api/write":
                self._j(gate_write(b))
            elif self.path == "/api/neutral":
                self._j(gate_neutral(b))
            elif self.path == "/api/policy":
                self._j(policy_on(b) if b.get("on") else policy_off(b))
            elif self.path == "/api/runner":          # start / stop the policy runner (it moves nothing by itself)
                op = b.get("op")
                if op == "start":
                    self._j(runner_start(b))
                elif op == "stop":
                    self._j(runner_stop())
                elif op == "dagger_split":
                    self._j(runner_dagger_split())
                else:
                    self._j({"error": "op must be start | stop | dagger_split"}, 400)
            elif self.path == "/api/policy/hold":     # the page's deadman heartbeat
                self._j(policy_hold(b))
            elif self.path == "/api/policy/act":      # the policy runner: from this Pi only
                if self.client_address[0] not in ("127.0.0.1", "::1"):
                    raise PermissionError("policy actions are accepted from the Pi itself only")
                self._j(policy_act(b))
            elif self.path == "/api/stop":     # ESC / X / STOP button: neutral + relay open, teleop + policy off.
                # Only the page's own STOP (same-origin JSON, discard_rec) also discards an episode being recorded:
                # the runner / replay_episode.py send /api/stop when they exit, and must never delete a cab demo.
                discard = bool(b.get("discard_rec")) and self._same_origin_json()
                try:
                    res = gate_stop(b)
                finally:
                    discarded = rec_discard_on_stop() if discard else False
                self._j(dict(res, rec_discarded=discarded) if isinstance(res, dict) else res)
            elif self.path == "/api/rec":      # recording never writes to the gate (it records the cab)
                op = b.get("op")
                if op == "start":
                    self._j(rec_start(b))
                elif op in ("save", "discard"):
                    self._j(rec_end(b, op))
                else:
                    self._j({"error": "op must be start | save | discard"}, 400)
            elif self.path == "/api/cams/assign":     # "Record as" on a tile (locked while recording: 409)
                self._j(cams_assign(b))
            elif self.path == "/api/camview/restart":  # managed camview only; 409 while recording / runner / Policy
                self._j(camview_restart(b))
            else:
                self._j({"error": "unknown"}, 404)
        except CamSetChanged as ex:            # the page asks, then resends with confirm_new_camera_set: true
            self._j({"error": str(ex), "confirm": ex.info}, 409)
        except PermissionError as ex:
            self._j({"error": str(ex)}, 409)
        except (ValueError, RuntimeError, OSError) as ex:
            self._j({"error": str(ex)}, 400)
        except Exception as ex:
            self._j({"error": str(ex)}, 500)


def make_server(port):
    return ThreadingHTTPServer(("0.0.0.0", port), H)


def parse_cam(spec):
    name, sep, pat = spec.partition("=")
    if not sep or name not in SLOTS or not pat:
        raise argparse.ArgumentTypeError("--cam wants NAME=<camview stream id>, NAME one of %s" % ", ".join(SLOTS))
    return name, pat


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=PORT, help="page + API port (default %(default)s)")
    ap.add_argument("--runner-dir", default=RUNNER["dir"], help="policy runner folder (control_loop.py, act/, datasets/, .env)")
    ap.add_argument("--runner-python", default=RUNNER["python"], help="python with pi-sdk for the runner (the venv)")
    ap.add_argument("--camview-url", default=CAMVIEW_URL, help="camview.py, as this server reaches it")
    ap.add_argument("--camview-public-url", default=None,
                    help="camview as the browser reaches it (default: the page's host, camview's port)")
    ap.add_argument("--camview", choices=("auto", "external", "off"), default="auto",
                    help="auto: start / restart / stop camview.py here (a camview.py of this user already on the port "
                         "is taken over with SIGTERM; anything else there is used as external); external: use what "
                         "answers on --camview-url, never start or signal anything; off: no supervisor "
                         "(default %(default)s)")
    ap.add_argument("--camview-python", default=os.path.join(HERE, ".venv", "bin", "python3"),
                    help="python for the managed camview (the venv with depthai: without it no OAK) "
                         "(default %(default)s; falls back to this python if missing)")
    ap.add_argument("--camview-script", default=SUP["script"], help="camview.py to run (default %(default)s)")
    ap.add_argument("--camview-log", default=SUP["log"], help="managed camview's output (default %(default)s)")
    ap.add_argument("--cam", type=parse_cam, action="append", default=[], metavar="NAME=ID",
                    help="optional: seed / override the camera assignment at startup (otherwise pick it on the page): "
                         "record camview stream ID (exact id, or a pattern like 'lux-*-color') as NAME (%s); "
                         "written to --assign-file; repeatable" % " / ".join(SLOTS))
    ap.add_argument("--assign-file", default=ASSIGN_FILE, metavar="PATH.json",
                    help="where the camera assignment is kept across restarts (default: %(default)s)")
    ap.add_argument("--exclude", action="append", default=[], metavar="PATTERN",
                    help="hide unassigned camview streams whose id or label matches (e.g. 'lux-*-depth'); repeatable")
    ap.add_argument("--no-auto", action="store_true", help="show only the assigned streams")
    ap.add_argument("--rec-fps", type=float, default=CFG["rec_fps"], help="recording steps per second")
    ap.add_argument("--episode-s", type=float, default=CFG["episode_s"], help="auto-save an episode after this long")
    ap.add_argument("--episodes-dir", default=CFG["episodes_dir"], help="where episodes are written")
    ap.add_argument("--sync", choices=camsync.SYNC_MODES, default=CFG["sync"],
                    help="how each step's frames are picked across cameras (camsync.py): aligned = all captured at the "
                         "same instant (the laggiest camera, normally the network world camera, sets it); latest = "
                         "each camera's newest frame. Written to meta.json and passed to the inference runner "
                         "(default %(default)s)")
    ap.add_argument("--cab-map", default=None, metavar="PATH.json",
                    help='override CAB_MAP / CAB_DEADBAND (cab joystick field + sign per channel), e.g. '
                         '{"arm": ["arm", -1.0], "deadband": 0.1}; listed entries replace the built-in ones')
    args = ap.parse_args()
    if args.rec_fps <= 0 or args.episode_s <= 0:
        ap.error("--rec-fps and --episode-s must be > 0")
    try:
        obj = {}
        if args.cab_map:
            with open(os.path.expanduser(args.cab_map)) as f:
                obj = json.load(f)
        cab_map, cab_db = cab_map_merge(obj, CAB_MAP, CAB_DEADBAND)
        CFG["cab_status"].update({ch: "--cab-map" for ch in obj if ch in CHANNELS})
    except (OSError, ValueError) as ex:
        ap.error("--cab-map %s: %s" % (args.cab_map, ex))
    names = [n for n, _ in args.cam]
    if len(set(names)) != len(names):
        ap.error("--cam names must be unique")
    if len({p for _, p in args.cam}) != len(args.cam):
        ap.error("--cam: one stream can't be recorded under two names")
    CFG.update(camview_url=args.camview_url, camview_public=args.camview_public_url, exclude=args.exclude,
               no_auto=args.no_auto, rec_fps=args.rec_fps, episode_s=args.episode_s,
               episodes_dir=os.path.expanduser(args.episodes_dir), poll_hz=max(10.0, args.rec_fps), sync=args.sync,
               assign_file=os.path.abspath(os.path.expanduser(args.assign_file)),
               cab_map=cab_map, cab_deadband=cab_db,
               cab_map_file=os.path.abspath(os.path.expanduser(args.cab_map)) if args.cab_map else None)
    CFG["port"] = args.port
    SUP.update(mode=args.camview, state=args.camview, script=os.path.abspath(os.path.expanduser(args.camview_script)),
               log=os.path.abspath(os.path.expanduser(args.camview_log)))
    if args.camview == "auto":
        py = os.path.abspath(os.path.expanduser(args.camview_python))
        if not os.path.isfile(py):
            SUP["note"] = ("no python at %s: camview runs with %s, so the OAK (depthai) path is unavailable"
                           % (py, sys.executable))
            py = sys.executable
        SUP["python"] = py
        host = camview_addr()[0]
        if host not in ("127.0.0.1", "localhost", "::1"):
            ap.error("--camview auto needs a loopback --camview-url (camview runs on this Pi); "
                     "use --camview external for %s" % CFG["camview_url"])
        if not os.path.isfile(SUP["script"]):
            ap.error("--camview auto: no camview script at %s (--camview-script, or --camview external / off)"
                     % SUP["script"])
        SUP["state"] = "starting"
    RUNNER.update(dir=os.path.abspath(args.runner_dir), python=os.path.abspath(args.runner_python))
    srv = make_server(args.port)
    load_recent()
    load_last_set()
    assign = {}
    if os.path.exists(CFG["assign_file"]):
        try:
            assign = load_assign(CFG["assign_file"])
        except (OSError, ValueError) as ex:
            print("WARNING: ignoring %s (%s): no camera assigned" % (CFG["assign_file"], ex), flush=True)
    seeded = dict(assign)
    for name, pat in args.cam:                   # --cam wins over the file (and takes its stream off other names)
        assign = {n: p for n, p in assign.items() if n != name and p != pat}
        assign[name] = pat
    if assign != seeded:
        try:
            save_assign(assign)
        except OSError as ex:
            print("WARNING: can't write %s (%s): the --cam assignment holds until restart" % (CFG["assign_file"], ex),
                  flush=True)
    apply_assign(assign)
    for fn in (watchdog, poller, camview_scanner, recorder, writer):
        threading.Thread(target=fn, name=fn.__name__, daemon=True).start()
    if SUP["mode"] == "auto":                    # long-lived: it forks camview (PR_SET_PDEATHSIG is per thread)
        threading.Thread(target=camview_supervisor, name="camview_supervisor", daemon=True).start()
    print("dashboard on http://0.0.0.0:%d  (ctrl-c to quit; quitting sends stop)" % args.port, flush=True)
    print("camview: %s" % {"auto": "managed by this dashboard (python %s, log %s)%s" % (
        SUP["python"], SUP["log"], "; " + SUP["note"] if SUP["note"] else ""),
        "external": "external (--camview external: never started or signalled here)",
        "off": "off (--camview off: not managed)"}[SUP["mode"]], flush=True)
    print("cameras from camview %s: %s  (assignment file %s)" % (CFG["camview_url"], ", ".join(
        "%s=%s" % (n, p) for n, p in assign.items()) or "none assigned yet: pick them on the page (Record as)",
        CFG["assign_file"]), flush=True)
    if LAST_SET["set"] is not None:
        print("last saved episode %s used cameras %s" % (LAST_SET["episode"], fmt_set(LAST_SET["set"])), flush=True)
    print("episodes: %s  (%g steps/s, auto-save after %g s), recording the cab joysticks" % (
        CFG["episodes_dir"], CFG["rec_fps"], CFG["episode_s"]), flush=True)
    print("cab map (%s): %s" % (CFG["cab_map_file"] or "built-in", ", ".join(
        "%s=%+g*(%s-%g) db %g [%s]" % (ch, s_, f_, o_, CFG["cab_deadband"][ch], CFG["cab_status"][ch])
        for ch, (f_, s_, o_) in CFG["cab_map"].items())), flush=True)

    quitting = []

    def _quit(signum, frame):                # ssh session closed (SIGHUP) / kill: same clean exit as Ctrl-C
        if quitting:                         # a second SIGTERM / SIGHUP must not cut the cleanup below short
            return
        quitting.append(signum)
        raise KeyboardInterrupt
    for s in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, _quit)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        quitting.append("finally")
        with SUP_LOCK:                       # the supervisor starts nothing from here on (camview_shutdown below)
            SUP["shutdown"] = True
        runner_stop(wait_s=6.0)
        with LOCK:
            TELE.update(on=False, epoch=None, n=-1, wrote=False)
            POL.update(on=False, epoch=None, n=-1, hold_t=0.0, wrote=False)
            _neutral_mark()
            try:
                rig.stop()
            except Exception:
                pass
        with REC_LOCK:                       # an open episode stays as <name>.partial (not deleted, not "saved")
            item = _rec_end("abort", "server stopped")[1] if REC["id"] is not None else None
        if item is not None:
            WQ.put(item)
            item[-1].wait(5.0)
            print("unsaved episode left as %s" % item[2], flush=True)
        RUNNING.clear()
        for c in list(CAMS.values()):
            c.close()
        camview_shutdown()                   # our managed camview only (SIGTERM, SIGKILL after 6 s); never an external


if __name__ == "__main__":
    main()
