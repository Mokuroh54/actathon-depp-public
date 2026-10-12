"""The real mini excavator, through the organisers' HTTP gate on its Raspberry Pi.

Action / observation schema (act/machine.py):
    action       {ch}.cmd for ch in CHANNELS, joystick commands in -1..1 (NOT positions), rig signs:
                 swing + = house right, boom + = down, arm + = out, bucket + = dump/open, thumb + = clamp in
    observation  {ch}.cmd = /state `sent`: what the gate is actually outputting (after caps and the
                 0.5 s hold), plus cameras. There is NO joint-angle feedback on this machine.
                 Extra readback (not in observation_features, so never in a dataset):
                 {ch}.cab (/state `controls`, the cab joystick / ADC input), {ch}.last_cmd (/state
                 `last_cmd`, the last command code sent), throttle, live (relay armed),
                 veto (gate's reason for refusing, "" if none), age_s.
                 Verified on hardware: while boom keys were held from code, `sent` and `last_cmd`
                 showed boom +/-0.2 and `controls.boom` stayed 0 (`controls` = cab input, not our output).

The gate listens on 127.0.0.1:8096 on the Pi; from the Mac: `ssh -L 8096:127.0.0.1:8096 excavator`.
    GET  /state   readback             POST /cmd {name: value}   POST /estop {}
Gate semantics this class relies on:
  * The FIRST /cmd write energises the relay. connect() therefore only reads /state.
  * Each /cmd is held for 0.5 s, then the gate goes neutral: the caller must send at >= 5 Hz
    (every control tick) to keep moving. Stopping sending is itself a stop.
  * Unlisted channels are set to 0; tracks and blade are forced to 0 by the gate.
  * The gate calls the swing channel "body" (raw controls.swing is something else).
HTTP / network errors propagate to the caller: the gate neutrals on its own within 0.5 s.
disconnect() sends /estop (neutral + relay open).
"""
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np

from act import camsync
from act.cameras import open_camera, parse_cameras
from act.machine import CHANNELS

DEFAULT_RIG_URL = "http://127.0.0.1:8096"
DEFAULT_DASHBOARD_URL = "http://127.0.0.1:8080"   # pi/act-athon/dashboard.py on the Pi (Policy mode)
TO_API = {"swing": "body"}  # our channel name -> gate name


@dataclass
class RigMachineConfig:
    base_url: str = DEFAULT_RIG_URL
    timeout_s: float = 1.0
    joint_names: list = field(default_factory=lambda: list(CHANNELS))  # fixed by the gate; don't reorder
    # camera name -> int (OpenCV device index) or URL string.
    cameras: dict = field(default_factory=dict)
    # Images are SQUARE crops of the native frame (every camera is captured 640x480): camera name -> (x, y, size) box
    # in native pixels; a camera not listed gets the centered square (x=80, y=0, size=480 on 640x480). The crop is
    # then resized to image_hw. Same code as scripts/convert_pi_episodes.py (prepare_image): training and deployment
    # must use the same crops (the converter writes them to <dataset>/crops.json, loadable with load_crop_file).
    crops: dict = field(default_factory=dict)
    image_hw: tuple = (480, 480)  # stored/returned size for every camera (square: the crop, resized if needed)
    capture_hw: tuple = (480, 640)  # size requested from OpenCV device-index cameras (ignored by stream URLs)
    # How one observation's frames are picked across cameras (camsync.select, docs/runner_sync.md): MUST be the mode the
    # training episodes were recorded with (episode meta.json "sync"; control_loop.py --sync).
    sync: str = camsync.DEFAULT_MODE
    # Freshness (the safety check): every camera's newest NEW frame, and every frame an observation uses, must have
    # arrived at most this long ago (= the converter's --max-age, so no training step had older frames).
    max_frame_age_s: float = 0.5


def parse_crop(spec):
    """'NAME=x,y,size' -> (name, (x, y, size)): a square crop box in native pixels."""
    name, sep, box = spec.partition("=")
    try:
        vals = tuple(int(v) for v in box.split(","))
    except ValueError:
        vals = ()
    if not sep or not name or len(vals) != 3:
        raise ValueError(f"bad crop {spec!r}: want NAME=x,y,size (pixels in the native frame)")
    return name, check_crop(vals, name)


def check_crop(box, name="?"):
    """Validate a crop box -> (x, y, size) ints (x, y >= 0, size >= 1)."""
    if (not isinstance(box, (list, tuple)) or len(box) != 3
            or any(isinstance(v, bool) or not isinstance(v, int) for v in box)):
        raise ValueError(f"crop for {name!r} must be [x, y, size] integers, got {box!r}")
    x, y, size = box
    if x < 0 or y < 0 or size < 1:
        raise ValueError(f"crop for {name!r} must have x, y >= 0 and size >= 1, got {list(box)}")
    return (x, y, size)


def load_crop_file(path):
    """JSON {camera: [x, y, size]} -> {camera: (x, y, size)} (the format of --crop-file and <dataset>/crops.json)."""
    with open(path) as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: want a JSON object {{camera: [x, y, size]}}")
    return {k: check_crop(v, k) for k, v in obj.items()}


def crop_box(frame_hw, crop=None):
    """The (x, y, size) square actually used on a frame of this (h, w): `crop` if given (must fit), else centered."""
    h, w = frame_hw
    if crop is None:
        size = min(h, w)
        return ((w - size) // 2, (h - size) // 2, size)
    x, y, size = check_crop(crop)
    if x + size > w or y + size > h:
        raise ValueError(f"crop {[x, y, size]} does not fit a {w}x{h} frame")
    return (x, y, size)


def prepare_image(bgr, crop, hw):
    """BGR frame -> RGB (hw, 3): square crop (crop_box) -> INTER_AREA resize to hw if needed -> RGB.
    THE image path for both training data (convert_pi_episodes.py) and deployment (RigMachine.get_observation)."""
    x, y, size = crop_box(bgr.shape[:2], crop)
    img = bgr[y:y + size, x:x + size]
    h, w = hw
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def make_machine(kind, cam_specs=None, rig_url=DEFAULT_RIG_URL, require_cameras=False, crops=None, image_hw=None,
                 dashboard_url=DEFAULT_DASHBOARD_URL, sync=camsync.DEFAULT_MODE):
    """kind "real" -> RigMachine, "dashboard" -> DashboardMachine (cameras only from --cam; `crops`
    {camera: (x, y, size)} and `image_hw` must match the training dataset: its crops.json and image size (converter
    --out-size); defaults = centered squares, 480x480). `sync`: camsync mode of the training episodes.
    Both have .cfg.joint_names / .cfg.cameras / .cfg.image_hw and the same features."""
    if kind not in ("real", "dashboard"):
        raise ValueError(f"unknown machine {kind!r} (real | dashboard)")
    cams = parse_cameras(cam_specs) if cam_specs else {}
    if require_cameras and not cams:
        raise SystemExit(f"--machine {kind} needs at least one --cam name=index|url")
    cfg = RigMachineConfig(base_url=rig_url, cameras=cams, crops=dict(crops or {}),
                           image_hw=tuple(image_hw) if image_hw else RigMachineConfig.image_hw, sync=sync)
    return DashboardMachine(cfg, dashboard_url) if kind == "dashboard" else RigMachine(cfg)


class RigMachine:
    def __init__(self, cfg: RigMachineConfig):
        self.cfg = cfg
        self.url = cfg.base_url.rstrip("/")
        self.caps = {}
        self.connected = False
        self.last_state, self.last_reply = None, None
        self.last_sync = None       # how the last observation's frames were picked (tick log): see _synced_frames
        self.last_picked = {}       # camera -> the RingFrame (camview JPEG bytes) the last observation used (--record-dagger)
        if cfg.sync not in camsync.SYNC_MODES:
            raise ValueError(f"sync {cfg.sync!r}: one of {camsync.SYNC_MODES}")
        for name, box in cfg.crops.items():
            check_crop(tuple(box), name)

    # ---------- LeRobot-style feature descriptions ----------
    @property
    def action_features(self):
        return {f"{j}.cmd": float for j in self.cfg.joint_names}

    @property
    def observation_features(self):
        h, w = self.cfg.image_hw
        feats = {f"{j}.cmd": float for j in self.cfg.joint_names}
        feats.update({name: (h, w, 3) for name in self.cfg.cameras})
        return feats

    # ---------- HTTP ----------
    def _http(self, path, payload=None):
        data = json.dumps(payload, allow_nan=False).encode() if payload is not None else None
        req = urllib.request.Request(self.url + path, data=data, method="POST" if data is not None else "GET",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as r:
            return json.loads(r.read())

    def read_state(self) -> dict:
        """GET /state -> {ch}.cmd (from `sent`) + {ch}.cab, {ch}.last_cmd, throttle, live, veto, age_s."""
        s = self._http("/state")
        self.last_state = s
        sent = s.get("sent")
        if not isinstance(sent, dict):
            raise RuntimeError(f"gate /state has no 'sent' dict (keys: {sorted(s)}): readback schema changed?")
        # `sent` = gate output after caps/hold, `controls` = cab input (verified on hardware, see module doc).
        c, last = s.get("controls", {}), s.get("last_cmd") or {}
        out = {}
        for j in self.cfg.joint_names:
            k = TO_API.get(j, j)  # swing <-> body; raw controls.swing is NOT our swing channel
            out[f"{j}.cmd"] = float(sent.get(k, 0.0))
            out[f"{j}.cab"] = float(c.get(k, 0.0))
            out[f"{j}.last_cmd"] = float(last.get(k, 0.0))
        out.update(throttle=c.get("throttle"), live=bool(s.get("armed")), veto=s.get("veto", ""), age_s=c.get("age"))
        return out

    # ---------- lifecycle ----------
    def connect(self):
        """Reads /state only (no write: the first /cmd energises the machine), then opens cameras."""
        state = self.read_state()
        writable = self.last_state.get("writable")
        missing = [j for j in self.cfg.joint_names if writable is not None and TO_API.get(j, j) not in writable]
        if missing:
            print(f"WARNING: gate reports {missing} not writable (writable: {writable})")
        print(f"rig gate {self.url}: live={state['live']} veto={state['veto']!r} mode={self.last_state.get('mode')!r}")
        for name, src in self.cfg.cameras.items():
            self.caps[name] = open_camera(src, self.cfg.capture_hw)
        self.connected = True

    def disconnect(self):
        """POST /estop (neutral + relay open), then release cameras."""
        try:
            if self.connected:
                self.last_reply = self._http("/estop", {})
        finally:
            for cap in self.caps.values():
                cap.release()
            self.caps.clear()
            self.connected = False

    def stop(self):
        """POST /estop without closing the session (cameras stay open)."""
        self.last_reply = self._http("/estop", {})
        return self.last_reply

    # ---------- robot API ----------
    def get_observation(self) -> dict:
        obs = self.read_state()
        for name, frame in self._synced_frames().items():
            crop = self.cfg.crops.get(name)
            obs[name] = prepare_image(frame, tuple(crop) if crop is not None else None, self.cfg.image_hw)
        return obs

    def send_action(self, action: dict) -> dict:
        """Clip to -1..1 and POST /cmd with all channels. Held 0.5 s by the gate: call every tick."""
        a = {j: float(np.clip(action[f"{j}.cmd"], -1.0, 1.0)) for j in self.cfg.joint_names}
        self.last_reply = self._http("/cmd", {TO_API.get(j, j): v for j, v in a.items()})
        return {f"{j}.cmd": v for j, v in a.items()}

    # ---------- cameras ----------
    def _synced_frames(self):
        """{camera: BGR frame} picked by camsync.select(cfg.sync), exactly like the dashboard recorder's _step (same
        rings of genuinely new frames, same max_lag, now = wall clock). Safety, in this order:
          1. every camera's check_fresh(): raises when a camera stopped delivering NEW frames (as read() always did);
          2. camsync.select picks one ring frame per camera (aligned: nearest the laggiest camera's newest capture);
          3. every picked frame must have ARRIVED within max_frame_age_s -- aligned mode can pick a frame older than a
             camera's newest; it must never use one older than the stale limit (the converter's --max-age rejected
             such steps, so the policy never trained on them). Raises -> the run ends (STOP).
        Only the picked JPEGs are decoded. self.last_sync = select's info + per camera (capture time, arrival age, seq)."""
        max_age = self.cfg.max_frame_age_s
        for name in self.cfg.cameras:
            self.caps[name].check_fresh(max_age)
        rings = {n: self.caps[n].ring for n in self.cfg.cameras}
        wall = time.time()
        chosen, info = camsync.select(rings, self.cfg.sync, camsync.MAX_LAG_S, wall)
        now = time.monotonic()
        picked, cams = {}, {}
        for name in self.cfg.cameras:
            src = self.caps[name].src
            if chosen.get(name) is None:
                raise RuntimeError(f"camera {src!r}: no frame to pick ({self.cfg.sync}): ring empty")
            t_cap, item = chosen[name]
            age = now - item.arrival
            if age > max_age:        # aligned: the anchor camera (the laggiest) drags the others back to its time
                why = (f"t_ref is set by {info['anchor']!r}, whose newest frame was captured "
                       f"{wall - info['t_ref']:.2f}s ago -> {info['anchor']!r} is lagging / freezing"
                       if info["anchor"] is not None else "no newer frame")
                raise RuntimeError(f"camera {src!r} ({name}) stale: the {self.cfg.sync}-picked frame arrived "
                                   f"{age:.2f}s ago (> {max_age}s); {why}")
            picked[name] = item
            cams[name] = {"cap": t_cap, "age": age, "seq": item.seq}
        self.last_sync = dict(info, cams=cams)
        self.last_picked = picked
        return {n: self.caps[n].frame_of(item) for n, item in picked.items()}


class DashboardMachine(RigMachine):
    """The real excavator driven THROUGH the Pi dashboard (pi/act-athon/dashboard.py, Policy mode); runs on the Pi.
    Reads /state from the gate (read only) and the cameras (camview streams) like RigMachine, but never writes to the
    gate: every action goes to the dashboard's POST /api/policy/act, which passes it on only while a human holds the
    deadman on the dashboard page (Policy on + SPACE / L1 / R1). `allowed` = the dashboard's verdict on the last action.
    disconnect() / stop() ask the dashboard for STOP (neutral + relay open, Policy off)."""

    def __init__(self, cfg: RigMachineConfig, dashboard_url=DEFAULT_DASHBOARD_URL):
        super().__init__(cfg)
        self.dash = dashboard_url.rstrip("/")
        self.allowed = self.policy_on = False

    def _dash(self, path, payload):
        req = urllib.request.Request(self.dash + path, data=json.dumps(payload, allow_nan=False).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"dashboard {path}: {json.loads(e.read() or b'{}').get('error', e)}") from e

    def connect(self):
        super().connect()                        # /state (read) + cameras; nothing is written
        with urllib.request.urlopen(self.dash + "/api/state", timeout=self.cfg.timeout_s) as r:
            pol = json.loads(r.read()).get("policy")
        if pol is None:
            raise SystemExit(f"{self.dash} has no Policy mode: update pi/act-athon/dashboard.py on the Pi")
        print(f"dashboard {self.dash}: Policy {'on' if pol['on'] else 'off (switch it on in the page)'}")

    def send_action(self, action: dict) -> dict:
        """POST the clipped action to the dashboard (every tick, also neutral ones: that is the runner heartbeat).
        Returns what the dashboard passed on: the action if allowed, else neutral."""
        a = {j: float(np.clip(action[f"{j}.cmd"], -1.0, 1.0)) for j in self.cfg.joint_names}
        self.last_reply = self._dash("/api/policy/act", {"cmd": a})
        self.allowed, self.policy_on = bool(self.last_reply.get("allowed")), bool(self.last_reply.get("on"))
        return {f"{j}.cmd": (v if self.allowed else 0.0) for j, v in a.items()}

    def disconnect(self):
        try:
            if self.connected:
                self.last_reply = self._dash("/api/stop", {})
        finally:
            for cap in self.caps.values():
                cap.release()
            self.caps.clear()
            self.connected = False

    def stop(self):
        self.last_reply = self._dash("/api/stop", {})
        return self.last_reply
