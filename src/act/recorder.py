"""DAgger episode recorder for scripts/control_loop.py --record-dagger (docs/dagger.md).

While the policy runner drives the excavator through the Pi dashboard, every control tick is recorded in the Pi
dashboard recorder's raw episode format (pi/act-athon/dashboard.py, Record panel: "act-athon-pi-episode/1"), so
scripts/convert_pi_episodes.py converts DAgger episodes unchanged:

    <episodes-dir>/<YYYYmmdd_HHMMSS>_<slug>.partial/   renamed without .partial when the episode is saved
        meta.json      the recorder's fields + "dagger": true, "policy": {...}, "control_counts", "aborted"
        steps.jsonl    one line per tick: i, t, mono, wall, teleop, action{5}, state{gate /state}, cams{...}, sync{...}
                       + control ("policy" | "human" | "idle"), policy_action, posted, cab_active,
                       cab_ineffective, relay_closed, policy_on,
                       phase (--subtasks), latch_active (--grip-latch)
        <cam>/<i>.jpg  the camsync-picked frame the policy saw this tick: camview's JPEG bytes, not re-encoded

Per tick (control_label):
    policy  the dashboard passed the runner's action to the gate (Policy on + deadman held -> reply "allowed")
    human   not allowed AND the relay is OPEN (gate /state "armed" false: after a dashboard STOP) AND the cab joysticks
            are active (any |cab_action| > 0 after the recorder's cab map + deadband)
    idle    anything else (nobody is driving). Verified on the machine (2026-09-26): the cab joysticks do NOT drive
            the machine while the code path holds the relay closed ("armed" true), so a deadman release alone does not
            hand control to the cab: cab input seen in /state controls then is recorded as idle (step
            cab_ineffective = true, meta cab_steps_relay_closed). Takeover = STOP in the dashboard, then the cab.
            A /state without "armed" counts as relay closed (never guess "human").
Recorded "action" = the command that actually drove the machine this tick: the action the dashboard passed on (policy),
the cab action (human), zeros (idle). "policy_action" = what the policy wanted (the controller's planned action; null
when its queue had no action for the tick).

The converter re-derives every step's action from state.controls with meta cab_map (it ignores step "action"). A
policy step's cab controls are at rest, so the recorder cab map would give zeros there. So each step's state.controls
keeps the gate's raw fields unchanged AND gets dagger_<ch> = the recorded action, and meta cab_map points at those
fields with deadband 0: the converter then reproduces exactly the recorded action for every step. The real cab map
used for human steps is in meta human_cab_map / human_cab_deadband. Never convert DAgger episodes with --cab-map.

An episode opens at the first tick that is not idle (the user engaged the policy or drove the cab) and is saved on
end(): quit, the split file, or an error (then meta "aborted": true, still saved). Idle ticks inside an
open episode are recorded. All file I/O happens on a writer thread; a step is DROPPED (counted, never blocks the
control loop) when more than `max_pending` steps wait for the disk.
"""
import json
import math
import os
import queue
import re
import socket
import threading
import time
import urllib.parse

from act.machine import CHANNELS

FORMAT = "act-athon-pi-episode/1"
SIGNS = {"swing": "+ = house right", "boom": "+ = down", "arm": "+ = out", "bucket": "+ = dump/open",
         "thumb": "+ = clamp in"}
# ---- same values / code as pi/act-athon/dashboard.py (CAB_MAP, CAB_DEADBAND, cab_action, SLOTS, TRACKS); keep in sync
CAB_MAP = {"swing": ("body", +1.0), "boom": ("boom", +1.0), "arm": ("arm", +1.0), "bucket": ("bucket", +1.0),
           "thumb": ("thumb", +1.0)}
CAB_DEADBAND = 0.08
SLOTS = {"front": "base_0_rgb", "wrist": "left_wrist_0_rgb", "side": "right_wrist_0_rgb", "world": "third_person_rgb"}
TRACKS = ("left_track", "right_track")
TRACK_WARN = 0.1
TO_API = {"swing": "body"}
DAGGER_FIELD = "dagger_{}"          # state.controls.dagger_<ch> = the recorded action (see module doc)
CONTROLS = ("policy", "human", "idle")


def human_cab_map():
    """The recorder's cab map in the dashboard's merged form: ({ch: [field, sign, offset]}, {ch: deadband})."""
    return ({ch: [f, s, 0.0] for ch, (f, s) in CAB_MAP.items()}, {ch: CAB_DEADBAND for ch in CHANNELS})


def cab_action(controls, cmap, deadband):
    """Raw /state `controls` -> {channel: -1..1} (rig signs). A missing / non-numeric field gives 0.
    Same code as dashboard.py / convert_pi_episodes.py cab_action."""
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


def control_label(allowed, cab, relay_closed):
    """allowed: the dashboard passed this tick's policy action to the gate; cab: this tick's cab action (deadbanded);
    relay_closed: gate /state "armed" (the code path holds the relay: the cab can't drive)."""
    if allowed:
        return "policy"
    return "human" if not relay_closed and any(v != 0.0 for v in cab.values()) else "idle"


def relay_closed(raw_state):
    """Gate /state -> True if the code path holds the relay ("armed"); a /state without "armed" counts as closed.
    None (no /state) -> False."""
    if raw_state is None:
        return False
    return bool(raw_state.get("armed", True))


def tracks_moving(controls):
    out = []
    for k in TRACKS:
        v = (controls or {}).get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and abs(v) > TRACK_WARN:
            out.append(k)
    return out


def gate_state(raw, poll_age):
    """Raw gate /state dict (RigMachine.last_state) -> the recorder's step "state" (dashboard parse_state fields)."""
    if not isinstance(raw, dict):
        return {"ok": False, "poll_age": poll_age, "error": "no /state"}
    c = raw.get("controls") if isinstance(raw.get("controls"), dict) else {}
    sent = raw.get("sent") if isinstance(raw.get("sent"), dict) else {}
    last = raw.get("last_cmd") if isinstance(raw.get("last_cmd"), dict) else {}
    return {"ok": True, "poll_age": poll_age, "error": None, "controls": dict(c),
            "sent": {k: float(sent.get(TO_API.get(k, k)) or 0.0) for k in CHANNELS},
            "last_cmd": {k: float(last.get(TO_API.get(k, k)) or 0.0) for k in CHANNELS},
            "armed": bool(raw.get("armed")), "veto": raw.get("veto") or "", "throttle": c.get("throttle"),
            "rabbit_button": c.get("rabbit_button"), "age": c.get("age")}


def jpeg_wh(b):
    """(width, height) from a JPEG's SOF segment, or None (dashboard.py jpeg_wh)."""
    i, n = 2, len(b)
    while i + 9 <= n and b[i] == 0xFF:
        m = b[i + 1]
        if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            return (b[i + 7] << 8) | b[i + 8], (b[i + 5] << 8) | b[i + 6]
        if m == 0xDA:
            return None
        i += 2 + ((b[i + 2] << 8) | b[i + 3])
    return None


def frame_wh(data):
    if isinstance(data, (bytes, bytearray)):
        return jpeg_wh(data) or (None, None)
    return int(data.shape[1]), int(data.shape[0])


def camview_id(src):
    """http://127.0.0.1:8100/stream/<id> -> <id>; anything else -> str(src)."""
    if isinstance(src, str) and "/stream/" in src:
        return urllib.parse.unquote(urllib.parse.urlsplit(src).path.split("/stream/", 1)[1])
    return str(src)


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40] or "episode"


def skew_summary(skews):
    if not skews:
        return None
    xs = sorted(skews)
    return {"p50": round(xs[len(xs) // 2] * 1e3, 1), "p95": round(xs[min(len(xs) - 1, int(0.95 * len(xs)))] * 1e3, 1),
            "max": round(xs[-1] * 1e3, 1), "n": len(xs)}


class Frame:
    """One camera's frame for a step: data = JPEG bytes (camview, written as is); arrival = time.monotonic() of arrival; cap = capture time (wall s); seq; age = arrival age (s)."""
    __slots__ = ("data", "arrival", "cap", "seq", "age")

    def __init__(self, data, arrival, cap, seq, age):
        self.data, self.arrival, self.cap, self.seq, self.age = data, arrival, cap, seq, age


class DaggerRecorder:
    def __init__(self, episodes_dir, fps, cameras, policy_info, sync_mode=None, max_pending=300, log=print):
        """cameras: {name: source} (the --cam specs); policy_info: meta "policy" block."""
        self.base = os.path.expanduser(episodes_dir)
        self.fps, self.cameras, self.policy_info = fps, dict(cameras), dict(policy_info)
        self.sync_mode, self.max_pending, self.log = sync_mode, max_pending, log
        self.q = queue.Queue()                  # unbounded: the bound is on steps (self.pending), never on open/close
        self.lock = threading.Lock()
        self.pending = 0
        self.write_errors, self.last_error = 0, None
        self.ep = None                          # the open episode (dict) or None
        self.saved = []                         # [(name, n_steps, how)] of this run
        self.total_dropped = 0
        self.thread = threading.Thread(target=self._writer, daemon=True, name="dagger-writer")
        self.thread.start()

    # ------------------------------------------------------------ loop thread
    def tick(self, *, prompt, now, wall, poll_age, raw_state, allowed, policy_on, drove, cab, policy_action, posted,
             frames, sync=None, phase=None, latch_active=None):
        """Record one control tick. drove / cab / policy_action / posted: {ch: float} (policy_action may be None);
        frames: {camera: Frame}. Returns the control label. Never blocks on disk."""
        closed = relay_closed(raw_state)
        cab_on = any(v != 0.0 for v in cab.values())
        control = control_label(allowed, cab, closed)
        if self.ep is None:
            if control == "idle":
                return control
            self._open(prompt, raw_state, frames, now, wall)
        ep = self.ep
        i = ep["n"]
        if i == 0:
            ep["t0"] = now
        elif now - ep["last_mono"] > 1.5 / self.fps:
            ep["late"] += 1                      # converter: a gap > 1.5 periods rejects the episode
        ep["last_mono"] = now
        state = gate_state(raw_state, round(poll_age, 4))
        state["controls"].update({DAGGER_FIELD.format(ch): float(drove[ch]) for ch in CHANNELS})
        tr = tracks_moving(state["controls"])
        ep["track_steps"] += bool(tr)
        step = {"i": i, "t": round(now - ep["t0"], 4), "mono": round(now, 4), "wall": round(wall, 4),
                "teleop": False, "action": {ch: float(drove[ch]) for ch in CHANNELS}, "state": state, "cams": {},
                "control": control, "cab_active": cab_on, "relay_closed": closed,
                "cab_ineffective": bool(cab_on and closed and not allowed), "policy_on": policy_on,
                "policy_action": None if policy_action is None else {ch: round(float(policy_action[ch]), 6)
                                                                    for ch in CHANNELS},
                "posted": {ch: round(float(posted[ch]), 6) for ch in CHANNELS}}
        if sync is not None:
            step["sync"] = {"t_ref": None if sync.get("t_ref") is None else round(sync["t_ref"], 4),
                            "anchor": sync.get("anchor"), "skew": round(sync.get("skew", 0.0), 4),
                            "left_out": list(sync.get("left_out", []))}
            if len(frames) >= 2 and not sync.get("left_out"):
                ep["skews"].append(sync.get("skew", 0.0))
        if phase is not None:
            step["phase"] = phase
        if latch_active is not None:
            step["latch_active"] = bool(latch_active)
        files = []
        for name in ep["cams"]:
            f = frames.get(name)
            if f is None:
                ep["missing"][name] += 1
                step["cams"][name] = {"file": None, "status": "no frame"}
                continue
            if f.seq is not None and f.seq == ep["last_seq"][name]:
                ep["repeated"][name] += 1
            ep["last_seq"][name] = f.seq
            rel = "%s/%06d.jpg" % (name, i)
            arr_wall = wall - (now - f.arrival)
            step["cams"][name] = {"file": rel, "seq": f.seq, "t": round(f.arrival - ep["t0"], 4),
                                  "mono": round(f.arrival, 4), "wall": round(arr_wall, 4), "age": round(f.age, 4),
                                  "cap": round(f.cap, 4), "lag": round(wall - f.cap, 4),
                                  "bytes": len(f.data) if isinstance(f.data, (bytes, bytearray)) else None}
            files.append((rel, f.data))
        with self.lock:
            full = self.pending >= self.max_pending
            if not full:
                self.pending += 1
        if full:                                  # the disk can't keep up: drop the recording step, never the tick
            ep["dropped"] += 1
            self.total_dropped += 1
            return control
        self.q.put(("step", ep["path"], step, files))
        ep["n"] += 1
        ep["counts"][control] += 1
        if cab_on and closed and not allowed:
            ep["cab_closed"] += 1                # cab moved but the relay was closed (no STOP): recorded as idle
        return control

    def _open(self, prompt, raw_state, frames, now, wall):
        os.makedirs(self.base, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name, k = "%s_%s" % (stamp, slug(prompt)), 1
        while os.path.exists(os.path.join(self.base, name)) or os.path.exists(os.path.join(self.base, name + ".partial")):
            k += 1
            name = "%s_%s_%d" % (stamp, slug(prompt), k)
        path = os.path.join(self.base, name + ".partial")
        os.makedirs(path)                           # claimed now (same-second names); the rest is the writer's job
        cams = list(self.cameras)
        hmap, hdb = human_cab_map()
        cam_meta = {}
        for c in cams:
            w, h = frame_wh(frames[c].data) if c in frames else (None, None)
            src = self.cameras[c]
            cam_meta[c] = {"camview_id": camview_id(src), "cam_arg": str(src), "label": None, "group": None,
                           "width": w, "height": h, "fps_at_start": None, "slot": SLOTS.get(c)}
        g = gate_state(raw_state, None) if raw_state is not None else {}
        meta = {"format": FORMAT, "task": prompt, "fps": self.fps, "episode_s": None, "channels": list(CHANNELS),
                "signs": SIGNS,
                "action_source": "cab",
                "action": "DAgger (control_loop.py --record-dagger): the command that drove the machine each tick -- "
                          "control policy: the runner's action the dashboard passed to the gate; human: "
                          "cab_action(state.controls, human_cab_map, human_cab_deadband); idle: 0. Also stored as "
                          "state.controls.dagger_<ch>, which cab_map below reads (deadband 0), so "
                          "convert_pi_episodes.py re-derives exactly this action. Do NOT convert with --cab-map.",
                "cab_map": {ch: [DAGGER_FIELD.format(ch), 1.0, 0.0] for ch in CHANNELS},
                "cab_deadband": {ch: 0.0 for ch in CHANNELS}, "cab_map_file": None,
                "cab_map_status": {ch: "dagger" for ch in CHANNELS},
                "human_cab_map": hmap, "human_cab_deadband": hdb,
                "state": "gate /state as read by the runner this tick: controls (raw, all fields, + dagger_<ch>), "
                         "sent (= gate output), last_cmd, armed, veto, throttle, rabbit_button, age",
                "cameras": cam_meta, "camera_slots": SLOTS,
                "camera_set": {c: [v["camview_id"], v["width"], v["height"]] for c, v in cam_meta.items()},
                "sync": {"mode": self.sync_mode, "rule": "camsync.select in the runner (RigMachine._synced_frames): "
                                                         "the frames the policy saw this tick"},
                "camera_set_changed_from": None, "camview_url": None,
                "throttle_at_start": g.get("throttle"), "rabbit_at_start": g.get("rabbit_button"),
                "armed_at_start": g.get("armed"), "dashboard_version": None, "dashboard_sha1": None,
                "recorder": "scripts/control_loop.py --record-dagger (act.recorder)",
                "host": socket.gethostname(), "started_wall": wall, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "dagger": True, "policy": self.policy_info,
                "control": "per step: policy = the dashboard passed the runner's action (Policy on + deadman held); "
                           "human = not passed, relay open (gate armed false, i.e. after a dashboard STOP) and the "
                           "cab joysticks active (|cab_action| > 0); idle = anything else (cab input while the relay "
                           "is closed does not drive the machine: idle, cab_ineffective)"}
        self.ep = {"name": name, "path": path, "meta": meta, "cams": cams, "n": 0, "t0": now, "last_mono": now,
                   "late": 0, "dropped": 0, "missing": {c: 0 for c in cams}, "repeated": {c: 0 for c in cams},
                   "last_seq": {c: None for c in cams}, "track_steps": 0, "skews": [],
                   "counts": {c: 0 for c in CONTROLS}, "cab_closed": 0}
        self.q.put(("open", path, meta, cams))
        self.log(f"\nDAgger: recording {name} (control {'/'.join(CONTROLS)} per tick) -> {self.base}")

    def end(self, reason, aborted=False):
        """Save the open episode (if any): meta.json completed, folder renamed without .partial. Returns a
        threading.Event set when the writer has done it, or None if nothing was open. Never blocks."""
        ep, self.ep = self.ep, None
        if ep is None:
            return None
        meta = dict(ep["meta"])
        meta.update(n_steps=ep["n"], duration_s=round(ep["n"] / self.fps, 3), late_ticks=ep["late"],
                    dropped_steps=ep["dropped"], frames_missing=ep["missing"], state_missing=0,
                    frames_repeated=ep["repeated"], sync_skew_ms=skew_summary(ep["skews"]),
                    track_steps=ep["track_steps"], track_warn=TRACK_WARN, end=reason, ended_wall=time.time(),
                    aborted=bool(aborted), control_counts=dict(ep["counts"]),
                    cab_steps_relay_closed=ep["cab_closed"])
        how = "save" if ep["n"] > 0 else "discard"
        done = threading.Event()
        final = os.path.join(os.path.dirname(ep["path"]), ep["name"])
        self.q.put(("close", how, ep["path"], final, meta, done))
        self.saved.append((ep["name"], ep["n"], how, dict(ep["counts"]), ep["dropped"], ep["late"]))
        self.log(f"\nDAgger: {'saved' if how == 'save' else 'discarded (no steps)'} {ep['name']} ({reason}"
                 f"{', ABORTED' if aborted else ''}): {ep['n']} steps {dict(ep['counts'])}, dropped {ep['dropped']}, "
                 f"late {ep['late']}")
        return done

    def close(self, reason, aborted=False, timeout=60.0):
        """End the open episode and wait (up to timeout) for the writer to finish every queued file."""
        self.end(reason, aborted)
        done = threading.Event()
        self.q.put(("stop", done))
        if not done.wait(timeout):
            self.log(f"\nDAgger: writer still busy after {timeout:.0f} s ({self.pending} steps queued): the last "
                     "episode may be incomplete (still .partial)")
        return done.is_set()

    def short(self):
        ep = self.ep
        if ep is None:
            return "rec: waiting to engage"
        return (f"rec {ep['n']} (p{ep['counts']['policy']} h{ep['counts']['human']} i{ep['counts']['idle']}"
                + (f" DROP {ep['dropped']}" if ep["dropped"] else "") + ")")

    # ------------------------------------------------------------ writer thread
    def _writer(self):
        fh, fh_path = None, None
        while True:
            item = self.q.get()
            kind = item[0]
            try:
                if kind == "step":
                    _, path, step, files = item
                    try:
                        for rel, data in files:
                            with open(os.path.join(path, rel), "wb") as f:
                                f.write(data)
                        if fh_path != path:
                            if fh is not None:
                                fh.close()
                            fh, fh_path = open(os.path.join(path, "steps.jsonl"), "a"), path
                        fh.write(json.dumps(step, separators=(",", ":")) + "\n")
                        fh.flush()
                    finally:
                        with self.lock:
                            self.pending -= 1
                elif kind == "open":
                    _, path, meta, cams = item
                    for c in cams:
                        os.makedirs(os.path.join(path, c), exist_ok=True)
                    with open(os.path.join(path, "meta.json"), "w") as f:
                        json.dump(meta, f, indent=1)
                elif kind == "close":
                    _, how, path, final, meta, done = item
                    try:
                        if fh is not None and fh_path == path:
                            fh.close()
                            fh, fh_path = None, None
                        if how == "discard":
                            for root, dirs, fs in os.walk(path, topdown=False):
                                for x in fs:
                                    os.remove(os.path.join(root, x))
                                for x in dirs:
                                    os.rmdir(os.path.join(root, x))
                            os.rmdir(path)
                        else:
                            with open(os.path.join(path, "meta.json"), "w") as f:
                                json.dump(meta, f, indent=1)
                            os.rename(path, final)
                    finally:
                        done.set()
                elif kind == "stop":
                    if fh is not None:
                        fh.close()
                    item[1].set()
                    return
            except Exception as ex:           # disk full / permissions: counted and shown, the loop keeps running
                self.write_errors += 1
                self.last_error = f"{type(ex).__name__}: {ex}"
