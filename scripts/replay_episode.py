"""Open-loop replay of a recorded cab episode through the Pi dashboard's Policy mode (the gate), to check that a value
sent through the GATE moves the arm like the same CAB joystick position (speed, dead band, partial-stick response).

    python replay_episode.py --episode ~/act-athon/episodes/<episode>                       # whole episode
    python replay_episode.py --episode <ep> --from-s 6.5 --to-s 12.5 --channels thumb        # grip only

Runs on the Pi from ~/act-athon/runner (PYTHONPATH=., the runner's act/ package), like control_loop.py --machine
dashboard: every action goes to the dashboard's POST /api/policy/act (DashboardMachine), which writes it to the gate
only while a human holds the deadman in the dashboard page (Policy on + SPACE / L1 / R1). Nothing here writes to the
gate directly; the gate's /state is only read (readback + liveness).

Playback: one recorded action per tick at the recorded step times (15 Hz), posted every tick (neutral too: that is the
runner heartbeat the dashboard watchdog expects). The trajectory ADVANCES ONLY while the dashboard says `allowed`:
  * released (allowed = False): the step that was just posted was NOT written (the dashboard wrote neutral instead),
    so it is not counted; from then on neutral is posted every tick and the position stays where it is;
  * re-engaged (a neutral comes back allowed): the next tick resumes at that same step.
The machine starts released: the first recorded action is posted only after the dashboard confirmed the deadman.

The recording has NO joint positions: the operator must put the machine in the recorded start pose by eye (compare
with the episode's front/ and world/ frames of the first step; the paths are printed) before typing "go".

Exit (end of trajectory, Ctrl-C, SIGTERM, any exception, gate /state unreachable or stale): one neutral, then
DashboardMachine.disconnect() = dashboard STOP (neutral + relay open, Policy off).

Log: <log-dir>/replay_<time>.csv, one row per tick: recorded action, posted action, what the dashboard passed on,
allowed, and the gate readback (/state `sent`, RigMachine's {ch}.cmd), so recorded vs actually output can be compared.
--cam name=url (optional, e.g. camview streams) saves JPEG snapshots every --snap-every s to <log-dir>/replay_<time>/;
the replay never depends on them (a camera that fails is dropped with a warning).
"""
import argparse
import csv
import json
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from act.cameras import open_camera
from act.machine import CHANNELS
from act.rig_machine import DEFAULT_DASHBOARD_URL, DEFAULT_RIG_URL, make_machine

MAX_SCALE = 1.5
ACTIVE = 0.3            # |action| above this counts as "stick in use" in the pre-run summary
MIN_DT, MAX_DT = 0.02, 0.2   # tick period clamp: the gate neutrals 0.5 s after a command, the dashboard after 0.3 s


def load_episode(ep_dir):
    """-> (meta dict, [step dicts]) from <ep>/meta.json + <ep>/steps.jsonl."""
    ep = Path(ep_dir).expanduser()
    steps_path = ep / "steps.jsonl"
    if not steps_path.is_file():
        raise SystemExit(f"{ep}: no steps.jsonl (want a raw episode folder: meta.json, steps.jsonl, front/, ...)")
    meta_path = ep / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    steps = [json.loads(line) for line in steps_path.read_text().splitlines() if line.strip()]
    if not steps:
        raise SystemExit(f"{steps_path}: empty")
    for s in steps:
        if not isinstance(s.get("action"), dict) or "t" not in s:
            raise SystemExit(f"{steps_path}: step {s.get('i')} has no 't' / 'action'")
    return meta, steps


def select_window(steps, from_s, to_s):
    lo = -np.inf if from_s is None else from_s
    hi = np.inf if to_s is None else to_s
    win = [s for s in steps if lo <= s["t"] <= hi]
    if not win:
        raise SystemExit(f"no steps in the window {from_s}..{to_s} s (episode t = {steps[0]['t']}..{steps[-1]['t']})")
    return win


def shape_action(rec, channels, scale):
    """Recorded {ch: v} -> the command to post: channels not in `channels` -> 0, the rest * scale, clipped to -1..1."""
    return {ch: (float(np.clip(float(rec.get(ch, 0.0)) * scale, -1.0, 1.0)) if ch in channels else 0.0)
            for ch in CHANNELS}


def step_periods(win):
    """Per step: how long it is held = recorded t[k+1] - t[k] (last: the median), clamped to MIN_DT..MAX_DT."""
    t = np.array([s["t"] for s in win], float)
    d = np.diff(t)
    med = float(np.median(d)) if len(d) else 1.0 / 15
    d = np.append(d, med)
    return np.clip(d, MIN_DT, MAX_DT), float(np.clip(med, MIN_DT, MAX_DT))


def active_runs(win, ch, thr=ACTIVE):
    """[(t_start, t_end, peak)] where |recorded action[ch]| > thr."""
    runs, cur = [], None
    for s in win:
        a = float(s["action"].get(ch, 0.0))
        if abs(a) > thr:
            if cur is None:
                cur = [s["t"], s["t"], a]
            cur[1] = s["t"]
            if abs(a) > abs(cur[2]):
                cur[2] = a
        elif cur is not None:
            runs.append(tuple(cur))
            cur = None
    if cur is not None:
        runs.append(tuple(cur))
    return runs


def print_summary(ep_dir, meta, steps, win, channels, scale, machine):
    ep = Path(ep_dir).expanduser()
    first = win[0]
    signs = meta.get("signs") or {}
    print(f"episode   {ep}")
    print(f"task      {meta.get('task', '?')!r}")
    print(f"recorded  {len(steps)} steps, t {steps[0]['t']:.2f}..{steps[-1]['t']:.2f} s, fps {meta.get('fps', '?')}, "
          f"action source {meta.get('action_source', '?')}")
    print(f"window    steps i={first['i']}..{win[-1]['i']} ({len(win)} steps), t {first['t']:.2f}..{win[-1]['t']:.2f} s "
          f"({win[-1]['t'] - first['t']:.1f} s at the recorded rate)")
    print(f"channels  {', '.join(ch for ch in CHANNELS if ch in channels)} (others neutral), scale {scale:g}")
    for ch in CHANNELS:
        if ch not in channels:
            continue
        runs = active_runs(win, ch)
        desc = ", ".join(f"{a:.1f}-{b:.1f}s peak {p:+.2f}" for a, b, p in runs) or "none"
        print(f"  {ch:6s} ({signs.get(ch, '')}): |a|>{ACTIVE} at {desc}")
    print(f"machine   {machine}")
    print()
    print("START POSE: the recording has NO joint positions. Before 'go', put the machine BY EYE in the pose of the")
    print(f"window's first step (i={first['i']}, t={first['t']:.2f} s) -- compare with:")
    for cam in ("front", "world", "wrist"):
        p = ep / cam / f"{int(first['i']):06d}.jpg"
        if p.is_file():
            print(f"    {p}")
    print("  recorded action at that step: " + " ".join(f"{ch} {float(first['action'].get(ch, 0)):+.2f}" for ch in CHANNELS))
    print()


def confirm(auto_yes):
    if auto_yes:
        print("--yes: starting without the 'go' prompt (motion still needs the dashboard deadman)")
        return True
    try:
        ans = input('type "go" to start (anything else aborts): ').strip().lower()
    except EOFError:
        ans = ""
    return ans == "go"


def open_snap_cameras(specs):
    """--cam name=src -> {name: camera}; a camera that fails to open is dropped with a warning (never fatal)."""
    cams = {}
    for spec in specs or []:
        name, sep, src = spec.partition("=")
        if not sep or not name or not src:
            print(f"WARNING: --cam {spec!r}: want name=url|index, ignored")
            continue
        try:
            cams[name] = open_camera(int(src) if src.isdigit() else src, (480, 640))
        except Exception as e:  # noqa: BLE001 -- cameras are optional
            print(f"WARNING: camera {name} ({src}) failed to open, no snapshots from it: {e}")
    return cams


def snapshot(cams, snap_dir, step_i, elapsed):
    """Save each camera's newest frame; returns the saved file names ('|'-joined). Failures are logged, not raised."""
    saved = []
    for name, cam in cams.items():
        try:
            frame = cam.read(max_age_s=2.0)
            fn = f"{name}_{elapsed:07.2f}s_i{step_i:06d}.jpg"
            cv2.imwrite(str(snap_dir / fn), frame)
            saved.append(fn)
        except Exception as e:  # noqa: BLE001
            print(f"\nWARNING: snapshot {name}: {e}")
    return "|".join(saved)


class StaleState(RuntimeError):
    pass


def readback(robot, max_state_age):
    """-> ({ch: gate output}, live, veto). Dashboard: the gate's /state (RigMachine.read_state: {ch}.cmd = `sent`);
    unreachable -> raises (the run stops); controls older than max_state_age -> StaleState."""
    st = robot.read_state()
    age = st.get("age_s")
    if max_state_age > 0 and isinstance(age, (int, float)) and age > max_state_age:
        raise StaleState(f"gate /state is stale: controls age {age:.2f} s > {max_state_age} s")
    return {ch: float(st[f"{ch}.cmd"]) for ch in CHANNELS}, st.get("live"), st.get("veto", "")


def fmt(a):
    return " ".join(f"{ch[:2]}{a[ch]:+.2f}" for ch in CHANNELS)


def on_sigterm(signum, frame):
    raise KeyboardInterrupt


def run(args):
    if not (0.0 < args.scale <= MAX_SCALE):
        raise SystemExit(f"--scale {args.scale}: refused (must be > 0 and <= {MAX_SCALE})")
    channels = set(CHANNELS)
    if args.channels:
        channels = {c.strip() for c in args.channels.split(",") if c.strip()}
        bad = sorted(channels - set(CHANNELS))
        if bad or not channels:
            raise SystemExit(f"--channels {args.channels!r}: unknown {bad}; choose from {','.join(CHANNELS)}")
    if args.from_s is not None and args.to_s is not None and args.to_s < args.from_s:
        raise SystemExit("--to-s must be >= --from-s")
    meta, steps = load_episode(args.episode)
    win = select_window(steps, args.from_s, args.to_s)
    periods, idle_period = step_periods(win)

    signal.signal(signal.SIGTERM, on_sigterm)
    robot = make_machine("dashboard", None, args.rig_url, dashboard_url=args.dashboard_url)  # no cameras here
    where = f"dashboard {args.dashboard_url} (gate /state read from {args.rig_url})"
    print_summary(args.episode, meta, steps, win, channels, args.scale, where)

    log_dir = Path(args.log_dir)
    stamp = time.strftime("%m%d_%H%M%S")
    cams, snap_dir = {}, None
    neutral = {ch: 0.0 for ch in CHANNELS}
    rows = 0
    stats = {"ticks": 0, "live_ticks": 0, "released_ticks": 0, "advanced": 0, "late": 0}
    diffs = []            # advanced ticks: gate readback - posted
    log_file = None
    ended = "?"
    started = False                # True once "go" was given: only then is a final neutral posted
    try:
        robot.connect()            # reads only (gate /state, dashboard /api/state); nothing is written
        if args.cam:               # optional, before the prompt (opening can take a few s); failures only warn
            cams = open_snap_cameras(args.cam)
            snap_dir = log_dir / f"replay_{stamp}"
            print(f"snapshots every {args.snap_every} s from {sorted(cams) or 'no camera'} -> {snap_dir}")
        if not confirm(args.yes):
            ended = "not confirmed (no 'go')"
            print("aborted: nothing was sent")
            return 2
        started = True
        log_dir.mkdir(parents=True, exist_ok=True)
        if cams:
            snap_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"replay_{stamp}.csv"
        log_file = open(log_path, "w", newline="")
        w = csv.writer(log_file)
        w.writerow(["wall", "t_run", "pos", "n", "ep_i", "rec_t", "state", "allowed", "policy_on", "advanced"]
                   + [f"rec_{c}" for c in CHANNELS] + [f"sent_{c}" for c in CHANNELS]
                   + [f"passed_{c}" for c in CHANNELS] + [f"gate_{c}" for c in CHANNELS]
                   + ["gate_live", "gate_veto", "snap"])
        print(f"log: {log_path}")
        print("hold the deadman in the dashboard page (Policy on + SPACE / L1 / R1) to play; release to pause")

        n = len(win)
        pos = 0
        live = False                # start released: the first recorded action waits for a confirmed deadman
        t0 = time.monotonic()
        next_snap = 0.0
        deadline = t0               # absolute tick schedule (no drift from sleep overshoot)
        while pos < n:
            tick = time.monotonic()
            step = win[pos]
            rec = {ch: float(step["action"].get(ch, 0.0)) for ch in CHANNELS}
            cmd = shape_action(rec, channels, args.scale) if live else dict(neutral)
            passed = robot.send_action({f"{ch}.cmd": v for ch, v in cmd.items()})
            allowed = bool(robot.allowed)
            policy_on = bool(robot.policy_on)
            advanced = live and allowed
            if live and not allowed:
                print(f"\nRELEASED at step {pos + 1}/{n} (t={step['t']:.2f} s): that step was not written; "
                      "paused, sending neutral" + ("" if policy_on else " (Policy is OFF in the page)"))
            elif not live and allowed:
                print(f"\nLIVE: deadman held, resuming at step {pos + 1}/{n} (t={step['t']:.2f} s)")
            gate, gate_live, veto = readback(robot, args.max_state_age)
            elapsed = tick - t0
            snap = ""
            if cams and elapsed >= next_snap:
                snap = snapshot(cams, snap_dir, int(step.get("i", pos)), elapsed)
                next_snap = elapsed + args.snap_every
            state = "LIVE" if live else "released"
            w.writerow([round(time.time(), 3), round(elapsed, 3), pos, n, step.get("i", pos), step["t"], state,
                        int(allowed), int(policy_on), int(advanced)]
                       + [round(rec[c], 4) for c in CHANNELS] + [round(cmd[c], 4) for c in CHANNELS]
                       + [round(float(passed[f"{c}.cmd"]), 4) for c in CHANNELS] + [round(gate[c], 4) for c in CHANNELS]
                       + [gate_live, veto, snap])
            log_file.flush()
            rows += 1
            stats["ticks"] += 1
            stats["live_ticks" if live else "released_ticks"] += 1
            if advanced:
                diffs.append([gate[c] - cmd[c] for c in CHANNELS])
                stats["advanced"] += 1
            print(f"\r[{'LIVE    ' if advanced else 'released'}] step {pos + 1:4d}/{n} t={step['t']:6.2f}s "
                  f"allowed={int(allowed)} sent {fmt(cmd)} | gate {fmt(gate)} live={gate_live} veto={veto!r}   ",
                  end="", flush=True)
            period = periods[pos] if advanced else idle_period
            if advanced:
                pos += 1
            live = allowed
            deadline += period
            spare = deadline - time.monotonic()
            stats["late"] += spare < 0
            if spare < -period:     # far behind (slow HTTP): don't burst to catch up, restart the schedule
                deadline = time.monotonic()
            if pos < n:
                time.sleep(max(0.0, spare))
        ended = "end of trajectory"
        return 0
    except KeyboardInterrupt:
        ended = "Ctrl-C / SIGTERM"
        return 130
    except Exception as e:  # noqa: BLE001 -- any failure: neutral + STOP in finally, then report
        ended = f"error: {type(e).__name__}: {e}"
        return 1
    finally:
        print(f"\n\nstopping ({ended}): " + ("neutral + STOP" if started else "STOP"))
        if started:
            try:
                robot.send_action({f"{ch}.cmd": 0.0 for ch in CHANNELS})
            except Exception as e:  # noqa: BLE001
                print(f"final neutral failed: {e}")
        try:
            if robot.connected:
                robot.disconnect()                    # dashboard: POST /api/stop (neutral + relay open, Policy off)
                print(f"dashboard STOP sent: {robot.last_reply}")
        except Exception as e:  # noqa: BLE001
            print(f"!!! dashboard STOP FAILED: {e} -- press STOP in the page / use the kill switch")
        for cam in cams.values():
            try:
                cam.release()
            except Exception:  # noqa: BLE001
                pass
        if log_file is not None:
            log_file.close()
            d = np.abs(np.array(diffs)) if diffs else None
            print(f"ticks {stats['ticks']} (live {stats['live_ticks']}, released {stats['released_ticks']}), "
                  f"steps played {stats['advanced']}/{len(win)}, late ticks {stats['late']}, rows {rows}")
            if d is not None:
                print("gate readback vs sent on played steps, mean |diff| per channel: "
                      + " ".join(f"{c} {m:.3f}" for c, m in zip(CHANNELS, d.mean(axis=0)))
                      + f" (max {d.max():.3f})")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episode", required=True, help="raw episode folder (meta.json + steps.jsonl)")
    ap.add_argument("--from-s", type=float, default=None, help="window start (recorded t, s)")
    ap.add_argument("--to-s", type=float, default=None, help="window end (recorded t, s, inclusive)")
    ap.add_argument("--channels", default=None,
                    help=f"comma list of channels to replay ({','.join(CHANNELS)}); others neutral. Default: all")
    ap.add_argument("--scale", type=float, default=1.0, help=f"multiply commands (clipped to -1..1; > {MAX_SCALE} refused)")
    ap.add_argument("--machine", choices=["dashboard"], default="dashboard")
    ap.add_argument("--dashboard-url", default=DEFAULT_DASHBOARD_URL)
    ap.add_argument("--rig-url", default=DEFAULT_RIG_URL, help="gate base URL, READ ONLY (/state readback)")
    ap.add_argument("--cam", action="append", help="name=url|index, optional, snapshots only (e.g. camview streams)")
    ap.add_argument("--snap-every", type=float, default=1.0, help="--cam: seconds between snapshots")
    ap.add_argument("--max-state-age", type=float, default=1.0,
                    help="stop if the gate's /state controls age exceeds this (s); 0 = no check")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--yes", action="store_true", help="skip the 'go' prompt (motion still needs the deadman)")
    return ap.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(parse_args()))
