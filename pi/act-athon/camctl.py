"""camctl -- lock USB webcam image controls (exposure, white balance, ...) and keep them locked.

V4L2 control values live in the camera and reset whenever the device re-enumerates (replug, Pi reboot, USB reset),
so they have to be re-applied. This tool reads cam_controls.json (next to this file) and writes the values with
v4l2-ctl. It only touches controls, never opens a stream, so it works while camview.py is streaming.

Config: {"<camview stream id>": {...}, "*": {...}}. "*" applies to every USB camera; a per-id entry overrides
single controls of "*" (a null value there means "leave this control alone on that camera"). Keys starting with
"_" are comments. Stream ids are camview's: "usb-" + the sysfs USB interface name (e.g. usb-3-1-1-1-0), i.e.
the physical USB port, stable across replugs into the same port; the /dev/videoN number is not.

    python3 camctl.py apply [--config PATH] [--dry-run]   # write + read back; exit 1 on any mismatch
    python3 camctl.py watch [--interval 5]                # re-apply whenever a camera's values drift
    python3 camctl.py show                                # read-only dump of every camera's controls

start_dashboard.sh runs apply once, then watch in the background for as long as dashboard.py runs.
Stdlib only. Needs v4l2-ctl (package v4l-utils) and read/write access to /dev/video* (group video).
See docs/camera-controls.md.
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "cam_controls.json")
DEFAULT_STATUS = os.path.join(HERE, "logs", "camctl_status.json")
# Same constant as camview.py; CAMCTL_SYSFS only exists so the tests can point it at a fake tree.
SYSFS_V4L = os.environ.get("CAMCTL_SYSFS", "/sys/class/video4linux")
V4L2_CTL = "v4l2-ctl"
CMD_TIMEOUT_S = 5.0          # a wedged camera must not hang apply / watch forever
FAIL_RETRY_S = 60.0          # watch: after a failed re-apply, wait this long before trying that camera again

# Controls that switch other controls between active and inactive. They must be written first: a UVC camera
# rejects writes to an inactive control (exposure_time_absolute while auto_exposure is on, white_balance_temperature
# while white_balance_automatic is on, focus_absolute while continuous autofocus is on). Old kernel names included.
MODE_FIRST = (
    "auto_exposure", "exposure_auto",
    "exposure_dynamic_framerate", "exposure_auto_priority",
    "white_balance_automatic", "white_balance_temperature_auto",
    "focus_automatic_continuous", "focus_auto",
    "gain_automatic", "autogain",
)
SKIP_NAME_PREFIXES = ("pispbe", "rpi-")    # the Pi's own ISP / HEVC / CSI nodes (also never behind a /usb path)


def log(msg):
    print("%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg), flush=True)


# ---------------------------------------------------------------- discovery (id derivation copied from camview.py)
def discover_usb_cameras():
    """Group /dev/videoN nodes by physical USB device (a camera's capture + metadata nodes share one).
    Excludes the Pi's own CSI/ISP/HEVC pipelines, which never sit behind a /usb sysfs path.
    Verbatim copy of camview.discover_usb_cameras (keep them identical: the ids must match camview's stream ids;
    the tests compare the two)."""
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


class CtlError(Exception):
    pass


def v4l2(dev, *args):
    """Run v4l2-ctl -d DEV ARGS; return (returncode, stdout, stderr). Raises CtlError if it can't run at all."""
    try:
        p = subprocess.run([V4L2_CTL, "-d", dev] + list(args), capture_output=True, text=True, timeout=CMD_TIMEOUT_S)
    except FileNotFoundError:
        raise CtlError("v4l2-ctl not found (sudo apt install v4l-utils)")
    except subprocess.TimeoutExpired:
        raise CtlError("v4l2-ctl %s timed out on %s" % (" ".join(args), dev))
    return p.returncode, p.stdout, p.stderr


_FMT_LINE = re.compile(r"^\s*\[\d+\]:")
_capture_cache = {}          # (node, sysfs realpath) -> is a capture node


def is_capture_node(node):
    """A capture node lists >= 1 pixel format; the UVC metadata sibling lists none. None = couldn't tell."""
    try:
        real = os.path.realpath(os.path.join(SYSFS_V4L, os.path.basename(node), "device"))
    except OSError:
        real = ""
    key = (node, real)
    if key in _capture_cache:
        return _capture_cache[key]
    try:
        rc, out, _ = v4l2(node, "--list-formats")
    except CtlError:
        return None
    if rc != 0:
        return None
    ok = any(_FMT_LINE.match(line) for line in out.splitlines())
    _capture_cache[key] = ok
    return ok


def cameras():
    """[{id, dev, label}] for every USB camera with a capture node, sorted by id."""
    out = []
    for cam_id, info in sorted(discover_usb_cameras().items()):
        if info["label"].startswith(SKIP_NAME_PREFIXES):
            continue
        for node in info["nodes"]:
            if is_capture_node(node):
                out.append({"id": cam_id, "dev": node, "label": info["label"]})
                break
    return out


# ---------------------------------------------------------------- controls
_CTRL_LINE = re.compile(r"^\s*([a-z0-9_]+)\s+0x[0-9a-fA-F]+\s+\((\w+)\)\s*:\s*(.*)$")


def list_controls(dev):
    """Parse `v4l2-ctl -l`: {name: {"type", "value", "min", "max", "flags"}} (value None for button controls)."""
    rc, out, err = v4l2(dev, "-l")
    if rc != 0:
        raise CtlError("v4l2-ctl -l failed on %s: %s" % (dev, (err or out).strip()[:200]))
    ctrls = {}
    for line in out.splitlines():
        m = _CTRL_LINE.match(line)
        if not m:
            continue
        name, typ, rest = m.groups()
        kv = dict(re.findall(r"(\w+)=(\S+)", rest))
        info = {"type": typ, "value": None, "min": None, "max": None,
                "flags": [f for f in kv.get("flags", "").split(",") if f]}
        for k in ("value", "min", "max"):
            try:
                info[k] = int(kv[k])
            except (KeyError, ValueError):
                pass
        ctrls[name] = info
    return ctrls


def read_controls(dev, names):
    """`v4l2-ctl -C a,b,c` -> {name: int}; names missing from the output are absent from the result."""
    if not names:
        return {}
    rc, out, err = v4l2(dev, "-C", ",".join(names))
    vals = {}
    for line in out.splitlines():
        m = re.match(r"^\s*([a-z0-9_]+)\s*:\s*(-?\d+)", line)
        if m:
            vals[m.group(1)] = int(m.group(2))
    if rc != 0 and not vals:
        raise CtlError("v4l2-ctl -C failed on %s: %s" % (dev, (err or out).strip()[:200]))
    return vals


def write_control(dev, name, value):
    """One control per call, so the order is exactly ours. Returns an error string or None. v4l2-ctl's exit code
    is not trusted either way: the read-back decides."""
    rc, out, err = v4l2(dev, "-c", "%s=%d" % (name, value))
    msg = (err or "").strip() or (out or "").strip()
    return msg[:200] if (rc != 0 or "error" in msg.lower() or "failed" in msg.lower()) else None


def ordered(names):
    """Mode controls first (MODE_FIRST order, then any other *auto* control), then the rest in config order."""
    first = [n for n in MODE_FIRST if n in names]
    autos = [n for n in names if n not in first and "auto" in n]
    return first + autos + [n for n in names if n not in first and n not in autos]


# ---------------------------------------------------------------- config
def load_config(path):
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("%s: top level must be an object {camera id: {control: value}}" % path)
    cfg = {}
    for key, ctrls in raw.items():
        if key.startswith("_"):
            continue
        if key != "*" and not key.startswith("usb-"):
            raise ValueError("%s: key %r is neither \"*\" nor a camview USB id (usb-...)" % (path, key))
        if not isinstance(ctrls, dict):
            raise ValueError("%s: %r must map control names to integers" % (path, key))
        clean = {}
        for name, val in ctrls.items():
            if name.startswith("_"):
                continue
            if val is not None and (isinstance(val, bool) or not isinstance(val, int)):
                raise ValueError("%s: %s.%s must be an integer (or null), got %r" % (path, key, name, val))
            clean[name] = val
        cfg[key] = clean
    return cfg


def desired_for(cfg, cam_id):
    """"*" merged with the camera's own entry (its values win; null removes a "*" control), in config order."""
    want = dict(cfg.get("*", {}))
    want.update(cfg.get(cam_id, {}))
    return {k: v for k, v in want.items() if v is not None}


# ---------------------------------------------------------------- apply
def apply_camera(cam, want, only=None, dry_run=False, warn_unsupported=True):
    """Write `want` (or just the names in `only`) in safe order, read everything back.
    Returns {"ok", "rows": [(name, want, got, note)], "skipped": [...], "errors": [...]}."""
    dev = cam["dev"]
    res = {"ok": True, "rows": [], "skipped": [], "errors": []}
    try:
        ctrls = list_controls(dev)
    except CtlError as e:
        res["ok"] = False
        res["errors"].append(str(e))
        return res
    supported = [n for n in want if n in ctrls]
    for n in want:
        if n not in ctrls:
            res["skipped"].append(n)
            if warn_unsupported:
                log("WARNING %s (%s): camera has no control %r -- skipped" % (cam["id"], dev, n))
    for n in supported:
        lo, hi = ctrls[n]["min"], ctrls[n]["max"]
        if lo is not None and hi is not None and not lo <= want[n] <= hi:
            log("WARNING %s: %s=%d is outside the camera's range %d..%d" % (cam["id"], n, want[n], lo, hi))
    to_write = ordered([n for n in supported if only is None or n in only])
    if dry_run:
        for n in to_write:
            log("dry-run %s (%s): would set %s=%d (now %s)" % (cam["id"], dev, n, want[n], ctrls[n]["value"]))
        for n in supported:
            res["rows"].append((n, want[n], ctrls[n]["value"], "dry-run"))
        return res
    write_err = {}
    for n in to_write:
        try:
            err = write_control(dev, n, want[n])
        except CtlError as e:
            err = str(e)
        if err:
            write_err[n] = err
    try:
        got = read_controls(dev, supported)
    except CtlError as e:
        res["ok"] = False
        res["errors"].append(str(e))
        got = {}
    for n in supported:
        g = got.get(n)
        note = "ok" if g == want[n] else ("MISMATCH" + (": " + write_err[n] if n in write_err else ""))
        if g != want[n]:
            res["ok"] = False
        res["rows"].append((n, want[n], g, note))
    return res


def print_report(cam, res):
    print("%s  %s  %s" % (cam["id"], cam["dev"], cam["label"]), flush=True)
    for n, w, g, note in res["rows"]:
        print("    %-30s want %-6s got %-6s %s" % (n, w, "?" if g is None else g, note), flush=True)
    for n in res["skipped"]:
        print("    %-30s unsupported by this camera: skipped (warning)" % n, flush=True)
    for e in res["errors"]:
        print("    ERROR %s" % e, flush=True)


def write_status(path, cam, res):
    """logs/camctl_status.json: {id: {dev, label, applied_at, ok, controls: {name: actual}, mismatched, skipped}}."""
    try:
        with open(path) as f:
            status = json.load(f)
        if not isinstance(status, dict):
            status = {}
    except (OSError, ValueError):
        status = {}
    status[cam["id"]] = {
        "dev": cam["dev"], "label": cam["label"],
        "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ok": res["ok"],
        "controls": {n: g for n, _, g, _ in res["rows"]},
        "mismatched": {n: {"want": w, "got": g} for n, w, g, note in res["rows"] if note != "ok"},
        "skipped": res["skipped"], "errors": res["errors"],
    }
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(status, f, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except OSError as e:
        log("WARNING could not write %s: %s" % (path, e))


def cmd_apply(args, cfg):
    cams = cameras()
    if not cams:
        log("WARNING no USB cameras with a capture node found -- nothing applied (watch applies them when they appear)")
        return 0
    all_ok = True
    for cam in cams:
        want = desired_for(cfg, cam["id"])
        if not want:
            print("%s  %s  %s\n    (no controls configured)" % (cam["id"], cam["dev"], cam["label"]), flush=True)
            continue
        res = apply_camera(cam, want, dry_run=args.dry_run)
        print_report(cam, res)
        if not args.dry_run:
            write_status(args.status, cam, res)
        all_ok = all_ok and res["ok"]
    if not all_ok:
        log("camctl apply: some controls did NOT take (see MISMATCH / ERROR above)")
    return 0 if all_ok else 1


def cmd_show(args, cfg):
    cams = cameras()
    if not cams:
        print("no USB cameras with a capture node found", flush=True)
        return 0
    for cam in cams:
        want = desired_for(cfg, cam["id"])
        print("%s  %s  %s" % (cam["id"], cam["dev"], cam["label"]), flush=True)
        try:
            ctrls = list_controls(cam["dev"])
        except CtlError as e:
            print("    ERROR %s" % e, flush=True)
            continue
        for n, c in ctrls.items():
            rng = "" if c["min"] is None else "%d..%d" % (c["min"], c["max"])
            cfgd = "" if n not in want else ("config %d%s" % (want[n], "" if want[n] == c["value"] else "  <-- DIFFERS"))
            print("    %-30s %-6s %-12s %-10s %s" % (n, "-" if c["value"] is None else c["value"], rng,
                                                   ",".join(c["flags"]), cfgd), flush=True)
        for n in want:
            if n not in ctrls:
                print("    %-30s (configured %d, but this camera has no such control)" % (n, want[n]), flush=True)
    return 0


# ---------------------------------------------------------------- watch
def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cmd_watch(args, cfg):
    stop = threading.Event()

    def on_signal(signum, _frame):
        log("camctl watch: got %s, exiting" % signal.Signals(signum).name)
        stop.set()

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, on_signal)
    try:
        cfg_mtime = os.stat(args.config).st_mtime
    except OSError:
        cfg_mtime = None
    log("camctl watch: every %gs, config %s, pid %d" % (args.interval, args.config, os.getpid()))
    seen = {}            # id -> dev last seen (for appear / disappear lines)
    warned = set()       # (id, dev, control) unsupported warnings already logged
    retry_at = {}        # (id, dev) -> time before which a failed camera is not retried
    last_err = {}        # id -> last read error logged (log each distinct error once)
    while not stop.is_set():
        if args.parent and not pid_alive(args.parent):
            log("camctl watch: parent %d is gone, exiting" % args.parent)
            break
        try:                                   # pick up edits to cam_controls.json without a restart
            m = os.stat(args.config).st_mtime
            if m != cfg_mtime:
                cfg_mtime = m
                cfg = load_config(args.config)
                retry_at.clear()
                log("camctl watch: reloaded %s" % args.config)
        except (OSError, ValueError) as e:
            log("WARNING camctl watch: keeping the previous config, %s unreadable: %s" % (args.config, e))
        cams = cameras()
        now_ids = {c["id"]: c["dev"] for c in cams}
        for cid, dev in now_ids.items():
            if seen.get(cid) != dev:
                log("camera %s at %s" % (cid, dev) + (" (was %s)" % seen[cid] if cid in seen else ""))
        for cid in set(seen) - set(now_ids):
            log("camera %s gone" % cid)
        seen = now_ids
        for cam in cams:
            if stop.is_set():
                break
            key = (cam["id"], cam["dev"])
            want = desired_for(cfg, cam["id"])
            if not want or time.time() < retry_at.get(key, 0):
                continue
            try:
                ctrls = list_controls(cam["dev"])
            except CtlError as e:
                if last_err.get(cam["id"]) != str(e):
                    log("WARNING %s: %s" % (cam["id"], e))
                    last_err[cam["id"]] = str(e)
                continue
            last_err.pop(cam["id"], None)
            for n in want:
                if n not in ctrls and (cam["id"], cam["dev"], n) not in warned:
                    warned.add((cam["id"], cam["dev"], n))
                    log("WARNING %s (%s): camera has no control %r -- skipped" % (cam["id"], cam["dev"], n))
            diff = [n for n in want if n in ctrls and ctrls[n]["value"] != want[n]]
            if not diff:
                continue
            log("re-apply %s (%s): %s" % (cam["id"], cam["dev"],
                                          ", ".join("%s %s->%d" % (n, ctrls[n]["value"], want[n]) for n in diff)))
            res = apply_camera(cam, want, only=diff, warn_unsupported=False)
            write_status(args.status, cam, res)
            if res["ok"]:
                log("re-apply %s: ok" % cam["id"])
                retry_at.pop(key, None)
            else:
                bad = ["%s want %s got %s" % (n, w, g) for n, w, g, note in res["rows"] if note != "ok"]
                log("WARNING re-apply %s FAILED (%s); retrying in %ds" % (cam["id"], "; ".join(bad + res["errors"]),
                                                                         FAIL_RETRY_S))
                retry_at[key] = time.time() + FAIL_RETRY_S
        stop.wait(args.interval)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, hlp in (("apply", "write the configured controls to every USB camera and read them back"),
                      ("watch", "re-apply whenever a camera's controls differ from the config"),
                      ("show", "print every camera's current control values (read-only)")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--config", default=DEFAULT_CONFIG, help="controls file (default %(default)s)")
        p.add_argument("--status", default=DEFAULT_STATUS, help="status file written after each apply")
        if name == "apply":
            p.add_argument("--dry-run", action="store_true", help="show what would be written, write nothing")
        if name == "watch":
            p.add_argument("--interval", type=float, default=5.0, help="seconds between checks (default 5)")
            p.add_argument("--parent", type=int, default=0, metavar="PID",
                           help="exit when this process (our parent, e.g. start_dashboard.sh) goes away")
    args = ap.parse_args()
    try:
        cfg = load_config(args.config)
    except (OSError, ValueError) as e:
        print("camctl: bad config: %s" % e, file=sys.stderr, flush=True)
        return 2
    if shutil.which(V4L2_CTL) is None:
        print("camctl: v4l2-ctl not found (sudo apt install v4l-utils)", file=sys.stderr, flush=True)
        return 1
    return {"apply": cmd_apply, "watch": cmd_watch, "show": cmd_show}[args.cmd](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
