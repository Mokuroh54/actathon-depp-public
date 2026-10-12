"""latency -- how old are the camera frames the recorder and the policy see? Run ON the Pi (read only: never touches
the gate, only reads MJPEG streams).

    python3 latency.py clock   http://<laptop>:8100            laptop - Pi clock offset (NTP-style, over HTTP)
    python3 latency.py measure http://<laptop>:8100 --secs 30  world camera: capture on the laptop -> arrival on the Pi
    .venv/bin/python3 latency.py snap --n 10                   stopwatch test: grab every camera at the same instant

measure: reads the laptop's /stream/world directly AND camview's relayed copy (net-world, same JPEG bytes: frames
are matched by hash) and reports, per frame, with the laptop clock mapped onto the Pi's by the measured offset:
    net     laptop capture -> arrival at this script, direct            (camera->laptop USB delay not included)
    relay   laptop capture -> arrival at this script via camview's relay (what the dashboard / runner get)
    gap     time between consecutive frames (jitter, stalls)
Results go to logs/latency_<time>.json. The recorder samples at 15 Hz, which adds on average half a frame period.

snap: holds one reader per camview stream (or --cams id,id), and every --every s saves ONE side-by-side JPEG of the
newest frame of each, stamped with the Pi time. Put a phone showing a stopwatch with milliseconds where every camera
sees it; the difference between the times shown in one snapshot is the relative delay of those cameras (a camera
showing an EARLIER time is that much older). Needs OpenCV (.venv) to compose; without it saves one JPEG per camera.
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import threading
import time
import urllib.request

try:                        # optional: only the snapshot contact sheet uses them (plain JPEGs without)
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

CAMVIEW = "http://127.0.0.1:8100"
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- MJPEG reader (Content-Length parts + headers)
class Reader:
    """Background reader of one multipart MJPEG stream: newest (jpeg, headers, arrival time) + an optional per-frame
    callback. Reconnects until stop()."""

    def __init__(self, url, on_frame=None):
        self.url, self.on_frame = url, on_frame
        self.latest, self.error, self.n = None, None, 0
        self.alive = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.alive:
            try:
                with urllib.request.urlopen(self.url, timeout=5.0) as r:
                    self.error = None
                    while self.alive:
                        part = read_part(r)
                        t = time.time()
                        if part is None:
                            continue
                        hdr, jpeg = part
                        self.n += 1
                        self.latest = (jpeg, hdr, t)
                        if self.on_frame:
                            self.on_frame(jpeg, hdr, t)
            except Exception as ex:
                self.error = "%s: %s" % (type(ex).__name__, ex)
                time.sleep(1.0)

    def stop(self):
        self.alive = False


def read_part(f):
    """Next part of a multipart stream -> (headers dict, jpeg bytes), or None for a part without Content-Length."""
    while True:
        line = f.readline(4096)
        if not line:
            raise EOFError("stream closed")
        if line.startswith(b"--"):
            break
    hdr = {}
    while True:
        line = f.readline(4096)
        if not line:
            raise EOFError("stream closed")
        if not line.strip():
            break
        k, _, v = line.decode("latin-1").partition(":")
        hdr[k.strip().lower()] = v.strip()
    n = int(hdr.get("content-length", "-1"))
    if n <= 0:
        return None
    data = f.read(n)
    if len(data) < n:
        raise EOFError("short part")
    return hdr, data


# ---------------------------------------------------------------- clock offset
def clock_offset(base, n=25):
    """laptop clock - Pi clock (s), from the probe with the smallest round trip: offset = t_laptop - (t0 + t1) / 2,
    error <= rtt / 2. -> (offset, rtt, all rtts)."""
    best, rtts = None, []
    for _ in range(n):
        t0 = time.time()
        with urllib.request.urlopen(base.rstrip("/") + "/api/time", timeout=2.0) as r:
            tl = json.loads(r.read())["t"]
        t1 = time.time()
        rtts.append(t1 - t0)
        if best is None or t1 - t0 < best[1]:
            best = (tl - (t0 + t1) / 2, t1 - t0)
        time.sleep(0.05)
    return best[0], best[1], rtts


def cmd_clock(a):
    off, rtt, rtts = clock_offset(a.laptop)
    print("laptop - Pi clock offset: %+.1f ms  (+-%.1f ms; http round trip min %.1f / median %.1f / max %.1f ms)"
          % (off * 1e3, rtt * 5e2, min(rtts) * 1e3, statistics.median(rtts) * 1e3, max(rtts) * 1e3))
    return off


# ---------------------------------------------------------------- measure
def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, round(p / 100 * (len(xs) - 1))))] if xs else float("nan")


def summary(xs):
    if not xs:
        return None
    ms = [x * 1e3 for x in xs]
    return {"n": len(ms), "mean": round(statistics.fmean(ms), 1), "p50": round(pct(ms, 50), 1),
            "p95": round(pct(ms, 95), 1), "max": round(max(ms), 1),
            "std": round(statistics.pstdev(ms), 1) if len(ms) > 1 else 0.0}


def fmt(s):
    return "-" if s is None else "p50 %6.1f  p95 %6.1f  max %6.1f  std %5.1f ms  (n=%d)" % (
        s["p50"], s["p95"], s["max"], s["std"], s["n"])


def cmd_measure(a):
    base = a.laptop.rstrip("/")
    print("clock offset ...", flush=True)
    off0, rtt0, _ = clock_offset(base)
    print("  laptop - Pi = %+.1f ms (+-%.1f ms)" % (off0 * 1e3, rtt0 * 5e2), flush=True)

    lock = threading.Lock()
    cap_by_hash = {}                     # jpeg hash -> laptop capture time (Pi clock)
    net, relay, gap_d, gap_r, relay_in = [], [], [], [], []
    last = {"d": None, "r": None}
    unmatched = [0]

    def direct(jpeg, hdr, t):
        tc = hdr.get("x-capture-time")
        if tc is None:
            return
        cap = float(tc) - off0
        with lock:
            cap_by_hash[hashlib.sha1(jpeg).digest()] = cap
            net.append(t - cap)
            if last["d"] is not None:
                gap_d.append(t - last["d"])
            last["d"] = t

    def relayed(jpeg, hdr, t):
        h = hashlib.sha1(jpeg).digest()
        deadline = time.time() + 0.5     # the direct copy of this frame may land a moment later
        while True:
            with lock:
                cap = cap_by_hash.get(h)
            if cap is not None or time.time() > deadline:
                break
            time.sleep(0.002)
        with lock:
            if last["r"] is not None:
                gap_r.append(t - last["r"])
            last["r"] = t
            if cap is None:
                unmatched[0] += 1
                return
            relay.append(t - cap)
            t_in = hdr.get("x-arrival-time") or hdr.get("x-capture-time")   # camview's arrival time (Pi clock);
            if t_in:                                                         # older camview sent it as X-Capture-Time
                relay_in.append(float(t_in) - cap)

    rd = Reader(base + "/stream/world", direct)
    rr = Reader(a.camview.rstrip("/") + "/stream/" + a.relay, relayed) if a.relay else None
    t_end = time.time() + a.secs
    while time.time() < t_end:
        time.sleep(1.0)
        with lock:
            n_d, n_r = len(net), len(relay)
            cur = summary(net[-30:])
        print("\r  %3.0f s left  direct %d frames%s  relayed %d%s   " % (
            t_end - time.time(), n_d, "" if cur is None else " (last 30: p50 %.0f ms)" % cur["p50"], n_r,
            "" if rr is None or not rr.error else " [relay: %s]" % rr.error), end="", flush=True)
    rd.stop()
    if rr:
        rr.stop()
    print()
    off1, rtt1, _ = clock_offset(base, n=10)
    res = {"laptop": base, "relay": a.relay, "secs": a.secs, "clock_offset_ms": round(off0 * 1e3, 2),
           "clock_offset_err_ms": round(rtt0 * 5e2, 2), "clock_drift_ms": round((off1 - off0) * 1e3, 2),
           "net": summary(net), "relay_client": summary(relay), "relay_arrival_at_camview": summary(relay_in),
           "gap_direct": summary(gap_d), "gap_relay": summary(gap_r), "relay_unmatched": unmatched[0],
           "direct_error": rd.error, "relay_error": rr.error if rr else None,
           "pi_time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    print("\nworld camera, laptop capture -> Pi (clock offset %+.1f ms +-%.1f, drift over the run %+.1f ms)"
          % (off0 * 1e3, rtt0 * 5e2, (off1 - off0) * 1e3))
    print("  direct from laptop   " + fmt(res["net"]))
    print("  via camview relay    " + fmt(res["relay_client"]) + "   <- what the dashboard / runner see")
    print("  frame gap (direct)   " + fmt(res["gap_direct"]))
    print("  frame gap (relay)    " + fmt(res["gap_relay"]))
    if a.relay and not relay:
        print("  relay: no matched frames (%s). Is %s in camview's list? (remote_cams.json: \"world\": \"%s/stream/world\")"
              % (rr.error or "%d unmatched" % unmatched[0], a.relay, base))
    s = res["relay_client"] or res["net"]
    if s:
        verdict = ("OK" if s["p95"] < 150 and s["std"] < 30 else
                   "HIGH / JITTERY: use a dedicated router or cable, lower --fps / --quality on the laptop")
        print("  -> %s (want p95 < 150 ms and std < 30 ms; add ~half a recorder tick, 33 ms at 15 Hz)" % verdict)
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    path = os.path.join(HERE, "logs", "latency_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print("  saved %s" % path)


# ---------------------------------------------------------------- snap (stopwatch test)
def cmd_snap(a):
    if a.cams:
        ids = a.cams.split(",")
    else:
        with urllib.request.urlopen(a.camview.rstrip("/") + "/api/cameras", timeout=3.0) as r:
            ids = [c["id"] for c in json.loads(r.read()) if (c.get("fps") or 0) > 0]
    if not ids:
        sys.exit("no live camview streams")
    readers = {i: Reader(a.camview.rstrip("/") + "/stream/" + i) for i in ids}
    t0 = time.time()
    while any(r.latest is None for r in readers.values()) and time.time() - t0 < 5:
        time.sleep(0.05)
    dead = [i for i, r in readers.items() if r.latest is None]
    if dead:
        print("no frames from %s (skipped)" % ", ".join(dead))
    out = os.path.join(HERE, "logs", "snap_%s" % time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    for k in range(a.n):
        time.sleep(a.every)
        now = time.time()
        shot = {i: r.latest for i, r in readers.items() if r.latest is not None}
        stamp = time.strftime("%H:%M:%S", time.localtime(now)) + ".%03d" % (now % 1 * 1000)
        if cv2 is None:
            for i, (jpeg, _, _) in shot.items():
                with open(os.path.join(out, "%02d_%s.jpg" % (k, i)), "wb") as f:
                    f.write(jpeg)
        else:
            tiles = []
            for i, (jpeg, hdr, t) in shot.items():
                img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                img = cv2.resize(img, (round(img.shape[1] * 480 / img.shape[0]), 480))
                cv2.putText(img, "%s  arrived %.0f ms before snap" % (i, (now - t) * 1e3), (8, 470), 0, 0.6,
                            (0, 255, 255), 2)
                tiles.append(img)
            sheet = np.hstack(tiles)
            cv2.putText(sheet, "Pi %s" % stamp, (8, 28), 0, 0.9, (0, 255, 255), 2)
            cv2.imwrite(os.path.join(out, "%02d.jpg" % k), sheet, (cv2.IMWRITE_JPEG_QUALITY, 90))
        print("snap %d/%d  %s" % (k + 1, a.n, stamp), flush=True)
    print("saved %s: read the stopwatch in each camera; the difference is their relative delay" % out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--camview", default=CAMVIEW, help="camview on this Pi (default %(default)s)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("clock", help="laptop - Pi clock offset")
    p.add_argument("laptop", help="worldcam base URL, e.g. http://192.168.1.204:8100")
    p = sub.add_parser("measure", help="world camera capture -> Pi arrival, direct and via the relay")
    p.add_argument("laptop", help="worldcam base URL, e.g. http://192.168.1.204:8100")
    p.add_argument("--secs", type=float, default=30.0)
    p.add_argument("--relay", default="net-world", help="camview's id of the relayed stream ('' = direct only)")
    p = sub.add_parser("snap", help="stopwatch test: same-instant frames of every camera")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--every", type=float, default=1.0, help="seconds between snapshots")
    p.add_argument("--cams", default="", help="camview stream ids, comma separated (default: every live stream)")
    a = ap.parse_args()
    {"clock": cmd_clock, "measure": cmd_measure, "snap": cmd_snap}[a.cmd](a)


if __name__ == "__main__":
    main()
