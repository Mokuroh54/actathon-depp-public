"""Quality report for recorded Pi episodes (dashboard.py's Record episode panel). Standard library only.

    python3 episode_qc.py <episode_dir> [...] [--json]
    python3 episode_qc.py --recent N [--episodes-dir DIR] [--json]

Reads meta.json and steps.jsonl once (JPEGs are only listed, never opened) and checks timing, cameras (missing /
repeated / stale frames), cross-camera skew, the cab inputs (idle lead-in, still tail), the machine (throttle, rabbit,
tracks, gate output), the camera set and the time of day. Each item gets a verdict ok / warn / bad (or n/a when the
episode has no data for it); the overall verdict is the worst one. Docs: docs/episode-qc.md.

Library use (dashboard.py imports it): qc_episode(path, prev_path=None) -> dict, render_text(report) -> str,
item_lines(report) -> one {item, verdict, text, notes} per item.
Exit code: 0 all ok, 1 some warn, 2 some bad (or unreadable episode).
"""
import argparse
import datetime
import glob
import json
import math
import os
import sys

CHANNELS = ("swing", "boom", "arm", "bucket", "thumb")

# ---------------------------------------------------------------- thresholds (periods = 1 / meta fps)
RATE_TOL = 0.05                  # bad: mean step rate off the meta fps by more than 5 %
INTERVAL_WARN_PERIODS = 1.5      # warn: a step interval longer than this (a late tick)
INTERVAL_BAD_PERIODS = 2.0       # bad: a step interval longer than this
AGE_WARN_PERIODS = 1.5           # warn: a camera frame older than 1.5 periods at its step
AGE_BAD_S = 0.5                  # bad: a camera frame older than this (the converter's --max-age)
REPEAT_WARN_FRAC = 0.10          # warn: more than 10 % of a camera's steps reuse the previous step's frame
SKEW_WARN_PERIODS = 1.0          # warn: p95 cross-camera skew above 1 period (capture times `cap` when recorded
                                 # with camsync, else arrival times)
SKEW_BAD_PERIODS = 2.0           # bad: p95 cross-camera arrival skew above 2 periods
ACTIVE_EPS = 1e-6                # a channel counts as moving when |action| > this (actions are already deadbanded)
LEAD_IN_WARN_S = 5.0             # warn: more than this idle before the first cab input
TAIL_WARN_S = 0.5                # warn: still tail shorter than this (AGENTS.md: end every demo still)
THROTTLE_RANGE_WARN = 0.1        # warn: throttle moved more than this during the episode
DAY_START_H = 8                  # warn: recorded before 08:00 or after 18:59 local (night: pipeline tests only)
DAY_END_H = 18

ORDER = {"n/a": 0, "ok": 0, "warn": 1, "bad": 2}
NA = "n/a"


def _worst(*vs):
    vs = [v for v in vs if v] or ["ok"]
    return max(vs, key=lambda v: ORDER.get(v, 0))


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def _r(x, nd=1):
    return NA if x is None else round(x, nd)


def _pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(math.ceil(q / 100.0 * len(s))) - 1))
    return s[k]


def _item(verdict, notes, **fields):
    d = {"verdict": verdict, "notes": notes}
    d.update(fields)
    return d


def _read(path):
    """-> (meta dict, list of step dicts, list of read problems). Never raises on bad content."""
    problems = []
    meta = {}
    try:
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        if not isinstance(meta, dict):
            problems.append("meta.json is not an object")
            meta = {}
    except OSError:
        problems.append("no meta.json")
    except ValueError as e:
        problems.append("meta.json unreadable: %s" % e)
    steps = []
    bad_lines = 0
    try:
        with open(os.path.join(path, "steps.jsonl")) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    s = json.loads(line)
                except ValueError:
                    bad_lines += 1
                    continue
                if isinstance(s, dict):
                    steps.append(s)
                else:
                    bad_lines += 1
    except OSError:
        problems.append("no steps.jsonl")
    if bad_lines:
        problems.append("%d unreadable lines in steps.jsonl" % bad_lines)
    return meta, steps, problems


def _step_time(s):
    return _num(s.get("mono")) if _num(s.get("mono")) is not None else _num(s.get("t"))


def _cam_names(meta, steps):
    names = list(meta.get("cameras") or {}) if isinstance(meta.get("cameras"), dict) else []
    for s in steps:
        for n in (s.get("cams") or {}):
            if n not in names:
                names.append(n)
    return names


# ---------------------------------------------------------------- checks
def _timing(meta, steps):
    fps = _num(meta.get("fps"))
    period = 1.0 / fps if fps else None
    ts = [_step_time(s) for s in steps]
    ts = [t for t in ts if t is not None]
    iv = [b - a for a, b in zip(ts, ts[1:])]
    n = len(steps)
    duration = (ts[-1] - ts[0]) if len(ts) > 1 else None
    rate = (len(ts) - 1) / duration if duration else None
    late = meta.get("late_ticks")
    dropped = meta.get("dropped_steps")
    notes = []
    v = "ok"
    meta_n = meta.get("n_steps")
    if n == 0:
        return _item("bad", ["no steps"], n_steps=0, fps=_r(fps), duration_s=NA, rate_hz=NA, interval_ms=NA,
                     late_ticks=late if late is not None else NA, dropped_steps=NA, meta_n_steps=meta_n or NA)
    idx = [s.get("i") for s in steps]
    gaps = sum(1 for a, b in zip(idx, idx[1:]) if isinstance(a, int) and isinstance(b, int) and b != a + 1)
    if isinstance(meta_n, int) and meta_n != n:
        v = _worst(v, "bad")
        notes.append("meta n_steps %d but steps.jsonl has %d" % (meta_n, n))
    if gaps:
        v = _worst(v, "bad")
        notes.append("step index jumps %d times" % gaps)
    if isinstance(dropped, int) and dropped > 0:
        v = _worst(v, "bad")
        notes.append("%d steps dropped (writer queue full)" % dropped)
    if period and rate:
        off = rate / fps - 1.0
        if abs(off) > RATE_TOL:
            v = _worst(v, "bad")
            notes.append("rate %.2f Hz is %+.1f%% off %.1f Hz" % (rate, 100 * off, fps))
    if period and iv:
        mx = max(iv)
        if mx > INTERVAL_BAD_PERIODS * period:
            v = _worst(v, "bad")
            notes.append("longest interval %.0f ms > %.1f periods" % (1000 * mx, INTERVAL_BAD_PERIODS))
        elif mx > INTERVAL_WARN_PERIODS * period:
            v = _worst(v, "warn")
            notes.append("longest interval %.0f ms > %.1f periods" % (1000 * mx, INTERVAL_WARN_PERIODS))
    if isinstance(late, int) and late > 0:
        v = _worst(v, "warn")
        notes.append("%d late ticks" % late)
    if fps is None:
        notes.append("no fps in meta: rate not checked")
    ivms = {"mean": _r(1000 * sum(iv) / len(iv)) if iv else NA, "min": _r(1000 * min(iv)) if iv else NA,
            "max": _r(1000 * max(iv)) if iv else NA}
    return _item(v, notes, n_steps=n, meta_n_steps=meta_n if meta_n is not None else NA, fps=_r(fps, 2),
                 duration_s=_r(duration, 2), rate_hz=_r(rate, 3), interval_ms=ivms,
                 late_ticks=late if late is not None else NA, dropped_steps=dropped if dropped is not None else NA)


def _list_files(path, name):
    try:
        return {e.name for e in os.scandir(os.path.join(path, name)) if e.is_file()}
    except OSError:
        return set()


def _cameras(path, meta, steps, names, period):
    out = {}
    fr_meta = meta.get("frames_repeated") if isinstance(meta.get("frames_repeated"), dict) else {}
    fm_meta = meta.get("frames_missing") if isinstance(meta.get("frames_missing"), dict) else {}
    n = len(steps)
    for name in names:
        on_disk = _list_files(path, name)
        refs, ages, lags, notes = [], [], [], []
        no_ref = file_absent = repeats = 0
        prev_seq = None
        for s in steps:
            s_mono = _num(s.get("mono"))
            c = (s.get("cams") or {}).get(name)
            if not isinstance(c, dict) or not c.get("file"):
                no_ref += 1
                prev_seq = None
                continue
            rel = c["file"]
            refs.append(rel)
            if os.path.basename(rel) not in on_disk:
                file_absent += 1
            seq = c.get("seq")
            if seq is not None and seq == prev_seq:
                repeats += 1
            prev_seq = seq
            age = _num(c.get("age"))
            if age is None and s_mono is not None and _num(c.get("mono")) is not None:
                age = s_mono - c["mono"]
            if age is not None:
                ages.append(age)
            lag = _num(c.get("lag"))
            if lag is not None:
                lags.append(lag)
        v = "ok"
        if no_ref or file_absent:
            v = "bad"
            if no_ref:
                notes.append("%d steps without a frame" % no_ref)
            if file_absent:
                notes.append("%d referenced files not on disk" % file_absent)
        mx = max(ages) if ages else None
        if mx is not None:
            if mx > AGE_BAD_S:
                v = _worst(v, "bad")
                notes.append("max frame age %.0f ms > %.0f ms" % (1000 * mx, 1000 * AGE_BAD_S))
            elif period and mx > AGE_WARN_PERIODS * period:
                v = _worst(v, "warn")
                notes.append("max frame age %.0f ms > %.1f periods" % (1000 * mx, AGE_WARN_PERIODS))
        if n and repeats / float(n) > REPEAT_WARN_FRAC:
            v = _worst(v, "warn")
            notes.append("%.0f%% of steps repeat the previous frame" % (100.0 * repeats / n))
        cm = (meta.get("cameras") or {}).get(name) if isinstance(meta.get("cameras"), dict) else None
        out[name] = _item(
            v, notes, steps=n, files_on_disk=len(on_disk), frame_refs=len(refs), unique_refs=len(set(refs)),
            steps_without_frame=no_ref, files_absent=file_absent,
            frames_missing_meta=fm_meta.get(name, NA), frames_repeated_meta=fr_meta.get(name, NA),
            repeats_from_seq=repeats if refs else NA,
            age_ms={"max": _r(1000 * mx) if mx is not None else NA,
                    "p95": _r(1000 * _pct(ages, 95)) if ages else NA},
            lag_ms={"max": _r(1000 * max(lags)) if lags else NA,       # capture -> step (camsync), info only
                    "p95": _r(1000 * _pct(lags, 95)) if lags else NA},
            camview_id=(cm or {}).get("camview_id", NA),
            resolution=("%sx%s" % (cm.get("width"), cm.get("height"))) if cm and cm.get("width") else NA)
    return out


def _skew(steps, names, period):
    if len(names) < 2:
        return _item(NA, ["fewer than 2 cameras"], cameras=len(names), steps_measured=0,
                     skew_ms={"median": NA, "p95": NA, "max": NA})
    sk, basis = [], "arrival"
    for s in steps:
        cams = s.get("cams") or {}
        ts = []
        for n in names:
            c = cams.get(n)
            if isinstance(c, dict) and c.get("file"):
                t = _num(c.get("cap"))                   # capture time (camsync): the skew the images really have
                if t is not None:
                    basis = "capture"
                else:
                    t = _num(c.get("mono"))
                if t is None:
                    t = _num(c.get("wall"))
                if t is not None:
                    ts.append(t)
        if len(ts) >= 2:
            sk.append(max(ts) - min(ts))
    if not sk:
        return _item(NA, ["no per-camera timestamps"], cameras=len(names), steps_measured=0,
                     skew_ms={"median": NA, "p95": NA, "max": NA})
    p95 = _pct(sk, 95)
    v, notes = "ok", []
    if period and p95 > SKEW_BAD_PERIODS * period:
        v = "bad"
        notes.append("p95 skew %.0f ms > %.0f periods" % (1000 * p95, SKEW_BAD_PERIODS))
    elif period and p95 > SKEW_WARN_PERIODS * period:
        v = "warn"
        notes.append("p95 skew %.0f ms > %.0f period" % (1000 * p95, SKEW_WARN_PERIODS))
    return _item(v, notes, cameras=len(names), steps_measured=len(sk), basis=basis,
                 skew_ms={"median": _r(1000 * _pct(sk, 50)), "p95": _r(1000 * p95), "max": _r(1000 * max(sk))})


def _cab(steps, fps):
    chans = {}
    active = []
    have_action = False
    for ch in CHANNELS:
        vals = []
        for s in steps:
            a = s.get("action")
            x = _num(a.get(ch)) if isinstance(a, dict) else None
            if x is not None:
                vals.append(x)
        have_action = have_action or bool(vals)
        nz = sum(1 for x in vals if abs(x) > ACTIVE_EPS)
        chans[ch] = {"min": _r(min(vals), 3) if vals else NA, "max": _r(max(vals), 3) if vals else NA,
                     "pct_nonzero": _r(100.0 * nz / len(vals)) if vals else NA}
    for k, s in enumerate(steps):
        a = s.get("action")
        if isinstance(a, dict) and any(abs(_num(a.get(ch)) or 0.0) > ACTIVE_EPS for ch in CHANNELS):
            active.append(k)
    if not have_action:
        return _item(NA, ["no action in steps"], channels=chans, first_active_step=NA, last_active_step=NA,
                     lead_in_s=NA, still_tail_s=NA)
    if not active:
        return _item("warn", ["no channel moved"], channels=chans, first_active_step=NA, last_active_step=NA,
                     lead_in_s=NA, still_tail_s=NA)
    t = [_step_time(s) for s in steps]

    def span(a, b):
        if t[a] is not None and t[b] is not None:
            return t[b] - t[a]
        return (b - a) / fps if fps else None

    first, last = active[0], active[-1]
    lead, tail = span(0, first), span(last, len(steps) - 1)
    v, notes = "ok", []
    if lead is not None and lead > LEAD_IN_WARN_S:
        v = "warn"
        notes.append("idle lead-in %.1f s > %.1f s" % (lead, LEAD_IN_WARN_S))
    if tail is not None and tail < TAIL_WARN_S:
        v = "warn"
        notes.append("still tail %.2f s < %.1f s" % (tail, TAIL_WARN_S))
    fi, li = steps[first].get("i", first), steps[last].get("i", last)
    return _item(v, notes, channels=chans, first_active_step=fi, last_active_step=li,
                 lead_in_s=_r(lead, 2), still_tail_s=_r(tail, 2))


def _machine(meta, steps):
    thr, rab = [], set()
    gate_steps = 0
    have_gate = False
    for s in steps:
        st = s.get("state")
        if not isinstance(st, dict):
            continue
        ctl = st.get("controls") if isinstance(st.get("controls"), dict) else {}
        x = _num(st.get("throttle"))
        if x is None:
            x = _num(ctl.get("throttle"))
        if x is not None:
            thr.append(x)
        r = _num(st.get("rabbit_button"))
        if r is None:
            r = _num(ctl.get("rabbit_button"))
        if r is not None:
            rab.add(r)
        sent = st.get("sent")
        if isinstance(sent, dict) or "armed" in st:
            have_gate = True
            if st.get("armed") is True or any(abs(_num(v) or 0.0) > ACTIVE_EPS for v in (sent or {}).values()):
                gate_steps += 1
    ts = meta.get("track_steps")
    v, notes = "ok", []
    rng = (max(thr) - min(thr)) if thr else None
    if rng is not None and rng > THROTTLE_RANGE_WARN:
        v = "warn"
        notes.append("throttle moved %.2f during the episode" % rng)
    if isinstance(ts, int) and ts > 0:
        v = "bad"
        notes.append("tracks moved in %d steps" % ts)
    if gate_steps:
        v = _worst(v, "warn")
        notes.append("gate armed / sending in %d steps (something else drove the gate)" % gate_steps)
    sm = meta.get("state_missing")
    if isinstance(sm, int) and sm > 0:
        v = _worst(v, "warn")
        notes.append("gate state missing in %d steps" % sm)
    if not thr:
        notes.append("no throttle recorded")
    return _item(v, notes,
                 throttle={"min": _r(min(thr), 3) if thr else NA, "max": _r(max(thr), 3) if thr else NA,
                           "mean": _r(sum(thr) / len(thr), 3) if thr else NA,
                           "at_start_meta": meta.get("throttle_at_start", NA)},
                 rabbit_button=sorted(rab) if rab else NA, rabbit_at_start_meta=meta.get("rabbit_at_start", NA),
                 track_steps=ts if ts is not None else NA,
                 gate_active_steps=gate_steps if have_gate else NA,
                 state_missing=sm if sm is not None else NA)


def _cam_set(meta):
    cams = meta.get("cameras")
    if isinstance(meta.get("camera_set"), dict):
        return {n: list(v) if isinstance(v, (list, tuple)) else v for n, v in meta["camera_set"].items()}
    if isinstance(cams, dict):
        return {n: [v.get("camview_id"), v.get("width"), v.get("height")] for n, v in cams.items()
                if isinstance(v, dict)}
    return None


def _fmt_set(cs):
    return ", ".join("%s=%s %sx%s" % (n, v[0], v[1], v[2]) if isinstance(v, list) and len(v) == 3 else
                     "%s=%s" % (n, v) for n, v in sorted(cs.items())) or "none"


def _camera_set(meta, prev_path):
    cs = _cam_set(meta)
    notes = []
    v = "ok" if cs else NA
    prev_name, prev_set, changed = NA, NA, NA
    if prev_path:
        pm, _, _ = _read_meta_only(prev_path)
        prev_name = os.path.basename(os.path.normpath(prev_path))
        ps = _cam_set(pm)
        if ps is not None and cs is not None:
            prev_set = _fmt_set(ps)
            changed = ps != cs
            if changed:
                v = "warn"
                notes.append("camera set differs from %s (%s): new dataset" % (prev_name, prev_set))
    cf = meta.get("camera_set_changed_from")
    if isinstance(cf, dict):
        v = _worst(v, "warn")
        notes.append("meta: camera set changed at this episode (was %s)" % _fmt_set(cf))
    return _item(v, notes, camera_set=_fmt_set(cs) if cs else NA, prev_episode=prev_name, prev_camera_set=prev_set,
                 changed_vs_prev=changed, camera_set_changed_from_meta=_fmt_set(cf) if isinstance(cf, dict) else NA)


def _read_meta_only(path):
    try:
        with open(os.path.join(path, "meta.json")) as f:
            m = json.load(f)
        return (m if isinstance(m, dict) else {}), True, None
    except (OSError, ValueError) as e:
        return {}, False, str(e)


def _when(meta, path):
    dt = None
    st = meta.get("started")
    if isinstance(st, str):                      # the Pi's local time with its UTC offset
        try:
            dt = datetime.datetime.strptime(st, "%Y-%m-%dT%H:%M:%S%z")
        except ValueError:
            dt = None
    if dt is None and _num(meta.get("started_wall")) is not None:
        dt = datetime.datetime.fromtimestamp(meta["started_wall"]).astimezone()
    if dt is None:                               # folder name YYYYmmdd_HHMMSS_<slug> (Pi local time)
        try:
            dt = datetime.datetime.strptime(os.path.basename(os.path.normpath(path))[:15], "%Y%m%d_%H%M%S")
        except ValueError:
            dt = None
    if dt is None:
        return _item(NA, ["no start time"], recorded_at=NA, hour=NA)
    v, notes = "ok", []
    if dt.hour < DAY_START_H or dt.hour > DAY_END_H:
        v = "warn"
        notes.append("recorded at %02d:%02d local: night / dusk, pipeline test only (not training)"
                     % (dt.hour, dt.minute))
    return _item(v, notes, recorded_at=dt.isoformat(), hour=dt.hour)


# ---------------------------------------------------------------- public API
def qc_episode(path, prev_path=None):
    """Quality report for one episode folder (JSON-able dict). prev_path: the previous saved episode, for the
    camera-set comparison. Never raises on missing / malformed content."""
    meta, steps, problems = _read(path)
    fps = _num(meta.get("fps"))
    period = 1.0 / fps if fps else None
    names = _cam_names(meta, steps)
    rep = {"episode": os.path.basename(os.path.normpath(path)), "path": os.path.abspath(path),
           "task": meta.get("task", NA), "format": meta.get("format", NA),
           "action_source": meta.get("action_source", NA), "problems": problems}
    items = {}
    items["timing"] = _timing(meta, steps)
    items["cameras"] = _cameras(path, meta, steps, names, period)
    items["skew"] = _skew(steps, names, period)
    items["cab"] = _cab(steps, fps)
    items["machine"] = _machine(meta, steps)
    items["camera_set"] = _camera_set(meta, prev_path)
    items["daylight"] = _when(meta, path)
    rep["items"] = items
    rep["recorded_at"] = items["daylight"]["recorded_at"]
    cam_v = _worst(*[c["verdict"] for c in items["cameras"].values()]) if items["cameras"] else "bad"
    if not items["cameras"]:
        items["timing"]["notes"].append("no cameras in this episode")
    verdicts = [cam_v] + [it["verdict"] for k, it in items.items() if k != "cameras"]
    if problems:
        verdicts.append("bad" if not steps else "warn")
    rep["verdict"] = _worst(*verdicts)
    return rep


def _line(label, it, body):
    return {"item": label, "verdict": it["verdict"], "text": body, "notes": list(it.get("notes") or [])}


def item_lines(report):
    """One line per item: [{"item", "verdict", "text", "notes"}] (what render_text prints; the dashboard shows it)."""
    it = report["items"]
    out = []
    t = it["timing"]
    iv = t["interval_ms"] if isinstance(t["interval_ms"], dict) else {}
    out.append(_line("timing", t, "%s steps, %s s, %s Hz (meta %s), interval ms mean/min/max %s/%s/%s, late %s"
                     % (t["n_steps"], t["duration_s"], t["rate_hz"], t["fps"], iv.get("mean", NA),
                        iv.get("min", NA), iv.get("max", NA), t["late_ticks"])))
    for n, c in it["cameras"].items():
        out.append(_line("cam " + n, c, "%s %s: %s/%s frames (%s unique, %s files), repeated %s (seq %s), "
                         "age ms max/p95 %s/%s" % (c["camview_id"], c["resolution"], c["frame_refs"], c["steps"],
                                                   c["unique_refs"], c["files_on_disk"], c["frames_repeated_meta"],
                                                   c["repeats_from_seq"], c["age_ms"]["max"], c["age_ms"]["p95"])))
    if not it["cameras"]:
        out.append({"item": "cameras", "verdict": "bad", "text": "none", "notes": []})
    s = it["skew"]
    out.append(_line("skew", s, "ms median/p95/max %s/%s/%s over %s steps"
                     % (s["skew_ms"]["median"], s["skew_ms"]["p95"], s["skew_ms"]["max"], s["steps_measured"])))
    c = it["cab"]
    moving = ", ".join("%s %s%%" % (ch, d["pct_nonzero"]) for ch, d in c["channels"].items())
    out.append(_line("cab", c, "active steps %s-%s, lead-in %s s, still tail %s s; nonzero: %s"
                     % (c["first_active_step"], c["last_active_step"], c["lead_in_s"], c["still_tail_s"], moving)))
    m = it["machine"]
    out.append(_line("machine", m, "throttle %s-%s (start %s), rabbit %s, track steps %s, gate active steps %s"
                     % (m["throttle"]["min"], m["throttle"]["max"], m["throttle"]["at_start_meta"],
                        m["rabbit_button"], m["track_steps"], m["gate_active_steps"])))
    cs = it["camera_set"]
    out.append(_line("cam set", cs, "%s (prev %s: %s)" % (cs["camera_set"], cs["prev_episode"],
                                                          "changed" if cs["changed_vs_prev"] is True else
                                                          "same" if cs["changed_vs_prev"] is False else NA)))
    d = it["daylight"]
    out.append(_line("daylight", d, "hour %s" % d["hour"]))
    return out


def render_text(report):
    """Short human rendering of qc_episode()'s report."""
    out = ["== %s  [%s]" % (report["episode"], report["verdict"].upper()),
           "task: %s   recorded: %s   source: %s" % (report["task"], report["recorded_at"], report["action_source"])]
    for p in report.get("problems") or []:
        out.append("      problem: %s" % p)
    for ln in item_lines(report):
        out.append("%-5s %-10s %s%s" % (ln["verdict"].upper(), ln["item"], ln["text"],
                                        ("  ! " + "; ".join(ln["notes"])) if ln["notes"] else ""))
    return "\n".join(out)


# ---------------------------------------------------------------- CLI
def saved_episodes(episodes_dir):
    """Saved episode folders (with meta.json, not .partial), oldest first (names start with the start time)."""
    out = []
    for p in sorted(glob.glob(os.path.join(episodes_dir, "*", "meta.json"))):
        d = os.path.dirname(p)
        if not d.endswith(".partial"):
            out.append(d)
    return out


def previous_episode(path, episodes_dir):
    name = os.path.basename(os.path.normpath(path))
    prev = None
    for d in saved_episodes(episodes_dir):
        b = os.path.basename(d)
        if b >= name:
            break
        prev = d
    return prev


def main(argv=None):
    ap = argparse.ArgumentParser(description="Quality report for recorded Pi episodes (docs/episode-qc.md).")
    ap.add_argument("episodes", nargs="*", help="episode folders")
    ap.add_argument("--recent", type=int, metavar="N", help="check the N newest saved episodes in --episodes-dir")
    ap.add_argument("--episodes-dir", help="episodes folder (default: ~/act-athon/episodes for --recent, else each "
                    "episode's parent folder) for the previous-episode camera-set check")
    ap.add_argument("--json", action="store_true", help="print JSON (one object, or a list for several)")
    a = ap.parse_args(argv)
    todo = []
    if a.recent:
        ed = os.path.expanduser(a.episodes_dir or "~/act-athon/episodes")
        eps = saved_episodes(ed)
        for k in range(max(0, len(eps) - a.recent), len(eps)):
            todo.append((eps[k], eps[k - 1] if k > 0 else None))
        if not eps:
            print("no saved episodes in %s" % ed, file=sys.stderr)
    for p in a.episodes:
        ed = os.path.expanduser(a.episodes_dir) if a.episodes_dir else os.path.dirname(os.path.abspath(p))
        todo.append((p, previous_episode(p, ed)))
    if not todo:
        ap.error("give episode folders or --recent N")
    reps = [qc_episode(p, prev) for p, prev in todo]
    if a.json:
        print(json.dumps(reps[0] if len(reps) == 1 else reps, indent=1))
    else:
        print("\n\n".join(render_text(r) for r in reps))
    worst = _worst(*[r["verdict"] for r in reps])
    return ORDER.get(worst, 0)


if __name__ == "__main__":
    sys.exit(main())
