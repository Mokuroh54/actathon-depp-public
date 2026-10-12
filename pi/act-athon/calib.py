"""Joint calibration for the excavator arm: jog to end stops, timed sweeps, dead-band ramps, inclinometer angles.
Served as a web page like dashboard.py (stdlib only, same path: rig -> 127.0.0.1:8096 -> gate).

Run ON the Pi from the act-athon repo root:  python3 calib.py   then open http://<pi>:8099 on a laptop.
Someone at the kill switch, nobody within 3 m of the arm. Procedure: docs/calibration.md in our repo.

Nothing is written to the gate until you jog or start a sweep / ramp (the first write energises the relay).

Jog (same keys as the dashboard's Teleoperate tab): hold SPACE + A/D swing, W/S arm, I/K boom, J/L bucket, U/O thumb;
    1 / 2 / 3 = speed 0.3 / 0.5 / 1.0. ESC or X = STOP (neutral + relay open), everywhere, always.
Sweep / ramp: pick joint, direction (and magnitude / timeout), click Arm, then HOLD SPACE to start.
    The command loop runs HERE on the Pi at 20 Hz (the network is too slow for the browser to time it).
    M       motion visibly started (ramp: records the command at that moment and ends the ramp)
    S or release SPACE   joint hit its end stop (sweep ends, neutral)
    Also ends (neutral): no heartbeat from the page for HB_TIMEOUT_S, window blur / tab hidden,
    the sweep timeout (default 15 s, max 30 s), the end of the ramp (0 -> 0.6 in 12 s).
Every sweep / ramp / angle entry is appended as one JSON line to calib/calib_<date>.jsonl next to this file.

Key presses are timestamped in the browser (performance.now) and mapped onto the Pi's clock with the smallest
observed (receive - send) offset of this run's messages, so network jitter does not end up in the timings; the raw
receive times are recorded too (*_rx).
"""
import argparse
import datetime
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig

PORT = 8099
WATCHDOG_S = 0.3         # jog: page stopped sending -> neutral (same as dashboard.py)
HB_TIMEOUT_S = 0.3       # sweep / ramp: heartbeat gap -> abort
RUN_HZ = 20.0            # sweep / ramp write rate
TICK_S = 0.01            # abort-check granularity of the run loop
DEFAULT_TIMEOUT_S = 15.0
MAX_TIMEOUT_S = 30.0
RAMP_MAX = 0.6           # dead-band ramp: |cmd| 0 -> RAMP_MAX over RAMP_S
RAMP_S = 12.0
SPEEDS = (0.3, 0.5, 1.0)
CALIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib")
LINKS = ("boom", "arm", "bucket", "thumb")
# joint angle = direction angle of the link - direction angle of its parent link (boom's parent: the chassis)
PARENT = {"boom": "chassis", "arm": "boom", "bucket": "arm", "thumb": "arm"}

LOCK = threading.Lock()   # every gate write / stop happens under this lock, so nothing can be written after a STOP
LAST = {"t": 0.0, "cmd": {}}          # jog
RUN = {"cur": None, "last": None}     # the active sweep / ramp and the last finished record
# last STOP from a page (its session id + browser time): a start that page sent BEFORE that STOP but that arrived
# after it (reordered on the bad wifi) is refused, so a STOP can never be followed by a stale start
LAST_STOP = {"sid": None, "ct": None}


# ---------------------------------------------------------------- helpers
def read_state():
    """rig.read() plus rabbit_button, from one /state call."""
    s = rig._http("/state")
    c = s.get("controls", {})
    out = {k: c.get(rig._TO_API.get(k, k), 0.0) for k in rig.KEYS}
    out.update(throttle=c.get("throttle"), rabbit=c.get("rabbit_button"), live=bool(s.get("armed")),
               veto=s.get("veto", ""), age_s=c.get("age"))
    return out


def calib_path():
    return os.path.join(CALIB_DIR, "calib_%s.jsonl" % datetime.date.today().isoformat())


def append(rec):
    os.makedirs(CALIB_DIR, exist_ok=True)
    with open(calib_path(), "a") as f:
        f.write(json.dumps(rec) + "\n")


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def link_angle(p, back):
    """Inclinometer pitch (deg from horizontal, + = far end above the pivot) -> direction angle in -180..180
    (0 = pointing forward, away from the cab; 90 = straight up). back = the far end is behind the pivot (toward the cab)."""
    if not back:
        return p
    return (180.0 if p >= 0 else -180.0) - p


def joint_angle(joint, links, chassis):
    """links: {name: direction angle}. Returns joint angle (deg) or None if a needed link is missing."""
    parent = PARENT[joint]
    pa = chassis if parent == "chassis" else links.get(parent)
    a = links.get(joint)
    if a is None or pa is None:
        return None
    return round(wrap180(a - pa), 2)


def num(v):
    if v is None or v == "":
        return None
    return float(v)


# ---------------------------------------------------------------- sweep / ramp loop
def cmd_at(r, now):
    if r["kind"] == "ramp":
        return r["dir"] * min(RAMP_MAX, RAMP_MAX / RAMP_S * (now - r["t_start_cmd"]))
    return r["dir"] * r["mag"]


def seen(r, ct, now):
    """Update this run's clock offset estimate (Pi monotonic - browser clock) from a message sent at browser time ct."""
    if ct is None:
        return
    off = now - float(ct)
    if r["off"] is None or off < r["off"]:
        r["off"] = off


def run_loop(r):
    period = 1.0 / RUN_HZ
    next_w = time.monotonic()
    reason = None
    while reason is None:
        with LOCK:
            now = time.monotonic()
            if r["abort"]:
                reason = r["abort"]
            elif now - r["hb"] > HB_TIMEOUT_S:
                reason = "heartbeat"
            elif r["t_start_cmd"] is not None and now - r["t_start_cmd"] > r["timeout_s"]:
                reason = "ramp_end" if r["kind"] == "ramp" else "timeout"
            elif now >= next_w:
                if r["t_start_cmd"] is None:
                    r["t_start_cmd"] = now
                c = round(cmd_at(r, now), 4)
                try:
                    rig.write(**{r["joint"]: c})
                except Exception as ex:
                    r["error"] = str(ex)
                    reason = "write_error"
                else:
                    r["n_writes"] += 1
                    r["t_last_write"] = now
                    r["last_cmd"] = c
                    next_w += period
                    if next_w < now:        # a slow write: don't burst to catch up
                        next_w = now + period
        if reason is None:
            time.sleep(TICK_S)
    finish(r, reason)


def finish(r, reason):
    with LOCK:
        # after a STOP (relay open) write nothing: a write would energise the relay again
        if r["abort"] != "estop" and r["t_start_cmd"] is not None:
            try:
                rig.write()
            except Exception:
                pass
        rec = record(r, reason, time.monotonic())
        RUN["last"] = rec
        if RUN["cur"] is r:
            RUN["cur"] = None
    try:
        append(rec)
    except Exception as ex:
        rec["save_error"] = str(ex)
        print("could not save: %s" % ex, flush=True)


def corrected(r, ct, rx):
    if rx is None:
        return None
    if ct is None or r["off"] is None:
        return rx
    return min(rx, float(ct) + r["off"])


def record(r, reason, t_end):
    t0 = r["t_start_cmd"]
    t_motion = corrected(r, r["ct_motion"], r["t_motion_rx"])
    t_stop = corrected(r, r["ct_stop"], r["t_stop_rx"])
    n = r["n_writes"]
    rec = {
        "kind": r["kind"], "id": r["id"], "wall_time": r["wall_time"], "joint": r["joint"], "dir": r["dir"],
        "throttle": r["throttle"], "rabbit_button": r["rabbit"], "state_error": r["state_error"],
        "end_reason": reason, "stop_key": r["stop_key"], "error": r["error"],
        "t_start_cmd": t0, "t_motion": t_motion, "t_stop": t_stop, "t_end": t_end,
        "t_motion_rx": r["t_motion_rx"], "t_stop_rx": r["t_stop_rx"], "clock_offset": r["off"],
        "n_writes": n,
        "write_hz": round((n - 1) / (r["t_last_write"] - t0), 2) if n > 1 and r["t_last_write"] > t0 else None,
        "lag_s": round(t_motion - t0, 3) if t_motion is not None and t0 is not None else None,
        "hb_timeout_s": HB_TIMEOUT_S, "note": r["note"],
    }
    if r["kind"] == "sweep":
        rec.update(mag=r["mag"], cmd=r["dir"] * r["mag"], timeout_s=r["timeout_s"], angle_deg=r["angle_deg"],
                   traverse_s=round(t_stop - t_motion, 3) if t_stop is not None and t_motion is not None else None)
    else:
        rate = round(RAMP_MAX / RAMP_S, 6)
        rec.update(ramp_rate_per_s=rate, ramp_max=RAMP_MAX, ramp_s=RAMP_S,
                   cmd_at_motion=round(r["dir"] * min(RAMP_MAX, rate * (t_motion - t0)), 4)
                   if t_motion is not None and t0 is not None else None,
                   last_cmd=r["last_cmd"])
    return rec


def start_run(b):
    kind, joint = b.get("kind"), b.get("joint")
    if kind not in ("sweep", "ramp"):
        raise ValueError("kind must be sweep or ramp")
    if joint not in rig.KEYS:
        raise ValueError("joint must be one of %s" % (rig.KEYS,))
    d = int(b.get("dir", 0))
    if d not in (-1, 1):
        raise ValueError("dir must be -1 or 1")
    mag = float(b.get("mag", 0.5))
    if not 0.0 < mag <= 1.0:
        raise ValueError("mag must be in (0, 1]")
    timeout = RAMP_S if kind == "ramp" else min(MAX_TIMEOUT_S, float(b.get("timeout_s") or DEFAULT_TIMEOUT_S))
    if timeout <= 0:
        raise ValueError("timeout_s must be > 0")
    throttle = rabbit = err = None
    try:
        st = read_state()           # read-only: /state
        throttle, rabbit = st["throttle"], st["rabbit"]
    except Exception as ex:
        err = str(ex)
    now = time.monotonic()
    r = {"id": "%s-%d" % (kind, int(time.time() * 1000)), "kind": kind, "joint": joint, "dir": d, "mag": mag,
         "timeout_s": timeout, "angle_deg": num(b.get("angle_deg")), "note": b.get("note") or "",
         "wall_time": datetime.datetime.now().isoformat(timespec="milliseconds"),
         "throttle": throttle, "rabbit": rabbit, "state_error": err,
         "hb": now, "off": None, "abort": None, "error": None, "stop_key": None,
         "t_start_cmd": None, "t_last_write": None, "n_writes": 0, "last_cmd": 0.0,
         "t_motion_rx": None, "ct_motion": None, "t_stop_rx": None, "ct_stop": None}
    seen(r, b.get("ct"), now)
    with LOCK:
        if (b.get("sid") is not None and b.get("sid") == LAST_STOP["sid"] and b.get("ct") is not None
                and LAST_STOP["ct"] is not None and float(b["ct"]) <= LAST_STOP["ct"]):
            raise RuntimeError("STOP was pressed after this start was sent")
        if RUN["cur"] is not None:
            raise RuntimeError("a %s is already running" % RUN["cur"]["kind"])
        LAST["cmd"] = {}            # jog watchdog off
        RUN["cur"] = r
    threading.Thread(target=run_loop, args=(r,), daemon=True).start()
    return {"id": r["id"], "throttle": throttle, "rabbit_button": rabbit, "state_error": err}


def cur_run(rid):
    r = RUN["cur"]
    return r if r is not None and r["id"] == rid else None


def status(r):
    if r is None:
        return {"running": False, "last": RUN["last"]}
    now = time.monotonic()
    t0 = r["t_start_cmd"]
    return {"running": True, "id": r["id"], "elapsed": round(now - t0, 2) if t0 else 0.0, "cmd": r["last_cmd"],
            "motion": r["t_motion_rx"] is not None, "n_writes": r["n_writes"]}


# ---------------------------------------------------------------- results
def load_records():
    try:
        with open(calib_path()) as f:
            return [json.loads(l) for l in f if l.strip()]
    except FileNotFoundError:
        return []


def fmt(v, nd=2):
    return "" if v is None else ("%.*f" % (nd, v))


def results():
    recs = load_records()
    # joint -> latest measured range (deg) from an angles record (angles are usually entered after the sweeps)
    rng = {x["joint"]: x["range_deg"] for x in recs if x.get("kind") == "angles" and x.get("range_deg") is not None}
    rows = []
    for rec in recs:
        k, j = rec.get("kind"), rec.get("joint")
        if k == "angles":
            a = rec.get("joint_deg", {})
            rows.append({"kind": "angles", "joint": j, "cmd": "",
                         "detail": "- end %s° · + end %s°  %s" % (fmt(a.get("-"), 1), fmt(a.get("+"), 1), rec.get("note", "")),
                         "result": ("range %s°" % fmt(rec.get("range_deg"), 1)) if rec.get("range_deg") is not None else ""})
        elif k == "sweep":
            r = rec.get("angle_deg") if j == "swing" else rng.get(j)
            tr = rec.get("traverse_s")
            ok = rec.get("end_reason") == "stop" and tr and r is not None
            rows.append({"kind": "sweep", "joint": j, "cmd": fmt(rec.get("cmd")),
                         "detail": "thr %s rab %s · lag %s s · traverse %s s · %s · %s Hz" % (
                             rec.get("throttle"), rec.get("rabbit_button"), fmt(rec.get("lag_s")), fmt(tr),
                             rec.get("end_reason"), fmt(rec.get("write_hz"), 1)),
                         "result": ("%s°/s" % fmt(abs(r) / tr, 1)) if ok else ""})
        elif k == "ramp":
            c = rec.get("cmd_at_motion")
            rows.append({"kind": "ramp", "joint": j, "cmd": "dir %+d" % rec.get("dir", 0),
                         "detail": "thr %s rab %s · %s" % (rec.get("throttle"), rec.get("rabbit_button"), rec.get("end_reason")),
                         "result": ("dead band ≈ %s (-%s reaction)" % (fmt(abs(c)), fmt(0.2 * rec.get("ramp_rate_per_s", 0), 3)))
                         if c is not None else "no motion marked"})
    th = {(x.get("throttle"), x.get("rabbit_button")) for x in recs if x.get("kind") in ("sweep", "ramp")}
    return {"file": os.path.basename(calib_path()), "rows": rows[::-1],
            "warning": ("throttle / rabbit changed during this session: %s" % sorted(map(str, th))) if len(th) > 1 else ""}


def save_angles(b):
    joint = b.get("joint")
    if joint not in LINKS:
        raise ValueError("joint must be one of %s" % (LINKS,))
    chassis = num(b.get("chassis")) or 0.0
    ends, links_out, jd = b.get("ends", {}), {}, {}
    for end in ("-", "+"):
        e = ends.get(end, {})
        raw, links = {}, {}
        for ln in LINKS:
            p = num(e.get(ln, {}).get("pitch"))
            back = bool(e.get(ln, {}).get("back"))
            if p is not None:
                if not -90.0 <= p <= 90.0:
                    raise ValueError("%s pitch %s outside -90..90" % (ln, p))
                raw[ln] = {"pitch": p, "back": back}
                links[ln] = round(link_angle(p, back), 2)
        links_out[end] = {"pitch": raw, "dir_deg": links}
        jd[end] = joint_angle(joint, links, chassis)
    rng = round(abs(wrap180(jd["+"] - jd["-"])), 2) if jd["+"] is not None and jd["-"] is not None else None
    rec = {"kind": "angles", "id": "angles-%d" % int(time.time() * 1000),
           "wall_time": datetime.datetime.now().isoformat(timespec="milliseconds"), "joint": joint,
           "chassis_pitch": chassis, "ends": links_out, "joint_deg": jd, "range_deg": rng, "note": b.get("note") or ""}
    append(rec)
    return rec


# ---------------------------------------------------------------- page
PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>joint calib</title><style>
:root{color-scheme:dark}body{margin:0;background:#0e0f12;color:#e8e8ea;font:16px -apple-system,system-ui,sans-serif;padding:16px;max-width:760px;margin:auto}
h1{font-size:18px;margin:4px 0 10px}h2{font-size:15px;margin:18px 0 6px;color:#c9cbd1}
#st{padding:10px 12px;border-radius:10px;background:#1a1c22;font-variant-numeric:tabular-nums;font-size:14px;line-height:1.5}
#st.live{background:#3a1216;border:1px solid #d33}#dm{margin:12px 0;padding:14px;border-radius:10px;text-align:center;font-weight:700;background:#23262e}
#dm.on{background:#3d7a3a}#dm.armed{background:#6a5412}#dm.run{background:#8a2a12}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{padding:6px 4px;border-bottom:1px solid #23262e;text-align:left;font-size:14px}
.k{color:#9a9ca3;font-size:13px}.v{text-align:right;width:70px}.cmd{text-align:right;width:70px;color:#8fb3ff}
#stop{width:100%;height:60px;border:0;border-radius:12px;background:#c0262c;color:#fff;font:inherit;font-size:20px;font-weight:700;margin-top:14px}
#focus{color:#e0a030;font-size:13px;margin-top:8px}.warn{color:#e0a030;font-size:13px}
.box{background:#16181d;border-radius:10px;padding:10px 12px}.row{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;margin:6px 0}
button.b,select,input{font:inherit;font-size:14px;background:#23262e;color:#e8e8ea;border:1px solid #333844;border-radius:8px;padding:6px 10px}
button.b.sel{background:#2d4d8a;border-color:#4d78c9}input.n{width:64px}input.t{width:100%;box-sizing:border-box}
#runinfo{font-variant-numeric:tabular-nums;font-size:14px;margin-top:6px;min-height:1.4em}
</style></head><body>
<h1>joint calib <span class="k">jog: hold SPACE + keys · 1/2/3 speed · ESC/X stop</span></h1>
<div id="st">connecting…</div>
<div id="dm">SPACE not held: disarmed</div>
<h2>Jog</h2>
<div class="row">speed <button class="b" data-s="0.3">1 · 0.3</button><button class="b" data-s="0.5">2 · 0.5</button><button class="b" data-s="1">3 · 1.0</button></div>
<table id="t"></table>
<h2>Sweep / dead-band ramp</h2>
<div class="box">
 <div class="row"><label><input type="radio" name="kind" value="sweep" checked> sweep</label><label><input type="radio" name="kind" value="ramp"> ramp 0→0.6 in 12 s</label></div>
 <div class="row">joint <select id="rj"></select> direction <select id="rd"></select>
  <span id="magw">magnitude <select id="rm"><option>0.3</option><option selected>0.5</option><option>1.0</option></select></span>
  <span id="tow">timeout <input class="n" id="rt" type="number" value="15" min="1" max="30"> s</span>
  <span id="angw">swept angle <input class="n" id="ra" type="number" placeholder="90"> °</span></div>
 <div class="row"><button class="b" id="arm">Arm (then hold SPACE)</button><span class="k">M = motion started · S or release SPACE = end stop</span></div>
 <div id="runinfo"></div>
</div>
<h2>Angles (inclinometer pitch, deg from horizontal; + = far end higher)</h2>
<div class="box">
 <div class="row">joint measured <select id="aj"><option>boom</option><option>arm</option><option>bucket</option><option>thumb</option></select>
  chassis pitch <input class="n" id="ac" type="number" placeholder="0"> °</div>
 <table id="at"></table>
 <div class="row"><input class="t" id="an" placeholder="note (phone position, app, anything odd)"></div>
 <div class="row"><button class="b" id="asave">Save angles</button><span id="amsg" class="k"></span></div>
</div>
<button id="stop">STOP (relay open) · ESC</button>
<div id="focus"></div>
<h2>Results <span class="k" id="rf"></span></h2>
<div id="rw" class="warn"></div>
<table id="res"><thead><tr><th>kind</th><th>joint</th><th>cmd</th><th>detail</th><th>result</th></tr></thead><tbody></tbody></table>
<script>
// channel: [key for -1, key for +1, label -, label +]
const MAP={swing:["KeyA","KeyD","left","right"],arm:["KeyS","KeyW","in","out"],boom:["KeyK","KeyI","up","down"],
           bucket:["KeyJ","KeyL","curl","dump"],thumb:["KeyU","KeyO","open","clamp"]};
const LINKS=["boom","arm","bucket","thumb"];
const $=id=>document.getElementById(id);
const down=new Set();let mag=0.3,timer=null,moving=false;
let mode="jog",run=null,hbTimer=null;  // mode: jog | armed | running
const now=()=>performance.now()/1000;
const SID=Math.random().toString(36).slice(2);  // this page instance (its performance.now clock)
for(const [k,[n,p,ln,lp]] of Object.entries(MAP)){const r=document.createElement("tr");
 r.innerHTML=`<td><b>${k}</b></td><td class="k">${n.slice(3)} ${ln} · ${p.slice(3)} ${lp}</td><td class="cmd" id="c_${k}">0.00</td><td class="v" id="v_${k}">0.00</td>`;$("t").appendChild(r);
 $("rj").add(new Option(k,k));}
$("rj").value="boom";
function dirOpts(){const [n,p,ln,lp]=MAP[$("rj").value];$("rd").innerHTML="";$("rd").add(new Option(`+ (${lp})`,"1"));$("rd").add(new Option(`- (${ln})`,"-1"));}
function kind(){return document.querySelector("input[name=kind]:checked").value;}
function form(){dirOpts();const sw=kind()==="sweep";$("magw").style.display=sw?"":"none";$("tow").style.display=sw?"":"none";
 $("angw").style.display=(sw&&$("rj").value==="swing")?"":"none";}
$("rj").onchange=form;document.querySelectorAll("input[name=kind]").forEach(e=>e.onchange=form);form();
function setMag(s){mag=s;document.querySelectorAll("button[data-s]").forEach(b=>b.classList.toggle("sel",+b.dataset.s===s));}
document.querySelectorAll("button[data-s]").forEach(b=>b.onclick=()=>{setMag(+b.dataset.s);b.blur();});setMag(0.3);
function post(p,b){return fetch(p,{method:"POST",body:JSON.stringify(b||{})}).then(r=>r.json()).catch(()=>({error:"network"}));}
// ---- jog (the dashboard's teleop logic)
function cmd(){const c={};if(mode!=="jog"||!down.has("Space"))return c;
 for(const [k,[n,p]] of Object.entries(MAP)){const v=(down.has(p)?1:0)-(down.has(n)?1:0);if(v)c[k]=v*mag;}return c;}
function dmShow(){const d=$("dm");
 if(mode==="armed"){d.className="armed";d.textContent="ARMED: hold SPACE to start the "+run.kind+" (ESC cancels)";}
 else if(mode==="running"){d.className="run";d.textContent=run.kind.toUpperCase()+" RUNNING: M = motion · S / release SPACE = end stop · ESC = STOP";}
 else{d.className=down.has("Space")?"on":"";d.textContent=down.has("Space")?"SPACE held: armed":"SPACE not held: disarmed";}}
function update(){const c=cmd();dmShow();
 for(const k in MAP)$("c_"+k).textContent=(c[k]||0).toFixed(2);
 if(Object.keys(c).length){post("/api/write",c);moving=true;if(!timer)timer=setInterval(()=>{const c2=cmd();if(Object.keys(c2).length)post("/api/write",c2);},100);}
 else if(moving){clearInterval(timer);timer=null;moving=false;post("/api/neutral");}}
function release(){down.clear();update();}
// ---- sweep / ramp
function endRun(msg){clearInterval(hbTimer);hbTimer=null;mode="jog";run=null;down.clear();update();if(msg)$("runinfo").textContent=msg;refresh();}
function arm(){if(mode!=="jog")return;release();
 const k=kind(),j=$("rj").value;
 run={kind:k,joint:j,dir:+$("rd").value,mag:+$("rm").value,timeout_s:+$("rt").value||15,angle_deg:$("ra").value};
 if(run.timeout_s>30){run.timeout_s=30;$("rt").value=30;}
 mode="armed";document.activeElement&&document.activeElement.blur();dmShow();
 $("runinfo").textContent=`armed: ${k} ${j} ${run.dir>0?"+":"-"}${k==="sweep"?run.mag:"0→0.6"}; hold SPACE to go`;}
async function go(){run.ct=now();run.sid=SID;mode="running";dmShow();
 const r=await post("/api/run/start",run);
 if(r.error||!r.id){endRun("start refused: "+(r.error||"no id"));return;}
 run.id=r.id;
 if(run.markCt)post("/api/run/mark",{id:run.id,ct:run.markCt});
 if(run.pending){run.stopSent=true;post("/api/run/stop",{id:run.id,ct:run.pendingCt,reason:run.pending});}$("runinfo").textContent=`running ${run.id} · throttle ${r.throttle} · rabbit ${r.rabbit_button}`;
 hbTimer=setInterval(hb,100);}
async function hb(){if(!run||!run.id)return;const j=await post("/api/hb",{id:run.id,ct:now()});
 if(j.error==="network")return;
 if(j.running){$("runinfo").textContent=`${run.id} · ${j.elapsed.toFixed(2)} s · cmd ${(+j.cmd).toFixed(3)} · writes ${j.n_writes}${j.motion?" · motion marked":""}`;}
 else if(mode==="running"){const L=(j.last&&j.last.id===run.id)?j.last:{};endRun(`ended: ${L.end_reason||"?"} · lag ${L.lag_s??"-"} s · traverse ${L.traverse_s??"-"} s${L.cmd_at_motion!=null?" · cmd at motion "+L.cmd_at_motion:""} · ${L.write_hz??"-"} Hz`);}}
// the page stays "running" until a heartbeat reply says the server-side run has ended (then shows its record)
function runStop(reason){if(mode==="armed"){endRun("disarmed ("+reason+")");return;}
 if(mode!=="running"||!run||run.stopSent)return;
 if(!run.id){if(!run.pending){run.pending=reason;run.pendingCt=now();}return;}  // start in flight: go() sends it
 run.stopSent=true;post("/api/run/stop",{id:run.id,ct:now(),reason}).then(hb);}
function mark(){if(mode!=="running"||!run)return;if(!run.id){run.markCt=run.markCt||now();return;}
 post("/api/run/mark",{id:run.id,ct:now()});}
function halt(){down.clear();clearInterval(timer);timer=null;moving=false;
 post("/api/stop",{sid:SID,ct:now()});
 if(mode==="armed")endRun("disarmed (STOP)");else if(mode==="running"&&run)run.stopSent=true;
 update();}
$("arm").onclick=arm;
// ---- keys
window.addEventListener("keydown",e=>{if(e.code==="Escape"||e.code==="KeyX"){e.preventDefault();halt();return;}
 if(mode==="armed"||mode==="running"){if(e.code==="Space"||e.code==="KeyM"||e.code==="KeyS")e.preventDefault();if(e.repeat)return;
  if(e.code==="Space"&&mode==="armed"){down.add("Space");go();}
  else if(e.code==="KeyM")mark();else if(e.code==="KeyS")runStop("S");return;}
 if(e.target.matches&&e.target.matches("input,select,textarea"))return;
 if(e.code==="Digit1"||e.code==="Digit2"||e.code==="Digit3"){setMag([0.3,0.5,1.0][+e.code.slice(5)-1]);update();return;}
 if(e.code==="Space"||Object.values(MAP).some(m=>m[0]===e.code||m[1]===e.code)){e.preventDefault();if(e.repeat)return;down.add(e.code);update();}});
window.addEventListener("keyup",e=>{if(e.code==="Space")e.preventDefault();
 if(mode==="running"&&e.code==="Space"){down.delete("Space");runStop("space");return;}
 if(down.delete(e.code))update();});
window.addEventListener("blur",()=>{if(mode!=="jog")runStop("blur");release();$("focus").textContent="window lost focus: click here to drive";});
window.addEventListener("focus",()=>{$("focus").textContent="";});
document.addEventListener("visibilitychange",()=>{if(document.hidden){if(mode!=="jog")runStop("hidden");release();}});
$("stop").onclick=halt;
// ---- angles
for(const end of ["-","+"]){const r=document.createElement("tr");
 r.innerHTML=`<td class="k" id="al${end==="+"?"p":"m"}"></td>`+LINKS.map(l=>`<td>${l}<br><input class="n" type="number" id="a${end==="+"?"p":"m"}_${l}"> <label class="k"><input type="checkbox" id="b${end==="+"?"p":"m"}_${l}">back</label></td>`).join("");
 $("at").appendChild(r);}
function angLabels(){const [n,p,ln,lp]=MAP[$("aj").value];$("alm").textContent=`− end (${ln})`;$("alp").textContent=`+ end (${lp})`;}
$("aj").onchange=angLabels;angLabels();
$("asave").onclick=async()=>{const ends={};
 for(const [end,s] of [["-","m"],["+","p"]]){ends[end]={};for(const l of LINKS){const v=$(`a${s}_${l}`).value;if(v!=="")ends[end][l]={pitch:+v,back:$(`b${s}_${l}`).checked};}}
 const j=await post("/api/angles",{joint:$("aj").value,chassis:$("ac").value,ends,note:$("an").value});
 $("amsg").textContent=j.error?("error: "+j.error):`saved: ${j.joint} − end ${j.joint_deg["-"]}° · + end ${j.joint_deg["+"]}° · range ${j.range_deg??"-"}°`;
 $("asave").blur();refresh();};
// ---- results + state
async function refresh(){try{const j=await (await fetch("/api/results")).json();$("rf").textContent=j.file;$("rw").textContent=j.warning;
 const tb=$("res").tBodies[0];tb.innerHTML="";
 for(const r of j.rows){const tr=document.createElement("tr");for(const c of ["kind","joint","cmd","detail","result"]){const td=document.createElement("td");td.textContent=r[c];tr.appendChild(td);}tb.appendChild(tr);}}catch(e){}}
async function poll(){try{const j=await (await fetch("/api/state")).json();
 for(const k in MAP)$("v_"+k).textContent=(+j[k]).toFixed(2);
 $("st").className=j.live?"live":"";$("st").innerHTML=`<b>${j.live?"LIVE: relay closed":"idle: relay open"}</b>${j.veto?" · VETO: "+j.veto:""}<br>throttle ${j.throttle} · rabbit ${j.rabbit} · age ${(+j.age_s).toFixed(2)} s · <span class="k">blue = sent, white = readback</span>`;}
 catch(e){$("st").textContent="calib server unreachable";}setTimeout(poll,200);}
poll();refresh();
</script></body></html>"""


# ---------------------------------------------------------------- server
class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _j(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/":
            b = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif self.path == "/api/state":
            try:
                self._j(read_state())
            except Exception as ex:
                self._j({"error": str(ex)}, 500)
        elif self.path == "/api/results":
            self._j(results())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        ln = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(ln) if ln else b"{}"
        now = time.monotonic()
        try:
            b = json.loads(body or b"{}")
            if self.path == "/api/write":            # jog
                cmd = {k: (max(-1.0, min(1.0, float(v))) if math.isfinite(float(v)) else 0.0) for k, v in b.items() if k in rig.KEYS}
                with LOCK:
                    if RUN["cur"] is not None:
                        return self._j({"error": "a sweep / ramp is running"}, 409)
                    LAST["t"] = time.time()
                    LAST["cmd"] = cmd
                    res = rig.write(**cmd)
                self._j(res)
            elif self.path == "/api/neutral":
                with LOCK:
                    if RUN["cur"] is not None:
                        return self._j({"ok": True, "skipped": "run active"})
                    LAST["cmd"] = {}
                    res = rig.write()
                self._j(res)
            elif self.path == "/api/stop":           # ESC / X / STOP button: neutral + relay open, ends any run
                with LOCK:
                    LAST["cmd"] = {}
                    if b.get("sid") is not None and b.get("ct") is not None:
                        LAST_STOP.update(sid=b["sid"], ct=float(b["ct"]))
                    r = RUN["cur"]
                    if r is not None:
                        r["abort"] = "estop"
                        r["stop_key"] = "estop"
                    res = rig.stop()
                self._j(res)
            elif self.path == "/api/run/start":
                self._j(start_run(b))
            elif self.path == "/api/hb":
                r = cur_run(b.get("id"))
                if r is not None:
                    r["hb"] = now
                    seen(r, b.get("ct"), now)
                self._j(status(r))
            elif self.path == "/api/run/mark":       # M
                r = cur_run(b.get("id"))
                if r is not None:
                    seen(r, b.get("ct"), now)
                    if r["t_motion_rx"] is None:
                        r["t_motion_rx"], r["ct_motion"] = now, b.get("ct")
                        if r["kind"] == "ramp":
                            r["abort"] = r["abort"] or "motion"
                self._j(status(r))
            elif self.path == "/api/run/stop":       # S / SPACE release = end stop; blur / hidden / released = abort
                r = cur_run(b.get("id"))
                if r is not None:
                    seen(r, b.get("ct"), now)
                    reason = str(b.get("reason", "stop"))
                    if reason in ("S", "space"):
                        if r["t_stop_rx"] is None:
                            r["t_stop_rx"], r["ct_stop"] = now, b.get("ct")
                        r["abort"] = r["abort"] or "stop"
                    else:
                        r["abort"] = r["abort"] or reason
                    r["stop_key"] = r["stop_key"] or reason
                self._j(status(r))
            elif self.path == "/api/angles":
                self._j(save_angles(b))
            else:
                self._j({"error": "unknown"}, 404)
        except (ValueError, RuntimeError) as ex:
            self._j({"error": str(ex)}, 400)
        except Exception as ex:
            self._j({"error": str(ex)}, 500)


def watchdog():
    """Jog: page stopped sending (tab closed, wifi drop): neutral within WATCHDOG_S."""
    while True:
        time.sleep(0.05)
        with LOCK:
            if LAST["cmd"] and time.time() - LAST["t"] > WATCHDOG_S:
                LAST["cmd"] = {}
                try:
                    rig.write()
                except Exception:
                    pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--hb-timeout", type=float, default=HB_TIMEOUT_S,
                    help="sweep / ramp abort after this heartbeat gap (s); raise it only if the wifi keeps aborting runs")
    args = ap.parse_args()
    HB_TIMEOUT_S = args.hb_timeout
    threading.Thread(target=watchdog, daemon=True).start()
    print("joint calib on http://0.0.0.0:%d  (ctrl-c to quit; quitting sends stop)  results -> %s"
          % (args.port, CALIB_DIR), flush=True)
    try:
        ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()
    finally:
        with LOCK:
            r = RUN["cur"]
            if r is not None:
                r["abort"] = "estop"
            try:
                rig.stop()
            except Exception:
                pass
