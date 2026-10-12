"""worldcam -- serve a laptop webcam (the static third-person "world" camera) to the excavator Pi as camview-style MJPEG.

Runs on the LAPTOP the camera is plugged into (macOS / Linux / Windows). Needs Python 3.9+ and OpenCV:
    pip3 install opencv-python
    python3 worldcam.py --list                  # which camera index is which (writes list_<i>.jpg to look at)
    caffeinate -dimsu python3 worldcam.py --cam 1   # macOS: caffeinate keeps the laptop awake while it streams
then open http://localhost:8100 to check the picture. On the Pi, camview.py relays it as stream "net-world" once
remote_cams.json has   {"world": "http://<laptop LAN ip>:8100/stream/world"}   and the dashboard records it as "world".

macOS: the first run asks for camera access for your terminal app (System Settings > Privacy & Security > Camera),
and may ask to accept incoming connections for python: allow both, then restart this script.

Endpoints (same shape as camview.py, so the Pi's relay, latency.py and a browser all read it the same way):
    GET /                 preview page
    GET /api/cameras      [{"id": "world", ...fps, age_s, width, height}]
    GET /stream/world     multipart MJPEG; every part has Content-Length, X-Frame-Seq and X-Capture-Time
                          (epoch s on THIS laptop's clock, taken when the frame came off the camera)
    GET /snapshot.jpg     newest frame
    GET /api/time         {"t": epoch s now}: the Pi's latency.py estimates the laptop-Pi clock offset from it

Only NEW frames are sent (a slow client skips frames, it never gets a backlog). Frames are kept at the camera's aspect
ratio and scaled down to --height (default 480, like every Pi camera); the square crop happens in conversion.
KEEP THE SETTINGS FIXED across every episode of a dataset and at inference: camera index, resolution, fps, quality.

Luxonis OAK (e.g. OAK-D S2) as the world camera: it is not a webcam, OpenCV can't open it. Use --oak (needs depthai,
same version as the Pi: pip3 install depthai==3.10.0). Color (CAM_A) is ISP-scaled to --width x --height and
MJPEG-encoded ON the OAK, like camview's OAK path, so only JPEGs cross USB. X-Capture-Time is the arrival time here,
the same convention camview uses for its own OAK / USB cameras. Only one program can hold an OAK: close depthai
viewers / examples first.
    caffeinate -dimsu python3 worldcam.py --oak               # first OAK found
    caffeinate -dimsu python3 worldcam.py --oak <device id>   # a specific one (ids: --list)
"""
import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import cv2
except ImportError:
    sys.exit("OpenCV missing: pip3 install opencv-python")
try:                        # optional: only --oak (Luxonis OAK) needs it
    import depthai as dai
except ImportError:
    dai = None

STREAM_ID = "world"
BOUNDARY = "worldcamframe"
REOPEN_S = 2.0          # no frame from the camera this long -> reopen it


class Camera:
    """Capture thread: newest JPEG + its capture time; clients wait on `cond` for a new seq."""

    def __init__(self, args):
        self.a = args
        self.cond = threading.Condition()
        self.jpeg, self.seq, self.t_cap = None, 0, 0.0
        self.w = self.h = 0
        self.fps, self.error, self.opened = 0.0, None, 0
        threading.Thread(target=self._run, daemon=True).start()

    def _open(self):
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.a.cam, backend)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.a.height)
        cap.set(cv2.CAP_PROP_FPS, self.a.fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # ignored by some backends; the read loop keeps up anyway
        except Exception:
            pass
        return cap

    def _run(self):
        rot = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(self.a.rotate)
        min_dt = 1.0 / self.a.fps
        while True:
            cap = self._open()
            if cap is None:
                self.error = "camera %d did not open (wrong --cam? camera permission?)" % self.a.cam
                print(self.error, flush=True)
                time.sleep(REOPEN_S)
                continue
            self.opened += 1
            self.error = None
            t_last_ok = t_last_pub = time.monotonic()
            while True:
                ok, frame = cap.read()
                t_cap = time.time()                  # off the camera: the timestamp every client gets
                now = time.monotonic()
                if not ok or frame is None:
                    if now - t_last_ok > REOPEN_S:
                        self.error = "no frames for %.0f s: reopening camera %d" % (REOPEN_S, self.a.cam)
                        print(self.error, flush=True)
                        break
                    time.sleep(0.005)
                    continue
                t_last_ok = now
                if now - t_last_pub < 0.9 * min_dt:  # camera faster than --fps: drop (backend ignored CAP_PROP_FPS)
                    continue
                if rot is not None:
                    frame = cv2.rotate(frame, rot)
                h, w = frame.shape[:2]
                if h > self.a.height:                # keep the aspect ratio, match the Pi cameras' height
                    s = self.a.height / h
                    frame = cv2.resize(frame, (round(w * s), self.a.height), interpolation=cv2.INTER_AREA)
                    h, w = frame.shape[:2]
                ok, buf = cv2.imencode(".jpg", frame, (cv2.IMWRITE_JPEG_QUALITY, self.a.quality))
                if not ok:
                    continue
                dt = now - t_last_pub
                t_last_pub = now
                with self.cond:
                    self.fps = 1.0 / dt if not self.fps else 0.9 * self.fps + 0.1 / max(dt, 1e-3)
                    self.jpeg, self.t_cap, self.w, self.h = buf.tobytes(), t_cap, w, h
                    self.seq += 1
                    self.cond.notify_all()
            cap.release()
            time.sleep(0.5)

    def wait_new(self, after, timeout=2.0):
        with self.cond:
            if self.seq == after:
                self.cond.wait(timeout)
            return self.jpeg, self.seq, self.t_cap

    def info(self):
        age = time.time() - self.t_cap if self.t_cap else None
        return {"id": STREAM_ID, "label": "world (laptop cam %d)" % self.a.cam, "group": "worldcam", "mode": "opencv",
                "error": self.error, "fps": round(self.fps, 1), "age_s": None if age is None else round(age, 3),
                "width": self.w, "height": self.h, "seq": self.seq, "reopens": max(0, self.opened - 1)}


class OakCamera(Camera):
    """Luxonis OAK color camera via depthai: on-device MJPEG, one JPEG per message. Same interface as Camera; the
    device is reopened (after REOPEN_S) whenever it fails or goes silent, e.g. unplugged."""

    def _run(self):
        if dai is None:
            raise ImportError("depthai not installed: pip3 install depthai==3.10.0 (needed for --oak)")
        rot = self.a.rotate
        while True:
            try:
                infos = [d for d in dai.Device.getAllAvailableDevices()
                         if self.a.oak in ("", d.getDeviceId(), d.name)]
                if not infos:
                    raise RuntimeError("no free OAK%s (unplugged, or another program holds it?)"
                                       % (" " + self.a.oak if self.a.oak else ""))
                with dai.Device(infos[0]) as device:
                    self.dev_id = infos[0].getDeviceId()
                    with dai.Pipeline(device) as pipeline:
                        color = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
                        if rot == 180:   # on the sensor, free
                            color.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
                        out = color.requestOutput((self.a.width, self.a.height), dai.ImgFrame.Type.NV12,
                                                  fps=self.a.fps)
                        enc = pipeline.create(dai.node.VideoEncoder).build(
                            out, frameRate=self.a.fps, profile=dai.VideoEncoderProperties.Profile.MJPEG)
                        enc.setQuality(self.a.quality)
                        q = enc.out.createOutputQueue(maxSize=1, blocking=False)
                        pipeline.start()
                        self.opened += 1
                        self.error = None
                        print("worldcam: OAK %s streaming" % self.dev_id, flush=True)
                        self._pump(pipeline, q)
            except Exception as ex:
                self.error = "OAK: %s: %s" % (type(ex).__name__, ex)
                print(self.error, flush=True)
            time.sleep(REOPEN_S)

    def _pump(self, pipeline, q):
        t_last_ok = t_last_pub = time.monotonic()
        while pipeline.isRunning():
            msg = q.tryGet()
            now = time.monotonic()
            if msg is None:
                if now - t_last_ok > REOPEN_S:
                    raise RuntimeError("no frames for %.0f s" % REOPEN_S)
                time.sleep(0.002)
                continue
            t_cap = time.time()                          # arrival here, like camview's OAK / USB cameras
            t_last_ok = now
            jpeg = bytes(msg.getData())
            if jpeg[:2] != b"\xff\xd8":
                continue
            if not self.w:                               # size from the first JPEG (SOF0 marker)
                i = jpeg.find(b"\xff\xc0")
                if i > 0:
                    self.h, self.w = int.from_bytes(jpeg[i + 5:i + 7], "big"), int.from_bytes(jpeg[i + 7:i + 9], "big")
            dt = now - t_last_pub
            t_last_pub = now
            with self.cond:
                self.fps = 1.0 / dt if not self.fps else 0.9 * self.fps + 0.1 / max(dt, 1e-3)
                self.jpeg, self.t_cap = jpeg, t_cap
                self.seq += 1
                self.cond.notify_all()
        raise RuntimeError("pipeline stopped")

    def info(self):
        d = super().info()
        d["label"], d["mode"] = "world (OAK %s)" % getattr(self, "dev_id", self.a.oak or "?"), "oak mjpeg"
        return d


PAGE = """<!doctype html><meta charset=utf-8><title>worldcam</title>
<style>body{background:#111;color:#ddd;font:14px system-ui;margin:16px}img{max-width:100%%;border:1px solid #333}</style>
<h3>worldcam: %s</h3><p id=s>...</p><img src="/stream/world">
<script>setInterval(async()=>{const c=(await (await fetch('/api/cameras')).json())[0];
document.getElementById('s').textContent=`${c.width}x${c.height} @ ${c.fps} fps, frame age ${c.age_s}s`+
(c.error?' | '+c.error:'')},1000)</script>"""


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    cam = None

    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/api/time":
            return self._send(200, "application/json", json.dumps({"t": time.time()}).encode())
        if p == "/":
            return self._send(200, "text/html", (PAGE % self.cam.info()["label"]).encode())
        if p == "/api/cameras":
            return self._send(200, "application/json", json.dumps([self.cam.info()]).encode())
        if p == "/snapshot.jpg":
            jpeg, _, t = self.cam.wait_new(-1, 0)
            if jpeg is None:
                return self._send(503, "text/plain", b"no frame yet")
            return self._send(200, "image/jpeg", jpeg)
        if p in ("/stream/" + STREAM_ID, "/stream", "/video"):
            return self._stream()
        self._send(404, "text/plain", b"not found")

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=" + BOUNDARY)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        seq = 0
        try:
            while True:
                jpeg, new, t_cap = self.cam.wait_new(seq)
                if jpeg is None or new == seq:
                    continue
                seq = new
                head = ("--%s\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\nX-Frame-Seq: %d\r\n"
                        "X-Capture-Time: %.6f\r\n\r\n" % (BOUNDARY, len(jpeg), seq, t_cap)).encode()
                self.wfile.write(head + jpeg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


def list_cameras():
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    for i in range(6):
        cap = cv2.VideoCapture(i, backend)
        if not cap.isOpened():
            print("cam %d: -" % i)
            continue
        ok, frame = False, None
        for _ in range(10):                  # some cameras return black / nothing for the first reads
            ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            cv2.imwrite("list_%d.jpg" % i, frame)
            print("cam %d: %dx%d  -> list_%d.jpg" % (i, frame.shape[1], frame.shape[0], i))
        else:
            print("cam %d: opens but gives no frame" % i)
    print("On a Mac, 0 is usually the built-in FaceTime camera; the USB camera is then 1 (or 2 with an iPhone nearby).")
    if dai is None:
        return print("depthai not installed: Luxonis OAK cameras not listed (pip3 install depthai==3.10.0)")
    for d in dai.Device.getAllAvailableDevices():
        print("OAK %s (%s, %s)  -> --oak %s" % (d.getDeviceId(), d.name, d.state.name, d.getDeviceId()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cam", type=int, default=0, help="OpenCV camera index (see --list)")
    ap.add_argument("--oak", nargs="?", const="", default=None, metavar="DEVICE_ID",
                    help="Luxonis OAK color camera via depthai instead of OpenCV (optional device id, see --list)")
    ap.add_argument("--list", action="store_true", help="probe camera indices 0-5, save a frame of each, exit")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--width", type=int, default=640, help="requested capture width")
    ap.add_argument("--height", type=int, default=480, help="requested capture height; frames are scaled down to it")
    ap.add_argument("--fps", type=float, default=30.0, help="max frame rate (default 30: the recorder at 15 Hz then "
                    "finds a frame < 33 ms old)")
    ap.add_argument("--quality", type=int, default=80, help="JPEG quality (camview uses 80)")
    ap.add_argument("--rotate", type=int, default=0, choices=(0, 90, 180, 270))
    a = ap.parse_args()
    if a.list:
        return list_cameras()
    if a.oak is not None and a.rotate not in (0, 180):
        ap.error("--oak supports --rotate 0 or 180 only")
    H.cam = OakCamera(a) if a.oak is not None else Camera(a)
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), H)
    srv.daemon_threads = True
    print("worldcam: %s on http://0.0.0.0:%d  (stream: /stream/%s, time: /api/time)  Ctrl-C to stop"
          % ("OAK" if a.oak is not None else "camera %d" % a.cam, a.port, STREAM_ID), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
