# Network (remote) cameras in camview

A camera that is **not wired to the Pi** (a phone, an ESP32-CAM, an IP camera) can stream to the Pi over the LAN and
show up in camview like the USB and OAK cameras: it gets a stream id `net-<name>`, is listed in `/api/cameras`, and
is served at `/stream/net-<name>`. The dashboard's recorder and the policy runner read it exactly like any other
camera (assign it under **Record as** on its tile).

## Adding one

1. Put the camera on the **same LAN as the Pi** (see [Network](#network)) and find its stream URL.
2. On the Pi, create or edit `pi/act-athon/remote_cams.json` (next to `camview.py`):

   ```json
   {
     "_comment": "keys starting with _ are ignored",
     "side-phone": {"url": "http://192.168.1.50:8080/video", "label": "side (phone)"},
     "side-esp":   {"url": "http://192.168.1.60:81/stream", "label": "side (ESP32-CAM)"},
     "side-ipcam": {"url": "rtsp://user:pass@192.168.1.70:554/stream2", "width": 640, "height": 480, "fps": 15}
   }
   ```

   camview re-reads the file within a few seconds of a change: new entries start, removed ones stop, changed ones
   restart. No camview restart needed. A file with broken JSON is reported in camview's log and the cameras already
   running keep running. A missing file means no network cameras.

   Alternatively (or in addition) on the command line: `camview.py --remote side-phone=http://192.168.1.50:8080/video`
   (repeatable; overrides the same name in the file). `--remote-config PATH` uses another file.
3. Open camview (`http://<pi>:8100`): the tile appears in group **Network camera**. Check its fps, age and link
   stats (below) before recording.

The name becomes the stream id (`side-phone` → `net-side-phone`): letters, digits and `-` only. Keep it fixed: the
dashboard binds its **Record as** assignment to the stream id.

### Entry fields

| key | meaning |
|---|---|
| `url` | required. `http(s)://` MJPEG stream or snapshot, or `rtsp://` / any other URL ffmpeg reads. `user:pass@` in an http URL is sent as HTTP basic auth; it is shown as `***` in logs and `/api/cameras`. |
| `label` | tile label (default: the name). |
| `mode` | `auto` (default), `mjpeg`, `ffmpeg` or `poll`, see below. |
| `rotate` | `0`, `90`, `180`, `270` (clockwise). Forces the ffmpeg path (decode + re-encode on the Pi). |
| `width`, `height` | ffmpeg path only: output size (after rotation). One of them alone keeps the aspect ratio. |
| `fps` | ffmpeg path: caps the output rate. `poll`: requests per second (default 10). |

### Modes

| mode | chosen automatically for | what the Pi does | Pi CPU |
|---|---|---|---|
| `mjpeg` | `http(s)://` URLs | reads the `multipart/x-mixed-replace` stream over one persistent HTTP connection and passes each JPEG through **byte for byte** (Content-Length per part, or JPEG markers when a sender omits it; chunked transfer and junk between parts are handled) | ~none |
| `poll` | `http(s)://…/*.jpg` / `*.jpeg` | GETs a snapshot URL `fps` times a second (keep-alive) | ~none, but one HTTP round-trip per frame |
| `ffmpeg` | `rtsp://` and everything else, or `rotate` set | `ffmpeg -rtsp_transport tcp -i URL` → decode → rotate / scale / fps → JPEG `-q:v 6` | decode + encode on the Pi (see below) |

`mode` in `/api/cameras` says which is running (`net mjpeg passthrough`, `net snapshot poll`,
`net ffmpeg decode+encode`, plus `+rot90` etc.).

**CPU cost of the ffmpeg path.** The Pi 5 has no H.264 hardware decoder, so RTSP/H.264 is decoded in software.
Rough estimate (not yet measured on the Pi): a 640x480 15 fps H.264 stream decode + JPEG encode takes a fraction of
one core; a 1080p main stream costs a full core or more. Use the camera's low-resolution **sub-stream** (often
`/stream2` or `/sub`) at 640x480, and check `top` on the Pi while it runs next to recording. After each ffmpeg
(re)start the first ~1–2 s of frames arrive in a quick burst (ffmpeg's stream probe), then real time.
Rotating an MJPEG source costs the same decode + encode; prefer mounting the camera the right way up.

## Examples

- **Phone as an IP camera.** Any "IP camera" / "IP webcam" app that serves MJPEG over HTTP works. The app shows its
  address, typically `http://<phone-ip>:8080/…`; the MJPEG stream path is app specific (often `/video` or
  `/mjpeg`), its snapshot path often `/shot.jpg` (→ `poll` mode, use only if there is no stream). In the app:
  resolution 640x480, fixed ~15 fps, JPEG quality ~70–80, **lock exposure / white balance / focus** if it offers
  it, disable the screen lock / battery optimisation for the app, keep the phone plugged into a power bank.
  Give the phone a fixed IP (DHCP reservation on the router) so the URL doesn't change.
- **ESP32-CAM** (Arduino `CameraWebServer` example): stream at `http://<ip>:81/stream`, snapshot at
  `http://<ip>/capture` (no `.jpg` suffix: set `"mode": "poll"` explicitly if you want snapshots). Set the frame size
  to VGA (640x480) in its web UI or sketch. Its 2.4 GHz Wi-Fi and small antenna make it the least stable option;
  an external antenna helps.
- **RTSP IP camera**: `rtsp://user:pass@<ip>:554/<path>` (path per vendor; pick the sub-stream). Always goes through
  ffmpeg (TCP transport, so no UDP loss or NAT issues). If the camera also offers an MJPEG HTTP URL, prefer that:
  no decode on the Pi.

## Network

- The camera must be on the **same LAN as the Pi**, reachable at a local address. The Pi is on wired Ethernet
  (`eth0`, 192.168.1.151) and also has Wi-Fi (`wlan0`). Best: camera on the same Wi-Fi access point / router the
  Pi's Ethernet goes to, with a strong signal where the camera is mounted.
- **Not via Tailscale or the internet.** Relay hops, NAT traversal and upstream bandwidth add latency and multi-second
  gaps; a camera whose frames arrive late is worse than no camera for recording and for pi07 inference.
- Bandwidth: 640x480 @ 15 fps MJPEG is about **3–6 Mbit/s** per camera (depends on scene detail and JPEG quality;
  `/api/cameras` shows the measured `net.mbit_s`). Fine for any 802.11n/ac link with decent signal; a crowded
  2.4 GHz band at an outdoor event is the realistic risk.
- Power: the camera needs its own supply (phone on a power bank, ESP32 on a USB power bank). A phone on battery alone
  will throttle or sleep.

## Judging link stability (`/api/cameras`)

Each `net-*` entry has the usual `fps`, `age_s`, `error`, `mode`, plus a `net` block:

| field | meaning |
|---|---|
| `fps` | new frames per second over the last 2 s (drops to 0 when frames stop) |
| `age_s` | seconds since the last new frame arrived at camview |
| `error` | `null` when healthy; otherwise the connection problem (`connection refused; reconnecting in 2.0 s`, `connect timed out`, `timed out (no data for 2 s)`, `ffmpeg exited …`), or `stale: no new frame for X s` |
| `net.connected` | a connection is up |
| `net.stale` | no new frame for more than 1 s (`REMOTE_STALE_S`) |
| `net.reconnects`, `net.reconnects_60s` | reconnect attempts since camview started / in the last 60 s |
| `net.last_gap_s_max_60s` | longest gap between two new frames in the last 60 s (including the current one) |
| `net.stale_60s` | gaps longer than 1 s in the last 60 s |
| `net.mbit_s` | received bandwidth over the last 2 s |
| `net.dup_dropped` | frames dropped because they were byte-identical to the previous one (sender repeating itself) |
| `net.bad_parts` | stream parts that held no usable JPEG |
| `net.last_error`, `net.url` | last disconnect reason; the URL with credentials hidden |

Before a recording session, watch it for a minute with the machine and people where they will be:
`fps` steady at the camera's rate, `last_gap_s_max_60s` well under 0.5 s, `stale_60s` and `reconnects_60s` at 0.
Quick check from a laptop: `curl -s http://<pi>:8100/api/cameras | python3 -m json.tool`.

## Stale-frame rules

- camview **never re-sends an old frame as new**, for every camera (USB, OAK and network): `/stream/<id>` writes a
  part only when a new frame arrived. A stalled camera sends nothing at all (earlier versions re-sent the last JPEG
  every 2 s, which made a frozen camera look fresh to readers that time frames by arrival).
- Every part carries `X-Frame-Seq` (strictly increasing per stream) and `X-Capture-Time` (epoch seconds when the
  frame arrived at camview) after `Content-Type` / `Content-Length`. Readers that ignore unknown part headers are
  unaffected.
- A new client gets the current frame immediately only if it is less than 1 s old (`STREAM_FIRST_MAX_AGE_S`);
  otherwise it waits for the next new one.
- Network sources: a JPEG byte-identical to the previous one is dropped (not counted, not pushed), so `age_s`, fps
  and seq follow genuinely new frames. After 1 s without a new frame the stream's `error` says `stale`; the stream
  stays listed (dashboard assignments don't flip to "missing"), and the dashboard's own stall detection refuses to
  start or flags the recording. Connections are dropped after 3 s connect / 2 s read silence (ffmpeg path: 10 s for
  the first frame, 5 s between frames) and retried forever with a backoff of 0.5 s doubling to 5 s. A dead camera
  never blocks the other cameras or the HTTP server.

## Same camera, same pose

A network camera used for training must be the **same device, at the same mount position and angle, with the same
settings (resolution, exposure, white balance, rotation)** at the demo. Tape-mark the mount. Adding a camera to the
recorded set changes the PI contract / training profile camera set (AGENTS.md: contracts are immutable → new
contract), so decide before collecting the training demos.

## Stability verdict

1. **Wired** (USB on the Pi): best. No link to lose.
2. **Same-LAN Wi-Fi** (this feature): OK for recording and inference, *with* the stale detection above: check
   `last_gap_s_max_60s` / `reconnects_60s` before each session, and discard episodes where the camera stalled.
   Expect occasional sub-second hiccups outdoors.
3. **Internet / Tailscale**: not OK. Latency and multi-second gaps are normal there.
