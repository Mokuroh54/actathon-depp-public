# Dashboard (cameras + status + teleop + recording)

`pi/act-athon/dashboard.py` runs on the excavator's Pi and serves one page that shows:

- a camera grid (every stream camview.py serves),
- the gate status (relay, page ⇄ Pi round trip; a gate veto or stale gate data shows up as a warning),
- what the gate is outputting on each channel (`/state` `sent`),
- a **Teleoperate** switch for keyboard / gamepad driving (positioning, tests; never recorded),
- a **Record episode** panel: demonstrations for training, driven by an operator **in the cab**,
- a **Cab monitor**: the cab joystick inputs the gate reads, live,
- a STOP button.

It uses only the standard library. It does not open cameras itself: `camview.py` (a teammate's camera server, port
8100) is the only process that does, because a V4L2 camera can't be opened by two programs at once. The dashboard
starts, restarts and stops camview itself (below): you don't run camview by hand.

## Run

**One command from the laptop** (world camera + tunnel here, `start_dashboard.sh` on the Pi, in the foreground):

```bash
scripts/excavator_up.sh                 # = --rec-fps 15; any dashboard.py flags pass through
scripts/excavator_up.sh --no-world      # without the laptop world camera
```

It starts `worldcam.py --oak --fps 15` (python from `WORLDCAM_PYTHON`, default `~/.venvs/worldcam/bin/python`) and
`ssh -R 18100:127.0.0.1:8100 excavator` unless they are already running (then they are reused and left running),
refuses to start if a dashboard already holds :8080 on the Pi, and stops what it started when the dashboard quits
(Ctrl-C). The steps it replaces follow.

Both files must be in `~/act-athon` on the Pi (`scp pi/act-athon/dashboard.py excavator:act-athon/`). Start only the
dashboard; it starts camview for you:

```bash
ssh excavator 'cd act-athon && python3 dashboard.py'              # page :8080, cameras (camview) :8100
```

Then open http://<pi-host>:8080 and choose the cameras to record on their tiles (**Record as**, below). No
camera flags are needed. Ctrl-C on the dashboard (or closing its ssh session, or `kill`) sends STOP and stops camview.

### camview is managed by the dashboard (`--camview auto`, the default)

- **Start.** The dashboard runs `camview.py` (next to `dashboard.py`) as its child with the venv's python,
  `.venv/bin/python3` (it has depthai: without it there is no OAK). If that python is missing it falls back to its own
  python and says so in the camview chip (the OAK stream is then missing).
- **A camview you started by hand is taken over.** If a `camview.py` of the same user already listens on port 8100
  (an old ssh session, a crashed dashboard), the dashboard sends it SIGTERM (clean shutdown), waits up to 6 s (then
  SIGKILL) and starts its own. It prints and logs which pid it replaced. Anything else on port 8100 (another user's
  process, a tunnel) is **never** signalled: the dashboard uses it as is and the chip says **external (not managed)**.
- **Restart.** The dashboard checks camview's `/api/cameras` every 2 s. If camview exits, or doesn't answer for 10 s,
  it is stopped (SIGTERM, SIGKILL after 6 s) and started again, after 1, 2, 4 ... up to 30 s (back to 1 s after a
  minute of running fine). It is **never** restarted while an episode is recording, the policy runner is running or
  Policy is on: the chip turns red and says what it is waiting for, and camview restarts as soon as that ends.
  (Recording Start already refuses cameras that aren't delivering.)
- **Stop.** Quitting the dashboard (Ctrl-C, SIGTERM, SIGHUP) stops camview (SIGTERM, SIGKILL after 6 s). If the
  dashboard dies without cleaning up (SIGKILL, crash), Linux sends camview SIGTERM anyway (`PR_SET_PDEATHSIG`), so no
  orphaned camview keeps the cameras.
- **On the page**, above the camera tiles: the camview chip (managed / external / off, pid, uptime, number of
  restarts; hover for the last error and the log path), a **Restart camview** button (every stream drops for a few
  seconds; refused while recording, while the runner runs or while Policy is on) and **camview log**, the last 20
  lines of its output (`GET /api/camview` returns the same with the status).
- **Log:** `~/act-athon/logs/camview.log` (camview's output plus the dashboard's `[dashboard ...]` lines about starts,
  stops and takeovers). Over 5 MB it moves to `camview.log.1` at the next start.
  `ssh excavator 'tail -f act-athon/logs/camview.log'`

`--camview external` uses whatever answers on `--camview-url` and never starts or signals anything (camview run by
hand, or on another host). `--camview off` turns the supervisor off completely. To run camview by hand for debugging,
start the dashboard with one of these, or stop the dashboard first: a normal dashboard start takes it over.

## Cameras

The dashboard asks camview for its streams every 2 s (`GET /api/cameras`), so cameras that camview picks up or loses
(hot-plug) show up or go away on their own. The tiles load the video straight from camview
(`http://<pi>:8100/stream/<id>`), on a different port from the page, so video never delays teleop requests.

### Choosing the recorded cameras (Record as)

Every camview stream gets a tile. Its caption shows the stream id, the camera, and its real resolution. Under the
caption, **Record as** chooses whether that stream is recorded and under which name:

| Record as | dataset key | pi0.5 image slot | pi07 training profile |
|---|---|---|---|
| off | not recorded (view only) | | |
| `front` | `observation.images.front` | `base_0_rgb` | base camera |
| `wrist` | `observation.images.wrist` | `left_wrist_0_rgb` | wrist camera |
| `side` | `observation.images.side` | `right_wrist_0_rgb` | |

This name → slot mapping must be the same in training and inference (AGENTS.md). It is `SLOTS` in `dashboard.py`,
and each episode's `meta.json` records it (`slot` per camera, `camera_slots`).

- A name is on **one** stream at a time. Giving `front` to another stream takes it off the old one; the message line
  above the tiles says what moved ("front moved from usb-3-1-1-0 to ..."). A stream has at most one name.
- Recorded tiles have a red frame and show fps, JPEG size and Mbit/s. Unassigned tiles are only loaded by your
  browser; the dashboard itself reads only the recorded streams.
- The choice is kept on the Pi (`~/act-athon/cam_assign.json`, next to `dashboard.py`; `--assign-file` to move it), so
  a dashboard or Pi restart keeps it. It is bound to the camview **stream id** (for USB cameras that is the USB port:
  keep each camera in its port).
- While an episode is recording, and while the policy runner is running, the selectors are locked (greyed out; the
  API answers 409). The runner reads its cameras from this assignment once, at start: the runner panel shows the
  mapping (`front ← usb-... (C270 HD WEBCAM) · ...`; before start from the current assignment for the cameras the
  selected dataset was trained on, missing or stalled ones in red; while running the mapping it was started with).
  It ends with `sync <mode>`: the camsync frame selection (`--sync aligned|latest`) passed to `control_loop.py`,
  currently the dashboard's own `--sync` (the mode it records with); it must match the training episodes'
  meta.json `"sync"` (episodes without one were recorded as `latest`).
- Cameras come and go (hot-plug, or another process takes one). A recorded stream that camview no longer lists stays
  as a greyed tile, **assigned but missing**; one that stops delivering frames is greyed as **assigned but stalled**.
  Start is refused until it is back, or until you set its tile to off.
- Start is refused while no camera is assigned.

`--cam NAME=ID` (optional) sets the same thing from the command line: at startup it overrides the saved assignment for
that name and writes it to the file. `NAME` is `front`, `wrist` or `side`; `ID` is a camview stream id or a pattern
(`lux-*-color`, bound to the first matching stream).

For each recorded stream the dashboard keeps one reader open to camview (over loopback) and remembers the newest JPEG
and when it arrived. When camview loses a camera, the reader waits and reconnects. When a camera freezes, camview keeps
re-sending its last picture every 2 s; the dashboard doesn't count those repeats as new frames, so the tile shows
**stalled** and the recorded frame age grows.

Hide streams nobody needs with `--exclude PATTERN` (for example `--exclude 'lux-*-depth'`; assigned streams are always
shown), or show only the assigned ones with `--no-auto`.

### One camera set per dataset

The camera set is the recorded names, each with its stream id and resolution. Episodes with different camera sets
must go into separate datasets (the converter also refuses to mix camera names or frame sizes). When you press
Start with a set that differs from the one used by the **last saved** episode (read from its `meta.json` in the
episodes folder, so it survives restarts), the page asks:

> Camera set changed (was: front=usb-3-1-1-0 640x480, wrist=..., now: ...). Episodes with different camera sets must
> go into separate datasets. Start anyway?

Cancel leaves everything as it was. OK records the episode and writes the old set in its `meta.json`
(`camera_set_changed_from`), so you can tell where the new dataset starts. Convert those episodes into a new `--root`.
Discarded episodes don't count. (API: `POST /api/rec {"op": "start", ..., "confirm_new_camera_set": true}`.)

### What camview captures

Every camera is captured at **640x480, 15 fps** (constants at the top of `camview.py`: `USB_FPS`, `OAK_FPS`,
`COLOR_RES`, `MONO_RES`):

- USB webcams (C270): the camera's own MJPG, passed through unchanged (no encoding on the Pi). A camera without MJPG
  falls back to raw YUYV + ffmpeg's JPEG encoder, also at 15 fps. `mode` in `/api/cameras` says which.
- OAK color, left, right: 640x480 (color is scaled on the camera from the full 4:3 sensor, so the full field of view)
  and **JPEG-encoded on the OAK** (depthai VideoEncoder, MJPEG); the Pi forwards the bytes unchanged (`mode`
  "oak mjpeg"). If the encoder can't be set up, or no JPEG arrives within 10 s of start, camview restarts that OAK
  with Pi-side encoding (`mode` "pi encode", raw frames over USB) and says so in its log; that holds until camview
  restarts. Depth is always colorized and encoded on the Pi. Each OAK output has its own reader thread, so every
  stream runs at its own rate.
- The OAK is **mounted upside down**. camview rotates every OAK stream 180° (constant `OAK_ROTATE_180`): color on
  the sensor, left/right on the OAK before the encoder, depth and the Pi-encode fallback on the Pi (`mode` gets
  "+rot", e.g. "oak mjpeg+rot"). **Episodes recorded before this change are upside down: don't mix them with later
  ones.**
- Only the OAK streams in `OAK_STREAMS` run (default: color only), to cut the OAK's power draw; the mono sensors and
  stereo only start when left, right or depth is listed.
- The OAK is on **USB 2** at the moment (sharing a hub with the C270). With on-device JPEG that is enough; the Pi-encode
  fallback sends raw frames and may not reach 15 fps on USB 2. If a USB 3 port and cable are available, plug the OAK
  into them.

`/api/cameras` and the camview page show each stream's measured fps and `mode`.

### Square crop (training images)

Frames are recorded at the camera's resolution (640x480 from camview's USB and OAK streams), uncropped. The images
the policy trains on and sees at deployment are a **square crop**,
made in conversion and in `RigMachine` (the same code, `act.rig_machine.prepare_image`): by default the centered
480x480 (x = 80..560, full height). Each recorded tile on the dashboard draws this default square as a dashed outline:
aim the cameras with it. **Everything that matters (bucket, thumb, the object, the drop zone) must be inside the
square.** The crop can be shifted per camera at conversion (`--crop`, below) without re-recording, as long as
it stays inside the frame.

The outline is only drawn on 4:3 frames. A stream with another aspect ratio says **non-4:3: set --crop in conversion**
under its tile: the default centered square may be the wrong part of that frame. Example: the ZED 2i streams
side-by-side stereo at 1344x376 (two 672x376 eyes). The centered square would straddle both eyes, so crop one eye in
conversion, e.g. the middle of the left eye: `--crop NAME=148,0,376` (x = (672 - 376) / 2).

## Flags

| flag | default | |
|---|---|---|
| `--port` | 8080 | page + API |
| `--camview` | `auto` | `auto`: start / restart / stop camview here (a camview.py of this user already on the port is taken over); `external`: use what answers on `--camview-url`, never start or signal it; `off`: no supervisor |
| `--camview-python` | `.venv/bin/python3` next to `dashboard.py` | python for the managed camview (needs depthai for the OAK); falls back to the dashboard's python if missing |
| `--camview-script` | `camview.py` next to `dashboard.py` | camview to run (its folder is the working directory) |
| `--camview-log` | `logs/camview.log` next to `dashboard.py` | managed camview's output |
| `--camview-url` | `http://127.0.0.1:8100` | camview, as the dashboard reaches it (the managed camview gets this URL's port; `auto` needs a loopback URL) |
| `--camview-public-url` | page host, port 8100 | camview, as the browser reaches it (e.g. through a tunnel) |
| `--cam NAME=ID` | | optional: record stream `ID` as `NAME` (`front` / `wrist` / `side`), overriding the saved assignment at startup; repeatable |
| `--assign-file PATH` | `cam_assign.json` next to `dashboard.py` | where the Record as choices are kept |
| `--exclude PATTERN` | | hide unassigned camview streams whose id or label matches; repeatable |
| `--no-auto` | off | show only the assigned streams |
| `--rec-fps` | 15 | recording steps per second (the camera rate) |
| `--episode-s` | 60 | auto-save an episode after this long |
| `--episodes-dir` | `~/act-athon/episodes` | where episodes are written |
| `--cab-map PATH.json` | built-in `CAB_MAP` | override the cab joystick map / deadband (see below) |

## Bandwidth and disk

The tiles get camview's full frame rate (15 fps); the dashboard can't lower it. Each recorded tile's caption shows the
JPEG size, fps and Mbit/s. Rough sizes:

- C270 (its own MJPG, 640x480): about 10-30 KB per frame (10 KB measured), so **1-4 Mbit/s** per tile.
- OAK color (640x480, JPEG from the OAK): about 20-40 KB per frame (estimate, not yet measured), so **2-5 Mbit/s** per
  tile. The laptop link is poor (88 ms RTT, spikes to 430 ms, 15 % loss), so turn off what you don't watch.

While driving, turn the streams off with the switch in the Cameras header if the link can't carry them, and
`--exclude` streams nobody watches. Turning the streams off doesn't affect recording.

Recording at `--rec-fps 15` stores one JPEG per recorded camera per step (900 per minute): about **9-27 MB per minute**
for the C270 and **18-36 MB per minute** for the OAK color stream (plus about 0.6 MB/min for `steps.jsonl`), so
roughly 30-65 MB per minute for both. The panel shows the free space; recording refuses to start below 300 MB.

## Recording from the cab

Demonstrations are recorded from a **human operator in the cab, driving with the machine's own joysticks**. Keyboard
and gamepad driving through this page is never recorded: it is bang-bang (-1 / 0 / +1 per key), while cab demos are
analog, and a policy should learn from one kind only.

While an episode is open the dashboard sends **nothing** to the gate:

- **Start** is refused while Teleoperate is on ("switch it off first").
- Teleoperate can't be switched on until the episode is saved or discarded.
- The relay stays open. Nobody else sends commands either (no policy run, no second dashboard).
- **STOP / Esc / X** still work at any time: they send `/estop` (neutral + relay open) as always, and
  **discard** the episode being recorded (the stop goes to the gate first). Only the page's STOP discards: the
  `/api/stop` the policy runner or `replay_episode.py` sends when it exits stops the gate but keeps the episode. The cab operator's own controls and the kill switch are what stop the
  machine during a cab demo (whether `/estop` affects cab driving hasn't been tested).

### Before the first episode: check the map with the Cab monitor

The gate's `/state` has a `controls` dict: the live inputs it reads (CAN + ADC + buttons), including the cab
joysticks. The dashboard turns them into the 5 rig channels with `CAB_MAP` in `dashboard.py`:

| channel (rig sign) | raw field | sign |
|---|---|---|
| swing (+ = house right) | `body` | +1 |
| boom (+ = down) | `boom` | +1 |
| arm (+ = out) | `arm` | +1 |
| bucket (+ = dump) | `bucket` | +1 |
| thumb (+ = clamp) | `thumb` | +1 |

plus a deadband of 0.08 (inside ±0.08 the action is 0; rest offsets are within ±0.04), clipped to -1..1. This map was
verified on the machine on 2026-09-26 (an operator calling out each move, relay open, `sent` = 0 throughout).
`controls.swing` and `controls.blade` are **not** house swing (they move only ±0.2, together): ignore them.

Check it at the start of each session anyway:

1. Open **Cab monitor** (under "Cab joysticks → recorded action"). It lists every numeric `controls` field live
   (`curl http://<pi>:8080/api/cab` returns the same). It only reads.
2. The operator moves one joystick axis at a time and calls it out. The matching field should move, the bar for the
   right channel should move in the right direction (rig signs above), and nothing else should move beyond the
   deadband (highlighted fields).
3. If a field or sign is wrong, fix it without editing code: write a JSON file and restart the dashboard with
   `--cab-map`:

   ```json
   {"arm": ["arm", -1.0], "deadband": 0.1}
   ```

   Listed channels replace the built-in entries, the rest are kept. An entry is `[raw field, sign]` or
   `[raw field, sign, rest offset]` (action = sign x (raw - offset)); `deadband` is a number or `{channel: number}`.
   The panel shows the map in use and marks entries that came from the file.

Every step also stores the whole raw `controls` dict, so a map that turns out wrong later can be fixed at conversion
(`convert_pi_episodes.py --cab-map`) without re-recording.

### Record

1. Open the **Record** tab (this switches Teleoperate / Inference off). Check the recorded cameras (Record as on the tiles; none greyed). Type the task prompt (or
   click a recent one). Press **Enter** to leave the text field.
2. Press **Enter** (or **Start**) to start, and tell the operator to go.
3. **Enter** (or **Save**) saves, **Backspace** (or **Discard**) deletes the episode. It saves on its own after
   `--episode-s`.

Enter and Backspace only work on the Record tab while no text field has focus. While an episode is recording the
other tabs can't be opened until you save or discard. In the text field only Esc (STOP) and Enter work, so
typing a prompt can't press X (STOP).

The policy only drives the arm (swing, boom, arm, bucket, thumb). If the operator moves the **tracks**
(`|left_track|` or `|right_track|` > 0.1) during an episode, the Record panel shows a red warning while it happens
and counts those steps (`track_steps` in meta.json). Recording goes on; discard the episode if the tracks mattered.

The server records on the Pi's clock, so the laptop link doesn't affect timing. At every step it stores:

- the action: the 5 channels derived from `controls` with the map (-1..1, rig signs, 0 inside the deadband),
- the raw `controls` dict (all fields),
- the rest of the gate `/state` from the same poller the page uses (`sent`, `last_cmd`, `armed`, `veto`,
  `throttle`, `rabbit_button`, `age`, plus how old that poll is),
- each recorded camera's newest frame (camview's JPEG bytes, unchanged), with its arrival time, age and camview
  sequence number.

The cameras run at the same 15 fps as the recorder, so tick jitter sometimes makes a step reuse the previous step's
frame. That is expected: meta.json counts it per camera (`frames_repeated`) and the converter prints it; it is not an
error. A camera that really stops shows up as a growing frame age (and as **stalled** on its tile after 5 s), and the
converter rejects an episode with a frame older than `--max-age` (0.5 s, about 7 steps).

```
~/act-athon/episodes/20260926_101500_pick_up_the_red_block/
  meta.json      task, fps, cameras (name, camview id, resolution, model slot), camera_set, camera_slots,
                 camera_set_changed_from, channel names / signs,
                 action_source "cab", cab_map, cab_deadband, cab_map_status, track_steps, frames_repeated,
                 dashboard version, throttle / rabbit at start
  steps.jsonl    one line per step
  front/000000.jpg ...
```

An episode is written to `<name>.partial` and renamed when it is saved. If the dashboard stops mid-episode, the
folder stays `.partial` and the converter skips it.

After a save, the **Last episode QC** card under the Record panel shows the episode's quality report (timing, frames, camera skew, still tail, throttle, tracks, camera set, daylight; `GET /api/episodes/qc?name=` / `?recent=N`); checks, thresholds and the CLI (`python3 episode_qc.py --recent 5`) are in `docs/episode-qc.md`.

## Fetch and convert (laptop)

```bash
rsync -av excavator:act-athon/episodes/ data/raw/
uv run scripts/convert_pi_episodes.py data/raw --root data/rig_cab --repo-id local/rig_cab --resume
```

This makes a LeRobot dataset with:

- action = the cab joysticks, **re-derived** from each step's raw `controls` with the map recorded in the episode
  (or `--cab-map fixed.json`, same format as the dashboard's, to correct it after the fact),
- observation.state = the **previous step's action** (zeros at the first step). The gate's `sent` is ~0 during a cab
  demo because the gate isn't driving; for code-driven runs `sent` is the last command given, so "previous command"
  is the same signal the policy sees at deployment,
- the camera images: the square crop of each frame (default the centered 480x480, stored at 480x480), and
  the task.

Set a crop per camera with `--crop NAME=x,y,size` (pixels in the recorded frame, e.g. 640x480, or 1344x376 for the ZED; repeatable) or a JSON file
`--crop-file crops.json` (`{"front": [160, 0, 480]}`; `--crop` wins over the file). A crop must be square and fit inside
the frame; the converter checks this and prints the crop it uses per camera. `--out-size N` resizes every crop to NxN
(needed when the crops differ in size); by default the crop size is kept. One dataset has one set of crops: `--resume`
refuses other crops. The crops are written to `<root>/crops.json` and to `pi_sources.json`. **Deployment must use the
same crops**: `RigMachine` crops the same way (default: centered 480x480), so a non-default crop or `--out-size` has to
be given to it too (`make_machine(..., crops=load_crop_file("<root>/crops.json"), image_hw=(N, N))`). Otherwise the
policy sees different pixels than it was trained on.

```bash
uv run scripts/convert_pi_episodes.py data/raw --root data/rig_cab_c2 --repo-id local/rig_cab_c2 --crop front=160,0,480
```

`--resume` only adds episodes that aren't in the dataset yet, so run it again after each rsync. `pi_sources.json`
in the dataset records, per episode, the map and deadband, the crops and the repeated-frame counts. The converter rejects an episode, and says why, if a
camera frame, the gate state or the raw `controls` is missing, or a frame / the gate state is older than 0.5 s, or
steps are missing, or a camera's frame size differs from the other episodes'. Episodes from the older keyboard recorder (no `action_source: "cab"` in meta.json) are skipped,
and `--resume` refuses a dataset that already holds keyboard episodes: use a new `--root` for cab demos (for example
`data/rig_cab`, and keep any old keyboard data in `data/rig_keys`).

## Runner options (Inference tab)

Under the training dataset the runner panel sets the `control_loop.py` flags that used to need an ssh start:

| option | flag | notes |
|---|---|---|
| zero-state | `--zero-state` | automatic: set when the dataset has `meta/zero_state.json` (the runner refuses a mismatch) |
| Subtasks | `--subtasks` | pi07 only (the phase goes out as PI's raw_text) |
| stop after release | `--stop-after-release S` | needs Subtasks; empty = off |
| Grip latch + threshold | `--grip-latch --grip-latch-threshold T` | default threshold 0.3 |
| chunk / request at | `--chunk N` (pi07) `--request-at M` | default 25 / 8; request at < chunk |
| state = the command that drove | `--state-from-drove` / `--no-state-from-drove` | real-state datasets only (hidden for zero-state); on by default |

The server checks the same combinations the runner refuses. The choices are kept in `runner/last.json`, and
**command** (under the camera line) shows the exact command line of the running runner.

## DAgger from the Inference tab

Tick **Record DAgger episodes** before **Start runner**: the runner then records every tick into the Record tab's
episodes folder (policy steps, and your cab corrections as "human"). To take over press **Stop inference** or **STOP** (both open the relay;
the cab can't drive while it is closed), drive from the cab, then **Start inference** + SPACE to hand back. **Save episode · start
new** saves the current episode. Details: `docs/dagger.md`.

## Driving

Teleoperate is **off** when the page opens. While it is off the dashboard sends nothing to the gate: it only reads
`/state`. The header has three tabs, **Teleoperate · Record · Inference**: a tab only picks the page shown in the left
column. Each page has its own switch: **Start teleoperating** / **Start inference** (the page also shows the gate output,
the gamepad and, on Teleoperate, the key map), and Start / Save / Discard on Record. Switching tabs switches Teleoperate off (one
neutral) and Inference off. **Stop inference** (its button, or leaving its tab) also opens the relay, like STOP, so
the cab can drive at once; while an episode is recording the Record tab stays in front. STOP is highlighted while
anything is live. To drive, press Start teleoperating, then:

| | keys | gamepad (standard mapping) |
|---|---|---|
| deadman (required) | hold **Space** | hold **L1** or **R1** |
| swing left / right | A / D | left stick X |
| arm in / out | S / W | left stick Y (forward = out) |
| boom up / down | K / I | right stick Y (forward = down) |
| bucket curl / dump | J / L | right stick X (left = curl) |
| thumb open / clamp | U / O | LT / RT |
| STOP (neutral + relay open) | **Esc** or **X**, or the STOP button | |

- The key map is the same as calib.py's jog.
- The gamepad uses the same 0.12 deadzone and 0.5 expo as `src/act/teleop.py`. Gamepads with a non-standard
  mapping are ignored.
- Commands are full scale (-1 / 0 / +1 per key; the gamepad sticks are analog). Speed is set on the machine (throttle / rabbit), not here.
- While you hold an input, commands repeat every 100 ms.
- The machine goes to neutral when you release the deadman or all inputs, click outside the window, or switch tabs.
- If the page stops sending (tab closed, network drop), the server sends neutral within 0.3 s. The gate itself goes
  neutral after 0.5 s.
- Switching Teleoperate off sends one neutral (only if you drove) and then nothing.
- After a STOP, nothing is sent until someone switches Teleoperate on again.
- If a request was sent before a STOP but arrives after it, it is refused.
- If another page switches Teleoperate on, this page's session ends.

## Safety

Someone holds the physical kill switch whenever the dashboard runs with Teleoperate on, and nobody is within 3 m of
the arm or the swing radius. STOP / Esc and Ctrl-C open the relay; the kill switch is the backstop for everything
else. During a cab recording the operator in the cab is driving: the same 3 m rule applies. Whether the dashboard's STOP
(`/estop`: neutral + relay open) has any effect on driving from the cab hasn't been tested, so don't count on it: the
operator and the kill switch are what stop the machine.
