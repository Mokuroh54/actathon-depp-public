# Joint calibration (dead band, speed, range, lag)

The excavator reports no joint angles, so its per-joint numbers (dead band, joint speed, range) are unknown until
measured. This procedure measures them with `pi/act-athon/calib.py`, a web page served by the Pi, and a phone
inclinometer.

What gets measured, per joint:

| quantity | how | gives |
|---|---|---|
| dead band | ramp \|cmd\| 0 → 0.6 over 12 s, press **M** when it moves | dead band |
| full-range traverse time | from one end stop to the other at 0.3 / 0.5 / 1.0, **M** at motion, **S** at the stop | max joint speed, deg/s (with the range) |
| response lag | start of command → **M** (includes your ~0.2 s reaction) | recorded only |
| range between end stops | phone pitch of each link at both end stops | joint limits, and the deg/s above |

Swing has no end stops: sweep between two tape marks on the ground and type in the angle between them.

## Before you start

- **People:** one person holds the kill switch for the whole session. Nobody within 3 m of the arm
  or the swing radius. One operator at the laptop, one person (outside 3 m) to read the phone.
- **Fix the throttle and rabbit (bunny) mode for the whole session and write them down.** Speed depends
  on both. The page shows them live and every sweep / ramp records them; the results table warns if
  they change mid-session. If you must change them, treat it as a new session and redo the sweeps.
- **End stops:** hitting an end stop drives the pump into relief, which heats the oil. Don't hold
  against a stop for more than about 1 s: press **S** (or let go of SPACE) the moment it stops moving.
  When jogging to a stop, let go as soon as it arrives.
- Park on level ground if you can. Otherwise measure the chassis pitch too (below).

## Start the page

`calib.py` must already be on the Pi (the coordinator copies it). On the laptop:

```bash
ssh excavator 'cd act-athon && python3 calib.py'      # then open http://<pi-host>:8099
```

Nothing is sent to the machine until you jog or start a sweep. The first write closes the relay.
**ESC**, **X** or the red STOP button open the relay at any time. Ctrl-C on the server also sends STOP.
If the tab loses focus or is hidden, any motion stops.

Controls:

- **Jog** (same keys as the dashboard's Teleoperate tab): hold SPACE, then A/D swing, W/S arm, I/K boom, J/L bucket,
  U/O thumb. Keys 1 / 2 / 3 set speed 0.3 / 0.5 / 1.0. Use jog to drive a joint to its start position.
- **Sweep / ramp:** choose sweep or ramp, joint, direction (and magnitude and timeout for a sweep), click
  **Arm**, then **hold SPACE** to start. The Pi runs the command loop itself at 20 Hz; the laptop only
  sends a heartbeat every 100 ms. Press **M** the moment the joint visibly starts to move. Press **S**, or
  let go of SPACE, the moment it reaches the end stop (or the tape mark for swing).
  A run also stops if the heartbeat is missing for more than 0.3 s (the network dropped), at the timeout
  (15 s by default, 30 s max), or for a ramp, when you press M or the ramp ends at 0.6.
  If runs keep stopping with `heartbeat` because of the wifi, restart with `python3 calib.py --hb-timeout 0.5`.
  The drawback is that the arm then keeps moving for up to 0.5 s after the laptop drops out.

## Phone inclinometer

Any level or clinometer app that shows degrees works (iPhone: Measure → Level; Android: a clinometer app).
Check the app on a flat table first. It should read 0.

- Hold the phone on its long edge, flat against the side plate of the link, with the long edge
  parallel to the line between the link's two pins:
  - **boom:** foot pin (at the swing frame) → boom-tip pin
  - **arm:** boom–arm pin → bucket pin
  - **bucket:** bucket pin → tooth tips (a straight edge of the side plate that points at the teeth is fine)
  - **thumb:** thumb pin → thumb tip
  - **chassis** (only if not level): the house deck, along the machine's forward axis
- Mark the phone position on each link with tape and use the same spot every time. A fixed placement
  offset cancels out in the range and speed. It only shifts the absolute limits.
- Sign: type the reading as **+ if the link's far end is higher than its pivot**, − if lower
  (the app may show only the magnitude). Chassis: + if the front is higher.
- Tick **back** if the far end is behind its pivot (toward the cab), e.g. the arm curled under the boom
  or a curled bucket. The page then maps the reading onto the full circle.
- **Joint angle = difference of adjacent link pitches**: boom = boom − chassis, arm = arm − boom,
  bucket = bucket − arm, thumb = thumb − arm. The page does the arithmetic. Enter every link the joint
  needs (e.g. for the arm: boom and arm) at **both** end stops, in the − end and + end rows. The labels show
  which physical end that is (e.g. boom − = up).

## Order

Do one joint at a time: boom, arm, bucket, thumb, then swing. For each joint:

1. **Dead band.** Ramp in each direction, 2–3 times per direction, starting away from the end stops.
   Press M at the first visible movement. The ramp ends there. If nothing moves by 0.6, it ends by itself;
   note that in the results.
   The recorded command is slightly high because of your reaction time (~0.2 s × 0.05/s ≈ +0.01).
   The ramp rate is stored so this can be corrected (below).
2. **Sweeps** at 0.3, 0.5 and 1.0, in each direction, at least 2 per setting. Jog to one end stop. Arm a
   sweep toward the other stop, hold SPACE, press M when it moves and S when it stops. Then sweep back.
   If a sweep ends with anything other than `stop` (heartbeat, timeout, blur, estop), redo it.
   A 0.3 sweep may be below the dead band. If it doesn't move, stop it (S) and note it.
3. **Angles.** Jog to the − end stop, read the needed links, jog to the + end stop, read them again,
   add a note (phone spot, app), then **Save angles**. The results table then shows deg/s for every
   completed sweep of that joint.

**Swing:** lay two tape marks on the ground under the bucket at a known angle apart (e.g. 90°, measured
from the swing centre). Put the bucket over the first mark, enter the angle in **swept angle**, and sweep
to the second mark. Press S as the bucket passes it. Stick to 0.3 and 0.5. At 1.0 the swing overshoots badly.
Swing has no angles step and no limits.

## Reading the results

Each sweep, ramp and angles entry is one JSON line in `calib/calib_<date>.jsonl` next to `calib.py` on the Pi.
Key fields:
- sweep: `t_start_cmd`, `t_motion`, `t_stop` (Pi monotonic clock), `lag_s`, `traverse_s`, `cmd`, `end_reason`,
  `throttle`, `rabbit_button`, `angle_deg` (swing), `write_hz`
- ramp: `cmd_at_motion`, `ramp_rate_per_s`
- angles: `joint_deg` per end, `range_deg`, and the raw pitches

Key-press times are stamped in the browser and mapped onto the Pi's clock, so network jitter doesn't end up in the
timings. The receive times are kept as `*_rx`.

Copy the results to the laptop:

```bash
mkdir -p data/calib && scp 'excavator:act-athon/calib/*.jsonl' data/calib/
```

Per joint: dead band = the mean over ramps of |`cmd_at_motion`| − `ramp_rate_per_s` × 0.2 s (the reaction
correction). Max rate = range / `traverse_s` from the |cmd| = 1 sweeps that ended with `stop`; if there are none,
extrapolate from lower settings (rate × (1 − dead band) / (|cmd| − dead band)). Limits come from the latest angles
entry, for boom, arm and bucket only; absolute values include the phone-placement offset.
