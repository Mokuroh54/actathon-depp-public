# Replaying a cab episode through the gate

Open question: does a value sent through the **gate** (dashboard Policy mode) move the arm like the same **cab
joystick** position? Speed at full stick, the dead band, and partial-stick response could all differ; the policy is
trained on cab values and deployed through the gate, so a mismatch means it will over- or under-drive every joint.

`scripts/replay_episode.py` answers it with an open-loop replay: it plays a recorded cab episode's actions
(`steps.jsonl` `action`, rig signs, -1..1) through the dashboard at the recorded rate, and logs what was recorded,
what was sent and what the gate actually output. Nothing closes the loop: if the arm drifts, it keeps drifting.

## How it drives the machine

- Same path as `control_loop.py --machine dashboard`: every tick it POSTs one action to the dashboard's
  `/api/policy/act` (`DashboardMachine`). The dashboard writes it to the gate only while the deadman is held in the
  page (Policy on + SPACE / L1 / R1). The gate's `/state` is only read.
- One action per tick at the recorded step times (15 Hz). Neutral ticks are posted too (runner heartbeat).
- **The trajectory advances only while the deadman is held.** Release: the step that was just posted was not written
  (the dashboard wrote neutral), the replay pauses and posts neutral. Hold again: it resumes at that same step. So you
  can stop anywhere and continue. It starts released: the first recorded step goes out only after the dashboard has
  confirmed the deadman on a neutral tick.
- End of the trajectory, Ctrl-C, SIGTERM, any error, gate `/state` unreachable or stale (controls age > 1 s,
  `--max-state-age`): one neutral, then dashboard **STOP** (neutral + relay open, Policy off). If STOP itself fails
  it prints `!!! dashboard STOP FAILED`: press STOP in the page / use the kill switch.
- Before anything moves it prints the task, the window, per-channel stick use, and the start-pose frames, then asks
  for `go` (`--yes` skips the prompt; motion still needs the deadman).

## Setup on the Pi

The coordinator deploys `replay_episode.py` next to `control_loop.py` in `~/act-athon/runner/` (it imports the
runner's `act/` package). Episodes are in `~/act-athon/episodes/`. Dashboard running, camview running, Policy mode
on in the page, **spotter on the kill switch**, nobody within 3 m. Throttle and rabbit mode as during recording
(the episode's `meta.json` has `throttle_at_start` / `rabbit_at_start`): joint speed depends on them.

**Start pose.** The recording has no joint positions. Drive the machine from the cab (or teleop) into the pose of the
window's first step, by eye: the script prints the episode's `front/`, `world/` and `wrist/` frames for that step.
Match the boom/arm/bucket angles and swing heading as closely as you can; the replay is open loop, so a start-pose
error stays in the whole run.

## Commands

Full replay of a clean session-2 episode (`20260926_163714`, 629 steps, 41.9 s, blue square -> red square; start
pose = its frame 000000):

```bash
cd ~/act-athon/runner
PYTHONPATH=. ../.venv/bin/python replay_episode.py \
    --episode ~/act-athon/episodes/20260926_163714_move_the_orange_cone_from_the_blue_squar \
    --cam front=http://127.0.0.1:8100/stream/usb-3-1-1-1-0 --cam world=http://127.0.0.1:8100/stream/net-world
```

Grip only (thumb, other channels neutral). The episode's thumb clamp (`thumb` > 0.3) runs from t = 6.8 to 12.07 s
at +1.0 (full clamp in), after an open run at about -0.96 from 2.0 to 4.9 s. Clamp only, starting with the thumb
open as at t = 6.5 s:

```bash
PYTHONPATH=. ../.venv/bin/python replay_episode.py \
    --episode ~/act-athon/episodes/20260926_163714_move_the_orange_cone_from_the_blue_squar \
    --from-s 6.5 --to-s 12.5 --channels thumb
```

Full open-then-clamp cycle, starting with the thumb shut: `--from-s 1.8 --to-s 12.5 --channels thumb`.

Other options: `--scale 0.5` (multiply every command, clipped to -1..1; values above 1.5, zero or negative are
refused), `--channels boom,arm`, `--snap-every 0.5`, `--dashboard-url` / `--rig-url` (defaults: the Pi's
127.0.0.1:8080 and :8096). The camview IDs above are the ones recorded in this episode's `meta.json`; check the
current IDs in the dashboard if a camera was replugged. Cameras are optional: one that fails to open prints a warning
and the replay runs without it.

## What to watch and record

- **Does the arm retrace the demo?** Watch the world camera side by side with the episode's `world/` frames (and
  `front/`). Note where the bucket ends up relative to the cone, the blue and red squares, and the final pose
  compared with the last recorded frame.
- **Speed at full stick.** The episode has long full-stick runs: boom -1.0 (up) 32.7-40.6 s, arm -1.0 (in)
  32.4-40.7 s, thumb +1.0 6.8-12.1 s, swing +1.0 1.0-1.9 s. Time them against the recorded frames: does each joint
  cover the same range in the same time, and does it hit an end stop earlier or later?
- **Partial stick / dead band.** Partial runs: swing -0.43 13.7-16.6 s, boom +0.44 25.6-27.0 s, arm +0.5
  16.7-18.0 s. If the arm barely moves where the cab moved it, the gate's dead band is larger (or its gain lower)
  than the cab's; if it overshoots, the gain is higher.
- **Grip test.** Does the thumb close fully in the same time, and does it hold the object?
- **Pauses.** Each release/re-engage restarts a joint from rest, which the recording didn't have: a replay with many
  pauses drifts more. For the comparison, run a window straight through.

Write down per run: episode, window, channels, scale, throttle / rabbit, start-pose quality, where it ended vs the
recording, and the log file name.

## The log

`logs/replay_<time>.csv` (in the runner folder), one row per tick:

| column | meaning |
|---|---|
| `wall`, `t_run` | wall clock; seconds since start |
| `pos`, `n`, `ep_i`, `rec_t` | position in the window, window length, episode step index (= frame file number), recorded t |
| `state`, `allowed`, `policy_on`, `advanced` | LIVE / released before this tick; dashboard verdict; Policy on; 1 if this tick played a step |
| `rec_<ch>` | the recorded cab action |
| `sent_<ch>` | what was posted (channels masked, scaled, clipped; neutral while released) |
| `passed_<ch>` | what the dashboard passed on (sent if allowed, else 0) |
| `gate_<ch>` | gate readback right after: `/state` `sent` (what the gate is outputting, after its caps) |
| `gate_live`, `gate_veto` | relay armed; the gate's refusal reason |
| `snap` | snapshot files written this tick (`logs/replay_<time>/`) |

At the end it prints the mean |gate - sent| per channel over played steps. Non-zero means the gate changed the
command (caps, clamps, veto): the mismatch then starts before the hydraulics.

## Reading a mismatch

- `gate_<ch>` differs from `sent_<ch>`: the gate caps or vetoes values; the policy's output is not what reaches the
  valves. Check the gate's caps (and `gate_veto`).
- Gate output matches but the arm moves slower / faster than in the demo: the gate's electrical output and the cab
  joystick produce different valve openings for the same number. The cab map (`cab_map` in `meta.json`, the
  converter) or a per-channel gain between the policy and the gate must correct it before the policy can work.
- Small commands do nothing through the gate but moved the arm from the cab: a larger dead band on the gate path.
  Expect the policy's small corrections to be lost; a dead-band offset on the gate side would fix it.
- Direction wrong on a channel: a sign error in the cab map for that channel. Stop at once (release the deadman).
- Arm matches the demo: gate values and cab values are interchangeable, and the policy's actions can be sent as is.

## Offline model replay with subtasks (`scripts/pi_replay.py`)

Not the gate replay above: `scripts/pi_replay.py` sends recorded frames to a hosted checkpoint and scores its chunks
against the recorded actions (nothing moves). `--subtasks LABELS.json` sends each step's labelled phase as the
subtask (`obs["subtask"]` → PI `raw_text`, as `control_loop.py --subtasks`) and adds MAE per phase; `--episode`
takes several indices (pooled). Thumb grip recall (recorded thumb ≥ 0.3 and predicted ≥ 0.3) is always printed.

```bash
PYTHONPATH=src .venv/bin/python scripts/pi_replay.py data/s24-nostate --zero-state \
    --gateway-url wss://api.pi-fleet.com/v1/models/<checkpoint-id> \
    --episode 0 6 11 18 20 22 --every 3 --subtasks outputs/subtask_labels/s24.json    # 1190 requests, ~2.5 min
```
