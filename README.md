# actathon

Our code from the act-athon physical-AI hackathon (2026-09-25 to 27): teaching a real mini excavator a cone
pick-and-place task with a vision-language-action policy. Human demos were driven from the cab and recorded on the
excavator's Raspberry Pi. We fine-tuned PI's pi07 through PI's Partner API; pi0.5 and SmolVLA on Modal were the
fallbacks. The fine-tuned policy then drove the machine through the organisers' control gate, from a web dashboard on
the Pi.

The hackathon is over; this repo is kept as a record. The notes in `AGENTS.md` are the plan as it stood during the event.
Side tools that weren't part of the final pipeline (the sim demo and fake machine, synthetic data, Modal eval, the first
keyboard teleop page, the legacy dig / push tasks, real-time chunking, the stall assist) were removed before
this release.

## How it fits together

```
cab operator drives ──► Pi: dashboard.py records episodes          (cameras via camview.py, frames picked by camsync)
                         │  rsync
                         ▼
laptop: convert_pi_episodes.py ──► LeRobot dataset ──► PI fine-tune (pi_finetune.py / pi_upload_spans.py)
                                                        or pi0.5 / SmolVLA on Modal (modal_jobs/)
                         ▼
Pi: dashboard Inference tab starts control_loop.py ──► hosted checkpoint ──► gate (127.0.0.1:8096) ──► machine
```

What the policy sees is kept identical between recording and deployment:
- **Frames:** the recorder and the runner pick each step's frames with the same `camsync.select` rule (`aligned` by
  default). The runner refuses to start if its copy of `camsync.py` differs from the recorder's.
- **Freshness:** a frame more than 0.5 s old rejects the episode at conversion and stops the run on the machine.
- **Pixels:** the same square crop (`crops.json` in the dataset) and resize, through one function (`prepare_image`).
- **State:** `observation.state` is the previous step's command, or zeros for `--zero-state` checkpoints.

## Layout

```
pi/act-athon/            runs ON the Pi (~/act-athon), next to the gate and the cameras
  dashboard.py           web UI on :8080. Tabs: Teleoperate · Record (episodes, QC) · Inference (starts the runner)
  camview.py             every camera as MJPEG on :8100 (USB, OAK, network cameras); stamps capture times
  camsync.py             one clock and one frame-selection rule for all cameras (recorder, camview, runner)
  camctl.py, start_dashboard.sh   lock exposure / white balance, then start the dashboard
  episode_qc.py          quality report per recorded episode
  worldcam/worldcam.py   a laptop webcam (or OAK) as the third-person "world" camera
  calib.py               joint calibration page (dead band, joint speed, range measurements)
  latency.py             measures how old frames are on arrival (read-only)
  raw_to_lerobot.py      alternative converter from the raw episodes on the HF Hub
src/act/                 shared library (the `act` package; on the Pi it is ~/act-athon/runner/act)
  rig_machine.py         the real excavator: gate I/O, synced camera frames, image crop / resize
  machine.py             the action / observation schema (CHANNELS)
  cameras.py, camsync.py camera readers with freshness checks; verbatim copy of the Pi's camsync.py
  policies.py            policy backends behind infer(obs, prompt): PI (pi07), HTTP (pi0.5 / SmolVLA)
  control.py             ChunkedController (ask ahead, drop stale steps, watchdog)
  assist.py, phase.py    grip latch; the reach → carry → release subtask phase
  recorder.py            DAgger episode recording from the runner
  teleop.py, dataset.py, video.py
scripts/                 laptop-side tools (control_loop.py and replay_episode.py also run on the Pi's runner)
  control_loop.py        the policy runner: real (over the gate tunnel) or through the dashboard (--machine dashboard)
  excavator_up.sh        one command for a session: world camera + ssh tunnel + dashboard on the Pi
  convert_pi_episodes.py Pi episodes -> LeRobot dataset (cab map, crops, freshness, --end-at-release)
  zero_state.py          copy of a dataset with observation.state zeroed
  subtask_labels.py      reach / carry / release labels from the cab commands
  pi_finetune.py, pi_upload_spans.py, pi_status.py, pi_dry_run.py   PI Partner API: upload, fine-tune, status
  pi_replay.py, pi_plan_probe.py   score a hosted checkpoint offline against recorded episodes (nothing moves)
  replay_episode.py      replay a cab episode through the gate (checks cab vs gate commands)
  camcheck.py            one observation through the runner's camera path (read-only)
  teleop.py              laptop gamepad / keyboard teleop (real machine, over the gate tunnel)
modal_jobs/              pi0.5 and SmolVLA fallbacks: train, and serve as an HTTPS policy endpoint
docs/                    one page per component (below)
```

## Setup

```bash
uv sync                                                   # Python 3.12, installs `act` in editable mode
uv run hf auth login
uv run modal setup                                        # only for modal_jobs/
```

PI's SDK comes from PI's package registry, and the `PI_API_KEY` goes in a git-ignored `.env` (see "Not included" below).

## A session on the machine

```bash
scripts/excavator_up.sh --rec-fps 15          # laptop: world camera + tunnel, dashboard on the Pi in the foreground
open http://<pi>:8080                          # Teleoperate / Record / Inference
```

- **Record:** a cab operator drives; Enter starts / saves an episode, Backspace discards it, and STOP / Esc also discards it.
- **Convert:**
  ```bash
  rsync -av excavator:act-athon/episodes/ data/raw/
  uv run scripts/convert_pi_episodes.py data/raw --root data/rig_cab --repo-id local/rig_cab --resume
  ```
- **Inference:** pick the checkpoint URL, prompt and training dataset on the Inference tab and start the runner. The
  runner options set subtasks, grip latch, chunk / request-at and DAgger recording. Stop inference opens the relay
  so the cab can take over.

## Action / observation schema

- **action:** `{ch}.cmd` for `ch` in `swing, boom, arm, bucket, thumb`, joystick commands in -1..1 (speeds, not positions).
- **observation.state:** the previous step's command (zeros at step 0; all zeros for `--zero-state` checkpoints).
  The gate's `sent` output is 0 while the cab drives, so recorded state comes from the previous action.
- **cameras:** `front`, `wrist` and `world`. `wrist` is only a slot name: both it and `front` are roof cameras,
  and there is no bucket camera.

| channel | + means            | keyboard (+ / -) | pad                   |
|---------|--------------------|------------------|-----------------------|
| swing   | house right        | D / A            | left stick right      |
| boom    | down (lower)       | I / K            | right stick forward   |
| arm     | out (extend)       | W / S            | left stick forward    |
| bucket  | dump / open        | L / J            | right stick right     |
| thumb   | clamp in           | O / U            | RT (LT = open)        |

- There is no joint-position feedback: `/state` only reports commands.
- The first command write closes the relay. While code holds it closed, the cab joysticks do not move the machine,
  so a human takeover needs STOP (relay open) first.
- Each command is held 0.5 s and then goes neutral, so the runner sends every tick. Tracks and blade are locked by the gate.
- **Safety:** someone at the kill switch whenever the relay may be live, and nobody within 3 m of the arm.

## Docs

| page | about |
|---|---|
| [excavator.md](docs/excavator.md) | Pi access, ssh, services |
| [dashboard.md](docs/dashboard.md) | the dashboard: teleop, recording, inference, runner options |
| [camera-controls.md](docs/camera-controls.md), [remote-camera.md](docs/remote-camera.md) | locked exposure; network / world cameras |
| [runner_sync.md](docs/runner_sync.md) | camsync: how frames are picked in recording and inference |
| [episode-qc.md](docs/episode-qc.md) | episode quality checks |
| [subtask_labels.md](docs/subtask_labels.md), [zero_state.md](docs/zero_state.md) | subtask spans; training without state |
| [dagger.md](docs/dagger.md), [replay.md](docs/replay.md) | DAgger collection; replays |
| [pi05_serving.md](docs/pi05_serving.md), [smolvla.md](docs/smolvla.md) | Modal fallbacks |
| [calibration.md](docs/calibration.md) | joint calibration measurements |


## Not included

- `pi/act-athon/rig/`, `example.py`, `ui.py`: the organisers' gate client and examples (Actor Labs' act-athon repo). `calib.py` expects `rig/` next to it on the Pi.
- Physical Intelligence's Partner API guide and SDK, available to PI partners. The PI scripts need `pi-sdk` from PI's package registry and a `PI_API_KEY`.
