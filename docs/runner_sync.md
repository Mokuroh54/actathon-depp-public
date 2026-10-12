# Runner camera sync (camsync)

The policy runner (`scripts/control_loop.py`, deployed as `~/act-athon/runner/`) picks each observation's camera frames
with the same rule the Pi dashboard recorder used when it wrote the training episodes: `camsync.select`.

## The rule (`src/act/camsync.py`)

`src/act/camsync.py` is a verbatim copy of the Pi's `~/act-athon/camsync.py` (sha1 `4b9ee430...`), with a header
comment added above it. It must stay byte-identical below that header. At startup the runner re-checks the sha1 and,
on the Pi, compares it with `~/act-athon/camsync.py`. A mismatch prints a `WARNING` line.

- **aligned**: `t_ref` is the oldest of the cameras' newest capture times, i.e. the capture time of the camera that
  lags the most. Every camera gets its frame captured nearest `t_ref`.
- **latest**: every camera gets its newest frame. This is the behaviour before camsync.
- A camera whose newest capture is older than `MAX_LAG_S` (0.5 s) is left out of `t_ref` and gets its newest frame.

Capture times come from camview's `X-Capture-Time` header, on the Pi wall clock. The runner clamps them to the arrival
wall time, exactly as the recorder does.

## Which mode

Record and run with the **same** mode. Each episode's `meta.json` `"sync"` block records which mode it used:

| episodes | mode |
|---|---|
| session 2 (2026-09-26 16:02 and later) | `aligned` |
| session 1 (14:01-14:21), no `"sync"` block | `latest` |

`convert_pi_episodes.py` does not carry the mode into the LeRobot dataset. So with `--dataset` on `--machine real` or
`--machine dashboard`, `--sync` is required. If a future converter writes `"sync"` into `<dataset>/pi_sources.json`,
the runner takes the mode from there and refuses a `--sync` that contradicts it. The dashboard's Policy launcher
already passes its own `--sync`.

## Safety: the stale-frame guarantees still hold

Each `MjpegCamera` keeps a `camsync.FrameRing` of `(capture time, RingFrame(jpeg, arrival, seq))`. It holds only
genuinely new frames: repeats never enter the ring. The ring is bounded by `RING_KEEP_S` (1.5 s) and `RING_MAX` (90).
`RigMachine.get_observation()`:

1. Runs `check_fresh(0.5)` on every camera. This is the unchanged stale check with the same error messages. It raises
   if any camera's newest new frame arrived more than 0.5 s ago, and the run ends.
2. Picks the frames with `camsync.select(mode)`.
3. Raises if any picked frame **arrived** more than 0.5 s ago. In aligned mode the laggiest camera drags the others
   back to its capture time. When that camera freezes or lags, this check ends the run, sometimes before the camera's
   own stale check fires. The error names that camera as the cause. The converter's `--max-age` rejected such steps
   too, so the policy never trained on them.

Only the picked JPEGs are decoded: one per camera per step.

## Tick log

`logs/ticks_*.csv` gains these columns: `sync_mode`, `sync_t_ref`, `sync_anchor` (the camera that set `t_ref`),
`sync_skew_ms` (newest minus oldest picked capture), `sync_left_out`, `cap_<cam>` (the picked frame's capture time)
and `age_ms_<cam>` (its arrival age).

## Differences from the recorder (not in the selection rule)

- The recorder clears a camera's ring when it reconnects. The runner keeps it: frames older than 1.5 s drop out anyway,
  and the stale checks cover the gap.
- The recorder skips a camera that is not ok, and the step has no frame for it. The runner raises instead: every
  camera the policy needs must be fresh.
