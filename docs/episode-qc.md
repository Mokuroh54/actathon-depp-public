# Episode QC

`pi/act-athon/episode_qc.py` checks recorded Pi episodes (the dashboard's Record episode panel, layout in
`docs/dashboard.md`) before they go into a dataset. Standard library only, so it runs on the Pi with the system
`python3`. It reads `meta.json` and `steps.jsonl` once and only lists the JPEG folders (it never opens a JPEG): well
under a second for a 1000-step episode.

## Run

On the Pi (copy the file next to `dashboard.py` in `~/act-athon`):

```bash
cd ~/act-athon
python3 episode_qc.py --recent 5                       # the 5 newest saved episodes in ~/act-athon/episodes
python3 episode_qc.py episodes/20260926_101500_pick_up_the_red_block
python3 episode_qc.py --recent 1 --json                # machine-readable report
```

On the laptop after rsync: `python3 pi/act-athon/episode_qc.py data/raw/<episode>` or
`python3 pi/act-athon/episode_qc.py --recent 20 --episodes-dir data/raw`.

`--recent` skips `.partial` folders (unsaved or interrupted episodes). The camera-set check compares each episode with
the previous saved one in `--episodes-dir` (by folder name, i.e. start time); for episodes given by path, that folder
defaults to the episode's parent. Exit code: 0 all ok, 1 some warn, 2 some bad.

From Python: `qc_episode(path, prev_path=None) -> dict` (JSON-serialisable), `render_text(report) -> str` and
`item_lines(report)` (one `{item, verdict, text, notes}` per item). The dashboard imports it (optional: without the file
the QC card is hidden and nothing else changes) and shows the report of each saved episode in the Record panel's
**Last episode QC** card, plus a **Recent episodes** list (last 10). API: `GET /api/episodes/qc?name=<episode folder>`
(only saved folders listed in the episodes folder; cached until the episode or its predecessor changes) and
`GET /api/episodes/qc?recent=N` (N <= 20, newest first).

## Checks

Every item gets **ok**, **warn** or **bad** (or **n/a** when the episode has no data for it, e.g. old-format episodes
without those fields; n/a counts as ok). The episode's verdict is the worst item. Missing fields show as `n/a`; the
tool never crashes on partial or malformed episodes (an episode without steps is bad). "Period" is 1 / meta `fps`
(66.7 ms at 15 Hz). Thresholds are constants at the top of the file.

| item | reports | warn | bad |
|---|---|---|---|
| timing | steps, duration, mean rate vs meta fps, step interval mean/min/max, `late_ticks`, `dropped_steps` | a `late_tick`, or an interval > 1.5 periods | rate off by > 5 %, an interval > 2 periods, steps dropped, step count / index doesn't match meta |
| cam `<name>` | camview id + resolution, frame refs vs steps, unique refs, files on disk, `frames_repeated` (meta) and repeats seen in the per-step camview `seq`, frame age max / p95 | max age > 1.5 periods (100 ms at 15 Hz), or > 10 % repeated frames | a step without a frame, a referenced JPEG missing on disk, or max age > 0.5 s (the converter's `--max-age`) |
| skew | per step, the spread of the recorded cameras' frame arrival times (`cams.<name>.mono`): median / p95 / max | p95 > 1 period | p95 > 2 periods |
| cab | per channel min / max and % of steps nonzero (actions are already deadbanded), first / last active step, idle lead-in and still tail in seconds | no channel moved, lead-in > 5 s, or still tail < 0.5 s (AGENTS.md: end every demo still) | |
| machine | throttle min / max / mean (and at start), rabbit values, `track_steps`, steps where the gate was armed or `sent` was nonzero | throttle range > 0.1, gate armed / sending (something else drove the gate during a cab episode), gate state missing | `track_steps` > 0 (no track driving) |
| cam set | recorded names → camview id + resolution, vs the previous saved episode, plus meta `camera_set_changed_from` | the set changed (the episode starts a new dataset) | |
| daylight | `recorded_at` (the Pi's local time from meta `started`) | hour < 8 or > 18: night / dusk episode, pipeline test only, not for training | |

Notes:

- Frame age is measured at the step: step time minus the frame's arrival time at the dashboard. With cameras at the
  same 15 fps as the recorder, a max just above one period (70-75 ms) and a few % repeated frames are normal tick
  jitter; a camera that stops shows up as growing age and many repeats.
- Skew uses arrival times at the dashboard, not sensor exposure times; it is a coarse check of how far apart in
  time the cameras' frames at one step are.
- Compare the throttle and rabbit values across episodes: AGENTS.md asks for fixed throttle and rabbit mode for the
  whole collection and the demo.
