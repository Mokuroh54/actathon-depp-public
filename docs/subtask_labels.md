# Subtask labels: reach → carry → release

pi07 gets two instructions per step: the episode task (`robot_task_string`) and a subtask (`raw_text` at inference,
the `subtask` of a `language_segment` span in training, see "How span annotations affect training" in PI's Partner API guide). We give
it the phase of the cone pick-and-place:

| phase (exact string) | from | to |
|---|---|---|
| `reach and grip the cone` | episode start | grip complete |
| `carry the cone to the target` | grip complete | release |
| `release the cone` | release | episode end (latched, no return phase) |

Machine tests failed exactly at these transitions (never lifting after a grip; weak / short grips). The phase comes from
the joystick commands with ONE causal rule, `act.phase.PhaseTracker` (`src/act/phase.py`), used by both sides:

- training: `scripts/subtask_labels.py` runs it over each recorded episode's cab commands → labels →
  `scripts/pi_upload_spans.py` uploads them as spans;
- inference: `scripts/control_loop.py --subtasks` runs it over the commands the runner actually sends → `raw_text`.

## The rule (act.phase)

Commands are -1..1 rig signs: thumb `+` = clamp in, `-` = open; bucket `+` = dump; swing. Values inside the
recording's ±0.08 deadband count as 0. Durations are seconds, converted with the fps (15 Hz).

1. **reach → carry (grip complete):** the first step where the thumb stops clamping (thumb ≤ 0.3) after a clamp of
   ≥ **1.2 s** of accumulated thumb > 0.3. Clamp pieces with gaps ≤ **0.5 s** count as one clamp.
2. **carried:** while in carry, once the swing has accumulated **1.0 s** of |swing| > 0.3 in the task's carry
   direction (blue→red: swing `-`; red→blue: swing `+`; either direction for an unknown task).
3. **carry → release:** once carried, at the first step after
   - a thumb-open run (thumb < -0.3) of ≥ **0.2 s**, or
   - a bucket-dump run (bucket > **0.7**) of ≥ **0.4 s**, also while the thumb clamps (`combined_dump`, the user's
     decision: the simultaneous clamp+dump counts as the release).
4. **Regrip:** a thumb open of ≥ 0.2 s in carry BEFORE the carry swing goes back to reach. The next full clamp
   completes a new grip.
5. **Release latches:** a stow clamp at the end of a demo doesn't count.

**Timing.** An event is fired by the command of step i. The new phase applies from step i+1. The runner builds tick
i's request before tick i's command is known, so it sees the new phase from tick i+1 too, and the labels use the same
boundary.

### Why these thresholds

They were chosen on the 18 session-2 episodes, checked against the camera frames.

- **1.2 s clamp.** Operators give a short 0.2-0.7 s clamp and then open wide before the real grip. Real grips last
  1.3-8.7 s. A short or weak grip therefore never counts: the runner stays in reach until the model has finished a
  full grip.
- **0.2 s thumb open.** Some releases are 0.2 s taps (171528, 170237).
- **Bucket dumps only after the carry.** Before the carry they are pre-grip positioning. About a third of the
  releases are bucket dumps, often with thumb+ at the same time.
- **0.7 dump threshold.** A 0.4 dump did not release the cone (165143).

**Known failure.** Thumb+ with bucket+ is sometimes only a rotation of the cone while it is still held (163805,
171110, 172745, 190439, 191108 at the first combined motion). In those episodes the rule releases a few seconds
early, and the labels were corrected by hand (below). The runner has the same weakness.

### Confidence and flags

| confidence | meaning | uploaded by default |
|---|---|---|
| `high` | clean grip → carry → thumb-open / clean-dump release | yes |
| `review` | labels written, check the sheet: regrip, combined thumb+/bucket+ release, release > 20 s after the carry, grip > 20 s before the carry | only with `--include-review` |
| `manual` | an override from a spot check (`--overrides`: corrected times, or an entry without times = confirmed) | yes |
| `none` | no labels: no grip at all, or no release (never carried while holding, or no open after the carry) | no |

For a regrip, the labels are hindsight: reach until the grip that is actually carried. The runner, being causal, shows
carry during the failed grip until the thumb reopens. In the 32-episode set this happens only in 164142 (72 steps).

## The 32-episode training set (s23-keep32)

- Labels: `outputs/subtask_labels/s23_keep32.json`, with overrides from
  `outputs/subtask_labels/s23_keep32_overrides.json`: 13 manual entries, each with a reason, all checked against the
  frames. `outputs/subtask_labels/s23_keep32_rule_only.json` is the rule alone.
- Spot-check sheets: `outputs/subtask_check/s23_keep32/<episode>.jpg`.

```bash
KEEP=_163714_,_163805_,...   # the 32 HHMMSS keys
uv run python scripts/subtask_labels.py data/raw/2026-09-26_session2 data/raw/2026-09-26_session3 --keep "$KEEP" \
    --overrides outputs/subtask_labels/s23_keep32_overrides.json --sheets outputs/subtask_check/s23_keep32 \
    --out outputs/subtask_labels/s23_keep32.json
```

Each sheet has a world row and a wrist row, with columns at grip -1 s / 0 / +1 s and release -1 s / 0 / +1 s, and
the commands overlaid. A correct sheet shows the cone:

- at GRIP-1s: in the closing jaws;
- at GRIP and GRIP+1s: clamped (lifting starts);
- at RELEASE-1s: still held at the target square;
- at RELEASE+1s: free in the target square.

To fix an episode, add `{"<episode folder>": {"grip_complete_s": .., "release_s": .., "why": ".."}}` (seconds from
the episode start) to an overrides file and label again.

## Upload with spans

`pi-upload-lerobot` can't attach spans: `ingest.upload_episode` calls `create_episode` without them.
`scripts/pi_upload_spans.py` plans and converts with the same pi-sdk ingest functions as `pi_dry_run.py`, then per
episode calls:

1. `create_episode(spans=[control_source(human), language_segment(task, subtask=<phase>) × 3])`;
2. `upload_shard` for each shard;
3. `finalize_episode_upload`.

Span placement:

- A boundary at step k is at `start + (k - 0.5)/fps`.
- The recording window is `[first step wall time, + (n-1)/fps]`, as the converter derives it.
- `--patch-existing` instead adds the spans to episodes that are already uploaded, with the public
  `Client.replace_annotations`.

The dry run applies pi-sdk's `validate_spans`, plus window, contiguity and per-step round-trip checks.

```bash
PY=~/.local/share/uv/tools/pi-sdk/bin/python
$PY scripts/pi_upload_spans.py data/s23-keep32-nostate --raw data/raw --labels outputs/subtask_labels/s23_keep32.json \
    --version s23ns --source-dataset-path excavator/s23ns --contract-json outputs/pi_s2/contract.json --dry-run
$PY scripts/pi_upload_spans.py data/s23-keep32-nostate --raw data/raw --labels outputs/subtask_labels/s23_keep32.json \
    --version s23ns --source-dataset-path excavator/s23ns --contract-id 01a0df9c-f2d4-7b31-ac37-28bc4a6e9fb2
```

Keep `truncate_actions_at_span_end` off (the default): chunks that cross a boundary teach the model to go on, for
example to lift right after the grip.

## Inference: `control_loop.py --subtasks` (--policy pi only)

- One `PhaseTracker` per run. The tracker uses the labeler's thresholds,
  `carry_sign(prompt)` and combined dump. A dashboard deadman release does NOT reset it.
- Each tick it is fed the command actually **sent**: after `--grip-latch` and the deadman / neutral handling, with
  the ±0.08 deadband applied first. On `--machine real` nothing is fed before the first deadman press.
- The phase travels in the observation (`obs["subtask"]`, `act.policies.SUBTASK_KEY`). Every request is therefore
  built with the phase of its own tick. This holds for the chunked controller's snapshot and `ZeroState`.
  `PIPolicy.make_input` sends `raw_text = phase` and `robot_task_string = prompt`; without the key,
  `raw_text = prompt`, as before.
- `--policy http` refuses `--subtasks`: our servers take no subtask.
- Status line: `| phase reach|carry|release`. Tick log: `phase` (what this tick's request was built with) and
  `phase_event` (`grip` / `regrip_open` / `release` / `override:<phase>`).
- Manual overrides, printed and logged:
  - `--start-phase reach|carry|release` sets the first tracker of the run.
  - The nudge file `--phase-file`, by default `phase_next` in the runner's working directory:
    - `touch phase_next` advances one phase (release → reach = a new attempt);
    - `echo carry > phase_next` jumps to that phase.

    The runner deletes the file when it consumes it.
- `--stop-after-release 1.5` (recommended) ends the run 1.5 s after the phase enters release (event or override),
  through the normal end (STOP, logs, DAgger save); leaving release first cancels it. Tick log: `auto_stop`.
- Without `--subtasks`, behaviour is unchanged: same tick-log columns, same status line, `raw_text` = prompt.

With `--grip-latch --subtasks`, a latched grip is a full clamp: when the latch ends and the sent thumb falls back
below 0.3, the tracker fires `grip` and the next request says `carry the cone to the target`.

Known risks of the causal rule at inference:

- A ≥ 0.2 s thumb-open blip during the carry swing, or a ≥ 0.4 s thumb+/bucket+ rotation after the swing, releases
  early.
- A one-tick thumb dropout inside a long clamp completes the grip early (seen in 191108 at 14.4 s).
- A dropped cone stays in carry until an open.

Use the nudge file to correct the phase by hand. Changing thresholds only in the runner breaks the agreement with
training: re-label and retrain instead.
