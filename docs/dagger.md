# DAgger data collection

Run the pi07 policy on the real excavator, take over from the cab when it goes wrong, and record both the policy's
steps and your corrections as training episodes. The runner (`scripts/control_loop.py --record-dagger`) writes
episodes in the same raw format as the dashboard's Record panel. The usual pipeline then handles them:
`convert_pi_episodes.py`, then `subtask_labels.py`, then `pi_upload_spans.py`. That last step turns each step's
controller into a PI `control_source` span. Training uses `dagger_data_filter=True` together with the regular cab
demos, which is the chosen setup (see the last section).

## Taking over requires STOP

This was verified on the machine on 2026-09-26. While the policy drives, the code path holds the relay closed. With the
relay closed, **the cab joysticks do not drive the machine**, even though `/state` `controls` still shows your input.
Releasing SPACE therefore does not hand control to the cab: the machine simply stops.

To take over, press **Stop inference** (or **STOP**) in the dashboard. Both open the relay and switch inference off,
and from then on the cab drives the machine.

The recorder follows the same rule. A step counts as `human` only when three things are true at once:

- the dashboard did not pass the policy's action;
- the gate reports the relay open (`/state` `"armed": false`);
- a cab joystick is active.

If you move the cab while the relay is closed (after a SPACE release without STOP), the step is recorded as `idle`. It
gets `cab_ineffective: true` and is counted in meta `cab_steps_relay_closed`. A `/state` without an `"armed"` field is
treated as "relay closed", so the recorder never guesses `human`.

## Run a session (on the Pi)

**Deploy first.** `~/act-athon/runner` needs this repo's `control_loop.py` and `act/`, including `act/recorder.py`
and the updated `act/rig_machine.py`. `PI_API_KEY` is in `runner/.env`.

1. Start the dashboard.
2. Set "Record as" on the camera tiles to front, wrist and world.
3. **From the dashboard:** on the **Inference** tab tick **Record DAgger episodes**, fill in the runner (backend, gateway
   URL, prompt, training dataset) and press **Start runner**, then **Start inference** and hold SPACE. The panel passes
   `--record-dagger --episodes-dir <the dashboard's --episodes-dir> --dagger-split-file dagger_next` (plus
   `--zero-state` for a zero-state dataset; `--state-from-drove` is then on by default for a real-state one), with its
   usual `--chunk 25 --request-at 8` and `--sync`. **Save episode · start new** touches `runner/dagger_next`. A split
   file left over from an earlier run is removed when the runner starts.
   The panel also sets `--subtasks`, `--grip-latch` (+ threshold), chunk / request at and
   `--state-from-drove` (docs/dashboard.md, Runner options).
4. **Or over ssh**, for flags the panel doesn't set (e.g. `--start-phase`, `--grip-latch-s`):

Zero-state checkpoint:
```bash
cd ~/act-athon/runner && PYTHONPATH=. ../.venv/bin/python -u control_loop.py --machine dashboard \
    --dashboard-url http://127.0.0.1:8080 --policy pi --gateway-url '<pi07 checkpoint gateway_url>' \
    --dataset datasets/s2-keep18-nostate --zero-state --sync aligned --grip-latch \
    --prompt "move the orange cone from the blue square to the red square" --chunk 15 --request-at 10 \
    --cam front=http://127.0.0.1:8100/stream/usb-3-1-1-1-0 \
    --cam wrist=http://127.0.0.1:8100/stream/usb-3-1-4-1-0 \
    --cam world=http://127.0.0.1:8100/stream/net-world \
    --record-dagger --episodes-dir ~/act-athon/episodes
```

Real-state checkpoint:

- Use `--dataset datasets/s2-keep18` and drop `--zero-state`.
- `--state-from-drove` is then on by default, because `--record-dagger` is set. The policy's state is the command that
  drove the machine on the previous tick: the passed policy action, your cab action after a STOP, or zeros when idle.
  It is not the gate's `sent`. That is how the training datasets define state (the previous recorded action), so after
  a handback the policy sees your last command.
- `--no-state-from-drove` turns this off.
- With `--zero-state` it is off and refused.

Other options:

- **Reverse direction:** `--prompt "move the orange cone from the red square to the blue square"`, in a separate run.
- **Subtasks:** add `--subtasks` if the checkpoint was trained with subtask spans. The phase tracker then follows what
  drove the machine, including your cab corrections.
- **Cameras and sync:** use the same `--cam` names and the same `--sync` as the checkpoint's training data.

### Recording and ending episodes

The runner's status line shows `policy`, `human` or `idle` for each tick, plus `rec N (pX hY iZ)`: the steps recorded
so far, split into policy, human and idle.

- An episode starts at the first tick where someone drives: the policy passed with Policy on and SPACE held, or the cab
  after a STOP.
- It is saved when you quit (Ctrl-C).
- It is also saved when you create the split file: run `touch ~/act-athon/runner/dagger_next`. This gives one episode
  per attempt without restarting the runner; the next episode opens at the next engagement.
- If the run ends on an error, for example a stale camera or a gate or dashboard error, the runner still sends STOP.
  The episode is saved with `"aborted": true` in meta.json. The upload skips aborted episodes (see below).

## The takeover procedure

1. **Policy drives.** Policy is on and you hold SPACE (or L1 / R1).
2. **Take over.** Press **STOP** in the dashboard. The relay opens and Policy switches off.
   - The runner keeps running and keeps posting its heartbeat. While the dashboard reports Policy off, the heartbeat is
     neutral: the runner offers no action at all. The dashboard would refuse one anyway, so nothing is ever written
     while Policy is off.
3. **Drive from the cab.** Every tick with cab input is recorded as `human`, with the cab action.
4. **Hand back.** Let go of the cab joysticks, switch **Policy** on in the page, then hold SPACE.
   - The first held tick sends a neutral, as always. After that the policy drives again, and those ticks are recorded
     as `policy`.

A tick where the dashboard passes the policy's action is always recorded as `policy`. If the cab is touched during such
a tick, `cab_active` shows it; the cab can't drive then anyway.

To pause between attempts, stop driving: those ticks are `idle`. To stop for real, press STOP and then Ctrl-C the
runner.

The existing safety behaviour does not change. The deadman, STOP, the dashboard watchdog and the runner's stale-frame
exit work as before. Recording runs on its own writer thread and never delays a control tick.

## What gets recorded

The files match `dashboard.py`'s recorder:

```
~/act-athon/episodes/<YYYYmmdd_HHMMSS>_<slug>/     (.partial until saved)
  steps.jsonl  <cam>/<i>.jpg  meta.json
```

### Fields added to each step

On top of the recorder's step fields, each step has:

| field | meaning |
|---|---|
| `control` | `policy`, `human` or `idle`, by the rule above. |
| `action` | The command that drove the machine: the passed policy action, the cab action, or zeros. |
| `policy_action` | What the policy wanted, even while you drove. It is null on ticks where the runner's queue had no action. |
| `posted` | What the runner posted, after `--grip-latch`. It is neutral while Policy is off. |
| `cab_active` | Whether the cab joysticks were active. |
| `relay_closed` | Gate `armed`. |
| `cab_ineffective` | Cab input while the relay was closed; the step is recorded as `idle`. |
| `policy_on` | Whether Policy was on. |
| `phase` | Present with `--subtasks`. |
| `latch_active` | Present with `--grip-latch`. |

The recorder's usual step fields are:

- `state`: the gate `/state` that this tick's observation read.
- `cams`: per camera, `file`, `seq`, `cap`, `lag`, `age` and `bytes`.
- `sync`: how the tick's frames were picked.

### Frames

The frames are the camsync-picked frames that the policy saw on that tick. They are camview's JPEG bytes, unchanged.

### meta.json

`meta.json` contains the recorder's fields plus:

- `"dagger": true`;
- `policy`: backend, gateway URL, dataset, zero_state, state_from_drove, sync, chunk, grip_latch and subtasks.
  It contains no keys.
- `control_counts`;
- `cab_steps_relay_closed`;
- `aborted`;
- `dropped_steps`.

### How the converter gets the right actions

`convert_pi_episodes.py` re-derives every action from `state.controls` with meta `cab_map`, and a policy step's cab
controls are at rest. So the recorder:

1. keeps the raw `controls` unchanged;
2. adds `controls.dagger_<ch>` set to the recorded action;
3. writes `cab_map` pointing at those fields, with deadband 0.

The real cab map is kept as `human_cab_map` and `human_cab_deadband`. The format stays `act-athon-pi-episode/1` with
`action_source: "cab"`.

**Never convert DAgger episodes with `--cab-map`.** It would replace the `dagger_<ch>` fields, and every policy step
would become zeros.

### Dropped steps

If the SD card can't keep up, a step waits in the writer queue. When more than `--dagger-queue` steps (default 300,
about 20 s) are waiting, new recording steps are dropped and counted in meta `dropped_steps`. The control tick itself
is never dropped.

An episode with dropped steps, or with loop overruns longer than 1.5 periods (`late_ticks`), has gaps. The converter
rejects episodes with gaps, as it does for the dashboard's own episodes.

## Convert, label, upload

On the laptop, follow the usual flow (PI's Partner API guide, docs/subtask_labels.md):

```bash
rsync -av excavator:act-athon/episodes/ data/raw/dagger-1/            # copy; never deletes on the Pi
uv run scripts/convert_pi_episodes.py data/raw/dagger-1 --root data/dagger-1 --repo-id local/dagger-1 --out-size <training size>
uv run python scripts/subtask_labels.py --dataset data/dagger-1 --raw data/raw --out outputs/subtask_labels/dagger-1.json
PY=~/.local/share/uv/tools/pi-sdk/bin/python
$PY scripts/pi_upload_spans.py data/dagger-1 --raw data/raw --labels outputs/subtask_labels/dagger-1.json \
    --version dagger1 --contract-json outputs/pi_s2/contract.json --dry-run        # offline check first
$PY scripts/pi_upload_spans.py data/dagger-1 --raw data/raw --labels outputs/subtask_labels/dagger-1.json \
    --version dagger1 --contract-id <the SAME platform contract as the cab demos>    # real upload: user's go-ahead
```

**Zero-state checkpoint.** Run `scripts/zero_state.py` on the converted dataset, as for the cab demos, and upload that
one. Everything must use the same contract as the cab demos: same cameras, same fps, same state and action.

**Aborted episodes.** The converter converts every episode, and does not read `aborted`. `pi_upload_spans.py` skips
episodes whose raw meta.json has `"aborted": true` and prints a note for each one. `--include-aborted` uploads them.
Every uploaded episode is marked `success=True` (pi_dry_run's metadata).

**Subtask labels.** `subtask_labels.py` works unchanged. Its rule is the causal tracker over each step's `action`,
which is now whatever drove the machine. A grip by the policy, a carry by you and a release by the policy still form
one reach, carry, release sequence. Each step's `phase` field is what the runner told the model at that tick; it is
useful for checking the labels.

### Control spans

For raw episodes with `"dagger": true`, `pi_upload_spans.py` adds one `control_source` span per contiguous run of steps
with the same controller, next to the subtask spans:

| step `control` | span value |
|---|---|
| `policy` | `"policy"` |
| `human` | `"human"` |
| `idle` | `"none"` |

The one exception is an `idle` run with human steps on both sides, which becomes `"human"`. That is you holding still
in the middle of a takeover. Splitting the takeover there would cut the training action chunks, because they end at
each human span's end.

The spans are contiguous and cover the whole recording. The dry run validates them with pi-sdk's validator and checks
that every step maps back to its own controller. Cab demos keep one `"human"` span over the whole episode.

## Training: the chosen setup

The setup is `dagger_data_filter=True`, over the regular cab demos plus the DAgger episodes.

The filter keeps only steps inside `human` control spans, and action chunks stop at the end of each human span. The cab
demos are one human span each, so they are kept whole. From the DAgger episodes, only your corrections are trained on.
The policy's own steps serve only as the context that led to each correction; they are never used as training targets.

### Recipe

This runs in the pi-sdk environment and needs the user's go-ahead: filters are immutable and fine-tunes cost quota.
State `num_steps × batch_size` and the episode and step counts first.

```python
from pi_sdk.finetuning import CheckpointPolicy, DatasetFilterInput, EpisodeFilter

contract = "<platform contract id of the cab demos>"      # DAgger episodes were uploaded with the same one
cab = finetuning_client.create_dataset_filter(
    "excavator-cab-<version>-human",
    EpisodeFilter(platform_contract_id=contract, source_dataset_path="excavator/<cab demo version>",
                  success=True, dagger_data_filter=True),          # cab demos: all human, kept whole
)
dagger = finetuning_client.create_dataset_filter(
    "excavator-dagger1-human",
    EpisodeFilter(platform_contract_id=contract, source_dataset_path="excavator/dagger1",
                  success=True, dagger_data_filter=True),          # DAgger: only the human corrections
)
finetune = finetuning_client.create_finetune(
    "excavator-dagger1-ft",
    [DatasetFilterInput(cab.dataset_filter_id), DatasetFilterInput(dagger.dataset_filter_id)],
    base_model="pi07",                   # or continue_from_checkpoint_id=<current checkpoint> (keeps contract + profile)
    num_steps=<N>, batch_size=<B>,
    checkpoint_policy=CheckpointPolicy("final_only"),
    training_profile_id="<the cab demos' training profile id>",
)
```

- **Mix.** Without weights, data is sampled in proportion to each filter's size. The corrections are few, so consider
  explicit weights, for example `DatasetFilterInput(dagger.dataset_filter_id, weight=...)` with a positive weight on
  every filter (see PI's Partner API guide).
- **Source path.** `source_dataset_path` defaults to `excavator/<--version>` in `pi_upload_spans.py`.
- **Alternatives, not chosen.** Training on all steps (no DAgger filter), or on span predicates on `control_source`
  (PI's Partner API guide, "How span annotations affect training").
