# Training without the state input (`--zero-state`)

Our `observation.state` is the previous command: the excavator has no joint sensors, so
`convert_pi_episodes.py` stores the previous step's action as state. A model trained on 7 demos
(`data/pilot-best7`) learned to copy it. On a training episode, its 15-step MAE was 0.085, against 0.086 for
"repeat the last command". On the machine it kept swinging right. A zero-state dataset removes that signal, so the
model has to act from the cameras.

## Make the dataset

```bash
uv run scripts/zero_state.py data/pilot-best7 data/pilot-best7-nostate      # --overwrite to redo it
```

This copies the dataset with `observation.state = 0.0` on every frame. The dtype and shape stay the same. Actions,
videos (hard-linked), tasks, `info.json`, `crops.json` and `pi_sources.json` are unchanged. The state stats in
`meta/stats.json` and the per-episode stats in `meta/episodes` become zeros (count kept). The script writes the
marker `meta/zero_state.json` and then checks the copy against the source with `LeRobotDataset`.

How lerobot 0.6.1 normalizes zero stats (checked by running `NormalizerProcessorStep`):

| mode | used by | a state of 0 becomes | any other state |
|---|---|---|---|
| MEAN_STD | SmolVLA | (0 − 0) / (0 + 1e-8) = 0 | ×1e8 |
| QUANTILES / MIN_MAX | pi0.5 | the zero-width range is replaced by eps, so −1 (a constant state token) | ×2e8 |

With a state of exactly 0 the result is finite in float32 and bfloat16. Neither policy converts state to float16
before the normalizer; in float16, eps rounds to 0 and the result is NaN. Any non-zero state blows up, so
inference has to send exact zeros.

### Noise variant (`--noise SIGMA`)

```bash
uv run scripts/zero_state.py data/s2-keep18 data/s2-keep18-noise --noise 0.01      # --seed 0 by default
```

PI's fine-tune on the all-zero `data/s2-keep18-nostate` failed with "Data preparation failed". Our unconfirmed
guess is the zero variance. With `--noise`, the state is i.i.d. N(0, SIGMA²) float32 noise, drawn from `--seed`
per global frame index, so it carries no information about the actions. The per-episode stats come from lerobot
0.6.1's `compute_episode_stats`, and `stats.json` is their `aggregate_stats`. This reproduces lerobot's own numbers
on pilot-best7 and s2-keep18 (the max difference is 2e-16).

The marker adds `train_noise_sigma` and `train_noise_seed`. Inference is unchanged: `--zero-state` sends zeros,
which is the noise mean. For `data/s2-keep18-noise` (SIGMA 0.01), lerobot 0.6.1 normalizes a zero state to:

| mode | per channel |
|---|---|
| MEAN_STD | 0.008, -0.014, 0.019, 0.003, -0.013 |
| QUANTILES | 0.008, -0.009, 0.031, 0.015, 0.010 |

All values are finite and close to 0. The Modal train jobs accept a noise marker only when the state stats have
|mean| < 0.2 SIGMA and std within 0.8 to 1.2 SIGMA, and serve the run as `zero_state`.

## Training and serving (Modal)

The pi0.5 and SmolVLA training jobs (`train_pi05.py`, `train_smolvla.py`) read the dataset's
`meta/zero_state.json`, from the volume folder or from the Hub. They write `"zero_state": true|false` into the run's
`dataset.json`. The marker has to agree with the state stats: a marker with non-zero stats, or all-zero stats with no
marker, stops the job before it trains.

When a run's `dataset.json` says `zero_state: true`, the server (the shared `serve_pi05.Handler`, used by both
`serve_pi05.py` and `serve_smolvla.py`) replaces every incoming state with zeros before preprocessing, whatever the
client sent. It reports `"zero_state"` in `/health` and in every `/infer` reply. A normal run's state is never zeroed.
Runs trained before this change have no field; they are served as normal runs, with a log line. `HTTPPolicy.connect`
refuses a server whose `/health` `zero_state` disagrees with the client's `--zero-state`, in either direction.
Servers deployed before this change don't report the flag, so the client can't check them.

**pi07 (`--policy pi`, PI's hosted gateway) has no server-side protection.** We don't control PI's server, and it
reports nothing about the training data. There, only the `--dataset` marker check and the `--zero-state` flag keep
the state at zero. Passing the wrong `--dataset` (one without the marker) to a zero-state pi07 checkpoint is not
caught.

## Inference

When `--dataset` has the marker, both scripts require `--zero-state`. With a dataset that has no marker, they
refuse `--zero-state`. Only the policy's view of the observation changes. The gate, the obs dict and the
`state_*` tick-log columns keep the real readback. The new `policy_state_*` columns show what the policy was sent.

```bash
uv run scripts/control_loop.py --machine real --cam front=<src> --cam wrist=<src> --policy http --url https://<url> \
    --dataset data/pilot-best7-nostate --zero-state --prompt "<training task>" --deadman --show
uv run scripts/pi_replay.py data/pilot-best7-nostate --url https://<url> --zero-state --episode 0
```

`pi_replay.py`'s "repeat" baseline always uses the recorded previous command, even with `--zero-state`. In the
zero-state copy that command survives only in the previous step's action, so the baseline numbers match a replay
on `data/pilot-best7`.
