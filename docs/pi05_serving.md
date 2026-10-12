# pi0.5 fallback: train and serve on Modal

pi07 through PI's gateway stays the primary path. This runbook is the fallback when PI's service
blocks us: fine-tune pi0.5 on Modal (`modal_jobs/train_pi05.py`), serve the checkpoint as an HTTPS endpoint
(`modal_jobs/serve_pi05.py`), and drive the machine with the same runner, switched to `--policy http`.

Every command below spends money or publishes data. Run them yourself, from the repo root on the laptop.

## 1. Put the dataset on a Modal volume

Any LeRobot dataset made by `scripts/convert_pi_episodes.py` works (daylight episodes only for the real run, see
AGENTS.md). Upload the folder that holds `meta/`, `data/` and `videos/`:

```bash
uv run modal volume create pi05-datasets                       # once (train_pi05.py would also create it)
uv run modal volume put pi05-datasets data/rig_cab_nighttest /rig_cab_nighttest
uv run modal volume ls pi05-datasets /rig_cab_nighttest        # expect meta/ data/ videos/ ...
```

To replace a dataset, upload it under a new name (e.g. `/rig_cab_v2`) instead of overwriting it, so a run's dataset
never changes under it. A dataset on the Hugging Face Hub still works with `--dataset <repo id>`.

## 2. Train

```bash
uv run modal run --detach modal_jobs/train_pi05.py --volume-dataset rig_cab_nighttest --steps 3000 --batch 32 \
    --rename '{"front": "base_0_rgb", "wrist": "left_wrist_0_rgb", "side": "right_wrist_0_rgb"}'
uv run modal app logs pi05-train                               # or the URL `modal run` printed
uv run modal volume ls pi05-runs                               # the run: pi05_<dataset>_<MMDD_HHMM>
```

`--rename` has to map every camera in the dataset to one pi0.5 slot, or the job stops before training starts (a
camera without a slot would be ignored by the model without any error). The run folder gets `checkpoints/`, plus
`rename.json` (the map) and `dataset.json` (image size, state dim, fps, cameras). The server reads both files.
The training prompt is the episode's task string. The runner's prompt has to match it exactly.

## 3. Create the bearer-token secret (once)

Modal web endpoints are public, so the server answers 401 unless the request carries
`Authorization: Bearer <PI05_SERVE_TOKEN>`:

```bash
TOKEN=$(openssl rand -hex 24)
uv run modal secret create pi05-serve PI05_SERVE_TOKEN=$TOKEN
echo "PI05_SERVE_TOKEN=$TOKEN" >> .env                          # laptop runs (git-ignored)
# and the same line in runner/.env on the Pi, next to PI_API_KEY
```

The client reads `PI05_SERVE_TOKEN` from the environment, or else from `.env` in its working directory. For the
dashboard runner, that is `runner/.env`.

## 4. Deploy the server

```bash
uv run modal deploy modal_jobs/serve_pi05.py                                        # latest run, L40S
PI05_SERVE_RUN=pi05_rig_cab_nighttest_0926_1400 uv run modal deploy modal_jobs/serve_pi05.py   # a specific run
PI05_SERVE_GPU=H100 uv run modal deploy modal_jobs/serve_pi05.py                    # another GPU
```

`modal deploy` prints the URL, for example `https://<workspace>--pi05-serve-policyserver-web.modal.run`. It is also
listed in the Modal dashboard under the app `pi05-serve`. The first request starts a container: loading the model
takes minutes (the runner waits on `GET /health`). After that, the container stays warm for 10 minutes after the
last request. At most one container runs at a time. The deploy fixes the run and the GPU in place. With no
`PI05_SERVE_RUN`, the container picks the newest run each time it starts, so a later training run changes what gets
served. Pin the run for anything that matters. `/health` and every reply name the served run.

Check it from the laptop (no machine involved):

```bash
curl -s -H "Authorization: Bearer $TOKEN" https://<url>/health        # run, cameras, image_hw, chunk_size
uv run scripts/pi_replay.py data/rig_cab_nighttest --url https://<url> --episode 0     # recorded frames, nothing moves
```

`pi_replay.py` sends a recorded episode's frames and state, so this checks connectivity, auth and the served
checkpoint against the recorded actions. On the machine, the runner's summary line shows rtt median / p95: keep
`--request-at` above rtt × fps.

Stop paying for a warm GPU right away: `uv run modal app stop pi05-serve`.

## 5. Drive the machine from the dashboard

The Pi's `runner/` folder must have the current `control_loop.py` and `act/` package (the `policies.py` with the
`HTTPPolicy` token / image-size changes), `runner/.env` with `PI05_SERVE_TOKEN`, and the training dataset in
`runner/datasets/<name>/` (meta/info.json + crops.json).

In the dashboard, switch Policy on and open the **Policy runner** panel:

| field | value |
|---|---|
| backend | **pi0.5 (Modal URL)** |
| URL | the URL from `modal deploy` (https://…modal.run, without `/infer`) |
| prompt | the training task string, exactly |
| training dataset | the same dataset the run was trained on |

Every camera in the dataset needs "Record as" set on its tile, as for pi07. Start runner, and wait for
`policy server: run …` in the runner log (a cold start takes minutes). Then hold SPACE / L1 / R1 to let the policy
act. Stop runner sends STOP. The runner sends the frames at the dataset's image size with JPEG quality 95. The server
resizes only frames that are not already that size, so no frame is resized twice.

## Cost (estimates: check modal.com/pricing before relying on them)

Assumed prices: L40S ≈ $1.95/h, H100 ≈ $3.95/h (GPU per-second billing; container CPU / RAM add a few percent).

| item | N | estimate |
|---|---|---|
| serving, warm L40S | per hour the container is up | ≈ $2/h |
| one test session | ~3 min cold start + your driving + 10 min warm tail | ≈ $0.50 + $2/h of driving |
| forgotten runner | the runner keeps requesting (it predicts while the deadman is released), so the container never idles out | $2/h until Stop runner / `modal app stop` |
| training, 2000 steps × batch 16 on H100 | ~1 s/step (assumed; read the real s/step from the first log lines) + ~10 min setup | ≈ 45 min ≈ $3 |
| training, 3000 steps × batch 32 on H100 | ~2 s/step (assumed) + setup | ≈ 1.8 h ≈ $7 |

Multiply by the number of runs: a night of 3 tries at 3000 × 32 ≈ $20. The first step times in `modal app logs
pi05-train` give the real s/step: redo the estimate with those numbers before launching a long run.

## Contract (for reference)

`POST <url>/infer` with `{"prompt": str, "state": [swing, boom, arm, bucket, thumb], "images": {"front": b64 JPEG, ...}}`
returns `{"actions": [[5 floats] × 50], "server_ms": float, "run": str, "ignored_cameras": [...]}`: the model's
whole chunk from one forward pass (`PI05Policy.predict_action_chunk`), unnormalized by the checkpoint's
postprocessor and clipped to -1..1. The server returns 400 on a malformed request, 401 on a missing or wrong token,
and 500 if the model outputs non-finite actions (these are never sent).
