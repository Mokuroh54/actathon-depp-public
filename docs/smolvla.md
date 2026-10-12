# SmolVLA: a fast-to-train policy on Modal

SmolVLA (`lerobot/smolvla_base`, about 450M parameters) is a third policy option, next to pi07 at PI and the pi0.5
fallback (`docs/pi05_serving.md`). It trains in about an hour on one L40S. It is served with the same `/infer` +
`/health` contract and the same bearer token as pi0.5, so the runner and the dashboard's **pi0.5 (Modal URL)**
backend work with it unchanged. The backend is generic HTTP: paste the SmolVLA URL there.

| file | what |
|---|---|
| `modal_jobs/train_smolvla.py` | `lerobot-train --policy.path=lerobot/smolvla_base`, dataset from the `pi05-datasets` volume, runs into the `smolvla-runs` volume |
| `modal_jobs/serve_smolvla.py` | Modal web endpoint (app `smolvla-serve`). It reuses `serve_pi05.Handler` for auth, validation, camera rename, resize and clipping, and adds `SmolVLAPolicy.predict_action_chunk` on the model side |

Every command below spends money. Run them yourself, from the repo root on the laptop.

## How our cameras map onto SmolVLA

`smolvla_base`'s `config.json` declares three image inputs: `observation.images.camera1`, `camera2` and `camera3`,
each 3×256×256 (declared size only: the model pads and resizes every image to 512×512 itself). It also declares
`empty_cameras: 0`, state 6 and action 6. State and action are zero-padded to 32 inside the model, so our 5 joints
fit. The model drops a slot that gets no image, so two cameras work.

| our camera | SmolVLA slot |
|---|---|
| front | camera1 (top / front view in the SmolVLA recipe) |
| wrist | camera2 (wrist) |
| side or world (later) | camera3 |

The default is `--rename '{"front": "camera1", "wrist": "camera2"}'`, which fits pilot-best7. The job stops before
any GPU time is spent unless every dataset camera maps to its own slot. An unmapped camera would otherwise be
ignored by the model without any error.

## 1. The dataset on the `pi05-datasets` volume

SmolVLA uses the same volume as pi0.5. The 7-episode pilot is `data/pilot-best7` (92 MB, 5,085 frames at 15 fps,
front + wrist 480×480, AV1). `data/cone-pnp-s1-pilot7` has the same content. First check whether it is already
uploaded:

```bash
uv run modal volume ls pi05-datasets /pilot-best7              # expect meta/ data/ videos/ ...
# only if missing:
uv run modal volume put pi05-datasets data/pilot-best7 /pilot-best7
```

## 2. Train

```bash
uv run modal run --detach modal_jobs/train_smolvla.py --volume-dataset pilot-best7
uv run modal app logs smolvla-train                            # or the URL `modal run` printed
uv run modal volume ls smolvla-runs                            # the run: smolvla_pilot-best7_<MMDD_HHMM>
```

Defaults: 3000 steps × batch 64 = 192k samples, which is about 38 passes over the 5,085 pilot frames. The run uses
an L40S with 8 CPU cores and 32 GB RAM for video decoding, 8 dataloader workers, and a checkpoint at step 1500 and
at the end. lerobot scales the LR warmup and cosine decay to the run length on its own (1000/30000 → 100/3000).
Options:

```bash
... --steps 2500 --batch 64                                    # ~31 passes
... --rename '{"front": "camera1", "wrist": "camera2", "side": "camera3"}'   # a 3-camera dataset
SMOLVLA_TRAIN_GPU=A100-80GB uv run modal run --detach modal_jobs/train_smolvla.py --volume-dataset pilot-best7
```

The run folder gets `checkpoints/`, `rename.json` and `dataset.json`, as the pi0.5 job does. The server reads both
files. The prompt at run time has to match the episode task string exactly. pilot-best7 has two:

- `pick the cone up from the blue square and move it onto the red square`
- `pick the cone up from the red square and move it onto the blue square`

## 3. Token (already done for pi0.5)

The server uses the same Modal secret `pi05-serve` / `PI05_SERVE_TOKEN` as pi0.5, so the existing `.env` and
`runner/.env` work as they are. If that secret doesn't exist yet, create it with step 3 of `docs/pi05_serving.md`.

## 4. Deploy

```bash
SMOLVLA_SERVE_RUN=smolvla_pilot-best7_0926_2100 uv run modal deploy modal_jobs/serve_smolvla.py   # pin the run
uv run modal deploy modal_jobs/serve_smolvla.py                # newest run in smolvla-runs (changes as you train)
SMOLVLA_SERVE_GPU=L4 uv run modal deploy modal_jobs/serve_smolvla.py                              # cheaper GPU
```

The URL is `https://<workspace>--smolvla-serve-policyserver-web.modal.run`. The app is `smolvla-serve`, separate from
`pi05-serve`, so both can be deployed at once. Each server has max 1 container, stays warm 10 min after the last
request, and does one warm-up forward pass at load. The server refuses to start if the run is not a SmolVLA checkpoint
or if `rename.json` disagrees with the rename map saved in the checkpoint. `SMOLVLA_SERVE_VOLUME` picks another runs
volume (default `smolvla-runs`).

```bash
curl -s -H "Authorization: Bearer $PI05_SERVE_TOKEN" https://<url>/health   # policy_type "smolvla", run, cameras, image_hw [480, 480]
uv run scripts/pi_replay.py data/pilot-best7 --url https://<url> --episode 0   # recorded frames, nothing moves
uv run modal app stop smolvla-serve                            # stop paying for the warm GPU
```

## 5. Dashboard

This works like pi0.5 (`docs/pi05_serving.md` step 5). Set the Policy runner fields as follows:

| field | value |
|---|---|
| backend | **pi0.5 (Modal URL)** (generic HTTP, works for SmolVLA) |
| URL | the SmolVLA URL from `modal deploy` |
| prompt | one of the task strings above |
| training dataset | pilot-best7 (needs `runner/datasets/pilot-best7/` with meta/info.json + crops.json) |

The runner log shows `policy server: run smolvla_…`, the served run name.

## Cost (estimates: check modal.com/pricing before relying on them)

Assumed prices: L40S $1.95/h, H100 $3.95/h, L4 $0.80/h, CPU $0.047/core·h, RAM $0.008/GiB·h. Assumed speed:
~1 s/step at batch 64 on L40S. That number is extrapolated from lerobot's "20k steps ≈ 4 h on one A100" for SmolVLA
at batch 64, and has not been measured here. Read the real s/step from the first `modal app logs smolvla-train` lines
and redo the estimate.

| item | N | estimate |
|---|---|---|
| training, 3000 × 64 on L40S | ~1 s/step × 3000 + ~10 min setup (first image build + downloads) ≈ 1 h × ($1.95 + 8 × $0.047 + 32 × $0.008 ≈ $2.6/h) | ≈ $2.6 per run |
| 3 tries in a night | 3 × $2.6 | ≈ $8 |
| serving, warm L40S | per hour up | ≈ $2/h (L4 ≈ $0.9/h) |
| one test session | ~2 min cold start + driving + 10 min warm tail | ≈ $0.40 + $2/h of driving |
| forgotten runner | keeps the container warm | $2/h until Stop runner / `modal app stop smolvla-serve` |

## Compared to pi0.5

| | pi0.5 (`train_pi05.py`) | SmolVLA (`train_smolvla.py`) |
|---|---|---|
| parameters | ~3B | ~450M (vision encoder frozen, action expert trained) |
| default training | 2000 × 16 on H100 ≈ 45 min ≈ $3 (3000 × 32 ≈ 1.8 h ≈ $7) | 3000 × 64 on L40S ≈ 1 h ≈ $2.6 |
| samples per $ | 3000 × 32 / $7 ≈ 14k | 192k / $2.6 ≈ 74k (≈ 5×) |
| cold start (serve) | minutes (~7 GB of weights) | shorter (~2 GB: the SmolVLM2 backbone + the checkpoint, cached in hf-cache / the runs volume) |
| per-request latency | slower (3B, 10 flow steps) | expected several times faster (450M, 10 flow steps). Measure with `server_ms` |
| chunk | 50 × 5 | 50 × 5 |

All the latency and speed figures above are estimates. The first real run's s/step and `server_ms` replace them.
