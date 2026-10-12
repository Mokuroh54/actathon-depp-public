# AGENTS.md

Context for coding agents working in this repo. Read README.md for commands, layout and the action schema;
this file covers the goal, the plan and the rules.

## Goal

Act-athon physical-AI hackathon. **Judging: Sunday 2026-09-27, 12:00 (noon), outdoors, in daylight.**
The task is ours to define: **trash pick-and-place** with the mini excavator (thumb + bucket grab an
object from a pickup area and drop it in a zone). The policy is **pi07** (Physical Intelligence), fine-tuned
on our cab demos through PI's Partner API (`pi-sdk`; PI's Partner API guide, not included here) and served by PI's hosted model gateway,
driven through `scripts/control_loop.py --machine real` with a `PIPolicy` backend in `src/act/policies.py`
(to write: wraps `pi_sdk.inference.PolicyClient` behind the same `infer(obs, prompt) -> (H, D)` interface).
The pi0.5-on-Modal path (`modal_jobs/`) stays in the repo as a fallback only if PI's service blocks us;
don't develop it further.

Prompts are passed explicitly (`--prompt`; the dashboard's Inference tab always passes one). Keep the prompt
strings identical between recording, training and the demo.

## Plan

| when | what |
|---|---|
| Fri night | make real recording work end to end: cab demo on the Pi dashboard → rsync → `convert_pi_episodes.py` → `pi-upload-lerobot` → short PI fine-tune (~1k steps) → hosted checkpoint (`gateway_url`) → `PIPolicy` on the machine. Night episodes are throwaway (pipeline test only). |
| Sat day | collect the training demos, late morning to mid-afternoon. Quick PI fine-tune on the first batch around midday to catch data bugs and to measure PI's turnaround (fine-tune + hosting time). |
| Sat night | main PI fine-tune; watch `list_finetune_evaluations` MAE; night run on the machine for mechanics/latency only, not task success. |
| Sun morning | real eval in daylight, pick the checkpoint; top-up (`continue_from_checkpoint_id`) only if the measured turnaround says it will be hosted before noon. |

## Recording

Demos are recorded **on the Pi**, by a human operator driving from the cab with the machine's own joysticks
(`docs/dashboard.md`): `pi/act-athon/camview.py` owns the cameras, `pi/act-athon/dashboard.py` (Pi :8080)
has the Record episode panel, and `scripts/convert_pi_episodes.py` turns rsynced episodes into a LeRobot dataset
(action = cab joysticks via the cab map; raw `controls` kept so the map can be fixed at conversion).
There is no laptop-side recorder any more (`scripts/record.py` was removed).

Joint calibration (`pi/act-athon/calib.py`, `docs/calibration.md`) only produces measurements (dead band, joint
speed, range); don't prioritise it.
Unverified: that a cab value and the same value sent through the gate move the arm alike. Check by replaying a
cab episode's actions through the gate before relying on it.

## pi07 via PI's Partner API

Guide: PI's Partner API guide, not included here (contract → episodes → training profile → dataset filter → fine-tune → hosted inference).

- **Two credentials.** PI's service-account JSON only authenticates installs from PI's package registry
  (`GOOGLE_APPLICATION_CREDENTIALS`, keyring + `keyrings.google-artifactregistry-auth`). The organisation API
  key `PI_API_KEY` (Partner Portal) authenticates every API call. Neither goes in the repo.
- **Two environments (lerobot version clash).** The project env has `pi-sdk[video,inference]` (pyproject pins it
  to PI's index, `explicit = true`) next to lerobot 0.6.1. `pi-upload-lerobot` needs lerobot 0.4.x, so it lives
  in a uv tool env (`uv tool install "pi-sdk[lerobot]" --extra-index-url <PI index>`). Never add
  `pi-sdk[lerobot]` to the project. Both lerobot versions use dataset format v3.0; the 0.6.1 → 0.4.x hand-off
  is unverified: test one episode first. Fallback: write PI MCAP shards straight from the raw Pi episodes with
  `pi_sdk.mcap.v1.Writer`.
- **Index order.** With uv, PI's registry goes in `--extra-index-url` (checked first); PyPI's `pi-sdk` is a
  0.0.2 placeholder. Never use `--index-strategy unsafe-best-match`.
- **Contract:** `system/step` at the recording rate (whole Hz, one rate per contract); state (5,) float32 =
  the previous command, action (5,) float32 = `{ch}.cmd` in -1..1; cameras as `VideoFrame` observation topics.
  Contracts and training profiles are immutable: a changed schema, camera set or rate means a new contract.
- **Training profile:** `front` → base camera, `wrist` → wrist camera, `action_space="absolute"`. Keep that
  mapping, the crops (`crops.json` in the dataset root), the prompts and the rate identical in training and
  inference.
- **Episodes:** stable `source_dataset_path` per collection (e.g. `excavator/cab-v1`); mark saved demos
  `success=True`; subtask spans (`language_segment`) are optional but feed the model's subtask instruction.
- **Inference:** `ClientConfig(action_keys=<profile action_topics>, execute_chunk_size=...)`. pi07 predicts
  50-step chunks (paper: robots at 20-50 Hz, executing 15-25 steps per chunk); start around 1-1.5 s of
  execution per chunk at our rate. `AsyncRTCPolicyRunner` keeps the arm moving between chunks.
- **Open questions for PI:** turnaround time, cost/quota, and how chunk length is handled at our rate.

## Data rules

- **Train only on daylight episodes near noon lighting.** Night / dusk episodes may be recorded for pipeline
  tests but must not go into the training set. Record the time and conditions per episode.
- Fixed camera mounts (tape-mark them) and fixed exposure / white balance between collection and the demo.
  Keep the camera → training-profile mapping (`front` → base camera, `wrist` → wrist camera) identical in
  training and inference.
- One recording rate for the whole training set (`--rec-fps`, a whole number): pi07 was trained at 20-50 Hz,
  so use the highest rate the cameras and the Pi sustain (check `late_ticks` and repeated frames per episode).
- Fix throttle and rabbit mode for collection and the demo, and write them down: joint speed depends on them,
  and the policy outputs joystick values, not positions.
- End every demo with a short still period (the policy otherwise repeats the task after finishing).
- Consistent demos beat many demos: same start pose, same recipe, decisive motions.
- Datasets recorded before the -1..1 `{ch}.cmd` schema (`data/fake_excavator`, `data/synth_excavator`:
  4 joints, `.vel`/`.pos`) are incompatible. Don't train on them.

## Safety (real machine)

- Someone holds the kill switch whenever the relay may be live. Nobody within 3 m of the arm or swing radius.
- While the `ssh -L 8096` tunnel is open, anything on the laptop talking to `127.0.0.1:8096` moves the machine.
  Agents must not run `--machine real`, the calibration page, or anything that writes to the gate unless
  the user asks for it in that turn.
- Tracks and blade are locked by the organisers' gate; don't try to work around it.

## Cost and compute

- Don't create PI resources (contracts, episodes, profiles, filters, fine-tunes) or start / cancel a PI
  fine-tune without the user's go-ahead: they are org-scoped, mostly immutable, and fine-tunes cost quota.
  Say the planned `num_steps` × `batch_size` and the episode / step count first.
- Same for Modal jobs (fallback path): estimate GPU-hours × runs and say the number before launching.

## Secrets

- Never commit tokens, keys or the Pi password. Put them in a git-ignored `.env` or Modal secrets.
  This includes `PI_API_KEY` and PI's service-account JSON (keep it outside the repo).
