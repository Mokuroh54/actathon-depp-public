"""Offline check of a checkpoint (PI-hosted pi07, or our Modal pi0.5 / SmolVLA server): replay a recorded episode's
frames + state to the model and compare its action chunks with what the operator actually did. Nothing moves (no
gate, no machine).

    uv run scripts/pi_replay.py data/pipeline-test --gateway-url <checkpoint gateway_url> [--episode 0] [--every 15]
    uv run scripts/pi_replay.py data/pilot-best7 --url https://<...>.modal.run            # serve_pi05 / serve_smolvla
    uv run scripts/pi_replay.py data/pilot-best7-nostate --url https://<...>.modal.run --zero-state
    uv run scripts/pi_replay.py data/s24 --gateway-url <...> --episode 0 5 11 18 20 23 --every 10 \
        --subtasks outputs/subtask_labels/s24.json

--gateway-url: a PI-hosted checkpoint (PIPolicy), PI_API_KEY from the environment or .env. --url: our HTTP policy
server (HTTPPolicy, bearer token PI05_SERVE_TOKEN from the environment or .env), frames sent at the dataset's image
size; its chunk is cut to --chunk steps so the numbers compare with the PI replays.
--zero-state: the model is sent state = zeros (checkpoints trained on a scripts/zero_state.py dataset); required iff
the dataset has meta/zero_state.json, as in control_loop.py.

Every --every steps it sends that step's images + observation.state with the episode's task string, and scores the
returned chunk against the recorded actions for the same steps (MAE per channel, -1..1 joystick units). Two baselines
on the same steps:
  zeros      always neutral
  repeat     keep repeating the previous command (the RECORDED observation.state, also with --zero-state; in a
             zero-state dataset, whose state column is zeros, the recorded previous command is the previous step's
             action, which is what convert_pi_episodes.py stores as state): a model that learned only "keep doing
             what you were doing" scores about this well while ignoring the cameras
--subtasks LABELS.json (scripts/subtask_labels.py output; --gateway-url only): every request also carries the labelled
phase of its own step as obs[SUBTASK_KEY], the way control_loop.py --subtasks does (PIPolicy sends it as raw_text).
Episodes are matched like pi_upload_spans.py (pi_sources.json: raw episode folder -> dataset episode); an episode
without labels is skipped. Adds the MAE per phase (each scored step counted under its own labelled phase).
Always reported: thumb grip recall = of the scored steps where the recorded thumb clamps (>= 0.3), the fraction where
the prediction clamps too.

A model that learned the episode beats both clearly, on the first action and over the chunk. On the training episode
itself (overfit check) that is a pipeline test; only held-out episodes say anything about generalisation.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from lerobot.datasets import LeRobotDataset

from act.machine import CHANNELS
from act.policies import (SUBTASK_KEY, HTTPPolicy, PIPolicy, ZeroState, check_zero_state, pi05_serve_token, pi_api_key,
                          zero_state_marker)

THUMB, CLAMP = CHANNELS.index("thumb"), 0.3     # thumb >= 0.3: clamping (act.phase's clamp threshold)


class DatasetCfg:
    """The bits of a machine config PIPolicy / HTTPPolicy need: joint order and camera names (dataset features)."""

    def __init__(self, cameras):
        self.joint_names, self.cameras = list(CHANNELS), {c: None for c in cameras}


def to_rgb_uint8(t):
    """LeRobot frame (C, H, W) float 0..1 -> (H, W, C) uint8 RGB, as RigMachine returns it."""
    return (t.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)


def mae(pred, true):
    return np.abs(pred - true).mean(axis=0)


def load_subtasks(root, path):
    """--subtasks: dataset episode index -> label segments. Keyed like pi_upload_spans.py: the labels are per raw
    episode folder, pi_sources.json maps that folder to the dataset episode. Episodes without segments are left out."""
    labels = json.loads(Path(path).read_text())
    sources = json.loads((root / "pi_sources.json").read_text())
    return {src["episode"]: labels[name]["segments"] for name, src in sources.items()
            if labels.get(name, {}).get("segments")}


def phase_at(segments, k):
    """The labelled subtask of episode step k (segments: start_step inclusive, end_step exclusive, contiguous)."""
    for seg in segments:
        if seg["start_step"] <= k < seg["end_step"]:
            return seg["subtask"]
    raise ValueError(f"step {k} is outside the labelled segments")


def infer_retry(policy, obs, prompt, tries=5):
    """policy.infer, backing off on quota errors (resource_exhausted / 429): waits 15, 30, 60, 120 s, then gives up."""
    for t in range(tries):
        try:
            return policy.infer(obs, prompt)
        except Exception as e:
            if t == tries - 1 or not any(m in str(e).lower() for m in ("resource_exhausted", "resourceexhausted", "429")):
                raise
            print(f"\n{type(e).__name__}: {e}: retrying in {15 * 2 ** t} s", flush=True)
            time.sleep(15 * 2 ** t)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path, help="LeRobot dataset (from convert_pi_episodes.py)")
    target = ap.add_mutually_exclusive_group(required=True)
    target.add_argument("--gateway-url", help="PI-hosted checkpoint (pi07)")
    target.add_argument("--url", help="our HTTP policy server (modal_jobs/serve_pi05.py or serve_smolvla.py)")
    ap.add_argument("--zero-state", action="store_true",
                    help="send state = zeros (checkpoint trained on a scripts/zero_state.py dataset)")
    ap.add_argument("--episode", type=int, nargs="+", default=[0], help="dataset episode index(es); scores are pooled")
    ap.add_argument("--every", type=int, default=15, help="query every N steps (default: one chunk apart)")
    ap.add_argument("--chunk", type=int, default=15,
                    help="actions scored per request (PI: execute_chunk_size; --url: the server's chunk cut to this)")
    ap.add_argument("--prompt", help="task string (default: the episode's recorded task)")
    ap.add_argument("--subtasks", type=Path,
                    help="subtask_labels.py JSON: send each step's labelled phase as the subtask (PI raw_text)")
    args = ap.parse_args()
    if args.subtasks and not args.gateway_url:
        ap.error("--subtasks is for --gateway-url (the phase goes out as PI's raw_text; our servers take no subtask)")

    check_zero_state(args.root, args.zero_state)
    zeroed = zero_state_marker(args.root) is not None
    ds = LeRobotDataset(f"local/{args.root.name}", root=args.root)
    cams = [k.removeprefix("observation.images.") for k in ds.meta.features if k.startswith("observation.images.")]
    episodes = args.episode
    if args.subtasks:
        subtasks = load_subtasks(args.root, args.subtasks)
        for e in [e for e in episodes if e not in subtasks]:
            print(f"skip episode {e}: no subtask labels in {args.subtasks}")
        episodes = [e for e in episodes if e in subtasks]
        if not episodes:
            raise SystemExit("no episode left to replay")
    print(f"episodes {episodes} at {ds.fps} fps, cameras {cams}, subtasks {args.subtasks or 'off'}")

    if args.gateway_url:
        policy = PIPolicy(args.gateway_url, DatasetCfg(cams), pi_api_key(), execute_chunk_size=args.chunk)
        print("connecting to PI (the first connection can take minutes while the model starts) ...")
        meta = policy.connect()
        print(f"PI server: cameras {policy.cam_map}, action horizon {meta.action_horizon}, action dim {meta.action_dim}")
    else:
        token = pi05_serve_token()
        if not token:
            print("no PI05_SERVE_TOKEN (environment or .env): sending no bearer token (the server will answer 401)")
        hw = {tuple(ds.meta.features[f"observation.images.{c}"]["shape"][:2]) for c in cams}
        if len(hw) != 1:
            raise SystemExit(f"expected one image size across cameras, got {hw}")
        # same as control_loop.py --policy http --dataset: frames at the training size, near-lossless JPEG
        policy = HTTPPolicy(args.url, DatasetCfg(cams), jpeg_quality=95, timeout_s=30.0, token=token, image_hw=hw.pop(),
                            zero_state=args.zero_state)
        print("connecting to the policy server (a cold start loads the model: can take minutes) ...")
        info = policy.connect()
        print(f"policy server: run {info.get('run')}, type {info.get('policy_type', 'pi05')}, cameras "
              f"{info.get('cameras')}, images {info.get('image_hw')}, chunk {info.get('chunk_size')}, "
              f"zero_state {info.get('zero_state')}")
    if args.zero_state:
        policy = ZeroState(policy, CHANNELS)
        print("--zero-state: the model is sent state = zeros")

    rows = {"model": [], "zeros": [], "repeat": []}
    first = {"model": [], "zeros": [], "repeat": []}
    pred_all = {"model": [], "zeros": [], "repeat": []}     # scored predictions, for the grip recall / spread
    true_all, phases, first_idx = [], [], []                # recorded actions, labelled phase per scored step
    t0 = time.monotonic()
    try:
        for e in episodes:
            ep = ds.meta.episodes[e]
            lo, hi = ep["dataset_from_index"], ep["dataset_to_index"]
            prompt = args.prompt or ep["tasks"][0]
            segs = subtasks[e] if args.subtasks else None
            actions = np.stack([ds.hf_dataset[i]["action"] for i in range(lo, hi)]).astype(float)   # no video decode
            print(f"episode {e}: {hi - lo} steps, task {prompt!r}")
            r0 = len(rows["model"])
            for i in range(lo, hi, args.every):
                x = ds[i]
                obs = {c: to_rgb_uint8(x[f"observation.images.{c}"]) for c in cams}
                state = x["observation.state"].numpy().astype(float)
                obs.update({f"{j}.cmd": float(v) for j, v in zip(CHANNELS, state)})
                if segs:
                    obs[SUBTASK_KEY] = phase_at(segs, i - lo)     # as control_loop.py --subtasks: this step's phase
                # recorded previous command for the "repeat" baseline (a zero-state dataset keeps it only in the actions)
                prev = (actions[i - lo - 1] if i > lo else np.zeros(len(CHANNELS))) if zeroed else state
                chunk = infer_retry(policy, obs, prompt)[:args.chunk]
                n = min(len(chunk), hi - i)
                true = actions[i - lo:i - lo + n]
                preds = {"model": chunk[:n], "zeros": np.zeros_like(true), "repeat": np.tile(prev, (n, 1))}
                first_idx.append(len(phases))                     # index of this request's first scored step
                for k, p in preds.items():
                    rows[k].append(np.abs(p - true))
                    first[k].append(np.abs(p[0] - true[0]))
                    pred_all[k].append(p)
                true_all.append(true)
                phases += [phase_at(segs, i - lo + t) if segs else "-" for t in range(n)]
                print(f"\rstep {i - lo:4d}/{hi - lo}  model first-action MAE so far "
                      f"{np.mean(first['model']):.3f}", end="", flush=True)
            print("\n  episode " + "   ".join(f"{k} {np.concatenate(rows[k][r0:]).mean():.3f}" for k in rows)
                  + " (whole chunk MAE)")
    finally:
        if hasattr(policy, "close"):
            policy.close()

    true_all, phases = np.concatenate(true_all), np.array(phases)
    print(f"\n{len(first['model'])} requests over {len(episodes)} episode(s) in {time.monotonic() - t0:.0f} s, "
          f"{len(true_all)} scored steps, {args.chunk}-step chunks; MAE in joystick units (-1..1), "
          f"channels {' '.join(CHANNELS)}")
    for k in rows:
        chunk_mae, first_mae = np.concatenate(rows[k]).mean(axis=0), np.mean(first[k], axis=0)
        print(f"  {k:6s}  first action {first_mae.mean():.3f} [{' '.join(f'{v:.2f}' for v in first_mae)}]"
              f"   whole chunk {chunk_mae.mean():.3f} [{' '.join(f'{v:.2f}' for v in chunk_mae)}]")
    if args.subtasks:
        print("per phase (whole chunk, each scored step under its own labelled phase):")
        for ph in dict.fromkeys(phases):
            sel = phases == ph
            print(f"  {ph} ({sel.sum()} steps)")
            for k in rows:
                m = np.concatenate(rows[k])[sel].mean(axis=0)
                print(f"    {k:6s}  {m.mean():.3f} [{' '.join(f'{v:.2f}' for v in m)}]")
    clamp = true_all[:, THUMB] >= CLAMP
    fclamp = clamp[first_idx]
    print(f"thumb grip recall (recorded thumb >= {CLAMP}: {clamp.sum()} scored steps, {fclamp.sum()} first actions; "
          f"fraction where the prediction is >= {CLAMP} too):")
    for k in rows:
        p = np.concatenate(pred_all[k])[:, THUMB] >= CLAMP
        print(f"  {k:6s}  whole chunk {p[clamp].mean() if clamp.any() else float('nan'):.2f}"
              f"   first action {p[first_idx][fclamp].mean() if fclamp.any() else float('nan'):.2f}"
              f"   (predicts a clamp on {p.mean():.2f} of all scored steps)")
    spread = np.concatenate(pred_all["model"]).std(axis=0)
    print(f"spread (std) of the model's actions [{' '.join(f'{v:.2f}' for v in spread)}], "
          f"recorded [{' '.join(f'{v:.2f}' for v in true_all.std(axis=0))}]")
    if policy.last_server_ms is not None:
        print(f"last server time {policy.last_server_ms:.0f} ms")


if __name__ == "__main__":
    main()
