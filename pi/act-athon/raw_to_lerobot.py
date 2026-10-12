"""Convert raw act-athon-pi-episode/1 episodes (HF Actathon-Depp/excavator-raw) into a LeRobot dataset.

Raw layout (per episode dir, e.g. 2026-09-26_session2/20260926_171736_move_the.../):
    meta.json     format/task/fps/channels/signs/action_source ("cab" = operator drove, actions are cab joysticks)
    steps.jsonl   one line per frame: {"i", "t", "action": {swing,boom,arm,bucket,thumb}, "state": {...gate readback}}
    front/ wrist/ world/   000000.jpg ... per-frame JPEGs, index == step "i"

Output: action = the 5 {ch}.cmd floats of the step; observation.state = the PREVIOUS step's action (zeros at step 0),
the "previous command" the runner (scripts/control_loop.py) shows the policy at deployment -- same as
scripts/convert_pi_episodes.py. (Using the step's own action as its state leaks the label: the policy learns
action = state and replays its last command.) Plus one observation.images.<name> video feature per camera.

Images: square center crop (the runner can only crop squares), written to <root>/crops.json in the runner's format so
control_loop.py --dataset crops the same pixels. Episodes with a camera frame or gate state older than --max-age
(the recorder's per-step `age` / `poll_age`, when present) are rejected, like convert_pi_episodes.py; a step with a
missing JPEG is skipped and counted (its action still becomes the next frame's state).

    .venv/bin/python pi/act-athon/raw_to_lerobot.py --out-repo Actathon-Depp/cone-pnp-s2 --push
    .venv/bin/python pi/act-athon/raw_to_lerobot.py --include "2026-09-26_session2/*" --cams front wrist world
"""
import argparse
import fnmatch
import json
from pathlib import Path

import av
import cv2
import numpy as np
import torch
from huggingface_hub import HfApi, snapshot_download
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets import LeRobotDataset

CHANNELS = ("swing", "boom", "arm", "bucket", "thumb")


def load_episode(ep_dir: Path, cams):
    meta = json.loads((ep_dir / "meta.json").read_text())
    steps = [json.loads(l) for l in (ep_dir / "steps.jsonl").read_text().splitlines() if l.strip()]
    return meta, steps, len(steps)


def stale_reason(steps, cams, max_age):
    """Why the episode must be rejected (a frame / gate state older than max_age), or None. Uses the recorder's
    per-step cams.<cam>.age (arrival age) and state.poll_age when present -- the checks convert_pi_episodes.py makes,
    so no training step has older inputs than the runner accepts (it stops beyond 0.5 s)."""
    for s in steps:
        st = s.get("state")
        if isinstance(st, dict) and st.get("poll_age") is not None and st["poll_age"] > max_age:
            return f"step {s['i']}: gate state {st['poll_age']:.2f} s old"
        for c in cams:
            v = (s.get("cams") or {}).get(c)
            if isinstance(v, dict) and v.get("age") is not None and v["age"] > max_age:
                return f"step {s['i']}: {c} frame {v['age']:.2f} s old"
    return None


def find_episodes(roots, cams, include=None):
    candidates = set()
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"raw directory does not exist: {root}")
        for meta in root.rglob("meta.json"):
            relative = meta.relative_to(root)
            if any(part.endswith(".partial") for part in meta.parts):
                continue
            if include and not fnmatch.fnmatchcase(relative.as_posix(), include):
                continue
            candidates.add(meta.parent.resolve())
    episodes = sorted(d for d in candidates
                      if all((d / c).is_dir() for c in cams) and (d / "steps.jsonl").is_file())
    print(f"{len(episodes)} saved episodes selected; "
          f"{len(candidates) - len(episodes)} excluded for missing cameras/steps", flush=True)
    return episodes


def read_image(path, center_crop):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    h, w = image.shape[:2]
    size = min(h, w)
    x, y = (w - size) // 2, (h - size) // 2
    crop = [x, y, size] if center_crop else [0, 0, w, h]
    if center_crop:
        image = image[y:y + size, x:x + size]
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), crop


def load_outcomes(path, episode_ids):
    if path is None:
        return {}
    outcomes = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        source_id = row["source_episode_id"]
        if source_id in outcomes:
            raise ValueError(f"duplicate outcome for {source_id}")
        if row["outcome"] not in {"success", "failure", "uncertain", "not_applicable"}:
            raise ValueError(f"invalid outcome for {source_id}: {row['outcome']}")
        outcomes[source_id] = row
    missing = set(episode_ids) - outcomes.keys()
    if missing:
        raise ValueError(f"missing outcome metadata for: {sorted(missing)}")
    return outcomes


def validate_video_counts(root, cameras, expected_frames):
    for camera in cameras:
        frames = 0
        for path in (root / "videos" / f"observation.images.{camera}").rglob("*.mp4"):
            with av.open(str(path)) as video:
                frames += video.streams.video[0].frames
        if frames != expected_frames:
            raise ValueError(f"{camera} video has {frames} frames; expected {expected_frames}")


def main(args):
    if args.raw_dir:
        roots = [Path(p).expanduser() for p in args.raw_dir]
    else:
        roots = [Path(snapshot_download(args.raw_repo, repo_type="dataset", max_workers=32,
                                       allow_patterns=[args.include] if args.include else None))]
    ep_dirs = find_episodes(roots, args.cams, args.include)
    if not ep_dirs:
        raise SystemExit(f"no complete episodes with cams {args.cams} under {roots}")
    print(f"{len(ep_dirs)} episodes to convert (cams: {args.cams})", flush=True)
    episode_ids = [ep.name for ep in ep_dirs]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("duplicate source episode names across raw directories")
    outcomes = load_outcomes(args.outcomes, episode_ids)
    if not args.low_memory and args.video_preset != "ultrafast":
        raise ValueError("--video-preset requires --low-memory")
    if not args.center_crop:
        raise SystemExit("--no-center-crop: the runner always square-crops (act.rig_machine.prepare_image), so a "
                         "full-frame dataset would train on other pixels than the policy sees")

    # image sizes per camera from the first episode (all episodes must match)
    sizes = {}
    for c in args.cams:
        image, _ = read_image(sorted((ep_dirs[0] / c).glob("*.jpg"))[0], args.center_crop)
        h, w = image.shape[:2]
        sizes[c] = (h, w)
    fps = json.loads((ep_dirs[0] / "meta.json").read_text()).get("fps", 15)

    names = [f"{ch}.cmd" for ch in CHANNELS]
    features = {
        "action": {"dtype": "float32", "shape": (5,), "names": names},
        "observation.state": {"dtype": "float32", "shape": (5,), "names": names},
        **{f"observation.images.{c}": {"dtype": "video", "shape": (*sizes[c], 3),
                                       "names": ["height", "width", "channels"]} for c in args.cams},
    }
    root = Path(args.root)
    if root.exists():
        raise SystemExit(f"{root} exists: remove it or pass a different --root")
    encoding = {}
    if args.low_memory:
        torch.set_num_threads(1)
        cv2.setNumThreads(1)
        encoding = {
            "video_backend": "pyav",
            # The real-time streaming writer drops frames on a full queue; offline conversion must not.
            "streaming_encoding": False,
            "encoder_threads": 1,
            "rgb_encoder": RGBEncoderConfig(vcodec="h264", crf=18, preset=args.video_preset),
        }
    ds = LeRobotDataset.create(repo_id=args.out_repo, fps=int(fps), features=features, root=root,
                               robot_type="excavator_real", use_videos=True,
                               image_writer_threads=0 if args.low_memory else 8, **encoding)

    sources, rejected, dataset_crops = {}, {}, {}
    joined_outcomes = []
    for k, ep in enumerate(ep_dirs):
        meta, steps, n = load_episode(ep, args.cams)
        stale = stale_reason(steps, args.cams, args.max_age)
        if stale:
            rejected[ep.name] = stale
            print(f"[{k + 1}/{len(ep_dirs)}] {ep.name[:60]}: rejected: {stale}", flush=True)
            continue
        skipped = 0
        frame_indices = []
        crops = {}
        prev = np.zeros(len(CHANNELS), dtype=np.float32)   # the previous STEP's command (kept or skipped)
        for s in steps:
            i = s["i"]
            imgs = {}
            for c in args.cams:
                p = ep / c / f"{i:06d}.jpg"
                if not p.exists():
                    break
                imgs[c], crops[c] = read_image(p, args.center_crop)
            a = np.array([float(s["action"].get(ch, 0.0)) for ch in CHANNELS], dtype=np.float32)
            state, prev = prev, a
            if len(imgs) != len(args.cams):
                skipped += 1
                continue
            ds.add_frame({"action": a, "observation.state": state,   # state = previous command (runner's)
                          **{f"observation.images.{c}": imgs[c] for c in args.cams},
                          "task": meta["task"]})
            frame_indices.append(i)
        if not frame_indices:
            rejected[ep.name] = "no step with every camera's frame"
            continue
        ds.save_episode()
        for c, crop in crops.items():
            if dataset_crops.setdefault(c, crop) != crop:
                raise SystemExit(f"{ep.name}: {c} crop {crop} differs from {dataset_crops[c]} (frame sizes differ)")
        sources[ep.name] = {
            "episode_index": len(sources),
            "source_path": str(ep),
            "task": meta["task"],
            "action_source": meta.get("action_source"),
            "cab_map": meta.get("cab_map"),
            "cameras": args.cams,
            "crops": crops,
            "source_frame_indices": frame_indices,
            "frames": len(frame_indices),
            "skipped_frames": skipped,
        }
        if outcomes:
            joined_outcomes.append({**outcomes[ep.name], "episode_index": sources[ep.name]["episode_index"]})
        print(f"[{k + 1}/{len(ep_dirs)}] {ep.name[:60]}: {n - skipped} frames"
              + (f" ({skipped} skipped)" if skipped else ""), flush=True)

    ds.finalize()
    validate_video_counts(root, args.cams, ds.meta.total_frames)
    (root / "source_episodes.json").write_text(json.dumps(sources, indent=2) + "\n")
    (root / "crops.json").write_text(json.dumps(dataset_crops) + "\n")   # control_loop.py --dataset reads it
    if rejected:
        (root / "rejected_episodes.json").write_text(json.dumps(rejected, indent=2) + "\n")
        print(f"{len(rejected)} episodes rejected (rejected_episodes.json)", flush=True)
    if joined_outcomes:
        (root / "meta" / "episode_outcomes.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in joined_outcomes)
        )
        (root / "meta" / "source_outcomes.jsonl").write_text(
            "".join(json.dumps({**row, "included_in_training": source_id in sources}) + "\n"
                    for source_id, row in sorted(outcomes.items()))
        )
    print(f"done: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames at {root}", flush=True)
    if args.push:
        api = HfApi()
        api.create_repo(args.out_repo, repo_type="dataset", private=True, exist_ok=True)
        if not api.dataset_info(args.out_repo).private:
            raise ValueError(f"refusing to upload raw-derived recordings to public repo {args.out_repo}")
        print(f"pushing to https://huggingface.co/datasets/{args.out_repo} (private) ...")
        ds.push_to_hub(private=True)
        print("pushed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-repo", default="Actathon-Depp/excavator-raw")
    ap.add_argument("--raw-dir", nargs="+", default=None,
                    help="local directories of saved raw episodes (skips the HF download)")
    ap.add_argument("--out-repo", default="Actathon-Depp/cone-pnp-s2")
    ap.add_argument("--root", default="data/cone_pnp_s2", help="local output dir (must not exist)")
    ap.add_argument("--include", default=None, help="glob of episode dirs to convert, e.g. '2026-09-26_session2/*'")
    ap.add_argument("--cams", nargs="+", default=["front", "wrist", "world"])
    ap.add_argument("--low-memory", action="store_true",
                    help="use synchronous disk-backed H.264 encoding rather than real-time queues (Pi)")
    ap.add_argument("--video-preset", default="ultrafast",
                    choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium"],
                    help="H.264 CPU/compression tradeoff when using --low-memory")
    ap.add_argument("--center-crop", action=argparse.BooleanOptionalAction, default=True,
                    help="center-crop each camera to a square (the runner's default crop; required)")
    ap.add_argument("--max-age", type=float, default=0.5,
                    help="reject an episode if a frame / gate state is older (s), as convert_pi_episodes.py")
    ap.add_argument("--outcomes", default=None,
                    help="JSONL weak outcome metadata keyed by source_episode_id; does not filter training data")
    ap.add_argument("--push", action="store_true")
    main(ap.parse_args())
