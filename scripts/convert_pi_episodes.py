"""Convert episodes recorded on the excavator's Pi (pi/act-athon/dashboard.py, Record panel) into a LeRobot dataset.

This is the recording path for the real machine: an operator drives from the cab with the machine's own joysticks while
the dashboard records (and sends nothing to the gate); copy the episode folders over, convert here.

    rsync -av excavator:act-athon/episodes/ data/raw/                   # copy (never deletes anything on the Pi)
    uv run scripts/convert_pi_episodes.py data/raw --root data/rig_cab --repo-id local/rig_cab
    uv run scripts/convert_pi_episodes.py data/raw --root data/rig_cab --repo-id local/rig_cab --resume
    uv run scripts/convert_pi_episodes.py data/raw/20260926_101500_pick_up_the_red_block ... --overwrite
    uv run scripts/convert_pi_episodes.py data/raw --root data/rig_cab_v2 --cab-map fixed_map.json   # re-map
    uv run scripts/convert_pi_episodes.py data/raw --root data/rig_cab_v3 --crop front=160,0,480     # shift a crop

Arguments are episode folders or folders that contain them. `<name>.partial` folders (unsaved: discarded in the
browser, or the dashboard stopped mid-episode) are skipped. With --resume, episodes already converted into this
dataset (listed in <root>/pi_sources.json) are skipped, so re-running after each rsync only adds the new ones
(--resume also creates the dataset if --root doesn't exist yet).

Only cab recordings are converted (meta.json action_source = "cab"). Episodes from the older keyboard-teleop recorder
are skipped with a message, and --resume refuses a dataset that already holds such episodes: keyboard demos are
bang-bang (-1/0/+1), cab demos are analog, so they don't belong in one dataset.

Schema (act.dataset.make_features on a RigMachine-shaped feature description):
    action              {ch}.cmd for ch in swing, boom, arm, bucket, thumb: the cab joysticks, RE-DERIVED here from
                        each step's stored raw gate `controls` with the cab map (-1..1, rig signs, 0 inside the
                        deadband). The map is the one recorded in the episode's meta.json unless --cab-map is given
                        (a JSON file in the dashboard's --cab-map format; listed entries replace the episode's), so a
                        map that turns out wrong is fixed by converting again, without re-recording.
    observation.state   {ch}.cmd = the PREVIOUS step's action (zeros at step 0). The gate's `sent` is ~0 during a cab
                        recording (the gate is not driving), so it carries nothing; the previous command is what
                        `sent` means for code-driven runs (the gate outputs the last command it was given), so the
                        policy sees the same "previous command" signal it gets at deployment.
    observation.images.<cam>   the camera's JPEG (captured 640x480), decoded, cropped to a SQUARE box -- default the
                        centered 480x480 (x=80, y=0), per camera --crop NAME=x,y,size / --crop-file (native pixels, must
                        fit the frame) -- and resized to --out-size (default: keep the crop size). Same code as
                        deployment (act.rig_machine.prepare_image); the crops go to <root>/crops.json and must be given
                        to RigMachine (crops=load_crop_file(...)) at deployment, or the policy sees other pixels.
    task                the prompt typed in the Record panel
    fps = the episode's --rec-fps; robot_type "excavator_real"; videos.
An episode is rejected (with the reason printed) if a step has no camera frame or no gate state (or no raw
`controls`), a frame or the gate state is older than --max-age, or two steps are more than 1.5 periods apart (dropped /
late ticks), or a camera's native size differs from the first episode's. Cameras run at about the record rate, so some
steps legitimately reuse the previous step's frame (tick jitter): these are counted and printed, not rejected (a
stalled camera is caught by --max-age). <root>/pi_sources.json records, per converted episode, the dataset index, the
cab map, the crops and the repeated-frame counts.

--end-at-release LABELS.json (scripts/subtask_labels.py --out; overrides applied there): end each episode
--release-margin s (default 0.5) after its labelled release step, dropping the return toward the start pose, so the
last frames show the released cone and not a pose that looks like the start. The release step is the step after the
thumb-open (or dump) run that released the cone ended. An episode without a release label is rejected. The cut goes
into pi_sources.json (trim), and --resume refuses a dataset converted with another cut. A trimmed episode ends
mid-motion, without a still period.
"""
import argparse
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets import LeRobotDataset
from lerobot.utils.feature_utils import build_dataset_frame

from act.dataset import make_features
from act.machine import CHANNELS
from act.rig_machine import RigMachine, RigMachineConfig, crop_box, load_crop_file, parse_crop, prepare_image

ROBOT_TYPE = "excavator_real"
SOURCES = "pi_sources.json"     # in the dataset root: {pi episode folder name: dataset episode index}
CROPS = "crops.json"            # in the dataset root: {camera: [x, y, size]} (act.rig_machine.load_crop_file format)
FORMAT = "act-athon-pi-episode/1"


# ---- cab map: same code as pi/act-athon/dashboard.py (cab_map_merge, cab_action); keep the two in sync
def cab_map_merge(obj, base_map, base_deadband):
    """Map JSON ({channel: [field, sign(, offset)], "deadband": number | {channel: number}}; listed entries replace
    base's) -> (map {ch: [field, sign, offset]}, deadband {ch: float}). Raises ValueError on a malformed entry."""
    if not isinstance(obj, dict):
        raise ValueError("cab map must be a JSON object")
    cmap = {ch: [str(e[0]), float(e[1]), float(e[2]) if len(e) > 2 else 0.0] for ch, e in base_map.items()}
    db = dict(base_deadband) if isinstance(base_deadband, dict) else {ch: float(base_deadband) for ch in CHANNELS}
    for k, v in obj.items():
        if k == "deadband":
            vals = v if isinstance(v, dict) else {ch: v for ch in CHANNELS}
            for ch, x in vals.items():
                if ch not in CHANNELS or isinstance(x, bool) or not isinstance(x, (int, float)) or x < 0:
                    raise ValueError("bad deadband entry %r: %r" % (ch, x))
                db[ch] = float(x)
        elif k in CHANNELS:
            if (not isinstance(v, (list, tuple)) or len(v) not in (2, 3) or not isinstance(v[0], str)
                    or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in v[1:])):
                raise ValueError("bad map entry %r: %r (want [raw field, sign] or [raw field, sign, offset])" % (k, v))
            cmap[k] = [v[0], float(v[1]), float(v[2]) if len(v) > 2 else 0.0]
        else:
            raise ValueError("unknown key %r (channels: %s, or deadband)" % (k, ", ".join(CHANNELS)))
    missing = [ch for ch in CHANNELS if ch not in cmap or ch not in db]
    if missing:
        raise ValueError("cab map has no entry for %s" % missing)
    return {ch: cmap[ch] for ch in CHANNELS}, {ch: db[ch] for ch in CHANNELS}


def cab_action(controls, cmap, deadband):
    """Raw /state `controls` -> {channel: -1..1} (rig signs). A missing / non-numeric field gives 0."""
    out = {}
    for ch in CHANNELS:
        field, sign, off = cmap[ch]
        v = controls.get(field) if isinstance(controls, dict) else None
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            out[ch] = 0.0
            continue
        x = float(v) - off
        out[ch] = 0.0 if abs(x) <= deadband[ch] else max(-1.0, min(1.0, sign * x))
    return out


def find_episodes(paths):
    eps, skipped = [], []
    for p in map(Path, paths):
        cands = [p] if (p / "meta.json").exists() or p.name.endswith(".partial") else sorted(
            d for d in p.iterdir() if d.is_dir()) if p.is_dir() else []
        if not cands:
            skipped.append((p, "not an episode folder (no meta.json) and contains none"))
        for d in cands:
            if d.name.endswith(".partial"):
                skipped.append((d, "unsaved (.partial)"))
            elif not (d / "meta.json").exists() or not (d / "steps.jsonl").exists():
                skipped.append((d, "no meta.json / steps.jsonl"))
            else:
                eps.append(d)
    return eps, skipped


def load_episode(d, max_age):
    """-> (meta, steps) or raises ValueError(reason)."""
    meta = json.loads((d / "meta.json").read_text())
    if meta.get("format") != FORMAT:
        raise ValueError(f"unknown format {meta.get('format')!r}")
    if meta.get("action_source") != "cab":
        raise ValueError(f"not a cab recording (action_source {meta.get('action_source')!r}): keyboard-teleop "
                         "episodes are not converted")
    if not isinstance(meta.get("cab_map"), dict) or meta.get("cab_deadband") is None:
        raise ValueError("meta.json has no cab_map / cab_deadband")
    steps = [json.loads(line) for line in (d / "steps.jsonl").read_text().splitlines() if line.strip()]
    if not steps:
        raise ValueError("no steps")
    if meta.get("n_steps") not in (None, len(steps)):
        raise ValueError(f"meta says {meta['n_steps']} steps, steps.jsonl has {len(steps)}")
    fps, cams = float(meta["fps"]), list(meta["cameras"])
    if fps != int(fps):
        raise ValueError(f"fps {fps} is not an integer (LeRobot needs one)")
    for a, b in zip(steps, steps[1:]):
        if b["i"] != a["i"] + 1 or b["t"] - a["t"] > 1.5 / fps:
            raise ValueError(f"gap between steps {a['i']} and {b['i']} ({b['t'] - a['t']:.3f} s)")
    for s in steps:
        st = s["state"]
        if not st.get("ok"):
            raise ValueError(f"step {s['i']}: no gate state ({st.get('error')})")
        if st.get("poll_age") is None or st["poll_age"] > max_age:
            raise ValueError(f"step {s['i']}: gate state {st.get('poll_age')} s old")
        if not isinstance(st.get("controls"), dict):
            raise ValueError(f"step {s['i']}: no raw controls")
        for c in cams:
            v = s["cams"].get(c) or {}
            if not v.get("file"):
                raise ValueError(f"step {s['i']}: no frame from {c} ({v.get('status')})")
            if v["age"] > max_age:
                raise ValueError(f"step {s['i']}: {c} frame {v['age']:.2f} s old")
            if not (d / v["file"]).exists():
                raise ValueError(f"step {s['i']}: missing {v['file']}")
    return meta, steps


def rig_features(cams, crops, hw):
    """The RigMachine (config + feature description) these episodes are converted for: nothing is opened or contacted."""
    return RigMachine(RigMachineConfig(cameras={c: c for c in cams}, crops=crops, image_hw=hw))


def decode(path):
    frame = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"cannot decode {path}")
    return frame


def read_image(path, crop, hw):
    """Same conversion as RigMachine._read_cam (act.rig_machine.prepare_image): BGR decode -> square crop -> INTER_AREA
    resize to hw if needed -> RGB."""
    return prepare_image(decode(path), crop, hw)


def native_hw(d, steps, cams):
    """{camera: (h, w)} of the episode's first frame per camera."""
    return {c: decode(d / steps[0]["cams"][c]["file"]).shape[:2] for c in cams}


def repeat_counts(steps, cams, fps):
    """{camera: [steps reusing the previous step's frame, of which with age >= 2 periods]}."""
    out = {}
    for c in cams:
        rep = old = 0
        for a, b in zip(steps, steps[1:]):
            if b["cams"][c].get("seq") is not None and b["cams"][c].get("seq") == a["cams"][c].get("seq"):
                rep += 1
                old += b["cams"][c]["age"] >= 2.0 / fps
        out[c] = [rep, old]
    return out


def resolve_crops(args, cams, hw_native):
    """--crop-file then --crop (later wins) -> ({camera: (x, y, size)} for every camera, validated to fit, output hw)."""
    want = {}
    try:
        if args.crop_file:
            want.update(load_crop_file(args.crop_file))
        want.update(dict(parse_crop(c) for c in args.crop))
    except (OSError, ValueError) as ex:
        raise SystemExit(f"--crop / --crop-file: {ex}")
    unknown = sorted(set(want) - set(cams))
    if unknown:
        raise SystemExit(f"crop given for {unknown}, but the cameras are {sorted(cams)}")
    crops = {}
    for c in cams:
        try:
            crops[c] = crop_box(hw_native[c], want.get(c))
        except ValueError as ex:
            raise SystemExit(f"crop for {c}: {ex}")
    sizes = sorted({b[2] for b in crops.values()})
    if args.out_size is None and len(sizes) > 1:
        raise SystemExit(f"the crops have different sizes {sizes}: pass --out-size N (every camera is stored NxN)")
    n = args.out_size or sizes[0]
    return crops, (n, n)


def open_dataset(args, fps, robot):
    root = Path(args.root)
    if args.resume and root.exists():
        ds = LeRobotDataset.resume(args.repo_id, root=root)
        if ds.fps != fps:
            raise SystemExit(f"{root} is {ds.fps} fps, the episodes are {fps} fps")
        want = {k: list(v["shape"]) for k, v in make_features(robot).items()}
        have = {k: list(v["shape"]) for k, v in ds.meta.features.items() if k.startswith(("action", "observation"))}
        if have != want:
            raise SystemExit(f"{root} has different features than these episodes (cameras {sorted(robot.cfg.cameras)})")
        return ds
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"{root} already exists: use --resume to append or --overwrite to start over")
        shutil.rmtree(root)
    return LeRobotDataset.create(repo_id=args.repo_id, fps=fps, features=make_features(robot), root=root,
                                 robot_type=ROBOT_TYPE, use_videos=True, image_writer_threads=4)


def episode_actions(meta, steps, override):
    """-> (map, deadband, actions per step, remapped?) with the episode's recorded map, overridden by --cab-map."""
    cmap, db = cab_map_merge(override, meta["cab_map"], meta["cab_deadband"])
    base = cab_map_merge({}, meta["cab_map"], meta["cab_deadband"])
    return cmap, db, [cab_action(s["state"]["controls"], cmap, db) for s in steps], (cmap, db) != base


def release_cut(labels, name, n_steps, fps, margin):
    """-> number of steps to keep (the labelled release step + margin), or raises ValueError(reason)."""
    lab = labels.get(name)
    if lab is None or lab.get("release_step") is None:
        raise ValueError(f"no release label ({(lab or {}).get('confidence', 'not in the labels file')})")
    if lab.get("n_steps") != n_steps:
        raise ValueError(f"labels are for {lab.get('n_steps')} steps, the episode has {n_steps}")
    return min(n_steps, lab["release_step"] + round(margin * fps))


def main(args):
    override = {}
    if args.cab_map:
        try:
            override = json.loads(Path(args.cab_map).read_text())
            cab_map_merge(override, {ch: ["x", 1.0] for ch in CHANNELS}, 0.0)     # validate once, up front
        except (OSError, ValueError) as ex:
            raise SystemExit(f"--cab-map {args.cab_map}: {ex}")
    if args.out_size is not None and args.out_size < 1:
        raise SystemExit("--out-size must be >= 1")
    try:                                                                        # validate once, up front
        if args.crop_file:
            load_crop_file(args.crop_file)
        for c in args.crop:
            parse_crop(c)
    except (OSError, ValueError) as ex:
        raise SystemExit(f"--crop / --crop-file: {ex}")
    labels, trim = None, None
    if args.end_at_release:
        try:
            labels = json.loads(Path(args.end_at_release).read_text())
        except (OSError, ValueError) as ex:
            raise SystemExit(f"--end-at-release {args.end_at_release}: {ex}")
        if args.release_margin < 0:
            raise SystemExit("--release-margin must be >= 0")
        trim = {"end_at_release": Path(args.end_at_release).name, "release_margin_s": args.release_margin}
    eps, skipped = find_episodes(args.episodes)
    root = Path(args.root)
    done = json.loads((root / SOURCES).read_text()) if args.resume and (root / SOURCES).exists() else {}
    old = [k for k, v in done.items() if not (isinstance(v, dict) and v.get("action_source") == "cab")]
    if old:
        raise SystemExit(f"{root} holds {len(old)} keyboard-teleop episodes (e.g. {old[0]}): cab demos go in a new "
                         "dataset (another --root)")
    other_trim = [k for k, v in done.items() if v.get("trim") != trim]
    if other_trim:
        raise SystemExit(f"{root} was converted with another end-of-episode cut ({done[other_trim[0]].get('trim')}; "
                         f"this run: {trim}): use another --root, or the same --end-at-release / --release-margin")
    good = []
    for d in eps:
        if d.name in done:
            skipped.append((d, f"already in the dataset (episode {done[d.name]['episode']})"))
            continue
        try:
            meta, steps = load_episode(d, args.max_age)
            cmap, db, actions, remapped = episode_actions(meta, steps, override)
            n_raw = len(steps)
            if labels is not None:
                keep_n = release_cut(labels, d.name, n_raw, float(meta["fps"]), args.release_margin)
                steps, actions = steps[:keep_n], actions[:keep_n]
        except (ValueError, KeyError, TypeError, IndexError, json.JSONDecodeError) as ex:
            skipped.append((d, f"rejected: {ex}"))
            continue
        if not remapped:     # the recorded action must be what the recorded map gives: else the formula changed
            bad = sum(any(abs(a[ch] - float(s["action"][ch])) > 1e-6 for ch in CHANNELS) for a, s in zip(actions, steps))
            if bad:
                print(f"warning {d.name}: {bad} steps' recorded action differs from the re-derived one (using the "
                      "re-derived one)")
        good.append((d, meta, steps, cmap, db, actions, remapped, n_raw))
    if good:   # one dataset = one fps + one camera set: keep the episodes that match the first one
        fps0, cams0 = int(good[0][1]["fps"]), sorted(good[0][1]["cameras"])
        keep, nat0 = [], None
        for g in good:
            meta = g[1]
            if int(meta["fps"]) != fps0 or sorted(meta["cameras"]) != cams0:
                skipped.append((g[0], f"rejected: {int(meta['fps'])} fps / cameras {sorted(meta['cameras'])}, but "
                                      f"this run is {fps0} fps / {cams0}"))
                continue
            try:
                nat = native_hw(g[0], g[2], cams0)
            except ValueError as ex:
                skipped.append((g[0], f"rejected: {ex}"))
                continue
            if keep and nat != nat0:
                skipped.append((g[0], f"rejected: native frame sizes {nat}, but this run is {nat0} (same crops for all)"))
                continue
            nat0 = nat0 or nat
            keep.append(g)
        good = keep
    for d, why in skipped:
        print(f"skip {d}: {why}")
    if not good:
        print("nothing to convert")
        return

    crops, hw = resolve_crops(args, cams0, nat0)
    want = {c: list(b) for c, b in crops.items()}
    other = [v.get("crops") for v in done.values() if v.get("crops") != want]
    if other:
        raise SystemExit(f"{root} was converted with other crops ({other[0]}); these are {want}: one dataset = one "
                         "set of crops (use another --root, or the same --crop / --crop-file)")
    for c in cams0:
        x, y, size = crops[c]
        print(f"crop {c}: x={x} y={y} size={size} of {nat0[c][1]}x{nat0[c][0]} -> {hw[1]}x{hw[0]}")
    robot = rig_features(cams0, crops, hw)
    ds = open_dataset(args, fps0, robot)
    features = ds.meta.features
    n_new = 0
    try:
        for d, meta, steps, cmap, db, actions, remapped, n_raw in good:
            prev = {ch: 0.0 for ch in CHANNELS}
            for s, a in zip(steps, actions):
                obs = {f"{ch}.cmd": prev[ch] for ch in CHANNELS}      # state = previous step's action (see top)
                obs.update({c: read_image(d / s["cams"][c]["file"], crops[c], hw) for c in cams0})
                act = {f"{ch}.cmd": a[ch] for ch in CHANNELS}
                ds.add_frame({**build_dataset_frame(features, obs, "observation"),
                              **build_dataset_frame(features, act, "action"), "task": meta["task"]})
                prev = a
            ds.save_episode()
            reps = repeat_counts(steps, cams0, fps0)
            done[d.name] = {"episode": ds.meta.total_episodes - 1, "action_source": "cab", "cab_map": cmap,
                            "cab_deadband": db, "remapped": remapped,
                            "crops": {c: list(b) for c, b in crops.items()}, "image_hw": list(hw),
                            "frames_repeated": reps,
                            "recorded_map_status": meta.get("cab_map_status"),
                            "track_steps": meta.get("track_steps"),
                            "trim": trim, "raw_steps": n_raw, "kept_steps": len(steps)}
            (root / SOURCES).write_text(json.dumps(done, indent=1))
            (root / CROPS).write_text(json.dumps({c: list(b) for c, b in crops.items()}))
            n_new += 1
            moving = sum(any(a.values()) for a in actions)
            print(f"episode {done[d.name]['episode']}: {d.name} | {len(steps)} frames ({len(steps) / fps0:.1f} s, "
                  f"{moving} with cab input)"
                  f"{f' | cut at release: {(n_raw - len(steps)) / fps0:.1f} s dropped' if trim else ''}"
                  f"{' | re-mapped (--cab-map)' if remapped else ''}"
                  f" | repeated frames {', '.join(f'{c} {r[0]} ({r[1]} >= 2 periods old)' for c, r in reps.items())}"
                  f"{' | tracks moved in %d steps' % meta['track_steps'] if meta.get('track_steps') else ''}"
                  f" | '{meta['task']}'")
    finally:
        try:        # after a failure mid-episode the buffer can be half-built: don't let cleanup hide the real error
            if ds.has_pending_frames():
                ds.clear_episode_buffer()
        except (KeyError, TypeError) as ex:
            print(f"note: could not clear the unsaved episode buffer ({type(ex).__name__}: {ex})")
        ds.finalize()
    print(f"dataset: {ds.root} | {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames | {n_new} new")
    if args.push and n_new:
        print(f"pushing to https://huggingface.co/datasets/{args.repo_id} (private) ...")
        ds.push_to_hub(private=True)
        print("pushed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("episodes", nargs="+", help="episode folders, or folders containing them (e.g. data/raw)")
    ap.add_argument("--root", default="data/rig_cab", help="local dataset folder")
    ap.add_argument("--repo-id", default="local/rig_cab", help="<hf_user>/<name>; needed for --push")
    ap.add_argument("--resume", action="store_true", help="append (skipping converted episodes); creates --root if missing")
    ap.add_argument("--overwrite", action="store_true", help="delete an existing dataset at --root first")
    ap.add_argument("--max-age", type=float, default=0.5, help="reject an episode if a frame / gate state is older (s)")
    ap.add_argument("--cab-map", default=None, metavar="PATH.json",
                    help="re-derive actions with this cab map (dashboard --cab-map format; listed entries replace "
                         "each episode's recorded map). Default: the map recorded in each episode's meta.json")
    ap.add_argument("--crop", action="append", default=[], metavar="NAME=x,y,size",
                    help="square crop box for camera NAME in native pixels (repeatable; overrides --crop-file). "
                         "Default: the centered square (80,0,480 on 640x480)")
    ap.add_argument("--crop-file", default=None, metavar="PATH.json",
                    help="JSON {camera: [x, y, size]} (the format of <root>/crops.json)")
    ap.add_argument("--out-size", type=int, default=None, metavar="N",
                    help="resize every square crop to NxN (default: keep the crop size, e.g. 480)")
    ap.add_argument("--end-at-release", default=None, metavar="LABELS.json",
                    help="end each episode at its labelled release (scripts/subtask_labels.py --out) + --release-margin")
    ap.add_argument("--release-margin", type=float, default=0.5, metavar="S",
                    help="seconds kept after the release step with --end-at-release (default 0.5)")
    ap.add_argument("--push", action="store_true", help="push to the HF Hub (private) when done")
    main(ap.parse_args())
