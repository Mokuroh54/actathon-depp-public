"""Derive per-episode subtask phases from the recorded cab joystick commands of raw Pi episodes, and write a
spot-check sheet per episode. Three phases (act.phase, exact strings):

    "reach and grip the cone" -> "carry the cone to the target" -> "release the cone"

The rule is the CAUSAL state machine act.phase.PhaseTracker, the same object scripts/control_loop.py --subtasks runs
over its own sent commands at inference, so training labels and runtime phases agree (docs/subtask_labels.md).

    uv run python scripts/subtask_labels.py data/raw/2026-09-26_session2 --keep 163714,163805,...
    uv run python scripts/subtask_labels.py --dataset data/s23-keep32 --raw data/raw --overrides <file>

Signals (step "action", -1..1, rig signs): thumb + = clamp in, - = open; bucket + = dump, - = curl; swing.
  reach -> carry     grip complete: the first step where the thumb stops clamping (thumb <= --thr) after a clamp of
                     at least --min-clamp s of accumulated thumb > --thr (pieces separated by gaps <= --clamp-gap
                     count as one clamp)
  in carry           "carried" once the swing has accumulated --min-carry s of |swing| > --swing-thr in the task's
                     carry direction (blue->red: swing -, red->blue: swing +)
  carry -> release   at the first step after an OPEN once carried: a thumb-open run (thumb < -thr) of >= --min-open s,
                     or a bucket dump run (bucket > --dump-thr) of >= --min-dump s; with --combined-dump always
                     (default) the dump counts even while the thumb clamps. An open BEFORE the carry is a regrip
                     (back to reach; the next full clamp completes a new grip).
  release is latched until the end of the episode.

Labels: [start, grip_complete) reach, [grip_complete, release) carry, [release, end) release, where grip_complete /
release are the step AFTER the one whose command fired the rule (the runner's request at a tick is built before that
tick's command, so it sees the new phase from the next tick on).
Flags (confidence "none", no labels written): no grip, no release (never carried, or no open after carrying).
Review (confidence "review", labels written, check the sheet): regrip (an earlier grip was opened again before the
carry), release by a combined thumb+/bucket+ dump, release more than --max-carry-to-release s after the carry, grip
complete more than --max-grip-to-carry s before the carry.

--overrides FILE: {"<episode folder>": {"grip_complete_s": 12.1, "release_s": 25.9, "why": "..."}} (seconds from the
episode start; an entry without times confirms the rule's labels) -> confidence "manual".

Output: --out JSON {episode_folder: {grip_complete_wall, release_wall, segments [{start, end, start_step, end_step,
subtask}], confidence (high / review / manual / none), flags, review, notes, events, params, ...}} and
--sheets DIR/<episode>.jpg (world + wrist rows; columns at grip complete -1 s / 0 / +1 s and release -1 s / 0 / +1 s).
A summary table is printed.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from act.phase import CARRY, REACH, RELEASE, PhaseTracker, carry_sign

SHEET_CAMS = ("world", "wrist")
TILE = (320, 240)


def find_episodes(paths):
    eps = []
    for p in map(Path, paths):
        if (p / "steps.jsonl").exists():
            eps.append(p)
        elif p.is_dir():
            eps += sorted(d for d in p.iterdir() if (d / "steps.jsonl").exists() and not d.name.endswith(".partial"))
    return eps


def dataset_episodes(root: Path, raw: Path):
    sources = json.loads((root / "pi_sources.json").read_text())
    eps = []
    for name in sorted(sources, key=lambda n: sources[n]["episode"]):
        hits = sorted(raw.rglob(f"{name}/steps.jsonl"))
        if not hits:
            raise SystemExit(f"no raw episode {name} under {raw}")
        eps.append(hits[0].parent)
    return eps


def load(d: Path):
    meta = json.loads((d / "meta.json").read_text())
    with (d / "steps.jsonl").open() as f:
        steps = [json.loads(line) for line in f if line.strip()]
    return meta, steps


def longest_swing(actions, sign, thr, fps):
    """Hindsight: the longest swing run in the carry direction (gaps <= 0.5 s joined) -> (start, end) steps."""
    gap, best, s, last = round(0.5 * fps), None, None, None
    for i, a in enumerate(actions):
        if (a["swing"] * sign if sign else abs(a["swing"])) > thr:
            if s is not None and i - last - 1 > gap:
                best = max(best or (s, last), (s, last), key=lambda r: r[1] - r[0])
                s = None
            s = i if s is None else s
            last = i
    if s is not None:
        best = max(best or (s, last), (s, last), key=lambda r: r[1] - r[0])
    return best


def run_rule(actions, fps, sign, args, combined):
    """Run PhaseTracker over the commands -> (events, step the carry registered, release kind or None).
    An event fired by step i's command is recorded at step i + 1: the first step whose observation is paired with the
    new phase, exactly as in the runner, where tick i's request is built before tick i's command is known."""
    tracker = PhaseTracker(fps, sign, thr=args.thr, min_clamp=args.min_clamp, clamp_gap=args.clamp_gap,
                           min_open=args.min_open, dump_thr=args.dump_thr, min_dump=args.min_dump,
                           swing_thr=args.swing_thr, min_carry=args.min_carry, combined_dump=combined)
    events, carried_at, release_kind = [], None, None
    for i, a in enumerate(actions):
        was_carried = tracker.carried
        ev = tracker.step(a)
        if tracker.carried and not was_carried and carried_at is None:
            carried_at = i
        if ev:
            events.append({"event": ev, "step": i + 1, "t": round((i + 1) / fps, 3)})
            if ev == "release":
                release_kind = tracker.last_kind
    return events, carried_at, release_kind


def label_episode(d: Path, args, overrides):
    meta, steps = load(d)
    fps = float(meta.get("fps") or 15.0)
    task = meta.get("task", "")
    sign = carry_sign(task)
    actions = [s["action"] for s in steps]
    n = len(steps)
    wall = [s["wall"] for s in steps]
    events, carried_at, release_kind = run_rule(actions, fps, sign, args, args.combined_dump == "always")
    fallback = False
    if args.combined_dump == "fallback" and release_kind is None and carried_at is not None:
        ev2, ca2, kind2 = run_rule(actions, fps, sign, args, True)
        if kind2 == "combined_dump":
            events, carried_at, release_kind, fallback = ev2, ca2, kind2, True
    grips = [e for e in events if e["event"] == "grip"]
    release = next((e for e in events if e["event"] == "release"), None)
    grip = grips[-1] if grips else None
    swing = longest_swing(actions, sign, args.swing_thr, fps)
    flags, review, notes = [], [], []
    if sign is None:
        notes.append(f"unknown task {task!r}: carry accepted in either swing direction")
    if not grips:
        flags.append("no_grip")
    elif release is None:
        flags.append(f"no_release (last grip complete at {grip['t']:.1f}s, "
                     + (f"carried from {carried_at / fps:.1f}s, no open after)" if carried_at is not None
                        else "never carried while holding)"))
    grip_step = grip["step"] if grip and release else None
    release_step = release["step"] if release else None
    if grip_step is not None:
        regrips = sum(1 for e in events if e["event"] == "regrip_open" and e["step"] < grip_step)
        if regrips:
            review.append(f"regrip ({regrips} grip(s) opened again before the carry)")
        if carried_at is not None and (carried_at - grip_step) / fps > args.max_grip_to_carry:
            review.append(f"grip complete {(carried_at - grip_step) / fps:.1f}s before the carry")
        if carried_at is not None and (release_step - carried_at) / fps > args.max_carry_to_release:
            review.append(f"release {(release_step - carried_at) / fps:.1f}s after the carry")
    if release_kind == "combined_dump":
        review.append("released by a combined thumb+/bucket+ dump (sometimes only a rotation of the held cone: "
                      "check the sheet)" + (" [fallback]" if fallback else ""))
    if swing and carried_at is not None and not (swing[0] - fps <= carried_at <= swing[1] + fps):
        notes.append("the carry the rule used is not the longest swing in the carry direction")
    confidence = "none" if flags else ("review" if review else "high")
    ov = overrides.get(d.name)
    if ov:
        grip_step = round(ov["grip_complete_s"] * fps) if "grip_complete_s" in ov else grip_step
        release_step = round(ov["release_s"] * fps) if "release_s" in ov else release_step
        confidence, notes = "manual", notes + [f"overridden: {ov}"]
        if grip_step is None or release_step is None:
            confidence = "none"
    rec = {
        "task": task, "fps": fps, "n_steps": n, "start_wall": wall[0],
        "end_wall": wall[-1] + 1.0 / fps,
        "grip_complete_step": grip_step, "release_step": release_step,
        "grip_complete_wall": wall[grip_step] if grip_step is not None else None,
        "release_wall": wall[release_step] if release_step is not None else None,
        "release_kind": release_kind if ov is None else "manual",
        "carried_step": carried_at,
        "longest_carry_swing_s": [round(swing[0] / fps, 2), round(swing[1] / fps, 2)] if swing else None,
        "segments": [], "confidence": confidence, "flags": flags, "review": review, "notes": notes,
        "events": events,
    }
    if confidence != "none":
        if not 0 < grip_step < release_step < n:
            rec["confidence"], rec["flags"] = "none", flags + [f"bad order: grip {grip_step} release {release_step}"]
        else:
            bounds = [(0, grip_step, REACH), (grip_step, release_step, CARRY), (release_step, n, RELEASE)]
            rec["segments"] = [{"start": wall[a], "end": wall[b] if b < n else rec["end_wall"], "start_step": a,
                                "end_step": b, "subtask": sub} for a, b, sub in bounds]
    return d, steps, rec


def tile(d: Path, steps, cam, i, label, fps):
    i = max(0, min(i, len(steps) - 1))
    cams = steps[i].get("cams") or {}
    img = None
    if cam in cams and cams[cam].get("file"):
        img = cv2.imread(str(d / cams[cam]["file"]))
    img = np.zeros((TILE[1], TILE[0], 3), np.uint8) if img is None else cv2.resize(img, TILE)
    a = steps[i]["action"]
    cv2.rectangle(img, (0, 0), (TILE[0], 34), (0, 0, 0), -1)
    cv2.putText(img, f"{label} {cam} t={i / fps:.1f}s", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(img, f"th{a['thumb']:+.1f} bk{a['bucket']:+.1f} bm{a['boom']:+.1f} sw{a['swing']:+.1f}", (4, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return img


def write_sheet(d: Path, steps, rec, out: Path):
    fps = rec["fps"]
    cols = []
    for name, key in (("GRIP", "grip_complete_step"), ("RELEASE", "release_step")):
        s = rec[key]
        if s is None:
            continue
        for off in (-1, 0, 1):
            cols.append((s + round(off * fps), f"{name}{off:+d}s" if off else name))
    if rec["confidence"] == "none":  # flagged: the rule's last grip (if any), then the rest of the episode
        n = len(steps)
        grips = [e["step"] for e in rec["events"] if e["event"] == "grip"]
        cols = [(grips[-1] + round(off * fps), "LASTGRIP" + (f"{off:+d}s" if off else "")) for off in (-1, 0, 1)
                ] if grips else []
        start = rec["carried_step"] if rec["carried_step"] is not None else (grips[-1] if grips else 0)
        cols += [(start + round(k * (n - 1 - start) / 4), "CARRY" if k == 0 and rec["carried_step"] else "t")
                 for k in range(5)]
    rows = [np.hstack([tile(d, steps, cam, i, lab, fps) for i, lab in cols]) for cam in SHEET_CAMS]
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="*", help="raw episode folders or folders containing them")
    ap.add_argument("--dataset", type=Path, help="label exactly this LeRobot dataset's episodes (its pi_sources.json)")
    ap.add_argument("--raw", type=Path, default=Path("data/raw"), help="with --dataset: where the raw episodes are")
    ap.add_argument("--keep", default="", help="comma-separated substrings (e.g. HHMMSS) an episode name must match")
    ap.add_argument("--out", type=Path, help="labels JSON (default outputs/subtask_labels/<dataset or dir>.json)")
    ap.add_argument("--sheets", type=Path, default=Path("outputs/subtask_check"))
    ap.add_argument("--no-sheets", action="store_true")
    ap.add_argument("--overrides", type=Path, help="JSON {episode: {grip_complete_s, release_s}} from spot checks")
    ap.add_argument("--thr", type=float, default=0.3, help="thumb clamp / open command threshold")
    ap.add_argument("--min-clamp", type=float, default=1.2, help="s of accumulated thumb clamp for a full grip")
    ap.add_argument("--clamp-gap", type=float, default=0.5, help="s: clamp pieces closer than this are one clamp")
    ap.add_argument("--min-open", type=float, default=0.2, help="s of thumb open that releases / regrips")
    ap.add_argument("--dump-thr", type=float, default=0.7, help="bucket dump threshold (release after carry)")
    ap.add_argument("--min-dump", type=float, default=0.4, help="s of clean bucket dump that releases after carry")
    ap.add_argument("--combined-dump", choices=("never", "fallback", "always"), default="always",
                    help="a bucket dump while the thumb clamps: never a release; a release only when an episode has "
                         "no other release after the carry (labels 'review'); always a release (causal, = runner)")
    ap.add_argument("--swing-thr", type=float, default=0.3)
    ap.add_argument("--min-carry", type=float, default=1.0, help="s of carry-direction swing while holding")
    ap.add_argument("--max-grip-to-carry", type=float, default=20.0, help="review if the carry starts later")
    ap.add_argument("--max-carry-to-release", type=float, default=20.0, help="review if the release is later")
    args = ap.parse_args()

    if args.dataset:
        eps = dataset_episodes(args.dataset, args.raw)
        default_name = args.dataset.name
    else:
        if not args.paths:
            ap.error("give raw episode paths or --dataset")
        eps = find_episodes(args.paths)
        default_name = Path(args.paths[0]).name
    keep = [k for k in args.keep.split(",") if k]
    if keep:
        eps = [d for d in eps if any(k in d.name for k in keep)]
        missing = [k for k in keep if not any(k in d.name for d in eps)]
        if missing:
            raise SystemExit(f"--keep entries with no episode: {missing}")
    if not eps:
        raise SystemExit("no episodes found")
    overrides = json.loads(args.overrides.read_text()) if args.overrides else {}
    out = args.out or Path("outputs/subtask_labels") / f"{default_name}.json"

    labels = {}
    print(f"{'episode':<22} {'dir':<5} {'len':>5} {'grip':>6} {'carry':>6} {'release':>8} {'kind':<11} {'conf':<7} "
          "flags / review")
    for d in eps:
        d, steps, rec = label_episode(d, args, overrides)
        labels[d.name] = rec
        if not args.no_sheets:
            write_sheet(d, steps, rec, args.sheets / f"{d.name}.jpg")
        fps = rec["fps"]
        fmt = lambda s: f"{s / fps:.1f}" if s is not None else "-"
        direction = {-1: "b->r", 1: "r->b"}.get(carry_sign(rec["task"]), "?")
        print(f"{d.name[:22]:<22} {direction:<5} {rec['n_steps'] / fps:>5.1f} {fmt(rec['grip_complete_step']):>6} "
              f"{fmt(rec['carried_step']):>6} {fmt(rec['release_step']):>8} {str(rec['release_kind']):<11} "
              f"{rec['confidence']:<7} {'; '.join(rec['flags'] + rec['review'])}")
    out.parent.mkdir(parents=True, exist_ok=True)
    params = {k: v for k, v in vars(args).items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    params["combined_dump"] = args.combined_dump
    for rec in labels.values():
        rec["params"] = params
    out.write_text(json.dumps(labels, indent=1))
    counts = {c: sum(r["confidence"] == c for r in labels.values()) for c in ("high", "review", "manual", "none")}
    print(f"{len(labels)} episodes: {counts['high']} high, {counts['review']} review, {counts['manual']} manual, "
          f"{counts['none']} flagged (no labels). Labels: {out}"
          + ("" if args.no_sheets else f"; spot-check sheets: {args.sheets}/<episode>.jpg"))


if __name__ == "__main__":
    main()
