"""Upload a LeRobot dataset's episodes to PI WITH subtask spans: per episode, the language_segment spans from
scripts/subtask_labels.py (the act.phase phase as `subtask`: "reach and grip the cone" / "carry the cone to the
target" / "release the cone"; the episode task as `task`) plus control_source spans: one "human" span over the whole
recording for a cab demo; for a DAgger episode (scripts/control_loop.py --record-dagger: each raw step has "control"
= policy / human / idle, docs/dagger.md) one span per contiguous run of the same controller. pi-upload-lerobot can't do this (pi_sdk.lerobot.ingest calls
create_episode without spans), so this script plans + converts with the same pi-sdk ingest code as
scripts/pi_dry_run.py and then registers each episode itself: create_episode(..., spans=...), upload_shard,
finalize_episode_upload (PI's Partner API guide, "Register, upload, and finalize the episode").

Runs in the pi-upload-lerobot tool env (lerobot 0.4.x):

    PY=~/.local/share/uv/tools/pi-sdk/bin/python
    # offline: full MCAP conversion + contract validation + span validation, no network, no API key
    $PY scripts/pi_upload_spans.py data/s23-keep32-nostate --raw data/raw --labels outputs/subtask_labels/s23_keep32.json \\
        --version s23ns --contract-json outputs/pi_s2/contract.json --dry-run
    # real upload (creates PI episodes: needs the user's go-ahead, PI_API_KEY in the env or .env)
    $PY scripts/pi_upload_spans.py data/s23-keep32-nostate --raw data/raw --labels outputs/subtask_labels/s23_keep32.json \\
        --version s23ns --contract-id <platform contract id>
    # or: add the spans to episodes ALREADY uploaded under --version (public Client.replace_annotations; replaces
    # the episode's annotation snapshot, i.e. any spans/attributes it had)
    $PY scripts/pi_upload_spans.py data/s2-keep18 --raw data/raw --labels ... --version s2st --patch-existing

Episode ids, start times, task and success come from pi_dry_run.write_metadata_csv (id = <version>-<raw episode
folder>, start = the wall clock of the raw episode's first step, success = true); source_dataset_path defaults to
excavator/<version>. Only episodes whose labels have confidence high / manual (plus review with --include-review) are
uploaded; the rest are listed and skipped.

Span times: the recording window is [start, start + (n-1)/fps] (the converter stamps step k at start + k/fps and
returns the last step's time as the recording end). A label boundary at step k (the first step of the new phase) is
placed half a step earlier, at start + (k - 0.5)/fps, so every step falls unambiguously on one side; the first span
starts at the recording start and the last ends at the recording end. The dry run checks that each step's subtask
recovered from the spans equals the label's.

DAgger control spans: "policy" -> control_source "policy", "human" -> "human", "idle" (nobody driving) -> "none",
EXCEPT an idle run with human steps on both sides, which becomes "human" (the operator holding still mid-takeover:
splitting the takeover there would cut dagger_data_filter's action chunks, which stop at each human span's end).
Episodes whose raw meta.json has "aborted": true (the runner ended on an error mid-episode) are skipped with a note;
--include-aborted uploads them too.
Spans are contiguous and cover the whole recording window; boundaries are placed half a step early like the subtask
boundaries, and the dry run checks that every step maps back to its own controller.

--windows WINDOWS.json ({"windows": {raw episode folder: [[start_s, end_s, subtask], ...]}}, seconds from the episode
start) uploads TIME SLICES instead of whole episodes: window k of an episode becomes its own episode
<version>-<folder>-grip<k>[-<--window-suffix>] with the frames start_s <= t < end_s (steps ceil(start_s*fps) ..
ceil(end_s*fps)-1), timestamps re-based so the window's first frame is the recording start, cameras re-encoded (the
cut rarely starts on a keyframe), one language_segment (the window's subtask) and one control_source over the window;
task stays the episode's prompt. --labels is optional then (used only to cross-check frame counts / start times).
Upload the same windows twice as distinct episodes with different --window-suffix values:

    $PY scripts/pi_upload_spans.py data/s24-nostate --raw data/raw --windows outputs/grip_windows/windows.json \\
        --version s24ns --window-suffix w1 --source-dataset-path excavator/gripwin-ns-w1 --contract-json ... --dry-run

Uses pi-sdk private names (ingest._load_lerobot_episode, _write_episode_mcaps, _episode_video_paths,
_episode_video_spans, _close_lerobot_episode, _is_video_feature, lerobot._video.read_step,
data._validation.validate_spans), pinned to pi-sdk 0.3.2 like
scripts/pi_dry_run.py; the network calls are all public Client methods.
"""
import argparse
import dataclasses
import datetime
import json
import math
import os
from pathlib import Path

from pi_sdk.data import (APIError, Client, EpisodeShard, McapContract, NdArrayContract, TopicContract, control_source,
                         language_segment)
from pi_sdk.data._validation import validate_spans
from pi_sdk.lerobot import ingest, plan_dataset_upload
from pi_sdk.lerobot._video import read_step
from pi_sdk.mcap.v1.validation import validate_mcap_shard

from pi_dry_run import ACTION_TOPIC, STATE_TOPIC, OfflineClient, camera_topic, make_contract, write_metadata_csv

# act.phase.PHASES (act isn't installed in the pi-upload-lerobot tool env); the runner sends these exact strings
SUBTASKS = ("reach and grip the cone", "carry the cone to the target", "release the cone")


def api_key() -> str:
    """PI_API_KEY from the environment, else from a git-ignored .env (KEY=VALUE lines). Never printed."""
    key = os.environ.get("PI_API_KEY")
    if not key and Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            name, sep, value = line.strip().partition("=")
            if sep and name.strip() == "PI_API_KEY":
                key = value.strip().strip('"').strip("'")
    if not key:
        raise SystemExit("PI_API_KEY is not set (export it, or put PI_API_KEY=... in .env)")
    return key


def load_contract(path):
    """A contract saved by pi_dry_run.py (dataclasses.asdict of McapContract) -> McapContract."""
    topics = []
    for t in json.loads(Path(path).read_text())["topics"]:
        nd = t.get("ndarray")
        topics.append(TopicContract(**{**t, "ndarray": NdArrayContract(shape=tuple(nd["shape"]), dtype=nd["dtype"])
                                       if nd else None,
                                       "required_in_snapshots": tuple(t.get("required_in_snapshots") or ())}))
    return McapContract(topics=tuple(topics))


def load_labels(paths):
    labels = {}
    for p in paths:
        labels.update(json.loads(Path(p).read_text()))
    return labels


CONTROL_VALUE = {"policy": "policy", "human": "human", "idle": "none"}


def raw_meta(raw: Path, name):
    """The raw episode's meta.json ({} if not found)."""
    hit = next(raw.rglob(f"{name}/meta.json"), None)
    return json.loads(hit.read_text()) if hit else {}


def raw_controls(raw: Path, name):
    """Per-step "control" of a raw DAgger episode (control_loop.py --record-dagger), or None for a cab demo."""
    hits = sorted(raw.rglob(f"{name}/steps.jsonl"))
    if not hits:
        return None
    meta_path = hits[0].parent / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    with hits[0].open() as f:
        controls = [json.loads(line).get("control") for line in f if line.strip()]
    if not meta.get("dagger"):
        if any(c is not None for c in controls):
            raise SystemExit(f"{name}: steps carry 'control' but meta.json has no \"dagger\": true")
        return None
    bad = sorted({str(c) for c in controls if c not in CONTROL_VALUE})
    if bad:
        raise SystemExit(f"{name}: unknown control label(s) {bad}")
    return controls


def control_runs(controls):
    """Per-step controls -> [(start_step, end_step, control_source value)], contiguous, covering every step. An idle
    run with human on both sides becomes human; other idle runs are "none"."""
    runs = []
    for k, c in enumerate(controls):
        if runs and runs[-1][2] == c:
            runs[-1][1] = k + 1
        else:
            runs.append([k, k + 1, c])
    for j in range(1, len(runs) - 1):
        if runs[j][2] == "idle" and runs[j - 1][2] == "human" and runs[j + 1][2] == "human":
            runs[j][2] = "human-gap"
    out = []
    for a, b, c in runs:
        v = "human" if c == "human-gap" else CONTROL_VALUE[c]
        if out and out[-1][2] == v:
            out[-1] = (out[-1][0], b, v)
        else:
            out.append((a, b, v))
    return out


def build_spans(rec, *, start, end, fps, task, controls=None):
    """Label record -> (spans, per-step subtask list, per-step control_source list). start/end: the recording window
    (UTC datetimes). controls: a DAgger episode's per-step controls (None: one human span, a cab demo)."""
    n = rec["n_steps"]

    def at(k):
        if k <= 0:
            return start
        if k >= n:
            return end
        return start + datetime.timedelta(seconds=(k - 0.5) / fps)

    if controls is None:
        spans, per_control = [control_source(start=start, end=end, value="human")], ["human"] * n
    else:
        if len(controls) != n:
            raise ValueError(f"{len(controls)} raw steps with a control label, labels have {n} steps")
        spans, per_control = [], [None] * n
        for a, b, v in control_runs(controls):
            spans.append(control_source(start=at(a), end=at(b), value=v))
            per_control[a:b] = [v] * (b - a)
    per_step = [None] * n
    for seg in rec["segments"]:
        a, b, sub = seg["start_step"], seg["end_step"], seg["subtask"]
        if sub not in SUBTASKS:
            raise ValueError(f"unexpected subtask {sub!r}")
        spans.append(language_segment(start=at(a), end=at(b), task=task, subtask=sub))
        for k in range(a, b):
            per_step[k] = sub
    return spans, per_step, per_control


def span_at(spans, t):
    """The span (sorted, contiguous) covering time t; the last span also covers its end."""
    return next((s for s in spans if s.start_time <= t < s.end_time or (s is spans[-1] and t == s.end_time)), None)


def check_spans(spans, per_step, *, start, end, fps, per_control=None):
    """pi-sdk's validator + window / overlap / coverage checks + per-step round trip. Returns a list of problems."""
    validate_spans(spans)  # raises pi_sdk.data.ValidationError
    problems = []
    for s in spans:
        if s.start_time < start or s.end_time > end:
            problems.append(f"span {s.start_time.isoformat()}..{s.end_time.isoformat()} outside the recording window")
    lang = sorted((s for s in spans if hasattr(s.kind, "subtask")), key=lambda s: s.start_time)
    if lang[0].start_time != start or lang[-1].end_time != end:
        problems.append("language segments don't cover the whole recording window")
    for a, b in zip(lang, lang[1:]):
        if a.end_time != b.start_time:
            problems.append(f"language segments not contiguous at {a.end_time.isoformat()}")
    mismatches = 0
    for k, want in enumerate(per_step):
        # the converter stamps steps from LeRobot's float32 timestamps, so the last step sits exactly on `end`
        # (a few us before start + (n-1)/fps); the last span ends there too
        t = min(start + datetime.timedelta(seconds=k / fps), end)
        got = next((s.kind.subtask for s in lang if s.start_time <= t < s.end_time
                    or (s is lang[-1] and t == s.end_time)), None)
        mismatches += got != want
    if mismatches:
        problems.append(f"{mismatches} step(s) map to a different subtask than the label")
    if per_control is not None:
        ctl = sorted((s for s in spans if hasattr(s.kind, "value")), key=lambda s: s.start_time)
        if not ctl or ctl[0].start_time != start or ctl[-1].end_time != end:
            problems.append("control_source spans don't cover the whole recording window")
        for a, b in zip(ctl, ctl[1:]):
            if a.end_time != b.start_time:
                problems.append(f"control_source spans not contiguous at {a.end_time.isoformat()}")
        wrong = 0
        for k, want in enumerate(per_control):
            s = span_at(ctl, min(start + datetime.timedelta(seconds=k / fps), end))
            wrong += s is None or s.kind.value != want
        if wrong:
            problems.append(f"{wrong} step(s) map to a different control_source than recorded")
    return problems


def fmt_controls(per_control, fps):
    counts = {v: per_control.count(v) for v in ("policy", "human", "none") if v in per_control}
    return "control " + ", ".join(f"{v} {n / fps:.1f}s" for v, n in counts.items())


def convert(ep):
    """pi-sdk's LeRobot -> PI MCAP conversion for one planned episode (same calls as pi_dry_run.py);
    returns the recording end the converter derived (the last step's timestamp)."""
    r = ep.result
    dataset = ingest._load_lerobot_episode(repo_id=ep.repo_id, root=ep.root, revision=ep.revision,
                                           episode_index=r.episode_index)
    try:
        return ingest._write_episode_mcaps(
            dataset, output_paths=r.output_paths, recording_start=r.declaration.recording_start_time,
            topic_map=ep.topic_map, features=ep.features, contract=ep.contract, h264_crf=ep.h264_crf,
            encoder_threads=ep.encoder_threads,
            video_sources=ingest._episode_video_paths(ep.root, ep.info, ep.features, ep.episode_row),
            reencode_video=ep.reencode_video, video_spans=ingest._episode_video_spans(ep.features, ep.episode_row))
    finally:
        ingest._close_lerobot_episode(dataset)


class WindowEpisode:
    """Steps [a, b) of a loaded LeRobot episode, as pi-sdk's MCAP writer reads an episode (len, fps, [i]): every
    camera decoded, timestamps re-based to step a's (t0)."""

    def __init__(self, dataset, a, b, video_keys):
        self.dataset, self.a, self.b, self.video_keys, self.fps = dataset, a, b, video_keys, dataset.fps
        self.t0 = float(read_step(dataset, a, [])["timestamp"])

    def __len__(self):
        return self.b - self.a

    def __getitem__(self, i):
        step = read_step(self.dataset, self.a + i, self.video_keys)
        return {**step, "timestamp": float(step["timestamp"]) - self.t0}


def convert_window(ep, a, b, output_paths, episode_start):
    """convert() for steps [a, b) of a planned episode, written to output_paths; no H.264 passthrough (a window
    rarely starts on a keyframe). Returns the window's (recording start, recording end)."""
    dataset = ingest._load_lerobot_episode(repo_id=ep.repo_id, root=ep.root, revision=ep.revision,
                                           episode_index=ep.result.episode_index)
    try:
        videos = [f for f in ep.topic_map if ingest._is_video_feature(ep.features[f])]
        window = WindowEpisode(dataset, a, b, videos)
        start = episode_start + datetime.timedelta(seconds=window.t0)
        end = ingest._write_episode_mcaps(
            window, output_paths=output_paths, recording_start=start, topic_map=ep.topic_map, features=ep.features,
            contract=ep.contract, h264_crf=ep.h264_crf, encoder_threads=ep.encoder_threads, video_sources={},
            reencode_video=True)
        return start, end
    finally:
        ingest._close_lerobot_episode(dataset)


def window_units(r, name, windows, n, fps, suffix):
    """One episode's windows -> [(episode id, one-segment label record, output paths, a, b)]; steps [a, b) are the
    frames with start_s <= k/fps < end_s."""
    units = []
    for k, (t0, t1, sub) in enumerate(windows):
        a, b = math.ceil(t0 * fps - 1e-6), min(math.ceil(t1 * fps - 1e-6), n)
        if not 0 <= a < b:
            raise SystemExit(f"{name}: window {k} [{t0}, {t1}) s is empty or outside the {n / fps:.1f} s episode")
        tag = f"grip{k}" + (f"-{suffix}" if suffix else "")
        rec = {"n_steps": b - a, "segments": [{"start_step": 0, "end_step": b - a, "subtask": sub}]}
        paths = tuple(p.parent.with_name(f"{p.parent.name}-{tag}") / p.name for p in r.output_paths)
        units.append((f"{r.declaration.episode_id}-{tag}", rec, paths, a, b))
    return units


def fmt_segments(rec, fps):
    return " | ".join(f"{s['subtask']} {s['start_step'] / fps:.1f}-{s['end_step'] / fps:.1f}s" for s in rec["segments"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path, help="LeRobot dataset from convert_pi_episodes.py")
    ap.add_argument("--raw", type=Path, required=True, help="rsynced Pi episodes (start times, tasks)")
    ap.add_argument("--labels", nargs="+", default=[],
                    help="subtask_labels.py JSON file(s) (required unless --windows; with it, only cross-checks)")
    ap.add_argument("--windows", type=Path, help="upload these time slices as their own episodes (module docstring)")
    ap.add_argument("--window-suffix", default="", help="--windows: extra episode-id suffix (e.g. w1) for re-uploads")
    ap.add_argument("--version", required=True, help="episode-id prefix (a new prefix per dataset version)")
    ap.add_argument("--out", type=Path, help="shards / metadata dir (default outputs/pi_spans_<version>)")
    ap.add_argument("--dry-run", action="store_true", help="convert + validate offline; no network, no API key")
    ap.add_argument("--patch-existing", action="store_true",
                    help="don't upload: replace_annotations on already-finalized <version>-* episodes")
    ap.add_argument("--contract-id", help="platform contract id (required for a real upload)")
    ap.add_argument("--contract-json", type=Path,
                    help="dry run / patch: validate against this saved contract (e.g. outputs/pi_s2/contract.json) "
                         "instead of the one pi_dry_run.make_contract builds from the dataset")
    ap.add_argument("--include-review", action="store_true", help="also upload episodes labelled 'review'")
    ap.add_argument("--include-aborted", action="store_true",
                    help="also upload DAgger episodes whose raw meta.json says \"aborted\": true (skipped by default)")
    ap.add_argument("--source-dataset-path", help="default excavator/<version>")
    ap.add_argument("--robot-id", default="")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--base-url", default=os.environ.get("PI_DATA_API_URL", "https://api.pi-fleet.com/v1/data"))
    args = ap.parse_args()
    if not args.dry_run and not args.patch_existing and not args.contract_id:
        ap.error("a real upload needs --contract-id (or use --dry-run / --patch-existing)")
    if not args.labels and not args.windows:
        ap.error("--labels is required (unless --windows)")
    out = args.out or Path(f"outputs/pi_spans_{args.version}")
    out.mkdir(parents=True, exist_ok=True)

    info = json.loads((args.root / "meta" / "info.json").read_text())
    fps = info["fps"]
    cams = sorted(k.removeprefix("observation.images.") for k in info["features"] if k.startswith("observation.images."))
    topic_map = {"observation.state": STATE_TOPIC, "action": ACTION_TOPIC,
                 **{f"observation.images.{c}": camera_topic(c) for c in cams}}
    csv_path = write_metadata_csv(args.root, args.raw, args.version, out)
    sources = json.loads((args.root / "pi_sources.json").read_text())
    name_of = {src["episode"]: name for name, src in sources.items()}
    labels = load_labels(args.labels)
    allowed = {"high", "manual"} | ({"review"} if args.include_review else set())
    windows = json.loads(args.windows.read_text())["windows"] if args.windows else None
    for name in sorted(set(windows or {}) - set(sources)):
        print(f"skip windows of {name}: not in the dataset")

    selected, skipped = [], []
    for idx in sorted(name_of):
        rec = labels.get(name_of[idx])
        if windows is not None and not windows.get(name_of[idx]):
            continue  # no window in this episode: not requested
        if raw_meta(args.raw, name_of[idx]).get("aborted") is True and not args.include_aborted:
            skipped.append((idx, "aborted episode (the runner ended on an error; --include-aborted to upload it)"))
        elif windows is not None:
            selected.append(idx)
        elif rec is None:
            skipped.append((idx, "no labels"))
        elif rec["confidence"] not in allowed or not rec["segments"]:
            skipped.append((idx, f"confidence {rec['confidence']}: {'; '.join(rec['flags'] + rec['review'])}"))
        else:
            selected.append(idx)
    selected = selected[:args.limit] if args.limit else selected
    for idx, why in skipped:
        print(f"skip episode {idx} {name_of[idx]}: {why}")
    if not selected:
        raise SystemExit("no episode to upload")

    client = None if args.dry_run else Client(api_key=api_key(), base_url=args.base_url)
    try:
        offline = load_contract(args.contract_json) if args.contract_json else make_contract(int(fps), cams)
        planner = OfflineClient(offline) if args.dry_run or args.patch_existing else client
        plan = plan_dataset_upload(planner, repo_id=f"local/{args.root.name}", root=args.root,
                                   platform_contract_id=args.contract_id or "offline", topic_map=topic_map,
                                   output_dir=out, metadata_csv=csv_path, robot_id=args.robot_id,
                                   source_dataset_path=args.source_dataset_path or f"excavator/{args.version}",
                                   episode_indices=selected)
        report, n_ok, n_units = {}, 0, 0
        for ep in plan.episodes:
            r = ep.result
            name = name_of[r.episode_index]
            rec = labels.get(name)
            n = ingest._episode_frame_count(ep.episode_row)
            if rec is not None and n != rec["n_steps"]:
                raise SystemExit(f"{name}: dataset has {n} frames, labels {rec['n_steps']} steps (re-label)")
            if rec is not None and abs(r.declaration.recording_start_time.timestamp() - rec["start_wall"]) > 1e-3:
                raise SystemExit(f"{name}: recording start {r.declaration.recording_start_time.timestamp()} "
                                 f"!= label start {rec['start_wall']}")
            controls = raw_controls(args.raw, name)       # None: a cab demo (one human span)
            if windows is None:
                units = [(r.declaration.episode_id, rec, r.output_paths, 0, n)]
            else:
                if controls is not None and len(controls) != n:
                    raise SystemExit(f"{name}: {len(controls)} raw steps with a control label, dataset has {n}")
                units = window_units(r, name, windows[name], n, fps, args.window_suffix)
            n_units += len(units)
            for eid, rec, paths, a, b in units:
                start = r.declaration.recording_start_time
                ctl = controls if windows is None or controls is None else controls[a:b]

                if args.patch_existing and not args.dry_run:
                    try:
                        existing = client.get_episode(eid)
                    except APIError as exc:
                        print(f"episode {r.episode_index} {eid}: not on PI ({exc.code}), skipped")
                        continue
                    if existing.upload_completed_at is None or existing.episode is None:
                        print(f"episode {r.episode_index} {eid}: not finalized, skipped")
                        continue
                    start = existing.episode.declaration.recording_start_time
                    end = existing.episode.declaration.recording_end_time
                    spans, per_step, per_control = build_spans(rec, start=start, end=end, fps=fps,
                                                               task=existing.episode.metadata.task, controls=ctl)
                    problems = check_spans(spans, per_step, start=start, end=end, fps=fps, per_control=per_control)
                    if problems:
                        raise SystemExit(f"{eid}: {problems}")
                    client.replace_annotations(eid, spans=spans)
                    n_ok += 1
                    print(f"episode {r.episode_index} {eid}: spans replaced ({fmt_segments(rec, fps)}; "
                          f"{fmt_controls(per_control, fps)})")
                    continue

                if args.patch_existing:  # dry run of a patch: the window is what the converter would derive
                    start += datetime.timedelta(seconds=a / fps)
                    end = start + datetime.timedelta(seconds=(rec["n_steps"] - 1) / fps)
                    shards_mb = 0.0
                else:
                    start, end = convert_window(ep, a, b, paths, start) if windows is not None else (start, convert(ep))
                    for path in paths:
                        validate_mcap_shard(path, ep.contract)
                    shards_mb = sum(p.stat().st_size for p in paths) / 1e6
                spans, per_step, per_control = build_spans(rec, start=start, end=end, fps=fps, task=r.metadata.task,
                                                           controls=ctl)
                problems = check_spans(spans, per_step, start=start, end=end, fps=fps, per_control=per_control)
                if problems:
                    raise SystemExit(f"{eid}: {problems}")
                report[eid] = {"episode_index": r.episode_index, "raw": name, "task": r.metadata.task,
                               "recording": [start.isoformat(), end.isoformat()], "confidence": rec.get("confidence"),
                               **({"window_steps": [a, b]} if windows is not None else {}),
                               "spans": [{"start": s.start_time.isoformat(), "end": s.end_time.isoformat(),
                                          **{k: v for k, v in dataclasses.asdict(s.kind).items() if v is not None}}
                                         for s in spans]}
                if args.dry_run:
                    n_ok += 1
                    shards = "no conversion (patch)" if args.patch_existing else (
                        f"{len(paths)} shards {shards_mb:.1f} MB valid")
                    print(f"episode {r.episode_index} {eid}: {shards}, {len(spans)} spans valid "
                          f"({fmt_segments(rec, fps)}; {fmt_controls(per_control, fps)})")
                    continue

                try:
                    existing = client.get_episode(eid)
                except APIError as exc:
                    if exc.code != "not_found":
                        raise
                    existing = None
                if existing is not None and existing.upload_completed_at is not None:
                    print(f"episode {r.episode_index} {eid}: already finalized on PI, skipped "
                          "(use --patch-existing to replace its spans)")
                    continue
                declaration = dataclasses.replace(r.declaration, episode_id=eid, recording_start_time=start,
                                                  recording_end_time=end)
                try:
                    client.create_episode(declaration, r.metadata, platform_contract_id=args.contract_id, spans=spans)
                except APIError as exc:
                    if exc.code == "failed_precondition":
                        print(f"episode {r.episode_index} {eid}: create refused ({exc.code}), skipped")
                        continue
                    raise
                for path in paths:
                    client.upload_shard(eid, EpisodeShard.create(path, episode_id=eid, recording_start_time=start,
                                                                 recording_end_time=end))
                client.finalize_episode_upload(eid)
                if client.get_episode(eid).upload_completed_at is None:
                    raise SystemExit(f"episode {eid} did not finalize")
                n_ok += 1
                print(f"episode {r.episode_index} {eid}: uploaded with {len(spans)} spans "
                      f"({fmt_segments(rec, fps)}; {fmt_controls(per_control, fps)})")
    finally:
        if client is not None:
            client.close()
    (out / "spans.json").write_text(json.dumps(report, indent=1))
    verb = "converted and validated offline (nothing sent to PI)" if args.dry_run else (
        "patched" if args.patch_existing else "uploaded")
    print(f"{n_ok}/{n_units} episode(s) {verb}; {len(skipped)} skipped. Spans: {out / 'spans.json'}")


if __name__ == "__main__":
    main()
