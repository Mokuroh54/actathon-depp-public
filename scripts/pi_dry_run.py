"""Offline dry run of the PI upload: build the excavator platform contract locally, convert a LeRobot dataset's
episodes into PI-native MCAP shards with pi-sdk's own LeRobot converter, and validate every shard against the
contract. Nothing is sent to PI (no API key, no contract / episode is created).

Runs in the pi-upload-lerobot tool env (lerobot 0.4.x; the project env has lerobot 0.6.1, see AGENTS.md):

    ~/.local/share/uv/tools/pi-sdk/bin/python scripts/pi_dry_run.py data/trash-pnp-dev --raw data/raw

--raw is where the rsynced Pi episodes live: each episode's start time is the wall clock of its first step
(the folder names are Pi local time, not UTC). Output in --out:
    contract.json      the platform contract (feed to ensure_platform_contract for a real upload)
    topic_map.json     LeRobot feature -> PI topic (pi-upload-lerobot --topic-map)
    custom_metadata.csv  per-episode id / start / task / success (pi-upload-lerobot --metadata-csv)
    episode-*/         the MCAP shards, each validated against the contract

Uses pi-sdk's private converter (pi_sdk.lerobot.ingest._write_episode_mcaps), the same code pi-upload-lerobot runs
before uploading, so the shards are what PI would receive. Pinned to pi-sdk 0.3.2; private names may change.
"""
import argparse
import csv
import dataclasses
import json
from pathlib import Path

from pi_sdk.data import McapContract, NdArrayContract, TopicContract
from pi_sdk.lerobot import ingest, plan_dataset_upload
from pi_sdk.mcap.v1.validation import validate_mcap_shard

CHANNELS = ("swing", "boom", "arm", "bucket", "thumb")  # act.machine.CHANNELS (act isn't in the tool env)
STATE_TOPIC = "observation/joystick/cmd"  # previous command, (5,) float32
ACTION_TOPIC = "action/joystick/cmd"      # cab joysticks {ch}.cmd in -1..1, (5,) float32


def camera_topic(cam):
    return f"observation/{cam}/rgb/image"


def make_contract(fps: int, cameras) -> McapContract:
    def array(topic, role):
        return TopicContract(topic=topic, role=role, schema_encoding="protobuf", schema_name="pi.data.v1.NdArray",
                             expected_frequency_hz=0, frequency_tolerance=0,
                             ndarray=NdArrayContract(shape=(len(CHANNELS),), dtype="float32"),
                             required_in_snapshots=("system/step",))

    def camera(cam):
        return TopicContract(topic=camera_topic(cam), role="observation", schema_encoding="protobuf",
                             schema_name="pi.data.v1.VideoFrame", expected_frequency_hz=0, frequency_tolerance=0,
                             required_in_snapshots=("system/step",))

    step = TopicContract(topic="system/step", role="system", schema_encoding="protobuf",
                         schema_name="pi.data.v1.SnapshotRef", expected_frequency_hz=fps, frequency_tolerance=0.1)
    return McapContract(topics=(step, array(STATE_TOPIC, "observation"), *[camera(c) for c in cameras],
                                array(ACTION_TOPIC, "action")))


class OfflineClient:
    """Stands in for pi_sdk.data.Client during planning: serves the local contract, has no network methods."""

    def __init__(self, contract):
        self.contract = contract

    def get_platform_contract(self, platform_contract_id):
        return self.contract


def episode_start(raw_root: Path, name: str) -> float:
    hits = sorted(raw_root.rglob(f"{name}/steps.jsonl"))
    if not hits:
        raise SystemExit(f"no raw episode {name} under {raw_root} (needed for its start time)")
    with hits[0].open() as f:
        return json.loads(f.readline())["wall"]


def write_metadata_csv(root: Path, raw_root: Path, version: str, out: Path) -> Path:
    sources = json.loads((root / "pi_sources.json").read_text())
    tasks = {}
    for name in sources:
        meta = next(raw_root.rglob(f"{name}/meta.json"), None)
        tasks[name] = json.loads(meta.read_text())["task"] if meta else ""
    path = out / "custom_metadata.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode_index", "episode_id", "start_timestamp", "task", "success"])
        w.writeheader()
        for name, src in sorted(sources.items(), key=lambda kv: kv[1]["episode"]):
            w.writerow({"episode_index": src["episode"], "episode_id": f"{version}-{name}",
                        "start_timestamp": f"{episode_start(raw_root, name):.6f}", "task": tasks[name],
                        "success": "true"})
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path, help="LeRobot dataset from convert_pi_episodes.py")
    ap.add_argument("--raw", type=Path, required=True, help="rsynced Pi episodes (for start times and tasks)")
    ap.add_argument("--out", type=Path, default=Path("outputs/pi_dry_run"))
    ap.add_argument("--version", default="dryrun", help="episode-id prefix (a new prefix per dataset version)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    info = json.loads((args.root / "meta" / "info.json").read_text())
    fps = info["fps"]
    if fps != int(fps):
        raise SystemExit(f"fps {fps} is not a whole number; the contract's system/step needs one")
    cams = sorted(k.removeprefix("observation.images.") for k in info["features"] if k.startswith("observation.images."))
    contract = make_contract(int(fps), cams)
    topic_map = {"observation.state": STATE_TOPIC, "action": ACTION_TOPIC,
                 **{f"observation.images.{c}": camera_topic(c) for c in cams}}
    (args.out / "contract.json").write_text(json.dumps(dataclasses.asdict(contract), indent=1))
    (args.out / "topic_map.json").write_text(json.dumps(topic_map, indent=1))
    csv_path = write_metadata_csv(args.root, args.raw, args.version, args.out)
    print(f"contract: system/step {int(fps)} Hz, state {STATE_TOPIC}, action {ACTION_TOPIC}, cameras {cams}")

    # the same planning pi-upload-lerobot does (feature / shape / dtype / fps checks against the contract)
    plan = plan_dataset_upload(OfflineClient(contract), repo_id=f"local/{args.root.name}", root=args.root,
                               platform_contract_id="offline", topic_map=topic_map, output_dir=args.out,
                               metadata_csv=csv_path, source_dataset_path=f"excavator/{args.version}",
                               limit=args.limit)
    for ep in plan.episodes:
        r = ep.result
        dataset = ingest._load_lerobot_episode(repo_id=ep.repo_id, root=ep.root, revision=ep.revision,
                                               episode_index=r.episode_index)
        try:
            end = ingest._write_episode_mcaps(
                dataset, output_paths=r.output_paths, recording_start=r.declaration.recording_start_time,
                topic_map=ep.topic_map, features=ep.features, contract=ep.contract, h264_crf=ep.h264_crf,
                encoder_threads=ep.encoder_threads,
                video_sources=ingest._episode_video_paths(ep.root, ep.info, ep.features, ep.episode_row),
                reencode_video=ep.reencode_video, video_spans=ingest._episode_video_spans(ep.features, ep.episode_row))
        finally:
            ingest._close_lerobot_episode(dataset)
        for path in r.output_paths:
            validate_mcap_shard(path, contract)
        mb = sum(p.stat().st_size for p in r.output_paths) / 1e6
        print(f"episode {r.episode_index} {r.declaration.episode_id}: {r.metadata.task!r}, "
              f"{r.declaration.recording_start_time:%Y-%m-%d %H:%M:%S} -> {end:%H:%M:%S} UTC, "
              f"{len(r.output_paths)} shards, {mb:.1f} MB, valid")
    print(f"{len(plan.episodes)} episode(s) converted and validated offline; nothing was sent to PI. Output: {args.out}")


if __name__ == "__main__":
    main()
