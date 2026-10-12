"""Fine-tune a PI model on a LeRobot excavator dataset through the PI Partner APIs.

Runs in its own venv because pi-sdk[lerobot] pins lerobot 0.4.x (the project uses 0.6.x):

    uv venv .venv-pi --python 3.12 && export GOOGLE_APPLICATION_CREDENTIALS=$PWD/<pi-registry-key>.json
    uv pip install --python .venv-pi/bin/python keyring keyrings.google-artifactregistry-auth
    PATH=$PWD/.venv-pi/bin:$PATH uv pip install --python .venv-pi/bin/python --keyring-provider subprocess \\
        "pi-sdk[video,inference,lerobot]>=0.3.2" \\
        --extra-index-url <PI package registry URL from PI onboarding>

Then, with PI_API_KEY set (IDs are saved to outputs/pi/state.json between steps):

    .venv-pi/bin/python scripts/pi_finetune.py contract
    .venv-pi/bin/python scripts/pi_finetune.py upload --task "pick up the trash and put it in the bin"
    .venv-pi/bin/python scripts/pi_finetune.py profile
    .venv-pi/bin/python scripts/pi_finetune.py finetune --base-model pi07_cosmic --num-steps 1000
    .venv-pi/bin/python scripts/pi_finetune.py status --wait
    .venv-pi/bin/python scripts/pi_finetune.py all --task "..."    # everything above in order
"""
import argparse
import csv
import datetime
import json
import os
import re
import time
from pathlib import Path

import pi_sdk
from huggingface_hub import snapshot_download
from pi_sdk.data import McapContract, NdArrayContract, TopicContract
from pi_sdk.finetuning import (
    ArraySlice,
    CheckpointPolicy,
    DatasetFilterInput,
    EpisodeFilter,
    TrainingProfileConfig,
)
from pi_sdk.lerobot.ingest import upload_dataset

FPS = 15
PLATFORM = "excavator_v1"
STATE_TOPIC = "observation/excavator/cmd"
ACTION_TOPIC = "action/excavator/cmd"
CAMERAS = {"front": "observation/front_camera/rgb/image", "wrist": "observation/wrist_camera/rgb/image"}
TOPIC_MAP = {
    "observation.state": STATE_TOPIC,
    "action": ACTION_TOPIC,
    **{f"observation.images.{name}": topic for name, topic in CAMERAS.items()},
}
LOCAL_TZ = datetime.timezone(datetime.timedelta(hours=-7))

OUT = Path("outputs/pi")
STATE_FILE = OUT / "state.json"


def load_state() -> dict:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}


def save_state(**updates) -> dict:
    state = {**load_state(), **updates}
    OUT.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")
    return state


def require(state: dict, key: str, step: str) -> str:
    if key not in state:
        raise SystemExit(f"{key} missing from {STATE_FILE}; run `{step}` first")
    return state[key]


def slug(repo_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", repo_id.split("/")[-1].lower()).strip("-")


def contract_topics() -> tuple[TopicContract, ...]:
    def array_topic(topic: str, role: str) -> TopicContract:
        return TopicContract(
            topic=topic,
            role=role,
            schema_encoding="protobuf",
            schema_name="pi.data.v1.NdArray",
            expected_frequency_hz=0,
            frequency_tolerance=0,
            ndarray=NdArrayContract(shape=(5,), dtype="float32"),
            required_in_snapshots=("system/step",),
        )

    def camera_topic(topic: str) -> TopicContract:
        return TopicContract(
            topic=topic,
            role="observation",
            schema_encoding="protobuf",
            schema_name="pi.data.v1.VideoFrame",
            expected_frequency_hz=0,
            frequency_tolerance=0,
            required_in_snapshots=("system/step",),
        )

    step = TopicContract(
        topic="system/step",
        role="system",
        schema_encoding="protobuf",
        schema_name="pi.data.v1.SnapshotRef",
        expected_frequency_hz=FPS,
        frequency_tolerance=0.1,
    )
    return (
        step,
        array_topic(STATE_TOPIC, "observation"),
        array_topic(ACTION_TOPIC, "action"),
        *(camera_topic(topic) for topic in CAMERAS.values()),
    )


def cmd_contract(args, data: pi_sdk.data.Client, ft: pi_sdk.finetuning.Client) -> None:
    result = data.ensure_platform_contract(
        PLATFORM,
        McapContract(topics=contract_topics()),
        description=f"Mini excavator: 5 joystick channels (swing, boom, arm, bucket, thumb) in -1..1, "
        f"front + wrist cameras, {FPS} Hz",
    )
    contract_id = result.platform_contract.platform_contract_id
    save_state(platform_contract_id=contract_id)
    print(f"platform contract {contract_id} (newly_created={result.newly_created})")


def write_metadata_csv(root: Path, *, repo_id: str, task: str | None) -> Path:
    """custom_metadata.csv for pi-upload-lerobot: stable episode IDs and absolute start times.

    Start times come from pi_sources.json keys (YYYYMMDD_HHMMSS_<name>, local time at the event).
    """
    sources = json.loads((root / "pi_sources.json").read_text())
    rows = []
    for key, source in sources.items():
        start = datetime.datetime.strptime(key[:15], "%Y%m%d_%H%M%S").replace(tzinfo=LOCAL_TZ)
        row = {
            "episode_index": source["episode"],
            "episode_id": f"{slug(repo_id)}-{key[:15].replace('_', '-')}",
            "start_timestamp": start.isoformat(),
            "success": "true",
        }
        if task:
            row["task"] = task
        rows.append(row)
    path = OUT / f"{slug(repo_id)}_metadata.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["episode_index"]))
    return path


def cmd_upload(args, data: pi_sdk.data.Client, ft: pi_sdk.finetuning.Client) -> None:
    state = load_state()
    contract_id = require(state, "platform_contract_id", "contract")
    root = Path(snapshot_download(args.repo_id, repo_type="dataset", local_dir=Path("data") / slug(args.repo_id)))
    metadata_csv = write_metadata_csv(root, repo_id=args.repo_id, task=args.task)
    source_path = f"{slug(args.repo_id)}/v1"
    results = upload_dataset(
        data,
        repo_id=args.repo_id,
        root=root,
        platform_contract_id=contract_id,
        topic_map=TOPIC_MAP,
        metadata_csv=metadata_csv,
        robot_id="excavator",
        output_dir=OUT / "shards" / slug(args.repo_id),
        source_dataset_path=source_path,
        dry_run=args.dry_run,
        progress=True,
    )
    for r in results:
        print(f"episode {r.episode_index} -> {r.declaration.episode_id} uploaded={r.uploaded} task={r.metadata.task!r}")
    if args.dry_run:
        return
    for r in results:
        if data.get_episode(r.declaration.episode_id).upload_completed_at is None:
            raise SystemExit(f"episode {r.declaration.episode_id} did not finalize")
    save_state(source_dataset_path=source_path, episode_ids=[r.declaration.episode_id for r in results])


def cmd_profile(args, data: pi_sdk.data.Client, ft: pi_sdk.finetuning.Client) -> None:
    state = load_state()
    contract_id = require(state, "platform_contract_id", "contract")
    source_path = require(state, "source_dataset_path", "upload")
    profile = ft.create_training_profile(
        f"{PLATFORM.replace('_', '-')}-training",
        contract_id,
        TrainingProfileConfig(
            base_camera_topics=(CAMERAS["front"],),
            wrist_camera_topics=(CAMERAS["wrist"],),
            state_slices=(ArraySlice(STATE_TOPIC),),
            action_topics=(ACTION_TOPIC,),
            action_space="absolute",
            # joystick commands are rates: a held stick repeats the same action while the arm moves
            keep_idle_actions=True,
        ),
    )
    dataset_filter = ft.create_dataset_filter(
        f"{source_path.replace('/', '-')}-successes",
        EpisodeFilter(platform_contract_id=contract_id, source_dataset_path=source_path, success=True),
    )
    save_state(
        training_profile_id=profile.training_profile_id, dataset_filter_id=dataset_filter.dataset_filter_id
    )
    print(f"training profile {profile.training_profile_id}, dataset filter {dataset_filter.dataset_filter_id}")


def cmd_finetune(args, data: pi_sdk.data.Client, ft: pi_sdk.finetuning.Client) -> None:
    state = load_state()
    name = args.name or f"{require(state, 'source_dataset_path', 'upload').split('/')[0]}-{args.base_model.replace('_', '-')}-{args.num_steps}"
    finetune = ft.create_finetune(
        name,
        [DatasetFilterInput(require(state, "dataset_filter_id", "profile"))],
        base_model=args.base_model,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        checkpoint_policy=CheckpointPolicy("final_only"),
        training_profile_id=require(state, "training_profile_id", "profile"),
    )
    save_state(run_id=finetune.run_id, finetune_name=name)
    print(f"fine-tune {name}: run_id={finetune.run_id} state={finetune.state}")


def cmd_status(args, data: pi_sdk.data.Client, ft: pi_sdk.finetuning.Client) -> None:
    run_id = require(load_state(), "run_id", "finetune")
    while True:
        finetune = ft.get_finetune(run_id)
        progress = f" {finetune.progress:.0%}" if finetune.progress is not None else ""
        print(f"{datetime.datetime.now():%H:%M:%S} {finetune.state}{progress}", flush=True)
        if not args.wait or finetune.state not in {"pending", "running"}:
            break
        time.sleep(60)
    if finetune.state in {"failed", "cancelled"}:
        raise SystemExit(finetune.terminal_reason or finetune.state)
    for metrics in ft.list_finetune_evaluations(run_id):
        print(f"  step {metrics.step}: train_mae={metrics.train_mae} validation_mae={metrics.validation_mae}")
    for checkpoint in ft.list_checkpoints(run_id).checkpoints:
        print(f"  checkpoint {checkpoint.checkpoint_id}: {checkpoint.state} {checkpoint.gateway_url or ''}")
        if checkpoint.state == "available" and checkpoint.gateway_url:
            save_state(checkpoint_id=checkpoint.checkpoint_id, gateway_url=checkpoint.gateway_url)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["contract", "upload", "profile", "finetune", "status", "all"])
    parser.add_argument("--repo-id", default="Actathon-Depp/trash-pnp-dev")
    parser.add_argument("--task", help="overall task instruction (overrides the dataset's task string)")
    parser.add_argument("--dry-run", action="store_true", help="upload: validate against the contract only")
    parser.add_argument("--base-model", default="pi07_cosmic")
    parser.add_argument("--num-steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, choices=[256, 512, 1024, 2048, 4096])
    parser.add_argument("--name", help="fine-tune name (lowercase kebab-case, unique in the org)")
    parser.add_argument("--wait", action="store_true", help="status: poll until the fine-tune finishes")
    args = parser.parse_args()

    api_key = os.environ.get("PI_API_KEY") or raise_missing_key()
    data = pi_sdk.data.Client(api_key=api_key)
    ft = pi_sdk.finetuning.Client(api_key=api_key)
    steps = {"contract": cmd_contract, "upload": cmd_upload, "profile": cmd_profile,
             "finetune": cmd_finetune, "status": cmd_status}
    for step in (list(steps) if args.step == "all" else [args.step]):
        steps[step](args, data, ft)


def raise_missing_key() -> str:
    raise SystemExit("set PI_API_KEY (Partner Portal key from partner.pi-fleet.com)")


if __name__ == "__main__":
    main()
