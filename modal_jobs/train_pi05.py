"""pi0.5 fine-tune on Modal (no SSH, pay per second).

One-time setup on the Mac:
    uv add modal
    uv run modal setup                                            # browser login
    uv run modal secret create huggingface HF_TOKEN=hf_xxx        # a token with read access (pi05_base download)

Dataset from the Hugging Face Hub (--dataset <repo id>) or from the pi05-datasets Modal volume (--volume-dataset <dir>):
    uv run modal volume create pi05-datasets                                      # once
    uv run modal volume put pi05-datasets data/rig_cab_nighttest /rig_cab_nighttest
    uv run modal volume ls pi05-datasets /rig_cab_nighttest                       # meta/ data/ videos/ ...

Run (keeps going if you close the laptop, thanks to --detach):
    uv run modal run --detach modal_jobs/train_pi05.py --dataset <hf_user>/synth_excavator
    uv run modal run --detach modal_jobs/train_pi05.py --volume-dataset rig_cab_nighttest --steps 3000 --batch 32 \
        --rename '{"front": "base_0_rgb", "wrist": "left_wrist_0_rgb", "side": "right_wrist_0_rgb"}'

Every dataset camera must be in --rename (a camera left out would be silently ignored by the model).
The run directory also gets rename.json (the --rename map) and dataset.json (image size, state dim, fps, cameras of
the training data), which modal_jobs/serve_pi05.py reads to serve the checkpoint with the training pixels.

Watch:  the URL printed by `modal run`, or `uv run modal app logs pi05-train`
Get the checkpoint afterwards:
    uv run modal volume ls pi05-runs
    uv run modal volume get pi05-runs <run_name>/checkpoints/last ./checkpoints/<run_name>
"""
import json
import subprocess
import time
from pathlib import Path

import modal

app = modal.App("pi05-train")

# pi0.5 base was trained with cameras named base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb.
# Map our camera names onto those slots; any slot we don't fill is passed as a masked empty image.
# Three cameras: --rename '{"front": "base_0_rgb", "wrist": "left_wrist_0_rgb", "side": "right_wrist_0_rgb"}'
DEFAULT_RENAME = '{"front": "base_0_rgb"}'
PI05_SLOTS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
DATASETS_MOUNT = "/datasets"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git")
    .uv_pip_install("lerobot[pi,training,dataset]")
    .env({"HF_HOME": "/cache/hf"})  # model + dataset downloads persist between runs
)
runs = modal.Volume.from_name("pi05-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
datasets = modal.Volume.from_name("pi05-datasets", create_if_missing=True)  # `modal volume put` LeRobot datasets here

with image.imports():  # container-only deps (huggingface_hub comes with lerobot)
    from huggingface_hub import file_exists, hf_hub_download


def dataset_info(dataset, data_dir):
    """meta/info.json of the training dataset: from the pi05-datasets volume, else from the Hub."""
    if data_dir:
        path = Path(DATASETS_MOUNT) / data_dir / "meta" / "info.json"
        if not path.exists():
            have = sorted(p.name for p in Path(DATASETS_MOUNT).iterdir()) if Path(DATASETS_MOUNT).is_dir() else []
            raise SystemExit(f"no {path} in the pi05-datasets volume (top-level dirs there: {have})")
    else:
        path = Path(hf_hub_download(dataset, "meta/info.json", repo_type="dataset"))
    return json.loads(path.read_text())


def serving_meta(info, rename, source, zero_state=False):
    """Checks the camera rename against the dataset; returns dataset.json for modal_jobs/serve_pi05.py
    (zero_state: dataset_zero_state(), the flag the server acts on)."""
    feats = info["features"]
    cams = sorted(k.split(".", 2)[2] for k in feats if k.startswith("observation.images."))
    if sorted(rename) != cams:
        order = [c for c in ("front", "wrist", "side") if c in cams] + [c for c in cams if c not in ("front", "wrist", "side")]
        raise SystemExit(f"--rename cameras {sorted(rename)} != dataset cameras {cams}: map every dataset camera, e.g. "
                         + json.dumps(dict(zip(order, PI05_SLOTS))))
    bad = [v for v in rename.values() if v not in PI05_SLOTS]
    if bad or len(set(rename.values())) != len(rename):
        raise SystemExit(f"--rename targets must be distinct pi0.5 camera slots {PI05_SLOTS}, got {list(rename.values())}")
    shapes = {tuple(feats[f"observation.images.{c}"]["shape"][:2]) for c in cams}
    if len(shapes) != 1:
        raise SystemExit(f"expected one image size across cameras, got {shapes}")
    return {"zero_state": zero_state, "source": source, "fps": info["fps"], "robot_type": info.get("robot_type") or "",
            "image_hw": list(shapes.pop()), "state_dim": feats["observation.state"]["shape"][0],
            "action_dim": feats["action"]["shape"][0], "cameras": cams}


def dataset_zero_state(dataset, data_dir):
    """True if the training dataset was made by scripts/zero_state.py (meta/zero_state.json), False if not; recorded
    in the run's dataset.json, from which serve_*.py zeros every incoming state for such a run. The marker must agree
    with the dataset's observation.state stats (all zero iff marked; a --noise marker's train_noise_sigma: mean ~0,
    std ~sigma): a zero-state checkpoint fed a real state sees it scaled by ~1e8 (~1/sigma for noise), so a
    disagreement stops the job before it trains. A noise run is served as zero_state too: inference sends zeros."""
    def fetch(rel):
        if data_dir:
            path = Path(DATASETS_MOUNT) / data_dir / rel
            return json.loads(path.read_text()) if path.exists() else None
        if not file_exists(dataset, rel, repo_type="dataset"):
            return None
        return json.loads(Path(hf_hub_download(dataset, rel, repo_type="dataset")).read_text())

    marker = fetch("meta/zero_state.json")
    stats = (fetch("meta/stats.json") or {}).get("observation.state")
    flat = stats is not None and all(float(x) == 0.0 for k, v in stats.items() if k != "count" for x in v)
    if marker is not None and "observation.state" not in (marker.get("zeroed") or []):
        raise SystemExit(f"meta/zero_state.json does not list observation.state in 'zeroed': {marker}")
    sigma = float((marker or {}).get("train_noise_sigma") or 0.0)
    if sigma > 0:   # zero_state.py --noise: state = N(0, sigma^2) noise in training, zeros (its mean) at inference
        if stats is None:
            raise SystemExit("meta/zero_state.json has train_noise_sigma but the dataset has no observation.state stats")
        std, mean = [float(x) for x in stats["std"]], [float(x) for x in stats["mean"]]
        if not all(0.8 * sigma < s < 1.2 * sigma for s in std) or not all(abs(m) < 0.2 * sigma for m in mean):
            raise SystemExit(f"meta/zero_state.json says N(0, {sigma}^2) noise but the observation.state stats have "
                             f"mean {mean}, std {std}: re-make the dataset with scripts/zero_state.py --noise")
    elif marker is not None and stats is not None and not flat:
        raise SystemExit("meta/zero_state.json is there but the observation.state stats are not all zero: "
                         "re-make the dataset with scripts/zero_state.py")
    if marker is None and flat:
        raise SystemExit("observation.state stats are all zero but there is no meta/zero_state.json: make the "
                         "dataset with scripts/zero_state.py (the marker is what makes the server zero the state)")
    print(f"zero_state: {marker is not None}" + (f" (trained on N(0, {sigma}^2) state noise)" if sigma else ""),
          flush=True)
    return marker is not None


@app.function(
    image=image,
    gpu="H100",  # "A100-80GB" also fits
    volumes={"/runs": runs, "/cache": hf_cache, DATASETS_MOUNT: datasets},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=4 * 3600,
)
def train(dataset: str, steps: int, batch: int, run_name: str, rename: str, extra: str = "", data_dir: str = ""):
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)
    rename_map = json.loads(rename)
    source = f"volume pi05-datasets:/{data_dir}" if data_dir else f"hub {dataset}"
    meta = serving_meta(dataset_info(dataset, data_dir), rename_map, source,  # fail before paying for training
                        dataset_zero_state(dataset, data_dir))
    if data_dir:  # a local LeRobot dataset: lerobot reads it from root and never asks the Hub for it
        data_args = [f"--dataset.repo_id=local/{data_dir}", f"--dataset.root={DATASETS_MOUNT}/{data_dir}"]
    else:
        data_args = [f"--dataset.repo_id={dataset}"]
    cmd = [
        "lerobot-train",
        *data_args,
        "--policy.path=lerobot/pi05_base",
        "--policy.dtype=bfloat16",
        "--policy.gradient_checkpointing=true",
        "--policy.push_to_hub=false",
        f"--steps={steps}",
        f"--batch_size={batch}",
        "--log_freq=50",
        f"--save_freq={max(steps // 2, 1)}",
        f"--output_dir=/runs/{run_name}",
        "--rename_map=" + json.dumps({f"observation.images.{k}": f"observation.images.{v}"
                                      for k, v in rename_map.items()}),
        *extra.split(),
    ]
    print(" ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True)
    finally:
        # after lerobot-train: it refuses to start if its output_dir already exists
        out = Path("/runs") / run_name
        out.mkdir(parents=True, exist_ok=True)
        (out / "rename.json").write_text(json.dumps(rename_map, indent=1))
        (out / "dataset.json").write_text(json.dumps(meta, indent=1))
        runs.commit()   # keep whatever checkpoints exist even if training crashed
        hf_cache.commit()
    print(f"done: checkpoints in volume pi05-runs at {run_name}/checkpoints (+ rename.json, dataset.json)")


@app.local_entrypoint()
def main(dataset: str = "", volume_dataset: str = "", steps: int = 2000, batch: int = 16, rename: str = DEFAULT_RENAME,
         extra: str = ""):
    if bool(dataset) == bool(volume_dataset):
        raise SystemExit("pass exactly one of --dataset <hf repo id> or --volume-dataset <dir in the pi05-datasets volume>")
    data_dir = volume_dataset.strip("/")
    run_name = f"pi05_{(dataset or data_dir).split('/')[-1]}_{time.strftime('%m%d_%H%M')}"
    json.loads(rename)  # fail here, not in the container
    print(f"run: {run_name}")
    train.remote(dataset, steps, batch, run_name, rename, extra, data_dir)
