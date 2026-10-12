"""SmolVLA fine-tune on Modal: the fast-to-train policy option next to pi0.5 (docs/smolvla.md).

Same shape as modal_jobs/train_pi05.py: lerobot-train from lerobot/smolvla_base, dataset from the pi05-datasets
Modal volume (--volume-dataset <dir>) or the Hub (--dataset <repo id>), runs into the smolvla-runs volume.

    uv run modal volume ls pi05-datasets /pilot-best7            # upload first if missing (docs/smolvla.md)
    uv run modal run --detach modal_jobs/train_smolvla.py --volume-dataset pilot-best7
    uv run modal run --detach modal_jobs/train_smolvla.py --volume-dataset pilot-best7 --steps 3000 --batch 64 \
        --rename '{"front": "camera1", "wrist": "camera2"}'
    SMOLVLA_TRAIN_GPU=A100-80GB uv run modal run --detach modal_jobs/train_smolvla.py --volume-dataset pilot-best7

smolvla_base's config declares three image inputs, observation.images.camera1/2/3, state 6 and action 6 (state and
action are zero-padded to 32 inside the model, so our 5 joints fit). A camera slot with no image is dropped: two
cameras are fine. Every dataset camera must be in --rename and map to a distinct slot, or the job stops before any
GPU time is spent (an unmapped camera would be silently ignored by the model).
The run directory also gets rename.json and dataset.json, which modal_jobs/serve_smolvla.py reads.

Watch:  the URL printed by `modal run`, or `uv run modal app logs smolvla-train`
"""
import json
import os
import subprocess
import time
from pathlib import Path

import modal

LEROBOT = "lerobot[smolvla,training,dataset]==0.6.1"  # the version the policy / processor code was checked against
GPU = os.environ.get("SMOLVLA_TRAIN_GPU", "L40S")     # ~450M params: fits L40S (48 GB) at batch 64
# smolvla_base's image inputs. The SmolVLA recipe orders views as camera1 = top/front, camera2 = wrist,
# camera3 = side, so side / world goes to camera3 if we ever record one.
DEFAULT_RENAME = '{"front": "camera1", "wrist": "camera2"}'
SMOLVLA_SLOTS = ("camera1", "camera2", "camera3")
DATASETS_MOUNT = "/datasets"

app = modal.App("smolvla-train")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git")
    .uv_pip_install(LEROBOT)
    .env({"HF_HOME": "/cache/hf", "SMOLVLA_TRAIN_GPU": GPU})  # HF downloads persist; container sees the same GPU
)
runs = modal.Volume.from_name("smolvla-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
datasets = modal.Volume.from_name("pi05-datasets", create_if_missing=True)  # shared with the pi0.5 jobs

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
    """Checks the camera rename against the dataset; returns dataset.json for modal_jobs/serve_smolvla.py
    (zero_state: dataset_zero_state(), the flag the server acts on)."""
    feats = info["features"]
    cams = sorted(k.split(".", 2)[2] for k in feats if k.startswith("observation.images."))
    if not cams:
        raise SystemExit("the dataset has no observation.images.* features")
    if sorted(rename) != cams:
        order = [c for c in ("front", "wrist", "side") if c in cams] + [c for c in cams if c not in ("front", "wrist", "side")]
        hint = json.dumps(dict(zip(order, SMOLVLA_SLOTS))) if len(cams) <= len(SMOLVLA_SLOTS) else "(too many cameras)"
        raise SystemExit(f"--rename cameras {sorted(rename)} != dataset cameras {cams}: map every dataset camera, e.g. "
                         + hint)
    bad = [v for v in rename.values() if v not in SMOLVLA_SLOTS]
    if bad or len(set(rename.values())) != len(rename):
        raise SystemExit(f"--rename targets must be distinct SmolVLA camera slots {SMOLVLA_SLOTS}, "
                         f"got {list(rename.values())}")
    shapes = {tuple(feats[f"observation.images.{c}"]["shape"][:2]) for c in cams}
    if len(shapes) != 1:
        raise SystemExit(f"expected one image size across cameras, got {shapes}")
    return {"zero_state": zero_state, "source": source, "fps": info["fps"], "robot_type": info.get("robot_type") or "",
            "image_hw": list(shapes.pop()), "state_dim": feats["observation.state"]["shape"][0],
            "action_dim": feats["action"]["shape"][0], "cameras": cams, "policy_type": "smolvla"}


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
    gpu=GPU,
    cpu=8.0,           # 2 AV1 video streams x batch 64 per step: decoding needs cores, not the 0.125-core default
    memory=32768,
    volumes={"/runs": runs, "/cache": hf_cache, DATASETS_MOUNT: datasets},
    secrets=[modal.Secret.from_name("huggingface")],  # smolvla_base is public; the token only avoids rate limits
    timeout=4 * 3600,
)
def train(dataset: str, steps: int, batch: int, run_name: str, rename: str, extra: str = "", data_dir: str = "",
          workers: int = 8):
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
        "--policy.path=lerobot/smolvla_base",
        "--policy.push_to_hub=false",
        f"--steps={steps}",
        f"--batch_size={batch}",
        f"--num_workers={workers}",
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
    print(f"done: checkpoints in volume smolvla-runs at {run_name}/checkpoints (+ rename.json, dataset.json)")


@app.local_entrypoint()
def main(dataset: str = "", volume_dataset: str = "", steps: int = 3000, batch: int = 64, rename: str = DEFAULT_RENAME,
         extra: str = "", workers: int = 8):
    # default 3000 x 64 = 192k samples = ~38 passes over the 5,085-frame pilot-best7 dataset
    if bool(dataset) == bool(volume_dataset):
        raise SystemExit("pass exactly one of --dataset <hf repo id> or --volume-dataset <dir in the pi05-datasets volume>")
    data_dir = volume_dataset.strip("/")
    run_name = f"smolvla_{(dataset or data_dir).split('/')[-1]}_{time.strftime('%m%d_%H%M')}"
    json.loads(rename)  # fail here, not in the container
    print(f"run: {run_name}  (GPU {GPU}; {steps} steps x batch {batch})")
    train.remote(dataset, steps, batch, run_name, rename, extra, data_dir, workers)
