"""Serve a fine-tuned SmolVLA checkpoint (smolvla-runs volume) on Modal with the SAME contract as serve_pi05.py.

Same client (act.policies.HTTPPolicy, the dashboard's "pi0.5 (Modal URL)" backend), same bearer token (the
pi05-serve secret's PI05_SERVE_TOKEN), same /infer + /health. Runbook: docs/smolvla.md.

    uv run modal deploy modal_jobs/serve_smolvla.py                                  # latest run in smolvla-runs
    SMOLVLA_SERVE_RUN=smolvla_pilot-best7_0926_2100 uv run modal deploy modal_jobs/serve_smolvla.py
    uv run modal app stop smolvla-serve                                              # stop paying for a warm GPU

The request handling (auth, validation, camera rename, resize, clip) is serve_pi05.Handler itself, imported and
shipped into the container with add_local_python_source, so the two servers cannot drift apart. Only the model side
differs: SmolVLAPolicy.predict_action_chunk (one flow-matching sample, the whole chunk) -> postprocessor per step.
/health additionally reports policy_type "smolvla". Zero-state runs (dataset.json "zero_state": true): the shared
Handler replaces every incoming state with zeros (serve_pi05.py docstring, docs/zero_state.md).
"""
import json
import os
import time
from pathlib import Path

import modal
import numpy as np

from serve_pi05 import TOKEN_ENV, Handler, camera_keys, find_checkpoint, served_zero_state

GPU = os.environ.get("SMOLVLA_SERVE_GPU", "L40S")         # ~450M params, ~2 GB of weights: any GPU fits
RUN = os.environ.get("SMOLVLA_SERVE_RUN", "")              # run dir in the runs volume; "" = latest run
RUNS_VOLUME = os.environ.get("SMOLVLA_SERVE_VOLUME", "smolvla-runs")
SCALEDOWN_S = 10 * 60                                      # idle container stays warm this long (a test session)
POLICY_TYPE = "smolvla"

app = modal.App("smolvla-serve")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git")
    .uv_pip_install("lerobot[smolvla,training,dataset]==0.6.1", "opencv-python-headless", "fastapi[standard]")
    # baked in at deploy time, so the container resolves the same run / GPU / volume as the deploying shell
    .env({"HF_HOME": "/cache/hf", "SMOLVLA_SERVE_GPU": GPU, "SMOLVLA_SERVE_RUN": RUN,
          "SMOLVLA_SERVE_VOLUME": RUNS_VOLUME})
    .add_local_python_source("serve_pi05")                 # Handler & co.; must stay the last image step
)
runs = modal.Volume.from_name(RUNS_VOLUME)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

with image.imports():  # container-only deps; skipped locally if missing (the request handling needs none)
    import torch
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.utils import prepare_observation_for_inference
    from starlette.concurrency import run_in_threadpool


def require_runs(base):
    """Before find_checkpoint picks the newest run: a SmolVLA-specific error when the volume has none."""
    if not any((p / "checkpoints").is_dir() for p in Path(base).iterdir()):
        raise SystemExit(f"no runs with checkpoints in the {RUNS_VOLUME} volume: train with modal_jobs/train_smolvla.py")


def make_handler(predict, run, run_dir, ckpt, token, gpu=GPU):
    """Torch-free: reads the checkpoint's config.json + the run's rename.json / dataset.json -> serve_pi05.Handler."""
    cfg = json.loads((Path(ckpt) / "config.json").read_text())
    if cfg.get("type") != POLICY_TYPE:
        raise SystemExit(f"{ckpt} is a {cfg.get('type')!r} checkpoint, not {POLICY_TYPE}: serve it with its own server")
    feats = {k: v for k, v in cfg["input_features"].items() if v.get("type") == "VISUAL"}
    cams = camera_keys(run_dir, ckpt, feats)
    meta_path = Path(run_dir) / "dataset.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    # the training pixels: the dataset's image size (the model then resizes with padding to 512x512, as in training)
    image_hw = meta.get("image_hw") or list(feats[next(iter(cams.values()))]["shape"][1:])
    state_dim = meta.get("state_dim") or cfg["output_features"]["action"]["shape"][0]
    if not meta:
        print(f"WARNING: no {meta_path}: images resized to the checkpoint's input size {image_hw}", flush=True)
    info = {"run": run, "checkpoint": str(ckpt), "policy_type": POLICY_TYPE, "chunk_size": cfg["chunk_size"],
            "n_action_steps": cfg.get("n_action_steps"), "num_steps": cfg.get("num_steps"),
            "dataset": meta or None, "gpu": gpu}
    return Handler(predict, cams, image_hw, state_dim, token, info, zero_state=served_zero_state(meta, meta_path)), meta


def load_policy(ckpt, device):
    """SmolVLAPolicy + the checkpoint's pre/post processors (rename, tokenizer, dataset-stat normalizers) on device."""
    policy = SmolVLAPolicy.from_pretrained(str(ckpt))
    policy.to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(ckpt),
                                         preprocessor_overrides={"device_processor": {"device": str(device)}})
    return policy, pre, post


def predict_chunk(policy, pre, post, device, obs, prompt, robot_type=""):
    """obs {observation.state (D,), model image key: HxWx3 uint8} -> (chunk_size, action_dim) float32, unnormalized."""
    with torch.inference_mode():
        batch = prepare_observation_for_inference(dict(obs), device, prompt, robot_type)
        batch = pre(batch)
        chunk = policy.predict_action_chunk(batch)      # (1, chunk_size, action_dim), normalized
        # the postprocessor works on (B, action_dim): unnormalize step by step, as lerobot's async PolicyServer
        steps = [post(chunk[:, i]) for i in range(chunk.shape[1])]
        return torch.stack(steps, dim=1)[0].float().cpu().numpy()


@app.cls(image=image, gpu=GPU, volumes={"/runs": runs, "/cache": hf_cache},
         secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("pi05-serve")],
         scaledown_window=SCALEDOWN_S, max_containers=1, timeout=600)
class PolicyServer:
    @modal.enter()
    def load(self):
        run = os.environ.get("SMOLVLA_SERVE_RUN", "")
        if not run:
            require_runs("/runs")
        run, ckpt = find_checkpoint(run)
        run_dir = Path("/runs") / run
        print(f"run {run}\ncheckpoint {ckpt}", flush=True)
        self.device = torch.device("cuda")
        token = os.environ.get(TOKEN_ENV, "")
        self.handler, meta = make_handler(self.predict, run, run_dir, ckpt, token,
                                          os.environ.get("SMOLVLA_SERVE_GPU", GPU))
        self.robot_type = meta.get("robot_type", "")
        self.policy, self.pre, self.post = load_policy(ckpt, self.device)
        if not token:
            print(f"WARNING: no {TOKEN_ENV} in the pi05-serve secret: every request gets 401", flush=True)
        h = self.handler
        # one throwaway forward pass, so the first real request doesn't pay for CUDA warm-up (client timeout ~2 s)
        blank = {"observation.state": np.zeros(h.state_dim, np.float32),
                 **{k: np.zeros((*h.image_hw, 3), np.uint8) for k in h.cameras.values()}}
        t0 = time.perf_counter()
        self.predict(blank, "warm up")
        print(f"warm-up forward pass {time.perf_counter() - t0:.1f} s", flush=True)
        print(f"serving smolvla: cameras {h.cameras}, images {list(h.image_hw)}, state dim {h.state_dim}, "
              f"chunk {h.info['chunk_size']}, zero_state {h.zero_state}", flush=True)

    def predict(self, obs, prompt):
        return predict_chunk(self.policy, self.pre, self.post, self.device, obs, prompt, self.robot_type)

    @modal.asgi_app()
    def web(self):
        api = FastAPI()

        async def route(request: Request):
            raw = await request.body()
            status, out = await run_in_threadpool(self.handler.respond, request.method, request.url.path,
                                                  request.headers.get("authorization"), raw)
            return JSONResponse(out, status_code=status)

        api.add_api_route("/infer", route, methods=["POST"])
        api.add_api_route("/health", route, methods=["GET"])
        return api
