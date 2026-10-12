"""Serve a fine-tuned pi0.5 checkpoint (pi05-runs volume) as an HTTPS policy endpoint on Modal.

Fallback to the PI-hosted pi07 path. Client: act.policies.HTTPPolicy (scripts/control_loop.py --policy http --url <url>).

    uv run modal secret create pi05-serve PI05_SERVE_TOKEN=$(openssl rand -hex 24)   # once; put the same value in .env
    uv run modal deploy modal_jobs/serve_pi05.py                                     # latest run in pi05-runs
    PI05_SERVE_RUN=pi05_rig_cab_0926_1400 PI05_SERVE_GPU=H100 uv run modal deploy modal_jobs/serve_pi05.py
    uv run modal app stop pi05-serve                                                 # stop paying for a warm GPU

The deploy prints the URL (https://<workspace>--pi05-serve-policyserver-web.modal.run). Contract (both need
`Authorization: Bearer <PI05_SERVE_TOKEN>`, else 401):
    GET  /health -> {run, checkpoint, cameras, image_hw, state_dim, chunk_size, ...}   (waits for the model to load)
    POST /infer  {"prompt": str, "state": [swing, boom, arm, bucket, thumb] (the {ch}.cmd readback),
                  "images": {our camera name: base64 JPEG}}
              -> {"actions": [[5 floats] x chunk_size] in -1..1, "server_ms": float, "run": str, "zero_state": bool}

Per request: our camera names -> the checkpoint's image keys via <run>/rename.json (written by train_pi05.py; for
older runs, the rename map saved in the checkpoint's policy_preprocessor.json), each image resized to the training
dataset's image size (<run>/dataset.json; older runs: the checkpoint's input-feature size), then lerobot's
preprocessor -> PI05Policy.predict_action_chunk (one forward pass, the whole chunk) -> postprocessor (unnormalize)
per step -> clip to -1..1. Non-finite actions are an error, never sent.

Zero-state runs (trained on a scripts/zero_state.py dataset: dataset.json "zero_state": true, written by the train
jobs from the dataset's meta/zero_state.json): the server replaces every incoming state with zeros before the
preprocessor, whatever the client sent. Such a checkpoint normalizes state with std = 0 (+ eps), so a real state would
reach the model scaled by ~1e8. /health and every /infer reply carry "zero_state". Runs whose dataset.json has no
zero_state field (older runs) are served as normal runs (state passed through), with a log line. A normal run's
state is never zeroed.

Cost: one container at most (max_containers=1), kept warm SCALEDOWN_S after the last request.
"""
import base64
import binascii
import hmac
import json
import os
import time
from pathlib import Path

import cv2
import modal
import numpy as np

GPU = os.environ.get("PI05_SERVE_GPU", "L40S")      # pi0.5 in bf16 needs ~8 GB; L40S (48 GB) is plenty
RUN = os.environ.get("PI05_SERVE_RUN", "")           # run dir in the pi05-runs volume; "" = latest run with checkpoints
SCALEDOWN_S = 10 * 60                                # idle container stays warm this long (a test session)
MAX_BODY = 8 * 1024 * 1024
TOKEN_ENV = "PI05_SERVE_TOKEN"

app = modal.App("pi05-serve")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git")
    .uv_pip_install("lerobot[pi,training,dataset]", "opencv-python-headless", "fastapi[standard]")
    # baked in at deploy time, so the container resolves the same run / GPU as the deploying shell
    .env({"HF_HOME": "/cache/hf", "PI05_SERVE_GPU": GPU, "PI05_SERVE_RUN": RUN})
)
runs = modal.Volume.from_name("pi05-runs")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

with image.imports():  # container-only deps; skipped locally if missing (the request handling below needs none)
    import torch
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.utils import prepare_observation_for_inference
    from starlette.concurrency import run_in_threadpool


def find_checkpoint(run, base=Path("/runs")):
    """The newest run with checkpoints unless `run` is given; last or highest step."""
    if not run:
        cands = sorted([p for p in base.iterdir() if (p / "checkpoints").is_dir()], key=lambda p: p.stat().st_mtime)
        if not cands:
            raise SystemExit("no runs with checkpoints in the pi05-runs volume")
        run = cands[-1].name
    ck = base / run / "checkpoints"
    last = ck / "last" / "pretrained_model"
    if not last.exists():  # 'last' is a symlink that may not survive; fall back to the highest step
        steps = sorted([p for p in ck.iterdir() if p.name.isdigit()], key=lambda p: int(p.name))
        last = steps[-1] / "pretrained_model"
    return run, last


def saved_rename(ckpt):
    """The rename map lerobot-train saved in the checkpoint's preprocessor: {'observation.images.x': '...y'}."""
    path = Path(ckpt) / "policy_preprocessor.json"
    if not path.exists():
        return {}
    for step in json.loads(path.read_text()).get("steps", []):
        if "rename" in str(step.get("registry_name", "")) and step.get("config", {}).get("rename_map"):
            return dict(step["config"]["rename_map"])
    return {}


def camera_keys(run_dir, ckpt, image_features):
    """Our camera name -> the checkpoint's image key, from <run>/rename.json (else the checkpoint's saved map).
    Both must agree when both exist; every target must be one of the checkpoint's image features."""
    from_ckpt = {k.split(".", 2)[2]: v for k, v in saved_rename(ckpt).items() if k.startswith("observation.images.")}
    path = Path(run_dir) / "rename.json"
    if path.exists():
        cams = {k: f"observation.images.{v}" for k, v in json.loads(path.read_text()).items()}
        if from_ckpt and from_ckpt != cams:
            raise SystemExit(f"{path} {cams} disagrees with the checkpoint's saved rename map {from_ckpt}")
    elif from_ckpt:
        cams = from_ckpt
    else:  # no rename at training time: dataset camera names are the model's image keys
        cams = {k.split(".", 2)[2]: k for k in image_features}
    bad = {c: k for c, k in cams.items() if k not in image_features}
    if bad:
        raise SystemExit(f"camera map {bad} targets keys the checkpoint doesn't have: {sorted(image_features)}")
    return cams


def served_zero_state(meta, meta_path):
    """dataset.json's "zero_state" (True: the run was trained on state = zeros, so the server zeros every incoming
    state). Missing (older runs, or no dataset.json): False, logged. Anything but a bool: refuse to serve."""
    if "zero_state" not in meta:
        print(f"zero_state: {meta_path} has no zero_state field (a run from before zero-state support): serving it "
              "as a normal run, the client's state is passed through", flush=True)
        return False
    z = meta["zero_state"]
    if not isinstance(z, bool):
        raise SystemExit(f"{meta_path}: zero_state must be true or false, got {z!r}")
    print(f"zero_state: {z} ({'every incoming state is replaced by zeros' if z else 'state passed through'})",
          flush=True)
    return z


class Handler:
    """Request handling without torch / FastAPI (tested locally with a stub predict).
    predict(obs: {key: array}, prompt) -> (H, D) array; obs = observation.state float32 (D_state,) + model image keys
    -> HxWx3 RGB uint8 at image_hw. zero_state=True (a run trained on a zero-state dataset): predict always gets
    observation.state = zeros, whatever the client sent (the request is still validated)."""

    def __init__(self, predict, cameras, image_hw, state_dim, token, info=None, zero_state=False):
        self.predict, self.cameras, self.image_hw = predict, dict(cameras), tuple(image_hw)
        self.state_dim, self.token, self.info = state_dim, token or "", dict(info or {})
        if not isinstance(zero_state, bool):
            raise TypeError(f"zero_state must be a bool, got {zero_state!r}")
        self.zero_state = zero_state

    def authorized(self, header):
        # fail closed: no configured token = nobody gets in
        if not self.token or not header or not header.startswith("Bearer "):
            return False
        return hmac.compare_digest(header[len("Bearer "):].strip().encode(), self.token.encode())

    def decode(self, name, b64):
        try:
            raw = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError, TypeError):
            raise ValueError(f"images.{name}: not base64")
        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"images.{name}: not a decodable JPEG/PNG")
        h, w = self.image_hw
        if bgr.shape[:2] != (h, w):
            shrink = bgr.shape[0] > h or bgr.shape[1] > w
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
        return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def infer(self, body):
        if not isinstance(body, dict):
            raise ValueError("body must be a JSON object")
        prompt, state, images = body.get("prompt"), body.get("state"), body.get("images")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 500:
            raise ValueError("prompt: a non-empty string (the training task, <= 500 chars)")
        if (not isinstance(state, list) or len(state) != self.state_dim
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in state)):
            raise ValueError(f"state: a list of {self.state_dim} numbers (the {{ch}}.cmd readback)")
        state = np.asarray(state, np.float32)
        if not np.all(np.isfinite(state)):
            raise ValueError("state: non-finite value")
        if self.zero_state:   # trained on state = zeros: anything else reaches the model scaled by ~1e8
            state = np.zeros(self.state_dim, np.float32)
        if not isinstance(images, dict):
            raise ValueError("images: {camera name: base64 JPEG}")
        missing = sorted(set(self.cameras) - set(images))
        if missing:
            raise ValueError(f"images: missing cameras {missing} (the checkpoint uses {sorted(self.cameras)})")
        obs = {"observation.state": state}
        for name, key in self.cameras.items():
            obs[key] = self.decode(name, images[name])
        actions = np.asarray(self.predict(obs, prompt), np.float64)
        if actions.ndim != 2 or actions.shape[0] < 1:
            raise RuntimeError(f"policy returned actions of shape {actions.shape}, want (H, D)")
        if not np.all(np.isfinite(actions)):
            raise RuntimeError("policy returned non-finite actions")
        out = {"actions": np.clip(actions, -1.0, 1.0).tolist(),
               "ignored_cameras": sorted(set(images) - set(self.cameras)), "zero_state": self.zero_state}
        return out

    def respond(self, method, path, auth, raw):
        """-> (status, JSON dict). The only entry point the web layer calls."""
        t0 = time.perf_counter()
        if not self.authorized(auth):
            return 401, {"error": "unauthorized: send Authorization: Bearer <PI05_SERVE_TOKEN>"}
        if method == "GET" and path == "/health":
            out = dict(self.info, ok=True, cameras=self.cameras, image_hw=list(self.image_hw),
                       state_dim=self.state_dim, zero_state=self.zero_state)
            return 200, out
        if method != "POST" or path != "/infer":
            return 404, {"error": f"no {method} {path}: POST /infer, GET /health"}
        if len(raw) > MAX_BODY:
            return 413, {"error": f"body over {MAX_BODY} bytes"}
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            return 400, {"error": "body is not JSON"}
        try:
            out = self.infer(body)
        except ValueError as e:
            return 400, {"error": str(e)}
        except RuntimeError as e:
            return 500, {"error": str(e)}
        out["server_ms"] = round(1e3 * (time.perf_counter() - t0), 1)
        out["run"] = self.info.get("run")
        return 200, out


@app.cls(image=image, gpu=GPU, volumes={"/runs": runs, "/cache": hf_cache},
         secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("pi05-serve")],
         scaledown_window=SCALEDOWN_S, max_containers=1, timeout=600)
class PolicyServer:
    @modal.enter()
    def load(self):
        run, ckpt = find_checkpoint(os.environ.get("PI05_SERVE_RUN", ""))
        run_dir = Path("/runs") / run
        print(f"run {run}\ncheckpoint {ckpt}", flush=True)
        self.device = torch.device("cuda")
        self.policy = PI05Policy.from_pretrained(str(ckpt))
        self.policy.to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=str(ckpt))
        feats = self.policy.config.image_features
        cams = camera_keys(run_dir, ckpt, feats)
        meta_path = run_dir / "dataset.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        # the training pixels: the dataset's image size (the model then resizes to its own resolution, as in training)
        image_hw = meta.get("image_hw") or list(feats[next(iter(cams.values()))].shape[1:])
        state_dim = meta.get("state_dim") or self.policy.config.output_features["action"].shape[0]
        self.robot_type = meta.get("robot_type", "")
        info = {"run": run, "checkpoint": str(ckpt), "chunk_size": self.policy.config.chunk_size,
                "dataset": meta or None, "gpu": os.environ.get("PI05_SERVE_GPU", GPU)}
        if not meta:
            print(f"WARNING: no {meta_path}: images resized to the checkpoint's input size {image_hw}", flush=True)
        self.handler = Handler(self.predict, cams, image_hw, state_dim, os.environ.get(TOKEN_ENV, ""), info,
                               zero_state=served_zero_state(meta, meta_path))
        if not os.environ.get(TOKEN_ENV):
            print(f"WARNING: no {TOKEN_ENV} in the pi05-serve secret: every request gets 401", flush=True)
        # one throwaway forward pass, so the first real request doesn't pay for CUDA warm-up (client timeout ~2 s)
        blank = {"observation.state": np.zeros(state_dim, np.float32),
                 **{k: np.zeros((*image_hw, 3), np.uint8) for k in cams.values()}}
        t0 = time.perf_counter()
        self.predict(blank, "warm up")
        print(f"warm-up forward pass {time.perf_counter() - t0:.1f} s", flush=True)
        print(f"serving: cameras {cams}, images {image_hw}, state dim {state_dim}, chunk {info['chunk_size']}, "
              f"zero_state {self.handler.zero_state}", flush=True)

    def predict(self, obs, prompt):
        with torch.inference_mode():
            batch = prepare_observation_for_inference(dict(obs), self.device, prompt, self.robot_type)
            batch = self.pre(batch)
            chunk = self.policy.predict_action_chunk(batch)          # (1, chunk_size, action_dim), normalized
            # the postprocessor works on (B, action_dim): unnormalize step by step, as lerobot's async PolicyServer
            steps = [self.post(chunk[:, i]) for i in range(chunk.shape[1])]
            return torch.stack(steps, dim=1)[0].float().cpu().numpy()

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
