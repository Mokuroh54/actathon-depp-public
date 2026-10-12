"""Policy backends with one interface: infer(obs, prompt) -> (H, D) actions, one per control step.

  HTTPPolicy      client for our policy server (JSON + JPEG frames): modal_jobs/serve_pi05.py, pi0.5 fallback, and
                  modal_jobs/serve_smolvla.py (same contract)
  PIPolicy        client for a PI-hosted fine-tuned checkpoint (pi07) through pi-sdk's PolicyClient
  ZeroState       wraps any backend so it sees state = zeros (checkpoints trained on scripts/zero_state.py datasets)
"""
import base64
import importlib.util
import json
import os
from pathlib import Path

import cv2
import numpy as np
import requests
from pi_sdk.inference import ClientConfig, InferenceInput, PolicyClient


class HTTPPolicy:
    """Client for our policy server (modal_jobs/serve_pi05.py): POST <url>/infer JSON {prompt, state, images: {name: b64
    jpeg}} -> {actions: (H, D), server_ms}. state = the {ch}.cmd readback (same as the dataset's observation.state);
    token: sent as Authorization: Bearer (serve_pi05 rejects requests without it).
    image_hw: send frames at exactly this size (the training dataset's; RigMachine's crops already are, so no resize
    happens and the server doesn't resize either); None = scale the long side to image_px.
    zero_state: whether this client runs with --zero-state (the ZeroState wrapper does the zeroing); connect() refuses
    a server whose /health reports the other setting. A zero-state server also zeros the state itself."""

    def __init__(self, url, cfg, image_px=224, jpeg_quality=80, timeout_s=2.0, token=None, image_hw=None,
                 zero_state=False):
        self.session, self.base, self.cfg = requests.Session(), url.rstrip("/"), cfg
        self.url = self.base + "/infer"
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        self.image_px, self.quality, self.timeout = image_px, jpeg_quality, timeout_s
        self.image_hw = tuple(image_hw) if image_hw else None
        self.last_payload_bytes, self.last_server_ms = 0, None
        self.zero_state = bool(zero_state)

    def reset(self):
        pass

    def connect(self, timeout_s=900):
        """GET <url>/health: waits for the server to start (a cold Modal container loads the model for minutes) and
        checks it wants cameras we have at our image size. Returns its info dict."""
        r = self.session.get(self.base + "/health", timeout=timeout_s)
        if r.status_code == 401:
            raise SystemExit("policy server: 401 unauthorized (PI05_SERVE_TOKEN missing or different from the "
                             "server's pi05-serve secret)")
        r.raise_for_status()
        info = r.json()
        missing = sorted(set(info.get("cameras") or {}) - set(self.cfg.cameras))
        if missing:
            raise SystemExit(f"policy server wants cameras {sorted(info['cameras'])}; we have no {missing}: "
                             "pass --cam with the names used in training")
        want = tuple(info.get("image_hw") or ())
        if self.image_hw and want and want != self.image_hw:
            raise SystemExit(f"policy server expects {want} images, --dataset gives {self.image_hw}: "
                             "checkpoint trained on another dataset?")
        served = info.get("zero_state")      # None: a server from before zero-state support (no check possible)
        if served is not None and bool(served) != self.zero_state:
            if served:
                raise SystemExit(f"policy server run {info.get('run')} was trained with state = zeros (zero_state): "
                                 "run with --zero-state and the zero-state --dataset (scripts/zero_state.py)")
            raise SystemExit(f"--zero-state, but policy server run {info.get('run')} was trained on real state: drop "
                             "--zero-state and use that run's own dataset, or serve the zero-state run")
        return info

    def encode(self, rgb):
        h, w = rgb.shape[:2]
        if self.image_hw:
            th, tw = self.image_hw
            img = rgb if (h, w) == (th, tw) else cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)
        else:
            s = self.image_px / max(h, w)
            img = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return base64.b64encode(buf.tobytes()).decode()

    def body(self, obs, prompt):
        """The /infer request body: prompt, state = the {ch}.cmd readback, every camera as base64 JPEG."""
        return {
            "prompt": prompt,
            "state": [float(obs[f"{j}.cmd"]) for j in self.cfg.joint_names],
            "images": {name: self.encode(obs[name]) for name in self.cfg.cameras},
        }

    def post(self, body):
        """POST /infer -> ((H, D) actions clipped to -1..1, the reply dict). ValueError on a bad shape / non-finite."""
        r = self.session.post(self.url, json=body, timeout=self.timeout)
        r.raise_for_status()
        self.last_payload_bytes = len(r.request.body or b"")
        data = r.json()
        self.last_server_ms = data.get("server_ms")
        chunk = np.array(data["actions"], float)
        if chunk.ndim != 2 or chunk.shape[1] != len(self.cfg.joint_names) or not np.all(np.isfinite(chunk)):
            raise ValueError(f"server returned actions of shape {chunk.shape} (or non-finite), "
                             f"want (H, {len(self.cfg.joint_names)})")
        return np.clip(chunk, -1.0, 1.0), data

    def infer(self, obs, prompt):
        return self.post(self.body(obs, prompt))[0]


# PI platform-contract topics for the excavator (same as scripts/pi_dry_run.py, which runs in the pi-upload-lerobot
# tool env and can't import act): state = the previous command, action = {ch}.cmd, both (5,) float32 in CHANNELS order
PI_STATE_TOPIC = "observation/joystick/cmd"
PI_ACTION_TOPIC = "action/joystick/cmd"
# obs key carrying the current subtask instruction (control_loop --subtasks: act.phase's phase). PIPolicy sends it as
# raw_text; absent -> raw_text = the prompt, as before. It travels inside obs so every request (the chunked controller's
# snapshot, the ZeroState view) carries the phase of the tick it was built on.
SUBTASK_KEY = "subtask"


def env_secret(name, env_file=".env"):
    """`name` from the environment, else from a git-ignored .env (KEY=VALUE lines, optional `export `); "" if unset.
    Never printed or logged."""
    val = os.environ.get(name, "")
    path = Path(env_file)
    if not val and path.exists():
        for line in path.read_text().splitlines():
            k, sep, v = line.strip().partition("=")
            if sep and k.strip().removeprefix("export ").strip() == name:
                val = v.strip().strip("'\"")
    return val


def pi_api_key(env_file=".env"):
    """PI_API_KEY from the environment, else from a git-ignored .env (KEY=VALUE lines). Never printed or logged."""
    key = env_secret("PI_API_KEY", env_file)
    if not key:
        raise SystemExit("PI_API_KEY is not set (export it, or put PI_API_KEY=... in .env)")
    return key


def pi05_serve_token(env_file=".env"):
    """PI05_SERVE_TOKEN (the bearer token of modal_jobs/serve_pi05.py, = the pi05-serve Modal secret) from the
    environment or .env; "" if unset."""
    return env_secret("PI05_SERVE_TOKEN", env_file)


def match_pi_cameras(server_names, ours):
    """Server camera name -> our camera name. A server name matches our camera `cam` if it is `cam` or has it as a
    path segment (e.g. "observation/front/rgb/image" <- "front"). Every server camera needs exactly one match."""
    out = {}
    for s in server_names:
        hits = [c for c in ours if s == c or f"/{c}/" in f"/{s}/"]
        if len(hits) != 1:
            raise SystemExit(f"server camera {s!r} matches {hits or 'none'} of our cameras {list(ours)}: "
                             f"pass --cam with the names used in training (server cameras: {list(server_names)})")
        out[s] = hits[0]
    return out


class PIPolicy:
    """Client for a PI-hosted checkpoint: pi-sdk PolicyClient over a websocket to the checkpoint's gateway_url.
    Mirrors the training data (scripts/convert_pi_episodes.py -> pi-upload-lerobot):
      images  obs[cam] = RigMachine's cropped RGB (the dataset's crops.json + image size), sent under the server's
              camera names (match_pi_cameras); PolicyClient resizes / compresses to what the server wants
      state   PI_STATE_TOPIC = the {ch}.cmd readback (gate `sent` = the previous command), float32, CHANNELS order
      prompt  robot_task_string = the prompt; raw_text = obs[SUBTASK_KEY] (control_loop --subtasks: the act.phase
              phase, as the training episodes' language_segment subtask) or, without it, the prompt
    infer returns the next execute_chunk_size actions, clipped to -1..1. The first connect can take minutes while
    PI starts the model: call connect() before the control loop starts."""

    def __init__(self, gateway_url, cfg, api_key, execute_chunk_size=15, sample_args=None):
        self.cfg = cfg
        self.client = PolicyClient(ClientConfig(
            gateway_url=gateway_url, api_key=api_key, execute_chunk_size=execute_chunk_size,
            action_keys=[PI_ACTION_TOPIC], sample_args=sample_args, app_name="actathon-excavator"))
        self.cam_map, self.reset_pending, self.sizes = None, False, {}
        self.last_payload_bytes, self.last_server_ms = 0, None
        # pi-sdk resizes with cykooz-resizer, which has no Linux arm64 wheel (the Pi). Without it, resize to the
        # server's resolution here with OpenCV (same filter family), so pi-sdk skips its own resize.
        self.presize = importlib.util.find_spec("cykooz_resizer") is None

    def connect(self):
        self.client.connect()
        meta = self.client.server_metadata
        if meta.action_keys and list(meta.action_keys) != [PI_ACTION_TOPIC]:
            raise SystemExit(f"server action keys {meta.action_keys} != [{PI_ACTION_TOPIC!r}]: "
                             "checkpoint trained on another contract / training profile?")
        self.cam_map = match_pi_cameras(meta.camera_names, list(self.cfg.cameras))
        pp = meta.image_preprocessing
        if self.presize and pp is not None:
            for s in self.cam_map:
                key = self.client._payload_key(s)       # pi-sdk 0.3.2 internals: the key its resize looks up
                res = pp.image_resolutions.get(key, pp.target_resolution)
                if res is not None:
                    self.sizes[s] = (tuple(res), pp.resize_mode, pp.interpolation or "lanczos")
        return meta

    def _fit(self, img, s):
        """Resize to the server's resolution like pi-sdk would (stretch, or 'pad': keep aspect + center on black)."""
        if s not in self.sizes:
            return img
        (h, w), mode, interp = self.sizes[s]
        if img.shape[:2] == (h, w):
            return img
        flag = cv2.INTER_LINEAR if interp == "bilinear" else cv2.INTER_LANCZOS4
        if mode != "pad":
            return cv2.resize(img, (w, h), interpolation=flag)
        r = max(img.shape[1] / w, img.shape[0] / h)
        nh, nw = max(1, int(img.shape[0] / r)), max(1, int(img.shape[1] / r))
        out = np.zeros((h, w, 3), np.uint8)
        top, left = (h - nh) // 2, (w - nw) // 2
        out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=flag)
        return out

    def reset(self):
        # applied before the next request: ChunkedController may have one in flight on another thread
        self.reset_pending = True

    def make_input(self, obs, prompt):
        """obs -> the pi-sdk request: images under the server's camera names, state = the {ch}.cmd readback."""
        state = np.array([obs[f"{j}.cmd"] for j in self.cfg.joint_names], np.float32)
        return InferenceInput(
            images={s: self._fit(obs[c], s) for s, c in self.cam_map.items()}, states={PI_STATE_TOPIC: state},
            robot_task_string=prompt, raw_text=obs.get(SUBTASK_KEY) or prompt)

    def infer(self, obs, prompt):
        if self.cam_map is None:
            self.connect()
        if self.reset_pending:
            self.reset_pending = False
            self.client.reset()
        res = self.client.infer(self.make_input(obs, prompt))
        m = res.metrics
        self.last_server_ms = m.server_total_time_ms if m is not None else res.server_time_ms
        chunk = np.asarray(res.actions_by_key[PI_ACTION_TOPIC], float)
        if chunk.ndim != 2 or chunk.shape[1] != len(self.cfg.joint_names):
            raise ValueError(f"PI returned actions of shape {chunk.shape}, want (H, {len(self.cfg.joint_names)})")
        if not np.all(np.isfinite(chunk)):     # a NaN would pass np.clip and become full stick at a max/min clamp
            raise ValueError("PI returned non-finite actions")
        return np.clip(chunk, -1.0, 1.0)

    def close(self):
        self.client.close()


ZERO_STATE_MARKER = "meta/zero_state.json"  # written by scripts/zero_state.py into a dataset whose state it zeroed


def zero_state_marker(root):
    """The dataset's zero-state marker (dict) if scripts/zero_state.py zeroed its observation.state, else None."""
    path = Path(root) / ZERO_STATE_MARKER
    return json.loads(path.read_text()) if path.exists() else None


def check_zero_state(root, zero_state):
    """A checkpoint trained on a zero-state dataset must see zeros as state, and one trained on real state must not:
    either mismatch silently wrecks the policy (SmolVLA normalizes state with the dataset's std = 0 + eps, so any
    non-zero state reaches the model scaled by ~1e8). SystemExit when --zero-state disagrees with the dataset marker."""
    marked = zero_state_marker(root) is not None
    if marked and not zero_state:
        raise SystemExit(f"{root} has {ZERO_STATE_MARKER} (trained with observation.state = 0): pass --zero-state")
    if zero_state and not marked:
        raise SystemExit(f"--zero-state, but {root} has no {ZERO_STATE_MARKER}: the checkpoint was trained on real "
                         "state (use the zero-state dataset made by scripts/zero_state.py, or drop --zero-state)")


def zero_state_view(obs, joint_names):
    """What a zero-state policy sees: a shallow copy of obs with every {ch}.cmd set to 0.0 (images shared, obs itself
    untouched, so the machine / gate / logs keep the real readback)."""
    view = dict(obs)
    view.update({f"{j}.cmd": 0.0 for j in joint_names})
    return view


class ZeroState:
    """Wraps a policy so every request carries state = zeros (models trained on scripts/zero_state.py datasets)."""

    def __init__(self, inner, joint_names):
        self.inner, self.joint_names = inner, list(joint_names)

    def reset(self):
        self.inner.reset()

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def infer(self, obs, prompt):
        return self.inner.infer(zero_state_view(obs, self.joint_names), prompt)

