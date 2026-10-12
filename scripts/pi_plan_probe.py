"""Offline probe of a PI-hosted checkpoint's FULL action plans at chosen moments of recorded demos. Nothing moves.

    uv run scripts/pi_plan_probe.py data/s2-keep18 --gateway-url <url> [--episodes 0 9 11] [--samples 2]
    uv run scripts/pi_plan_probe.py data/s2-keep18-nostate --gateway-url <url> --zero-state

The runner only ever executes steps ~4-8 of each 15-step chunk (--chunk 15 --request-at 10 at ~4 ticks of latency).
This asks for the whole horizon (--horizon, default 50) at three moments per episode and prints it next to what the
operator actually did, to see whether the plan's tail holds the transition the runner never reached:
  reach     1 s into the first sustained arm-out push (arm > 0.5): does the plan stop reaching?
  clamp     1 s into the first sustained thumb clamp (thumb > 0.7): does the plan keep clamping, or release / lift?
  pre-lift  0.5 s before that clamp ends: does the plan contain the lift (boom up = boom < 0) / swing?
With real state (no --zero-state) each moment is also sent with the state's moving channel zeroed ("state-0"),
to measure how much the plan copies its state.
Requests: episodes x moments x samples (x2 with real state). PI_API_KEY comes from the environment or .env.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from lerobot.datasets import LeRobotDataset

from act.machine import CHANNELS
from act.policies import PIPolicy, ZeroState, check_zero_state, pi_api_key

SHOW = [0, 4, 8, 12, 16, 20, 25, 30, 40, 49]
IDX = {c: i for i, c in enumerate(CHANNELS)}


class DatasetCfg:
    def __init__(self, cameras):
        self.joint_names, self.cameras = list(CHANNELS), {c: None for c in cameras}


def to_rgb_uint8(t):
    return (t.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)


def first_run(mask, min_len):
    """(start, end) of the first run of True at least min_len long, else None."""
    run = 0
    for i, v in enumerate(mask):
        run = run + 1 if v else 0
        if run == min_len:
            s = i - min_len + 1
            e = s
            while e < len(mask) and mask[e]:
                e += 1
            return s, e
    return None


def moments(actions, fps):
    out = {}
    r = first_run(actions[:, IDX["arm"]] > 0.5, 8)
    if r:
        out["reach"] = (min(r[0] + fps, r[1] - 2), "arm")
    r = first_run(actions[:, IDX["thumb"]] > 0.7, fps)
    if r:
        out["clamp"] = (r[0] + fps, "thumb")
        if r[1] - fps // 2 > r[0] + fps:
            out["pre-lift"] = (r[1] - fps // 2, "thumb")
    return out


def show(label, rows):
    """One line per channel: the values at the SHOW steps."""
    for j, c in enumerate(CHANNELS):
        print(f"  {label if j == 0 else '':14s} {c:6s} " + " ".join(f"{v:+.2f}" for v in rows[:, j]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--gateway-url", required=True)
    ap.add_argument("--zero-state", action="store_true")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 9, 11])
    ap.add_argument("--samples", type=int, default=2, help="requests per condition (the model samples)")
    ap.add_argument("--horizon", type=int, default=50, help="execute_chunk_size to ask for (pi07 horizon: 50)")
    ap.add_argument("--out", type=Path, default=None, help="JSON with every plan (default outputs/plan_probe_<ckpt>.json)")
    args = ap.parse_args()

    check_zero_state(args.root, args.zero_state)
    ds = LeRobotDataset(f"local/{args.root.name}", root=args.root)
    cams = [k.removeprefix("observation.images.") for k in ds.meta.features if k.startswith("observation.images.")]
    policy = PIPolicy(args.gateway_url, DatasetCfg(cams), pi_api_key(), execute_chunk_size=args.horizon)
    print("connecting to PI (the first connection can take minutes while the model starts) ...")
    meta = policy.connect()
    print(f"PI server: cameras {policy.cam_map}, action horizon {meta.action_horizon}")
    if args.zero_state:
        policy = ZeroState(policy, CHANNELS)

    ckpt = args.gateway_url.rstrip("/").rsplit("/", 1)[-1]
    out_path = args.out or Path("outputs") / f"plan_probe_{ckpt}.json"
    results = []
    print(f"\nchannels per step: {' '.join(CHANNELS)}; steps shown {SHOW} (15 Hz; the runner used ~4-8)")
    try:
        for e in args.episodes:
            ep = ds.meta.episodes[e]
            lo, hi = ep["dataset_from_index"], ep["dataset_to_index"]
            prompt = ep["tasks"][0]
            actions = np.stack([ds.hf_dataset[i]["action"] for i in range(lo, hi)]).astype(float)
            for name, (k, ch) in moments(actions, ds.fps).items():
                x = ds[lo + k]
                obs = {c: to_rgb_uint8(x[f"observation.images.{c}"]) for c in cams}
                state = actions[k - 1] if k > 0 else np.zeros(len(CHANNELS))   # previous command (= state column)
                conds = {"state": state}
                if not args.zero_state:
                    s0 = state.copy()
                    s0[IDX[ch]] = 0.0
                    conds["state-0"] = s0
                true = actions[k:k + args.horizon]
                print(f"\n=== episode {e} {name} (t={k / ds.fps:.1f} s, state {' '.join(f'{v:+.2f}' for v in state)})")
                shown = [i for i in SHOW if i < len(true)]
                show("demo", true[shown])
                for cname, st in conds.items():
                    for s in range(args.samples):
                        o = dict(obs)
                        o.update({f"{j}.cmd": float(v) for j, v in zip(CHANNELS, st)})
                        policy.reset()
                        t0 = time.monotonic()
                        plan = policy.infer(o, prompt)
                        ms = 1e3 * (time.monotonic() - t0)
                        sh = [i for i in SHOW if i < len(plan)]
                        show(f"{cname} #{s} {ms:.0f}ms", plan[sh])
                        results.append({"episode": e, "moment": name, "t_s": k / ds.fps, "cond": cname, "sample": s,
                                        "state": st.tolist(), "plan": plan.tolist(), "demo": true.tolist(), "ms": ms})
    finally:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"gateway_url": args.gateway_url, "dataset": str(args.root),
                                        "channels": list(CHANNELS), "results": results}))
        print(f"\n{len(results)} plans -> {out_path}")
        if hasattr(policy, "close"):
            policy.close()


if __name__ == "__main__":
    main()
