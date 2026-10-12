"""Copy a LeRobot v3 dataset with observation.state set to 0.0 on every frame (for training without the state input),
or to small i.i.d. Gaussian noise (--noise SIGMA) for trainers that choke on a zero-variance feature.

    uv run scripts/zero_state.py data/pilot-best7 data/pilot-best7-nostate
    uv run scripts/zero_state.py data/pilot-best7 data/pilot-best7-nostate --overwrite
    uv run scripts/zero_state.py data/s2-keep18 data/s2-keep18-noise --noise 0.01 [--seed 0]

Why: our observation.state is the PREVIOUS COMMAND (the excavator has no joint sensors), so a model can learn the
shortcut "keep doing what you were doing" and ignore the cameras. A model trained on the copy gets no state signal
and has to act from the images; run it with --zero-state (scripts/control_loop.py, scripts/pi_replay.py), which
feeds it zeros as state. docs/zero_state.md.

Everything else is identical: actions, timestamps and indices (the data parquet is rewritten row group by row group
with only the observation.state column replaced), videos (hard-linked; copied if the destination is on another
filesystem; never symlinked), tasks, info.json, crops.json, pi_sources.json.

Zeros (default): the stats are the true stats of an all-zero feature: meta/stats.json and the per-episode stats in
meta/episodes get min = max = mean = std = q* = 0 (count kept). lerobot 0.6.1's normalizer turns that into a
constant, finite model input for an exactly-zero state (MEAN_STD: (0 - 0) / (0 + 1e-8) = 0 for SmolVLA; QUANTILES /
MIN_MAX: the 0-width range gets eps -> -1 for pi0.5), in float32 and bfloat16. A NON-zero state would be scaled by
~1e8, hence the marker meta/zero_state.json, which control_loop.py and pi_replay.py use to require --zero-state.

--noise SIGMA: state = i.i.d. N(0, SIGMA^2) float32 per frame and channel, drawn from --seed indexed by the global
frame index (reproducible, independent of the actions). The per-episode stats are computed from those values with
lerobot 0.6.1's compute_episode_stats and meta/stats.json is their aggregate_stats, as lerobot does. The marker gets
"train_noise_sigma": at inference the model is sent zeros (the noise mean), so --zero-state works unchanged; a zero
state then normalizes to ~0 (MEAN_STD: -mean/std; QUANTILES: 2(0 - q01)/(q99 - q01) - 1).

The copy is built in a temporary folder next to the destination and renamed into place, then loaded back with
LeRobotDataset and checked against the source (length, fps, features, tasks, the new state and its stats, other
columns and stats equal, a sample of decoded frames byte-identical).
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets import LeRobotDataset
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.processor.normalize_processor import NormalizerProcessorStep

from act.policies import ZERO_STATE_MARKER

ZEROED = ("observation.state",)
KEY = ZEROED[0]


def make_noise(total_frames, dim, sigma, seed):
    """(total_frames, dim) float32 N(0, sigma^2), row = global frame index; None for sigma 0 (zeros)."""
    if not sigma:
        return None
    return (np.random.default_rng(seed).standard_normal((total_frames, dim)) * sigma).astype(np.float32)


def new_state(index, dim, noise):
    """The replacement state for the rows with these global frame indices."""
    return np.zeros((len(index), dim), np.float32) if noise is None else noise[np.asarray(index)]


def state_stats(values, feature):
    """lerobot 0.6.1's per-episode stats of one (N, D) state array."""
    return compute_episode_stats({KEY: values}, {KEY: feature})[KEY]


def zero_data_parquet(src, dst, dim, noise):
    """Rewrite one data parquet with the state column replaced (zeros, or the noise rows of each frame's global index),
    one row group at a time (lerobot writes one row group per episode), schema + metadata unchanged."""
    f = pq.ParquetFile(src)
    compression = f.metadata.row_group(0).column(0).compression if f.metadata.num_row_groups else "snappy"
    with pq.ParquetWriter(dst, f.schema_arrow, compression=compression.lower()) as w:
        for g in range(f.metadata.num_row_groups):
            t = f.read_row_group(g)
            if KEY in t.column_names:
                i = t.column_names.index(KEY)
                vals = new_state(t.column("index").to_numpy(), dim, noise)
                t = t.set_column(i, t.field(i), pa.array(vals.tolist(), type=t.column(i).type))
            w.write_table(t)


def zero_stats(stats):
    """A stats dict {min, max, mean, std, count, q01..q99} -> the stats of an all-zero feature (count kept)."""
    return {k: (v if k == "count" else np.zeros_like(np.asarray(v, float)).tolist()) for k, v in stats.items()}


def episode_state_stats(episodes, dim, noise, feature):
    """Per-episode state stats {episode_index: stats} (numpy) computed from the new values, in episode order. Zeros:
    exact zeros (lerobot's histogram quantiles would give ~1e-14 for an all-zero feature)."""
    rows = episodes.select(["episode_index", "dataset_from_index", "dataset_to_index"]).to_pylist()
    if noise is None:
        n = lambda r: r["dataset_to_index"] - r["dataset_from_index"]
        return {r["episode_index"]: {**{k: np.zeros(dim) for k in ("min", "max", "mean", "std", "q01", "q10", "q50",
                                                                      "q90", "q99")}, "count": np.array([n(r)])}
                for r in rows}
    return {r["episode_index"]: state_stats(new_state(np.arange(r["dataset_from_index"], r["dataset_to_index"]), dim,
                                                      noise), feature) for r in rows}


def zero_episodes_parquet(src, dst, ep_stats):
    """meta/episodes parquet: the stats/observation.state/<stat> columns replaced by ep_stats (count kept)."""
    t = pq.read_table(src)
    eps = t.column("episode_index").to_pylist()
    for i, name in enumerate(t.column_names):
        parts = name.split("/")
        if len(parts) == 3 and parts[0] == "stats" and parts[1] == KEY and parts[2] != "count":
            col = t.column(i)
            vals = [np.asarray(ep_stats[e][parts[2]], float).tolist() for e in eps]
            t = t.set_column(i, t.field(i), pa.array(vals, type=col.type))
    compression = pq.read_metadata(src).row_group(0).column(0).compression
    pq.write_table(t, dst, compression=compression.lower())


def link_or_copy(src, dst):
    """Hard link (same filesystem), else a real copy. Never a symlink: the folder gets uploaded to a Modal volume."""
    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def build(src, tmp, dim, noise, feature):
    counts, global_stats = {}, None
    for d, _, files in os.walk(src):
        rel_dir = Path(d).relative_to(src)
        (tmp / rel_dir).mkdir(parents=True, exist_ok=True)
        for name in files:
            rel = rel_dir / name
            s, t = src / rel, tmp / rel
            if rel.parts[0] == "data" and name.endswith(".parquet"):
                zero_data_parquet(s, t, dim, noise)
                how = "rewritten"
            elif rel.parts[:2] == ("meta", "episodes") and name.endswith(".parquet"):
                ep_stats = episode_state_stats(pq.read_table(s), dim, noise, feature)
                zero_episodes_parquet(s, t, ep_stats)
                agg = aggregate_stats([{KEY: st} for st in ep_stats.values()])[KEY]
                global_stats = agg if global_stats is None else aggregate_stats([{KEY: global_stats}, {KEY: agg}])[KEY]
                how = "rewritten"
            elif rel == Path("meta/stats.json"):
                continue                     # written after all episode files are read
            elif name.endswith(".mp4"):
                how = link_or_copy(s, t)
            else:
                shutil.copy2(s, t)
                how = "copied"
            counts[how] = counts.get(how, 0) + 1
    stats = json.loads((src / "meta/stats.json").read_text())
    if noise is None:
        stats[KEY] = zero_stats(stats[KEY])
    else:
        stats[KEY] = {k: (stats[KEY]["count"] if k == "count" else np.asarray(v, float).tolist())
                      for k, v in global_stats.items()}
    (tmp / "meta/stats.json").write_text(json.dumps(stats, indent=4))
    counts["rewritten"] = counts.get("rewritten", 0) + 1
    return counts


def normalized_zero(stats):
    """What lerobot 0.6.1's normalizer makes of a zero state with these stats: {mode: (D,) array} for MEAN_STD
    (SmolVLA) and QUANTILES (pi0.5), float32."""
    st = {KEY: {k: np.asarray(v, np.float32) for k, v in stats.items() if k != "count"}}
    feats = {KEY: PolicyFeature(type=FeatureType.STATE, shape=(len(stats["mean"]),))}
    out = {}
    for mode in (NormalizationMode.MEAN_STD, NormalizationMode.QUANTILES):
        step = NormalizerProcessorStep(features=feats, norm_map={FeatureType.STATE: mode}, stats=st)
        x = torch.zeros((1, len(stats["mean"])), dtype=torch.float32)
        out[mode.value] = step._apply_transform(x, KEY, FeatureType.STATE)[0].numpy()
    return out


def arrow_columns(ds):
    """The dataset's frame table as {column: numpy array} (no video decode)."""
    table = ds.hf_dataset.data.table if hasattr(ds.hf_dataset.data, "table") else ds.hf_dataset.data
    return {c: np.array(table.column(c).to_pylist()) for c in table.column_names}


def verify(src, dst, n_frames, noise=None, sigma=0.0):
    """Load both with LeRobotDataset and compare. Returns a list of failures (empty = ok)."""
    a = LeRobotDataset(f"local/{src.name}", root=src)
    b = LeRobotDataset(f"local/{dst.name}", root=dst)
    bad = []
    check = lambda ok, what: ok or bad.append(what)
    check(len(a) == len(b), f"len {len(a)} != {len(b)}")
    check(a.fps == b.fps, f"fps {a.fps} != {b.fps}")
    check(a.meta.features == b.meta.features, "features differ")
    check(a.meta.total_episodes == b.meta.total_episodes, "episode count differs")
    check(a.meta.tasks.equals(b.meta.tasks), "tasks differ")
    ca, cb = arrow_columns(a), arrow_columns(b)
    check(sorted(ca) == sorted(cb), f"columns {sorted(ca)} != {sorted(cb)}")
    for c in ca:
        if c == KEY:
            want = new_state(cb["index"], ca[c].shape[1], noise)
            check(cb[c].shape == ca[c].shape and np.array_equal(cb[c].astype(np.float32), want),
                  f"{c} is not the intended zeros / noise")
            check(bool(ca[c].any()), f"{c} was already all zeros in the source (nothing to do?)")
        else:
            check(np.array_equal(ca[c], cb[c]), f"column {c} differs")
    feature = b.meta.features[KEY]
    want_eps = episode_state_stats(pq.read_table(next((dst / "meta/episodes").rglob("*.parquet"))), feature["shape"][0],
                                   noise, feature) if noise is not None else None
    for key, st in a.meta.stats.items():
        for k, v in st.items():
            if key != KEY or k == "count":
                check(np.array_equal(np.asarray(b.meta.stats[key][k]), np.asarray(v)), f"stats {key}/{k}")
            elif noise is None:
                check(not np.asarray(b.meta.stats[key][k]).any(), f"stats {key}/{k} not zero")
    if noise is not None:
        agg = aggregate_stats([{KEY: s} for s in want_eps.values()])[KEY]
        for k, v in agg.items():
            if k != "count":
                check(np.allclose(np.asarray(b.meta.stats[KEY][k]), v, rtol=0, atol=1e-12),
                      f"stats {KEY}/{k} != lerobot aggregate of the episode stats")
    for col in b.meta.episodes.column_names:
        parts = col.split("/")
        if len(parts) == 3 and parts[0] == "stats" and parts[1] == KEY and parts[2] != "count":
            got = np.asarray(list(b.meta.episodes[col]), float)
            if noise is None:
                check(not got.any(), f"episode {col} not zero")
            else:
                want = np.stack([np.asarray(want_eps[e][parts[2]], float) for e in b.meta.episodes["episode_index"]])
                check(np.allclose(got, want, rtol=0, atol=1e-12), f"episode {col} != lerobot compute_episode_stats")
        else:
            check(list(a.meta.episodes[col]) == list(b.meta.episodes[col]), f"episode column {col} differs")
    idx = sorted({0, len(a) - 1, *np.linspace(0, len(a) - 1, max(n_frames, 2)).round().astype(int).tolist()})
    for i in idx:
        xa, xb = a[i], b[i]
        for k, v in xa.items():
            if k == KEY:
                check(np.array_equal(xb[k].numpy(), new_state([xb["index"].item()], v.shape[0], noise)[0]),
                      f"frame {i}: {k} not the intended value")
            elif isinstance(v, torch.Tensor):
                check(torch.equal(v, xb[k]), f"frame {i}: {k} differs")
            else:
                check(v == xb[k], f"frame {i}: {k} differs")
    for p in src.rglob("*.mp4"):
        q = dst / p.relative_to(src)
        check(q.exists() and not q.is_symlink() and q.stat().st_size == p.stat().st_size, f"video {q}")
    print(f"verified: {len(b)} frames, {b.meta.total_episodes} episodes, {b.fps} fps; "
          f"{len(idx)} decoded frames compared ({', '.join(map(str, idx))})")
    s = cb[KEY].astype(np.float64)
    if noise is not None:
        act = cb["action"].astype(np.float64)
        corr = np.corrcoef(np.hstack([s, act]).T)[:s.shape[1], s.shape[1]:]
        prev = ca[KEY].astype(np.float64)
        corr_prev = np.corrcoef(np.hstack([s, prev]).T)[:s.shape[1], s.shape[1]:]
        print(f"noise state: mean {np.array2string(s.mean(0), precision=5)}, std {np.array2string(s.std(0), precision=5)}"
              f" (sigma {sigma}); max |corr| with action {np.abs(corr).max():.4f}, with the original state "
              f"{np.abs(corr_prev).max():.4f} (~{2 / np.sqrt(len(s)):.4f} expected by chance)")
        check(np.all(np.abs(s.std(0) / sigma - 1) < 0.05) and np.all(np.abs(s.mean(0)) < 0.05 * sigma),
              "noise mean / std off")
        check(np.abs(corr).max() < 5 / np.sqrt(len(s)), "noise correlates with the actions")
    nz = normalized_zero({k: np.asarray(v, float) for k, v in b.meta.stats[KEY].items()})
    for mode, v in nz.items():
        print(f"lerobot 0.6.1 normalizer, zero state, {mode}: {np.array2string(v, precision=4)}")
        check(np.all(np.isfinite(v)), f"{mode}: zero state normalizes to non-finite values")
        if noise is not None:
            check(np.all(np.abs(v) < 0.1), f"{mode}: zero state does not normalize to ~0")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=Path, help="source LeRobot v3 dataset root")
    ap.add_argument("dst", type=Path, help="new dataset root (must not exist unless --overwrite)")
    ap.add_argument("--noise", type=float, default=0.0, metavar="SIGMA",
                    help="state = i.i.d. N(0, SIGMA^2) instead of zeros (inference still sends zeros)")
    ap.add_argument("--seed", type=int, default=0, help="--noise: RNG seed")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing destination")
    ap.add_argument("--verify-frames", type=int, default=8, help="decoded frames compared with the source")
    args = ap.parse_args()

    src, dst = args.src.resolve(), args.dst.resolve()
    if not (np.isfinite(args.noise) and 0 <= args.noise <= 0.1):
        raise SystemExit("--noise: a sigma in 0..0.1 (joystick units; inference sends the mean, 0)")
    info = src / "meta" / "info.json"
    if not info.exists():
        raise SystemExit(f"{src}: no meta/info.json (not a LeRobot dataset)")
    info = json.loads(info.read_text())
    if info.get("codebase_version") != "v3.0":
        raise SystemExit(f"{src}: codebase_version {info.get('codebase_version')}, this script handles v3.0")
    if KEY not in info["features"]:
        raise SystemExit(f"{src}: no feature {KEY}")
    feature = info["features"][KEY]
    if feature["dtype"] != "float32" or len(feature["shape"]) != 1:
        raise SystemExit(f"{src}: {KEY} is {feature['dtype']} {feature['shape']}, expected float32 (D,)")
    if (src / ZERO_STATE_MARKER).exists():
        raise SystemExit(f"{src} already has {ZERO_STATE_MARKER}: it is a zero-state dataset")
    if dst == src or src in dst.parents or dst in src.parents:
        raise SystemExit("destination must be a separate folder (not the source, inside it, or containing it)")
    if dst.exists() and not args.overwrite:
        raise SystemExit(f"{dst} exists: pass --overwrite to replace it")

    dim = feature["shape"][0]
    noise = make_noise(info["total_frames"], dim, args.noise, args.seed)
    marker = {"zeroed": list(ZEROED), "value": 0.0}
    if noise is not None:
        marker.update(train_noise_sigma=args.noise, train_noise_seed=args.seed)
    marker.update(source=str(args.src), source_abs=str(src), created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  note=("observation.state is 0.0 on every frame" if noise is None else
                        f"observation.state is i.i.d. N(0, {args.noise}^2) noise, uncorrelated with the actions")
                  + "; run checkpoints trained on this with --zero-state (they are sent zeros)")
    tmp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        counts = build(src, tmp, dim, noise, feature)
        (tmp / ZERO_STATE_MARKER).write_text(json.dumps(marker, indent=1) + "\n")
        if dst.exists():
            shutil.rmtree(dst)
        tmp.rename(dst)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    print(f"{src} -> {dst}: files {counts}; marker {ZERO_STATE_MARKER}")

    bad = verify(src, dst, args.verify_frames, noise, args.noise)
    if bad:
        print("VERIFY FAILED:\n  " + "\n  ".join(bad))
        sys.exit(1)
    what = "all zeros" if noise is None else f"N(0, {args.noise}^2) noise with lerobot stats"
    print(f"ok: state {what}; actions, other columns, tasks, other stats and sampled frames identical")


if __name__ == "__main__":
    main()
