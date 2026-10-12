"""Benchmark classical deinterleavers on local TSRD under the official TDC protocol.

Protocol (matches examples/identity_model.ipynb): every pulse train is cut into
non-overlapping 1,024-pulse windows, windows with fewer than 2 emitters are
dropped, each window is clustered on its own and scored with the official
``evaluate_model_on_pulse_train`` (default_label=-1), and the leaderboard number
is the median over windows.

Stages:
  stats  - per-feature transform statistics from a sample of TRAIN files (no labels)
  tune   - grid search per method on a VALIDATION window sample, V-measure only
  test   - full official metrics on TEST windows (all scan, a seeded stare sample)

Labels are read only to score; no method sees them.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.metrics import v_measure_score
from sklearn.neighbors import NearestNeighbors

from turing_deinterleaving_challenge.models.evaluate import evaluate_model_on_pulse_train
from turing_deinterleaving_challenge.models.model import Deinterleaver, IdentityModel

WINDOW = 1024
MIN_EMITTERS = 2
SPLIT_DIRS = {
    ("scan", "train"): "scan/train_scan",
    ("scan", "validation"): "scan/val_scan",
    ("scan", "test"): "scan/test_scan",
    ("stare", "train"): "stare/train_stare",
    ("stare", "validation"): "stare/val_stare",
    ("stare", "test"): "stare/test_stare",
}
TOA, CF, PW, AOA, AMP = range(5)


# --------------------------------------------------------------------------- data


def split_files(root: Path, mode: str, split: str) -> list[Path]:
    """Files in the order DeinterleavingChallengeDataset uses (by numeric suffix)."""

    files = (root / SPLIT_DIRS[(mode, split)]).glob("*.h5")
    return sorted(files, key=lambda p: int(p.stem.split("_")[-1]))


def eligible_windows(path: Path) -> list[tuple[int, int]]:
    """(start, n_emitters) of every full window with >= MIN_EMITTERS emitters."""

    with h5py.File(path, "r") as f:
        labels = f["labels"][:].squeeze()
    out = []
    for w in range(labels.shape[0] // WINDOW):
        n = len(np.unique(labels[w * WINDOW : (w + 1) * WINDOW]))
        if n >= MIN_EMITTERS:
            out.append((w * WINDOW, n))
    return out


def window_index(root: Path, mode: str, split: str, workers: int) -> pd.DataFrame:
    files = split_files(root, mode, split)
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(eligible_windows, p): p for p in files}
        for fut in as_completed(futures):
            p = futures[fut]
            rows.extend({"file": str(p), "start": s, "n_emitters": n} for s, n in fut.result())
    order = {str(p): i for i, p in enumerate(files)}
    df = pd.DataFrame(rows)
    df["file_order"] = df.file.map(order)
    return df.sort_values(["file_order", "start"]).reset_index(drop=True)


# ----------------------------------------------------------------------- features


def transform(data: np.ndarray, kind: str, stats: dict | None) -> np.ndarray:
    """Feature view for one window. Uses only the window and train-split stats."""

    x = np.asarray(data, dtype=np.float64)
    if kind == "raw":
        return x
    if kind == "raw_no_toa":
        return x[:, [CF, PW, AOA, AMP]]
    if kind == "zscore_window":
        sd = x.std(0)
        return (x - x.mean(0)) / np.where(sd > 0, sd, 1.0)
    if kind == "physical":
        # log CF / PW (both span orders of magnitude), amplitude, AoA on the circle.
        # Scaled by TRAIN-split statistics so one unit means the same in every window.
        a = np.deg2rad(x[:, AOA])
        cols = np.column_stack(
            [np.log(np.clip(x[:, CF], 1e-3, None)), np.log(np.clip(x[:, PW], 1e-6, None)), x[:, AMP]]
        )
        cols = (cols - stats["mean"]) / stats["std"]
        return np.column_stack([cols, stats["aoa_weight"] * np.cos(a), stats["aoa_weight"] * np.sin(a)])
    raise ValueError(kind)


def train_stats(root: Path, mode: str, n_files: int, seed: int) -> dict:
    files = split_files(root, mode, "train")
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(files), size=min(n_files, len(files)), replace=False)
    chunks = []
    for i in pick:
        with h5py.File(files[i], "r") as f:
            n = f["data"].shape[0]
            if n == 0:
                continue
            idx = np.sort(rng.choice(n, size=min(n, 20_000), replace=False))
            chunks.append(f["data"][idx])
    x = np.concatenate(chunks).astype(np.float64)
    cols = np.column_stack([np.log(np.clip(x[:, CF], 1e-3, None)), np.log(np.clip(x[:, PW], 1e-6, None)), x[:, AMP]])
    return {
        "mean": cols.mean(0).tolist(),
        "std": cols.std(0).tolist(),
        "aoa_weight": 1.0,
        "n_files": len(chunks),
        "n_pulses": int(x.shape[0]),
    }


# ------------------------------------------------------------------------ methods


class FeatureClusterer(Deinterleaver):
    """Feature transform + density clustering, optionally absorbing noise."""

    def __init__(self, features: str, algo: str, params: dict, stats: dict | None, absorb_noise: bool = False):
        super().__init__(default_label=-1)
        self.features, self.algo, self.params = features, algo, dict(params)
        self.stats = None if stats is None else {k: np.asarray(v) if k in ("mean", "std") else v for k, v in stats.items()}
        self.absorb_noise = absorb_noise

    def __call__(self, data: np.ndarray) -> np.ndarray:
        x = transform(data, self.features, self.stats)
        if self.algo == "hdbscan":
            labels = HDBSCAN(copy=True, **self.params).fit_predict(x)
        elif self.algo == "dbscan":
            labels = DBSCAN(**self.params).fit_predict(x)
        else:
            raise ValueError(self.algo)
        if self.absorb_noise:
            labels = absorb_noise(x, labels)
        return labels


def absorb_noise(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Give each noise point the label of its nearest clustered point."""

    noise = labels == -1
    if not noise.any() or noise.all():
        return labels
    nn = NearestNeighbors(n_neighbors=1).fit(x[~noise])
    _, j = nn.kneighbors(x[noise])
    out = labels.copy()
    out[noise] = labels[~noise][j[:, 0]]
    return out


@dataclass(frozen=True)
class MethodSpec:
    name: str
    features: str
    algo: str
    grid: dict  # param -> candidate values; a single value means fixed
    absorb_noise: bool = False
    official: bool = False


METHODS = [
    MethodSpec("M0 official HDBSCAN (raw PDWs)", "raw", "hdbscan", {}, official=True),
    MethodSpec("M1 HDBSCAN, ToA dropped", "raw_no_toa", "hdbscan", {}),
    MethodSpec("M2 HDBSCAN, per-window z-score", "zscore_window", "hdbscan", {}),
    MethodSpec("M3 HDBSCAN, physical features", "physical", "hdbscan", {}),
    MethodSpec(
        "M4 HDBSCAN, physical + tuned",
        "physical",
        "hdbscan",
        {"min_cluster_size": [3, 5, 8, 12, 20], "min_samples": [1, 3, 5, None], "cluster_selection_epsilon": [0.0, 0.05, 0.1, 0.2]},
    ),
    MethodSpec(
        "M5 HDBSCAN, physical + tuned + noise absorbed",
        "physical",
        "hdbscan",
        {"min_cluster_size": [3, 5, 8, 12, 20], "min_samples": [1, 3, 5, None], "cluster_selection_epsilon": [0.0, 0.05, 0.1, 0.2]},
        absorb_noise=True,
    ),
    MethodSpec(
        "M6 DBSCAN, physical + tuned",
        "physical",
        "dbscan",
        {"eps": [0.02, 0.05, 0.1, 0.2, 0.3, 0.5], "min_samples": [2, 3, 5, 10]},
    ),
]


def build(spec: MethodSpec, params: dict, stats: dict | None) -> Deinterleaver:
    if spec.official:
        return IdentityModel(clusterer="hdbscan", cl_params={"copy": True}, default_label=-1)
    return FeatureClusterer(spec.features, spec.algo, params, stats, spec.absorb_noise)


def grid_points(grid: dict) -> list[dict]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in product(*(grid[k] for k in keys))]


# ---------------------------------------------------------------------- execution


def _load_windows(file: str, starts: list[int]):
    with h5py.File(file, "r") as f:
        for s in starts:
            yield s, f["data"][s : s + WINDOW], f["labels"][s : s + WINDOW].squeeze()


def tune_file(file: str, starts: list[int], spec: MethodSpec, points: list[dict], stats: dict | None):
    rows = []
    for s, data, labels in _load_windows(file, starts):
        for k, params in enumerate(points):
            pred = build(spec, params, stats)(data)
            rows.append((k, file, s, v_measure_score(labels, pred)))
    return rows


def test_file(file: str, starts: list[int], specs: list[tuple[MethodSpec, dict]], stats: dict | None):
    rows = []
    for s, data, labels in _load_windows(file, starts):
        for spec, params in specs:
            model = build(spec, params, stats)
            t0 = time.perf_counter()
            pred = model(data)
            elapsed = time.perf_counter() - t0
            scores = evaluate_model_on_pulse_train(model, data, labels)
            rows.append(
                {
                    "method": spec.name,
                    "file": file,
                    "start": s,
                    "n_emitters": len(np.unique(labels)),
                    "n_clusters": len(np.unique(pred[pred != -1])),
                    "noise_frac": float((pred == -1).mean()),
                    "seconds": elapsed,
                    **{k: float(v) for k, v in scores.items()},
                }
            )
    return rows


def run_by_file(fn, windows: pd.DataFrame, workers: int, *args):
    out = []
    groups = windows.groupby("file").start.apply(list)
    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(fn, f, starts, *args) for f, starts in groups.items()]
        for i, fut in enumerate(as_completed(futures), 1):
            out.extend(fut.result())
            if i % 25 == 0 or i == len(futures):
                print(f"    {i}/{len(futures)} files", flush=True)
    return out


def sample(windows: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or n >= len(windows):
        return windows
    return windows.sample(n=n, random_state=seed).sort_values(["file_order", "start"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("/srv/NAS_BACKUP_20260916/TuringData"))
    ap.add_argument("--out", type=Path, default=Path.home() / "tdc_runs" / "tsrd_benchmark")
    ap.add_argument("--modes", nargs="+", default=["scan", "stare"])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tune-windows", type=int, default=1500)
    ap.add_argument("--test-windows", type=int, nargs=2, default=[0, 20000], metavar=("SCAN", "STARE"),
                    help="test windows per mode; 0 = all")
    ap.add_argument("--methods", nargs="*", default=None, help="method name prefixes, e.g. M0 M4")
    args = ap.parse_args()

    specs = [m for m in METHODS if not args.methods or any(m.name.startswith(p) for p in args.methods)]
    for mode in args.modes:
        out = args.out / mode
        out.mkdir(parents=True, exist_ok=True)
        print(f"== {mode}", flush=True)

        stats_path = out / "train_stats.json"
        if not stats_path.exists():
            stats_path.write_text(json.dumps(train_stats(args.root, mode, n_files=200, seed=args.seed), indent=2))
        stats = json.loads(stats_path.read_text())

        idx = {}
        for split in ("validation", "test"):
            p = out / f"windows_{split}.csv"
            if not p.exists():
                print(f"  indexing {split} windows", flush=True)
                window_index(args.root, mode, split, args.workers).to_csv(p, index=False)
            idx[split] = pd.read_csv(p)
            print(f"  {split}: {len(idx[split])} eligible windows", flush=True)

        chosen_path = out / "chosen_params.json"
        chosen = json.loads(chosen_path.read_text()) if chosen_path.exists() else {}
        tune_set = sample(idx["validation"], args.tune_windows, args.seed)
        for spec in specs:
            if spec.name in chosen:
                continue
            points = grid_points(spec.grid)
            if len(points) == 1:
                chosen[spec.name] = {"params": points[0], "val_median_v": None, "grid_size": 1}
                continue
            print(f"  tuning {spec.name}: {len(points)} configs x {len(tune_set)} windows", flush=True)
            res = pd.DataFrame(run_by_file(tune_file, tune_set, args.workers, spec, points, stats),
                               columns=["k", "file", "start", "v"])
            med = res.groupby("k").v.median()
            best = int(med.idxmax())
            pd.DataFrame([{**points[k], "val_median_v": med[k]} for k in med.index]).to_csv(
                out / f"tuning_{spec.name.split()[0]}.csv", index=False)
            chosen[spec.name] = {"params": points[best], "val_median_v": float(med[best]), "grid_size": len(points)}
            chosen_path.write_text(json.dumps(chosen, indent=2))
        chosen_path.write_text(json.dumps(chosen, indent=2))

        n_test = args.test_windows[0 if mode == "scan" else 1] or None
        test_set = sample(idx["test"], n_test, args.seed)
        print(f"  testing {len(specs)} methods on {len(test_set)} of {len(idx['test'])} test windows", flush=True)
        rows = run_by_file(test_file, test_set, args.workers, [(s, chosen[s.name]["params"]) for s in specs], stats)
        per_window = pd.DataFrame(rows)
        tag = "_".join(s.name.split()[0] for s in specs)
        per_window.to_csv(out / f"test_per_window_{tag}.csv.gz", index=False)
        summary = per_window.drop(columns=["file", "start"]).groupby("method").median(numeric_only=True)
        summary["n_windows"] = per_window.groupby("method").size()
        summary.to_csv(out / f"test_summary_{tag}.csv")
        print(summary[["V-measure", "Homogeneity", "Completeness", "Adjusted Mutual Information", "n_windows"]].round(3), flush=True)


if __name__ == "__main__":
    main()
