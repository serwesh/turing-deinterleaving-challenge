"""Next steps on TSRD under the official TDC protocol: M7-M10 against the M6 control.

M6   control: DBSCAN on physical features (tsrd_benchmark.py), rerun here for paired scores
M7   DBSCAN on M6's physical features joined with a CAESAR checkpoint's pulse embedding
M8   M6 with a per-window radius: shrink eps when the window holds many clusters, or set
     it from the window's own nearest-neighbour distances
M9   M6 followed by a PRI split: a cluster holding two interleaved pulse trains with
     different PRIs is cut in two (ToA differences only; never splits sequential trains)
M9b  DBSCAN on physical features plus each pulse's gap to its nearest same-signature pulse
     (a per-pulse PRI estimate)
M10  M7 + M8 + M9 stacked, re-tuned jointly on a small grid around their tuned settings

Windows, train statistics, the validation tuning sample and the stare test sample are the
ones tsrd_benchmark.py wrote, so every score here pairs window-for-window with M0-M6.

Scoring: ``fast_scores`` reproduces the official ``evaluate_labels`` (its cluster-wise
MCC/F1 from a contingency table instead of one sklearn call per emitter-cluster pair);
the test stage checks it against the official scorer on the first --check windows and
stops on any difference above 1e-9.

Embeddings are computed on CPU inside each worker (the encoders are < 1 MB MLPs).
Run with ``PYTHONPATH=~/CAESAR`` so the CAESAR feature code and checkpoints load.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    v_measure_score,
)
from sklearn.neighbors import NearestNeighbors

from tsrd_benchmark import AMP, AOA, CF, PW, TOA, WINDOW, _load_windows, grid_points, sample, transform
from turing_deinterleaving_challenge.models.evaluate import evaluate_labels

CAESAR_RUNS = Path("/srv/DRDO_EXPERIMENTS/DEB/tsrd_local/runs")
CHECKPOINTS = {
    "v4": CAESAR_RUNS / "tsrd_p1v4/tsrd_p1v3__masked_dual_residual__soft_triple/pointwise"
    "/masked_dual_residual_soft_triple_type/checkpoints/model_masked_dual_residual_soft_triple_32d_19in.pt",
    "v5": CAESAR_RUNS / "tsrd_p1v5/tsrd_p1v5__residual__semi_hard/pointwise/residual_semi_hard_type"
    "/checkpoints/model_residual_semi_hard_16d_19in.pt",
    "v6": CAESAR_RUNS / "tsrd_p1v6/tsrd_p1v6__residual__semi_hard/pointwise/residual_semi_hard_type"
    "/checkpoints/model_residual_semi_hard_16d_19in.pt",
}
M6_PARAMS = {"eps": 0.5, "min_samples": 2}


# ------------------------------------------------------------------------ scoring


def _cluster_wise(true: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """Official cluster_wise_score for MCC and F1: min over emitters of the best cluster."""

    _, t = np.unique(true, return_inverse=True)
    _, c = np.unique(pred, return_inverse=True)
    n = float(len(true))
    tab = np.zeros((t.max() + 1, c.max() + 1))
    np.add.at(tab, (t, c), 1.0)
    nt, nc = tab.sum(1, keepdims=True), tab.sum(0, keepdims=True)
    tp, fp, fn = tab, nt - tab, nc - tab
    tn = n - tp - fp - fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros_like(tab), where=den > 0)
    f1 = 2 * tp / (nt + nc)
    return float(mcc.max(1).min()), float(f1.max(1).min())


def fast_scores(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """``evaluate_labels`` for default_label=-1 on TSRD (labels never negative, discount 1)."""

    mcc, f1 = _cluster_wise(true, pred)
    return {
        "Homogeneity": homogeneity_score(true, pred),
        "Completeness": completeness_score(true, pred),
        "V-measure": v_measure_score(true, pred),
        "Adjusted Rand Index": adjusted_rand_score(true, pred),
        "Adjusted Mutual Information": adjusted_mutual_info_score(true, pred),
        "MCC": mcc,
        "F1": f1,
        "discount": 1.0,
    }


# ----------------------------------------------------------------------- features

_ENCODERS: dict[str, tuple] = {}


def _encoder(name: str, path: str):
    if name not in _ENCODERS:
        import torch

        from caesar.repr.inference import InferenceContract
        from caesar.repr.utils.weight_manager import WeightManager

        torch.set_num_threads(1)
        with contextlib.redirect_stdout(io.StringIO()):
            model, ckpt = WeightManager().load_checkpoint(path, "cpu", allow_turing_measurement=True)
        _ENCODERS[name] = (model, InferenceContract.from_checkpoint(path, ckpt, allow_turing_measurement=True))
    return _ENCODERS[name]


def embedding(data: np.ndarray, name: str, path: str, cache: dict) -> np.ndarray:
    """Unit-length CAESAR embedding of one window, temporal features built inside the window.
    The engineered feature matrix is shared by checkpoints with the same feature contract."""

    from caesar.repr.inference import apply_persisted_scaler, build_feature_matrix, embed

    model, contract = _encoder(name, path)
    key = ("raw", tuple(contract.feature_names))
    if key not in cache:
        frame = pd.DataFrame(
            {
                "time_ms": data[:, TOA] / 1000.0,
                "Freq(MHz)": data[:, CF],
                "PW(µs)": data[:, PW],
                "Azimuth(º)": data[:, AOA],
                "Power(dBm)": data[:, AMP],
                "source_file": "window",
                "scenario_id": "window",
            }
        )
        with contextlib.redirect_stdout(io.StringIO()):
            cache[key] = build_feature_matrix(frame, contract)
    e = embed(model, apply_persisted_scaler(cache[key], contract.feature_names, contract.scaler_params))
    return e / np.clip(np.linalg.norm(e, axis=1, keepdims=True), 1e-9, None)


# ------------------------------------------------------------------------ methods


def dbscan(x: np.ndarray, eps: float, min_samples: int = 2) -> np.ndarray:
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(x)


def adaptive_eps(x: np.ndarray, p: dict, cache: dict) -> float:
    """Per-window radius. ``count``: shrink eps0 by (K / kref)^-gamma once DBSCAN at eps0
    finds K > kref clusters. ``knn``: c times the q-quantile of 2nd-neighbour distances."""

    if p["rule"] == "count":
        key = ("base", p["eps0"])
        if key not in cache:
            cache[key] = dbscan(x, p["eps0"])
        k = len(np.unique(cache[key][cache[key] != -1]))
        return p["eps0"] * min(1.0, (max(k, 1) / p["kref"]) ** -p["gamma"])
    if "knn" not in cache:
        d, _ = NearestNeighbors(n_neighbors=3).fit(x).kneighbors(x)
        cache["knn"] = d[:, 2]
    return p["c"] * float(np.quantile(cache["knn"], p["q"]))


def _periodic(t: np.ndarray, tol: float) -> tuple[float | None, np.ndarray]:
    """Dominant PRI of a sorted ToA run (peak of 1st-3rd order differences on a log grid of
    relative width tol) and the pulses that have a partner exactly one such PRI away."""

    n = len(t)
    d = np.concatenate([t[k:] - t[:-k] for k in (1, 2, 3) if n > k]) if n > 1 else np.empty(0)
    d = d[d > 1e-6]
    if d.size < 4:
        return None, np.zeros(n, bool)
    b = np.floor(np.log(d) / tol).astype(np.int64)
    vals, counts = np.unique(b, return_counts=True)
    lookup = dict(zip(vals.tolist(), counts.tolist()))
    smooth = np.array([lookup.get(v - 1, 0) + c + lookup.get(v + 1, 0) for v, c in zip(vals.tolist(), counts.tolist())])
    best = vals[int(np.argmax(smooth))]
    near = np.abs(b - best) <= 1
    p = float(np.exp(np.log(d[near]).mean()))
    lo, hi = p * (1 - tol), p * (1 + tol)
    fwd = np.searchsorted(t, t + hi, "right") > np.searchsorted(t, t + lo, "left")
    bwd = np.searchsorted(t, t - lo, "right") > np.searchsorted(t, t - hi, "left")
    return p, fwd | bwd


def _split_once(t: np.ndarray, p: dict) -> np.ndarray | None:
    """Mask of one pulse train inside a cluster sorted by ToA, or None to keep it whole."""

    if len(t) < 2 * p["min_size"]:
        return None
    p1, a = _periodic(t, p["tol"])
    if p1 is None or not p["min_frac"] <= a.mean() <= 1 - p["min_frac"]:
        return None
    rest = t[~a]
    if len(rest) < p["min_size"]:
        return None
    p2, b = _periodic(rest, p["tol"])
    if p2 is None or b.mean() < p["verify"] or abs(np.log(p2 / p1)) < 3 * p["tol"]:
        return None
    # Two trains that take turns (switch-dwell, a sliding PRI) are one emitter: require
    # the trains to run at the same time, not one after the other.
    ta = t[a]
    overlap = min(ta[-1], rest[-1]) - max(ta[0], rest[0])
    if overlap < 0.5 * min(ta[-1] - ta[0], rest[-1] - rest[0]):
        return None
    return a


def pri_split(toa: np.ndarray, labels: np.ndarray, p: dict) -> np.ndarray:
    """Split clusters (up to two levels) that hold interleaved trains with distinct PRIs."""

    if not p.get("on", True):
        return labels
    out = labels.copy()
    nxt = out.max() + 1 if (out >= 0).any() else 0
    queue = [(lab, 0) for lab in np.unique(out[out >= 0])]
    while queue:
        lab, depth = queue.pop()
        idx = np.flatnonzero(out == lab)
        order = idx[np.argsort(toa[idx], kind="stable")]
        a = _split_once(toa[order], p)
        if a is None:
            continue
        out[order[~a]] = nxt
        if depth + 1 < 2:
            queue += [(lab, depth + 1), (nxt, depth + 1)]
        nxt += 1
    return out


def signature_pri(phys: np.ndarray, toa: np.ndarray, radius: float) -> np.ndarray:
    """log of each pulse's ToA gap to the nearest pulse with an almost identical signature
    (within ``radius`` in physical-feature space): a per-pulse estimate of its emitter's PRI.
    CAESAR's own PRI_est is a rolling median over the interleaved stream, so it measures
    pulse density, not any one emitter's PRI."""

    near = NearestNeighbors(radius=radius).fit(phys).radius_neighbors(phys, return_distance=False)
    gap = np.full(len(toa), np.nan)
    for i, js in enumerate(near):
        d = np.abs(toa[js] - toa[i])
        d = d[d > 1e-6]
        if d.size:
            gap[i] = d.min()
    fill = np.nanmedian(gap) if np.isfinite(gap).any() else 1.0
    return np.log(np.where(np.isfinite(gap), gap, fill))


@dataclass
class Spec:
    name: str
    kind: str  # control | embed | adaptive | pri | stacked
    grid: list[dict] = field(default_factory=list)


def predict(kind: str, p: dict, data: np.ndarray, cache: dict, stats: dict, ckpts: dict[str, str]) -> np.ndarray:
    """One method/setting on one window. ``cache`` holds per-window work shared by settings."""

    if "phys" not in cache:
        cache["phys"] = transform(data, "physical", stats)
    phys = cache["phys"]
    if kind == "control":
        return dbscan(phys, **M6_PARAMS)
    if kind == "adaptive":
        return dbscan(phys, adaptive_eps(phys, p, cache))
    if kind == "pri":
        if "m6" not in cache:
            cache["m6"] = dbscan(phys, **M6_PARAMS)
        return pri_split(data[:, TOA], cache["m6"], p)
    if kind == "prifeat":
        if not p["w"]:
            return dbscan(phys, p["eps"])
        key = ("spri", p["radius"])
        if key not in cache:
            cache[key] = signature_pri(phys, data[:, TOA], p["radius"])
        return dbscan(np.column_stack([phys, p["w"] * cache[key]]), p["eps"])
    # embed / stacked: physical features joined with a weighted unit embedding
    emb = ("emb", p["ckpt"])
    if emb not in cache:
        cache[emb] = embedding(data, p["ckpt"], ckpts[p["ckpt"]], cache)
    key = ("z", p["ckpt"], p["w"])
    if key not in cache:
        cache[key] = np.hstack([phys, p["w"] * cache[emb]])
    z = cache[key]
    if kind == "embed":
        return dbscan(z, p["eps"])
    sub = cache.setdefault(("sub", p["ckpt"], p["w"]), {})
    labels = dbscan(z, adaptive_eps(z, {"rule": "count", **p}, sub)) if p["gamma"] else dbscan(z, p["eps0"])
    return pri_split(data[:, TOA], labels, p["pri"])


# ---------------------------------------------------------------------- execution


def tune_file(file, starts, kind, points, stats, ckpts):
    rows = []
    for s, data, labels in _load_windows(file, starts):
        cache: dict = {}
        for k, p in enumerate(points):
            rows.append((k, file, s, v_measure_score(labels, predict(kind, p, data, cache, stats, ckpts))))
    return rows


def test_file(file, starts, methods, stats, ckpts, check):
    rows = []
    for i, (s, data, labels) in enumerate(_load_windows(file, starts)):
        cache: dict = {}
        for name, kind, p in methods:
            t0 = time.perf_counter()
            pred = predict(kind, p, data, cache, stats, ckpts)
            elapsed = time.perf_counter() - t0
            scores = fast_scores(pred, labels)
            if i < check:
                ref = evaluate_labels(pred, labels)
                bad = {k: (scores[k], ref[k]) for k in ref if abs(scores[k] - ref[k]) > 1e-9}
                if bad:
                    raise AssertionError(f"fast scorer disagrees on {file}:{s} {name}: {bad}")
            rows.append(
                {
                    "method": name,
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


def run_by_file(fn, windows: pd.DataFrame, workers: int, *args, chunk: int = 0):
    """Jobs per file (or per ``chunk`` windows of a file, to balance big stare files)."""

    jobs = []
    for f, starts in windows.groupby("file").start.apply(list).items():
        step = chunk or len(starts)
        jobs += [(f, starts[i : i + step]) for i in range(0, len(starts), step)]
    out = []
    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(fn, f, st, *args) for f, st in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            out.extend(fut.result())
            if i % max(1, len(futures) // 10) == 0 or i == len(futures):
                print(f"    {i}/{len(futures)} jobs", flush=True)
    return out


def grids(chosen: dict, ckpt_names: list[str]) -> list[Spec]:
    pri = [{"on": False}] + [
        {"on": True, "tol": tol, "min_frac": mf, "verify": v, "min_size": 8}
        for tol in (0.003, 0.01, 0.03)
        for mf in (0.15, 0.3)
        for v in (0.5, 0.75)
    ]
    adaptive = (
        [{"rule": "count", "eps0": 0.5, "kref": 1, "gamma": 0.0}]
        + [
            {"rule": "count", "eps0": e0, "kref": kref, "gamma": g}
            for e0 in (0.5, 0.6, 0.75)
            for kref in (4, 8, 12)
            for g in (0.15, 0.3, 0.5)
        ]
        + [{"rule": "knn", "q": q, "c": c} for q in (0.5, 0.9) for c in (1.5, 2, 3, 4, 6)]
    )
    embed = grid_points({"ckpt": ckpt_names, "w": [0.25, 0.5, 0.75, 1.0, 1.5], "eps": [0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25]})
    specs = [
        Spec("M6 DBSCAN, physical + tuned (control)", "control", [{}]),
        Spec("M7 DBSCAN, physical + CAESAR embedding", "embed", embed),
        Spec("M8 DBSCAN, physical, adaptive radius", "adaptive", adaptive),
        Spec("M9 M6 + PRI split", "pri", pri),
        Spec("M9b DBSCAN, physical + signature PRI feature", "prifeat",
             [{"radius": 0.05, "w": 0.0, "eps": 0.5}]
             + grid_points({"radius": [0.05, 0.15], "w": [0.05, 0.1, 0.2, 0.4], "eps": [0.5, 0.6]})),
    ]
    m7, m9 = chosen.get(specs[1].name), chosen.get(specs[3].name)
    if m7 and m9:
        e = m7["params"]["eps"]
        stacked = [
            {"ckpt": m7["params"]["ckpt"], "w": m7["params"]["w"], "eps0": round(e * f, 3), "kref": kref, "gamma": g, "pri": pr}
            for f in (1.0, 1.2, 1.5)
            for kref in (4, 8, 12)
            for g in (0.0, 0.15, 0.3, 0.5)
            for pr in ({"on": False}, m9["params"])
            if not (g == 0.0 and kref != 4)
        ]
        specs.append(Spec("M10 M7 + M8 + M9 stacked", "stacked", stacked))
    return specs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", type=Path, default=Path.home() / "tdc_runs" / "tsrd_benchmark",
                    help="tsrd_benchmark.py output (windows, train stats)")
    ap.add_argument("--out", type=Path, default=Path.home() / "tdc_runs" / "tsrd_next")
    ap.add_argument("--modes", nargs="+", default=["scan", "stare"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tune-windows", type=int, default=1500)
    ap.add_argument("--test-windows", type=int, nargs=2, default=[0, 20000], metavar=("SCAN", "STARE"))
    ap.add_argument("--ckpt", action="append", default=[], metavar="NAME=PATH",
                    help="extra CAESAR checkpoint for M7/M10 (repeatable)")
    ap.add_argument("--methods", nargs="*", default=None, help="method name prefixes to test, e.g. M6 M10")
    ap.add_argument("--tag", default="", help="suffix for test output files")
    ap.add_argument("--check", type=int, default=3, help="windows per job checked against the official scorer")
    ap.add_argument("--chunk", type=int, default=0, help="max windows per test job (0 = whole file)")
    args = ap.parse_args()

    ckpts = {k: str(v) for k, v in CHECKPOINTS.items()}
    ckpts.update(dict(c.split("=", 1) for c in args.ckpt))
    for mode in args.modes:
        bench, out = args.bench / mode, args.out / mode
        out.mkdir(parents=True, exist_ok=True)
        print(f"== {mode}", flush=True)
        stats = json.loads((bench / "train_stats.json").read_text())
        stats = {k: np.asarray(v) if k in ("mean", "std") else v for k, v in stats.items()}
        idx = {s: pd.read_csv(bench / f"windows_{s}.csv") for s in ("validation", "test")}
        tune_set = sample(idx["validation"], args.tune_windows, args.seed)

        chosen_path = out / "chosen_params.json"
        chosen = json.loads(chosen_path.read_text()) if chosen_path.exists() else {}
        for _ in range(2):  # second pass builds M10 from the tuned M7 and M9
            for spec in grids(chosen, list(ckpts)):
                if spec.name in chosen:
                    continue
                if len(spec.grid) == 1:
                    chosen[spec.name] = {"kind": spec.kind, "params": spec.grid[0], "val_median_v": None, "grid_size": 1}
                    continue
                print(f"  tuning {spec.name}: {len(spec.grid)} configs x {len(tune_set)} windows", flush=True)
                res = pd.DataFrame(run_by_file(tune_file, tune_set, args.workers, spec.kind, spec.grid, stats, ckpts),
                                   columns=["k", "file", "start", "v"])
                med = res.groupby("k").v.median()
                best = int(med.idxmax())
                pd.DataFrame([{"params": json.dumps(spec.grid[k]), "val_median_v": med[k]} for k in med.index]).sort_values(
                    "val_median_v", ascending=False).to_csv(out / f"tuning_{spec.name.split()[0]}.csv", index=False)
                chosen[spec.name] = {"kind": spec.kind, "params": spec.grid[best], "val_median_v": float(med[best]),
                                     "grid_size": len(spec.grid)}
                chosen_path.write_text(json.dumps(chosen, indent=2))
                print(f"    best {spec.grid[best]} -> {med[best]:.4f}", flush=True)
        chosen_path.write_text(json.dumps(chosen, indent=2))

        names = [n for n in chosen if not args.methods or any(n.startswith(p) for p in args.methods)]
        n_test = args.test_windows[0 if mode == "scan" else 1] or None
        test_set = sample(idx["test"], n_test, args.seed)
        print(f"  testing {len(names)} methods on {len(test_set)} of {len(idx['test'])} test windows", flush=True)
        methods = [(n, chosen[n]["kind"], chosen[n]["params"]) for n in names]
        rows = run_by_file(test_file, test_set, args.workers, methods, stats, ckpts, args.check, chunk=args.chunk)
        per_window = pd.DataFrame(rows)
        tag = "_".join(n.split()[0] for n in names) + (f"_{args.tag}" if args.tag else "")
        per_window.to_csv(out / f"test_per_window_{tag}.csv.gz", index=False)
        summary = per_window.drop(columns=["file", "start"]).groupby("method").median(numeric_only=True)
        summary["n_windows"] = per_window.groupby("method").size()
        summary.to_csv(out / f"test_summary_{tag}.csv")
        print(summary[["V-measure", "Homogeneity", "Completeness", "MCC", "n_windows"]].round(3), flush=True)


if __name__ == "__main__":
    main()
