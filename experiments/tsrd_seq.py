"""Sequence encoders x metric losses on TSRD under the official TDC protocol.

The full Cartesian grid of CAESAR's sequential encoders {gru, bigru, lstm, bilstm} and
CAESAR's sequential losses {triplet (batch-all), ntxent, supcon, multi_similarity},
trained separately for each receiver mode.

Each encoder reads one 1,024-pulse window as a sequence and emits a 16-d embedding per
pulse. Training labels are emitter identities inside the window: CAESAR's native
sequential losses mine positives and negatives within each sequence, which is the
deinterleaving task itself. (CAESAR's own sequential pipeline trains on radar-type
labels, which merges same-type emitters.)

Per-pulse inputs: M6's physical features (log CF, log PW and amplitude scaled with train
statistics; AoA as cos/sin) plus the standardised log(1 + gap to the previous pulse, us).

Stages
  build  (tdc_env)      train-window index and feature tensors; the validation tuning
                        windows and test windows of tsrd_benchmark.py, as raw arrays
  train  (CAESAR venv)  one encoder/loss/mode on one GPU; keeps the best checkpoint on
                        held-out validation windows; embeds the tuning and test windows
  queue  (CAESAR venv)  the whole grid, --per-gpu jobs at a time on each listed GPU
  eval   (tdc_env)      HDBSCAN on the embedding alone and joined to the physical
                        features (min_cluster_size x min_samples as in M4), tuned on
                        the 1,500 validation windows (median V-measure), then scored
                        on the test windows

  PYTHONPATH=~/CAESAR ~/tdc_env/bin/python tsrd_seq.py build
  ~/CAESAR/.venv/bin/python tsrd_seq.py queue --gpus 0 1 --per-gpu 4
  PYTHONPATH=~/CAESAR ~/tdc_env/bin/python tsrd_seq.py eval
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

WINDOW = 1024
TOA, CF, PW, AOA, AMP = range(5)
MODELS = ("gru", "bigru", "lstm", "bilstm")
LOSSES = ("triplet", "ntxent", "supcon", "multi_similarity")
ROOT = Path("/srv/NAS_BACKUP_20260916/TuringData")
BENCH = Path.home() / "tdc_runs" / "tsrd_benchmark"
NEXT = Path.home() / "tdc_runs" / "tsrd_next"
OUT = Path.home() / "tdc_runs" / "tsrd_seq"
CAESAR = Path.home() / "CAESAR"


# ----------------------------------------------------------------------- features


def seq_features(raw: np.ndarray, stats: dict) -> np.ndarray:
    """[N, L, 5] raw PDWs -> [N, L, 6] encoder inputs (see module docstring)."""

    x = np.asarray(raw, dtype=np.float64)
    cols = np.stack(
        [np.log(np.clip(x[..., CF], 1e-3, None)), np.log(np.clip(x[..., PW], 1e-6, None)), x[..., AMP]], axis=-1
    )
    cols = (cols - np.asarray(stats["mean"])) / np.asarray(stats["std"])
    a = np.deg2rad(x[..., AOA])
    gap = np.diff(x[..., TOA], axis=-1, prepend=x[..., :1, TOA])
    g = (np.log1p(np.clip(gap, 0, None)) - stats["gap_mean"]) / stats["gap_std"]
    return np.concatenate(
        [cols, stats["aoa_weight"] * np.cos(a)[..., None], stats["aoa_weight"] * np.sin(a)[..., None], g[..., None]],
        axis=-1,
    ).astype(np.float32)


def relabel(labels: np.ndarray) -> np.ndarray:
    """Emitter ids -> 0..k-1 inside each window (losses only compare within a window)."""

    out = np.empty(labels.shape, dtype=np.int64)
    for i, row in enumerate(labels):
        out[i] = np.unique(row, return_inverse=True)[1]
    return out


# -------------------------------------------------------------------------- build


def _read(file: str, starts: list[int]):
    import h5py

    with h5py.File(file, "r") as f:
        data = np.stack([f["data"][s : s + WINDOW] for s in starts]).astype(np.float32)
        labels = np.stack([f["labels"][s : s + WINDOW].reshape(-1) for s in starts]).astype(np.int32)
    return file, starts, data, labels


def read_windows(windows, workers: int):
    from concurrent.futures import ProcessPoolExecutor

    pos = {(f, s): i for i, (f, s) in enumerate(zip(windows.file, windows.start))}
    raw = np.empty((len(windows), WINDOW, 5), np.float32)
    lab = np.empty((len(windows), WINDOW), np.int32)
    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(_read, f, st) for f, st in windows.groupby("file").start.apply(list).items()]
        for fut in futures:
            f, starts, data, labels = fut.result()
            for s, d, y in zip(starts, data, labels):
                raw[pos[(f, s)]], lab[pos[(f, s)]] = d, y
    return raw, lab


def build(args) -> None:
    import pandas as pd

    from tsrd_benchmark import sample, window_index

    for mode in args.modes:
        out = OUT / mode / "data"
        out.mkdir(parents=True, exist_ok=True)
        bench = BENCH / mode
        idx_train_path = out / "windows_train_index.csv"
        if not idx_train_path.exists():
            print(f"[{mode}] indexing train windows", flush=True)
            window_index(ROOT, mode, "train", args.workers).to_csv(idx_train_path, index=False)
        idx_train = pd.read_csv(idx_train_path)
        idx_val = pd.read_csv(bench / "windows_validation.csv")
        idx_test = pd.read_csv(bench / "windows_test.csv")

        tune = sample(idx_val, 1500, 0)  # the tuning windows of tsrd_benchmark / tsrd_next
        rest = idx_val.drop(tune.index)
        splits = {
            "train": sample(idx_train, args.train_windows, 0),
            "val": sample(rest, args.val_windows, 1),  # early stopping only; disjoint from tuning
            "tune": tune,
            "test": sample(idx_test, None if mode == "scan" else 20000, 0),
        }
        stats = json.loads((bench / "train_stats.json").read_text())
        for name, win in splits.items():
            if (out / f"raw_{name}.npy").exists():
                continue
            print(f"[{mode}] reading {len(win)} {name} windows", flush=True)
            raw, lab = read_windows(win, args.workers)
            win[["file", "start", "n_emitters"]].to_csv(out / f"windows_{name}.csv", index=False)
            np.save(out / f"lab_{name}.npy", lab)
            np.save(out / f"raw_{name}.npy", raw)
            if name == "train":
                gap = np.diff(raw[..., TOA], axis=-1, prepend=raw[..., :1, TOA])
                g = np.log1p(np.clip(gap, 0, None))
                stats.update(gap_mean=float(g.mean()), gap_std=float(g.std()))
                (out / "feature_stats.json").write_text(json.dumps(stats, indent=2))
        stats = json.loads((out / "feature_stats.json").read_text())
        for name in ("train", "val"):
            if not (out / f"feat_{name}.npy").exists():
                np.save(out / f"feat_{name}.npy", seq_features(np.load(out / f"raw_{name}.npy"), stats))
        print(f"[{mode}] data ready in {out}", flush=True)


# -------------------------------------------------------------------------- train


def run_dir(mode: str, model: str, loss: str) -> Path:
    return OUT / mode / "runs" / f"{model}__{loss}"


def train(args) -> None:
    import types

    import torch

    sys.path.insert(0, str(CAESAR))
    from applications.sequential.factory import resolve_model
    from applications.sequential.losses.factory import get_sequence_loss

    data, out = OUT / args.mode / "data", run_dir(args.mode, args.model, args.loss)
    out.mkdir(parents=True, exist_ok=True)
    (out / "error.txt").unlink(missing_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(f"cuda:{args.gpu}")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True

    stats = json.loads((data / "feature_stats.json").read_text())
    x_tr, y_tr = np.load(data / "feat_train.npy"), relabel(np.load(data / "lab_train.npy"))
    x_va, y_va = np.load(data / "feat_val.npy"), relabel(np.load(data / "lab_val.npy"))

    spec = resolve_model(args.model)
    opts = spec.parse_impl_opts([f"embed_dim={args.embed_dim}"])
    model = spec.build_model(in_dim=x_tr.shape[-1], opts=opts).to(dev)
    criterion = get_sequence_loss(
        args.loss, config=types.SimpleNamespace(),
        opts=types.SimpleNamespace(subsample=args.subsample, anchor_chunk=64, loss_sampling_scope="native"),
    ).to(dev)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + np.cos(np.pi * min(s, args.steps) / args.steps))
    )

    def batch(x, y, idx):
        xb = torch.from_numpy(np.ascontiguousarray(x[idx])).to(dev)
        yb = torch.from_numpy(np.ascontiguousarray(y[idx])).to(dev)
        return xb, yb, torch.ones(yb.shape, dtype=torch.bool, device=dev)

    def val_loss() -> float:
        model.eval()
        total = 0.0
        with torch.no_grad():
            for s in range(0, len(x_va), args.batch):
                xb, yb, m = batch(x_va, y_va, np.arange(s, min(s + args.batch, len(x_va))))
                total += float(criterion(model(xb, m), yb, m).loss) * len(yb)
        model.train()
        return total / len(x_va)

    history, best, best_step, bad, t0 = [], float("inf"), 0, 0, time.time()
    model.train()
    for step in range(1, args.steps + 1):
        xb, yb, m = batch(x_tr, y_tr, rng.choice(len(x_tr), args.batch, replace=False))
        out_loss = criterion(model(xb, m), yb, m).loss
        if not torch.isfinite(out_loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        optim.zero_grad(set_to_none=True)
        out_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        sched.step()
        if step % args.val_every == 0 or step == args.steps:
            v = val_loss()
            history.append({"step": step, "train_loss": float(out_loss), "val_loss": v, "seconds": time.time() - t0})
            print(f"  step {step}: train {float(out_loss):.4f} val {v:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if v < best - 1e-4:
                best, best_step, bad = v, step, 0
                torch.save(model.state_dict(), out / "best_state.pt")
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"  early stop at step {step}; best step {best_step}", flush=True)
                    break

    model.load_state_dict(torch.load(out / "best_state.pt", map_location=dev))
    model.eval()

    def embed(raw: np.ndarray) -> np.ndarray:
        res = np.empty((len(raw), WINDOW, args.embed_dim), np.float32)
        with torch.no_grad():
            for s in range(0, len(raw), 32):
                xb = torch.from_numpy(seq_features(raw[s : s + 32], stats)).to(dev)
                res[s : s + 32] = model(xb, torch.ones(xb.shape[:2], dtype=torch.bool, device=dev)).float().cpu().numpy()
        return res

    # Cosine-geometry losses cluster on unit vectors; Euclidean ones on a fixed global
    # scale (RMS norm over train windows) so one radius grid serves every run.
    normalized = bool(getattr(criterion, "requires_normalized", False))
    scale = 1.0
    if not normalized:
        e = embed(np.load(data / "raw_train.npy", mmap_mode="r")[:256])
        scale = float(np.sqrt((e**2).sum(-1).mean()))
    for name in ("tune", "test"):
        e = embed(np.load(data / f"raw_{name}.npy", mmap_mode="r"))
        e = e / np.clip(np.linalg.norm(e, axis=-1, keepdims=True), 1e-9, None) if normalized else e / scale
        np.save(out / f"emb_{name}.npy", e.astype(np.float16))
    meta = {
        "mode": args.mode, "model": args.model, "loss": args.loss, "canonical_loss": criterion.name,
        "requires_normalized": normalized, "scale": scale, "best_step": best_step, "best_val_loss": best,
        "steps_run": history[-1]["step"], "train_seconds": history[-1]["seconds"],
        "params": int(sum(p.numel() for p in model.parameters())), "opts": {k: v for k, v in vars(opts).items()},
        "args": {k: v for k, v in vars(args).items() if k != "func"}, "history": history,
        "loss_provenance": criterion.provenance(),
    }
    torch.save({"state_dict": model.state_dict(), "meta": meta, "feature_stats": stats}, out / "checkpoint.pt")
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"  done: best val {best:.4f} at step {best_step}", flush=True)


def queue(args) -> None:
    jobs = [(mode, m, l) for mode in args.modes for m in MODELS for l in LOSSES]
    pending = [j for j in jobs if not (run_dir(*j) / "meta.json").exists()]
    running: dict[int, list] = {g: [] for g in args.gpus}
    print(f"{len(pending)} of {len(jobs)} runs to train on GPUs {args.gpus}", flush=True)
    while pending or any(running.values()):
        for gpu, procs in running.items():
            for job, proc, log in list(procs):
                if proc.poll() is not None:
                    log.close()
                    status = "ok" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
                    print(f"{time.strftime('%H:%M:%S')} gpu{gpu} {'/'.join(job)} {status}", flush=True)
                    procs.remove((job, proc, log))
            while pending and len(procs) < args.per_gpu:
                job = pending.pop(0)
                d = run_dir(*job)
                d.mkdir(parents=True, exist_ok=True)
                log = open(d / "train.log", "w")
                cmd = [sys.executable, "-u", __file__, "train", "--mode", job[0], "--model", job[1], "--loss", job[2],
                       "--gpu", str(gpu), "--steps", str(args.steps)]
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                        env={**os.environ, "OMP_NUM_THREADS": "2",
                                             "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
                procs.append((job, proc, log))
                print(f"{time.strftime('%H:%M:%S')} gpu{gpu} start {'/'.join(job)}", flush=True)
        time.sleep(20)
    print("queue finished", flush=True)


# ------------------------------------------------------- CAESAR pointwise checkpoints

_CK: dict = {}


def _ck_init(path: str) -> None:
    import contextlib
    import io

    import torch

    sys.path.insert(0, str(CAESAR))
    from caesar.repr.inference import InferenceContract

    torch.set_num_threads(1)
    with contextlib.redirect_stdout(io.StringIO()):
        ck = torch.load(path, map_location="cpu", weights_only=False)
    _CK["contract"] = InferenceContract.from_checkpoint(path, ck, allow_turing_measurement=True)


def _ck_features(raw: np.ndarray) -> np.ndarray:
    """CAESAR's own 19 inputs for one window (temporal features built inside the window)."""

    import contextlib
    import io

    import pandas as pd
    from caesar.repr.inference import apply_persisted_scaler, build_feature_matrix

    c = _CK["contract"]
    frame = pd.DataFrame({"time_ms": raw[:, TOA] / 1000.0, "Freq(MHz)": raw[:, CF], "PW(µs)": raw[:, PW],
                          "Azimuth(º)": raw[:, AOA], "Power(dBm)": raw[:, AMP],
                          "source_file": "window", "scenario_id": "window"})
    with contextlib.redirect_stdout(io.StringIO()):
        x = build_feature_matrix(frame, c)
    return apply_persisted_scaler(x, c.feature_names, c.scaler_params).astype(np.float32)


def embed_caesar(args) -> None:
    """Embed the tuning and test windows with a CAESAR pointwise checkpoint so that
    ``eval`` scores it exactly like a grid run (unit-length embeddings, as in M7)."""

    import contextlib
    import io
    from multiprocessing import Pool

    import torch

    sys.path.insert(0, str(CAESAR))
    from caesar.repr.utils.weight_manager import WeightManager

    out = OUT / args.mode / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    dev = f"cuda:{args.gpu}"
    with contextlib.redirect_stdout(io.StringIO()):
        model, ck = WeightManager().load_checkpoint(args.checkpoint, dev, allow_turing_measurement=True)
    model.eval()
    data = OUT / args.mode / "data"
    with Pool(args.workers, initializer=_ck_init, initargs=(args.checkpoint,)) as pool:
        for split in ("tune", "test"):
            raw = np.load(data / f"raw_{split}.npy", mmap_mode="r")
            emb = np.empty((len(raw), WINDOW, int(ck["emb_dim"])), np.float16)
            for s in range(0, len(raw), 256):
                feats = pool.map(_ck_features, [np.asarray(w) for w in raw[s : s + 256]])
                x = torch.from_numpy(np.concatenate(feats)).to(dev)
                with torch.no_grad():
                    e = model(x).float()
                e = torch.nn.functional.normalize(e, dim=-1).cpu().numpy().reshape(len(feats), WINDOW, -1)
                emb[s : s + len(feats)] = e.astype(np.float16)
            np.save(out / f"emb_{split}.npy", emb)
            print(f"  {split}: {len(raw)} windows embedded", flush=True)
    meta = {"mode": args.mode, "model": args.label_model, "loss": args.label_loss, "checkpoint": args.checkpoint,
            "requires_normalized": True, "best_step": ck.get("best_epoch"), "train_seconds": None,
            "params": int(sum(p.numel() for p in model.parameters())),
            "identity_target": (ck.get("dataset_metadata") or {}).get("identity_target")}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


# --------------------------------------------------------------------------- eval

HDBSCAN_GRID = [(mcs, ms) for mcs in (3, 5, 8, 12, 20) for ms in (1, 3, 5, None)]  # as M4
JOIN_WEIGHTS = (0.5, 1.0)


def _configs():
    return [("alone", None, mcs, ms) for mcs, ms in HDBSCAN_GRID] + [
        ("joined", w, mcs, ms) for w in JOIN_WEIGHTS for mcs, ms in HDBSCAN_GRID
    ]


def _cluster(phys, emb, variant, w, mcs, ms):
    from sklearn.cluster import HDBSCAN

    z = emb if variant == "alone" else np.hstack([phys, w * emb])
    return HDBSCAN(min_cluster_size=mcs, min_samples=ms, copy=True).fit_predict(z)


def _eval_job(mode: str, run: str, split: str, lo: int, hi: int, configs: list, test: bool):
    from sklearn.metrics import v_measure_score

    from tsrd_benchmark import transform
    from tsrd_next import fast_scores

    data = OUT / mode / "data"
    stats = json.loads((BENCH / mode / "train_stats.json").read_text())
    stats = {k: np.asarray(v) if k in ("mean", "std") else v for k, v in stats.items()}
    raw = np.load(data / f"raw_{split}.npy", mmap_mode="r")
    lab = np.load(data / f"lab_{split}.npy", mmap_mode="r")
    emb = np.load(OUT / mode / "runs" / run / f"emb_{split}.npy", mmap_mode="r")
    rows = []
    for i in range(lo, hi):
        phys = transform(np.asarray(raw[i]), "physical", stats)
        e, y = np.asarray(emb[i], dtype=np.float64), np.asarray(lab[i])
        for k, (variant, w, mcs, ms) in enumerate(configs):
            pred = _cluster(phys, e, variant, w, mcs, ms)
            if test:
                rows.append({"run": run, "variant": variant, "i": i, "n_emitters": len(np.unique(y)),
                             "n_clusters": len(np.unique(pred[pred != -1])), "noise_frac": float((pred == -1).mean()),
                             **fast_scores(pred, y)})
            else:
                rows.append((run, k, i, v_measure_score(y, pred)))
    return rows


def evaluate(args) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    import pandas as pd

    for mode in args.modes:
        out = OUT / mode / "eval"
        out.mkdir(parents=True, exist_ok=True)
        runs_dir = OUT / mode / "runs"
        runs = sorted(p.name for p in runs_dir.iterdir() if (p / "emb_test.npy").exists()
                      and (p / "meta.json").exists()) if runs_dir.exists() else []
        if not runs:
            continue
        chosen_path = out / "chosen.json"
        chosen = json.loads(chosen_path.read_text()) if chosen_path.exists() else {}
        todo = [r for r in runs if r not in chosen]
        configs = _configs()
        n_tune = len(np.load(OUT / mode / "data" / "lab_tune.npy", mmap_mode="r"))
        if todo:
            print(f"[{mode}] tuning {len(todo)} runs x {len(configs)} configs x {n_tune} windows", flush=True)
            res = []
            with ProcessPoolExecutor(args.workers) as pool:
                futs = [pool.submit(_eval_job, mode, r, "tune", lo, min(lo + 50, n_tune), configs, False)
                        for r in todo for lo in range(0, n_tune, 50)]
                for fut in as_completed(futs):
                    res.extend(fut.result())
            res = pd.DataFrame(res, columns=["run", "k", "i", "v"])
            med = res.groupby(["run", "k"]).v.median().reset_index()
            for r in todo:
                m = med[med.run == r].set_index("k").v
                pick = {}
                for variant in ("alone", "joined"):
                    ks = [k for k, c in enumerate(configs) if c[0] == variant]
                    best = max(ks, key=lambda k: m[k])
                    _, w, mcs, ms = configs[best]
                    pick[variant] = {"w": w, "min_cluster_size": mcs, "min_samples": ms, "val_median_v": float(m[best])}
                chosen[r] = pick
            chosen_path.write_text(json.dumps(chosen, indent=2))
            med.to_csv(out / "tuning_all.csv", mode="a", header=not (out / "tuning_all.csv").exists(), index=False)

        def tested(r: str) -> bool:  # files written before noise share was recorded are redone
            f = out / f"test_{r}.csv.gz"
            return f.exists() and "noise_frac" in pd.read_csv(f, nrows=0).columns

        todo = [r for r in runs if not tested(r)]
        n_test = len(np.load(OUT / mode / "data" / "lab_test.npy", mmap_mode="r"))
        if todo:
            print(f"[{mode}] testing {len(todo)} runs on {n_test} windows", flush=True)
            with ProcessPoolExecutor(args.workers) as pool:
                futs = {}
                for r in todo:
                    cfg = [(v, chosen[r][v]["w"], chosen[r][v]["min_cluster_size"], chosen[r][v]["min_samples"])
                           for v in ("alone", "joined")]
                    for lo in range(0, n_test, 400):
                        futs[pool.submit(_eval_job, mode, r, "test", lo, min(lo + 400, n_test), cfg, True)] = r
                rows: dict[str, list] = {r: [] for r in todo}
                left = {r: sum(1 for v in futs.values() if v == r) for r in todo}
                for fut in as_completed(futs):
                    r = futs[fut]
                    rows[r].extend(fut.result())
                    left[r] -= 1
                    if left[r] == 0:
                        pd.DataFrame(rows.pop(r)).to_csv(out / f"test_{r}.csv.gz", index=False)
                        print(f"  {r} tested", flush=True)
        summarize(mode)


def summarize(mode: str) -> None:
    import pandas as pd

    out = OUT / mode / "eval"
    chosen = json.loads((out / "chosen.json").read_text())
    win = pd.read_csv(OUT / mode / "data" / "windows_test.csv")
    rows = []
    for f in sorted(out.glob("test_*.csv.gz")):
        d = pd.read_csv(f)
        run = d.run.iloc[0]
        meta = json.loads((OUT / mode / "runs" / run / "meta.json").read_text())
        for variant, g in d.groupby("variant"):
            rows.append({
                "model": meta["model"], "loss": meta["loss"], "variant": variant,
                "val_median_v": chosen[run][variant]["val_median_v"],
                **{k: g[k].median() for k in ("V-measure", "Homogeneity", "Completeness", "Adjusted Rand Index",
                                             "Adjusted Mutual Information", "MCC", "F1", "noise_frac", "n_clusters")
                   if k in g},
                "mean_v": g["V-measure"].mean(), "n_windows": len(g), "best_step": meta["best_step"],
                "train_seconds": meta["train_seconds"], "params": meta["params"],
                "min_cluster_size": chosen[run][variant]["min_cluster_size"],
                "min_samples": chosen[run][variant]["min_samples"], "w": chosen[run][variant]["w"],
            })
    if not rows:
        return
    s = pd.DataFrame(rows).sort_values("V-measure", ascending=False)
    s.to_csv(out / "summary.csv", index=False)
    print(s[["model", "loss", "variant", "val_median_v", "V-measure", "Homogeneity", "Completeness", "MCC"]]
          .round(4).to_string(index=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--modes", nargs="+", default=["scan", "stare"])
    b.add_argument("--workers", type=int, default=32)
    b.add_argument("--train-windows", type=int, default=20000)
    b.add_argument("--val-windows", type=int, default=1000)
    b.set_defaults(func=build)
    t = sub.add_parser("train")
    t.add_argument("--mode", required=True)
    t.add_argument("--model", required=True, choices=MODELS)
    t.add_argument("--loss", required=True, choices=LOSSES)
    t.add_argument("--gpu", type=int, default=0)
    t.add_argument("--steps", type=int, default=4000)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--embed-dim", type=int, default=16)
    t.add_argument("--subsample", type=int, default=256,
                   help="pulses sampled per window for the loss, the same for every loss (0 = all; batch-all triplets on all 1,024 run out of GPU memory)")
    t.add_argument("--val-every", type=int, default=250)
    t.add_argument("--patience", type=int, default=6, help="validation checks without improvement")
    t.add_argument("--seed", type=int, default=11)
    t.set_defaults(func=train)
    q = sub.add_parser("queue")
    q.add_argument("--modes", nargs="+", default=["scan", "stare"])
    q.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    q.add_argument("--per-gpu", type=int, default=4)
    q.add_argument("--steps", type=int, default=4000)
    q.set_defaults(func=queue)
    c = sub.add_parser("embed-caesar")
    c.add_argument("--mode", required=True)
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--name", required=True, help="run directory name")
    c.add_argument("--label-model", default="caesar_residual")
    c.add_argument("--label-loss", default="semi_hard")
    c.add_argument("--gpu", type=int, default=0)
    c.add_argument("--workers", type=int, default=16)
    c.set_defaults(func=embed_caesar)
    e = sub.add_parser("eval")
    e.add_argument("--modes", nargs="+", default=["scan", "stare"])
    e.add_argument("--workers", type=int, default=32)
    e.set_defaults(func=evaluate)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
