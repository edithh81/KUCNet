"""Per-batch benchmark for KUCNet: #messages, wall time, peak GPU memory.

Runs the original model (no architecture changes) for ``--num_batches``
forward[+backward] passes at ``--batch_size`` and records per-batch stats.
Useful for paper efficiency tables — the "biggest" (max) row across the
recorded batches is what you typically report.

Examples
--------
Train-mode (forward + backward + optimizer step), 100 batches of size 1::

    python benchmark.py --data_path data/last-fm/ --num_batches 100 --batch_size 1 --mode train

Eval-mode (forward only, no_grad), 100 batches of size 1::

    python benchmark.py --data_path data/amazon-book/ --num_batches 100 --batch_size 1 --mode test

Outputs are written to ``results/benchmark/<dataset>_<mode>_bs<BS>_<ts>.{jsonl,txt}``.
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch

from load_data import DataLoader
from models import KUCNet_trans
from utils import cal_bpr_loss


# ---------------------------------------------------------------------------
# Dataset → hyperparameters (mirrors train.py so we load the same model shape).
# ---------------------------------------------------------------------------
_DEFAULTS = dict(
    lr=0.0002, decay_rate=0.9938, lamb=0.0001,
    hidden_dim=48, attn_dim=5, n_layer=3, dropout=0.02, act="idd",
    n_batch=20, n_tbatch=20, K=50,
)

_DATASET_OPTS = {
    "new_alibaba-fashion": dict(lr=0.00005, decay_rate=0.999, lamb=0.0001, dropout=0.01, n_layer=5, K=50),
    "alibaba-fashion":     dict(lr=10 ** -6.5, decay_rate=0.998, lamb=0.00001, dropout=0.2, act="relu", n_layer=5, K=70),
    "last-fm":             dict(lr=0.0004, decay_rate=0.994, lamb=0.00014, n_layer=3, K=35),
    "new_last-fm":         dict(lr=0.0004, decay_rate=0.994, lamb=0.00014, n_layer=3, K=50),
    "amazon-book":         dict(lr=0.0012, decay_rate=0.994, lamb=0.000014, n_layer=3, K=120),
    "new_amazon-book":     dict(lr=0.0005, decay_rate=0.994, lamb=0.000014, dropout=0.01, n_layer=3, K=170),
    "Dis_5fold_item":      dict(lr=0.0005, decay_rate=0.994, lamb=0.00001, dropout=0.01, n_layer=5, K=35),
    "Dis_5fold_user":      dict(lr=0.001, decay_rate=0.994, lamb=0.00001, dropout=0.01, n_layer=3, K=550),
}


class _Opts:
    pass


def _dataset_name(data_path: str) -> str:
    parts = data_path.rstrip("/").split("/")
    return parts[-1]


def _make_opts(ds_name: str, user_K: int | None = None) -> _Opts:
    cfg = {**_DEFAULTS, **_DATASET_OPTS.get(ds_name, {})}
    if user_K is not None and ds_name not in _DATASET_OPTS:
        cfg["K"] = user_K
    opts = _Opts()
    for k, v in cfg.items():
        setattr(opts, k, v)
    return opts


def _agg(arr: np.ndarray) -> dict:
    return {
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
    }


def parse_args():
    p = argparse.ArgumentParser(description="KUCNet per-batch benchmark")
    p.add_argument("--data_path", type=str, required=True,
                   help="e.g. data/last-fm/ or data/amazon-book/")
    p.add_argument("--num_batches", type=int, default=100,
                   help="Number of batches to record (excludes 1 warmup batch)")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--mode", choices=["train", "test"], default="train",
                   help="'train' = forward+backward+step; 'test' = forward under no_grad")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--K", type=int, default=50,
                   help="Fallback K for datasets without a preset")
    p.add_argument("--ppr_device", type=str, default=None,
                   help="'cuda' or 'cpu'; default = cuda if available")
    p.add_argument("--out_dir", type=str, default="results/benchmark")
    return p.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    ds_name = _dataset_name(args.data_path)
    print(f"[bench] dataset={ds_name} mode={args.mode} bs={args.batch_size} "
          f"num_batches={args.num_batches} gpu={args.gpu}")

    loader = DataLoader(args.data_path)

    opts = _make_opts(ds_name, user_K=args.K)
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel
    opts.n_users = loader.n_users
    opts.n_items = loader.n_items
    opts.n_nodes = loader.n_nodes

    # Build model. ``get_ppr`` picks up ``args.ppr_device`` — default cuda if available.
    import models as _models_mod
    _orig_get_ppr = _models_mod.get_ppr

    def _patched_get_ppr(loader):
        return _orig_get_ppr(loader, device=args.ppr_device)

    _models_mod.get_ppr = _patched_get_ppr
    try:
        model = KUCNet_trans(opts, loader)
    finally:
        _models_mod.get_ppr = _orig_get_ppr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.lamb)

    # --- forward hooks: count edges consumed per GNN layer per forward pass ---
    edge_counts_this_forward: list[int] = []

    def _hook(module, inputs, output):
        # GNNLayer returns a 7-tuple ending in ``edges``; fail loudly if that changes.
        assert isinstance(output, tuple) and len(output) == 7, (
            f"GNNLayer forward must return a 7-tuple (got {type(output).__name__} "
            f"len={len(output) if isinstance(output, tuple) else 'n/a'}); "
            "update benchmark.py if the signature changed."
        )
        edges = output[6]
        edge_counts_this_forward.append(int(edges.size(0)))

    for layer in model.gnn_layers:
        layer.register_forward_hook(_hook)

    # -------------------------------- run loop --------------------------------
    if args.mode == "train":
        model.train()
        n_data = loader.n_train
        data_tag = "train"
    else:
        model.eval()
        n_data = loader.n_test
        data_tag = "test"

    bs = args.batch_size
    n_available = n_data // bs + (n_data % bs > 0)
    n_to_run = min(args.num_batches, n_available)
    if n_to_run < args.num_batches:
        print(f"[bench] only {n_available} batches available at bs={bs}, "
              f"capping num_batches to {n_to_run}")

    records = []

    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Warmup (excluded from stats). Also triggers lazy cudnn allocs.
    warmup_idx = np.arange(0, min(bs, n_data))
    if args.mode == "train":
        subs, rels, pos, neg = loader.get_batch(warmup_idx, data="train")
        edge_counts_this_forward.clear()
        _sync()
        scores = model(subs, rels)
        loss = cal_bpr_loss(opts.n_users, pos, neg, scores)
        loss.backward()
        optimizer.step()
        model.zero_grad()
    else:
        subs, rels, _objs = loader.get_batch(warmup_idx, data="test")
        edge_counts_this_forward.clear()
        with torch.no_grad():
            model(subs, rels, mode="test")
    _sync()
    print("[bench] warmup done")

    for i in range(n_to_run):
        start = (i * bs) % max(n_data, 1)
        end = min(n_data, start + bs)
        batch_idx = np.arange(start, end)

        if args.mode == "train":
            subs, rels, pos, neg = loader.get_batch(batch_idx, data="train")
        else:
            subs, rels, _objs = loader.get_batch(batch_idx, data="test")

        _sync()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        edge_counts_this_forward.clear()
        t0 = time.perf_counter()

        if args.mode == "train":
            model.zero_grad()
            scores = model(subs, rels)
            loss = cal_bpr_loss(opts.n_users, pos, neg, scores)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                model(subs, rels, mode="test")

        _sync()
        batch_time = time.perf_counter() - t0
        batch_msgs = int(sum(edge_counts_this_forward))
        gpu_peak_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                       if torch.cuda.is_available() else 0.0)

        rec = {
            "batch": i + 1,
            "messages": batch_msgs,
            "time_s": round(batch_time, 6),
            "gpu_peak_mb": round(gpu_peak_mb, 2),
        }
        records.append(rec)

        if (i + 1) % max(10, n_to_run // 10) == 0 or i == 0:
            print(f"[bench] {i+1:>4d}/{n_to_run}  msgs={batch_msgs:>9d}  "
                  f"time={batch_time*1000:.1f}ms  peak={gpu_peak_mb:.1f}MB")

    # -------------------------------- aggregate + write -----------------------
    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{ds_name}_{args.mode}_bs{bs}_{ts}"
    jsonl_path = os.path.join(args.out_dir, f"{run_id}.jsonl")
    summary_path = os.path.join(args.out_dir, f"{run_id}_summary.txt")

    config_blob = {
        "type": "config",
        "model": "kucnet-original",
        "dataset": ds_name,
        "mode": args.mode,
        "batch_size": int(bs),
        "num_batches": int(len(records)),
        "seed": int(args.seed),
        "gpu": int(args.gpu),
        "K": int(opts.K),
        "n_layer": int(opts.n_layer),
        "hidden_dim": int(opts.hidden_dim),
        "attn_dim": int(opts.attn_dim),
        "dropout": float(opts.dropout),
        "act": str(opts.act),
        "n_users": int(opts.n_users),
        "n_items": int(opts.n_items),
        "n_nodes": int(opts.n_nodes),
        "n_ent": int(opts.n_ent),
        "n_rel": int(opts.n_rel),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": (torch.cuda.get_device_name(args.gpu)
                     if torch.cuda.is_available() else None),
    }

    with open(jsonl_path, "w") as f:
        f.write(json.dumps(config_blob) + "\n")
        for r in records:
            f.write(json.dumps({"type": "batch", **r}) + "\n")

    msgs = np.array([r["messages"] for r in records], dtype=np.float64)
    times = np.array([r["time_s"] for r in records], dtype=np.float64)
    peaks = np.array([r["gpu_peak_mb"] for r in records], dtype=np.float64)

    summary = {
        "type": "summary",
        "messages": _agg(msgs),
        "time_s": _agg(times),
        "gpu_peak_mb": _agg(peaks),
        "n_batches": len(records),
        "paper_row": {
            "dataset": ds_name,
            "batch_size": bs,
            "messages_max": int(msgs.max()),
            "time_ms_max": round(float(times.max()) * 1000, 2),
            "gpu_peak_mb_max": round(float(peaks.max()), 2),
        },
    }
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(summary) + "\n")

    # Human-readable table
    bar = "=" * 78
    with open(summary_path, "w") as f:
        f.write(f"{bar}\n")
        f.write(f" KUCNet (original) benchmark — {ds_name}  [{args.mode}]\n")
        f.write(f" batch_size={bs}  num_batches={len(records)}  "
                f"seed={args.seed}  gpu={args.gpu}\n")
        f.write(f" torch={torch.__version__}  "
                f"cuda={torch.cuda.is_available()}  "
                f"gpu_name={config_blob['gpu_name']}\n")
        f.write(f" {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"{bar}\n\n")
        f.write(f" Config: K={opts.K}  n_layer={opts.n_layer}  "
                f"hidden={opts.hidden_dim}  attn={opts.attn_dim}  "
                f"dropout={opts.dropout}  act={opts.act}\n\n")
        f.write(f"{'Metric':<14}{'Max':>14}{'Mean':>14}{'Median':>14}"
                f"{'P90':>14}{'P99':>14}\n")
        f.write(f"{'-'*78}\n")
        f.write(f"{'messages':<14}"
                f"{msgs.max():>14.0f}{msgs.mean():>14.1f}"
                f"{np.median(msgs):>14.1f}"
                f"{np.percentile(msgs, 90):>14.1f}"
                f"{np.percentile(msgs, 99):>14.1f}\n")
        f.write(f"{'time_ms':<14}"
                f"{times.max()*1000:>14.2f}{times.mean()*1000:>14.2f}"
                f"{np.median(times)*1000:>14.2f}"
                f"{np.percentile(times, 90)*1000:>14.2f}"
                f"{np.percentile(times, 99)*1000:>14.2f}\n")
        f.write(f"{'gpu_peak_mb':<14}"
                f"{peaks.max():>14.2f}{peaks.mean():>14.2f}"
                f"{np.median(peaks):>14.2f}"
                f"{np.percentile(peaks, 90):>14.2f}"
                f"{np.percentile(peaks, 99):>14.2f}\n\n")
        pr = summary["paper_row"]
        f.write(f"{bar}\n")
        f.write(f" Paper row (biggest):\n"
                f"   dataset:      {pr['dataset']}\n"
                f"   batch_size:   {pr['batch_size']}\n"
                f"   #messages:    {pr['messages_max']}\n"
                f"   peak_mem_MB:  {pr['gpu_peak_mb_max']}\n"
                f"   time_ms:      {pr['time_ms_max']}\n")
        f.write(f"{bar}\n")

    print(f"\n[bench] wrote {jsonl_path}")
    print(f"[bench] wrote {summary_path}")
    print(f"[bench] biggest  msgs={int(msgs.max())}  "
          f"peak={peaks.max():.1f}MB  time={times.max()*1000:.1f}ms")


if __name__ == "__main__":
    main()
