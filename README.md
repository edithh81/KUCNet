# KUCNet

The code for our paper "Knowledge-Enhanced Recommendation with User-Centric Subgraph Network", accepted by 2024 IEEE 40th International Conference on Data Engineering (ICDE).

Link to paper: https://ieeexplore.ieee.org/document/10597876, https://arxiv.org/pdf/2403.14377

> This fork ports the codebase to modern PyTorch / Python (2.8 / 3.12) and adds a per-batch benchmarking script. The model architecture and training logic are unchanged.

## Environment Requirements

Tested with:

- Python 3.12
- PyTorch 2.8.0
- numpy >= 1.26
- scipy >= 1.11
- tqdm >= 4.62
- `torch_scatter` 2.1.2 (optional — a native-PyTorch fallback in `scatter_shim.py` is used if it's not installed)

`torchdrug` is **no longer required** — the only symbol we used (`variadic_topk`) is re-implemented in `torchdrug_shim.py` using core PyTorch.

Install with:

```
pip install -r requirements.txt
```

## Run the Codes

For traditional recommendation:

    python train.py --data_path=data/last-fm/

For new-item recommendation:

```
python train.py --data_path=data/new_last-fm/
```

For disease-gene prediction (new-item setting):

```
python train.py --data_path=data/Dis_5fold_item/
```

For disease-gene prediction (new-user setting):

```
python train.py --data_path=data/Dis_5fold_user/
```

Results land in `results/`.

## PPR computation device

`get_ppr()` in `ppr.py` now accepts an optional `device=` argument (defaults
to `cuda` if available, else `cpu`). This lets you run the power-iteration on
either GPU (fast) or CPU (when GPU memory is the bottleneck).

## Benchmarking (efficiency table)

`benchmark.py` records per-batch **#messages**, **wall time**, and **peak GPU
memory** — suitable for the efficiency table in the paper.

Usage:

```
# 100 training batches of size 1 on last-fm
python benchmark.py --data_path data/last-fm/ --num_batches 100 --batch_size 1 --mode train

# 100 inference batches of size 1 on amazon-book
python benchmark.py --data_path data/amazon-book/ --num_batches 100 --batch_size 1 --mode test
```

Outputs:

- `results/benchmark/<dataset>_<mode>_bs<BS>_<timestamp>.jsonl` — one JSON line per batch (`messages`, `time_s`, `gpu_peak_mb`), plus a header config line and a final summary line.
- `results/benchmark/<dataset>_<mode>_bs<BS>_<timestamp>_summary.txt` — human-readable table with `max / mean / median / p90 / p99`, and a "Paper row (biggest)" block you can copy directly.

Notes:

- One warmup batch is run before timing begins and excluded from all statistics.
- GPU peak memory is measured via `torch.cuda.max_memory_allocated()` with a reset each batch, so each recorded value is the peak within that single batch.
- Message count is the total number of edges (post-pruning) consumed across all `n_layer` GNN layers in a single forward pass.
