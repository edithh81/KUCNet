"""Personalised PageRank for KUCNet.

Refined drop-in replacement for the original sparse-rank power-iteration.
Same return shape as before — a dense ``[n_users, n_nodes]`` CPU tensor —
so ``models.py``'s ``self.ppr[u, v]`` indexing keeps working unchanged.

Speed wins vs. the original implementation:

  * dense ``rank`` vector (sparse-dense ``torch.sparse.mm`` is much faster
    than the original sparse-sparse mm);
  * out-degrees via ``torch.bincount`` instead of ``unique + counts``;
  * personalisation matrix ``P`` filled via ``Tensor.fill_`` + indexed
    write instead of repeated ``torch.cat`` of growing index/value tensors;
  * early-stop when ``||next_rank - rank||_2 < tol``;
  * automatic CPU fallback on CUDA OOM / cublas / cusparse errors.

A ``topk=`` mode is also provided (returns a dict of per-user top-k indices
and scores) for memory-constrained settings — note that this changes the
return type, so it requires corresponding model-side changes.
"""

import time

import torch
from tqdm import tqdm


def _normalize_device(device):
    if isinstance(device, torch.device):
        if device.type == "cuda" and not torch.cuda.is_available():
            print("CUDA requested for PPR, but CUDA is unavailable. Falling back to CPU.")
            return torch.device("cpu")
        return device
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = str(device).lower()
    if dev in {"auto", "cuda_if_available"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested for PPR, but CUDA is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def compress_ppr_to_topk(ppr: torch.Tensor, topk: int, score_dtype: torch.dtype = torch.float32):
    """Compress a dense ``[n_users, n_nodes]`` PPR matrix to per-user top-k lists.

    Useful when the dense PPR fits in RAM at compute time but you want to
    cache only a sparse summary on disk.
    """
    if ppr.dim() != 2:
        raise ValueError(f"Expected a 2D PPR tensor, got shape {tuple(ppr.shape)}")

    _, n_nodes = ppr.shape
    k = min(max(int(topk), 1), n_nodes)
    scores, indices = torch.topk(ppr.float(), k=k, dim=1)
    return {
        "mode": "topk",
        "indices": indices.to(dtype=torch.int32).cpu().contiguous(),
        "scores": scores.to(dtype=score_dtype).cpu().contiguous(),
        "ppr_topk": int(k),
    }


def _run_ppr_once(
    loader,
    bs,
    N,
    alpha,
    beta,
    tol,
    topk,
    compute_dtype,
    cache_dtype,
    device,
):
    n_nodes = int(loader.n_nodes)
    n_users = int(loader.n_users)

    tkg = torch.as_tensor(loader.tKG, dtype=torch.long, device=device)
    heads = tkg[:, 0]
    tails = tkg[:, 2]

    # Random-walk transition matrix M[i, j] = 1 / out_deg(i)  (row-stochastic).
    out_deg = torch.bincount(heads, minlength=n_nodes).clamp_min(1)
    values = (1.0 / out_deg[tails]).to(dtype=compute_dtype)
    indices = torch.stack([heads, tails], dim=0)
    M = torch.sparse_coo_tensor(
        indices, values, (n_nodes, n_nodes), dtype=compute_dtype, device=device
    ).coalesce()

    keep_topk = topk is not None and int(topk) > 0 and int(topk) < n_nodes
    if keep_topk:
        k = int(topk)
        final_indices = torch.empty((n_users, k), dtype=torch.int32)
        final_scores = torch.empty((n_users, k), dtype=cache_dtype)
    else:
        final_rank = torch.empty((n_users, n_nodes), dtype=cache_dtype)

    n_batch = n_users // bs + int(n_users % bs > 0)
    s_time = time.time()

    for i in tqdm(range(n_batch), desc=f"[PPR] power-iter on {device.type}"):
        start = i * bs
        end = min(n_users, start + bs)
        tbs = end - start
        u_list = list(range(start, end))

        # rank starts as one-hot per user (dense, [n_nodes, tbs]).
        rank = torch.zeros((n_nodes, tbs), dtype=compute_dtype, device=device)
        user_idx = torch.tensor(u_list, dtype=torch.long, device=device)
        col_idx = torch.arange(tbs, dtype=torch.long, device=device)
        rank[user_idx, col_idx] = 1.0

        # Personalisation P[:, col]: beta/|prefs| on known prefs,
        #                            (1-beta)/(n_nodes - |prefs|) elsewhere.
        P = torch.empty((n_nodes, tbs), dtype=compute_dtype, device=device)
        for col, uid in enumerate(u_list):
            p_set = loader.known_user_set[uid]
            n_pref = len(p_set)

            if n_pref >= n_nodes:
                P[:, col].fill_(1.0 / n_nodes)
                continue

            denom = max(1, n_nodes - n_pref)
            base = (1.0 - beta) / denom
            P[:, col].fill_(base)

            if n_pref > 0:
                pref_idx = torch.as_tensor(list(p_set), dtype=torch.long, device=device)
                P[pref_idx, col] = beta / n_pref

        # Power iteration with early stop.
        for _ in range(N):
            next_rank = (1.0 - alpha) * P + alpha * torch.sparse.mm(M, rank)
            if torch.norm(next_rank - rank).item() < tol:
                rank = next_rank
                break
            rank = next_rank

        rank_t = rank.transpose(0, 1).cpu()  # [tbs, n_nodes]
        if keep_topk:
            scores, idx = torch.topk(rank_t.float(), k=k, dim=1)
            final_indices[start:end] = idx.to(dtype=torch.int32)
            final_scores[start:end] = scores.to(dtype=cache_dtype)
        else:
            final_rank[start:end] = rank_t.to(dtype=cache_dtype)

        if device.type == "cuda":
            del rank, P, rank_t
            torch.cuda.empty_cache()

    print(f"PPR done on {device.type}. time: {time.time() - s_time:.1f}s")

    if keep_topk:
        return {
            "mode": "topk",
            "indices": final_indices.contiguous(),
            "scores": final_scores.contiguous(),
            "ppr_topk": int(k),
        }
    return final_rank.contiguous()


def get_ppr(
    loader,
    bs=128,
    N=20,
    alpha=0.85,
    beta=0.8,
    tol=1e-6,
    topk=None,
    compute_dtype=torch.float32,
    cache_dtype=torch.float32,
    device=None,
    fallback_to_cpu=True,
):
    """Compute Personalised PageRank for every user.

    Drop-in for the original ``get_ppr`` — same default return type
    (dense ``[n_users, n_nodes]`` CPU tensor in float32), much faster.

    Args:
        loader:        ``DataLoader`` exposing ``tKG``, ``n_nodes``,
                       ``n_users`` and ``known_user_set``.
        bs:            user batch size for the power iteration.
        N:             max power-iteration steps.
        alpha:         teleport weight (default 0.85).
        beta:          weight on the preference set vs. uniform in the
                       personalisation vector (default 0.8).
        tol:           early-stop threshold on ``||next_rank - rank||_2``.
        topk:          if a positive int < n_nodes, return per-user top-k
                       indices+scores as a dict (changes the return type
                       and requires model-side adaptation). Default ``None``
                       returns the full dense matrix.
        compute_dtype: dtype during iteration (default float32).
        cache_dtype:   dtype for the returned tensor (default float32 to
                       match original numerics; pass ``torch.float16`` to
                       halve memory at the cost of underflow on very small
                       PPR values).
        device:        torch.device, 'cuda', 'cpu', 'auto', or None
                       (default: cuda if available, else cpu).
        fallback_to_cpu: if a CUDA run raises an OOM / cublas / cusparse
                       error, retry once on CPU before giving up.
    """
    compute_device = _normalize_device(device)

    try:
        return _run_ppr_once(
            loader=loader,
            bs=bs,
            N=N,
            alpha=alpha,
            beta=beta,
            tol=tol,
            topk=topk,
            compute_dtype=compute_dtype,
            cache_dtype=cache_dtype,
            device=compute_device,
        )
    except RuntimeError as e:
        msg = str(e).lower()
        oom_like = any(
            token in msg
            for token in ("out of memory", "cuda error", "cublas", "cusparse")
        )
        if compute_device.type == "cuda" and fallback_to_cpu and oom_like:
            print(f"GPU PPR failed ({e}). Falling back to CPU once.")
            torch.cuda.empty_cache()
            return _run_ppr_once(
                loader=loader,
                bs=bs,
                N=N,
                alpha=alpha,
                beta=beta,
                tol=tol,
                topk=topk,
                compute_dtype=compute_dtype,
                cache_dtype=cache_dtype,
                device=torch.device("cpu"),
            )
        raise
