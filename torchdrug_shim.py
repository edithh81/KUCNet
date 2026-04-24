"""Drop-in replacement for ``torchdrug.layers.functional.variadic_topk``.

KUCNet originally depended on ``torchdrug==0.2.0.post1``, which does not ship
wheels for Python 3.12 / PyTorch 2.8. This module provides the single symbol
the model code actually uses (``variadic_topk``) as a vectorised PyTorch-only
implementation.

Usage (in ``models.py``)::

    import torchdrug_shim as functional
    functional.variadic_topk(values, sizes, k)
"""

import torch


def variadic_topk(input, size, k, largest=True):
    """Top-k values/indices over variadic-length groups.

    Args:
        input:   1-D tensor, groups concatenated contiguously.
        size:    1-D LongTensor of per-group sizes; ``size.sum() == input.numel()``.
        k:       number of top elements to return per group.
        largest: if True, return largest; else smallest.

    Returns:
        (values, indices) — each shaped ``(n_groups * k,)``.
        ``indices`` are **global** offsets into ``input`` (i.e. include each
        group's start offset), matching the original torchdrug behaviour.

        For groups shorter than ``k``, trailing slots are zero-valued and
        point at that group's start. Empty groups yield all-zero indices.
    """
    n_groups = size.numel()
    device = input.device

    if n_groups == 0:
        return (
            input.new_zeros(0),
            torch.zeros(0, dtype=torch.long, device=device),
        )

    max_count = int(size.max().item())
    padded_len = max(max_count, k)

    starts = torch.zeros_like(size)
    if n_groups > 1:
        starts[1:] = size[:-1].cumsum(0)

    pad_fill = float("-inf") if largest else float("inf")
    padded = input.new_full((n_groups, padded_len), pad_fill)
    if input.numel() > 0:
        group_idx = torch.repeat_interleave(
            torch.arange(n_groups, device=device), size
        )
        pos_in_group = (
            torch.arange(input.numel(), device=device) - starts[group_idx]
        )
        padded[group_idx, pos_in_group] = input

    values, local_idx = torch.topk(padded, k, dim=1, largest=largest)

    valid = local_idx < size.unsqueeze(1)
    values = torch.where(valid, values, torch.zeros_like(values))
    local_idx = torch.where(valid, local_idx, torch.zeros_like(local_idx))

    global_idx = local_idx + starts.unsqueeze(1)

    empty = (size == 0).unsqueeze(1)
    if empty.any():
        global_idx = torch.where(empty, torch.zeros_like(global_idx), global_idx)

    return values.reshape(-1), global_idx.reshape(-1)
