"""Drop-in replacement for ``torchdrug.layers.functional.variadic_topk``.

KUCNet originally depended on ``torchdrug==0.2.0.post1``, which does not
ship wheels for Python 3.12 / PyTorch 2.8. This module provides the single
symbol the model code actually uses (``variadic_topk``) as a vectorised
PyTorch-only implementation matching torchdrug's original contract.

Contract (matches torchdrug 0.2.0):
    variadic_topk(input, size, k) -> (values, indices)

    - ``values``  has shape ``(n_groups, k)``
    - ``indices`` has shape ``(n_groups, k)`` — **local** to each group
      (i.e. positions within ``input[starts[g] : starts[g] + size[g]]``).

The caller (``models.GNNLayer.forward``) adds the per-group offset itself
via ``topk_index + cnt_sum.unsqueeze(1)``, so we must NOT add it here.

Usage (in ``models.py``)::

    import torchdrug_shim as functional
    topk_value, topk_index = functional.variadic_topk(values, sizes, k)
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
        (values, indices) — both shaped ``(n_groups, k)``.

        ``indices`` are **local** positions within each group's slice
        (in ``[0, size[g])`` for valid slots). For groups shorter than
        ``k`` the trailing slots get zero-valued ``values`` and a
        zero-valued ``indices`` entry — after the caller adds the
        group's start offset, those padded slots collapse onto the
        first edge of the group and are removed by ``torch.unique``.
    """
    n_groups = size.numel()
    device = input.device

    if n_groups == 0:
        return (
            input.new_zeros((0, k)),
            torch.zeros((0, k), dtype=torch.long, device=device),
        )

    max_count = int(size.max().item()) if input.numel() > 0 else 0
    padded_len = max(max_count, k)

    starts = torch.zeros_like(size)
    if n_groups > 1:
        starts[1:] = size[:-1].cumsum(0)

    # Scatter ``input`` into a dense [n_groups, padded_len] matrix. Empty
    # slots are filled with ±inf so ``torch.topk`` never picks them.
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

    # Slots beyond a group's true length are padding — zero their value
    # and (local) index. The caller adds the group's start offset later,
    # so these padded slots will all alias to ``starts[g]`` (the first
    # valid edge of the group) and be deduplicated by ``torch.unique``.
    valid = local_idx < size.unsqueeze(1)
    values = torch.where(valid, values, torch.zeros_like(values))
    local_idx = torch.where(valid, local_idx, torch.zeros_like(local_idx))

    return values, local_idx
