"""``torch_scatter.scatter`` fallback using only core PyTorch.

If ``torch_scatter`` is installed (fast CUDA kernels) we re-export it.
Otherwise we provide a minimal native implementation that covers the call
sites used by KUCNet: ``scatter(src, index, dim=0, dim_size=N, reduce='sum')``.

The native path uses ``Tensor.index_add_`` for 'sum' and
``Tensor.scatter_reduce_`` (PyTorch >= 1.12) for 'mean' / 'max' / 'min'.
"""

import torch

try:
    from torch_scatter import scatter as _scatter  # type: ignore[import-not-found]

    scatter = _scatter

except Exception:  # torch_scatter not installed / unavailable

    def scatter(src, index, dim=0, out=None, dim_size=None, reduce="sum"):
        if dim != 0:
            raise NotImplementedError(
                "scatter_shim fallback only supports dim=0 (KUCNet's only usage)"
            )

        if dim_size is None:
            dim_size = int(index.max().item()) + 1 if index.numel() else 0

        if out is None:
            shape = (dim_size,) + tuple(src.shape[1:])
            out = torch.zeros(shape, dtype=src.dtype, device=src.device)

        if reduce == "sum":
            out.index_add_(0, index, src)
            return out

        # Broadcast index to src's trailing dims for scatter_reduce_.
        if src.dim() > 1:
            idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
        else:
            idx = index

        reduce_map = {"mean": "mean", "max": "amax", "min": "amin"}
        if reduce not in reduce_map:
            raise NotImplementedError(
                f"scatter_shim fallback: reduce='{reduce}' not supported"
            )
        out.scatter_reduce_(0, idx, src, reduce=reduce_map[reduce], include_self=False)
        return out
