"""Memory-preserving helpers for the original HUMEA training objective."""

from collections.abc import Callable
from contextlib import nullcontext

import torch
from torch.utils.checkpoint import checkpoint


def checkpoint_call(function: Callable, *args, **kwargs):
    """Call a function with non-reentrant activation recomputation."""

    return checkpoint(function, *args, use_reentrant=False, **kwargs)


def checkpoint_loss(criterion: Callable, *args) -> torch.Tensor:
    """Evaluate a loss with recomputation during backward to bound GPU memory."""

    return checkpoint_call(criterion, *args)


def saved_tensor_offload():
    """Move tensors saved for CUDA backward to pinned CPU memory."""

    if not torch.cuda.is_available():
        return nullcontext()
    return torch.autograd.graph.save_on_cpu(pin_memory=True)
