"""Memory-preserving helpers for the original HUMEA training objective."""

from collections.abc import Callable

import torch
from torch.utils.checkpoint import checkpoint


def checkpoint_loss(criterion: Callable, *args) -> torch.Tensor:
    """Evaluate a loss with recomputation during backward to bound GPU memory."""

    return checkpoint(criterion, *args, use_reentrant=False)
