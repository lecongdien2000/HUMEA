"""Memory-preserving helpers for the original HUMEA training objective."""

from collections.abc import Callable
import torch
from torch.utils.checkpoint import checkpoint


def checkpoint_call(function: Callable, *args, **kwargs):
    """Call a function with non-reentrant activation recomputation."""

    return checkpoint(function, *args, use_reentrant=False, **kwargs)


def checkpoint_loss(criterion: Callable, *args) -> torch.Tensor:
    """Evaluate a loss with recomputation during backward to bound GPU memory."""

    return checkpoint_call(criterion, *args)


def gradient_proxy(tensor: torch.Tensor) -> torch.Tensor:
    """Create a leaf that receives a loss gradient without retaining that loss graph."""

    return tensor.detach().requires_grad_(True)


def backward_through_gradient_bridge(originals, proxies) -> None:
    """Propagate accumulated proxy gradients through their original tensors."""

    if len(originals) != len(proxies):
        raise ValueError("Gradient bridge inputs must have matching lengths")
    active = [
        (original, proxy.grad)
        for original, proxy in zip(originals, proxies)
        if original.requires_grad and proxy.grad is not None
    ]
    if active:
        torch.autograd.backward(
            [original for original, _ in active],
            [gradient for _, gradient in active],
        )
