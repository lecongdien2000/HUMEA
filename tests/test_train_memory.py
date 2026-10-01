import torch
from pathlib import Path

from memory_utils import checkpoint_call, checkpoint_loss, saved_tensor_offload


def test_checkpoint_loss_preserves_value_and_gradients():
    def quadratic_loss(tensor, scale):
        return (tensor.square() * scale).sum()

    direct_input = torch.randn(8, requires_grad=True)
    checkpoint_input = direct_input.detach().clone().requires_grad_(True)

    direct = quadratic_loss(direct_input, 0.25)
    checkpointed = checkpoint_loss(quadratic_loss, checkpoint_input, 0.25)
    direct.backward()
    checkpointed.backward()

    torch.testing.assert_close(checkpointed, direct)
    torch.testing.assert_close(checkpoint_input.grad, direct_input.grad)


def test_checkpoint_call_supports_nested_outputs_and_keyword_arguments():
    def encoder(tensor, *, scale):
        return [tensor * scale, tensor.square()], {"sum": tensor.sum()}

    value = torch.randn(8, requires_grad=True)
    outputs, metadata = checkpoint_call(encoder, value, scale=0.5)
    (outputs[0].sum() + outputs[1].sum() + metadata["sum"]).backward()

    assert value.grad is not None


def test_cuda_cache_is_released_immediately_before_backward():
    source = (Path(__file__).parents[1] / "train.py").read_text(encoding="utf-8")
    backward = source.index("sum(loss_all).backward()")
    cache_release = source.rfind("torch.cuda.empty_cache()", 0, backward)

    assert cache_release != -1
    assert source[cache_release:backward].strip() == "torch.cuda.empty_cache()"


def test_saved_tensor_offload_is_a_noop_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    value = torch.tensor(2.0, requires_grad=True)

    with saved_tensor_offload():
        result = value.square()
    result.backward()

    assert value.grad.item() == 4.0


def test_training_offloads_saved_tensors_across_forward_and_backward():
    source = (Path(__file__).parents[1] / "train.py").read_text(encoding="utf-8")

    enter = source.index("saved_tensors.__enter__()")
    forward = source.index("self.multimodal_encoder,", enter)
    backward = source.index("sum(loss_all).backward()", forward)
    exit_context = source.index("saved_tensors.__exit__", backward)

    assert enter < forward < backward < exit_context
