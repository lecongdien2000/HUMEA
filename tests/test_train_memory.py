import torch

from memory_utils import checkpoint_call, checkpoint_loss


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
