import torch

from layers import combine_expert_outputs


def test_combine_expert_outputs_matches_materialized_reference_and_gradients():
    torch.manual_seed(7)
    gates = torch.randn(9, 5, dtype=torch.float64, requires_grad=True)
    experts = torch.randn(9, 5, 13, dtype=torch.float64, requires_grad=True)
    reference_gates = gates.detach().clone().requires_grad_(True)
    reference_experts = experts.detach().clone().requires_grad_(True)

    actual = combine_expert_outputs(gates, experts)
    expected = (reference_gates.unsqueeze(-1) * reference_experts).sum(dim=-2)

    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(gates.grad, reference_gates.grad)
    torch.testing.assert_close(experts.grad, reference_experts.grad)
