import torch
import pytest

from layers import SpecialSpmmFunction, combine_expert_outputs, sparse_value_gradients


@pytest.mark.parametrize("dropout", [0.0, 0.3])
def test_whitening_projection_matches_centered_reference(dropout):
    import copy
    from layers import PWLayer

    torch.manual_seed(42)
    actual = PWLayer(13, 7, dropout=dropout).double()
    with torch.no_grad():
        actual.bias.normal_()
    reference = copy.deepcopy(actual)
    x = torch.randn(9, 13, dtype=torch.float64, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)
    torch.manual_seed(5)
    result = actual(x)
    torch.manual_seed(5)
    expected = reference.lin(reference.dropout(ref_x) - reference.bias)
    result.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(x.grad, ref_x.grad)
    for parameter, ref_parameter in zip(actual.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, ref_parameter.grad)


def test_whitening_projection_does_not_save_a_centered_feature_table():
    from layers import PWLayer

    layer = PWLayer(13, 7)
    features = torch.randn(9, 13)
    saved = []

    def pack(tensor):
        if tensor.shape == features.shape:
            saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        layer(features).square().sum().backward()
    assert saved
    assert all(tensor.data_ptr() == features.data_ptr() for tensor in saved)


def test_combine_expert_outputs_matches_materialized_reference_and_gradients():
    torch.manual_seed(7)
    gates = torch.randn(9, 5, dtype=torch.float64, requires_grad=True)
    experts = torch.randn(9, 5, 13, dtype=torch.float64, requires_grad=True)
    reference_gates = gates.detach().clone().requires_grad_(True)
    reference_experts = experts.detach().clone().requires_grad_(True)

    actual = combine_expert_outputs(gates, list(experts.unbind(dim=1)))
    expected = (reference_gates.unsqueeze(-1) * reference_experts).sum(dim=-2)

    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(gates.grad, reference_gates.grad)
    torch.testing.assert_close(experts.grad, reference_experts.grad)


def test_sparse_spmm_backward_matches_dense_reference():
    indices = torch.tensor([[0, 0, 1, 3, 4, 5], [1, 4, 2, 0, 5, 3]])
    values = torch.randn(indices.shape[1], dtype=torch.float64, requires_grad=True)
    features = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    reference_values = values.detach().clone().requires_grad_(True)
    reference_features = features.detach().clone().requires_grad_(True)
    upstream = torch.randn(6, 4, dtype=torch.float64)

    actual = SpecialSpmmFunction.apply(indices, values, (6, 6), features)
    dense = torch.zeros(6, 6, dtype=torch.float64)
    dense = dense.index_put(tuple(indices), reference_values, accumulate=True)
    expected = dense.matmul(reference_features)

    torch.testing.assert_close(actual, expected)
    (actual * upstream).sum().backward()
    (expected * upstream).sum().backward()
    torch.testing.assert_close(values.grad, reference_values.grad)
    torch.testing.assert_close(features.grad, reference_features.grad)


def test_sparse_value_gradients_only_compute_existing_edges():
    indices = torch.tensor([[0, 2, 3], [1, 0, 2]])
    grad_output = torch.randn(4, 5)
    features = torch.randn(4, 5)

    actual = sparse_value_gradients(grad_output, features, indices, chunk_size=2)
    dense_reference = grad_output.matmul(features.t())

    torch.testing.assert_close(actual, dense_reference[indices[0], indices[1]])
