import pytest
import torch

import loss


def test_rowwise_normalization_commutes_with_training_row_selection():
    embedding = torch.randn(10, 4)
    rows = torch.tensor([0, 2, 7])

    normalize_then_select = torch.nn.functional.normalize(embedding, dim=1)[rows]
    select_then_normalize = torch.nn.functional.normalize(embedding[rows], dim=1)

    torch.testing.assert_close(select_then_normalize, normalize_then_select)


@pytest.mark.parametrize(
    ("criterion", "expected_calls"),
    [
        (loss.icl_loss(device="cpu"), 2),
        (loss.ial_loss(device="cpu"), 4),
    ],
)
def test_losses_normalize_only_selected_training_rows(
    monkeypatch, criterion, expected_calls
):
    embedding = torch.randn(10, 4)
    links = torch.tensor([[0, 5], [2, 7]])
    normalized_shapes = []
    original_normalize = loss.F.normalize

    def record_normalize(tensor, *args, **kwargs):
        normalized_shapes.append(tuple(tensor.shape))
        return original_normalize(tensor, *args, **kwargs)

    monkeypatch.setattr(loss.F, "normalize", record_normalize)

    if isinstance(criterion, loss.ial_loss):
        criterion(embedding, embedding.clone(), links)
    else:
        criterion(embedding, links)

    assert normalized_shapes == [(2, 4)] * expected_calls
