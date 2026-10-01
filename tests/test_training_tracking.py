from types import SimpleNamespace

import numpy as np
import torch

from train import HUMEA


class RecordingTracker:
    def __init__(self):
        self.events = []

    def log_losses(self, epoch, metrics):
        assert all(isinstance(value, float) for value in metrics.values())
        self.events.append((epoch, metrics))


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.4))

    def forward(self, device, indices, *features, exp_outputs=False):
        table = (indices.float()[:, None] + 1) * self.weight
        modalities = [table * (index + 1) for index in range(6)]
        return modalities + [torch.cat(modalities, dim=1)], {"img": [table, table.square()]}


class TinyEstimator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.2))

    def forward(self, embeddings):
        return embeddings["img"][0].mean() * self.weight

    def train_estimator(self, embeddings):
        return self(embeddings).square()


def make_training(tracker):
    model = object.__new__(HUMEA)
    model.args = SimpleNamespace(bsize=2, epochs=2, mi_loss=0.001, il_start=-1,
                                 without=0, check_point=10)
    model.device = torch.device("cpu")
    model.ENT_NUM = 4
    model.train_ill = np.array([[0, 1], [2, 3]])
    model.tracker = tracker
    model.multimodal_encoder = TinyEncoder()
    model.mi_estimator = TinyEstimator()
    model.multi_loss_layer = torch.nn.Identity()
    model.align_multi_loss_layer = torch.nn.Identity()
    model.optimizer = torch.optim.SGD(model.multimodal_encoder.parameters(), lr=0.001)
    model.mi_optimizer = torch.optim.SGD(model.mi_estimator.parameters(), lr=0.001)
    for name in ("adj", "img_features", "rel_features", "att_features",
                 "att_txt_features", "rel_txt_features"):
        setattr(model, name, None)
    model.inner_view_loss = lambda *inputs: sum(x.square().mean() for x in inputs[:-1])
    model.kl_alignment_loss = lambda *inputs: sum(x.square().mean() for x in inputs[:-1])
    model.criterion_cl = lambda embeddings, links: embeddings.square().mean()
    return model


def test_training_logs_scalar_losses_each_epoch_without_changing_updates():
    tracker = RecordingTracker()
    recorded = make_training(tracker)
    baseline = make_training(None)
    np.random.seed(42)
    baseline.train()
    np.random.seed(42)
    recorded.train()
    torch.testing.assert_close(recorded.multimodal_encoder.weight,
                               baseline.multimodal_encoder.weight)
    torch.testing.assert_close(recorded.mi_estimator.weight, baseline.mi_estimator.weight)
    assert [epoch for epoch, metrics in tracker.events] == [0, 1]
    assert all({"total", "mi", "inner", "alignment", "joint", "estimator"}
               <= metrics.keys() for epoch, metrics in tracker.events)
