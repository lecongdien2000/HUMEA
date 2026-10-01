"""Kaggle orchestration for the six main HUMEA reproduction experiments."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Experiment:
    """One main-table HUMEA experiment and its published target metrics."""

    id: str
    dataset: str
    rate: float
    fusion_weight_dim: int
    hits1: float
    hits5: float
    hits10: float
    mrr: float


EXPERIMENTS = {
    experiment.id: experiment
    for experiment in (
        Experiment("db15k-20", "FB15K_DB15K", 0.2, 512, 0.5118, 0.6997, 0.7643, 0.5980),
        Experiment("db15k-50", "FB15K_DB15K", 0.5, 0, 0.6949, 0.8434, 0.8803, 0.7630),
        Experiment("db15k-80", "FB15K_DB15K", 0.8, 0, 0.8094, 0.9146, 0.9353, 0.8560),
        Experiment("yago15k-20", "FB15K_YAGO15K", 0.2, 128, 0.4393, 0.6278, 0.6989, 0.5280),
        Experiment("yago15k-50", "FB15K_YAGO15K", 0.5, 0, 0.6491, 0.8032, 0.8474, 0.7200),
        Experiment("yago15k-80", "FB15K_YAGO15K", 0.8, 0, 0.7737, 0.8904, 0.9196, 0.8270),
    )
}


def build_train_command(
    repo_root: Path,
    experiment: Experiment,
    *,
    epochs: int = 1000,
    checkpoint: int = 10,
) -> list[str]:
    """Build the direct ``train.py`` command used for one paper experiment."""

    del repo_root  # The subprocess cwd selects the repository; paths stay portable.
    return [
        sys.executable,
        "train.py",
        "--file_dir",
        f"data/mmkb-datasets/{experiment.dataset}",
        "--rate",
        str(experiment.rate),
        "--lr",
        ".0005",
        "--epochs",
        str(epochs),
        "--hidden_units",
        "300,300,300",
        "--check_point",
        str(checkpoint),
        "--bsize",
        "512",
        "--il_start",
        "500",
        "--csls",
        "--csls_k",
        "3",
        "--seed",
        "42",
        "--tau_cl",
        "0.1",
        "--tau_al",
        "4.0",
        "--fusion_weight_dim",
        str(experiment.fusion_weight_dim),
        "--without",
        "0",
        "--al_loss",
        "0.1",
        "--cl_loss",
        "1.0",
    ]
