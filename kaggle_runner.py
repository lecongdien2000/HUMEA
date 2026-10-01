"""Kaggle orchestration for the six main HUMEA reproduction experiments."""

from __future__ import annotations

import sys
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


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

COMMON_DATASET_FILES = (
    "ent_ids_1",
    "ent_ids_2",
    "ill_ent_ids",
    "triples_1",
    "triples_2",
    "training_attrs_1",
    "training_attrs_2",
    "attribute_feature_dict.pkl",
    "triples_feature_dict.pkl",
)

BEST_RESULT_PATTERN = re.compile(
    r"Best avg epoch <(?P<epoch>\d+)>:.*?"
    r"acc@\[[^\]]+\]=\[(?P<accuracy>[^\]]+)\],\s*"
    r"mr=[0-9.eE+-]+,\s*mrr=(?P<mrr>[0-9.eE+-]+)",
    re.MULTILINE,
)


def required_dataset_files(dataset: str) -> tuple[str, ...]:
    """Return all released files consumed by ``train.py`` for a dataset."""

    return (*COMMON_DATASET_FILES, f"{dataset}_id_img_feature_dict.pkl")


def find_missing_files(
    repo_root: Path, experiments: Iterable[Experiment]
) -> list[Path]:
    """Return required paths that are absent, relative to the repository root."""

    missing: set[Path] = set()
    for experiment in experiments:
        relative_dir = Path("data") / "mmkb-datasets" / experiment.dataset
        for filename in required_dataset_files(experiment.dataset):
            relative_path = relative_dir / filename
            if not (repo_root / relative_path).is_file():
                missing.add(relative_path)
    return sorted(missing, key=lambda path: path.as_posix())


def parse_best_metrics(text: str) -> dict[str, int | float] | None:
    """Parse the final best-result record emitted by the authors' ``train.py``."""

    matches = list(BEST_RESULT_PATTERN.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    accuracy = [
        float(value)
        for value in re.split(r"[\s,]+", match.group("accuracy").strip())
        if value
    ]
    if len(accuracy) != 3:
        return None
    return {
        "epoch": int(match.group("epoch")),
        "hits1": accuracy[0],
        "hits5": accuracy[1],
        "hits10": accuracy[2],
        "mrr": float(match.group("mrr")),
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
