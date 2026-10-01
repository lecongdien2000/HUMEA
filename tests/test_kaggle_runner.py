import sys
from pathlib import Path

from kaggle_runner import (
    EXPERIMENTS,
    build_train_command,
    find_missing_files,
    parse_best_metrics,
    required_dataset_files,
)


def test_registry_matches_main_table():
    assert [
        (experiment.id, experiment.dataset, experiment.rate, experiment.fusion_weight_dim)
        for experiment in EXPERIMENTS.values()
    ] == [
        ("db15k-20", "FB15K_DB15K", 0.2, 512),
        ("db15k-50", "FB15K_DB15K", 0.5, 0),
        ("db15k-80", "FB15K_DB15K", 0.8, 0),
        ("yago15k-20", "FB15K_YAGO15K", 0.2, 128),
        ("yago15k-50", "FB15K_YAGO15K", 0.5, 0),
        ("yago15k-80", "FB15K_YAGO15K", 0.8, 0),
    ]


def test_full_command_matches_paper_configuration(tmp_path: Path):
    command = build_train_command(tmp_path, EXPERIMENTS["db15k-20"])

    assert command[:2] == [sys.executable, "train.py"]
    assert command[command.index("--file_dir") + 1] == (
        "data/mmkb-datasets/FB15K_DB15K"
    )
    assert command[command.index("--rate") + 1] == "0.2"
    assert command[command.index("--epochs") + 1] == "1000"
    assert command[command.index("--check_point") + 1] == "10"
    assert command[command.index("--fusion_weight_dim") + 1] == "512"
    assert "--csls" in command
    assert "--train_ill_path" not in command


def test_smoke_command_runs_twelve_epochs(tmp_path: Path):
    command = build_train_command(
        tmp_path,
        EXPERIMENTS["db15k-20"],
        epochs=12,
        checkpoint=10,
    )

    assert command[command.index("--epochs") + 1] == "12"
    assert command[command.index("--check_point") + 1] == "10"


def test_validation_accepts_complete_released_dataset(tmp_path: Path):
    experiment = EXPERIMENTS["db15k-20"]
    dataset_dir = tmp_path / "data" / "mmkb-datasets" / experiment.dataset
    dataset_dir.mkdir(parents=True)
    for relative_path in required_dataset_files(experiment.dataset):
        (dataset_dir / relative_path).touch()

    assert find_missing_files(tmp_path, [experiment]) == []


def test_validation_reports_exact_missing_path(tmp_path: Path):
    experiment = EXPERIMENTS["yago15k-20"]
    dataset_dir = tmp_path / "data" / "mmkb-datasets" / experiment.dataset
    dataset_dir.mkdir(parents=True)
    for relative_path in required_dataset_files(experiment.dataset):
        (dataset_dir / relative_path).touch()
    (dataset_dir / "triples_feature_dict.pkl").unlink()

    assert find_missing_files(tmp_path, [experiment]) == [
        Path(
            "data/mmkb-datasets/FB15K_YAGO15K/triples_feature_dict.pkl"
        )
    ]


def test_parse_best_metrics_reads_real_log_shape():
    text = (
        "2025-01-01 | INFO | training complete\n"
        "Best avg epoch <330>: acc@[1, 5, 10]="
        "[0.51175 0.6997  0.7643 ], mr=46.123, mrr=0.598\n"
    )

    assert parse_best_metrics(text) == {
        "epoch": 330,
        "hits1": 0.51175,
        "hits5": 0.6997,
        "hits10": 0.7643,
        "mrr": 0.598,
    }


def test_parse_best_metrics_returns_none_when_training_did_not_finish():
    assert parse_best_metrics("epoch 10 checkpoint") is None
