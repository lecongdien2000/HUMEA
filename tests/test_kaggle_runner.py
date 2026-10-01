import sys
from pathlib import Path

from kaggle_runner import EXPERIMENTS, build_train_command


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
