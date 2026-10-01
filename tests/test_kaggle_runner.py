import sys
import threading
from pathlib import Path

import kaggle_runner
import pytest
from kaggle_runner import (
    EXPERIMENTS,
    assign_gpu_queues,
    build_train_command,
    execute_queues,
    find_missing_files,
    load_manifest,
    main,
    parse_best_metrics,
    required_dataset_files,
    run_experiment,
    should_skip,
    write_summary_csv,
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


def test_command_builder_is_defined_before_script_entrypoint():
    source = Path(kaggle_runner.__file__).read_text(encoding="utf-8")

    assert source.index("def build_train_command") < source.index(
        'if __name__ == "__main__"'
    )


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


def test_one_gpu_queue_preserves_table_order():
    queues = assign_gpu_queues(list(EXPERIMENTS), ["0"])

    assert queues == {"0": list(EXPERIMENTS)}


def test_two_gpu_queues_split_datasets():
    queues = assign_gpu_queues(list(EXPERIMENTS), ["0", "1"])

    assert queues == {
        "0": ["db15k-20", "db15k-50", "db15k-80"],
        "1": ["yago15k-20", "yago15k-50", "yago15k-80"],
    }


def test_resume_skips_only_successful_run_with_metrics():
    manifest = {
        "version": 1,
        "experiments": {
            "success": {"status": "success", "metrics": {"mrr": 0.5}},
            "failed": {"status": "failed", "metrics": None},
            "incomplete": {"status": "running", "metrics": None},
        },
    }

    assert should_skip(manifest, "success", force=False)
    assert not should_skip(manifest, "success", force=True)
    assert not should_skip(manifest, "failed", force=False)
    assert not should_skip(manifest, "incomplete", force=False)
    assert not should_skip(manifest, "unknown", force=False)


class FakeProcess:
    def __init__(self, lines: list[str], returncode: int = 0):
        self.stdout = iter(lines)
        self.returncode = returncode

    def wait(self) -> int:
        return self.returncode


def test_run_experiment_records_log_manifest_and_metrics(tmp_path, monkeypatch):
    best_line = (
        "Best avg epoch <330>: acc@[1, 5, 10]="
        "[0.51175 0.6997 0.7643], mr=46.123, mrr=0.598\n"
    )
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess(["training\n", best_line])

    monkeypatch.setattr(kaggle_runner.subprocess, "Popen", fake_popen)
    manifest = load_manifest(tmp_path / "artifacts" / "manifest.json")

    success = run_experiment(
        repo_root=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        experiment=EXPERIMENTS["db15k-20"],
        gpu_id="1",
        manifest=manifest,
        manifest_lock=threading.Lock(),
    )

    assert success
    assert captured["kwargs"]["cwd"] == tmp_path
    assert captured["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert (
        captured["kwargs"]["env"]["PYTORCH_ALLOC_CONF"]
        == "expandable_segments:True"
    )
    log_text = (tmp_path / "artifacts" / "logs" / "db15k-20.log").read_text()
    assert "training" in log_text
    record = manifest["experiments"]["db15k-20"]
    assert record["status"] == "success"
    assert record["gpu"] == "1"
    assert record["metrics"]["hits1"] == 0.51175
    assert record["command"] == captured["command"]
    assert record["duration_seconds"] >= 0
    assert (tmp_path / "artifacts" / "manifest.json").is_file()


def test_cpu_run_overrides_train_device(tmp_path, monkeypatch):
    best_line = (
        "Best avg epoch <10>: acc@[1, 5, 10]="
        "[0.1 0.2 0.3], mr=50.0, mrr=0.15\n"
    )
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        return FakeProcess([best_line])

    monkeypatch.setattr(kaggle_runner.subprocess, "Popen", fake_popen)
    manifest = {"version": 1, "experiments": {}}

    assert run_experiment(
        repo_root=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        experiment=EXPERIMENTS["db15k-20"],
        gpu_id="cpu",
        manifest=manifest,
        manifest_lock=threading.Lock(),
        epochs=12,
    )
    assert captured["command"][captured["command"].index("--device") + 1] == "cpu"


def test_summary_marks_metrics_close_to_paper(tmp_path):
    manifest = {
        "version": 1,
        "experiments": {
            "db15k-20": {
                "status": "success",
                "gpu": "0",
                "duration_seconds": 60.0,
                "metrics": {
                    "epoch": 330,
                    "hits1": 0.51175,
                    "hits5": 0.6997,
                    "hits10": 0.7643,
                    "mrr": 0.598,
                },
            }
        },
    }

    summary_path = tmp_path / "summary.csv"
    write_summary_csv(summary_path, manifest)

    summary = summary_path.read_text()
    assert "db15k-20,FB15K_DB15K,0.2,0,success,330" in summary
    assert summary.rstrip().endswith(",close")


def test_failed_experiment_stops_only_its_gpu_queue(tmp_path, monkeypatch):
    calls = []

    def fake_run_experiment(**kwargs):
        item = (kwargs["gpu_id"], kwargs["experiment"].id)
        calls.append(item)
        return kwargs["experiment"].id != "db15k-20"

    monkeypatch.setattr(kaggle_runner, "run_experiment", fake_run_experiment)
    queues = {
        "0": ["db15k-20", "db15k-50", "db15k-80"],
        "1": ["yago15k-20", "yago15k-50", "yago15k-80"],
    }

    success = execute_queues(
        repo_root=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        queues=queues,
        manifest={"version": 1, "experiments": {}},
    )

    assert not success
    assert ("0", "db15k-20") in calls
    assert ("0", "db15k-50") not in calls
    assert ("0", "db15k-80") not in calls
    assert [item for item in calls if item[0] == "1"] == [
        ("1", "yago15k-20"),
        ("1", "yago15k-50"),
        ("1", "yago15k-80"),
    ]


def make_complete_data(repo_root: Path, experiment_ids: list[str]) -> None:
    for experiment_id in experiment_ids:
        experiment = EXPERIMENTS[experiment_id]
        dataset_dir = repo_root / "data" / "mmkb-datasets" / experiment.dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for relative_path in required_dataset_files(experiment.dataset):
            (dataset_dir / relative_path).touch(exist_ok=True)


def test_cli_validate_checks_all_released_inputs(tmp_path, monkeypatch, capsys):
    make_complete_data(tmp_path, list(EXPERIMENTS))
    monkeypatch.setattr(kaggle_runner, "validate_runtime_dependencies", lambda: [])
    monkeypatch.setattr(kaggle_runner, "discover_gpu_ids", lambda: ["0", "1"])

    exit_code = main(["--repo-root", str(tmp_path), "validate"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Data validation passed for 6 experiment(s)" in output
    assert "Visible CUDA devices: 2" in output


def test_cli_rejects_unknown_single_experiment(tmp_path):
    with pytest.raises(SystemExit):
        main(
            [
                "--repo-root",
                str(tmp_path),
                "single",
                "--experiment",
                "not-an-experiment",
            ]
        )


def test_cli_rejects_full_training_without_gpu(tmp_path, monkeypatch, capsys):
    make_complete_data(tmp_path, ["db15k-20"])
    monkeypatch.setattr(kaggle_runner, "validate_runtime_dependencies", lambda: [])
    monkeypatch.setattr(kaggle_runner, "discover_gpu_ids", lambda: [])

    exit_code = main(
        [
            "--repo-root",
            str(tmp_path),
            "single",
            "--experiment",
            "db15k-20",
        ]
    )

    assert exit_code == 2
    assert "No CUDA GPU is visible" in capsys.readouterr().err


def test_cli_allow_cpu_invokes_single_experiment(tmp_path, monkeypatch):
    make_complete_data(tmp_path, ["db15k-20"])
    monkeypatch.setattr(kaggle_runner, "validate_runtime_dependencies", lambda: [])
    monkeypatch.setattr(kaggle_runner, "discover_gpu_ids", lambda: [])
    captured = {}

    def fake_execute_queues(**kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr(kaggle_runner, "execute_queues", fake_execute_queues)

    exit_code = main(
        [
            "--repo-root",
            str(tmp_path),
            "--allow-cpu",
            "single",
            "--experiment",
            "db15k-20",
        ]
    )

    assert exit_code == 0
    assert captured["queues"] == {"cpu": ["db15k-20"]}
    assert captured["epochs"] == 1000


def test_cli_smoke_uses_resume_key_distinct_from_full_run(
    tmp_path, monkeypatch
):
    make_complete_data(tmp_path, ["db15k-20"])
    monkeypatch.setattr(kaggle_runner, "validate_runtime_dependencies", lambda: [])
    monkeypatch.setattr(kaggle_runner, "discover_gpu_ids", lambda: ["0"])
    captured = {}

    def fake_execute_queues(**kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr(kaggle_runner, "execute_queues", fake_execute_queues)

    assert main(["--repo-root", str(tmp_path), "smoke"]) == 0
    assert captured["run_id_prefix"] == "smoke-"
    smoke_manifest = {
        "version": 1,
        "experiments": {
            "smoke-db15k-20": {
                "status": "success",
                "metrics": {"mrr": 0.01},
            }
        },
    }
    assert not should_skip(smoke_manifest, "db15k-20", force=False)


def test_cli_rejects_unwritable_artifact_target(tmp_path, monkeypatch, capsys):
    make_complete_data(tmp_path, ["db15k-20"])
    blocked_path = tmp_path / "blocked-artifacts"
    blocked_path.write_text("this is a file, not a directory")
    monkeypatch.setattr(kaggle_runner, "validate_runtime_dependencies", lambda: [])
    monkeypatch.setattr(kaggle_runner, "discover_gpu_ids", lambda: ["0"])

    exit_code = main(
        [
            "--repo-root",
            str(tmp_path),
            "--artifacts-dir",
            str(blocked_path),
            "single",
            "--experiment",
            "db15k-20",
        ]
    )

    assert exit_code == 2
    assert "Artifact directory is not writable" in capsys.readouterr().err
