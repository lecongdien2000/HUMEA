"""Kaggle orchestration for the six main HUMEA reproduction experiments."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
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


def assign_gpu_queues(
    experiment_ids: list[str], gpu_ids: list[str]
) -> dict[str, list[str]]:
    """Assign dataset families to GPUs while preserving experiment order."""

    if not gpu_ids:
        return {}
    if len(gpu_ids) == 1:
        return {gpu_ids[0]: list(experiment_ids)}
    db15k = [item for item in experiment_ids if EXPERIMENTS[item].dataset == "FB15K_DB15K"]
    yago15k = [item for item in experiment_ids if EXPERIMENTS[item].dataset == "FB15K_YAGO15K"]
    queues: dict[str, list[str]] = {}
    if db15k:
        queues[gpu_ids[0]] = db15k
    if yago15k:
        queues[gpu_ids[1]] = yago15k
    return queues


def load_manifest(path: Path) -> dict:
    """Load runner state, or return a new manifest when none exists."""

    if not path.is_file():
        return {"version": 1, "experiments": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1 or not isinstance(data.get("experiments"), dict):
        raise ValueError(f"Unsupported or invalid manifest: {path}")
    return data


def save_manifest(path: Path, manifest: dict) -> None:
    """Atomically persist runner state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def should_skip(manifest: dict, experiment_id: str, *, force: bool) -> bool:
    """Return whether an experiment has a complete reusable result."""

    if force:
        return False
    record = manifest.get("experiments", {}).get(experiment_id, {})
    return record.get("status") == "success" and bool(record.get("metrics"))


def run_experiment(
    *,
    repo_root: Path,
    artifacts_dir: Path,
    experiment: Experiment,
    gpu_id: str,
    manifest: dict,
    manifest_lock: Lock,
    force: bool = False,
    epochs: int = 1000,
    checkpoint: int = 10,
) -> bool:
    """Run one experiment, stream its log, and record an atomic result."""

    manifest_path = artifacts_dir / "manifest.json"
    with manifest_lock:
        if should_skip(manifest, experiment.id, force=force):
            print(f"[skip] {experiment.id}: completed result already exists")
            return True

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    log_dir = artifacts_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{experiment.id}.log"
    command = build_train_command(
        repo_root,
        experiment,
        epochs=epochs,
        checkpoint=checkpoint,
    )
    environment = os.environ.copy()
    if gpu_id != "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = gpu_id

    started_at = datetime.now(timezone.utc)
    started_clock = time.monotonic()
    running_record = {
        "status": "running",
        "gpu": gpu_id,
        "command": command,
        "started_at": started_at.isoformat(),
        "metrics": None,
    }
    with manifest_lock:
        manifest["experiments"][experiment.id] = running_record
        save_manifest(manifest_path, manifest)

    output_lines: list[str] = []
    exit_code = -1
    error: str | None = None
    try:
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("Training subprocess did not expose stdout")
        with log_path.open("w", encoding="utf-8") as log_file:
            for line in process.stdout:
                print(f"[{experiment.id}] {line}", end="")
                log_file.write(line)
                output_lines.append(line)
        exit_code = process.wait()
    except Exception as exc:  # The manifest must preserve launch/runtime failures.
        error = f"{type(exc).__name__}: {exc}"

    duration = time.monotonic() - started_clock
    metrics = parse_best_metrics("".join(output_lines))
    success = exit_code == 0 and metrics is not None
    final_record = {
        **running_record,
        "status": "success" if success else "failed",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(duration, 3),
        "exit_code": exit_code,
        "metrics": metrics,
        "log": str(log_path.relative_to(repo_root))
        if log_path.is_relative_to(repo_root)
        else str(log_path),
    }
    if error:
        final_record["error"] = error
    elif exit_code == 0 and metrics is None:
        final_record["error"] = "Training exited successfully without final best metrics"

    with manifest_lock:
        manifest["experiments"][experiment.id] = final_record
        save_manifest(manifest_path, manifest)
    return success


def _comparison(experiment: Experiment, metrics: dict | None) -> str:
    if not metrics:
        return ""
    targets = {
        "hits1": experiment.hits1,
        "hits5": experiment.hits5,
        "hits10": experiment.hits10,
        "mrr": experiment.mrr,
    }
    return (
        "close"
        if all(abs(float(metrics[key]) - target) <= 0.01 for key, target in targets.items())
        else "different"
    )


def write_summary_csv(path: Path, manifest: dict) -> None:
    """Write stable tabular results for Kaggle display and download."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "dataset",
        "rate",
        "gpu",
        "status",
        "epoch",
        "hits1",
        "hits5",
        "hits10",
        "mrr",
        "duration_seconds",
        "comparison",
    ]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        records = manifest.get("experiments", {})
        for experiment_id, experiment in EXPERIMENTS.items():
            if experiment_id not in records:
                continue
            record = records[experiment_id]
            metrics = record.get("metrics") or {}
            writer.writerow(
                {
                    "experiment": experiment_id,
                    "dataset": experiment.dataset,
                    "rate": experiment.rate,
                    "gpu": record.get("gpu", ""),
                    "status": record.get("status", ""),
                    "epoch": metrics.get("epoch", ""),
                    "hits1": metrics.get("hits1", ""),
                    "hits5": metrics.get("hits5", ""),
                    "hits10": metrics.get("hits10", ""),
                    "mrr": metrics.get("mrr", ""),
                    "duration_seconds": record.get("duration_seconds", ""),
                    "comparison": _comparison(experiment, metrics),
                }
            )


def execute_queues(
    *,
    repo_root: Path,
    artifacts_dir: Path,
    queues: dict[str, list[str]],
    manifest: dict,
    force: bool = False,
    epochs: int = 1000,
    checkpoint: int = 10,
) -> bool:
    """Run one sequential experiment queue per GPU, concurrently."""

    manifest_lock = Lock()

    def run_queue(gpu_id: str, experiment_ids: list[str]) -> bool:
        for experiment_id in experiment_ids:
            if not run_experiment(
                repo_root=repo_root,
                artifacts_dir=artifacts_dir,
                experiment=EXPERIMENTS[experiment_id],
                gpu_id=gpu_id,
                manifest=manifest,
                manifest_lock=manifest_lock,
                force=force,
                epochs=epochs,
                checkpoint=checkpoint,
            ):
                return False
        return True

    outcomes: list[bool] = []
    with ThreadPoolExecutor(max_workers=max(1, len(queues))) as executor:
        futures = [
            executor.submit(run_queue, gpu_id, experiment_ids)
            for gpu_id, experiment_ids in queues.items()
        ]
        for future in as_completed(futures):
            outcomes.append(future.result())

    write_summary_csv(artifacts_dir / "summary.csv", manifest)
    return bool(outcomes) and all(outcomes)


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
