"""Optional experiment tracking with local, reproducible provenance."""

import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import os
import platform
import shutil
import sys
import zipfile
from numbers import Real
from pathlib import Path

_EXCLUDED = {
    ".git",
    ".worktrees",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "data",
    "log",
    "logs",
    "artifacts",
    "checkpoints",
    "save_pkl",
    "wandb",
}
_MANIFESTS = {
    "pyproject.toml",
    "uv.lock",
    "poetry.lock",
    "Pipfile",
    "Pipfile.lock",
    "setup.cfg",
    "environment.yml",
    "environment.yaml",
}
_EVAL_KEYS = ("hits1", "hits5", "hits10", "mr", "mrr")


def _source_archive(repo_root, output_dir=None):
    root = Path(repo_root).resolve()
    output = Path(output_dir).resolve() if output_dir else None
    files = []
    for directory, children, names in os.walk(root, followlinks=False):
        directory = Path(directory)
        children[:] = sorted(
            child
            for child in children
            if child.lower() not in _EXCLUDED
            and not (directory / child).is_symlink()
            and (output is None or (directory / child).resolve() != output)
        )
        if output is not None and directory == output:
            continue
        for name in names:
            path = directory / name
            if path.is_symlink():
                continue
            if (
                path.suffix in {".py", ".ipynb", ".sh"}
                or name in _MANIFESTS
                or (name.startswith("requirements") and path.suffix == ".txt")
            ):
                files.append(path)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
            info = zipfile.ZipInfo(
                path.relative_to(root).as_posix(), (1980, 1, 1, 0, 0, 0)
            )
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    return buffer.getvalue()


def source_fingerprint(repo_root: Path) -> str:
    """SHA256 of the portable source archive, independent of paths and mtimes."""
    return hashlib.sha256(_source_archive(repo_root)).hexdigest()


def _runtime_environment():
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": dict(
            sorted(
                (distribution.metadata["Name"], distribution.version)
                for distribution in importlib.metadata.distributions()
                if distribution.metadata["Name"]
            )
        ),
    }
    try:
        torch = importlib.import_module("torch")
        available = torch.cuda.is_available()
        environment["torch"] = {
            "version": str(torch.__version__),
            "cuda": torch.version.cuda,
            "cuda_available": available,
            "gpus": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
            if available
            else [],
        }
    except ImportError:
        environment["torch"] = None
    return environment


def _scalars(metrics):
    values = {}
    for key, value in metrics.items():
        if (
            not isinstance(key, str)
            or not isinstance(value, Real)
            or isinstance(value, bool)
        ):
            raise TypeError("Metrics must have string keys and scalar numeric values")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Metrics must be finite scalar values")
        values[key] = value
    return values


class ExperimentTracker:
    """Keep SDK imports lazy; never downgrade requested online/offline tracking."""

    @classmethod
    def start(
        cls,
        config: dict,
        repo_root: Path,
        output_dir: Path,
        mode: str = "disabled",
        project: str = "humea-reproduction",
        entity: str | None = None,
        name: str | None = None,
        job_type: str = "main",
    ):
        if mode not in {"disabled", "online", "offline"}:
            raise ValueError("Tracking mode must be disabled, online, or offline")
        tracker = cls()
        tracker.output_dir = Path(output_dir).resolve()
        tracker.output_dir.mkdir(parents=True, exist_ok=True)
        tracker.mode, tracker.run = mode, None
        tracker.project, tracker.entity = project, entity
        tracker.run_id, tracker.run_url = None, None
        tracker.source_sha256 = None
        tracker._finished, tracker._best_mrr = False, float("-inf")
        tracker.summary = {}
        try:
            archive = _source_archive(repo_root, tracker.output_dir)
            fingerprint = hashlib.sha256(archive).hexdigest()
            tracker.source_sha256 = fingerprint
            configuration = {
                **config,
                "provenance": {
                    "source_sha256": fingerprint,
                    "python": sys.version,
                    "platform": platform.platform(),
                },
            }
            (tracker.output_dir / "config.json").write_text(
                json.dumps(configuration, indent=2), encoding="utf-8"
            )
            (tracker.output_dir / "metrics.jsonl").write_text("", encoding="utf-8")
            (tracker.output_dir / "source.zip").write_bytes(archive)
            environment = _runtime_environment()
            environment["source_sha256"] = fingerprint
            environment_path = tracker.output_dir / "runtime-environment.json"
            environment_path.write_text(
                json.dumps(environment, indent=2), encoding="utf-8"
            )
            if mode != "disabled":
                try:
                    wandb = importlib.import_module("wandb")
                except ImportError as error:
                    raise RuntimeError(
                        "Requested tracking requires the wandb SDK; install wandb"
                    ) from error
                tracker.run = wandb.init(
                    project=project,
                    entity=entity,
                    name=name,
                    job_type=job_type,
                    mode=mode,
                    dir=str(tracker.output_dir),
                    config=configuration,
                )
                if tracker.run is None:
                    raise RuntimeError("wandb.init did not return a tracking run")
                tracker.run_id = tracker.run.id
                tracker.run_url = tracker.run.url
                tracker.run.define_metric("epoch")
                for namespace in ("train/*", "eval/*"):
                    tracker.run.define_metric(namespace, step_metric="epoch")
                for path in (environment_path, tracker.output_dir / "source.zip"):
                    tracker.run.save(
                        str(path), base_path=str(tracker.output_dir), policy="now"
                    )
            tracker._write_status("running")
        except Exception as error:
            try:
                tracker.finish(1)
            except Exception as cleanup_error:  # noqa: BLE001 - SDK cleanup must preserve the original error.
                error.add_note(f"Tracking cleanup also failed: {cleanup_error}")
            raise RuntimeError(
                f"Experiment tracking initialization failed: {error}"
            ) from error
        return tracker

    def _write_status(self, status):
        record = {
            "run_id": self.run_id,
            "run_url": self.run_url,
            "mode": self.mode,
            "project": self.project,
            "entity": self.entity,
            "status": status,
            "source_sha256": self.source_sha256,
            "summary": self.summary,
        }
        (self.output_dir / "tracking.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )

    def log_losses(self, epoch: int, metrics: dict[str, float]):
        values = _scalars(metrics)
        self._log(
            {
                "epoch": int(epoch),
                **{f"train/{key}": value for key, value in values.items()},
            }
        )

    def _log(self, values):
        with (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as history:
            history.write(json.dumps(values) + "\n")
        if self.run is not None:
            self.run.log(values)

    def attach_file(self, path: Path):
        """Preserve a raw run file locally and attach it to an enabled SDK run."""
        path = Path(path).resolve()
        destination = self.output_dir / path.name
        if path != destination:
            shutil.copyfile(path, destination)
        if self.run is not None:
            self.run.save(
                str(destination), base_path=str(self.output_dir), policy="now"
            )

    def log_evaluation(self, epoch: int, metrics: dict[str, float]):
        if set(metrics) != set(_EVAL_KEYS):
            raise ValueError(f"Evaluation metrics must contain exactly {_EVAL_KEYS}")
        values = _scalars(metrics)
        snapshot = {"epoch": int(epoch), **values}
        updates = {f"final/{key}": value for key, value in snapshot.items()}
        if values["mrr"] > self._best_mrr:
            self._best_mrr = values["mrr"]
            updates.update({f"best/{key}": value for key, value in snapshot.items()})
        self.summary.update(updates)
        self._log(
            {
                "epoch": int(epoch),
                **{f"eval/{key}": value for key, value in values.items()},
            }
        )
        if self.run is not None:
            self.run.summary.update(updates)

    def finish(self, exit_code: int = 0):
        if self._finished:
            return
        self._finished = True
        status = "completed" if exit_code == 0 else "failed"
        self._write_status(status)
        if self.run is not None:
            try:
                self.run.summary["status"] = status
                self.run.finish(exit_code=exit_code)
            except Exception:
                self._write_status("failed")
                raise
