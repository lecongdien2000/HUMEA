import importlib
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class TrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import optional runtime dependencies before patch.dict restores sys.modules;
        # torch's native extension cannot be unloaded and imported again safely.
        importlib.import_module("experiment_tracking")._runtime_environment()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "repo"
        self.repo.mkdir()
        (self.repo / "train.py").write_text("print(1)\n")
        self.output = self.repo / "artifacts" / "run"

    def tracker_module(self):
        return importlib.import_module("experiment_tracking")

    def fake_sdk(self):
        run = SimpleNamespace(
            id="abc",
            url="https://example/run/abc",
            summary={},
            logs=[],
            definitions=[],
            artifacts=[],
            finishes=[],
        )
        run.log = lambda values: run.logs.append(values)
        run.define_metric = lambda *args, **kwargs: run.definitions.append(
            (args, kwargs)
        )
        run.save = lambda path, **kwargs: run.artifacts.append(Path(path))
        run.finish = lambda **kwargs: run.finishes.append(kwargs)
        sdk = SimpleNamespace(init=lambda **kwargs: run)
        return sdk, run

    def test_disabled_requires_no_sdk_and_finishes_idempotently(self):
        with patch.dict(sys.modules, {"wandb": None}):
            tracker = self.tracker_module().ExperimentTracker.start(
                {}, self.repo, self.output
            )
            tracker.log_losses(0, {"total": 1.0})
            tracker.log_evaluation(
                0, {"hits1": 0.1, "hits5": 0.2, "hits10": 0.3, "mr": 10, "mrr": 0.2}
            )
            tracker.finish()
            tracker.finish(1)
        metadata = json.loads((self.output / "tracking.json").read_text())
        self.assertEqual(metadata["status"], "completed")
        self.assertEqual(metadata["mode"], "disabled")
        self.assertIsNone(tracker.run_id)
        self.assertTrue((self.output / "runtime-environment.json").exists())

    def test_logs_metrics_and_best_snapshot_from_same_epoch(self):
        sdk, run = self.fake_sdk()
        with patch.dict(sys.modules, {"wandb": sdk}):
            tracker = self.tracker_module().ExperimentTracker.start(
                {"seed": 42}, self.repo, self.output, mode="offline"
            )
            tracker.log_losses(2, {"total": 2})
            tracker.log_evaluation(
                2, {"hits1": 0.1, "hits5": 0.2, "hits10": 0.3, "mr": 10, "mrr": 0.4}
            )
            tracker.log_evaluation(
                3, {"hits1": 0.9, "hits5": 0.9, "hits10": 0.9, "mr": 2, "mrr": 0.3}
            )
            tracker.log_evaluation(
                4, {"hits1": 0.8, "hits5": 0.8, "hits10": 0.8, "mr": 3, "mrr": 0.4}
            )
            tracker.finish()
        self.assertEqual(run.logs[0], {"epoch": 2, "train/total": 2.0})
        self.assertEqual(len(run.logs), 4)
        self.assertEqual(run.summary["best/epoch"], 2)
        self.assertEqual(run.summary["best/hits1"], 0.1)
        self.assertEqual(run.summary["final/epoch"], 4)
        self.assertEqual(run.summary["final/hits1"], 0.8)
        self.assertIn((("eval/*",), {"step_metric": "epoch"}), run.definitions)
        self.assertEqual(len(run.artifacts), 2)
        self.assertEqual(run.finishes, [{"exit_code": 0}])

    def test_failed_finish_records_failure(self):
        sdk, run = self.fake_sdk()
        with patch.dict(sys.modules, {"wandb": sdk}):
            tracker = self.tracker_module().ExperimentTracker.start(
                {}, self.repo, self.output, mode="online"
            )
            tracker.finish(1)
        self.assertEqual(
            json.loads((self.output / "tracking.json").read_text())["status"], "failed"
        )
        self.assertEqual(run.summary["status"], "failed")

    def test_requested_tracking_cannot_silently_fall_back(self):
        with (
            patch.dict(sys.modules, {"wandb": None}),
            self.assertRaisesRegex(RuntimeError, "wandb"),
        ):
            self.tracker_module().ExperimentTracker.start(
                {}, self.repo, self.output, mode="online"
            )
        self.assertEqual(
            json.loads((self.output / "tracking.json").read_text())["status"], "failed"
        )

    def test_authentication_failure_is_recorded(self):
        def fail(**kwargs):
            raise ValueError("authentication failed")

        with (
            patch.dict(sys.modules, {"wandb": SimpleNamespace(init=fail)}),
            self.assertRaisesRegex(RuntimeError, "authentication failed"),
        ):
            self.tracker_module().ExperimentTracker.start(
                {}, self.repo, self.output, mode="online"
            )
        self.assertEqual(
            json.loads((self.output / "tracking.json").read_text())["status"], "failed"
        )

    def test_cleanup_failure_does_not_hide_initialization_error(self):
        sdk, run = self.fake_sdk()

        def fail_setup(*args, **kwargs):
            raise ValueError("source upload failed")

        def fail_finish(**kwargs):
            raise ValueError("cleanup failed")

        run.save, run.finish = fail_setup, fail_finish
        with (
            patch.dict(sys.modules, {"wandb": sdk}),
            self.assertRaisesRegex(RuntimeError, "source upload failed"),
        ):
            self.tracker_module().ExperimentTracker.start(
                {}, self.repo, self.output, mode="online"
            )
        self.assertEqual(
            json.loads((self.output / "tracking.json").read_text())["status"], "failed"
        )

    def test_finish_failure_persists_failed_status(self):
        sdk, run = self.fake_sdk()

        def fail(**kwargs):
            raise ValueError("network failure")

        run.finish = fail
        with patch.dict(sys.modules, {"wandb": sdk}):
            tracker = self.tracker_module().ExperimentTracker.start(
                {}, self.repo, self.output, mode="online"
            )
            with self.assertRaisesRegex(ValueError, "network failure"):
                tracker.finish()
            tracker.finish()
        self.assertEqual(
            json.loads((self.output / "tracking.json").read_text())["status"], "failed"
        )

    def test_metrics_reject_tensors_and_nonfinite_scalars(self):
        tracker = self.tracker_module().ExperimentTracker.start(
            {}, self.repo, self.output
        )
        for value in ([1], float("nan"), float("inf"), True):
            with self.assertRaises((ValueError, TypeError)):
                tracker.log_losses(1, {"loss": value})
        tracker.finish()

    def test_local_history_config_and_summary_survive_failed_run(self):
        tracker = self.tracker_module().ExperimentTracker.start(
            {"seed": 42}, self.repo, self.output
        )
        tracker.log_losses(2, {"total": 1})
        tracker.log_evaluation(
            2, {"hits1": 0.1, "hits5": 0.2, "hits10": 0.3, "mr": 4, "mrr": 0.5}
        )
        tracker.finish(1)
        configuration = json.loads((self.output / "config.json").read_text())
        self.assertEqual(configuration["seed"], 42)
        history = [
            json.loads(line)
            for line in (self.output / "metrics.jsonl").read_text().splitlines()
        ]
        self.assertEqual(history[0], {"epoch": 2, "train/total": 1.0})
        self.assertEqual(history[1]["eval/mrr"], 0.5)
        tracking = json.loads((self.output / "tracking.json").read_text())
        self.assertEqual(tracking["summary"]["best/epoch"], 2)
        self.assertEqual(tracking["summary"]["final/mrr"], 0.5)
        self.assertEqual(
            tracking["source_sha256"], configuration["provenance"]["source_sha256"]
        )
        self.assertEqual(tracking["status"], "failed")

    def test_attaches_raw_training_log_and_shell_entrypoints(self):
        (self.repo / "run.sh").write_text("python train.py\n")
        raw_log = Path(self.temp.name) / "training.log"
        raw_log.write_text("raw training output\n")
        sdk, run = self.fake_sdk()
        with patch.dict(sys.modules, {"wandb": sdk}):
            tracker = self.tracker_module().ExperimentTracker.start(
                {}, self.repo, self.output, mode="offline"
            )
            tracker.attach_file(raw_log)
            tracker.finish()
        self.assertEqual(
            (self.output / "training.log").read_bytes(), raw_log.read_bytes()
        )
        self.assertIn((self.output / "training.log").resolve(), run.artifacts)
        with zipfile.ZipFile(self.output / "source.zip") as archive:
            self.assertIn("run.sh", archive.namelist())

    def test_source_is_deterministic_and_excludes_data_and_outputs(self):
        module = self.tracker_module()
        for directory in ("data", "log", ".git", ".worktrees", ".venv", "artifacts"):
            folder = self.repo / directory
            folder.mkdir(exist_ok=True)
            (folder / "private.py").write_text("secret")
        (self.repo / ".env").write_text("API_KEY=private")
        (self.repo / "pyproject.toml").write_text("[project]\n")
        initial = module.source_fingerprint(self.repo)
        other = Path(self.temp.name) / "other"
        other.mkdir()
        for name in ("train.py", "pyproject.toml"):
            (other / name).write_bytes((self.repo / name).read_bytes())
        self.assertEqual(initial, module.source_fingerprint(other))
        os.utime(self.repo / "train.py", (100, 100))
        self.assertEqual(initial, module.source_fingerprint(self.repo))
        tracker = module.ExperimentTracker.start({}, self.repo, self.output)
        with zipfile.ZipFile(self.output / "source.zip") as archive:
            self.assertEqual(archive.namelist(), ["pyproject.toml", "train.py"])
        import hashlib

        self.assertEqual(
            initial,
            hashlib.sha256((self.output / "source.zip").read_bytes()).hexdigest(),
        )
        (self.repo / "train.py").write_text("print(2)\n")
        self.assertNotEqual(initial, module.source_fingerprint(self.repo))
        tracker.finish()


if __name__ == "__main__":
    unittest.main()
