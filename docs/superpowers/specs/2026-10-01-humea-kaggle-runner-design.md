# HUMEA Kaggle Reproduction Runner Design

## Purpose

Add a Kaggle-ready workflow to the upstream HUMEA repository that can validate the released precomputed features, smoke-test the training pipeline, run one selected main-table experiment, or run all six main-table experiments. The workflow must preserve the authors' model and optimization code so that results remain comparable with the paper.

## Scope

The project will support the six non-iterative main-result configurations, all using seed 42 and 1,000 epochs:

| ID | Dataset | Alignment rate | Fusion weight dimension |
|---|---|---:|---:|
| `db15k-20` | `FB15K_DB15K` | 0.2 | 512 |
| `db15k-50` | `FB15K_DB15K` | 0.5 | 0 |
| `db15k-80` | `FB15K_DB15K` | 0.8 | 0 |
| `yago15k-20` | `FB15K_YAGO15K` | 0.2 | 128 |
| `yago15k-50` | `FB15K_YAGO15K` | 0.5 | 0 |
| `yago15k-80` | `FB15K_YAGO15K` | 0.8 | 0 |

The workflow will reuse the authors' released image, attribute, and relation feature files. It will not regenerate ResNet or language-model embeddings, run pseudo-label experiments, reproduce ablations, or change HUMEA's neural architecture.

## Architecture

The Kaggle notebook will be a thin interface over a repository-owned Python runner. The runner will contain the experiment registry, dataset validation, command construction, GPU scheduling, result-state tracking, and summary generation. The notebook will handle only Kaggle-specific setup: locating or downloading the archive, extracting it, checking the GPU environment, selecting a mode, invoking the runner, and packaging output.

This separation keeps orchestration testable outside Kaggle. Live validation later required memory-equivalent T4 compatibility changes: select mini-batch rows before row-wise normalization in `loss.py` and gradient-checkpoint the loss calls plus main multimodal-encoder forward. `model.py`, `layers.py`, and `utils.py` remain unchanged.

## User Modes

The runner will expose four commands:

- `validate`: verify Python dependencies, CUDA visibility, dataset directories, and every feature/input file required by the selected experiments.
- `smoke`: run `db15k-20` for 12 epochs with evaluation at epoch 10.
- `single EXPERIMENT_ID`: run one full 1,000-epoch experiment.
- `all`: schedule all six full experiments across available GPUs.

The notebook will default to `single db15k-20`, making the first paper experiment the safe initial action. Running all experiments will require changing one visible configuration variable to `all`.

## GPU Scheduling

For `all`, the runner will create at most one training process per visible CUDA device. On Kaggle T4 x2, GPU 0 will process the three `FB15K_DB15K` configurations sequentially while GPU 1 processes the three `FB15K_YAGO15K` configurations sequentially. Each subprocess will receive a single device through `CUDA_VISIBLE_DEVICES` and will execute `train.py` directly with the paper's arguments.

If only one GPU is visible, all six experiments will run sequentially. CPU-only full training will be rejected unless the user explicitly overrides the safety check.

The expected wall time is 4.5–6.5 hours on Kaggle T4 x2, including one-time setup and result export. A single first experiment is expected to take 45–75 minutes after setup.

## Data Flow

1. The user attaches the authors' `data.zip` as a private Kaggle Dataset or enables Internet access for a one-time download.
2. The notebook finds an attached archive first; if none exists and downloading is enabled, it downloads the exact Google Drive file named in the HUMEA README.
3. The notebook extracts the archive under `/kaggle/working/HUMEA/data` only when the required dataset directories are absent.
4. The runner validates both dataset directories and required files before launching training.
5. Each training process writes combined stdout/stderr to a unique file under `artifacts/logs/` while the authors' Loguru output continues to be written under `log/`.
6. On success, the runner records experiment ID, command, GPU, start/end timestamps, duration, exit status, and parsed best metrics in `artifacts/manifest.json` and `artifacts/summary.csv`.
7. The notebook packages `artifacts/` and the matching author logs into a ZIP in `/kaggle/working` for Kaggle output persistence.

## Resume Semantics

The authors' training loop does not implement model checkpoint/resume. The runner therefore resumes at experiment granularity:

- A completed experiment with a successful manifest entry and parsed best metrics is skipped by default.
- A failed or interrupted experiment is rerun from epoch 0.
- `--force` reruns a completed experiment.

This avoids altering training behavior while preventing completed multi-hour runs from being repeated unnecessarily.

## Error Handling

Validation will fail before training with an actionable message when:

- a dataset directory or required file is missing;
- PyTorch or another runtime dependency cannot be imported;
- full training is requested without CUDA;
- an experiment ID is unknown;
- an output directory is not writable.

If a child training process fails, the runner will preserve its log, record the nonzero exit code, stop that GPU's queue, allow the other GPU queue to finish its current experiment, and exit nonzero. It will never report a failed run as completed.

## Result Verification

The summary parser will read the best-result line emitted by `train.py` and write Hits@1, Hits@5, Hits@10, MRR, and best epoch. The notebook will display the summary alongside the paper targets. A result within 0.01 absolute of the corresponding published metric will be labeled `close`; larger differences will be shown without being hidden or automatically retried.

The first experiment target is Hits@1 0.5118, Hits@5 0.6997, Hits@10 0.7643, and MRR 0.5980.

## Files

- `kaggle/humea_kaggle.ipynb`: reproducible Kaggle setup, configuration, execution, verification, and artifact export.
- `kaggle/README.md`: upload/attach instructions, notebook settings, modes, runtime estimates, and troubleshooting.
- `kaggle_runner.py`: experiment registry, validation, scheduling, manifest handling, metric parsing, and CLI.
- `tests/test_kaggle_runner.py`: unit tests for registry values, command construction, validation, resume decisions, metric parsing, and GPU queue assignment.
- `.gitignore`: ignore downloaded datasets and generated Kaggle artifacts without hiding source files.

## Testing Strategy

Unit tests will mock subprocess execution and CUDA discovery so they run without the 1.66 GiB dataset or a GPU. A local CLI validation test will use temporary fixture directories. The generated notebook will be parsed as JSON and executed only through its setup-independent cells during automated checks. The final verification will include `pytest`, CLI help, notebook JSON validation, and a clean Git diff review.

## Acceptance Criteria

- A Kaggle user can attach or download the authors' processed archive and run the first experiment without editing HUMEA model code.
- `single db15k-20` emits a complete log and structured result summary.
- `all` uses two GPUs concurrently when two are visible and falls back to one GPU sequentially.
- Successfully completed experiments are not repeated unless `--force` is supplied.
- Failures are visible in both process exit status and the manifest.
- The notebook and README contain exact, copyable setup instructions with no dependency on `uv`.
- All orchestration tests pass without downloading the dataset or using a GPU.
