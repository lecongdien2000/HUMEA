# HUMEA experiment reproduction

Execution instructions for the group's public [HUMEA reproduction fork](https://github.com/lecongdien2000/HUMEA), forked from the authors' [upstream HUMEA](https://github.com/mikumifa/HUMEA). Experimental results and method descriptions are presented in the accompanying report.

## Get the code

```bash
git clone https://github.com/lecongdien2000/HUMEA.git
cd HUMEA
```

Kaggle users should then follow [the notebook setup guide](kaggle/README.md). The entry point is [kaggle/humea_kaggle.ipynb](kaggle/humea_kaggle.ipynb); it invokes `kaggle_runner.py`, which launches `train.py` for the selected configuration.

For the baseline and loss ablations using the exact source attached to the recorded
main reproduction, use [kaggle/humea_reference_ablations.ipynb](kaggle/humea_reference_ablations.ipynb).
This is the ablation entry point; `humea_strict_replay.ipynb` is a separate diagnostic.
The reference notebook restores [reproduction/reference-source.zip](reproduction/reference-source.zip),
verifies its SHA256 and uses a separate orchestration directory so historical training
files remain unchanged. File checksums and recorded settings are in
[reproduction/reference-source.json](reproduction/reference-source.json).

Choose `MODE = "smoke"` for three 12-epoch checks, or `MODE = "campaign"` for those
checks followed by fresh 1,000-epoch baseline, `without7`, and `without8` runs.
Keep GPU T4, Internet and the `WANDB_API_KEY` secret enabled. The notebook uses seed 42,
batch size 512, alignment rate 0.2 and fusion dimension 512, with online W&B logging.
Run it in a fresh Kaggle session; it refuses to overwrite an existing source directory.
It stops on a failed job, changed tracked source fingerprint, or the 5.5-hour training
budget, and preserves evidence in `/kaggle/working/HUMEA-results.zip`.
This restores historical code/settings; it does not enforce deterministic GPU execution.

## Environment and hardware

Use Kaggle with a CUDA-enabled NVIDIA Tesla T4 GPU (16 GB). Run each configuration separately with `MODE = "single"` to stay within Kaggle's session limits. The runner supports two GPUs for independent experiment queues, but a combined six-experiment job can exceed the session limit.

- Recorded Kaggle environment: Linux x86_64, Python 3.12.13, PyTorch 2.10.0+cu128 and CUDA 12.8.
- Recorded scientific packages: NumPy 2.0.2, SciPy 1.16.3 and scikit-learn 1.6.1. Preserve Kaggle's preinstalled CUDA-enabled PyTorch and scientific stack; if its image changes, compare it with the saved environment before treating the rerun as the same environment.
- Notebook additions are pinned in `requirements-kaggle.txt`: Loguru 0.7.3, gdown 5.2.0 and W&B 0.30.0.
- Each tracked execution saves exact package versions, Python, PyTorch, CUDA and GPU information in `runtime-environment.json`. Use the record attached to the corresponding W&B run when reproducing a reported experiment.
- Local CPU tests check code behavior; CPU execution is not the recommended reproduction hardware.

Follow [the Kaggle guide](kaggle/README.md) and import [kaggle/humea_kaggle.ipynb](kaggle/humea_kaggle.ipynb).

For a local Linux machine with an NVIDIA GPU, install [uv](https://docs.astral.sh/uv/), select Python 3.12, and use the committed dependency lock:

```bash
uv sync --frozen
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

A compatible NVIDIA driver is required. This locked local environment is distinct from Kaggle's preinstalled environment; consult the environment record for the experiment being reproduced.

## Source revision

Use the source revision referenced in the report or W&B run. Each tracked run saves `source.zip` and its SHA256 fingerprint, including when its execution directory has no Git metadata. Cite the group's GitHub repository and exact source revision in the report.

The source archive attached to the group's main runs has SHA256 `545b95884f15dfc94ceae0d985cbe0968da5cc733f718ab611be9688e2e51b34`. Use that attached archive for the exact recorded source; a new Git archive includes updated documentation and will have a different archive hash.

To package source for Kaggle, commit the intended changes and run:

```bash
git archive --format=zip --output HUMEA-kaggle-bundle.zip HEAD
```

This includes committed files only. Use the same bundle identified by the corresponding experiment.

## Data preparation

Download the authors' [processed data archive](https://drive.google.com/file/d/1a3aou1qe7Yzq6y_khyTq1La_kf3UI7sC/view?usp=drive_link). Use the released features directly; preprocessing scripts are not needed for this workflow. Extract it so the repository contains:

```text
data/mmkb-datasets/FB15K_DB15K/
data/mmkb-datasets/FB15K_YAGO15K/
```

Keep the released files and split settings unchanged. Validation reports missing required files before training.

## W&B recording

Each experiment creates a distinct W&B run. The runner defaults to `offline`, recording history locally for later synchronization. Online recording requires an account/team, project and authentication:

```bash
uv run wandb login
uv run python kaggle_runner.py --wandb-mode online --wandb-entity YOUR_ACCOUNT_OR_TEAM --wandb-project humea-reproduction single --experiment db15k-20
```

On Kaggle, set `WANDB_MODE = "online"`, configure `WANDB_ENTITY`, and enable a secret named `WANDB_API_KEY`. Keep the key out of the notebook and repository. Internet is required for installation, downloads and online logging.

Offline recording does not provide a public experiment link. Download the results archive and synchronize each `offline-run-*` directory:

```bash
uv run wandb login
uv run wandb sync --entity YOUR_ACCOUNT_OR_TEAM --project humea-reproduction PATH_TO_OFFLINE_RUN
```

Before submission, make the project publicly viewable or create a view-only report link for a private/team project. Check the link while signed out. Link the relevant main/ablation run set or report from the matching report section. See [W&B report sharing](https://docs.wandb.ai/models/reports/cross-project-reports).

`--wandb-mode disabled` is for debugging and does not meet the W&B requirement. Smoke runs are marked `smoke`, normal runner executions `main`, and direct executions with nonzero `--without` are marked `ablation`.

## Execution

For a strict repeatability check, use [kaggle/humea_strict_replay.ipynb](kaggle/humea_strict_replay.ipynb),
or run the following after preparing the same data and W&B credentials:

```bash
python replay_runner.py --epochs 35 --wandb-mode online --wandb-entity YOUR_ACCOUNT_OR_TEAM
```

This launches two separate DB15K-20 baseline processes on GPU 0 with seed 42,
strict deterministic algorithms, and unchanged loss/fusion settings. Use a fresh
`--artifacts-dir` for each pair. Unsupported deterministic operations fail explicitly.
The exit code is zero only if inputs, source, settings, initial state and every
epoch's model/optimizer/RNG/order hashes, routing, losses and recorded evaluations match.
It does not start a full experiment automatically.

The `artifacts/strict-replay` directory contains the comparison, raw logs and each
run's `replay` directory. Input checksums and routing/state traces are attached to
W&B; large diagnostic checkpoints stay in the Kaggle output archive. Checkpoints
include initialization and epochs 20, 22, 23 and 24. Download that archive before
discarding the Kaggle session.

For a direct diagnostic run, append `--strict_reproducibility --replay_diagnostics`
to the corresponding `train.py` command. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` and
`PYTHONHASHSEED=42` before starting Python. To resume, use the same command/settings,
add `--resume_diagnostic PATH_TO_CHECKPOINT`, and use a fresh tracking output directory.
Resume only checkpoints from your own trusted runs; these include Python state.
Repeatability must be rechecked when the hardware or software environment changes.

Run from the repository root. On Kaggle use `python`; in a local locked environment prefix commands with `uv run`.

```bash
python kaggle_runner.py validate
python kaggle_runner.py --wandb-mode offline smoke
python kaggle_runner.py --wandb-mode online --wandb-entity YOUR_ACCOUNT_OR_TEAM single --experiment db15k-20
```

| Experiment ID | Dataset | Alignment rate | Fusion dimension |
|---|---|---:|---:|
| `db15k-20` | FB15K_DB15K | 0.2 | 512 |
| `db15k-50` | FB15K_DB15K | 0.5 | 0 |
| `db15k-80` | FB15K_DB15K | 0.8 | 0 |
| `yago15k-20` | FB15K_YAGO15K | 0.2 | 128 |
| `yago15k-50` | FB15K_YAGO15K | 0.5 | 0 |
| `yago15k-80` | FB15K_YAGO15K | 0.8 | 0 |

The full commands use seed 42, batch 512, learning rate 0.0005, 1,000 epochs and evaluation every 10 epochs. Change the selected experiment ID to run another configuration. Keep these settings when reproducing the reported main experiments. Run each ID in its own Kaggle session; `smoke` runs only 12 epochs and is a setup check.

On a machine with enough uninterrupted GPU time, all six can also be scheduled with:

```bash
python kaggle_runner.py --wandb-mode online --wandb-entity YOUR_ACCOUNT_OR_TEAM all
```

With two visible GPUs the datasets have separate queues; with one GPU experiments execute sequentially. A GPU's queue stops when an experiment fails.

For a component-removal experiment, run `train.py` with full baseline arguments and the selected `--without` flag:

```bash
python train.py --file_dir data/mmkb-datasets/FB15K_DB15K --rate 0.2 --seed 42 --lr 0.0005 --epochs 1000 --hidden_units 300,300,300 --check_point 10 --bsize 512 --il_start 500 --csls --csls_k 3 --tau_cl 0.1 --tau_al 4.0 --fusion_weight_dim 512 --al_loss 0.1 --cl_loss 1.0 --without 1
```

For direct execution, configure `WANDB_MODE`, `WANDB_ENTITY` and `WANDB_PROJECT` in the environment. CLI flag mapping: 1 image, 2 graph, 3 relation, 4 relation text, 5 attribute, 6 attribute text, 7 inner-view alignment, 8 cross-view alignment. This is an execution example; consult the report for which ablation configurations were actually performed.

## Outputs and interruption handling

```text
artifacts/
  logs/                       # complete subprocess logs
  manifest.json               # status, command, timestamps and W&B URL
  summary.csv                 # extracted summary and online W&B URL
  tracking/<attempt>/
    config.json
    metrics.jsonl
    runtime-environment.json
    source.zip
    tracking.json
    wandb/                    # SDK records
```

The notebook exports `HUMEA-results.zip`, including tracking records. Online runs attach source/environment evidence and the training log to W&B. Keep the archive and W&B records for the report.

Resume works at experiment granularity. Successful experiments with parsed metrics are skipped. An interrupted experiment starts at epoch 0 because training does not save model checkpoints. Restore `artifacts/` before resuming. To rerun a completed configuration:

```bash
python kaggle_runner.py --force single --experiment db15k-20
```

When W&B recording is requested, legacy results without completed W&B evidence and disabled-mode results are rerun. Completed offline records can be reused in offline mode and synchronized later. Online mode requires a completed online run in the configured project/account; it does not silently reuse an unsynchronized offline run.

## Checks and troubleshooting

```bash
uv run --with pytest python -m pytest -q
python kaggle_runner.py --help
```

- Missing input: inspect the path reported by `validate` and check archive nesting.
- No GPU: enable a Kaggle GPU or verify the local CUDA PyTorch/driver installation.
- W&B initialization failure: check the SDK, entity/project, Internet and credentials. Tracking does not silently fall back to another mode.
- Training failure: inspect `artifacts/logs/`; failures are recorded separately from completed runs.
- Out of memory: use the exact patched source snapshot before changing reproduction settings.

## Attribution

Original code: [mikumifa/HUMEA](https://github.com/mikumifa/HUMEA), accompanying *On Modality Weighting and Specificity for Multi-Modal Entity Alignment*. The original authors' commits are retained in this fork; subsequent reproduction commits record the group's additions. Consult the authors' distribution terms before redistributing datasets or pretrained features.
