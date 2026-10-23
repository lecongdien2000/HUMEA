# Run HUMEA on Kaggle

Start with the [root reproduction guide](../README.md) for environment, commands, configuration, W&B sharing and interruption handling.

## Prepare source and data

1. Clone the public fork using the root guide. Select the source revision associated with the reported experiment and create `HUMEA-kaggle-bundle.zip` using `git archive`. For the exact recorded archive, download the run's `source.zip`, verify its SHA256 against the root guide, and rename it `HUMEA-kaggle-bundle.zip`.
2. Upload the bundle to a Kaggle Dataset and attach it to the notebook.
3. Attach the authors' processed `data.zip`, or an extracted Dataset whose root contains `mmkb-datasets/`. The notebook can download the original archive with Internet and `DOWNLOAD_DATA_IF_MISSING` enabled.
4. Import [humea_kaggle.ipynb](humea_kaggle.ipynb).

The notebook discovers attached source files or the ZIP automatically. Data must contain both `FB15K_DB15K` and `FB15K_YAGO15K` under `mmkb-datasets/` for full validation.

## Configure the session

Select an NVIDIA Tesla T4 accelerator. One GPU runs one queue; T4 x2 can run two dataset queues concurrently. Preserve Kaggle's preinstalled CUDA-enabled PyTorch and scientific packages. Setup installs pinned additions from `requirements-kaggle.txt`; Internet is required unless those packages are already supplied locally.

```python
MODE = "single"                  # validate | smoke | single | all
EXPERIMENT = "db15k-20"
BATCH_SIZE = 512
DOWNLOAD_DATA_IF_MISSING = True
WANDB_MODE = "online"            # offline supports later synchronization
WANDB_PROJECT = "humea-reproduction"
WANDB_ENTITY = "diencongle"
```

For online recording, add `WANDB_API_KEY` in Kaggle Secrets and grant the notebook access. The notebook reads it without printing or embedding it. Enable Internet for W&B synchronization.

The group's destination is the public [diencongle/humea-reproduction](https://forge.coreweave.com/wandb/diencongle/humea-reproduction) project. Verify each experiment or report link in a signed-out browser before including it in the report.

Run the notebook as follows:

1. Set `MODE = "validate"` and run all cells. Check that source, data and GPU validation succeed.
2. Set `MODE = "smoke"` and run all cells. This performs a 12-epoch setup check; verify that W&B recording succeeds.
3. Set `MODE = "single"` and choose an `EXPERIMENT` from the root guide. Keep `BATCH_SIZE = 512` and online W&B recording enabled.
4. Save a notebook version with **Save & Run All** for the full 1,000-epoch execution. This runs all cells in a fresh session, as described in [Kaggle's notebook documentation](https://www.kaggle.com/docs/notebooks). Confirm completion in the saved version's logs and download its output archive.
5. Repeat step 3â€“4 for each remaining experiment ID, using a separate saved execution for each configuration.

Each run logs scalar losses per epoch and evaluation history every 10 epochs, with source/environment evidence. Avoid `MODE = "all"` on Kaggle: the combined queue can exceed a session's time limit.

## Save and share evidence

Download `/kaggle/working/HUMEA-results.zip` after execution. It includes logs, manifests, configuration, W&B records, source snapshots and environment versions. The failure path also attempts to package available evidence.

Offline history must be synchronized using `wandb sync` before it provides a public experiment link. Follow the root guide's sharing instructions. A Kaggle notebook link or local CSV alone does not meet the lecturer's W&B requirement.

Keep the source revision, notebook configuration, dataset source and W&B run URL together. Cite the group's GitHub repository in the report's reproduction section and the corresponding public W&B links in each results section.

## Session limits

Interrupted experiments restart at epoch 0. Other successful experiments can be skipped if their `artifacts/` directory is restored. The notebook creates a clean source directory, so restore previous artifacts after the source-copy cell and before training.

For errors, inspect `artifacts/logs/<experiment-id>.log` and `manifest.json`. Document changes to batch size, temperatures or other reproduction settings in the report.

## Matched loss ablations

Import [humea_loss_ablations.ipynb](humea_loss_ablations.ipynb) and pin `SOURCE_REVISION` to the Git commit being reproduced. Enable the existing `WANDB_API_KEY` secret. The notebook reuses attached processed data and downloads the public source fork; it does not use the older attached source bundle.

`MODE = "campaign"` runs 12-epoch checks for `without` 0, 7 and 8, then the three full jobs sequentially on GPU 0. Each full job uses DB15K-20, seed 42, batch 512, fusion dimension 512 and 1,000 epochs. Only `--without` changes: 0 baseline, 7 removes unimodal alignment, 8 removes unimodal distillation. These are controlled comparisons against our baseline; the upstream ablation shell script uses different fusion/split settings.

Each job has separate logs, manifest entries and online W&B records under group `db15k20-loss-ablation-20261023`. Filter out `smoke-` runs in the report. The campaign stops on failure and exports available evidence to `/kaggle/working/HUMEA-results.zip` after each job. Training has a 5.5-hour budget, excluding initial source/data setup; verify sufficient quota before launch. Do not run another GPU notebook concurrently.

For one job from a configured shell, use:

```bash
python kaggle_runner.py --wandb-mode online --wandb-entity diencongle --without 7 single --experiment db15k-20
```

Use `--without 8` for distillation removal and `--without 0` for baseline. The runner gives ablations separate record IDs and leaves their baseline-paper comparison field empty. Put the measured comparisons and public W&B experiment links in the report.
