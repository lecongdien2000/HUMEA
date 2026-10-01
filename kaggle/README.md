# Reproducing HUMEA on Kaggle

This workflow runs the six non-iterative main-result experiments from the HUMEA paper. It reuses the authors' released image, attribute, and relation features; it still trains the HUMEA alignment model from scratch for 1,000 epochs.

The model files and training loop are unchanged. `kaggle_runner.py` adds preflight checks, experiment selection, two-GPU scheduling, experiment-level resume, logs, and a CSV summary.

## Expected runtime

| Work | Kaggle T4 estimate |
|---|---:|
| Setup and archive extraction | 10–30 minutes |
| 12-epoch smoke test | 5–15 minutes |
| First experiment (`db15k-20`) | 45–75 minutes |
| All six on one T4 | 7–10 hours |
| All six on T4 x2 | 4–6 hours |
| Full T4 x2 workflow including setup/export | 4.5–6.5 hours |

Times are estimates. Kaggle load, CUDA/PyTorch versions, and evaluation time can change them.

## 1. Create the source bundle

From the completed HUMEA repository, create a ZIP containing only tracked files:

```bash
git archive --format=zip --output HUMEA-kaggle-bundle.zip HEAD
```

Create a private Kaggle Dataset and upload `HUMEA-kaggle-bundle.zip`. The notebook supports either Kaggle-expanded source files or the attached ZIP itself, so the Dataset slug does not matter.

## 2. Attach the processed data

Download the authors' [`data.zip`](https://drive.google.com/file/d/1a3aou1qe7Yzq6y_khyTq1La_kf3UI7sC/view?usp=drive_link) and upload it to another private Kaggle Dataset. Attach both private Datasets to the notebook.

Attaching the data is recommended because it avoids a Google Drive download on every clean Kaggle session. The notebook supports three data layouts, in this order:

1. An attached, already-extracted directory containing `mmkb-datasets/`.
2. An attached file named `data.zip`.
3. A one-time Google Drive download when Internet is enabled.

After setup, the runner requires these directories:

```text
/kaggle/working/HUMEA/data/mmkb-datasets/FB15K_DB15K
/kaggle/working/HUMEA/data/mmkb-datasets/FB15K_YAGO15K
```

Do not run `data_process/` for the main reproduction. The archive already contains the feature dictionaries consumed by `train.py`.

## 3. Import and configure the notebook

Import [`humea_kaggle.ipynb`](humea_kaggle.ipynb) into Kaggle and choose a GPU accelerator. Select T4 x2 when it is available. Internet can remain off when both source and data Datasets are attached.

The configuration cell contains:

```python
MODE = "single"       # validate | smoke | single | all
EXPERIMENT = "db15k-20"
BATCH_SIZE = 512          # Paper setting; checkpointed for Kaggle T4
DOWNLOAD_DATA_IF_MISSING = True
```

Use the modes in this order:

1. `validate`: check dependencies, all released inputs, and visible GPUs without training.
2. `smoke`: run `db15k-20` for 12 epochs and evaluate at epoch 10.
3. `single`: run the experiment selected by `EXPERIMENT` for 1,000 epochs.
4. `all`: run all six 1,000-epoch configurations.

The accepted experiment IDs are:

```text
db15k-20  db15k-50  db15k-80
yago15k-20  yago15k-50  yago15k-80
```

For `all` on two visible GPUs, GPU 0 runs the three DB15K configurations sequentially while GPU 1 runs the three YAGO15K configurations sequentially. With one GPU, all six run sequentially.

## Direct runner commands

The notebook calls the following commands from `/kaggle/working/HUMEA`. They can also be used in a terminal:

```bash
python kaggle_runner.py validate
python kaggle_runner.py smoke
python kaggle_runner.py --batch-size 512 single --experiment db15k-20
python kaggle_runner.py --batch-size 512 all
```

Use direct Python execution on Kaggle. Do not run `uv sync`: Kaggle already supplies the CUDA-enabled PyTorch stack, and recreating the lockfile environment downloads an unnecessary CUDA stack.

## Outputs and resume behavior

Each run produces:

```text
artifacts/
├── logs/<experiment-id>.log
├── manifest.json
└── summary.csv
```

The notebook also creates `/kaggle/working/HUMEA-results.zip`. Save a Kaggle notebook version with outputs enabled before ending the session.

The runner skips a successful experiment only when its manifest entry contains parsed final metrics. Use global option `--force` before the subcommand to rerun it:

```bash
python kaggle_runner.py --force single --experiment db15k-20
```

HUMEA does not save training checkpoints. A failed or interrupted experiment restarts at epoch 0, while other experiments already completed in `manifest.json` remain reusable if the Kaggle output is restored into the next session.

## Expected first result

The paper target for `db15k-20` is:

| Hits@1 | Hits@5 | Hits@10 | MRR |
|---:|---:|---:|---:|
| 0.5118 | 0.6997 | 0.7643 | 0.5980 |

`summary.csv` records the batch size and labels a result `close` when all four metrics are within 0.01 absolute of their paper targets. This label is diagnostic, not a replacement for reporting the actual values.

The upstream losses originally normalized every entity embedding before selecting each mini-batch; live T4 tests showed that this retained redundant full-table autograd graphs. The included `loss.py` compatibility patch selects the batch rows before applying the same row-wise L2 normalization. These operations are mathematically equivalent and use less memory. The authors also retain every mini-batch loss graph until one final backward pass. The runner therefore applies PyTorch non-reentrant gradient checkpointing to the loss calls, each MI estimator after its expert pair has been selected, and the main multimodal-encoder forward, and uses CUDA's asynchronous allocator to avoid split-block fragmentation during recomputation. Finally, the MoE adaptor computes the same gated expert sum with batched matrix multiplication instead of materializing a second entity-by-expert-by-feature tensor. Regression tests cover both output and gradient equivalence. These changes preserve the computation at the cost of recomputation. Because checkpointing avoids retaining the quadratic loss intermediates, the notebook uses the paper batch size 512. Report the implementation-level memory patches; the loss formula, model architecture, seed, features, learning rate, batch size, and 1,000-epoch schedule are unchanged.

## Troubleshooting

- **`kaggle_runner.py` not found:** attach the private Dataset created from `HUMEA-kaggle-bundle.zip`.
- **Missing dataset files:** inspect the validation list and confirm `data.zip` contains both `mmkb-datasets` directories at the expected nesting level.
- **No CUDA GPU is visible:** enable a GPU accelerator and restart the session. `--allow-cpu` exists for runner tests, not practical reproduction.
- **CUDA out of memory:** confirm the batch-before-normalization and gradient-checkpoint patches are present. If a smaller GPU still fails, use a GPU with more memory; reducing the batch after checkpointing can increase checkpoint-record overhead because this implementation accumulates all batch losses before backward.
- **A process exits without metrics:** open `artifacts/logs/<experiment-id>.log`. The manifest records the exit code and does not mark the run successful.
- **Session interruption:** restore the previous `artifacts/` directory before rerunning. Completed experiments are skipped; the interrupted one starts over.
