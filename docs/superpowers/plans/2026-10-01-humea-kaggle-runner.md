# HUMEA Kaggle Reproduction Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tested Kaggle workflow that validates the authors' processed data, runs one or all six main HUMEA experiments, uses both Kaggle GPUs safely, resumes completed experiments, and exports structured metrics.

**Architecture:** Keep all paper model files unchanged. Put reusable orchestration in `kaggle_runner.py`, expose it through a small Kaggle notebook, and cover command construction, validation, parsing, resume logic, and GPU assignment with CPU-only unit tests.

**Tech Stack:** Python 3.11+, standard library (`argparse`, `csv`, `dataclasses`, `json`, `pathlib`, `subprocess`, `threading`), PyTorch only for runtime CUDA discovery, pytest, Jupyter notebook JSON.

---

## File map

- Create `kaggle_runner.py`: experiment registry, data validation, command generation, metrics parsing, manifest storage, subprocess execution, GPU queue assignment, and CLI.
- Create `tests/test_kaggle_runner.py`: CPU-only tests of all runner behavior that does not require real training.
- Create `kaggle/humea_kaggle.ipynb`: Kaggle setup, data preparation, mode selection, runner invocation, summary display, and artifact packaging.
- Create `kaggle/README.md`: exact Kaggle instructions, runtime estimates, outputs, and troubleshooting.
- Modify `.gitignore`: ignore extracted data, generated artifacts, notebook checkpoints, and Python caches.
- Modify `README.md`: link the Kaggle workflow without changing the upstream training instructions.

### Task 1: Define and test the experiment registry and commands

**Files:**
- Create: `tests/test_kaggle_runner.py`
- Create: `kaggle_runner.py`

- [ ] **Step 1: Write failing registry and command tests**

Create tests that assert the six exact IDs and configurations, then assert that `build_train_command()` starts with the active Python interpreter and `train.py`, uses a relative dataset path, contains the paper hyperparameters, emits 1,000 epochs for a full run, and emits 12 epochs/checkpoint 10 for a smoke run.

```python
def test_registry_matches_main_table():
    assert [(e.id, e.dataset, e.rate, e.fusion_weight_dim) for e in EXPERIMENTS.values()] == [
        ("db15k-20", "FB15K_DB15K", 0.2, 512),
        ("db15k-50", "FB15K_DB15K", 0.5, 0),
        ("db15k-80", "FB15K_DB15K", 0.8, 0),
        ("yago15k-20", "FB15K_YAGO15K", 0.2, 128),
        ("yago15k-50", "FB15K_YAGO15K", 0.5, 0),
        ("yago15k-80", "FB15K_YAGO15K", 0.8, 0),
    ]
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `python -m pytest tests/test_kaggle_runner.py -q`

Expected: collection fails because `kaggle_runner` does not exist.

- [ ] **Step 3: Implement the immutable registry and command builder**

Define a frozen `Experiment` dataclass with ID, dataset, rate, fusion dimension, and paper targets. Define insertion-ordered `EXPERIMENTS`. Implement `build_train_command(repo_root, experiment, epochs=1000, checkpoint=10)` with these fixed arguments:

```python
[sys.executable, "train.py", "--file_dir", f"data/mmkb-datasets/{experiment.dataset}",
 "--rate", str(experiment.rate), "--lr", ".0005", "--epochs", str(epochs),
 "--hidden_units", "300,300,300", "--check_point", str(checkpoint),
 "--bsize", "512", "--il_start", "500", "--csls", "--csls_k", "3",
 "--seed", "42", "--tau_cl", "0.1", "--tau_al", "4.0",
 "--fusion_weight_dim", str(experiment.fusion_weight_dim), "--without", "0",
 "--al_loss", "0.1", "--cl_loss", "1.0"]
```

- [ ] **Step 4: Run registry and command tests**

Run: `python -m pytest tests/test_kaggle_runner.py -q`

Expected: the new tests pass.

- [ ] **Step 5: Commit the registry slice**

```bash
git add kaggle_runner.py tests/test_kaggle_runner.py
git commit -m "feat: define HUMEA main experiments"
```

### Task 2: Add preflight validation and metric parsing

**Files:**
- Modify: `kaggle_runner.py`
- Modify: `tests/test_kaggle_runner.py`

- [ ] **Step 1: Write failing validation and parser tests**

Use `tmp_path` to create all nine common dataset files plus the dataset-specific image pickle. Assert no missing files, delete one, and assert its exact relative path is reported. Test parsing this real log shape:

```text
Best avg epoch <330>: acc@[1, 5, 10]=[0.51175 0.6997  0.7643 ], mr=46.123, mrr=0.598
```

Expected parsed metrics: epoch 330, Hits@1 0.51175, Hits@5 0.6997, Hits@10 0.7643, MRR 0.598. Also test that text without the line returns `None`.

- [ ] **Step 2: Run focused tests and verify failures**

Run: `python -m pytest tests/test_kaggle_runner.py -k "validation or parse" -q`

Expected: failures for undefined validation/parser functions.

- [ ] **Step 3: Implement required-file validation**

Require `ent_ids_1`, `ent_ids_2`, `ill_ent_ids`, `triples_1`, `triples_2`, `training_attrs_1`, `training_attrs_2`, `attribute_feature_dict.pkl`, `triples_feature_dict.pkl`, and `<dataset>_id_img_feature_dict.pkl`. Return all missing paths and make the CLI render each on its own line.

- [ ] **Step 4: Implement final-metric parsing**

Use one compiled multiline regular expression for the final `Best avg epoch` line. Split the accuracy vector on whitespace and commas, require exactly three values, and return a JSON-serializable dictionary.

- [ ] **Step 5: Run all tests**

Run: `python -m pytest tests/test_kaggle_runner.py -q`

Expected: all current tests pass.

- [ ] **Step 6: Commit validation and parsing**

```bash
git add kaggle_runner.py tests/test_kaggle_runner.py
git commit -m "feat: validate HUMEA data and parse metrics"
```

### Task 3: Add manifests, resume behavior, and dual-GPU scheduling

**Files:**
- Modify: `kaggle_runner.py`
- Modify: `tests/test_kaggle_runner.py`

- [ ] **Step 1: Write failing state and scheduling tests**

Test that one GPU receives all six IDs in table order. Test that two GPUs receive DB15K IDs on GPU `0` and YAGO15K IDs on GPU `1`. Test that a successful manifest entry containing metrics is skipped unless `force=True`, while failed/incomplete entries run again. Mock `subprocess.Popen` with output containing a valid final metrics line and assert the manifest records success, duration, GPU ID, command, and metrics.

- [ ] **Step 2: Run focused tests and verify failures**

Run: `python -m pytest tests/test_kaggle_runner.py -k "queue or manifest or resume or run_experiment" -q`

Expected: failures for undefined scheduling/execution functions.

- [ ] **Step 3: Implement atomic manifest persistence**

Store state at `artifacts/manifest.json` under an `experiments` object. Write to `manifest.json.tmp`, flush it, and replace the final file. Protect in-process reads and writes with a `threading.Lock`.

- [ ] **Step 4: Implement subprocess execution**

Create `artifacts/logs/<experiment-id>.log`, run from the repository root, set `CUDA_VISIBLE_DEVICES` to the assigned physical GPU, stream combined stdout/stderr to both terminal and log, parse final metrics, and mark success only for exit code 0 plus valid metrics.

- [ ] **Step 5: Implement queue assignment and execution**

For one GPU, use one six-experiment queue. For at least two GPUs, create a DB15K queue on the first GPU and YAGO15K queue on the second. Run queues with `ThreadPoolExecutor`; a failed item stops only its queue, and overall exit is nonzero after other active queues finish.

- [ ] **Step 6: Implement CSV summary generation**

Write `artifacts/summary.csv` with ID, dataset, rate, GPU, status, epoch, Hits@1, Hits@5, Hits@10, MRR, duration, and comparison (`close` only when all four published metrics differ by at most 0.01).

- [ ] **Step 7: Run all tests**

Run: `python -m pytest tests/test_kaggle_runner.py -q`

Expected: all tests pass without CUDA or dataset downloads.

- [ ] **Step 8: Commit orchestration**

```bash
git add kaggle_runner.py tests/test_kaggle_runner.py
git commit -m "feat: orchestrate resumable multi-GPU runs"
```

### Task 4: Add the CLI and Kaggle notebook

**Files:**
- Modify: `kaggle_runner.py`
- Modify: `tests/test_kaggle_runner.py`
- Create: `kaggle/humea_kaggle.ipynb`

- [ ] **Step 1: Write failing CLI tests**

Call `main()` with patched arguments for `validate`, `smoke`, `single db15k-20`, and `all`. Mock CUDA discovery and process execution. Assert unknown IDs are rejected, full runs without GPUs fail with a clear message, and `--allow-cpu` permits test-only CPU execution.

- [ ] **Step 2: Run CLI tests and verify failures**

Run: `python -m pytest tests/test_kaggle_runner.py -k cli -q`

Expected: CLI behavior is not implemented yet.

- [ ] **Step 3: Implement the CLI**

Add subcommands `validate`, `smoke`, `single`, and `all`, common options `--repo-root`, `--artifacts-dir`, `--force`, and `--allow-cpu`, and `--experiment` for `single`. Detect GPUs with `torch.cuda.device_count()` only after argument parsing so `--help` remains lightweight.

- [ ] **Step 4: Create the notebook as valid nbformat 4 JSON**

Add cells for: GPU/environment inspection; cloning only when `/kaggle/working/HUMEA` is absent; installing `loguru` and `gdown`; locating an attached `data.zip` before using the README Google Drive URL; safe extraction; `MODE = "single"` and `EXPERIMENT = "db15k-20"`; validation; smoke/single/all command execution; pandas summary display; and ZIP artifact export. Every command must use `/kaggle/working/HUMEA` and direct `python`, never `uv`.

- [ ] **Step 5: Validate tests and notebook JSON**

Run:

```bash
python -m pytest tests/test_kaggle_runner.py -q
python -m json.tool kaggle/humea_kaggle.ipynb > /dev/null
python kaggle_runner.py --help
```

Expected: tests pass, JSON validation exits 0, and help lists all four subcommands.

- [ ] **Step 6: Commit CLI and notebook**

```bash
git add kaggle_runner.py tests/test_kaggle_runner.py kaggle/humea_kaggle.ipynb
git commit -m "feat: add Kaggle reproduction notebook"
```

### Task 5: Document the workflow and ignore generated data

**Files:**
- Create: `kaggle/README.md`
- Modify: `README.md`
- Modify: `.gitignore`

- [ ] **Step 1: Extend ignore rules**

Add `/artifacts/`, `/.ipynb_checkpoints/`, `kaggle/.ipynb_checkpoints/`, `__pycache__/`, and `.pytest_cache/`. Keep the existing `data/*` rule.

- [ ] **Step 2: Write Kaggle instructions**

Document T4 x2 and Internet settings; the processed-feature distinction; attach-first and download fallback data paths; validate, smoke, single, and all modes; 45–75 minute single-run and 4.5–6.5 hour all-run estimates; output files; experiment-granularity resume; and OOM/interruption troubleshooting.

- [ ] **Step 3: Link the workflow from the main README**

Add a `Kaggle reproduction` section pointing to `kaggle/README.md` and `kaggle/humea_kaggle.ipynb`. State that this path runs only the six main non-iterative experiments and uses released features.

- [ ] **Step 4: Check documentation for stale or contradictory instructions**

Run: `rg -n "Kaggle|precomputed|4.5|db15k-20|uv" README.md kaggle/README.md`

Expected: Kaggle documentation points to direct Python execution; upstream local setup may still mention `uv`.

- [ ] **Step 5: Commit documentation**

```bash
git add .gitignore README.md kaggle/README.md
git commit -m "docs: explain Kaggle HUMEA reproduction"
```

### Task 6: Final verification and handoff

**Files:**
- Verify: all changed project files

- [ ] **Step 1: Run the complete CPU-only test suite**

Run: `python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Run static entry-point checks**

Run:

```bash
python -m py_compile kaggle_runner.py
python kaggle_runner.py --help
python -m json.tool kaggle/humea_kaggle.ipynb > /dev/null
```

Expected: all commands exit 0.

- [ ] **Step 3: Verify source isolation**

Run: `git diff af60da5 -- train.py model.py layers.py loss.py utils.py`

Expected: no output, proving the paper model/training implementation is unchanged.

- [ ] **Step 4: Review repository status and diff**

Run: `git status --short` and `git diff --check HEAD~5..HEAD`.

Expected: clean status and no whitespace errors.

- [ ] **Step 5: Record verification evidence**

Add the exact test count, CLI check result, notebook JSON result, and source-isolation result to the final handoff. Explicitly state that real 1,000-epoch training was not run locally because the released dataset and Kaggle GPUs are external runtime inputs.
