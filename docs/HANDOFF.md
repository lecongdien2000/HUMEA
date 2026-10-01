# HUMEA current work

Updated 2026-10-01 (Asia/Bangkok). Read this instead of old chat history.

## Lecturer requirements update

- User wants all conversation/work communication in English.
- New implementation is on local branch `chore/reproduction-tracking`.
- `experiment_tracking.py` records W&B epoch losses/evaluation, complete best-MRR snapshots, latest evaluation, raw logs, exact runtime versions, and deterministic source snapshots. Default execution mode is offline; online credentials come from the environment/Kaggle Secrets.
- Runner records tracking metadata/W&B URL in manifest and CSV; failed tracking prevents an experiment from being marked successful. SDK is pinned to W&B 0.30.0.
- Root and Kaggle READMEs now provide execution instructions and sharing requirements, without architecture explanations or score tables.
- Final verification: 58 tests passed; real W&B offline smoke succeeded; notebook code compiled; `uv lock --check` and `git diff --check` passed. Review found and fixed a legacy-result resume bypass: requested tracking now requires completed evidence before reuse or success, and online reuse checks destination. Focused re-review and final suite passed. Notebook compilation and dependency lock checks also passed.
- W&B entity/project and the group's GitHub destination were requested asynchronously but have not been supplied. Public sharing and repository publication therefore remain pending. Local/offline logging alone does not satisfy the public-link requirement.
- Active Kaggle version 27 still uses the original source without W&B; it was not interrupted. Preserve its original bundle at `D:/Github/HUMEA-kaggle-deploy/archive/v27/HUMEA-kaggle-bundle.zip`. Future executions should use the new committed bundle.

## Objective

Run the first paper experiment (`db15k-20`, seed 42, batch 512, 1,000 epochs) on Kaggle T4 with the released features and unchanged training objective. Six-experiment reproduction comes after a working first run.

## Current state

- Repository: `D:/Github/HUMEA-analysis`, origin `https://github.com/mikumifa/HUMEA.git`.
- HEAD on resumption: `182517d` (`fix: release retained epoch tensors`). Earlier commits reduce loss, MoE, and sparse attention memory.
- Kaggle version 24 failed during `sum(loss_all).backward()`: 10.90 GiB allocated, requested 434.27 MiB, only 30.81 MiB free.
- Uncommitted gradient-bridge implementation inherited from the previous session is now completed locally. Losses use detached encoder/expert outputs; accumulated proxy gradients then propagate through the encoder in one backward call.
- This session additionally skips original tensors without gradients (modality ablations), releases all named proxy references before the estimator forward, and releases final loss references before encoder backward.
- Version 25 got through loss backward but failed in encoder backward (11.62 GiB allocated; requested 434.27 MiB; 32.81 MiB free). Logs: `D:/Github/HUMEA-kaggle-v25/HUMEA/artifacts/logs/smoke-db15k-20.log`.
- Root-cause follow-up: each whitening expert saved a separate full-width centered feature table. The image table is 434.27 MiB and there are five image experts. `PWLayer` now computes `W(dropout(x)) - W(b)` instead of `W(dropout(x) - b)`, avoiding these copies and their full-width bias gradients. This is algebraically equivalent; floating-point operation ordering differs.
- Verification: `python -m pytest -q`: **42 passed**, one existing PyTorch sparse-invariant warning. Tests include output/input/parameter-gradient equivalence with dropout, plus saved-tensor storage checks. `git diff --check` passed.
- Local PyTorch is CPU-only; GPU memory success cannot be inferred from local tests.

## Active run

- Notebook: https://www.kaggle.com/code/lecongdien/humea-first-experiment
- Dataset: https://www.kaggle.com/datasets/lecongdien/humea-kaggle-reproduction
- Source bundle was rebuilt from tracked working-tree files (including uncommitted patches), byte-checked, and uploaded successfully.
- **Version 26 smoke PASSED**: 12 epochs, epoch-10 evaluation, exit 0; training 89.7164 seconds, subprocess 112.633 seconds. Metrics at epoch 10: Hits@1 0.20605, Hits@5 0.39380002, Hits@10 0.489, MRR 0.299. These early metrics are not final reproduction results. Evidence: `D:/Github/HUMEA-kaggle-v26/HUMEA/artifacts/`.
- **Notebook version 27 launches the full 1,000-epoch `db15k-20` run**, same `NvidiaTeslaT4` and batch 512. Pushed successfully; final result pending.
- Deployment directory: `D:/Github/HUMEA-kaggle-deploy` (`source` and `kernel` subdirectories).
- Repository and deployment notebooks now both use `single`; smoke was used only for validation.

## Next actions

1. Check `kaggle kernels status lecongdien/humea-first-experiment`.
2. When complete/error, download only logs/manifest/summary (avoid the multi-GB dataset output):
   `kaggle kernels output lecongdien/humea-first-experiment -p D:/Github/HUMEA-kaggle-v27 --file-pattern '.*artifacts/.*(log|manifest.json|summary.csv)$'`
3. Verify the full-run manifest and final metrics, then compare them with paper targets. Do not relaunch while version 27 is running.
4. On failure, inspect the exact new traceback before any further memory changes.
5. Update this handoff with the result and active notebook version.

## CLI details

- Run dataset upload **from its source directory**, using `kaggle datasets version -p . -m 'message'`. An absolute `-p D:/...` hits a Windows upload-cache path error.
- Launch from the kernel directory: `kaggle kernels push -p . --accelerator NvidiaTeslaT4`.
- Set `PYTHONIOENCODING=utf-8` when fetching outputs; the CLI may otherwise crash while printing Unicode notebook logs after downloading the requested files.
- READMEs now contain operational instructions only. Historical debugging and measurements are recorded here; full-run timing remains unverified.
