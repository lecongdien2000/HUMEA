# Reproduction tracking implementation plan

**Goal:** Implement the user's approved W&B tracking and GitHub reproduction-guide changes without disturbing Kaggle version 27.

**Design:** Keep training mathematics unchanged. A small tracker logs detached scalar metrics, a complete best-MRR evaluation snapshot, the latest evaluation, and source/environment evidence. Runner-selected experiment names distinguish smoke/main; direct ablations use their `without` flag. Offline W&B is supported until account configuration is supplied. Public viewing is configured and verified separately from logging.

## Tasks and acceptance

- [x] Add `experiment_tracking.py` and meaningful tests in `tests/test_experiment_tracking.py`: disabled mode requires no SDK; offline runs initialize; histories use epoch; best metrics stay at a single epoch; failed runs finish as failed; source fingerprint changes with code; environment capture excludes credentials.
- [x] Integrate scalar loss logging and evaluation snapshots into `train.py`; finalize tracking on success and exceptions. Never retain loss tensors for logging or change optimizer/evaluation behavior.
- [x] Add runner tracking configuration and run URL evidence; test subprocess environment propagation and separate offline/disabled behavior.
- [x] Update Kaggle notebook: install W&B, read online credentials from Kaggle Secrets, export offline W&B data and exact environment metadata. Preserve version 27; do not push a replacement while it runs.
- [x] Rewrite both READMEs as execution guides. Include exact commands, tested hardware, data layout, dependency capture, W&B sharing/sync, and interruption behavior. Remove scores, architecture, and solution explanations.
- [x] Capture the exact patched source revision locally. Public publication remains pending the group repository/account details; no guessed destination or API keys in code.
- [x] Verify `python -m pytest -q`, real W&B offline logging, notebook JSON/code compilation, runner help, and `git diff --check`; request independent review before completion.

The lecturer's public-viewing requirement remains unfulfilled until an online run/project or report has been synced and its link is checked without authentication. Existing version 27 logs may be imported later with explicit provenance; they cannot recover unrecorded losses.

## Pending external configuration

- [ ] Publish to the group GitHub repository once its destination is supplied.
- [ ] Run/sync to the requested W&B account/project and verify a public-viewing link.
