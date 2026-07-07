# LyHAM-CO Repository Safety Audit

Date: 2026-07-07

## Conclusion

This repository can enter the public release stage only through a clean allowlisted public branch. The current local `master` history must not be pushed directly because the baseline commit already tracks private goal/state/audit files and some of those files contain absolute local paths.

## Current Git State

- Local repository: target LyHAM-CO experiment code directory. The absolute local path is intentionally omitted from the public audit.
- Current branch before cleanup: `master`
- Local baseline commit: `27254a5 Baseline Lyapunov edge unloading experiment code`
- Local `origin` before setup: absent
- Target remote: `https://github.com/1037229249/LyHAM-coding.git`
- Target remote state before publish: `main` at `5b32254051d432657f054fc8355b07e7673f345f`
- Staged files during initial audit: none
- Tracked file count during initial audit: 56
- Untracked, not ignored file count during initial audit: 31

## Risk Findings

- Tracked private/state files existed in local history, including `active_goal.md`, `goal_progress.md`, `goal_state.json`, `long_term_experiment_requirements.md`, and multiple `*_audit*.md` files.
- Tracked private files contained absolute local paths such as Codex attachment paths and local Python/runtime paths.
- Untracked private handoff and contract files existed, including `CURRENT_EXECUTION_HANDOFF.md`, `GLOBAL_TMC_GOAL_CONTRACT.md`, and `CURRENT_BLOCKER_CONTRACT.md`.
- Large generated artifacts existed locally, including `_test_artifacts/`, `smoke_artifacts/`, `tmp/`, `paper_ready_*`, `实验结果/`, and `Training results/`.
- `ARTIFACT_STATUS_MATRIX.csv` was treated as private and not approved for public release.

## Public Allowlist

Only these categories are allowed in the public branch:

- Source code under `utils/`.
- Tests under `tests/`.
- `.gitignore`.
- Public README files.
- Public safety documentation: `REPO_SAFETY_AUDIT.md` and `GIT_BRANCH_POLICY.md`.
- Explicitly reviewed configuration templates or run scripts, if present and secret-free.

## Forbidden Public Content

The following must not be staged, committed, or pushed to GitHub:

- Manuscripts, submission materials, PDFs, DOCX/PPTX files, and LaTeX build output.
- Unpublished figures, paper-ready figure packages, result images, and generated tables.
- Logs, caches, temporary files, virtual environments, IDE state, and local Codex state.
- Raw experiment outputs, large CSV/JSON dumps, binary data, and model checkpoints.
- Handoff, goal, state, contract, brief, audit, cleanup, and do-not-repeat files.
- Tokens, secrets, credentials, private keys, and absolute local configuration.

## Required Release Method

Do not push the current `master` branch. Build a clean public branch from an allowlist and push that branch to remote `main` with `--force-with-lease` only after verification confirms no forbidden files are present.

## Public Readiness

The clean `public-main` branch is the only branch eligible for public publication after the release checks pass. The original local `master` is treated as private history and is not safe for direct publication.
