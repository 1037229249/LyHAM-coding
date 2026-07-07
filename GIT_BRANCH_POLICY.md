# LyHAM-CO Git Branch Policy

Date: 2026-07-07

## Branch Roles

- `main`: public GitHub branch for sanitized source code, tests, README, and public safety documents only.
- `master`: local/private working history. Do not push this branch to GitHub.
- `public-main`: clean publication branch built from an explicit allowlist. This branch may be pushed to remote `main` after verification.
- Private handoff, state, audit, contract, or experiment-recovery branches must remain local.

## Staging Rules

- Never use `git add .` for public release work.
- Stage explicit allowlisted paths only.
- Stop immediately if `git diff --cached --name-only` contains private state, logs, results, manuscripts, generated figures, model weights, or credentials.

## Remote Rules

- Remote repository: `https://github.com/1037229249/LyHAM-coding.git`.
- The remote is dedicated to this experiment code, so a verified clean public branch may overwrite remote `main`.
- Use `git push --force-with-lease origin public-main:main`.
- Do not use bare `--force`.
- Do not push local `master`.

## Public Content Rules

Allowed public content:

- `utils/**/*.py`
- `tests/**/*.py`
- `.gitignore`
- `README*.md`
- `REPO_SAFETY_AUDIT.md`
- `GIT_BRANCH_POLICY.md`
- Secret-free configuration templates and reproducibility notes after explicit review

Forbidden public content:

- Manuscripts, submissions, PDFs, DOCX/PPTX files, and LaTeX build output
- Unpublished figures and paper-ready result packages
- Logs, caches, temporary files, virtual environments, and IDE state
- Raw experiment output, large CSV/JSON result dumps, binary data, and model checkpoints
- Handoff, goal, state, contract, brief, audit, cleanup, and do-not-repeat files
- Tokens, secrets, credentials, private keys, and local absolute-path configuration

## Recovery Rule

If private files are accidentally staged, unstage them before any commit or push. If private files are accidentally pushed, stop further pushes and perform a dedicated history-remediation procedure before continuing public work.
