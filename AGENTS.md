# Repository Guidelines

## Project Structure & Modules
- `sgelabs/`: core Python package (parser, IR, solver, analysis, plotting, utils).
- `examples/`: small runnable models and CLI demos.
- `tests/`: pytest suite for parser, linearization, solver, and analysis.
- `model_base/`: curated Dynare `.mod` references and assets (read‑only).
- `pyproject.toml`: build config; enable editable installs and dev tools.

## Build, Test, and Development
- Setup: `python -m venv .venv && source .venv/bin/activate` (Windows: `./.venv/Scripts/activate`).
- Install (dev): `pip install -e .[dev]`.
- Lint/format: `ruff check .` and `ruff format .`.
- Type check: `mypy sgelabs`.
- Tests: `pytest -q`.
- CLI (spec): `sgelabs load model.mod`, `sgelabs solve --ir model.ir.json`, `sgelabs irf --ss model.ss.json`.

## Coding Style & Naming
- Python ≥ 3.11, 4‑space indent, exhaustive type hints in public APIs.
- Names: `snake_case` functions/vars, `PascalCase` classes, module‑private with leading `_`.
- Structure modules by concern: `io/`, `ir/`, `solve/`, `analysis/`, `plotting/`, `utils/`.
- Keep functions small; prefer pure, deterministic code (seed RNG in tests).

## Testing Guidelines
- Framework: `pytest` with `tests/` mirroring package layout.
- Files: `tests/<area>/test_<unit>.py`; parametrized cases for edge coverage.
- Targets: parser round‑trips, BK conditions, IRF/FEVD shapes, error messages.
- Run: `pytest -q`; aim for meaningful coverage on touched code.

## Commit & PR Guidelines
- Commits: Conventional Commits (e.g., `feat(parser): support varobs block`).
- Scope PRs narrowly; include a clear description, linked issues, and CLI/output snippets or plots when relevant.
- Checklist: updated tests, docs/examples if behavior changes, `ruff`/`mypy` clean.

## Security & Assets
- Do not modify contents under `model_base/` except to add metadata or new curated examples.
- Avoid committing large binaries elsewhere; prefer lightweight text fixtures.
- No network calls in library code; make behavior deterministic.

## Agent‑Specific Notes
- Prefer minimal diffs and focused patches; align with existing layout in `spec.md`.
- Update docs alongside code; add tests for new behavior before or with changes.
