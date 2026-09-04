# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-04

First release associated with the accepted IEEE TGRS paper:
*"Symbolic Discovery of Explicit and Auditable Spectral Models for Land Cover Mapping"*
(DOI: [10.1109/TGRS.2026.3731741](https://doi.org/10.1109/TGRS.2026.3731741)).

### Added
- Public API exports in `symbolearn/__init__.py`:
  `SymbolicClassifier`, `SymbolicRegressor`, `SymbolicTransformer`, `Expression`,
  `ExpressionSet`, `Fitness`, `__version__`.
- `__version__` is exposed at runtime; keep it in sync with `[project] version`
  in `pyproject.toml`.
- README badge for IEEE TGRS (Accepted 2026) plus the highlighted DOI link.
- README citation block: full citation text and a `BibTeX` entry
  (`note = {Accepted for publication, to appear}` until the formal volume/issue
  /page numbers become available).
- `.gitignore` covers `data/`, `outputs/`, `__pycache__/`, `.pytest_cache/`,
  `.codegraph/`, `.vscode/`, `.workbuddy/` (project data only).

### Fixed
- `population.find_top_n` / `find_oldest_n`: `np.argpartition` would raise
  `ValueError: kth(=N) out of bounds (N)` when `n == len(fitnesses)`. Both
  methods now short-circuit to `np.arange(total)` at that boundary.
- `utils.stratified_train_test_split` (balanced branch): `train_size` was
  treated as an absolute count even when given as a ratio (e.g. `0.8`), causing
  a `TypeError`. Ratios are now expanded to integer targets before allocating
  per-class quotas.
- `metrics/transformer.py`: `silhouette_loss`, `davies_bouldin_loss`,
  `calinski_harabasz_loss` were passing 1-D `X` to scikit-learn clustering
  metrics. The 2-D `y_pred` is now passed as features; labels use
  `y_true.ravel()`.
- `tests/test_fitness.py`: the three penalty tests now opt in to bias
  regularization via `regularize_bias=True`, matching the documented behavior
  that pure additive-bias constants are excluded by default.
- `train_sc.py`: `train_size` was forwarded as `float` to
  `spatially_disjoint_train_test_split`, which raises on non-integer input.
  Both CLI branches now cast to `int()`.
- Repository title alignment: README top heading, the in-text reference
  paragraph, and the BibTeX entry title all use the accepted paper title
  (previously the README still referenced an early working title).
- `pyproject.toml`: package description aligned with the accepted paper title.

### Changed
- Project description in `pyproject.toml` rewritten to match the accepted
  paper title (removed the working subtitle and old working title).
- README author line links each author to their ORCID profile (Yanmin Jiang
  has no ORCID on record and is kept as plain text).
- README wording around the paper status clarifies "Accepted for publication"
  with a note that the DOI is pending activation until publication.

### Repository hygiene
- Tracked Git LFS / large files: none added.
- Generated outputs (e.g. `outputs/models*/SC/...`) are runtime artefacts and
  are excluded from version control via `outputs/` in `.gitignore`.
