# AGENTS.md

Symbolic-identification research framework: scikit-learn-compatible genetic-programming library (`SymbolicRegressor`, `SymbolicClassifier`, `SymbolicTransformer`) with island-model evolution and Pareto-front selection, used for hyperspectral-image classification experiments.

## Quick Start

```bash
# Install from repo root (see pyproject.toml for full dependency list):
pip install -e .
```

```python
from symbolearn.symbolic_estimators import SymbolicClassifier
sc = SymbolicClassifier(
    maxsize=15, niterations=10, populations=31, population_size=27,
    ncycles_per_iteration=38, n_jobs=8, random_state=42,
    metric='hinge_loss', out_func='identity',        # see "API gotchas" below
    spatial_stats='mean', valid_window_sizes=3,
    spectral_stats=('mean', 'min', 'max', 'slope'),
    operators=('+', '-', '*', '/', 'sigmoid', 'tanh', 'softplus'),
)
sc.fit(image_3d, y_train_map)        # image_3d: (H, W, D); y_train_map: (H, W) with NaN=unlabeled
print(sc.score(image_3d, y_test_map))
df = sc.get_hof()                    # Pareto front as a DataFrame
```

## Commands

There is **no Makefile, no CI workflow, no linter, no formatter** in this repo. Everything is run as plain scripts from the repo root.

```bash
# CLI trainer (used in the paper experiments); run a single dataset.
# Supports: Houston, "Pavia University", Salinas, KSC, "Indian Pines", Salinas-A,
#           WHU_Hi_LongKou, WHU_Hi_HanChuan, WHU_Hi_HongHu, Trento, "Pavia Centre", Botswana
python train_sc.py --dataset "Pavia University" --niterations 100 --n_jobs 16

# Smoke test on a hyperspectral dataset (uses Salinas by default)
python test.py
```

## Architecture

`symbolearn/` is the library. The tree shape is:

- `symbolic_estimators.py` — public sklearn API (`SymbolicRegressor`, `SymbolicClassifier`, `SymbolicTransformer`)
- `base.py` — `BaseSymbolic`: island-model orchestrator, migration, hall-of-fame, parallel init via `joblib`
- `population.py` — one island; steady-state evolution, tournament selection, batched fitness
- `expression.py` — `Expression` / `ExpressionSet` (multi-output for multi-class classification)
- `generator.py` + `tree.py` — random expression generation with parsimony pressure; `count_trees` is a DP pre-compute
- `gpoperator.py` — mutation + crossover with semantic validity checks
- `node.py` — `Operator` / `Variable` / `Constant` / `DynamicAggregation`; **uses numba `@njit/@vectorize`** — first call is a slow compile
- `fitness.py` + `symbolearn/metrics/{regression,classification,transformer}.py` — fitness functions, all numba-accelerated
- `halloffame.py` — Pareto-front archive with dominance checks
- `log.py` — `EvolutionLogger` / `LogAnalyzer`
- `utils.py` — `stratified_train_test_split`, `extract_and_aggregate_spatial` (NaN-aware window aggregation over `(H,W,D)` arrays), `poisson_sample`, `_idx_model_selection`
- `tree_parser.py` — load expressions from CSV

Tests under `tests/` use **pytest**. Run all tests from the repo root:

```bash
pytest                          # quiet, 174 tests
pytest -v                       # verbose with test names
pytest tests/test_node.py       # single file
```

Test configuration lives in `[tool.pytest.ini_options]` inside `pyproject.toml` (sets `testpaths`, `pythonpath`, and suppresses `UserWarning`).

## API Gotchas (worth knowing before editing)

- `metric='hinge_loss'` requires `out_func='identity'` — a `UserWarning` is raised otherwise, and multi-class auto-forces `out_func='softmax'` for non-hinge metrics.
- `SymbolicClassifier` stores the label encoder in `self.classes_`; integer class indices in the returned `predict` / `predict_proba` map back to the original labels via `self.classes_.take(...)`.
- 3-D `(H, W, D)` `predict` only works when the classifier was constructed with a `spatial_stats` value. Tabular `predict` requires the same `n_features_in_` as `fit`.
- `score()` overrides `ClassifierMixin.score` to ignore NaN pixels in spatial mode — do not "simplify" this back to the sklearn default.
- `extract_and_aggregate_spatial` requires an **odd** `window_size` and uses `np.pad(mode='reflect')`; it is NaN-aware only when `skip_nan=True` (default).
- `SymbolicClassifier.fit` with spatial mode expects `X.ndim == 3` and `y.ndim == 2`; spatial dims must match.
- `stratified_train_test_split` switches between **tabular** (1-D `y`) and **map** (2-D/3-D `y`) modes automatically; in map mode `preserve_shape=True` returns full `(H,W)` arrays with NaN for unselected pixels.
- `Variable(variable_index, name=...)` — the first arg is an int index (feature column), not a string. Use `Variable(0, name="x0")`.
- `Fitness.penalty` accepts `'l1'` | `'l2'` | `'elasticnet'` | `None`. `'no_pen'` is not a valid value (use `None` for no penalty).
- `Population.__init__` takes `(population_size, generator, gpoperator, metric, ...)` — does NOT accept `random_state` directly; pass it via `generator` and `gpoperator` instead.
- `NodeContent` is a base class (not abstract). Prefer concrete subclasses (`Variable`, `Constant`, `Operator`).
- `generate_random_tree(size, degrees, count_memo, random_state)` — does NOT take `variable`, `constants`, operator kwargs. Use `ExprGenerator` instead.

## Repo Hygiene / Don'ts

- Dependencies are in `pyproject.toml`. The old `requirements.txt` has been removed — don't recreate it.
- `.gitignore` excludes `data/`, `__pycache__/`, `.codegraph/`, `.vscode/` and `.pytest_cache/`. 
- Output files are written with `compress=9` (`joblib.dump`) and CSV Pareto fronts include dominated solutions (`include_dominated=True`); consumers should expect both.

## Project-Rules Note

No prior `AGENTS.md` / `CLAUDE.md` / `.cursorrules` existed when this file was created. If you add `.cursor/rules/` or `CLAUDE.md`, keep them in sync with this file rather than forking the guidance.
