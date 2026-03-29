# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for dependency management.

```bash
# Install dependencies (including dev)
uv sync --all-groups

# Run linting
uv run ruff check src/

# Run formatting
uv run ruff format src/

# Run tests
uv run pytest

# Run a single test file
uv run pytest tests/path/to/test_file.py

# Build the package
uv run hatchling build

# Launch Jupyter for running example notebooks
uv run jupyter lab
```

## Architecture

The package is under `src/cga2m_plus/` and consists of two modules:

### `cga2m.py` — `Constraint_GA2M` class

The core model. Training is a three-phase pipeline:

1. **`train()`** — Outer loop that greedily adds pairwise interaction terms one at a time. Each iteration runs `backfitting()` over the current set of main and interaction features, then picks the next best interaction from `remaining_interaction_features` by lowest residual MSE. Stops early when interaction importance drops below `threshold` for two consecutive rounds.

2. **`prune_and_retrain()`** — Computes feature importance (L1 norm of predictions / total output norm) and drops any feature or interaction pair below `threshold`. Calls `retrain()` on the surviving terms.

3. **`higher_order_train()`** — Trains a single unconstrained LightGBM booster on the residuals of the GA2M terms. This is the "higher-order" term; it improves accuracy without being directly interpretable.

**Backfitting** (`backfitting()`): Iteratively trains each univariate and pairwise LightGBM model on the residuals of all other terms. Monotone constraints from `monotone_constraints` are passed per-feature to LightGBM. Mean-centering is applied to each term so the model is identifiable.

**`predict()`**: Sums contributions from all main terms, interaction terms, and optionally the higher-order term, then adds back `y_train_mean`.

**Internal state after training:**
- `main_model_dict` — `{feature_idx: lgb.Booster}`
- `interaction_model_dict` — `{(i, j): lgb.Booster}`
- `higher_model` — `lgb.Booster` (after `higher_order_train()`)
- `use_main_features`, `use_interaction_features` — which terms survived pruning
- `train_main_mean`, `train_interaction_mean` — centering offsets for each term

### `visualize.py` — plotting utilities

Standalone functions (not methods) that take a trained `Constraint_GA2M` instance and `X`:

- `plot_main(ga2m, X)` — line plots for each univariate term
- `plot_interaction(ga2m, X, mode="3d")` — 2D contour or 3D surface for each pairwise term
- `show_importance(ga2m, after_prune, higher_mode)` — horizontal bar chart of feature importance

## Package layout

```
src/cga2m_plus/
    __init__.py      # empty (all imports done explicitly by users)
    cga2m.py         # Constraint_GA2M class
    visualize.py     # plot_main, plot_interaction, show_importance
examples/
    How_to_use_CGA2M+.ipynb   # end-to-end usage example
```

## Notes

- All input data must be `numpy.ndarray`. The constructor deep-copies inputs and mean-centers `y_train`.
- `X_test` passed to the constructor is used as a validation set during training (not a held-out test set — naming is a known inconsistency).
- `monotone_constraints` is a list of `1`, `-1`, or `0` with length equal to number of features. Defaults to all zeros (no constraint).
- The package is published to PyPI as `cga2m-plus`.
