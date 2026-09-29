# Phase 1 repairs

## Concrete changes

- Removed a Mac-specific Downloads path and unnecessary deep-learning/grid-search dependencies.
- Removed unsupported causal conclusions, insulting employee cluster labels, and the impossible satisfaction range of 3.5–4.5.
- Retained the original dataset schema while adding reproducible preprocessing, independent validation, measured comparisons, and permutation importance.

## Scope and remaining limits

There are no employee identifiers, observation dates, or defined prediction horizon. Random benchmark holdouts do not establish future-employer performance. Feature importance and department rates are associations, not causal evidence or grounds for individual employment decisions.

Two compact model candidates (one for text) replace expensive grids and unnecessary neural networks.
The current notebook is an executable walkthrough of the Python implementation. Historical versions remain in Git history.
Source folders and unrelated local datasets were read without modification during discovery.

## Validation

- Full CLI execution generated the committed metrics, reports, plots, and held-out predictions.
- The updated notebook was executed with a fresh kernel.
- Saved-model batch inference was checked against the corresponding held-out scores.
- 10 focused pytest checks passed; they cover label validation, split isolation, preprocessing leakage, and task-specific failure cases.

See [results/REPORT.md](results/REPORT.md) and [results/metrics.json](results/metrics.json) for measured evidence.

The root-level `image 1.png` through `image 6.png` are historical plots. Current results are under `results/`.
