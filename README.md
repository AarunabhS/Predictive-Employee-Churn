# Predictive Employee Churn

Predict employee departure in a public HR benchmark and describe which input features the model relies on.

**Start with the [executed notebook](Portfolio_Final_Arunabho.ipynb) or the [results report](results/REPORT.md).**
The report includes held-out predictions, a baseline, a precision–recall curve, confusion counts,
and the assumptions needed to interpret the results.

## Measured result

The validation-selected **random_forest** achieves test average precision **0.9595**,
precision **97.8%**, recall **90.5%**, and F1 **0.9399**.
Test log loss is **0.0852**; baseline log loss is **0.4493**.
These are the recorded benchmark results from the supplied input, not a deployment claim.

There are no employee identifiers, observation dates, or defined prediction horizon. Random benchmark holdouts do not establish future-employer performance. Feature importance and department rates are associations, not causal evidence or grounds for individual employment decisions.

## Run locally

Tested with Python **3.13.2**. Use Python 3.12 or newer and the pinned requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python fetch_data.py
python analysis.py
```

After the documented public input is present, the analysis runs locally without cloud services or credentials.
The downloader verifies checksums and leaves a differing existing input untouched.
For your own labelled CSV, use `python analysis.py --data path/to/input.csv --output results/custom_run`.
The data must match the [documented schema](data/README.md).

For predictions on new unlabelled rows, first run training, then:

```bash
python predict.py --data examples/new_rows.csv --output results/new_predictions.csv
```

The model bundle contains the fitted preprocessing and the validation-selected threshold.
Positive-score meanings: 0=stayed; 1=left. Score scales reflect the training benchmark.

## Data and modelling choices

14,999 HR benchmark rows with satisfaction, evaluation, workload, tenure, accidents, promotion, department, salary, and `left` labels. This matches the schema and row count used in the historical notebook. There are 3,008 exact duplicate rows; deduplication leaves 11,991. Dataset source provenance is limited: the file is a public teaching benchmark, not verified records of a named employer. A different IBM HR ZIP was present locally, but its schema did not match this project; it was left untouched.

Deduplicate before stratified train/validation/test splitting. Fit imputation, scaling, and one-hot encoding inside each training pipeline. Compare logistic regression and a bounded random forest against a prevalence baseline. Select on validation average precision, then select the validation F1 threshold. Compute feature importance on validation and descriptive department rates on training only.

Sources, licence, sampling, schema, and SHA-256 checksums are in [data/README.md](data/README.md).

## Reviewer map

| File | What it demonstrates |
|---|---|
| `analysis.py` | Input validation, preprocessing, bounded model comparison, held-out evaluation |
| `workflow.py` | Split isolation, validation-only decisions, metrics, report generation |
| `predict.py` | Batch inference using the saved pipeline |
| `Portfolio_Final_Arunabho.ipynb` | Executed walkthrough with current outputs |
| `results/REPORT.md` | Findings and limitations |
| `results/metrics.csv` | Validation and test metrics for every candidate |
| `results/predictions.csv` | Auditable held-out scores and labels |
| `results/splits.csv` | Split membership for every analysed row |
| `results/metrics.json` | Input hash, package versions, seed, and run settings |
| `tests/` | Input handling, leakage controls, metric correctness |

## Verification

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The notebook and CLI call the same implementation. Fixed seeds, pinned dependencies, and recorded input hashes
make the published run reproducible. Generated model files are local and ignored by Git.
See [REVIEW_NOTES.md](REVIEW_NOTES.md) for the repairs and remaining limits.
