# Employee Churn Benchmark: case study

## Problem and outcome

Evaluate departure classification and explain error patterns without causal claims.

Average precision **0.9593**, precision **97.8%**, recall **90.5%**. Log loss **0.0763** versus baseline **0.4493**.

## Data and evaluation

The original-schema HR benchmark is retained: 11,991 unique, non-conflicting rows after cleanup. Its missing employee IDs, measurement dates, and departure horizon limit external validation. Random benchmark splits are appropriate for this demonstration, with future-company evaluation still outstanding.

Three development-only validation splits compare fixed-model performance and permutation-importance ranks. `importance_stability.csv` reports mean rank and variation. `subgroup_errors.csv` shows missed departures and false flags by department, salary, and satisfaction band. These groups describe model behavior; they do not establish equitable treatment or intervention effectiveness.

Model decisions use validation only. Preprocessing fits on fitting rows; probability calibration uses a disjoint reserve.
The test split is evaluated after model selection. Reusing known benchmarks during development is distinct from a new external validation.

## Findings, errors, and uncertainty

The [executed notebook](Churn_Phase2.ipynb) displays results and uncertainty from the same Python implementation.
[Current report](results/phase2/REPORT.md), [reliability plot](results/phase2/reliability.png),
[validation stability](results/phase2/validation_stability.csv), and [uncertainty intervals](results/phase2/uncertainty.json)
show performance with its practical limits. [Error rows](results/phase2/errors.csv) expose failures rather than hiding them.

Read `FEATURE_AVAILABILITY.md` before adapting the model to a company. All fields need a documented pre-departure measurement window. A validated prediction horizon requires dated employee snapshots and departure events, which this benchmark does not contain.

## Reproduce

Tested with the phase-one pinned Python requirements. Use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python fetch_data.py
python phase2.py
```

Phase-one results and notebooks remain available. Phase-two outputs are written to `results/phase2/`.
Model bundles and locally retained raw inputs are ignored by Git. Data downloaders verify pinned hashes and preserve differing existing inputs.
Install `requirements-dev.txt` for `python -m pytest -q` or to execute the notebook.

## Batch prediction with the phase-two model

```bash
python predict.py --model results/phase2/model.joblib --data examples/new_rows.csv --output results/phase2/local/new_predictions.csv
```
