# Data source and schema

Source: [Public HR_comma_sep.csv benchmark, distributed by liujiaqi on Kaggle](https://www.kaggle.com/datasets/liujiaqi/hr-comma-sepcsv).

Licence: **CC0: Public Domain, as stated in Kaggle dataset metadata**. Dataset rights are separate from project code.

14,999 HR benchmark rows with satisfaction, evaluation, workload, tenure, accidents, promotion, department, salary, and `left` labels. This matches the schema and row count used in the historical notebook. There are 3,008 exact duplicate rows; deduplication leaves 11,991. Dataset source provenance is limited: the file is a public teaching benchmark, not verified records of a named employer. A different IBM HR ZIP was present locally, but its schema did not match this project; it was left untouched.

## Schema

CSV columns: `satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, Work_accident, left, promotion_last_5years, sales, salary`.

Target: `left` — 0=stayed; 1=left.
Extra columns are excluded by the explicit feature list in `analysis.py`.
Missing or non-binary labels are rejected, never inferred as negatives.
Numeric missing features are imputed from training data; completely missing numeric columns are rejected.

## Reproduce and verify

`python fetch_data.py` verifies the local file or downloads from the exact source above if it is missing.
Both the upstream bytes and the converted CSV are checked against the SHA-256 hashes in
[provenance.json](provenance.json). Existing differing files are left untouched.
CSV conversions are deterministic with the tested requirements.

See the [results report](../results/REPORT.md) for the actual cohort, partition sizes, and limitations.
