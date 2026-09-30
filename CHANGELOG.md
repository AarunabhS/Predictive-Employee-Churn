# Change log

## Phase two — 30 September 2026

- Added `phase2.py`, `review_tools.py`, executed `Churn_Phase2.ipynb`, and `CASE_STUDY.md`.
- Added independent calibration, uncertainty estimates, development-only stability checks, and error analysis.
- Updated the README to lead with current evidence while preserving its phase-one content.
- Added focused tests. Preserved existing files, phase-one results, notebooks, and Git history.
- Read `FEATURE_AVAILABILITY.md` before adapting the model to a company. All fields need a documented pre-departure measurement window. A validated prediction horizon requires dated employee snapshots and departure events, which this benchmark does not contain.

Data provenance and download checks are documented in `data/`. File paths and before/after SHA-256 hashes
are recorded in `AUDIT/phase2-file-changes.json`. Commit and push receipts are retained in the local audit log.
