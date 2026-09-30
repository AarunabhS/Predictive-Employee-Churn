"""Phase two: calibrated scores, uncertainty, stability, and operating trade-offs."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'results/phase2/local/mpl-cache'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(__file__).resolve().parent/'results/phase2/local/cache'))
import argparse
import importlib.metadata
import platform
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import analysis
from workflow import choose_threshold, measure, input_metadata, json_write
from review_tools import (partition_training, fit_sigmoid, bootstrap_intervals, reliability_table,
                          plot_reliability, operating_policies, ranking_curve, subgroup_errors, export_splits)

ROOT = Path(__file__).resolve().parent
PROJECT = 'churn'


def build_models(n_fit):
    if PROJECT == 'fraud':
        return {
            'weighted_logistic': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                LogisticRegression(class_weight='balanced', max_iter=700, random_state=42)),
            'random_forest': make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(
                n_estimators=60, max_depth=12, min_samples_leaf=2, max_samples=min(40000, n_fit),
                class_weight='balanced_subsample', random_state=42, n_jobs=2))}
    if PROJECT == 'churn':
        return {'logistic_regression': make_pipeline(analysis.preprocessing(), LogisticRegression(max_iter=600, random_state=42)),
                'random_forest': make_pipeline(analysis.preprocessing(), RandomForestClassifier(
                    n_estimators=100, max_depth=14, min_samples_leaf=3, random_state=42, n_jobs=2))}
    return {'logistic_regression': make_pipeline(analysis.preprocessing(), LogisticRegression(max_iter=600, C=0.5, random_state=42)),
            'random_forest': make_pipeline(analysis.preprocessing(), RandomForestClassifier(
                n_estimators=70, max_depth=12, min_samples_leaf=12, random_state=42, n_jobs=2))}


def load(path):
    if PROJECT == 'fraud':
        X, y, original, info = analysis.load_data(path)
        raw = pd.read_csv(path).drop_duplicates(subset=['Time', 'Class', *analysis.FEATURES]).sort_values('Time', kind='stable').reset_index(drop=True)
        times, groups = raw.Time.to_numpy(), None
    elif PROJECT == 'churn':
        X, y, info = analysis.load_data(path)
        original, times, groups = analysis.make_splits(y), None, None
    else:
        X, y, groups, info = analysis.load_data(path)
        original, times = analysis.make_splits(y, groups), None
    return X, y, partition_training(original, y, groups=groups, times=times), info, groups, times


def stability_folds(y, splits, groups, times):
    if times is not None:
        for number, end in enumerate([0.5333333333, 0.6666666667, 0.8]):
            a, b, c = np.quantile(times, [end*0.5625, end*0.75, end])
            yield number, {'fit': np.flatnonzero(times < a),
                          'calibration': np.flatnonzero((times >= a) & (times < b)),
                          'validation': np.flatnonzero((times >= b) & (times < c))}
    else:
        development = np.sort(np.concatenate([splits[key] for key in ['fit', 'calibration', 'validation']]))
        for number, seed in enumerate([43, 44, 45]):
            if groups is None:
                train, val = train_test_split(development, test_size=0.25, stratify=y.iloc[development], random_state=seed)
            else:
                a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(development, y.iloc[development], groups.iloc[development]))
                train, val = development[a], development[b]
            four = partition_training({'train': train, 'validation': val, 'test': splits['test']}, y, groups=groups, seed=seed)
            yield number, four


def run(data=None, output=None):
    started = time.monotonic()
    defaults = {'fraud': 'transactions_full.csv', 'churn': 'hr.csv', 'ctr': 'clicks.csv'}
    data = ROOT / 'data' / defaults[PROJECT] if data is None else Path(data)
    output = ROOT / 'results/phase2' if output is None else Path(output)
    output.mkdir(parents=True, exist_ok=True)
    X, y, splits, info, groups, times = load(data)
    fit, cal, val, test = (splits[key] for key in ['fit', 'calibration', 'validation', 'test'])
    beta = 2 if PROJECT == 'fraud' else 1
    models, records = {}, []
    candidates = {'baseline': DummyClassifier(strategy='prior'), **build_models(len(fit))}
    for name, base in candidates.items():
        base.fit(X.iloc[fit], y.iloc[fit])
        models[name] = base
        if name != 'baseline':
            models[name+'_sigmoid'] = fit_sigmoid(base, X.iloc[cal], y.iloc[cal])
    for name, model in models.items():
        score = model.predict_proba(X.iloc[val])[:, 1]
        threshold = 0.5 if name == 'baseline' else choose_threshold(y.iloc[val], score, beta)
        records.append({'model': name, 'split': 'validation', **measure(y.iloc[val], score, threshold)})
    options = [row for row in records if row['model'] != 'baseline']
    selected = (max(options, key=lambda r: (round(r['average_precision'], 12), -r['log_loss'])) if PROJECT == 'fraud'
                else min(options, key=lambda r: r['log_loss']))
    name, threshold = selected['model'], selected['threshold']
    model = models[name]
    # All model and threshold decisions are complete before evaluation of test labels.
    score = model.predict_proba(X.iloc[test])[:, 1]
    validation_score = model.predict_proba(X.iloc[val])[:, 1]
    predictions = pd.DataFrame({'row_id': test, 'actual': y.iloc[test].to_numpy(),
                                'selected_score': score, 'predicted': (score >= threshold).astype(int)})
    for current, estimator in models.items():
        scores = estimator.predict_proba(X.iloc[test])[:, 1]
        current_threshold = next(row['threshold'] for row in records if row['model'] == current)
        records.append({'model': current, 'split': 'test', **measure(y.iloc[test], scores, current_threshold)})
        predictions[current+'_score'] = scores
    pd.DataFrame(records).to_csv(output / 'metrics.csv', index=False, float_format='%.8f')
    predictions.to_csv(output / 'predictions.csv', index=False, float_format='%.8f')
    export_splits(y, splits, output)
    raw_name = name.removesuffix('_sigmoid')
    reliability = [reliability_table(y.iloc[test], models[key].predict_proba(X.iloc[test])[:, 1], key,scheme=scheme)
                   for key in [raw_name, raw_name+'_sigmoid'] for scheme in ['quantile','uniform']]
    plot_reliability(reliability, output, {'fraud':'Full fraud benchmark', 'churn':'Employee churn benchmark', 'ctr':'Downsampled click benchmark'}[PROJECT])
    intervals = bootstrap_intervals(y.iloc[test], score, threshold,
                                    groups=groups.iloc[test] if groups is not None else None)
    json_write(output / 'uncertainty.json', intervals)
    policies = operating_policies(y.iloc[val], validation_score, y.iloc[test], score, output)
    ranking_curve(y.iloc[test], score, output)
    joblib.dump({'model': model, 'threshold': threshold, 'phase': 2}, output / 'model.joblib')
    validation = pd.DataFrame({'row_id': val, 'actual': y.iloc[val].to_numpy(), 'score': validation_score})
    validation.to_csv(output / 'validation_predictions.csv', index=False, float_format='%.8f')
    stability, importances = [], []
    for number, fold in stability_folds(y, splits, groups, times):
        assert not set(fold['fit']) & set(test) and not set(fold['calibration']) & set(test) and not set(fold['validation']) & set(test)
        base = build_models(len(fold['fit']))[raw_name]
        base.fit(X.iloc[fold['fit']], y.iloc[fold['fit']])
        fitted = fit_sigmoid(base, X.iloc[fold['calibration']], y.iloc[fold['calibration']]) if name.endswith('_sigmoid') else base
        current_score = fitted.predict_proba(X.iloc[fold['validation']])[:, 1]
        cut = choose_threshold(y.iloc[fold['validation']], current_score, beta)
        stability.append({'fold': number+1, 'fit_rows': len(fold['fit']), 'calibration_rows': len(fold['calibration']),
                          'validation_rows': len(fold['validation']), **measure(y.iloc[fold['validation']], current_score, cut)})
        if PROJECT == 'churn':
            importance = permutation_importance(fitted, X.iloc[fold['validation']], y.iloc[fold['validation']],
                                               n_repeats=3, scoring='average_precision', random_state=42+number, n_jobs=1)
            part = pd.DataFrame({'feature': X.columns, 'ap_decrease': importance.importances_mean})
            part['rank'], part['fold'] = part.ap_decrease.rank(ascending=False), number+1
            importances.append(part)
    pd.DataFrame(stability).to_csv(output / 'validation_stability.csv', index=False, float_format='%.8f')
    if importances:
        importance = pd.concat(importances)
        importance.to_csv(output / 'importance_by_fold.csv', index=False, float_format='%.8f')
        importance.groupby('feature').agg(mean_rank=('rank','mean'), rank_std=('rank','std'), mean_ap_decrease=('ap_decrease','mean')).sort_values('mean_rank').to_csv(output / 'importance_stability.csv', float_format='%.8f')
    subgroup_columns = {'fraud':['Amount'], 'churn':['department','salary','satisfaction_level'], 'ctr':['depth','position']}[PROJECT]
    subgroup_errors(X.iloc[test], y.iloc[test], score, threshold, subgroup_columns, output)
    errors = predictions.loc[predictions.actual != predictions.predicted].copy()
    if PROJECT == 'churn':
        examples = X.iloc[errors.row_id].reset_index(drop=True)
        errors = pd.concat([errors.reset_index(drop=True), examples], axis=1)
    errors.to_csv(output / 'errors.csv', index=False, float_format='%.8f')
    summary = {'project': PROJECT, **input_metadata(data), **info,
               'selected_model': name, 'threshold': threshold,
               'selection': 'validation AP; log loss breaks AP ties' if PROJECT == 'fraud' else 'validation log loss',
               'protocol': 'fit 45%, independent sigmoid calibration 15%, validation 20%, test 20%; grouped or chronological where available',
               'splits': {key:{'rows':len(rows),'positives':int(y.iloc[rows].sum())} for key,rows in splits.items()},
               'test': next(row for row in records if row['model']==name and row['split']=='test'),
               'baseline_test': next(row for row in records if row['model']=='baseline' and row['split']=='test'),
               'runtime_seconds': round(time.monotonic()-started, 2), 'seed':42, 'python':platform.python_version(),
               'packages':{key:importlib.metadata.version(key) for key in ['numpy','pandas','scikit-learn','matplotlib']},
               'stability_design': 'Three expanding chronological validation windows; test excluded' if PROJECT=='fraud' else 'Three development-only validation splits; final test excluded',
               'fit_limit': '60 trees, at most 40,000 training rows per tree' if PROJECT=='fraud' else 'Bounded phase-one models',
               'limits': 'Benchmark redevelopment is not a newly collected external test. No production-performance or causal claims. '
                         'CTR probabilities describe downsampled benchmark prevalence; natural-prevalence logs and timestamps remain unavailable.'}
    json_write(output / 'metrics.json', summary)
    t = summary['test']
    recall, precision = intervals['recall_wilson'], intervals['precision_wilson']
    report = f'''# Phase two: evidence and decision trade-offs

{len(y):,} deduplicated observations. **{name}** was selected using {summary['selection']}.
{summary['protocol']}. Calibration uses disjoint reserve labels; preprocessing fits only on fit rows.

| Test measure | Selected | Prior baseline |
|---|---:|---:|
| Average precision | {t['average_precision']:.4f} | {summary['baseline_test']['average_precision']:.4f} |
| Log loss | {t['log_loss']:.4f} | {summary['baseline_test']['log_loss']:.4f} |
| Brier score | {t['brier_score']:.4f} | {summary['baseline_test']['brier_score']:.4f} |
| Precision | {t['precision']:.1%} | {summary['baseline_test']['precision']:.1%} |
| Recall | {t['recall']:.1%} | {summary['baseline_test']['recall']:.1%} |

At the validation-selected threshold, {t['tp']} positives are found, {t['fn']} are missed, and {t['fp']} false flags occur.
95% Wilson recall interval: {recall['low']:.1%}–{recall['high']:.1%}; precision: {precision['low']:.1%}–{precision['high']:.1%}.
The intervals describe this benchmark sample under the stated sampling assumptions.

![Calibration reliability and bin support](reliability.png)
![Ranking budget and captured positives](ranking.png)

[Operating policies](operating_policies.csv) choose thresholds exclusively from validation scores for fixed workload budgets.
Ties stay together and can leave a budget partly unused. Applying the same thresholds to test can exceed the validation budget;
the table records actual counts. No dollar-saving or intervention-effect claim is inferred from these retrospective labels.

[Uncertainty](uncertainty.json) contains 200 bootstrap replicates, with clusters resampled when user/context groups exist.
[Validation stability](validation_stability.csv) tests a fixed model structure on three development-only splits.
[Subgroup errors](subgroup_errors.csv) gives support and missed/false-flag counts; small groups warrant caution.
[Error rows](errors.csv), [all candidate metrics](metrics.csv), [test predictions](predictions.csv), and [partition membership](splits.csv) are auditable.

The final test labels are not used to choose calibration, models, budgets, or thresholds. Previously published benchmark results
were known during redevelopment; an independent external evaluation remains necessary. Calibration assumes the same target population.
For CTR, calibration cannot undo unknown upstream negative sampling. For HR, observation dates and a departure horizon are missing;
feature-importance consistency is association, not causality. The fraud benchmark spans two days and anonymized PCA features.
'''
    (output / 'REPORT.md').write_text(report)
    print(f'{PROJECT}: selected {name}; AP={t["average_precision"]:.4f}; log loss={t["log_loss"]:.4f}; elapsed={summary["runtime_seconds"]}s', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run(args.data, args.output)


if __name__ == '__main__':
    main()
