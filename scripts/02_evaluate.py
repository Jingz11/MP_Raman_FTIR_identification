"""Run the seven-model within-domain and cross-library evaluations."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from core import SEED, cross_library_models, within_domain_models


def load_inputs(matrix_path, metadata_path):
    matrix = pd.read_csv(matrix_path, index_col="spectrum_id")
    meta = pd.read_csv(metadata_path)
    meta = meta.set_index("spectrum_id").loc[matrix.index].reset_index()
    return matrix.to_numpy(float), meta


def scores(y, pred):
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
    }


def group_bootstrap(y, pred, groups, repeats=2000):
    """Class-stratified group bootstrap of fixed predictions."""
    frame = pd.DataFrame({"y": y, "pred": pred, "group": groups})
    if (frame.groupby("group").y.nunique() > 1).any():
        raise ValueError("each group_id must belong to one polymer")
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(repeats):
        parts = []
        for label, block in frame.groupby("y", sort=True):
            ids = block.group.drop_duplicates().to_numpy()
            sampled = rng.choice(ids, size=len(ids), replace=True)
            parts.extend(block[block.group.eq(g)] for g in sampled)
        boot = pd.concat(parts, ignore_index=True)
        draws.append(balanced_accuracy_score(boot.y, boot.pred))
    return np.quantile(draws, [0.025, 0.975])


def add_predictions(rows, analysis, modality, train_library, test_library, model, ids, y, pred, groups, folds=None):
    for i, sid in enumerate(ids):
        rows.append({
            "analysis": analysis, "modality": modality, "train_library": train_library,
            "test_library": test_library, "model": model, "spectrum_id": sid,
            "group_id": groups[i], "true_polymer": y[i], "predicted_polymer": pred[i],
            "fold": "" if folds is None else int(folds[i]),
        })


def run_within(X, meta, repeats):
    summaries, predictions = [], []
    for (modality, library), idx in meta.groupby(["modality", "library"], sort=True).groups.items():
        idx = np.asarray(list(idx), dtype=int)
        counts = meta.loc[idx, "polymer"].value_counts()
        keep_labels = counts[counts >= 10].index
        idx = idx[meta.loc[idx, "polymer"].isin(keep_labels).to_numpy()]
        if len(keep_labels) < 2:
            continue
        y = meta.loc[idx, "polymer"].astype(str).to_numpy()
        groups = meta.loc[idx, "group_id"].astype(str).to_numpy()
        group_counts = pd.DataFrame({"y": y, "g": groups}).drop_duplicates().groupby("y").size()
        n_splits = min(5, int(group_counts.min()))
        if n_splits < 2:
            continue
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        splits = list(cv.split(X[idx], y, groups))
        for name, estimator in within_domain_models().items():
            pred = np.empty(len(idx), dtype=object)
            fold_id = np.empty(len(idx), dtype=int)
            for fold, (train, test) in enumerate(splits):
                fitted = clone(estimator).fit(X[idx][train], y[train])
                pred[test], fold_id[test] = fitted.predict(X[idx][test]), fold
            row = {"analysis": "within", "modality": modality, "train_library": library,
                   "test_library": library, "model": name, "n": len(idx), "classes": len(keep_labels),
                   "folds": n_splits, **scores(y, pred)}
            lo, hi = group_bootstrap(y, pred, groups, repeats)
            row.update(ba_ci_low=lo, ba_ci_high=hi)
            summaries.append(row)
            add_predictions(predictions, "within", modality, library, library, name,
                            meta.loc[idx, "spectrum_id"].to_numpy(), y, pred, groups, fold_id)
    return summaries, predictions


def run_cross(X, meta, repeats):
    summaries, predictions = [], []
    for modality, mod_idx in meta.groupby("modality", sort=True).groups.items():
        libraries = sorted(meta.loc[list(mod_idx), "library"].astype(str).unique())
        for source in libraries:
            for target in libraries:
                if source == target:
                    continue
                train = meta.index[(meta.modality == modality) & (meta.library.astype(str) == source)].to_numpy()
                test = meta.index[(meta.modality == modality) & (meta.library.astype(str) == target)].to_numpy()
                a, b = meta.loc[train, "polymer"].value_counts(), meta.loc[test, "polymer"].value_counts()
                common = sorted(set(a[a >= 5].index) & set(b[b >= 5].index))
                if len(common) < 2:
                    continue
                train = train[meta.loc[train, "polymer"].isin(common).to_numpy()]
                test = test[meta.loc[test, "polymer"].isin(common).to_numpy()]
                y_train = meta.loc[train, "polymer"].astype(str).to_numpy()
                y_test = meta.loc[test, "polymer"].astype(str).to_numpy()
                groups = meta.loc[test, "group_id"].astype(str).to_numpy()
                for name, estimator in cross_library_models().items():
                    pred = clone(estimator).fit(X[train], y_train).predict(X[test])
                    row = {"analysis": "cross_library", "modality": modality,
                           "train_library": source, "test_library": target, "model": name,
                           "n": len(test), "classes": len(common), "folds": 0, **scores(y_test, pred)}
                    lo, hi = group_bootstrap(y_test, pred, groups, repeats)
                    row.update(ba_ci_low=lo, ba_ci_high=hi)
                    summaries.append(row)
                    add_predictions(predictions, "cross_library", modality, source, target, name,
                                    meta.loc[test, "spectrum_id"].to_numpy(), y_test, pred, groups)
    return summaries, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    X, meta = load_inputs(args.matrix, args.metadata)
    within_s, within_p = run_within(X, meta, args.bootstrap)
    cross_s, cross_p = run_cross(X, meta, args.bootstrap)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(within_s + cross_s).to_csv(args.output / "model_summary.csv", index=False)
    pd.DataFrame(within_p + cross_p).to_csv(args.output / "predictions.csv", index=False)
    print(f"experiments={len(within_s) + len(cross_s)} predictions={len(within_p) + len(cross_p)}")


if __name__ == "__main__":
    main()
