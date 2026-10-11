"""Registered directed transfers and the 17 fixed classifier/similarity configurations."""
import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from core import cross_library_models
from analysis_utils import load_matrix, prepare_pair, transfer_pairs, group_ids


def model_specs():
    base = cross_library_models()
    base["RBF-SVM"] = SVC(C=10, gamma="scale", kernel="rbf", random_state=42)
    for name, model in base.items():
        yield name, False, clone(model)
        if name != "RF":
            yield name, True, make_pipeline(StandardScaler(), clone(model))


def similarity_predict(xs, ys, xt, pearson=False):
    if pearson:
        xs, xt = xs-xs.mean(axis=1, keepdims=True), xt-xt.mean(axis=1, keepdims=True)
    den = np.maximum(np.linalg.norm(xt, axis=1)[:, None]*np.linalg.norm(xs, axis=1)[None, :],
                     np.finfo(float).tiny)
    return ys[np.argmax(xt @ xs.T / den, axis=1)]


def evaluate(x, meta, modality, workflow, primary_only=False):
    rows, preds = [], []
    for source, target in transfer_pairs(modality):
        tr, te, classes = prepare_pair(meta, source, target)
        xs, xt = x[tr], x[te]
        ys, yt = meta.loc[tr, "polymer"].to_numpy(), meta.loc[te, "polymer"].to_numpy()
        exp = source+"_to_"+target
        specs = list(model_specs())
        if primary_only:
            specs = [s for s in specs if s[0] in {"NC", "SVM"} and not s[1]]
        for name, scaled, model in specs + ([] if primary_only else [
                ("Cosine-1NN", False, None), ("Pearson-1NN", False, None)]):
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                if model is None:
                    p = similarity_predict(xs, ys, xt, name == "Pearson-1NN")
                else:
                    p = model.fit(xs, ys).predict(xt)
            rows.append(dict(workflow=workflow, modality=modality, experiment_id=exp,
                             source_domain=source, target_domain=target, model=name, scaler_used=scaled,
                             n_train=len(xs), n_test=len(xt), classes=";".join(classes),
                             balanced_accuracy=balanced_accuracy_score(yt, p), accuracy=accuracy_score(yt, p),
                             fit_warnings=" | ".join(str(w.message) for w in ws)))
            frame = meta.loc[te, [c for c in ["spectrum_id", "library", "subdomain"] if c in meta]].copy()
            frame["group_id"] = group_ids(meta.loc[te])
            frame["true_polymer"], frame["pred_polymer"] = yt, p
            for key in ["workflow", "modality", "experiment_id", "source_domain", "target_domain",
                        "model", "scaler_used"]:
                frame[key] = rows[-1][key]
            preds.append(frame)
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--modality", required=True, choices=["FTIR", "Raman"])
    parser.add_argument("--workflow", required=True, choices=["Reference", "Harmonized"])
    parser.add_argument("--primary-only", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    x, meta = load_matrix(args.matrix, args.metadata)
    rows, preds = evaluate(x, meta, args.modality, args.workflow, args.primary_only)
    args.output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output/"transfer_panel.csv", index=False)
    preds.to_csv(args.output/"predictions.csv.gz", index=False, compression="gzip")
    print(f"transfer/model rows={len(rows)}; predictions={len(preds)}")


if __name__ == "__main__":
    main()
