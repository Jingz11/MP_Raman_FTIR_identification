"""Fixed-source independent FTIR or paired Reference/ALS Raman validation."""
import argparse
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.svm import SVC
from core import cross_library_models
from analysis_utils import group_ids, paired_bootstrap


def family(a):
    return np.where(np.isin(a, ["PET", "PES"]), "Polyester", a)


def balanced_accuracy_score(y, pred):
    """Mean true-class recall, retaining predictions outside target label coverage."""
    y, pred = np.asarray(y), np.asarray(pred)
    return float(np.mean([(pred[y==c]==c).mean() for c in np.unique(y)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--source-metadata", required=True, type=Path)
    parser.add_argument("--target-metadata", required=True, type=Path)
    parser.add_argument("--modality", required=True, choices=["FTIR", "Raman"])
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    sm, tm = pd.read_csv(args.source_metadata), pd.read_csv(args.target_metadata)
    if sm.spectrum_id.duplicated().any() or tm.spectrum_id.duplicated().any():
        raise ValueError("source and target metadata must have unique IDs")
    ys = sm.polymer.to_numpy()
    known = tm.polymer.isin(np.unique(ys)).to_numpy()
    if not known.any():
        raise ValueError("no source-supported target labels")
    if set(sm.spectrum_id) & set(tm.spectrum_id):
        raise ValueError("source and external target IDs overlap")
    args.output.mkdir(parents=True, exist_ok=True)
    tm.assign(source_supported=known).to_csv(args.output/"target_coverage.csv", index=False)
    target = tm[known].reset_index(drop=True)
    y = target.polymer.to_numpy()
    scoring_y = family(y) if args.modality == "Raman" else y
    groups = group_ids(target)
    models = cross_library_models()
    models = {{"1NN": "1-NN", "3NN": "3-NN", "PLSDA": "PLS-DA", "SVM": "LinearSVC"}.get(k, k): v
              for k, v in models.items()}
    if args.modality == "FTIR":
        models["LinearSVC"].set_params(max_iter=5000)
    else:
        models["RBF-SVM"] = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
    with np.load(args.features, allow_pickle=False) as z:
        if not np.array_equal(z["source_id"].astype(str), sm.spectrum_id.astype(str)):
            raise ValueError("source NPZ order differs from metadata")
        if not np.array_equal(z["target_id"].astype(str), tm.spectrum_id.astype(str)):
            raise ValueError("target NPZ order differs from metadata")
        arms = [("Reference", "source_reference", "target_reference")]
        if args.modality == "Raman":
            arms += [("Harmonized_ALS", "source_ALS", "target_ALS")]
        features = {arm: (z[sk].copy(), z[tk][known].copy()) for arm, sk, tk in arms}
    design = dict(modality=args.modality, n_train=len(sm), n_target_total=len(tm),
                  n_target_supported=len(target), bootstrap=args.bootstrap, seed=42,
                  source_classes=sorted(set(ys)), fitting="All source classes retained; no separate StandardScaler",
                  scoring="Polyester family primary, strict source labels secondary" if args.modality=="Raman" else "Strict source labels",
                  uncertainty="Class-stratified target-group resampling of fixed predictions; no model refitting",
                  model_parameters={k: v.get_params() for k, v in models.items()})
    (args.output/"design.json").write_text(json.dumps(design, indent=2), encoding="utf-8")
    rows, preds, recalls, pmap = [], [], [], {}
    for arm, (xs, xt) in features.items():
        if xs.shape[0] != len(sm) or xt.shape[0] != len(target) or xs.shape[1] != xt.shape[1]:
            raise ValueError("source/target feature shapes do not align")
        if not np.isfinite(xs).all() or not np.isfinite(xt).all():
            raise ValueError("external features must be finite")
        for name, estimator in models.items():
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                pred = clone(estimator).fit(xs, ys).predict(xt)
            scoring_pred = family(pred) if args.modality == "Raman" else pred
            pmap[(arm, name)] = scoring_pred
            rows.append(dict(workflow=arm, model=name, n_train=len(sm), n_test=len(target),
                             family_BA=balanced_accuracy_score(scoring_y, scoring_pred),
                             strict_BA=balanced_accuracy_score(y, pred),
                             fit_warnings=" | ".join(str(w.message) for w in ws)))
            frame = pd.DataFrame(dict(spectrum_id=target.spectrum_id, group_id=groups,
                                      true_label=y, prediction=pred, family_true=scoring_y,
                                      family_prediction=scoring_pred, workflow=arm, model=name))
            preds.append(frame)
            for c in sorted(set(scoring_y)):
                recalls.append(dict(workflow=arm, model=name, polymer=c, n=int((scoring_y==c).sum()),
                                    recall=float((scoring_pred[scoring_y==c]==c).mean())))
    changes = []
    for name in models:
        p0 = pmap[("Reference", name)]
        p1 = pmap[("Harmonized_ALS", name)] if args.modality=="Raman" else p0
        changes.append(dict(model=name, Reference_BA=balanced_accuracy_score(scoring_y,p0),
                            comparison_BA=balanced_accuracy_score(scoring_y,p1),
                            delta=balanced_accuracy_score(scoring_y,p1)-balanced_accuracy_score(scoring_y,p0),
                            **paired_bootstrap(scoring_y,p0,p1,groups,args.bootstrap)))
    pd.DataFrame(rows).to_csv(args.output/"scores.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(args.output/"predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(recalls).to_csv(args.output/"per_class.csv", index=False)
    pd.DataFrame(changes).to_csv(args.output/"paired_changes.csv", index=False)
    print(f"source={len(sm)}; supported external target={len(target)}; models/arms={len(rows)}")


if __name__ == "__main__":
    main()
