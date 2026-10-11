"""Fixed-design centered CORAL benchmark; no target-label parameter search."""
import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestCentroid
from sklearn.svm import LinearSVC
from analysis_utils import load_matrix, prepare_pair, group_ids, paired_bootstrap


def coral(xs, xt):
    """(Xs-mean_s) @ (Cs+I)^(-1/2) @ (Ct+I)^(1/2) + mean_t."""
    ms, mt = xs.mean(0), xt.mean(0)
    cs, ct = np.cov(xs, rowvar=False), np.cov(xt, rowvar=False)
    eye = np.eye(xs.shape[1])
    es, us = eigh(cs+eye, check_finite=False)
    et, ut = eigh(ct+eye, check_finite=False)
    invs, roott = (us*es**(-.5)) @ us.T, (ut*et**.5) @ ut.T
    a = invs @ roott
    adapted = (xs-ms) @ a + mt
    error = np.linalg.norm(a.T@(cs+eye)@a-(ct+eye))/np.linalg.norm(ct+eye)
    if error >= 1e-9 or not np.allclose(adapted.mean(0), mt, atol=1e-10):
        raise ValueError("CORAL matrix-power validation failed")
    return adapted, float(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--modality", required=True, choices=["FTIR", "Raman"])
    parser.add_argument("--workflow", required=True, choices=["Reference", "Harmonized"])
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    x, meta = load_matrix(args.matrix, args.metadata)
    source, target = ("ATR_FLOPP", "ATR_FLOPPe") if args.modality=="FTIR" else ("Raman_SLoPP", "Raman_RaSPI")
    tr, te, classes = prepare_pair(meta, source, target)
    xs, xt = x[tr], x[te]
    ys, yt = meta.loc[tr,"polymer"].to_numpy(), meta.loc[te,"polymer"].to_numpy()
    xa, error = coral(xs, xt)
    groups = group_ids(meta.loc[te])
    rows, predictions = [], []
    for name in ["NC", "LinearSVC"]:
        factory = lambda: NearestCentroid(metric="euclidean") if name=="NC" else LinearSVC(C=1,max_iter=1000000,random_state=42)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            p0, p1 = factory().fit(xs,ys).predict(xt), factory().fit(xa,ys).predict(xt)
        rows.append(dict(modality=args.modality, workflow=args.workflow, source=source, target=target,
                         model=name, baseline_BA=balanced_accuracy_score(yt,p0),
                         coral_BA=balanced_accuracy_score(yt,p1),
                         delta=balanced_accuracy_score(yt,p1)-balanced_accuracy_score(yt,p0),
                         regularization=1.0, regularized_alignment_relative_error=error,
                         fit_warnings=" | ".join(str(w.message) for w in ws),
                         **paired_bootstrap(yt,p0,p1,groups,args.bootstrap)))
        predictions.append(pd.DataFrame(dict(spectrum_id=meta.loc[te,"spectrum_id"], group_id=groups,
                                              true_polymer=yt, baseline_prediction=p0, coral_prediction=p1,
                                              model=name, modality=args.modality, workflow=args.workflow)))
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output/"paired_summary.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(args.output/"predictions.csv", index=False)
    print(f"CORAL {source} -> {target}; fixed sample classes={classes}")


if __name__ == "__main__":
    main()
