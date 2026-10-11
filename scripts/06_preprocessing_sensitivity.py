"""Fixed-population baseline, ALS-lambda and normalization sensitivity."""
import argparse
from pathlib import Path
import importlib
import json
import warnings
import numpy as np
import pandas as pd
from pybaselines import Baseline
from core import als_baseline, snv, minmax
from analysis_utils import load_matrix
panel = importlib.import_module("04_transfer_panel")
pavi = importlib.import_module("03_pavi")


def feature_conditions(pre, grid):
    """Input is MA7-smoothed intensity before baseline correction and SNV."""
    yield "none_minmax", np.vstack([minmax(snv(v)) for v in pre])
    yield "none_snv", np.vstack([snv(v) for v in pre])
    for lam in [1e4, 3e4, 1e5, 3e5, 1e6]:
        residual = np.vstack([v-als_baseline(v, lam=lam) for v in pre])
        yield f"ALS_{lam:g}_minmax", np.vstack([minmax(snv(v)) for v in residual])
        if lam == 1e5:
            yield "ALS_100000_snv", np.vstack([snv(v) for v in residual])
    fitter = Baseline(x_data=grid)
    for name in ["airPLS", "ModPoly", "rubberband"]:
        residual = []
        for v in pre:
            if name == "airPLS":
                baseline, info = fitter.airpls(v, lam=1e6, max_iter=50, tol=.001, normalize_weights=False)
            elif name == "ModPoly":
                baseline, info = fitter.modpoly(v, poly_order=2, max_iter=1000, tol=.001)
                if info["tol_history"][-1] >= .001:
                    raise ValueError("ModPoly did not converge at max_iter=1000")
            else:
                baseline, info = fitter.rubberband(v, segments=1, lam=None)
            residual.append(minmax(snv(v-baseline)))
        yield name+"_minmax", np.vstack(residual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=Path, help="Raman NPZ: spectrum_id, pre_snv_ma7, grid_cm1")
    parser.add_argument("--matrix", required=True, type=Path, help="Fixed published processed matrix")
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--modality", required=True, choices=["FTIR", "Raman"])
    parser.add_argument("--workflow", required=True, choices=["Reference", "Harmonized"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    x, meta = load_matrix(args.matrix, args.metadata)
    args.output.mkdir(parents=True, exist_ok=True)
    conditions = [("none_minmax", x), ("none_snv", np.vstack([snv(v) for v in x]))]
    if args.modality == "Raman":
        if args.components is None:
            parser.error("--components is required for Raman")
        with np.load(args.components, allow_pickle=False) as z:
            ids = z["spectrum_id"].astype(str)
            if len(set(ids)) != len(ids):
                raise ValueError("component IDs must be unique")
            positions = pd.Series(np.arange(len(ids)), index=ids).reindex(meta.spectrum_id.astype(str))
            if positions.isna().any():
                raise ValueError("some fixed sample IDs are absent from the component cache")
            pre = z["pre_snv_ma7"][positions.to_numpy(int)]
            grid = z["grid_cm1"].copy()
        conditions = feature_conditions(pre, grid)
    allmetrics, pavis, checks = [], [], []
    for label, features in conditions:
        if not np.isfinite(features).all():
            raise ValueError(f"nonfinite features: {label}")
        metrics, predictions = panel.evaluate(features, meta, args.modality, label, primary_only=True)
        metrics["sample_workflow"] = args.workflow
        allmetrics.append(metrics)
        predictions.to_csv(args.output/(label+"_predictions.csv.gz"), index=False, compression="gzip")
        cells, _, _ = pavi.cells_from_predictions(pavi.canonical_predictions(predictions))
        pavis.append(pavi.summarize(cells).assign(condition=label, sample_workflow=args.workflow))
        baseline_label = "ALS_100000_minmax" if args.workflow == "Harmonized" else "none_minmax"
        if label == baseline_label:
            difference = float(np.max(np.abs(features-x)))
            if difference >= 1e-9:
                raise ValueError(f"baseline matrix reproduction failed: {difference}")
            checks.append(dict(check="published_matrix_reproduction", max_abs_difference=difference))
    identity = np.max(np.abs(np.vstack([minmax(snv(v)) for v in x])-np.vstack([minmax(v) for v in x])))
    checks.append(dict(check="SNV_then_minmax_equals_minmax", max_abs_difference=float(identity)))
    pd.concat(allmetrics, ignore_index=True).to_csv(args.output/"metrics.csv", index=False)
    pd.concat(pavis, ignore_index=True).to_csv(args.output/"PAVI_sensitivity.csv", index=False)
    (args.output/"checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(f"fixed-population conditions={len(allmetrics)}")


if __name__ == "__main__":
    main()
