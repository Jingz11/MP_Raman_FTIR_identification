"""Polymer Analytical Vulnerability Indicator, current zero-inclusive definition."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

REFERENCE_SOURCES = {"ATR_FLOPP", "ATR_FLOPPe", "Raman_SLoPP", "Raman_RaSPI",
                     "Raman_pristine", "Raman_weathered",
                     "FLOPP", "FLOPP-e", "SLoPP", "SLoPP-E", "RaSPI"}


def canonical_predictions(data, reference_sources=None):
    data = data.copy()
    if "analysis" in data:
        data = data[data.analysis.eq("cross_library")]
    col = "method" if "method" in data else "model"
    data = data[data[col].isin(["NC", "NC_euclid"])].copy()
    if "scaler_used" in data:
        data = data[~data.scaler_used.astype(str).str.lower().isin(["true", "1"])].copy()
    data = data.rename(columns={"predicted_polymer": "pred_polymer", "train_library": "source_domain",
                               "test_library": "target_domain"})
    if "experiment_id" not in data:
        data["experiment_id"] = data.source_domain.astype(str) + "_to_" + data.target_domain.astype(str)
    for col, value in [("workflow", "unspecified"), ("library", None), ("subdomain", "")]:
        if col not in data:
            data[col] = data.target_domain if value is None else value
    group = next((c for c in ["group_id", "group_key_corrected", "group_key"] if c in data), None)
    if group is None or data[group].isna().any() or data[group].astype(str).eq("").any():
        raise ValueError("PAVI requires recorded target groups")
    data["subdomain"] = data.subdomain.fillna("")
    data["gid"] = data.library.astype(str) + "||" + data.subdomain.astype(str) + "||" + data[group].astype(str)
    refs = REFERENCE_SOURCES if reference_sources is None else set(reference_sources)
    data["reference_trained"] = data.source_domain.isin(refs)
    if data.empty or not data.reference_trained.any():
        raise ValueError("no unscaled NC reference-trained predictions; check --reference-sources")
    if data.duplicated(["workflow", "modality", "experiment_id", "library", "subdomain", "spectrum_id"]).any():
        raise ValueError("duplicate target predictions in a transfer")
    return data


def cells_from_predictions(frame):
    rows, matrices, groupkeys = [], [], []
    for (exp, c), g in frame.groupby(["experiment_id", "true_polymer"], sort=True):
        labels = sorted(set(g.pred_polymer) | {c})
        table = pd.crosstab(g.gid, g.pred_polymer).reindex(columns=labels, fill_value=0)
        counts = table.to_numpy(float); correct = labels.index(c)
        wrong = counts.sum(axis=0); wrong[correct] = 0
        rows.append(dict(experiment_id=exp, polymer=c, n=len(g), n_groups=len(table),
                         error=float(g.pred_polymer.ne(c).mean()), dominant=float(wrong.max()/len(g)),
                         reference_trained=bool(g.reference_trained.iloc[0]), correct_i=correct))
        matrices.append(counts); groupkeys.append(table.index.tolist())
    return pd.DataFrame(rows), matrices, groupkeys


def summarize(cells, minimum=5, weight=.5):
    rows = []
    for polymer, g in cells[cells.n.ge(minimum)].groupby("polymer", sort=True):
        ref = g[g.reference_trained]
        a, b = float(g.error.mean()), float(ref.dominant.mean()) if len(ref) else np.nan
        rows.append(dict(polymer=polymer, component_A=a, component_B=b, PAVI_score=weight*a+(1-weight)*b,
                         n_A=len(g), n_B=len(ref), min_target_n=int(g.n.min()),
                         min_target_groups=int(g.n_groups.min()), target_appearances=int(g.n.sum()),
                         low_support=bool(g.n.min()<10 or len(g)<5)))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["PAVI_rank"] = out.PAVI_score.rank(ascending=False, method="average")
    return out


def bootstrap_cells(frame, cells, matrices, groupkeys, repeats, rng):
    if (frame.groupby("gid").true_polymer.nunique() > 1).any():
        raise ValueError("target groups must have one true polymer")
    membership = frame.groupby("gid").experiment_id.agg(lambda x: ";".join(sorted(set(x))))
    unique = frame.drop_duplicates("gid").set_index("gid")
    unique["membership"] = membership
    keys = sorted(unique.index); pos = {k: i for i, k in enumerate(keys)}
    weights = np.zeros((repeats, len(keys)), dtype=np.int32)
    for _, g in unique.groupby(["library", "subdomain", "true_polymer", "membership"], sort=True):
        ix = [pos[k] for k in g.index]; n = len(ix)
        weights[:, ix] = rng.multinomial(n, np.full(n, 1/n), size=repeats)
    errors, dominant = [], []
    for i, row in cells.iterrows():
        counts = weights[:, [pos[k] for k in groupkeys[i]]] @ matrices[i]
        den = counts.sum(axis=1)
        if np.any(den == 0):
            raise ValueError("bootstrap produced an empty cell")
        errors.append(1-counts[:, int(row.correct_i)]/den)
        counts[:, int(row.correct_i)] = 0
        dominant.append(counts.max(axis=1)/den)
    return np.array(errors), np.array(dominant)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path, nargs="+")
    parser.add_argument("--workflow-names", nargs="+")
    parser.add_argument("--reference-sources", nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    if args.bootstrap < 0 or (args.workflow_names and len(args.workflow_names) != len(args.predictions)):
        parser.error("bootstrap must be nonnegative; supply one workflow name per input")
    frames = []
    for i, path in enumerate(args.predictions):
        frame = pd.read_csv(path)
        if args.workflow_names:
            frame["workflow"] = args.workflow_names[i]
        frames.append(frame)
    data = canonical_predictions(pd.concat(frames, ignore_index=True), args.reference_sources)
    outputs, sensitivities, support = [], [], []
    rng = np.random.default_rng(args.seed)
    panels = sorted(data.groupby(["workflow", "modality"], sort=False),
                    key=lambda item: ({"Reference": 0, "Harmonized": 1}.get(item[0][0], 2), item[0]))
    for (workflow, modality), frame in panels:
        cells, matrices, keys = cells_from_predictions(frame)
        out = summarize(cells)
        if args.bootstrap:
            e, d = bootstrap_cells(frame, cells, matrices, keys, args.bootstrap, rng)
            draws = []
            for polymer in out.polymer:
                a = cells.index[cells.polymer.eq(polymer) & cells.n.ge(5)]
                b = cells.index[cells.polymer.eq(polymer) & cells.n.ge(5) & cells.reference_trained]
                draws.append(.5*(e[a].mean(axis=0)+d[b].mean(axis=0)) if len(b) else np.full(args.bootstrap, np.nan))
            draws = np.array(draws).T
            ranks = rankdata(-draws, axis=1, method="average", nan_policy="omit")
            for j in range(len(out)):
                out.loc[j, "ci_low"], out.loc[j, "ci_high"] = np.quantile(draws[:, j], [.025, .975])
                out.loc[j, "top1_frequency"] = np.mean(ranks[:, j] == 1)
                out.loc[j, "top3_frequency"] = np.mean(ranks[:, j] <= 3)
                out.loc[j, "degenerate_interval"] = np.ptp(draws[:, j]) < 1e-12
        outputs.append(out.assign(workflow=workflow, modality=modality))
        support.append(cells.assign(workflow=workflow, modality=modality))
        for n in [5, 10, 15, 20]:
            for w in [.25, .5, .75]:
                sensitivities.append(summarize(cells, n, w).assign(workflow=workflow, modality=modality,
                                                                  min_target_n_threshold=n, component_A_weight=w))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(outputs, ignore_index=True).to_csv(args.output, index=False)
    pd.concat(sensitivities, ignore_index=True).to_csv(args.output.with_name(args.output.stem+"_sensitivity.csv"), index=False)
    pd.concat(support, ignore_index=True).to_csv(args.output.with_name(args.output.stem+"_cell_support.csv"), index=False)
    print(f"PAVI rows={sum(map(len, outputs))}; fixed-model bootstrap={args.bootstrap}")


if __name__ == "__main__":
    main()
