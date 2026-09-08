"""Rebuild the Polymer Analytical Variability Index from NC transfer results."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = pd.read_csv(args.predictions)
    data = data[(data.analysis == "cross_library") & (data.model == "NC")].copy()
    rows = []
    for modality, mod in data.groupby("modality", sort=True):
        experiment_ba, wrong_rates = {}, []
        keys = ["train_library", "test_library"]
        for key, exp in mod.groupby(keys, sort=True):
            experiment_ba[key] = balanced_accuracy_score(exp.true_polymer, exp.predicted_polymer)
            for polymer, block in exp.groupby("true_polymer", sort=True):
                wrong = block[block.predicted_polymer != polymer].predicted_polymer.value_counts()
                rate = 0.0 if wrong.empty else float(wrong.iloc[0] / len(block))
                wrong_rates.append({"experiment": key, "polymer": polymer, "rate": rate,
                                    "wrong_target": "" if wrong.empty else wrong.index[0]})
        wr = pd.DataFrame(wrong_rates)
        for polymer in sorted(mod.true_polymer.unique()):
            eligible = {tuple(x) for x in mod.loc[mod.true_polymer == polymer, keys].drop_duplicates().to_numpy()}
            comp_a = float(np.mean([1.0 - experiment_ba[k] for k in eligible]))
            positive = wr[(wr.polymer == polymer) & (wr.rate > 0)]
            comp_b = 0.0 if positive.empty else float(positive.rate.mean())
            dominant = positive.sort_values("rate", ascending=False).head(1)
            rows.append({
                "modality": modality, "polymer": polymer,
                "component_A": comp_a, "component_B": comp_b,
                "PAVI_score": (comp_a + comp_b) / 2.0,
                "n_transfer_directions": len(eligible),
                "dominant_wrong_target": "" if dominant.empty else dominant.iloc[0].wrong_target,
                "dominant_wrong_rate": 0.0 if dominant.empty else dominant.iloc[0].rate,
            })
    out = pd.DataFrame(rows).sort_values(["modality", "PAVI_score"], ascending=[True, False])
    out["PAVI_rank"] = out.groupby("modality").cumcount() + 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"PAVI rows={len(out)}")


if __name__ == "__main__":
    main()
