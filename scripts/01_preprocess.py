"""Build analysis-ready matrices from downloaded two-column spectra."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core import preprocess_spectrum

REQUIRED = {"spectrum_id", "source_file", "library", "modality"}


def read_two_column(path: Path):
    """Read comma/tab/semicolon/whitespace separated files, with or without headers."""
    attempts = [
        dict(sep=None, engine="python", header=None, comment="#"),
        dict(sep=r"\s+", engine="python", header=None, comment="#"),
    ]
    for kwargs in attempts:
        try:
            frame = pd.read_csv(path, **kwargs)
            numeric = frame.apply(pd.to_numeric, errors="coerce")
            usable = [c for c in numeric if numeric[c].notna().sum() >= 2]
            if len(usable) >= 2:
                pair = numeric[usable[:2]].dropna()
                return pair.iloc[:, 0].to_numpy(float), pair.iloc[:, 1].to_numpy(float)
        except Exception:
            pass
    raise ValueError("could not identify two numeric columns")


def value(row, name, default):
    item = row.get(name, default)
    return default if pd.isna(item) or str(item).strip() == "" else item


def normalized_modality(value):
    text = str(value).strip().upper()
    return "FTIR" if text in {"FTIR", "ATR", "ATR-FTIR"} else text


def load_label_map(path):
    if path is None or not path.exists():
        return {}, {}
    table = pd.read_json(path)
    exact, fallback = {}, {}
    for _, row in table.iterrows():
        record = (row.standard_label, value(row, "exclusion_reason", ""))
        exact[(str(row.library), normalized_modality(row.modality), str(row.raw_label))] = record
        fallback.setdefault((str(row.library), str(row.raw_label)), []).append(record)
    fallback = {key: records[0] for key, records in fallback.items() if len(set(records)) == 1}
    return exact, fallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label-map", type=Path,
                        default=Path(__file__).with_name("label_mapping.json"))
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    missing = REQUIRED - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    if manifest.spectrum_id.astype(str).duplicated().any():
        raise ValueError("spectrum_id must be unique")
    if "polymer" not in manifest and "raw_label" not in manifest:
        raise ValueError("manifest needs polymer, or raw_label for lookup in label_mapping.json")
    exact_map, fallback_map = load_label_map(args.label_map)

    vectors, metadata, failures = [], [], []
    for _, row in manifest.iterrows():
        sid = str(row.spectrum_id)
        source = Path(str(row.source_file))
        if not source.is_absolute():
            source = args.manifest.parent / source
        try:
            polymer = value(row, "polymer", "")
            if not polymer:
                raw_label = str(row.raw_label)
                key = (str(row.library), normalized_modality(row.modality), raw_label)
                mapped = exact_map.get(key, fallback_map.get((str(row.library), raw_label)))
                if mapped is None:
                    raise ValueError(f"no harmonized label for raw_label={raw_label!r}")
                polymer, exclusion = mapped
                if exclusion:
                    raise ValueError(f"excluded by label map: {exclusion}")
            axis, intensity = read_two_column(source)
            result = preprocess_spectrum(
                axis, intensity,
                modality=row.modality,
                representation=value(row, "representation", "native_intensity"),
                axis_unit=value(row, "axis_unit", "cm-1"),
                excitation_nm=value(row, "excitation_nm", None),
                raman_workflow=value(row, "raman_workflow", "harmonized"),
            )
            vectors.append(pd.Series(result.values, index=[f"{x:g}" for x in result.axis], name=sid))
            metadata.append({
                "spectrum_id": sid, "library": row.library, "modality": row.modality,
                "polymer": polymer, "group_id": value(row, "group_id", sid),
                "source_file": str(source), "coverage_fraction": result.coverage_fraction,
                "duplicate_points_removed": result.duplicate_points_removed,
                "conversion": result.conversion, "smoothing": result.smoothing,
                "baseline": result.baseline, "normalization": "SNV_then_minmax_0_1",
            })
        except Exception as exc:
            failures.append({"spectrum_id": sid, "source_file": str(source), "error": str(exc)})

    args.output.mkdir(parents=True, exist_ok=True)
    if not vectors:
        pd.DataFrame(failures).to_csv(args.output / "preprocessing_failures.csv", index=False)
        raise RuntimeError("no spectra were successfully preprocessed")
    matrix = pd.DataFrame(vectors)
    matrix.index.name = "spectrum_id"
    matrix.to_csv(args.output / "spectral_matrix.csv")
    pd.DataFrame(metadata).to_csv(args.output / "metadata.csv", index=False)
    pd.DataFrame(failures, columns=["spectrum_id", "source_file", "error"]).to_csv(
        args.output / "preprocessing_failures.csv", index=False
    )
    print(f"processed={len(vectors)} failed={len(failures)} features={matrix.shape[1]}")


if __name__ == "__main__":
    main()
