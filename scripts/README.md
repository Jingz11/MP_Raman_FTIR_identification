# Core reproducibility scripts

Only the code needed to reproduce the main workflow is retained here.

## Files

- `core.py`: FTIR/Raman preprocessing and the seven fixed model configurations.
- `01_preprocess.py`: reads downloaded two-column spectra and builds the processed matrix plus QC metadata.
- `02_evaluate.py`: grouped within-domain validation, cross-library transfer, and 2,000-repeat group bootstrap intervals.
- `03_pavi.py`: rebuilds PAVI from nearest-centroid cross-library predictions.
- `smoke_test.py`: fast check that does not require downloaded data.
- `manifest_template.csv`: input manifest header.
- `label_mapping.json`: frozen raw-to-harmonized polymer labels and exclusions.

## Input manifest

Create one row per raw spectrum. Paths may be absolute or relative to the manifest.
`polymer` may contain the harmonized label used for modelling. If it is blank,
`raw_label` is resolved through `label_mapping.json`. `group_id` identifies
technical or biological replicates that must remain in the same CV fold.

For FLOPP/FLOPP-e FTIR use `representation=percent_transmittance`; use
`native_intensity` for absorbance/intensity sources. For Raman wavelength files use
`axis_unit=nm`. When `excitation_nm` is blank, the recorded MicroPlastiX rule is used:
first wavelength below 700 nm -> 532.13 nm, otherwise 785 nm. For the authoritative
harmonized Raman analysis use `raman_workflow=harmonized`; `reference` omits ALS.

## Run

```bash
python scripts/01_preprocess.py --manifest data/manifest.csv --output results/processed
python scripts/02_evaluate.py --matrix results/processed/spectral_matrix.csv --metadata results/processed/metadata.csv --output results/models
python scripts/03_pavi.py --predictions results/models/predictions.csv --output results/PAVI.csv
python scripts/smoke_test.py
```

The fixed grids are 682-3998 cm-1 at 2 cm-1 for FTIR (1,659 variables) and
225-1997 cm-1 at 2 cm-1 for Raman (887 variables). Preprocessing is sort/deduplicate,
conversion when required, linear interpolation, MA7, optional Raman ALS
(lambda=1e5, p=0.01, 10 iterations), SNV, and per-spectrum min-max normalization.
