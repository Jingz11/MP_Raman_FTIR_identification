# Core reproducibility code

This repository contains the core code needed to reproduce the spectral preprocessing and machine-learning analyses reported in the study. Third-party spectra are not redistributed. They should be downloaded from the sources listed in the manuscript and organized using the input manifest described in `scripts/README.md`.

## Included

- FTIR/ATR and Raman preprocessing
- Label harmonization
- Fixed configurations for seven classifiers
- Grouped within-domain validation
- Cross-library transfer evaluation
- Group-bootstrap confidence intervals
- PAVI calculation from nearest-centroid transfer predictions

## Installation

```bash
conda env create -f environment.yml
conda activate polymer-transfer-reference
