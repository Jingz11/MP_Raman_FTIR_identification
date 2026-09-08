"""Fast, data-free verification of preprocessing and model configuration."""
import numpy as np

from core import FTIR_GRID, RAMAN_GRID, cross_library_models, preprocess_spectrum, within_domain_models


def main():
    rng = np.random.default_rng(42)
    x = np.linspace(4100, 650, 1800)
    ftir = preprocess_spectrum(x, 70 + 10 * np.sin(x / 180), modality="FTIR",
                               representation="percent_transmittance")
    x = np.linspace(210, 2020, 1000)
    raman = preprocess_spectrum(x, 0.001 * x + np.sin(x / 35) + rng.normal(0, 0.03, len(x)),
                                modality="Raman", raman_workflow="harmonized")
    assert len(FTIR_GRID) == len(ftir.values) == 1659
    assert len(RAMAN_GRID) == len(raman.values) == 887
    assert np.isclose([ftir.values.min(), raman.values.min()], 0).all()
    assert np.isclose([ftir.values.max(), raman.values.max()], 1).all()
    names = {"NC", "1NN", "3NN", "RF", "SVM", "LR", "PLSDA"}
    assert set(within_domain_models()) == set(cross_library_models()) == names
    print("PASS: preprocessing grids, ALS, normalization, and seven model factories")


if __name__ == "__main__":
    main()
