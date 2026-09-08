"""Core preprocessing and fixed model definitions for the manuscript."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import solveh_banded
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC

SEED = 42
FTIR_GRID = np.arange(682.0, 3998.0 + 1e-9, 2.0)   # 1,659 variables
RAMAN_GRID = np.arange(225.0, 1997.0 + 1e-9, 2.0)  # 887 variables
ALS_LAMBDA = 1e5
ALS_P = 0.01
ALS_ITERATIONS = 10


@dataclass(frozen=True)
class PreprocessingResult:
    axis: np.ndarray
    values: np.ndarray
    coverage_fraction: float
    duplicate_points_removed: int
    conversion: str
    smoothing: str
    baseline: str


def sort_and_deduplicate(axis, values):
    """Remove non-finite pairs, sort ascending, and average duplicate axes."""
    frame = pd.DataFrame({"axis": axis, "value": values}).apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < 2:
        raise ValueError("fewer than two finite axis/value pairs")
    before = len(frame)
    frame = frame.groupby("axis", as_index=False, sort=True)["value"].mean()
    return frame.axis.to_numpy(float), frame.value.to_numpy(float), before - len(frame)


def wavelength_to_raman_shift(wavelength_nm, excitation_nm):
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    return (1.0 / float(excitation_nm) - 1.0 / wavelength_nm) * 1e7


def percent_transmittance_to_absorbance(percent_t):
    """A = -log10(max(%T/100, 1e-4))."""
    return -np.log10(np.maximum(np.asarray(percent_t, dtype=float) / 100.0, 1e-4))


def moving_average(values, window=7):
    return pd.Series(np.asarray(values, dtype=float)).rolling(window, center=True, min_periods=1).mean().to_numpy(float)


def als_baseline(values, lam=ALS_LAMBDA, p=ALS_P, iterations=ALS_ITERATIONS):
    """Asymmetric least-squares baseline used after MA7 for harmonized Raman."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 3:
        raise ValueError("ALS needs at least three points")
    main = np.full(n, 6.0)
    main[:2], main[-2:] = [1.0, 5.0], [5.0, 1.0]
    first = np.full(n - 1, -4.0)
    first[0] = first[-1] = -2.0
    second = np.ones(n - 2)
    weights = np.ones(n)
    baseline = values.copy()
    for _ in range(iterations):
        band = np.zeros((3, n))
        band[0] = weights + lam * main
        band[1, :-1] = lam * first
        band[2, :-2] = lam * second
        baseline = solveh_banded(band, weights * values, lower=True, check_finite=False)
        weights = np.where(values > baseline, p, 1.0 - p)
    return baseline


def snv(values):
    values = np.asarray(values, dtype=float)
    sd = float(np.std(values))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("SNV is undefined for a constant spectrum")
    return (values - np.mean(values)) / sd


def minmax(values):
    values = np.asarray(values, dtype=float)
    lo, hi = float(np.min(values)), float(np.max(values))
    if not np.isfinite(lo + hi) or hi <= lo:
        raise ValueError("min-max normalization is undefined")
    return (values - lo) / (hi - lo)


def preprocess_spectrum(axis, values, *, modality, representation="native_intensity",
                        axis_unit="cm-1", excitation_nm=None, raman_workflow="harmonized"):
    """Run the manuscript pipeline on one FTIR/ATR or Raman spectrum."""
    modality = str(modality).upper()
    modality = "FTIR" if modality in {"FTIR", "ATR", "ATR-FTIR"} else modality
    if modality not in {"FTIR", "RAMAN"}:
        raise ValueError("modality must be FTIR/ATR or Raman")
    axis = np.asarray(axis, dtype=float)
    values = np.asarray(values, dtype=float)
    conversion = "none"
    if modality == "RAMAN" and str(axis_unit).lower() in {"nm", "nanometre", "nanometer"}:
        finite = axis[np.isfinite(axis)]
        if excitation_nm in {None, "", "auto"}:
            excitation_nm = 532.13 if finite[0] < 700 else 785.0
        axis = wavelength_to_raman_shift(axis, float(excitation_nm))
        conversion = f"wavelength_to_shift_{float(excitation_nm):g}nm"
    axis, values, removed = sort_and_deduplicate(axis, values)
    target = FTIR_GRID if modality == "FTIR" else RAMAN_GRID
    coverage = float(np.mean((target >= axis.min()) & (target <= axis.max())))
    if coverage < 1.0:
        raise ValueError(f"target-grid coverage is incomplete ({coverage:.3%})")
    if str(representation).lower() in {"percent_transmittance", "%t", "transmittance"}:
        if modality != "FTIR":
            raise ValueError("percent-transmittance conversion applies to FTIR only")
        values = percent_transmittance_to_absorbance(values)
        conversion = "percent_transmittance_to_absorbance_floor_1e-4"
    processed = moving_average(np.interp(target, axis, values), window=7)
    baseline = "none"
    if modality == "RAMAN" and str(raman_workflow).lower() == "harmonized":
        processed = processed - als_baseline(processed)
        baseline = "ALS_lambda_1e5_p_0.01_iterations_10"
    processed = minmax(snv(processed))
    return PreprocessingResult(target, processed, coverage, int(removed), conversion, "moving_average_7", baseline)


class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    """Five-component PLS-DA with one-hot targets and argmax prediction."""
    def __init__(self, n_components=5):
        self.n_components = n_components

    def fit(self, X, y):
        self.encoder_ = LabelEncoder().fit(y)
        y_onehot = np.eye(len(self.encoder_.classes_))[self.encoder_.transform(y)]
        n = max(1, min(self.n_components, X.shape[1], X.shape[0] - 1, len(self.encoder_.classes_) - 1))
        self.model_ = PLSRegression(n_components=n).fit(X, y_onehot)
        self.classes_ = self.encoder_.classes_
        return self

    def predict(self, X):
        return self.encoder_.inverse_transform(np.argmax(self.model_.predict(X), axis=1))


def within_domain_models():
    """Seven fixed within-domain models; all except RF include StandardScaler."""
    return {
        "NC": make_pipeline(StandardScaler(), NearestCentroid(metric="euclidean")),
        "1NN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1)),
        "3NN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)),
        "RF": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")),
        "LR": make_pipeline(StandardScaler(), LogisticRegression(C=1, solver="lbfgs", max_iter=5000,
                                                                   class_weight="balanced", random_state=SEED)),
        "PLSDA": make_pipeline(StandardScaler(), PLSDAClassifier(n_components=5)),
    }


def cross_library_models():
    """Seven fixed transfer models; no separate scaler is fitted across libraries."""
    return {
        "NC": NearestCentroid(metric="euclidean"),
        "1NN": KNeighborsClassifier(n_neighbors=1),
        "3NN": KNeighborsClassifier(n_neighbors=3),
        "RF": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "SVM": LinearSVC(C=1, max_iter=1_000_000, random_state=SEED),
        "LR": LogisticRegression(C=1, solver="lbfgs", max_iter=5000, class_weight="balanced", random_state=SEED),
        "PLSDA": PLSDAClassifier(n_components=5),
    }
