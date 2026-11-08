#!/usr/bin/env python3
"""Fast integrity checks for the publication-facing repository."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/make_figures"))
from quantum_geometry import matrix_laplacian, validate_operators  # noqa: E402


def main():
    files = sorted((ROOT / "results/trained_operators").glob("*.npz"))
    if not files:
        raise RuntimeError("No public operator files found")
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            operators = validate_operators(archive["operators"])
            metadata = json.loads(str(archive["metadata"]))
        laplacian = matrix_laplacian(operators)
        if not np.allclose(laplacian, laplacian.conj().T, atol=2e-6):
            raise AssertionError(f"Non-Hermitian Laplacian: {path}")
        if metadata.get("experiment") != path.stem:
            raise AssertionError(f"Metadata mismatch: {path}")
        print(f"OK {path.name}: operators {operators.shape}")

    expected_wbc = {
        "wbc_N8.npz",
        "wbc_N8_w0p02.npz",
        "wbc_N8_w0p05.npz",
        "wbc_N8_w0p2.npz",
        "wbc_N8_w0p5.npz",
        "wbc_N8_w0p8.npz",
    }
    actual_wbc = {path.name for path in files if path.name.startswith("wbc_N8")}
    if actual_wbc != expected_wbc:
        raise AssertionError(
            f"Unexpected WBC operator set: {sorted(actual_wbc)}"
        )

    wbc_derived = ROOT / "results/derived_quantities/wbc_all_weights.npz"
    with np.load(wbc_derived, allow_pickle=False) as archive:
        weights = archive["weights"]
        spectra = archive["mean_response_eigenvalues"]
        dimensions = archive["dimensions"]
        coordinates_02 = archive["eigenmap_coordinates_w0p2"]
        coordinates_05 = archive["eigenmap_coordinates_w0p5"]
    if weights.shape != (6,) or spectra.shape != (6, 30):
        raise AssertionError("Unexpected WBC response-spectrum shapes")
    if dimensions.tolist() != [7, 5, 4, 3, 2, 1]:
        raise AssertionError(f"Unexpected WBC dimensions: {dimensions}")
    if coordinates_02.shape != (569, 3) or coordinates_05.shape != (569, 3):
        raise AssertionError("Unexpected WBC eigenmap-coordinate shapes")
    print("OK WBC six-weight spectra and dimensions")

    with (ROOT / "results/derived_quantities/wbc_analysis_summary.json").open(
        encoding="utf-8"
    ) as stream:
        wbc_summary = json.load(stream)
    if wbc_summary["w0p5_main_cloud_count"] != 567:
        raise AssertionError("Unexpected WBC w=0.5 main-cloud count")
    if sorted(wbc_summary["w0p5_isolated_labels"]) != ["M", "M"]:
        raise AssertionError("Unexpected WBC w=0.5 isolated labels")
    if not np.isclose(
        wbc_summary["w0p2_weyl_fit"]["dimension"], 2.5005, atol=5e-4
    ):
        raise AssertionError("Unexpected WBC w=0.2 Weyl-law dimension")
    print("OK WBC isolated-point and Weyl-law summaries")

    with (ROOT / "results/derived_quantities/wbc_classifier_results.json").open(
        encoding="utf-8"
    ) as stream:
        classifier_results = json.load(stream)["results"]
    expected_classifier_metrics = {
        ("0.2", "Logistic regression"): (0.9733, 0.9069),
        ("0.2", "Gradient boosting"): (0.9813, 0.9385),
        ("0.5", "Logistic regression"): (0.9642, 0.9122),
        ("0.5", "Gradient boosting"): (0.9704, 0.9296),
    }
    for key, expected in expected_classifier_metrics.items():
        weight, classifier = key
        aggregate = classifier_results[weight][classifier]["aggregate"]
        actual = (aggregate["roc_auc_mean"], aggregate["accuracy_mean"])
        if not np.allclose(actual, expected, atol=5e-4):
            raise AssertionError(
                f"Unexpected WBC classifier metrics for {weight}, {classifier}: "
                f"{actual}"
            )
    print("OK WBC classifier summaries")

    stability = sorted(
        (ROOT / "results/stability/two_spheres_seed_w").glob("*.npz")
    )
    if len(stability) != 34:
        raise AssertionError(
            f"Expected 34 two-sphere stability archives, found {len(stability)}"
        )
    combinations = set()
    for path in stability:
        with np.load(path, allow_pickle=False) as archive:
            operators = validate_operators(archive["operators"])
            metadata = json.loads(archive["metadata"].item())
            spectrum = archive["laplacian_spectrum"]
        if operators.shape != (3, 8, 8) or spectrum.shape != (64,):
            raise AssertionError(f"Unexpected stability shapes: {path}")
        combinations.add((metadata["data_seed"], metadata["training_seed"],
                          metadata["w"]))
    if len(combinations) != 34:
        raise AssertionError("Duplicate stability parameter combinations")
    print("OK two-sphere stability sweep: 34 unique runs")

if __name__ == "__main__":
    main()
