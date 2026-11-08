#!/usr/bin/env python3
"""Recompute all public WBC spectra and Table 1 quantities."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGURE_SCRIPTS = ROOT / "scripts/make_figures"
sys.path.insert(0, str(FIGURE_SCRIPTS))

from breast_cancer import (  # noqa: E402
    eigenmap_coordinates,
    laplacian_modes,
    load_operators,
    load_wbc,
)
from quantum_geometry import response_eigenvalues  # noqa: E402


WEIGHTS = ("0p02", "0p05", "0p1", "0p2", "0p5", "0p8")
WEIGHT_VALUES = np.asarray((0.02, 0.05, 0.1, 0.2, 0.5, 0.8))
THRESHOLD = 0.5
WEYL_WINDOW_FACTOR = np.sqrt(10.0)


def isolated_partition(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split coordinates at the largest gap in the leading eigenmap."""
    order = np.argsort(coordinates[:, 0])
    split = int(np.argmax(np.diff(coordinates[order, 0]))) + 1
    left, right = order[:split], order[split:]
    return (left, right) if len(left) <= len(right) else (right, left)


def weyl_dimension(spectrum: np.ndarray) -> dict:
    """Fit one log decade centered on the noncommutative spectral scale."""
    positive = np.asarray(spectrum)[1:]
    counts = np.arange(1, len(positive) + 1, dtype=float)
    lambda_nc = float(np.sqrt(positive[0] * positive[-1]))
    lower = lambda_nc / WEYL_WINDOW_FACTOR
    upper = lambda_nc * WEYL_WINDOW_FACTOR
    selected = (positive >= lower) & (positive <= upper)
    x = np.log(positive[selected])
    y = np.log(counts[selected])
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = np.sum((y - fitted) ** 2)
    total = np.sum((y - y.mean()) ** 2)
    return {
        "dimension": float(2.0 * slope),
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(1.0 - residual / total),
        "lambda_nc": lambda_nc,
        "window_lower": float(lower),
        "window_upper": float(upper),
        "window_mode_count": int(np.count_nonzero(selected)),
        "counting_convention": "trivial zero mode excluded; positive modes count from 1",
        "window_convention": "one log decade centered on lambda_nc",
    }


def main() -> None:
    output = ROOT / "results/derived_quantities"
    output.mkdir(parents=True, exist_ok=True)
    data, labels = load_wbc()

    mean_spectra = []
    dimensions = []
    arrays: dict[str, np.ndarray] = {}
    operator_metadata = {}

    for tag, weight in zip(WEIGHTS, WEIGHT_VALUES):
        operators, metadata = load_operators(tag, include_metadata=True)
        energies, cloud, response_values = response_eigenvalues(data, operators)
        mean_descending = np.sort(response_values, axis=1)[:, ::-1].mean(axis=0)
        dimension = int(np.count_nonzero(mean_descending > THRESHOLD))
        mean_spectra.append(mean_descending)
        dimensions.append(dimension)
        arrays[f"response_eigenvalues_w{tag}"] = response_values
        arrays[f"mean_response_eigenvalues_w{tag}"] = mean_descending
        operator_metadata[tag] = metadata

        if tag in {"0p2", "0p5"}:
            laplacian_spectrum, modes = laplacian_modes(operators, count=31)
            coordinates = eigenmap_coordinates(data, operators, modes[:3])
            overlaps = np.abs(
                np.einsum("mij,aji->am", modes, operators)
            )
            arrays[f"laplacian_spectrum_w{tag}"] = laplacian_spectrum
            arrays[f"eigenmap_coordinates_w{tag}"] = coordinates
            arrays[f"feature_overlaps_w{tag}"] = overlaps
            arrays[f"qcml_cloud_w{tag}"] = cloud
            arrays[f"ground_energies_w{tag}"] = energies

    mean_spectra_array = np.asarray(mean_spectra)
    dimensions_array = np.asarray(dimensions, dtype=np.int64)
    metadata = {
        "experiment": "wbc_all_weights",
        "weights": WEIGHT_VALUES.tolist(),
        "dimension_threshold": THRESHOLD,
        "standardization": (
            "all 569 samples; zero mean and population unit variance"
        ),
        "laplacian_normalization": "Delta / 30",
        "operator_metadata": operator_metadata,
    }
    np.savez_compressed(
        output / "wbc_all_weights.npz",
        weights=WEIGHT_VALUES,
        mean_response_eigenvalues=mean_spectra_array,
        dimensions=dimensions_array,
        labels=labels,
        metadata=json.dumps(metadata, sort_keys=True),
        **arrays,
    )

    with (output / "wbc_response_spectra.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["weight", "response_index", "mean_response_eigenvalue", "above_0.5"]
        )
        for weight, spectrum in zip(WEIGHT_VALUES, mean_spectra_array):
            for index, value in enumerate(spectrum, start=1):
                writer.writerow(
                    [f"{weight:g}", index, f"{value:.12g}", bool(value > THRESHOLD)]
                )

    with (output / "wbc_intrinsic_dimensions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["weight", "dimension", "threshold"])
        for weight, dimension in zip(WEIGHT_VALUES, dimensions_array):
            writer.writerow([f"{weight:g}", int(dimension), THRESHOLD])

    w05_coordinates = arrays["eigenmap_coordinates_w0p5"]
    isolated, main_cloud = isolated_partition(w05_coordinates)
    isolated_distance = float(
        np.linalg.norm(w05_coordinates[isolated[0]] - w05_coordinates[isolated[1]])
    )
    summary = {
        "weights": WEIGHT_VALUES.tolist(),
        "dimensions": dimensions_array.tolist(),
        "threshold": THRESHOLD,
        "w0p5_isolated_indices_zero_based": isolated.tolist(),
        "w0p5_isolated_labels": labels[isolated].tolist(),
        "w0p5_isolated_distance": isolated_distance,
        "w0p5_main_cloud_count": int(len(main_cloud)),
        "w0p5_main_cloud_y1_range": [
            float(w05_coordinates[main_cloud, 0].min()),
            float(w05_coordinates[main_cloud, 0].max()),
        ],
        "w0p2_weyl_fit": weyl_dimension(arrays["laplacian_spectrum_w0p2"]),
    }
    with (output / "wbc_analysis_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
