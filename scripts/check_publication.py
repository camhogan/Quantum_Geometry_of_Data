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
