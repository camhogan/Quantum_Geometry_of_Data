#!/usr/bin/env python3
"""Reproduce the matrix-Laplacian analyses for the WBC example (Figure 14)."""

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from quantum_geometry import matrix_laplacian  # noqa: E402


def main():
    operators = np.load(ROOT/"results/trained_operators/wbc_N8.npz")["operators"]
    laplacian = matrix_laplacian(operators)/operators.shape[0]
    values, vectors = np.linalg.eigh(laplacian)
    modes = vectors[:, 1:32].T.reshape(31, 8, 8)
    modes = .5*(modes+modes.conj().transpose(0,2,1))
    # Match the author's notebook definition: |Tr(Y_i X_j)|.
    overlaps = np.abs(np.einsum("mij,aji->am", modes, operators))
    output = ROOT/"figures/generated/figure_14_breast_cancer"; output.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    count = len(values); axes[0].plot(np.arange(count), values[:count], "o-")
    axes[0].set(xlabel="Mode", ylabel="Eigenvalue", title="Partial matrix-Laplacian spectrum")
    yy, xx = np.indices(overlaps.shape)
    # Use overlap-proportional marker area, as in the paper.  Keeping the
    # absolute scale (instead of normalizing every plot to one giant bubble)
    # makes weak overlaps disappear quickly and leaves the grid legible.
    marker_areas = 22.0 * overlaps.ravel()
    scatter=axes[1].scatter(
        xx.ravel()+1, yy.ravel(), s=marker_areas, c=overlaps.ravel(),
        cmap="Blues", vmin=0.0, vmax=4.0, edgecolors="none"
    )
    base=["radius","texture","perimeter","area","smoothness","compactness","concavity","concave_points","symmetry","fractal_dimension"]
    names=[f"{name}{group}" for group in (1,2,3) for name in base]
    axes[1].set(xlabel="",ylabel="",xticks=np.arange(31)+1,xticklabels=[f"Y[{i}]" for i in range(1,32)],
                yticks=np.arange(30),yticklabels=names)
    axes[1].invert_yaxis()
    axes[1].xaxis.tick_top(); axes[1].tick_params(axis="x",rotation=90); axes[1].grid(True)
    fig.colorbar(scatter, ax=axes[1]); fig.tight_layout()
    fig.savefig(output/"figure_14_wbc.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote WBC plots to {output}")


if __name__ == "__main__": main()
