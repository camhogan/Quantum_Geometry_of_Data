#!/usr/bin/env python3
"""Reproduce the uniform-sphere validation in paper Figures 2 and 4."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from quantum_geometry import matrix_laplacian  # noqa: E402


# Author notebook grid-search result. The omitted fourth column is the local
# energy gap; Figure 2 uses only these three spatial coordinates.
DEGENERACIES = np.array([
    [-0.0100, -0.0080, -0.0069],
    [0.0014, 0.0009, -0.0006],
    [0.0052, -0.0042, 0.0146],
])


def reference_sphere(ax):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    ax.plot_surface(
        np.outer(np.cos(u), np.sin(v)),
        np.outer(np.sin(u), np.sin(v)),
        np.outer(np.ones_like(u), np.cos(v)),
        color="gray", alpha=0.08, edgecolor="none", zorder=0,
    )


def style_3d(ax):
    ax.set(xlabel="x", ylabel="y", zlabel="z", xlim=(-1.1, 1.1),
           ylim=(-1.1, 1.1), zlim=(-1.1, 1.1))
    ax.set_box_aspect((1, 1, 1))


def fuzzy_sphere_spectrum(j):
    return np.concatenate([
        np.full(2 * ell + 1, ell * (ell + 1), dtype=float)
        for ell in range(int(2 * j) + 1)
    ])


def main():
    data = np.load(ROOT / "data/input/spheres/sphere_uniform.npy")
    operators = np.load(
        ROOT / "results/trained_operators/sphere_uniform_N4.npz"
    )["operators"]
    fuzzy = np.load(
        ROOT / "results/derived_quantities/sphere_uniform_N4.npz"
    )["fuzzy_manifold"]
    output = ROOT / "figures/generated/figure_02_04_uniform_sphere"
    output.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 6))
    left = fig.add_subplot(121, projection="3d")
    left.scatter(*data.T, color="black", s=2)
    reference_sphere(left)
    style_3d(left)

    right = fig.add_subplot(122, projection="3d")
    right.scatter(*fuzzy.T, color="green", s=2)
    right.scatter(*DEGENERACIES.T, color="blue", s=50, zorder=20)
    reference_sphere(right)
    style_3d(right)
    fig.tight_layout()
    fig.savefig(output / "figure_02_uniform_sphere.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    learned = np.sort(np.linalg.eigvalsh(matrix_laplacian(operators)).real)
    j, w = 1.5, 0.1
    exact = fuzzy_sphere_spectrum(j) / (j + w) ** 2
    fig, ax = plt.subplots(figsize=(8, 6))
    indices = np.arange(operators.shape[1] ** 2)
    ax.plot(indices, learned, "o-", label="Learned spectrum", zorder=1)
    ax.scatter(indices, exact, color="red", s=20, alpha=0.6,
               label=r"$S_4^2$ spectrum", zorder=2)
    ax.set_xlabel("Eigenvalue index", fontsize=16)
    ax.set_ylabel(r"$\lambda$", fontsize=16)
    ax.legend(fontsize=14)
    fig.tight_layout()
    fig.savefig(output / "figure_04_laplacian_spectrum.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote paper Figures 2 and 4 to {output}")


if __name__ == "__main__":
    main()
