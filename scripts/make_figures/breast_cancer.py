#!/usr/bin/env python3
"""Reproduce the WBC matrix-Laplacian analyses in Figures 14--16."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from quantum_geometry import (  # noqa: E402
    coherent_states,
    expectation_values,
    matrix_laplacian,
)


def load_wbc():
    raw = np.genfromtxt(
        ROOT / "data/input/breast_cancer/wdbc.data", delimiter=",", dtype=str
    )
    labels = raw[:, 1]
    data = raw[:, 2:].astype(float)
    data = (data - data.mean(axis=0)) / data.std(axis=0, ddof=0)
    return data, labels


def load_operators(weight: str):
    path = ROOT / f"results/trained_operators/wbc_N8_w{weight}.npz"
    with np.load(path, allow_pickle=False) as archive:
        return archive["operators"]


def laplacian_modes(operators, count=31):
    laplacian = matrix_laplacian(operators) / operators.shape[0]
    values, vectors = np.linalg.eigh(laplacian)
    n = operators.shape[1]
    modes = vectors[:, 1 : count + 1].T.reshape(count, n, n)
    modes = 0.5 * (modes + modes.conj().transpose(0, 2, 1))
    return values.real, modes


def eigenmap_coordinates(data, operators, modes):
    _, states = coherent_states(data, operators)
    return expectation_values(states, modes)


def plot_figure_14(output, operators):
    values, modes = laplacian_modes(operators)
    overlaps = np.abs(np.einsum("mij,aji->am", modes, operators))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(np.arange(len(values)), values, "o-")
    axes[0].set(
        xlabel="Eigenvalue index",
        ylabel=r"$\lambda$",
        title=r"Matrix-Laplacian spectrum, $w=0.2$",
    )

    yy, xx = np.indices(overlaps.shape)
    scatter = axes[1].scatter(
        xx.ravel() + 1,
        yy.ravel(),
        s=22.0 * overlaps.ravel(),
        c=overlaps.ravel(),
        cmap="Blues",
        vmin=0.0,
        vmax=4.0,
        edgecolors="none",
    )
    base = [
        "radius",
        "texture",
        "perimeter",
        "area",
        "smoothness",
        "compactness",
        "concavity",
        "concave_points",
        "symmetry",
        "fractal_dimension",
    ]
    names = [f"{name}{group}" for group in (1, 2, 3) for name in base]
    axes[1].set(
        xlabel="",
        ylabel="",
        xticks=np.arange(31) + 1,
        xticklabels=[f"Y[{i}]" for i in range(1, 32)],
        yticks=np.arange(30),
        yticklabels=names,
    )
    axes[1].invert_yaxis()
    axes[1].xaxis.tick_top()
    axes[1].tick_params(axis="x", rotation=90)
    axes[1].grid(True)
    fig.colorbar(scatter, ax=axes[1])
    fig.tight_layout()
    fig.savefig(output / "figure_14_wbc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def diagnosis_scatter(ax, coordinates, labels, dimensions=3):
    colors = np.where(labels == "M", "red", "green")
    if dimensions == 3:
        ax.scatter(*coordinates[:, :3].T, c=colors, s=8, marker=".")
        ax.set(xlabel="Y1", ylabel="Y2", zlabel="Y3")
    else:
        ax.scatter(*coordinates[:, :2].T, c=colors, s=8, marker=".")
        ax.set(xlabel="Y1", ylabel="Y2")
    for diagnosis, color in (("M", "red"), ("B", "green")):
        ax.scatter([], [], c=color, s=12, label=diagnosis)
    ax.legend()


def plot_figure_15(output, data, labels, operators):
    _, modes = laplacian_modes(operators, count=3)
    coordinates = eigenmap_coordinates(data, operators, modes)

    fig = plt.figure(figsize=(13, 5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)
    diagnosis_scatter(ax3d, coordinates, labels, dimensions=3)
    diagnosis_scatter(ax2d, coordinates, labels, dimensions=2)
    ax3d.set_title(r"WBC Laplacian eigenmaps, $w=0.2$")
    ax2d.set_title("Projection onto the two leading eigenmaps")
    fig.tight_layout()
    fig.savefig(output / "figure_15_wbc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_figure_16(output, data, labels, operators):
    values, modes = laplacian_modes(operators, count=3)
    coordinates = eigenmap_coordinates(data, operators, modes)

    fig = plt.figure(figsize=(13, 5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_spectrum = fig.add_subplot(1, 2, 2)
    diagnosis_scatter(ax3d, coordinates, labels, dimensions=3)
    ax3d.set_title(r"WBC Laplacian eigenmaps, $w=0.5$")
    ax_spectrum.plot(np.arange(len(values)), values, "o-")
    ax_spectrum.set(
        xlabel="Eigenvalue index",
        ylabel=r"$\lambda$",
        title=r"Matrix-Laplacian spectrum, $w=0.5$",
    )
    fig.tight_layout()
    fig.savefig(output / "figure_16_wbc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    output = ROOT / "figures/generated/figure_14_breast_cancer"
    output.mkdir(parents=True, exist_ok=True)
    data, labels = load_wbc()
    operators_02 = load_operators("0p2")
    operators_05 = load_operators("0p5")

    plot_figure_14(output, operators_02)
    plot_figure_15(output, data, labels, operators_02)
    plot_figure_16(output, data, labels, operators_05)
    print(f"Wrote WBC Figures 14--16 to {output}")


if __name__ == "__main__":
    main()
