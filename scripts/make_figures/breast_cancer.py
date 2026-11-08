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


def load_operators(weight: str, include_metadata: bool = False):
    filenames = {
        "0p02": "wbc_N8_w0p02.npz",
        "0p05": "wbc_N8_w0p05.npz",
        "0p1": "wbc_N8.npz",
        "0p2": "wbc_N8_w0p2.npz",
        "0p5": "wbc_N8_w0p5.npz",
        "0p8": "wbc_N8_w0p8.npz",
    }
    if weight not in filenames:
        raise ValueError(f"Unsupported WBC weight tag: {weight}")
    path = ROOT / "results/trained_operators" / filenames[weight]
    with np.load(path, allow_pickle=False) as archive:
        operators = archive["operators"]
        if include_metadata:
            import json

            return operators, json.loads(str(archive["metadata"]))
        return operators


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


def diagnosis_scatter(ax, coordinates, labels, dimensions=3, marker_size=28):
    colors = np.where(labels == "M", "red", "green")
    spans = np.ptp(coordinates[:, :dimensions], axis=0)
    padding = np.maximum(0.025 * spans, 1e-6)
    lower = coordinates[:, :dimensions].min(axis=0) - padding
    upper = coordinates[:, :dimensions].max(axis=0) + padding
    if dimensions == 3:
        ax.scatter(
            *coordinates[:, :3].T,
            c=colors,
            s=marker_size,
            marker="o",
            alpha=0.85,
            edgecolors="none",
            depthshade=False,
        )
        ax.set(xlabel="Y1", ylabel="Y2", zlabel="Y3")
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_zlim(lower[2], upper[2])
        ax.set_box_aspect(spans)
        ax.view_init(elev=20, azim=-120)
    else:
        ax.scatter(
            *coordinates[:, :2].T,
            c=colors,
            s=marker_size,
            marker="o",
            alpha=0.85,
            edgecolors="none",
        )
        ax.set(xlabel="Y1", ylabel="Y2")
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
    for diagnosis, color in (("M", "red"), ("B", "green")):
        ax.scatter(
            [], [], c=color, s=marker_size, marker="o", label=diagnosis
        )
    ax.legend()


def plot_laplacian_spectrum_panel(output, operators, weight, filename):
    values, _ = laplacian_modes(operators)
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.plot(np.arange(len(values)), values, "o-", label=rf"$w={weight}$")
    ax.set(xlabel="Eigenvalue index", ylabel=r"$\lambda$")
    ax.grid(True, linestyle=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_overlap_panel(output, operators, filename):
    _, modes = laplacian_modes(operators)
    overlaps = np.abs(np.einsum("mij,aji->am", modes, operators))
    base = [
        "radius", "texture", "perimeter", "area", "smoothness",
        "compactness", "concavity", "concave_points", "symmetry",
        "fractal_dimension",
    ]
    names = [f"{name}{group}" for group in (1, 2, 3) for name in base]
    rows, cols = overlaps.shape
    fig, ax = plt.subplots(figsize=(cols * 0.4, rows * 0.4))
    maximum = float(overlaps.max())
    for row in range(rows):
        for col in range(cols):
            value = float(overlaps[row, col])
            circle = plt.Circle(
                (col, row),
                0.4 * value / maximum,
                color="navy",
                alpha=value / maximum,
            )
            ax.add_patch(circle)
    for row in range(rows + 1):
        ax.axhline(row - 0.5, color="gray", linewidth=0.5)
    for col in range(cols + 1):
        ax.axvline(col - 0.5, color="gray", linewidth=0.5)
    ax.set(
        xlim=(-0.5, cols - 0.5),
        ylim=(-0.5, rows - 0.5),
        xticks=np.arange(cols),
        yticks=np.arange(rows),
        xticklabels=[f"Y[{index}]" for index in range(1, cols + 1)],
        yticklabels=names,
    )
    ax.tick_params(axis="x", labelrotation=90, labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.xaxis.tick_top()
    ax.invert_yaxis()
    ax.set_aspect("equal")
    scalar = plt.cm.ScalarMappable(
        cmap="Blues", norm=plt.Normalize(vmin=0.0, vmax=maximum)
    )
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=ax, shrink=0.6, pad=0.02)
    colorbar.set_label("Overlap", fontsize=16)
    fig.tight_layout()
    fig.savefig(output / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_eigenmap_panel(
    output, data, labels, operators, weight, dimensions, filename
):
    _, modes = laplacian_modes(operators, count=3)
    coordinates = eigenmap_coordinates(data, operators, modes)
    if dimensions == 3:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        marker_size = 38 if weight == "0.5" else 28
        diagnosis_scatter(
            ax, coordinates, labels, dimensions=3, marker_size=marker_size
        )
    else:
        fig, ax = plt.subplots(figsize=(10, 10))
        diagnosis_scatter(ax, coordinates, labels, dimensions=2)
        ax.grid(True)
    ax.legend(title=rf"$w={weight}$")
    fig.tight_layout()
    fig.savefig(output / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_manuscript_panels(output, data, labels, operators_02, operators_05):
    prefix = "Figure-WBC_features30_points569-N8-E63878-MBatch100"
    plot_laplacian_spectrum_panel(
        output, operators_02, "0.2", f"{prefix}-W0.2-Laplacian-spectrum.png"
    )
    plot_overlap_panel(output, operators_02, f"{prefix}-W0.2-XYoverlap.png")
    plot_eigenmap_panel(
        output, data, labels, operators_02, "0.2", 3,
        f"{prefix}-W0.2-L3-emapCS-diagnosis.png",
    )
    plot_eigenmap_panel(
        output, data, labels, operators_02, "0.2", 2,
        f"{prefix}-W0.2-L2-emapCS-diagnosis.png",
    )
    plot_eigenmap_panel(
        output, data, labels, operators_05, "0.5", 3,
        f"{prefix}-W0.5-L3-emapCS-diagnosis.png",
    )
    plot_laplacian_spectrum_panel(
        output, operators_05, "0.5", f"{prefix}-W0.5-Laplacian-spectrum.png"
    )


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
    plot_manuscript_panels(
        output, data, labels, operators_02, operators_05
    )
    print(f"Wrote WBC Figures 14--16 to {output}")


if __name__ == "__main__":
    main()
