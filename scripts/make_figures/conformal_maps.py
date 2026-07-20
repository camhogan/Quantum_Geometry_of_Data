#!/usr/bin/env python3
"""Reproduce the public analyses for the conformal-map example (Figures 9--13)."""

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from quantum_geometry import coherent_states, matrix_laplacian, expectation_values  # noqa: E402


def infer_parameters(data):
    """Infer `a` for the author convention f(z)=(z-a)/(1-conj(a)z)."""
    base = data[0, 0::2] + 1j*data[0, 1::2]
    result = []
    for row in data:
        mapped = row[0::2] + 1j*row[1::2]
        c, b = mapped*base, base-mapped
        # a - (w z) conj(a) = z - w, written as a real 2x2 system.
        design = np.vstack((np.column_stack((1-c.real, -c.imag)),
                            np.column_stack((-c.imag, 1+c.real))))
        target = np.concatenate((b.real, b.imag))
        u, v = np.linalg.lstsq(design, target, rcond=None)[0]
        result.append(u+1j*v)
    return np.asarray(result)


def main():
    data = np.load(ROOT/"data/input/conformal_maps/conformal_maps.npy")
    operators = np.load(ROOT/"results/trained_operators/conformal_maps_N8.npz")["operators"]
    derived = np.load(ROOT/"results/derived_quantities/conformal_maps_N8.npz")
    fuzzy = derived["fuzzy_manifold"]
    metric_eigenvalues = derived["metric_eigenvalues"]
    parameters = infer_parameters(data)
    output = ROOT/"figures/generated/figure_09_13_conformal_maps"; output.mkdir(parents=True, exist_ok=True)

    picks = [0, 3, 10, 13]
    fig, axes = plt.subplots(1, len(picks), figsize=(15, 3))
    for ax, index in zip(axes, picks):
        ax.scatter(data[index, 0::2], data[index, 1::2], s=7, c="black", label="data")
        ax.scatter(fuzzy[index, 0::2], fuzzy[index, 1::2], s=7, c="tab:orange", label="QCML")
        circle=np.linspace(0,2*np.pi,300); ax.plot(np.cos(circle),np.sin(circle),"r--")
        title = "#0 Reference set\na = 0" if index == 0 else (
            f"#{index}, arg(a) = {np.angle(parameters[index])/np.pi:.2f}π\n"
            f"|a| = {abs(parameters[index]):.2f}"
        )
        ax.set_aspect("equal"); ax.set_title(title)
        ax.axis("off")
    axes[0].legend(); fig.tight_layout(); fig.savefig(output/"figure_09_maps.png", dpi=300); plt.close(fig)

    laplacian = matrix_laplacian(operators)/operators.shape[0]
    lap_values, lap_vectors = np.linalg.eigh(laplacian)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ordered = np.sort(metric_eigenvalues, axis=1)[:, ::-1]
    means=ordered.mean(0); lows=ordered.min(0); highs=ordered.max(0)
    axes[0].vlines(np.arange(10),lows[:10],highs[:10],color=".82",linewidth=8)
    axes[0].plot(np.arange(10),means[:10],"ko",label="Average")
    axes[0].set(xlabel="Eigenvalue Index",ylabel="Metric Eigenvalue"); axes[0].legend(); axes[0].grid(True)
    positive = np.sort(lap_values[lap_values > 1e-6])
    counts=np.arange(1,len(positive)+1); axes[1].loglog(positive,counts,label=r"$N(\lambda)$")
    scale=np.median(counts/positive); axes[1].loglog(positive,scale*positive,"--",label=r"$\lambda^{d/2}, d=2$, Weyl law")
    axes[1].set(xlabel=r"$\lambda$",ylabel=r"$N(\lambda)$ - Counting function"); axes[1].legend(); axes[1].grid(True,which="both",linestyle=":")
    fig.tight_layout(); fig.savefig(output/"figure_10_metric_and_weyl.png", dpi=300); plt.close(fig)

    modes = lap_vectors[:, 1:33].T.reshape(32, 8, 8)
    modes=.5*(modes+modes.conj().transpose(0,2,1))
    # Match the author's notebook definition: |Tr(Y_i X_j)|.
    overlaps = np.abs(np.einsum("mij,aji->am", modes, operators))
    rows=np.arange(16); fig, ax = plt.subplots(figsize=(16, 8))
    yy, xx = np.indices((len(rows), overlaps.shape[1])); values = overlaps[rows]
    marker_areas = 140.0 * values.ravel()
    scatter = ax.scatter(
        xx.ravel(), yy.ravel(), s=marker_areas, c=values.ravel(),
        cmap="Blues", vmin=0.0, vmax=1.7, edgecolors="none"
    )
    ax.set(xlabel="",ylabel="",xticks=range(32),xticklabels=[f"Y[{i}]" for i in range(1,33)],
           yticks=range(16),yticklabels=[f"X[{i}]" for i in range(16)])
    ax.invert_yaxis()
    ax.xaxis.tick_top(); ax.tick_params(axis="x",rotation=90); ax.grid(True)
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Overlap")
    fig.tight_layout(); fig.savefig(output/"figure_11_operator_overlap.png", dpi=300); plt.close(fig)

    _, states = coherent_states(data, operators)
    coordinates = expectation_values(states, modes[:3])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, color, title in zip(axes, (np.abs(parameters), np.angle(parameters)), ("|a|", "arg(a)")):
        sc=ax.scatter(coordinates[:,0], coordinates[:,1], c=color, s=8, cmap="hsv" if "arg" in title else "viridis")
        ax.set_aspect("equal"); ax.set_title(title); fig.colorbar(sc, ax=ax)
    fig.tight_layout(); fig.savefig(output/"figure_12_eigenmap_2d.png", dpi=300); plt.close(fig)

    fig = plt.figure(figsize=(11, 5))
    for i, (color, title) in enumerate(((np.abs(parameters), "|a|"), (np.angle(parameters), "arg(a)")), 1):
        ax=fig.add_subplot(1,2,i,projection="3d"); sc=ax.scatter(*coordinates.T,c=color,s=8,cmap="hsv" if i==2 else "viridis")
        ax.set_title(title); fig.colorbar(sc,ax=ax,shrink=.65)
    fig.tight_layout(); fig.savefig(output/"figure_13_eigenmap_3d.png", dpi=300); plt.close(fig)
    print(f"Wrote conformal-map plots to {output}")


if __name__ == "__main__": main()
