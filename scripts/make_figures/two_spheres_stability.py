#!/usr/bin/env python3
"""Summarize the two-sphere training-seed and w stability study."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/stability/two_spheres_seed_w"
OUTPUT = ROOT / "figures/generated/two_spheres_seed_w_stability"


def load_runs():
    runs = []
    for path in sorted(RESULTS.glob("two_spheres_*.npz")):
        archive = np.load(path)
        metadata = json.loads(archive["metadata"].item())
        metadata["path"] = path
        metadata["fuzzy_manifold"] = archive["fuzzy_manifold"]
        spectrum = np.sort(archive["laplacian_spectrum"].real)
        metadata["lambda_1"] = float(spectrum[1])
        runs.append(metadata)
    if len(runs) != 34:
        raise RuntimeError(f"Expected 34 stability runs, found {len(runs)}")
    return runs


def circles(ax):
    theta = np.linspace(0, 2 * np.pi, 200)
    for center, radius in ((0, 1.5), (3, 1.0)):
        ax.plot(radius * np.cos(theta), center + radius * np.sin(theta),
                color="0.75", linewidth=0.7, zorder=0)


def training_grid(runs):
    seeds = [40, 41, 42, 43, 44]
    weights = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]
    fig, axes = plt.subplots(len(weights), len(seeds), figsize=(12, 16),
                             sharex=True, sharey=True)
    for row, w in enumerate(weights):
        for column, seed in enumerate(seeds):
            run = next(r for r in runs if r["data_seed"] == 42
                       and r["training_seed"] == seed and r["w"] == w)
            points = run["fuzzy_manifold"]
            ax = axes[row, column]
            circles(ax)
            ax.scatter(points[:, 0], points[:, 2], s=0.7, color="green",
                       alpha=0.55, rasterized=True)
            ax.set_aspect("equal")
            ax.set_xlim(-1.8, 1.8)
            ax.set_ylim(-1.8, 4.2)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"training seed {seed}", fontsize=10)
            if column == 0:
                ax.set_ylabel(f"w = {w:g}", fontsize=10)
    fig.suptitle("Two-sphere learned geometry: fixed data seed 42", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT / "training_seed_w_geometry_grid.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def data_seed_grid(runs):
    seeds = [40, 41, 42, 43, 44]
    fig, axes = plt.subplots(1, len(seeds), figsize=(12, 4), sharex=True,
                             sharey=True)
    for ax, seed in zip(axes, seeds):
        run = next(r for r in runs if r["data_seed"] == seed
                   and r["training_seed"] == 42 and r["w"] == 0.1)
        points = run["fuzzy_manifold"]
        circles(ax)
        ax.scatter(points[:, 0], points[:, 2], s=0.7, color="green",
                   alpha=0.55, rasterized=True)
        ax.set(aspect="equal", xlim=(-1.8, 1.8), ylim=(-1.8, 4.2),
               xticks=[], yticks=[])
        ax.set_title(f"data seed {seed}", fontsize=10)
    fig.suptitle("Dataset stability: training seed 42, w = 0.1", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT / "data_seed_geometry_grid.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def metric_summary(runs):
    weights = np.array([0.0, 0.01, 0.03, 0.1, 0.3, 1.0])
    metrics = [
        ("mean_surface_distance", "Mean distance to reference spheres"),
        ("bridge_fraction_0p25", "Fraction farther than 0.25"),
        ("final_variance", "Mean learned variance"),
        ("lambda_1", r"Second Laplacian eigenvalue $\lambda_1$"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (key, label) in zip(axes.ravel(), metrics):
        means, deviations = [], []
        for w in weights:
            values = [r[key] for r in runs if r["data_seed"] == 42
                      and r["w"] == w]
            means.append(np.mean(values))
            deviations.append(np.std(values, ddof=1))
        ax.errorbar(weights, means, yerr=deviations, fmt="o-", capsize=3)
        ax.axvline(0.1, color="0.5", linestyle="--", linewidth=1,
                   label="paper w = 0.1")
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_xlabel("w")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("Two-sphere stability across five training seeds")
    fig.tight_layout()
    fig.savefig(OUTPUT / "stability_metrics_vs_w.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def write_summary(runs):
    keys = [
        "data_seed", "training_seed", "probe_seed", "w", "epochs",
        "minibatch_size", "final_loss", "final_bias", "final_variance",
        "mean_surface_distance", "p95_surface_distance",
        "bridge_fraction_0p25", "component_0_fraction",
        "component_1_fraction", "lambda_1", "training_seconds",
    ]
    with (RESULTS / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for run in sorted(runs, key=lambda r: (r["data_seed"], r["w"],
                                               r["training_seed"])):
            writer.writerow({key: run[key] for key in keys})


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    training_grid(runs)
    data_seed_grid(runs)
    metric_summary(runs)
    write_summary(runs)
    print(f"Wrote two-sphere stability figures to {OUTPUT}")


if __name__ == "__main__":
    main()
