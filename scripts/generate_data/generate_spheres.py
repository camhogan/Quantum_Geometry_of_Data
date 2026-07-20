#!/usr/bin/env python3
"""Generate uniform or polar-weighted noisy sphere point clouds."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def uniform_spheres(centers, radii, total_points, noise=0.0, seed=42):
    rng = np.random.RandomState(seed)
    radii = np.asarray(radii, dtype=float)
    counts = np.round(radii**2 / np.sum(radii**2) * total_points).astype(int)
    clouds = []
    for center, radius, count in zip(centers, radii, counts):
        phi = rng.uniform(0, 2*np.pi, count)
        cos_theta = rng.uniform(-1, 1, count)
        sin_theta = np.sqrt(1-cos_theta**2)
        cloud = radius*np.column_stack((
            sin_theta*np.cos(phi), sin_theta*np.sin(phi), cos_theta
        )) + np.asarray(center)
        if noise:
            cloud += rng.normal(0, noise, cloud.shape)
        clouds.append(cloud)
    return np.vstack(clouds)


def weighted_sphere(radius, points, noise=0.1, seed=42):
    """Reproduce the author-code distribution used by the saved experiment.

    Its polar-angle density is (1 + cos(theta))^3 sin(theta). This code-level
    convention is intentionally retained for exact compatibility with the
    trained operators and differs from the squared expression in the paper.
    """
    rng = np.random.RandomState(seed)
    grid = np.linspace(0, np.pi, 1000)
    density = lambda t: (1+np.cos(t))**3*np.sin(t)
    maximum = density(grid).max()
    theta = []
    while len(theta) < points:
        candidate = rng.uniform(0, np.pi)
        if rng.uniform(0, maximum) < density(candidate):
            theta.append(candidate)
    theta = np.asarray(theta)
    phi = rng.uniform(0, 2*np.pi, points)
    cloud = radius*np.column_stack((
        np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
    ))
    if noise:
        cloud += rng.normal(0, noise, cloud.shape)
    return cloud


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("uniform", "two", "weighted"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=None)
    parser.add_argument("--noise", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.experiment == "uniform":
        data = uniform_spheres([(0,0,0)], [1], args.points or 1000,
                               0.0 if args.noise is None else args.noise, args.seed)
    elif args.experiment == "two":
        data = uniform_spheres([(0,0,0), (0,0,3)], [1.5,1], args.points or 2000,
                               0.1 if args.noise is None else args.noise, args.seed)
    else:
        data = weighted_sphere(1, args.points or 2000,
                               0.1 if args.noise is None else args.noise, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, data)
    print(f"Saved {data.shape} to {args.output}")


if __name__ == "__main__":
    main()
