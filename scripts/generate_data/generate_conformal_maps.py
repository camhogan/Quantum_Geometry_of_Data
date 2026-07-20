#!/usr/bin/env python3
"""Generate point-cloud representations of Poincare-disk automorphisms."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def generate(n_maps=2000, points_per_map=100, a_max=.9, z_max=1., noise=0., seed=42):
    rng = np.random.RandomState(seed)
    radius = z_max*np.sqrt(rng.rand(points_per_map))
    angle = 2*np.pi*rng.rand(points_per_map)
    base = radius*np.exp(1j*angle)
    data, parameters = [], []
    transforms = [(0j, 1+0j)]
    transforms.extend((
        a_max*np.sqrt(rng.rand())*np.exp(2j*np.pi*rng.rand()), 1+0j
    ) for _ in range(n_maps-1))
    for a, phase in transforms:
        mapped = phase*(base-a)/(1-np.conj(a)*base)
        real = mapped.real + rng.normal(0, noise, points_per_map)
        imag = mapped.imag + rng.normal(0, noise, points_per_map)
        data.append(np.column_stack((real, imag)).ravel())
        parameters.append(a)
    return np.asarray(data), np.asarray(parameters)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maps", type=int, default=2000)
    parser.add_argument("--points-per-map", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data, parameters = generate(args.maps, args.points_per_map, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, data)
    np.save(args.output.with_name(args.output.stem + "_parameters.npy"), parameters)
    print(f"Saved {data.shape} to {args.output}")


if __name__ == "__main__":
    main()

