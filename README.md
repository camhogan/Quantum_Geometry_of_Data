# Quantum Geometry of Data

This repository accompanies the paper [*Quantum Geometry of Data*](https://arxiv.org/abs/2507.21135) by Alexander G. Abanov et al. It contains
the input datasets, final trained Hermitian operators, post-training analysis
code, and figure-generation scripts for the four examples in Section 5:

1. Two disconnected spheres with noise (Figures 5--6).
2. A sphere with a non-uniform distribution of points (Figures 7--8).
3. A dataset of conformal maps (Figures 9--13).
4. Wisconsin Breast Cancer data (Figure 14).

The HSM implementation and operator-training pipeline are not part of this
release. For access to these materials, please contact Alexander G. Abanov
directly. The trained operators are supplied directly as portable NumPy files;
the public scripts do not depend on the private model class or on PyTorch.

## Layout

- `data/input/`: input datasets used by the experiments.
- `results/trained_operators/`: final complex Hermitian operator arrays.
- `results/derived_quantities/`: selected post-training arrays used by plots.
- `scripts/generate_data/`: reproducible synthetic-data generators.
- `scripts/make_figures/`: public post-training analysis and plotting code.
- `figures/generated/`: numbered paper figures produced by the public scripts.

Each `.npz` result contains JSON metadata in a scalar `metadata` field. Trained
operator files contain an `operators` array with shape `(D, N, N)`, where `D`
is the input dimension and `N` is the Hilbert-space dimension.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce the paper examples

```bash
python scripts/make_figures/two_spheres.py
python scripts/make_figures/nonuniform_sphere.py
python scripts/make_figures/conformal_maps.py
python scripts/make_figures/breast_cancer.py
```

Run the fast public-artifact integrity checks with:

```bash
python scripts/check_publication.py
```

These commands recompute the applicable expectation-value geometry,
uncertainty, Berry curvature, metric information, matrix-Laplacian spectra,
operator/eigenmode overlaps, and Laplacian eigenmaps. Outputs are written below
`figures/generated/`.

## Regenerate the synthetic inputs

```bash
python scripts/generate_data/generate_spheres.py --experiment weighted \
  --output data/input/spheres/sphere_nonuniform_regenerated.npy
python scripts/generate_data/generate_spheres.py --experiment two \
  --output data/input/spheres/two_spheres_regenerated.npy
python scripts/generate_data/generate_conformal_maps.py \
  --output data/input/conformal_maps/conformal_maps_regenerated.npy
```

The archived author code and saved datasets are treated as canonical. The
non-uniform sphere therefore uses polar-angle density
`(1 + cos(theta))^3 sin(theta)`, and conformal maps use
`f(z) = (z - a)/(1 - conj(a) z)`. These conventions are retained even where
the manuscript presents an algebraically different expression.

The uniform sphere remains as an auxiliary basic-reference artifact from the
earlier explanatory sections; it is not one of the four Section 5 examples.

See the README files within `data/` and `results/` for format details.
