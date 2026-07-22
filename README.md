# Quantum Geometry of Data

This repository accompanies the paper [*Quantum Geometry of Data*](https://arxiv.org/abs/2507.21135) by Alexander G. Abanov et al. It contains
the input datasets, final trained Hermitian operators, post-training analysis
code, and figure-generation scripts for the uniform-sphere validation and the
four examples in Section 5:

1. Uniform sphere validation (Figures 2 and 4).
2. Two disconnected spheres with noise (Figures 5--6).
3. A sphere with a non-uniform distribution of points (Figures 7--8).
4. A dataset of conformal maps (Figures 9--13).
5. Wisconsin Breast Cancer data (Figure 14).

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
python scripts/make_figures/uniform_sphere.py
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

## Reproducibility parameters

All controlled random processes use seed 42 for the paper runs. Dataset,
training, and plotting seeds are listed separately because they control
different sources of randomness. 

| Experiment | Dataset seed | Training seed | Plot/probe seed | Points | Input dimension | Noise | N | w |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Uniform sphere | 42 | 42 | not applicable | 1,000 | 3 | 0 | 4 | 0.1 |
| Two spheres | 42 | 42 | 42 | 2,000 | 3 | 0.1 | 8 | 0.1 |
| Non-uniform sphere | 42 | 42 | 42 | 2,000 | 3 | 0.1 | 8 | 0.1 |
| Conformal maps | not recorded (archive); 42 (public generator) | 42 | not applicable | 2,000 | 200 | 0 | 8 | 0.1 |
| Wisconsin Breast Cancer | not applicable | 42 | not applicable | 569 | 30 | not applicable | 8 | 0.1 |

`N` denotes Hilbert-space dimension. The synthetic sphere noise is isotropic
Gaussian standard deviation. Conformal inputs consist of 2,000 maps evaluated
on 100 shared disk points, hence input dimension 200; `a_max=0.9`, `z_max=1`,
and global rotation is zero. The two spheres have centers `(0,0,0)` and
`(0,0,3)`, radii 1.5 and 1, and point counts proportional to surface area.

| Experiment | Epochs | Minibatch | Initial learning rate | Learning-rate schedule |
|---|---:|---:|---:|---|
| Uniform sphere | 20,000 | 50 | 0.3 | exponential; final factor 0.1 |
| Two spheres | 10,000 | 50 | 0.2 | exponential; final factor 0.1 |
| Non-uniform sphere | 10,000 | 50 | 0.2 | exponential; final factor 0.1 |
| Conformal maps | 100,000 | 100 | 0.1 | exponential; gamma 0.999995 |
| Wisconsin Breast Cancer | 63,878 | 100 | 0.1 | exponential; gamma 0.999994 |

The author training workflow uses Adam with `betas=(0.99, 0.99)`, zero weight
decay, and AMSGrad. Figures in this repository are reproducible from the
supplied frozen operators. Exact end-to-end retraining additionally depends on
the private HSM implementation, numerical libraries, and hardware and is not
part of this public release.

## Two-sphere seed and w stability study

The robustness study contains 34 unique 10,000-epoch training runs. Its main
grid fixes the two-sphere dataset at seed 42 and crosses training seeds
`40, 41, 42, 43, 44` with `w = 0, 0.01, 0.03, 0.1, 0.3, 1`. A companion study
fixes training seed 42 and `w=0.1`, then varies the dataset seed over the same
five values. This separates sensitivity to model training from sensitivity to
the sampled noisy dataset.

Each public stability archive contains only compact trained operators, its
learned QCML cloud, Laplacian spectrum, metrics, and complete JSON metadata.
Regenerate the public comparisons with:

```bash
python scripts/make_figures/two_spheres_stability.py
```

The resulting figures show that the learned geometry is consistent across the
five seeds at fixed `w`. Increasing `w` reduces learned quantum variance, while
large `w` increasingly contracts the learned geometry away from the reference
spheres; `w=1` strongly distorts the smaller component. The paper value
`w=0.1` lies in the stable intermediate regime.

## Regenerate the synthetic inputs

```bash
python scripts/generate_data/generate_spheres.py --experiment uniform \
  --seed 42 --output data/input/spheres/sphere_uniform_regenerated.npy
python scripts/generate_data/generate_spheres.py --experiment weighted \
  --seed 42 --output data/input/spheres/sphere_nonuniform_regenerated.npy
python scripts/generate_data/generate_spheres.py --experiment two \
  --seed 42 --output data/input/spheres/two_spheres_regenerated.npy
python scripts/generate_data/generate_conformal_maps.py \
  --seed 42 --maps 2000 --points-per-map 100 --a-max 0.9 --z-max 1 \
  --noise 0 \
  --output data/input/conformal_maps/conformal_maps_regenerated.npy
```


See the README files within `data/` and `results/` for format details.
