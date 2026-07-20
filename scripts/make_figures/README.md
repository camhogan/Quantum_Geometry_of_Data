# Figure-generation entry points

- `two_spheres.py`: analyses underlying Figures 5 and 6.
- `nonuniform_sphere.py`: analyses underlying Figures 7 and 8.
- `conformal_maps.py`: analyses underlying Figures 9 through 13.
- `breast_cancer.py`: analyses underlying Figure 14.
- `quantum_geometry.py`: shared NumPy post-training calculations.

These scripts load frozen Hermitian matrices from `results/trained_operators/`.
They neither contain nor invoke HSM training code.

For Figures 5--8, the scripts use the monopole coordinates and topological
charges produced by the author notebook's grid search. Re-running that costly
search is not required to regenerate the paper panels.
