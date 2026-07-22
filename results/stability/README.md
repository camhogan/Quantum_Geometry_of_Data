# Stability studies

`two_spheres_seed_w/` contains 34 compact post-training archives for the
two-sphere robustness study. Each archive provides the trained Hermitian
operators, learned QCML cloud, matrix-Laplacian spectrum, and JSON metadata.
No private HSM implementation or training checkpoint is included.

The main factorial study fixes the dataset at seed 42 and varies training seed
over `40, 41, 42, 43, 44` and `w` over `0, 0.01, 0.03, 0.1, 0.3, 1`. A second
study fixes training seed 42 and `w=0.1`, then varies the dataset seed over the
same five values. The shared seed-42 run appears only once, giving 34 unique
runs.

Run `python scripts/make_figures/two_spheres_stability.py` to regenerate the
comparison grids, metric summary, and `summary.csv`.
