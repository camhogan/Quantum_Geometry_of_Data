# Public experiment results

`trained_operators/` contains portable NumPy archives with:

- `operators`: complex Hermitian matrices of shape `(D, N, N)`.
- `metadata`: a JSON string describing the source experiment.

`derived_quantities/` contains selected class-independent arrays saved from the
same experiment. These are provided for direct plotting and verification, but
the public analysis code can recompute its core geometry from `operators`.

For WBC, `wbc_all_weights.npz` contains the response spectra for
`w = 0.02, 0.05, 0.1, 0.2, 0.5, 0.8`, together with the `w=0.2` and `w=0.5`
Laplacian spectra, eigenmap coordinates, and feature overlaps. The same mean
response spectra and dimensions are provided in CSV form. The classifier JSON
records every fold as well as aggregate ROC AUC, accuracy, sensitivity, and
specificity for the operator-dependent rows of the manuscript table.

PyTorch state dictionaries are intentionally excluded because they depend on
the unpublished model implementation. Full training checkpoints are also
excluded because they contain redundant arrays and, in one case, exceed 600 MB.

`stability/` contains compact operator ensembles and summary data for robustness
studies. See its README for the exact experimental grid.
