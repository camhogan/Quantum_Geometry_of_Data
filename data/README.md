# Input data

`input/` contains observations supplied to the unpublished training process.
Synthetic datasets are stored as NumPy arrays. The breast-cancer directory
contains the original UCI data and its accompanying description.

For exact reproducibility, the public generators follow the archived author
code. The non-uniform sphere uses polar-angle density
`(1 + cos(theta))^3 sin(theta)`. The conformal-map data use
`f(z) = (z - a)/(1 - conj(a) z)` with zero global rotation.

The uniform, non-uniform, and two-sphere archives were generated with NumPy
seed 42. The archived conformal-map realization was generated without a
recorded seed and is supplied directly. Its public generator defaults to seed
42 so newly generated conformal datasets are reproducible, although they are
not identical to the archived realization.
