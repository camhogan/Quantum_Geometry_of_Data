"""Post-training quantum-geometry calculations using only NumPy."""

from __future__ import annotations

import numpy as np


def validate_operators(operators: np.ndarray, atol: float = 1e-5) -> np.ndarray:
    operators = np.asarray(operators)
    if operators.ndim != 3 or operators.shape[1] != operators.shape[2]:
        raise ValueError("operators must have shape (D, N, N)")
    if not np.allclose(operators, operators.conj().transpose(0, 2, 1), atol=atol):
        raise ValueError("all operators must be Hermitian")
    return operators


def hamiltonians(points: np.ndarray, operators: np.ndarray) -> np.ndarray:
    """Return H(x)=1/2 sum_a (X_a-x_a I)^2 for every input point."""
    operators = validate_operators(operators)
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != operators.shape[0]:
        raise ValueError("points must have shape (samples, D)")
    n = operators.shape[1]
    a2 = np.einsum("aij,ajk->ik", operators, operators)
    xa = np.einsum("sa,aij->sij", points, operators)
    x2 = np.einsum("sa,sa->s", points, points)[:, None, None] * np.eye(n)
    return 0.5 * (a2[None] - 2.0 * xa + x2)


def coherent_states(points: np.ndarray, operators: np.ndarray):
    values, vectors = np.linalg.eigh(hamiltonians(points, operators))
    return values[:, 0], vectors[:, :, 0]


def expectation_values(states: np.ndarray, operators: np.ndarray) -> np.ndarray:
    return np.einsum("si,aij,sj->sa", states.conj(), operators, states).real


def uncertainty(states: np.ndarray, operators: np.ndarray) -> np.ndarray:
    means = expectation_values(states, operators)
    squared = np.einsum("aij,ajk->aik", operators, operators)
    second = np.einsum("si,aij,sj->sa", states.conj(), squared, states).real
    return np.sum(second - means**2, axis=1)


def quantum_metric(points: np.ndarray, operators: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(hamiltonians(points, operators))
    ground = vectors[:, :, 0]
    excited = vectors[:, :, 1:]
    elements = np.einsum("si,aij,sjn->san", ground.conj(), operators, excited)
    gaps = values[:, 1:] - values[:, :1]
    return (2.0 * np.einsum(
        "san,sbn,sn->sab", elements, elements.conj(), 1.0 / gaps
    )).real


def berry_curvature(points: np.ndarray, operators: np.ndarray) -> np.ndarray:
    """Return the antisymmetric Berry-curvature tensor at each point."""
    values, vectors = np.linalg.eigh(hamiltonians(points, operators))
    ground = vectors[:, :, 0]
    excited = vectors[:, :, 1:]
    elements = np.einsum("si,aij,sjn->san", ground.conj(), operators, excited)
    gaps2 = (values[:, 1:] - values[:, :1])**2
    tensor = -2.0*np.imag(np.einsum(
        "san,sbn,sn->sab", elements, elements.conj(), 1.0/gaps2
    ))
    return 0.5*(tensor-tensor.transpose(0, 2, 1))


def matrix_laplacian(operators: np.ndarray) -> np.ndarray:
    """Return Delta(A)=sum_a [X_a,[X_a,A]] in vectorized form.

    This is the unnormalized convention used for the spectra in the paper.
    """
    operators = validate_operators(operators)
    d, n, _ = operators.shape
    sum_x2 = np.einsum("aik,akj->ij", operators, operators)
    eye = np.eye(n, dtype=operators.dtype)
    lap = np.kron(sum_x2, eye) + np.kron(eye, sum_x2.T)
    lap -= 2.0 * np.einsum("aik,alj->ijkl", operators, operators).reshape(n*n, n*n)
    return lap
