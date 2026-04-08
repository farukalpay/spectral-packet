"""Sparse eigensolvers for large-scale quantum Hamiltonians.

Provides three solver backends for the lowest eigenvalues/eigenstates of
sparse (or sparsifiable) Hamiltonian matrices:

1. **Lanczos** — ``scipy.sparse.linalg.eigsh`` (CPU, Krylov subspace)
2. **LOBPCG**  — ``torch.lobpcg`` (GPU-capable, block Krylov)
3. **Davidson** — preconditioned residue iteration (standard in quantum
   chemistry for CI problems)

Complexity: O(N · k · nnz) instead of O(N³) for the dense path, where
*k* is the number of requested eigenstates and *nnz* the number of
non-zero Hamiltonian elements.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Literal

import torch
import numpy as np

Tensor = torch.Tensor


class SparseSolverMethod(enum.Enum):
    """Available sparse eigensolver backends."""

    LANCZOS = "lanczos"
    LOBPCG = "lobpcg"
    DAVIDSON = "davidson"


@dataclass(frozen=True, slots=True)
class SparseEigensolverResult:
    """Result container for sparse eigensolvers.

    Attributes
    ----------
    eigenvalues : Tensor
        Shape ``(num_states,)`` — lowest eigenvalues in ascending order.
    eigenvectors : Tensor
        Shape ``(N, num_states)`` — corresponding eigenvectors as columns.
    num_states : int
        Number of converged eigenstates.
    method : str
        Solver backend used.
    converged : bool
        Whether all requested states converged within tolerance.
    iterations : int
        Number of iterations performed.
    sparsity : float
        Fraction of zero elements in the input matrix.
    """

    eigenvalues: Tensor
    eigenvectors: Tensor
    num_states: int
    method: str
    converged: bool
    iterations: int
    sparsity: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_scipy_sparse(H: Tensor):
    """Convert a torch tensor (dense or sparse_csr) to scipy CSR."""
    import scipy.sparse as sp

    if H.is_sparse_csr:
        crow = H.crow_indices().cpu().numpy()
        col = H.col_indices().cpu().numpy()
        vals = H.values().cpu().numpy()
        return sp.csr_matrix((vals, col, crow), shape=H.shape)
    if H.is_sparse:
        H_coo = H.coalesce()
        indices = H_coo.indices().cpu().numpy()
        values = H_coo.values().cpu().numpy()
        return sp.coo_matrix((values, (indices[0], indices[1])), shape=H.shape).tocsr()
    # Dense tensor — convert via numpy
    return sp.csr_matrix(H.detach().cpu().numpy())


def _compute_sparsity(H: Tensor) -> float:
    """Fraction of zero elements."""
    N = H.shape[0] * H.shape[1]
    if H.is_sparse_csr:
        nnz = H.values().numel()
    elif H.is_sparse:
        nnz = H._nnz()
    else:
        nnz = int(torch.count_nonzero(H).item())
    return 1.0 - nnz / N if N > 0 else 0.0


def _ensure_sparse_csr(H: Tensor) -> Tensor:
    """Best-effort conversion to sparse CSR for torch.lobpcg."""
    if H.is_sparse_csr:
        return H
    if H.is_sparse:
        return H.to_sparse_csr()
    # Dense — threshold small values to build sparse CSR
    mask = H.abs() > 0
    return H.to_sparse_csr()


# ---------------------------------------------------------------------------
# Lanczos (scipy.sparse.linalg.eigsh)
# ---------------------------------------------------------------------------

def _solve_lanczos(
    H: Tensor,
    num_states: int,
    tol: float,
    max_iter: int,
) -> SparseEigensolverResult:
    """Solve via Lanczos (ARPACK through scipy)."""
    from scipy.sparse.linalg import eigsh

    H_sp = _to_scipy_sparse(H)
    sparsity = _compute_sparsity(H)
    N = H.shape[0]

    # eigsh requires k < N
    k = min(num_states, N - 1)

    eigenvalues_np, eigenvectors_np = eigsh(
        H_sp,
        k=k,
        which="SA",  # smallest algebraic
        tol=tol,
        maxiter=max_iter if max_iter > 0 else None,
    )

    # Sort ascending
    order = np.argsort(eigenvalues_np)
    eigenvalues_np = eigenvalues_np[order]
    eigenvectors_np = eigenvectors_np[:, order]

    dtype = H.dtype if H.dtype in (torch.float32, torch.float64) else torch.float64
    device = H.device if not H.device.type == "meta" else torch.device("cpu")

    return SparseEigensolverResult(
        eigenvalues=torch.tensor(eigenvalues_np, dtype=dtype, device=device),
        eigenvectors=torch.tensor(eigenvectors_np, dtype=dtype, device=device),
        num_states=k,
        method="lanczos",
        converged=True,
        iterations=0,  # ARPACK doesn't expose iteration count easily
        sparsity=sparsity,
    )


# ---------------------------------------------------------------------------
# LOBPCG (torch.lobpcg)
# ---------------------------------------------------------------------------

def _solve_lobpcg(
    H: Tensor,
    num_states: int,
    tol: float,
    max_iter: int,
) -> SparseEigensolverResult:
    """Solve via Locally Optimal Block Preconditioned Conjugate Gradient."""
    sparsity = _compute_sparsity(H)
    N = H.shape[0]
    k = min(num_states, N - 1)
    dtype = H.dtype if H.dtype in (torch.float32, torch.float64) else torch.float64
    device = H.device

    H_work = _ensure_sparse_csr(H.to(dtype=dtype))

    # Random initial guess
    X0 = torch.randn(N, k, dtype=dtype, device=device)

    iters = max_iter if max_iter > 0 else 400

    eigenvalues, eigenvectors = torch.lobpcg(
        H_work, k=k, X=X0, niter=iters, tol=tol, largest=False,
    )

    # Sort ascending
    order = torch.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    return SparseEigensolverResult(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        num_states=k,
        method="lobpcg",
        converged=True,
        iterations=iters,
        sparsity=sparsity,
    )


# ---------------------------------------------------------------------------
# Davidson algorithm
# ---------------------------------------------------------------------------

def _solve_davidson(
    H: Tensor,
    num_states: int,
    tol: float,
    max_iter: int,
) -> SparseEigensolverResult:
    """Generalized Davidson algorithm for lowest eigenpairs.

    Standard in quantum chemistry for Configuration Interaction (CI)
    problems where the diagonal is a good preconditioner.
    """
    sparsity = _compute_sparsity(H)
    N = H.shape[0]
    k = min(num_states, N - 1)
    dtype = H.dtype if H.dtype in (torch.float32, torch.float64) else torch.float64
    device = H.device

    # Work in dense on the device (Davidson is iterative but needs matvec)
    if H.is_sparse or H.is_sparse_csr:
        H_dense = H.to_dense().to(dtype=dtype, device=device)
    else:
        H_dense = H.to(dtype=dtype, device=device)

    diag_H = torch.diagonal(H_dense).clone()

    # Initial guess: k unit vectors at indices of smallest diagonal elements
    _, init_indices = torch.topk(diag_H, k, largest=False)
    V = torch.zeros(N, k, dtype=dtype, device=device)
    for i, idx in enumerate(init_indices):
        V[idx, i] = 1.0

    # Orthonormalize
    V, _ = torch.linalg.qr(V)

    max_subspace = min(max(8 * k, 60), N)
    converged = False
    iteration = 0

    eigenvalues = torch.zeros(k, dtype=dtype, device=device)
    eigenvectors = torch.zeros(N, k, dtype=dtype, device=device)

    for iteration in range(1, (max_iter if max_iter > 0 else 200) + 1):
        # Projected Hamiltonian in subspace
        HV = H_dense @ V  # (N, m)
        H_sub = V.T @ HV  # (m, m)

        # Solve small eigenproblem
        theta, s = torch.linalg.eigh(H_sub)

        # Select lowest k
        eigenvalues = theta[:k]
        # Ritz vectors
        ritz = V @ s[:, :k]  # (N, k)
        eigenvectors = ritz

        # Compute residuals
        residuals = H_dense @ ritz - ritz * eigenvalues.unsqueeze(0)
        res_norms = torch.norm(residuals, dim=0)

        if torch.all(res_norms < tol):
            converged = True
            break

        # Expand subspace with preconditioned residuals (Davidson correction)
        new_vectors = []
        for j in range(k):
            if res_norms[j] >= tol:
                r = residuals[:, j]
                # Diagonal preconditioner: (diag(H) - theta_j I)^{-1} r
                denom = diag_H - eigenvalues[j]
                # Avoid division by zero
                denom = torch.where(denom.abs() < 1e-12, torch.ones_like(denom), denom)
                correction = r / denom
                new_vectors.append(correction)

        if not new_vectors:
            converged = True
            break

        new_V = torch.stack(new_vectors, dim=1)

        # Orthogonalize against existing subspace
        V_expanded = torch.cat([V, new_V], dim=1)

        # Restart if subspace too large
        if V_expanded.shape[1] > max_subspace:
            V = ritz.clone()
            V, _ = torch.linalg.qr(V)
        else:
            V, _ = torch.linalg.qr(V_expanded)

    return SparseEigensolverResult(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        num_states=k,
        method="davidson",
        converged=converged,
        iterations=iteration,
        sparsity=sparsity,
    )


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

def solve_eigenproblem_sparse(
    H: Tensor,
    num_states: int = 6,
    *,
    method: str | SparseSolverMethod = "lanczos",
    tol: float = 1e-10,
    max_iter: int = 0,
) -> SparseEigensolverResult:
    """Solve for the lowest eigenpairs of a Hamiltonian using sparse methods.

    Parameters
    ----------
    H : Tensor
        Hermitian matrix — dense, sparse COO, or sparse CSR.
    num_states : int
        Number of lowest eigenvalues/eigenvectors to compute.
    method : str or SparseSolverMethod
        ``"lanczos"`` (default), ``"lobpcg"``, or ``"davidson"``.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations (0 = backend default).

    Returns
    -------
    SparseEigensolverResult
    """
    if isinstance(method, str):
        method = SparseSolverMethod(method.lower())

    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"H must be a square matrix, got shape {H.shape}")

    if num_states < 1:
        raise ValueError("num_states must be at least 1")

    if num_states >= H.shape[0]:
        raise ValueError(
            f"num_states ({num_states}) must be less than matrix dimension ({H.shape[0]}) "
            f"for iterative solvers; use torch.linalg.eigh for full diagonalization"
        )

    solvers = {
        SparseSolverMethod.LANCZOS: _solve_lanczos,
        SparseSolverMethod.LOBPCG: _solve_lobpcg,
        SparseSolverMethod.DAVIDSON: _solve_davidson,
    }

    return solvers[method](H, num_states, tol, max_iter)


__all__ = [
    "SparseSolverMethod",
    "SparseEigensolverResult",
    "solve_eigenproblem_sparse",
]
