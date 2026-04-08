"""Tests for the sparse eigensolver module."""
from __future__ import annotations

import pytest
import torch
import numpy as np

from spectral_packet_engine.sparse_eigensolver import (
    SparseSolverMethod,
    SparseEigensolverResult,
    solve_eigenproblem_sparse,
)


def _harmonic_hamiltonian(N: int = 64, omega: float = 50.0) -> torch.Tensor:
    """Build a dense Hamiltonian for the harmonic oscillator on [0,1]."""
    from spectral_packet_engine.domain import InfiniteWell1D
    from spectral_packet_engine.basis import eigenenergies, sine_basis_matrix
    from spectral_packet_engine.eigensolver import harmonic_potential
    from spectral_packet_engine.domain import coerce_tensor

    dtype = torch.float64
    domain = InfiniteWell1D(
        left=torch.tensor(0.0, dtype=dtype),
        right=torch.tensor(1.0, dtype=dtype),
    )
    modes = torch.arange(1, N + 1, dtype=dtype)
    E_n = eigenenergies(domain, modes)
    num_quad = max(4 * N, 512)
    quad_grid = domain.grid(num_quad)
    dx = quad_grid[1] - quad_grid[0]
    V_vals = coerce_tensor(
        harmonic_potential(quad_grid, omega=omega, domain=domain),
        dtype=dtype,
    )
    B = sine_basis_matrix(domain, modes, quad_grid)
    weights = torch.ones(num_quad, dtype=dtype) * dx
    weights[0] /= 2.0
    weights[-1] /= 2.0
    V_mat = B.T @ ((V_vals * weights).unsqueeze(1) * B)
    return torch.diag(E_n) + V_mat


@pytest.fixture
def harmonic_H():
    return _harmonic_hamiltonian(64)


@pytest.fixture
def dense_reference(harmonic_H):
    """Dense eigensolution as ground truth."""
    eigenvalues, eigenvectors = torch.linalg.eigh(harmonic_H)
    return eigenvalues[:6], eigenvectors[:, :6]


class TestLanczos:
    def test_eigenvalues_match_dense(self, harmonic_H, dense_reference):
        result = solve_eigenproblem_sparse(harmonic_H, num_states=6, method="lanczos")
        dense_evals = dense_reference[0]
        assert torch.allclose(result.eigenvalues, dense_evals, atol=1e-8)

    def test_method_label(self, harmonic_H):
        result = solve_eigenproblem_sparse(harmonic_H, num_states=4, method="lanczos")
        assert result.method == "lanczos"
        assert result.num_states == 4

    def test_eigenvectors_orthonormal(self, harmonic_H):
        result = solve_eigenproblem_sparse(harmonic_H, num_states=4, method="lanczos")
        V = result.eigenvectors
        overlap = V.T @ V
        assert torch.allclose(overlap, torch.eye(4, dtype=torch.float64), atol=1e-8)


class TestLOBPCG:
    def test_eigenvalues_match_dense(self, harmonic_H, dense_reference):
        result = solve_eigenproblem_sparse(harmonic_H, num_states=6, method="lobpcg")
        dense_evals = dense_reference[0]
        # LOBPCG is less precise, allow wider tolerance
        assert torch.allclose(result.eigenvalues, dense_evals, atol=1e-4)

    def test_method_label(self, harmonic_H):
        result = solve_eigenproblem_sparse(harmonic_H, num_states=4, method="lobpcg")
        assert result.method == "lobpcg"


class TestDavidson:
    def test_eigenvalues_match_dense(self, harmonic_H, dense_reference):
        result = solve_eigenproblem_sparse(harmonic_H, num_states=6, method="davidson")
        dense_evals = dense_reference[0]
        assert torch.allclose(result.eigenvalues, dense_evals, atol=1e-6)

    def test_convergence(self, harmonic_H):
        result = solve_eigenproblem_sparse(harmonic_H, num_states=4, method="davidson", tol=1e-10)
        assert result.converged

    def test_method_label(self, harmonic_H):
        result = solve_eigenproblem_sparse(harmonic_H, num_states=4, method="davidson")
        assert result.method == "davidson"
        assert result.iterations > 0


class TestSparseInput:
    def test_sparse_csr_input(self, harmonic_H, dense_reference):
        H_sparse = harmonic_H.to_sparse_csr()
        result = solve_eigenproblem_sparse(H_sparse, num_states=6, method="lanczos")
        dense_evals = dense_reference[0]
        assert torch.allclose(result.eigenvalues, dense_evals, atol=1e-8)

    def test_sparse_coo_input(self, harmonic_H, dense_reference):
        H_sparse = harmonic_H.to_sparse()
        result = solve_eigenproblem_sparse(H_sparse, num_states=6, method="lanczos")
        dense_evals = dense_reference[0]
        assert torch.allclose(result.eigenvalues, dense_evals, atol=1e-8)

    def test_sparsity_computed(self, harmonic_H):
        result = solve_eigenproblem_sparse(harmonic_H, num_states=4, method="lanczos")
        assert 0.0 <= result.sparsity <= 1.0


class TestValidation:
    def test_non_square_raises(self):
        H = torch.randn(10, 5)
        with pytest.raises(ValueError, match="square"):
            solve_eigenproblem_sparse(H, num_states=3)

    def test_too_many_states_raises(self):
        H = torch.eye(5, dtype=torch.float64)
        with pytest.raises(ValueError, match="num_states"):
            solve_eigenproblem_sparse(H, num_states=5)

    def test_zero_states_raises(self):
        H = torch.eye(5, dtype=torch.float64)
        with pytest.raises(ValueError, match="num_states"):
            solve_eigenproblem_sparse(H, num_states=0)


class TestAllMethodsAgree:
    def test_three_methods_agree(self, harmonic_H):
        """All three solvers should produce the same lowest eigenvalues."""
        r_lanczos = solve_eigenproblem_sparse(harmonic_H, num_states=4, method="lanczos")
        r_davidson = solve_eigenproblem_sparse(harmonic_H, num_states=4, method="davidson")
        r_lobpcg = solve_eigenproblem_sparse(harmonic_H, num_states=4, method="lobpcg")

        # Lanczos and Davidson should be very close
        assert torch.allclose(r_lanczos.eigenvalues, r_davidson.eigenvalues, atol=1e-6)
        # LOBPCG has wider tolerance
        assert torch.allclose(r_lanczos.eigenvalues, r_lobpcg.eigenvalues, atol=1e-3)
