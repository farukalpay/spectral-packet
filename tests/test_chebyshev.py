"""Tests for the Chebyshev spectral basis."""

import torch
import pytest

from spectral_packet_engine.chebyshev import (
    ChebyshevBasis,
    chebyshev_matrix,
    chebyshev_nodes,
    chebyshev_quadrature_weights,
)
from spectral_packet_engine.domain import InfiniteWell1D


@pytest.fixture
def domain():
    return InfiniteWell1D(left=torch.tensor(0.0, dtype=torch.float64), right=torch.tensor(1.0, dtype=torch.float64))


@pytest.fixture
def wide_domain():
    return InfiniteWell1D(left=torch.tensor(-2.0, dtype=torch.float64), right=torch.tensor(3.0, dtype=torch.float64))


class TestChebyshevNodes:
    def test_node_count(self, domain):
        nodes = chebyshev_nodes(domain, 10)
        assert nodes.shape == (10,)

    def test_nodes_within_domain(self, domain):
        nodes = chebyshev_nodes(domain, 50)
        assert torch.all(nodes >= domain.left - 1e-12).item()
        assert torch.all(nodes <= domain.right + 1e-12).item()

    def test_endpoints_included(self, domain):
        nodes = chebyshev_nodes(domain, 20)
        # First node should be near right, last near left (cosine ordering)
        assert torch.isclose(nodes[0], domain.right, atol=1e-12).item()
        assert torch.isclose(nodes[-1], domain.left, atol=1e-12).item()

    def test_wide_domain(self, wide_domain):
        nodes = chebyshev_nodes(wide_domain, 15)
        assert torch.all(nodes >= wide_domain.left - 1e-12).item()
        assert torch.all(nodes <= wide_domain.right + 1e-12).item()


class TestChebyshevMatrix:
    def test_shape(self, domain):
        grid = domain.grid(100)
        modes = torch.arange(5)
        T = chebyshev_matrix(domain, modes, grid)
        assert T.shape == (100, 5)

    def test_T0_is_one(self, domain):
        grid = domain.grid(50)
        T = chebyshev_matrix(domain, torch.tensor([0]), grid)
        assert torch.allclose(T[:, 0], torch.ones(50, dtype=torch.float64), atol=1e-12)

    def test_T1_is_linear(self, domain):
        grid = domain.grid(50)
        T = chebyshev_matrix(domain, torch.arange(2), grid)
        xi = 2 * (grid - domain.left) / domain.length - 1
        assert torch.allclose(T[:, 1], xi, atol=1e-12)

    def test_recurrence_T2(self, domain):
        """T_2(x) = 2x^2 - 1"""
        grid = domain.grid(50)
        T = chebyshev_matrix(domain, torch.arange(3), grid)
        xi = 2 * (grid - domain.left) / domain.length - 1
        expected = 2 * xi ** 2 - 1
        assert torch.allclose(T[:, 2], expected, atol=1e-10)


class TestChebyshevBasis:
    def test_creation(self, domain):
        basis = ChebyshevBasis(domain, num_modes=10)
        assert basis.num_modes == 10

    def test_invalid_modes(self, domain):
        with pytest.raises(ValueError, match="positive"):
            ChebyshevBasis(domain, num_modes=0)

    def test_evaluate(self, domain):
        basis = ChebyshevBasis(domain, num_modes=8)
        grid = domain.grid(100)
        T = basis.evaluate(grid)
        assert T.shape == (100, 8)

    def test_project_reconstruct_polynomial(self, domain):
        """A polynomial of degree < N should be exactly representable."""
        basis = ChebyshevBasis(domain, num_modes=10)
        nodes = basis.nodes(64)
        # f(x) = 3x^2 - 2x + 1, degree 2 — should be exact with 3+ modes
        f_values = 3 * nodes ** 2 - 2 * nodes + 1
        coeffs = basis.project(f_values, nodes)
        recon = basis.reconstruct(coeffs, nodes)
        assert torch.allclose(recon, f_values, atol=1e-6)

    def test_project_reconstruct_smooth(self, domain):
        """Smooth function should converge exponentially."""
        basis = ChebyshevBasis(domain, num_modes=20)
        nodes = basis.nodes(64)
        f_values = torch.sin(torch.pi * nodes)
        coeffs = basis.project(f_values, nodes)
        recon = basis.reconstruct(coeffs, nodes)
        error = torch.max(torch.abs(recon - f_values))
        assert error < 1e-4  # Should be very good with 20 modes

    def test_batch_project(self, domain):
        """Project a batch of profiles."""
        basis = ChebyshevBasis(domain, num_modes=10)
        nodes = basis.nodes(32)
        batch = torch.stack([
            torch.sin(torch.pi * nodes),
            torch.cos(torch.pi * nodes),
            nodes ** 2,
        ])
        coeffs = basis.project(batch, nodes)
        assert coeffs.shape == (3, 10)


class TestChebyshevDifferentiation:
    def test_derivative_of_linear(self, domain):
        """d/dx[x] = 1"""
        basis = ChebyshevBasis(domain, num_modes=5)
        nodes = basis.nodes(32)
        f_values = nodes.clone()
        coeffs = basis.project(f_values, nodes)
        d_coeffs = basis.differentiate(coeffs)
        deriv = basis.reconstruct(d_coeffs, nodes)
        expected = torch.ones_like(nodes)
        assert torch.allclose(deriv, expected, atol=1e-4)

    def test_derivative_of_quadratic(self, domain):
        """d/dx[x^2] = 2x"""
        basis = ChebyshevBasis(domain, num_modes=8)
        nodes = basis.nodes(32)
        f_values = nodes ** 2
        coeffs = basis.project(f_values, nodes)
        d_coeffs = basis.differentiate(coeffs)
        deriv = basis.reconstruct(d_coeffs, nodes)
        expected = 2 * nodes
        assert torch.allclose(deriv, expected, atol=1e-3)

    def test_second_derivative_of_sine(self, domain):
        """d^2/dx^2[sin(pi*x)] = -pi^2 * sin(pi*x)"""
        basis = ChebyshevBasis(domain, num_modes=20)
        nodes = basis.nodes(64)
        f_values = torch.sin(torch.pi * nodes)
        coeffs = basis.project(f_values, nodes)
        d2_coeffs = basis.differentiate(coeffs, order=2)
        d2 = basis.reconstruct(d2_coeffs, nodes)
        expected = -(torch.pi ** 2) * torch.sin(torch.pi * nodes)
        # Interior points (away from boundaries)
        interior = (nodes > 0.1) & (nodes < 0.9)
        error = torch.max(torch.abs(d2[interior] - expected[interior]))
        assert error < 0.5  # Spectral derivative should be reasonable

    def test_differentiation_matrix_shape(self, domain):
        basis = ChebyshevBasis(domain, num_modes=10)
        D = basis.differentiation_matrix()
        assert D.shape == (10, 10)

    def test_zero_order_is_identity(self, domain):
        basis = ChebyshevBasis(domain, num_modes=5)
        coeffs = torch.randn(5, dtype=torch.float64)
        result = basis.differentiate(coeffs, order=0)
        assert torch.allclose(result, coeffs)


class TestQuadratureWeights:
    def test_integrates_constant(self):
        """Integral of 1 over [-1,1] = 2."""
        w = chebyshev_quadrature_weights(20, dtype=torch.float64, device=None)
        assert torch.isclose(torch.sum(w), torch.tensor(2.0, dtype=torch.float64), atol=1e-10).item()

    def test_integrates_T1(self):
        """Integral of x over [-1,1] = 0."""
        N = 20
        w = chebyshev_quadrature_weights(N, dtype=torch.float64, device=None)
        nodes = torch.cos(torch.pi * torch.arange(N, dtype=torch.float64) / (N - 1))
        result = torch.sum(w * nodes)
        assert torch.isclose(result, torch.tensor(0.0, dtype=torch.float64), atol=1e-10).item()
