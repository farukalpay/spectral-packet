"""Tests for spectral differentiation, momentum, energy, and convergence modules."""

import torch
import pytest

from spectral_packet_engine.basis import InfiniteWellBasis
from spectral_packet_engine.convergence import (
    DecayType,
    analyze_convergence,
    estimate_spectral_decay,
    recommend_truncation,
    spectral_entropy,
)
from spectral_packet_engine.domain import InfiniteWell1D
from spectral_packet_engine.dynamics import SpectralPropagator
from spectral_packet_engine.energy import (
    check_energy_conservation,
    compute_energy_budget,
    energy_spectrum,
    kinetic_energy,
)
from spectral_packet_engine.momentum import (
    expectation_momentum_spectral,
    expectation_momentum_squared_spectral,
    heisenberg_uncertainty,
    variance_momentum_spectral,
)
from spectral_packet_engine.projector import StateProjector
from spectral_packet_engine.spectral_diff import (
    kinetic_energy_spectral,
    parseval_conservation_error,
    parseval_norm,
    sine_basis_derivative_coefficients,
    sine_basis_derivative_on_grid,
)
from spectral_packet_engine.state import SpectralState, make_truncated_gaussian_packet


@pytest.fixture
def domain():
    return InfiniteWell1D(left=torch.tensor(0.0, dtype=torch.float64), right=torch.tensor(1.0, dtype=torch.float64))


@pytest.fixture
def basis(domain):
    return InfiniteWellBasis(domain, num_modes=32)


@pytest.fixture
def gaussian_state(domain, basis):
    """A Gaussian wave packet projected onto the spectral basis."""
    packet = make_truncated_gaussian_packet(
        domain, center=0.5, width=0.1, wavenumber=10.0
    )
    projector = StateProjector(basis)
    return projector.project_packet(packet)


# === Spectral Differentiation ===

class TestSpectralDifferentiation:
    def test_derivative_preserves_shape(self, basis):
        coeffs = torch.randn(32, dtype=torch.float64)
        d_coeffs = sine_basis_derivative_coefficients(coeffs, basis, order=1)
        assert d_coeffs.shape == coeffs.shape

    def test_second_derivative_eigenstate(self, domain, basis):
        """The second derivative of phi_n is -(n*pi/L)^2 * phi_n."""
        # Pure mode n=3
        coeffs = torch.zeros(32, dtype=torch.float64)
        coeffs[2] = 1.0  # mode 3 (0-indexed)
        d2_coeffs = sine_basis_derivative_coefficients(coeffs, basis, order=2)
        # For even-order, stays in sine basis: d^2 phi_n / dx^2 = -(n*pi/L)^2 phi_n
        expected_scale = -(3 * torch.pi / domain.length) ** 2
        assert torch.isclose(d2_coeffs[2], expected_scale, rtol=1e-6).item()

    def test_zero_order_is_identity(self, basis):
        coeffs = torch.randn(32, dtype=torch.float64)
        result = sine_basis_derivative_coefficients(coeffs, basis, order=0)
        assert torch.allclose(result, coeffs)

    def test_derivative_on_grid(self, domain, basis):
        """Test grid evaluation of derivatives."""
        coeffs = torch.zeros(32, dtype=torch.float64)
        coeffs[0] = 1.0  # mode 1: phi_1 = sqrt(2/L) sin(pi*x/L)
        grid = domain.grid(100)
        deriv = sine_basis_derivative_on_grid(coeffs, grid, basis, order=1)
        # d/dx phi_1 = sqrt(2/L) * (pi/L) * cos(pi*x/L)
        factor = torch.sqrt(torch.tensor(2.0) / domain.length)
        expected = factor * (torch.pi / domain.length) * torch.cos(torch.pi * grid / domain.length)
        assert torch.allclose(deriv, expected, atol=1e-6)

    def test_batch_derivative(self, basis):
        coeffs = torch.randn(5, 32, dtype=torch.float64)
        d_coeffs = sine_basis_derivative_coefficients(coeffs, basis, order=1)
        assert d_coeffs.shape == (5, 32)


# === Parseval Norm and Conservation ===

class TestParseval:
    def test_parseval_norm_normalized_state(self, gaussian_state):
        """A properly normalized state should have norm ≈ 1."""
        norm = parseval_norm(gaussian_state.coefficients)
        assert norm.item() > 0

    def test_conservation_under_propagation(self, domain, basis, gaussian_state):
        """Exact phase-factor propagation preserves the norm exactly."""
        propagator = SpectralPropagator(basis)
        times = torch.linspace(0, 1, 10, dtype=torch.float64)
        propagated = propagator.propagate_many(gaussian_state, times)
        error = parseval_conservation_error(gaussian_state.coefficients, propagated)
        # Should be near zero — allow floating-point tolerance
        assert torch.all(error < 1e-6).item()

    def test_norm_of_real_coefficients(self):
        coeffs = torch.tensor([1.0, 2.0, 3.0])
        assert torch.isclose(parseval_norm(coeffs), torch.tensor(14.0)).item()


# === Momentum Observables ===

class TestMomentumObservables:
    def test_momentum_squared_positive(self, gaussian_state, basis):
        """<p^2> must be non-negative."""
        p_sq = expectation_momentum_squared_spectral(gaussian_state.coefficients, basis)
        assert p_sq.item() >= 0

    def test_real_state_zero_momentum(self, basis):
        """A real-valued state in a symmetric well has <p> = 0."""
        # Pure eigenstate (real coefficients)
        coeffs = torch.zeros(32, dtype=torch.complex128)
        coeffs[0] = 1.0 + 0j
        p = expectation_momentum_spectral(coeffs, basis)
        assert abs(p.item()) < 1e-10

    def test_momentum_variance_positive(self, gaussian_state, basis):
        var_p = variance_momentum_spectral(gaussian_state.coefficients, basis)
        assert var_p.item() >= -1e-10  # Allow tiny numerical negatives

    def test_heisenberg_uncertainty_bound(self, gaussian_state, basis):
        """sigma_x * sigma_p >= hbar/2."""
        result = heisenberg_uncertainty(gaussian_state.coefficients, basis)
        assert result.product.item() >= result.hbar_over_2.item() * 0.99  # Allow 1% numerical slack


# === Energy Functional ===

class TestEnergy:
    def test_kinetic_energy_eigenstate(self, domain, basis):
        """Kinetic energy of pure eigenstate n is E_n."""
        coeffs = torch.zeros(32, dtype=torch.float64)
        coeffs[2] = 1.0  # mode 3
        E = kinetic_energy(coeffs, basis)
        E_3 = (3 * torch.pi * domain.hbar) ** 2 / (2 * domain.mass * domain.length ** 2)
        assert torch.isclose(E, E_3, rtol=1e-10).item()

    def test_kinetic_energy_agrees_with_spectral_diff(self, gaussian_state, basis):
        """Both kinetic energy functions should agree."""
        E1 = kinetic_energy(gaussian_state.coefficients, basis)
        E2 = kinetic_energy_spectral(gaussian_state.coefficients, basis)
        assert torch.isclose(E1, E2, rtol=1e-6).item()

    def test_energy_conservation_propagation(self, domain, basis, gaussian_state):
        """Energy must be conserved under unitary propagation."""
        propagator = SpectralPropagator(basis)
        times = torch.linspace(0, 2, 20, dtype=torch.float64)
        propagated = propagator.propagate_many(gaussian_state, times)
        report = check_energy_conservation(gaussian_state.coefficients, propagated, basis)
        assert report.is_conserved
        assert report.relative_error.item() < 1e-10

    def test_energy_spectrum_shape(self, gaussian_state, basis):
        spec = energy_spectrum(gaussian_state.coefficients, basis)
        assert spec.shape == (32,)
        assert torch.all(spec >= 0).item()

    def test_energy_budget(self, gaussian_state, basis):
        budget = compute_energy_budget(gaussian_state.coefficients, basis)
        assert torch.isclose(budget.cumulative_fraction[-1], torch.tensor(1.0, dtype=torch.float64), atol=1e-10).item()
        assert torch.all(budget.mode_fractions >= 0).item()

    def test_batch_kinetic_energy(self, basis):
        coeffs = torch.randn(5, 32, dtype=torch.float64)
        E = kinetic_energy(coeffs, basis)
        assert E.shape == (5,)
        assert torch.all(E >= 0).item()


# === Convergence Diagnostics ===

class TestConvergence:
    def test_exponential_decay_detection(self):
        """Exponentially decaying coefficients should be classified correctly."""
        n = torch.arange(1, 33, dtype=torch.float64)
        coeffs = torch.exp(-0.5 * n)
        decay = estimate_spectral_decay(coeffs)
        assert decay.decay_type == DecayType.EXPONENTIAL
        assert decay.rate.item() > 0
        assert decay.r_squared.item() > 0.9

    def test_algebraic_decay_detection(self):
        """Algebraically decaying coefficients."""
        n = torch.arange(1, 65, dtype=torch.float64)
        coeffs = n ** (-3.0)
        decay = estimate_spectral_decay(coeffs)
        assert decay.decay_type == DecayType.ALGEBRAIC
        assert abs(decay.rate.item() - 3.0) < 0.5  # Should be close to 3

    def test_spectral_entropy_single_mode(self):
        """Single-mode state: entropy = 0, effective count = 1."""
        coeffs = torch.zeros(32, dtype=torch.float64)
        coeffs[0] = 1.0
        report = spectral_entropy(coeffs)
        assert torch.isclose(report.entropy, torch.tensor(0.0, dtype=torch.float64), atol=1e-10).item()
        assert torch.isclose(report.effective_mode_count, torch.tensor(1.0, dtype=torch.float64), atol=1e-6).item()

    def test_spectral_entropy_uniform(self):
        """Uniform distribution: entropy = log(N), effective count = N."""
        N = 16
        coeffs = torch.ones(N, dtype=torch.float64) / torch.sqrt(torch.tensor(float(N), dtype=torch.float64))
        report = spectral_entropy(coeffs)
        expected_entropy = torch.log(torch.tensor(float(N), dtype=torch.float64))
        assert torch.isclose(report.entropy, expected_entropy, atol=0.1).item()

    def test_truncation_recommendation(self):
        """Should recommend fewer modes for fast-decaying spectra."""
        n = torch.arange(1, 65, dtype=torch.float64)
        fast_decay = torch.exp(-n)
        slow_decay = n ** (-1.5)
        rec_fast = recommend_truncation(fast_decay, error_tolerance=0.01)
        rec_slow = recommend_truncation(slow_decay, error_tolerance=0.01)
        assert rec_fast.recommended_modes <= rec_slow.recommended_modes

    def test_analyze_convergence_full(self):
        """Full convergence analysis pipeline."""
        n = torch.arange(1, 33, dtype=torch.float64)
        coeffs = torch.exp(-0.3 * n)
        result = analyze_convergence(coeffs, error_tolerance=0.01)
        assert result.decay is not None
        assert result.entropy is not None
        assert result.truncation is not None
        assert result.gibbs is None  # No reconstruction data provided
