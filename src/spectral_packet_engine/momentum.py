"""Momentum-space observables and Heisenberg uncertainty analysis.

For a particle in an infinite well with eigenstates phi_n(x), the
momentum operator p = -i*hbar*d/dx acts on the basis as:

    p phi_n(x) = -i*hbar * sqrt(2/L) * (n*pi/L) * cos(n*pi*(x-a)/L)

The expectation value <p> and variance <p^2> - <p>^2 can be computed
from the spectral coefficients without returning to the grid, using:

    <p^2> = sum_n |c_n|^2 * (n*pi*hbar/L)^2 = 2*m * <T>

where <T> is the kinetic energy.  For <p> of a complex wavefunction,
we need the off-diagonal matrix elements of p in the sine basis.

The Heisenberg uncertainty product sigma_x * sigma_p >= hbar/2
provides a fundamental consistency check for any quantum state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from spectral_packet_engine.basis import InfiniteWellBasis
from spectral_packet_engine.domain import InfiniteWell1D, coerce_tensor

Tensor = torch.Tensor


def _momentum_matrix_elements(basis: InfiniteWellBasis) -> Tensor:
    """Compute the matrix elements <phi_m | p | phi_n> in the sine basis.

    For the infinite well with phi_n(x) = sqrt(2/L) sin(n*pi*(x-a)/L):

        <phi_m | p | phi_n> = -i*hbar * (2/L) * integral sin(m*k*x) * n*k * cos(n*k*x) dx

    where k = pi/L.  The integral evaluates to:

        <m|p|n> = -i*hbar * (2/(L)) * n*pi/L * { L/(pi) * [m/(m^2-n^2)] * [(-1)^{m+n} - 1] }  if m != n
                = 0  if m == n (diagonal vanishes for real basis)

    Simplified: <m|p|n> = -i*hbar * 2*m*n / (L*(m^2-n^2)) * [(-1)^{m+n} - 1]
    which is nonzero only when m+n is odd.
    """
    N = basis.num_modes
    L = basis.domain.length
    hbar = basis.domain.hbar
    dtype = basis.domain.real_dtype
    device = basis.domain.device

    modes = torch.arange(1, N + 1, dtype=dtype, device=device)
    m = modes[:, None]  # (N, 1)
    n = modes[None, :]  # (1, N)

    # m^2 - n^2, with safe division
    diff_sq = m ** 2 - n ** 2
    safe_diff = torch.where(diff_sq != 0, diff_sq, torch.ones_like(diff_sq))

    # 1 - (-1)^{m+n} is +2 when m+n is odd, 0 when m+n is even
    m_int = torch.arange(1, N + 1, device=device)
    parity = (m_int[:, None] + m_int[None, :]) % 2  # 1 if odd, 0 if even
    sign_factor = 2.0 * parity.to(dtype=dtype)

    # <m|p|n> = -i*hbar * 4*m*n / (L*(m^2-n^2)) when m+n is odd
    # We store the real part: matrix_real such that <m|p|n> = -i * matrix_real
    matrix_real = hbar * 2 * m * n / (L * safe_diff) * sign_factor
    # Zero out diagonal (m == n)
    matrix_real = torch.where(diff_sq != 0, matrix_real, torch.zeros_like(matrix_real))

    return matrix_real  # Real part; actual <m|p|n> = -i * matrix_real


def expectation_momentum_spectral(
    coefficients: Tensor,
    basis: InfiniteWellBasis,
) -> Tensor:
    """Compute <p> from spectral coefficients.

    <p> = sum_{m,n} c_m* <m|p|n> c_n = -i * sum_{m,n} c_m* P_mn c_n

    where P_mn is the real-valued momentum matrix element.
    For a real-valued state (all c_n real), <p> = 0 by symmetry.
    """
    coeffs = coerce_tensor(coefficients, dtype=basis.domain.complex_dtype, device=basis.domain.device)
    P = _momentum_matrix_elements(basis)  # Real-valued (N, N)

    if coeffs.ndim == 1:
        # <p> = -i * c^dagger @ P @ c
        inner = torch.conj(coeffs) @ P.to(dtype=coeffs.dtype) @ coeffs
        return (-1j * inner).real
    if coeffs.ndim == 2:
        # Batch: each row is a set of coefficients
        Pc = (P.to(dtype=coeffs.dtype) @ coeffs.T).T  # (batch, N)
        inner = torch.sum(torch.conj(coeffs) * Pc, dim=-1)
        return (-1j * inner).real
    raise ValueError("coefficients must be one- or two-dimensional")


def expectation_momentum_squared_spectral(
    coefficients: Tensor,
    basis: InfiniteWellBasis,
) -> Tensor:
    """Compute <p^2> from spectral coefficients.

    Since the sine basis diagonalizes the kinetic energy operator:
        <p^2> = 2*m*<T> = sum_n |c_n|^2 * (n*pi*hbar/L)^2

    This is exact and avoids the need for matrix elements.
    """
    coeffs = coerce_tensor(coefficients, device=basis.domain.device)
    k_n = basis.wavenumbers  # n*pi/L
    p_n_sq = (basis.domain.hbar * k_n) ** 2

    if torch.is_complex(coeffs):
        weights = torch.abs(coeffs) ** 2
    else:
        weights = coeffs.to(dtype=basis.domain.real_dtype) ** 2

    if weights.ndim == 1:
        return torch.sum(weights * p_n_sq)
    if weights.ndim == 2:
        return torch.sum(weights * p_n_sq[None, :], dim=-1)
    raise ValueError("coefficients must be one- or two-dimensional")


def variance_momentum_spectral(
    coefficients: Tensor,
    basis: InfiniteWellBasis,
) -> Tensor:
    """Compute Var(p) = <p^2> - <p>^2 from spectral coefficients."""
    mean_p = expectation_momentum_spectral(coefficients, basis)
    mean_p_sq = expectation_momentum_squared_spectral(coefficients, basis)
    return mean_p_sq - mean_p ** 2


def expectation_position_spectral(
    coefficients: Tensor,
    basis: InfiniteWellBasis,
) -> Tensor:
    """Compute <x> from spectral coefficients and the sine-basis position matrix."""
    coeffs = coerce_tensor(coefficients, dtype=basis.domain.complex_dtype, device=basis.domain.device)
    X = _position_matrix_elements(basis).to(dtype=coeffs.dtype)

    if coeffs.ndim == 1:
        return (torch.conj(coeffs) @ X @ coeffs).real
    if coeffs.ndim == 2:
        Xc = (X @ coeffs.T).T
        return torch.sum(torch.conj(coeffs) * Xc, dim=-1).real
    raise ValueError("coefficients must be one- or two-dimensional")


def variance_position_spectral(
    coefficients: Tensor,
    basis: InfiniteWellBasis,
) -> Tensor:
    """Compute Var(x) from spectral coefficients and the sine-basis position matrix."""
    return _position_variance_spectral(coefficients, basis)


@dataclass(frozen=True, slots=True)
class UncertaintyProduct:
    """Heisenberg uncertainty analysis for a quantum state."""
    sigma_x: Tensor
    sigma_p: Tensor
    product: Tensor
    hbar_over_2: Tensor
    saturates_bound: Tensor  # True if product ≈ hbar/2 (coherent state)


def heisenberg_uncertainty(
    coefficients: Tensor,
    basis: InfiniteWellBasis,
    *,
    position_variance: Tensor | None = None,
) -> UncertaintyProduct:
    """Compute the Heisenberg uncertainty product sigma_x * sigma_p.

    The fundamental bound is sigma_x * sigma_p >= hbar/2.
    Equality holds only for Gaussian (coherent) states.

    Parameters
    ----------
    coefficients : Tensor
        Spectral coefficients of the state.
    basis : InfiniteWellBasis
        The spectral basis.
    position_variance : Tensor, optional
        Pre-computed position variance. If not provided, it is estimated
        from the spectral coefficients using the position matrix elements.
        For higher accuracy, compute from grid-based observables and pass here.
    """
    var_p = variance_momentum_spectral(coefficients, basis)
    sigma_p = torch.sqrt(torch.clamp(var_p, min=0))

    if position_variance is not None:
        var_x = coerce_tensor(position_variance, dtype=basis.domain.real_dtype, device=basis.domain.device)
    else:
        # Estimate position variance from spectral coefficients via matrix elements
        var_x = _position_variance_spectral(coefficients, basis)

    sigma_x = torch.sqrt(torch.clamp(var_x, min=0))
    product = sigma_x * sigma_p
    hbar_half = basis.domain.hbar / 2

    # Check saturation: product within 1% of the lower bound
    saturates = product < 1.01 * hbar_half

    return UncertaintyProduct(
        sigma_x=sigma_x,
        sigma_p=sigma_p,
        product=product,
        hbar_over_2=hbar_half,
        saturates_bound=saturates,
    )


def _position_matrix_elements(basis: InfiniteWellBasis) -> Tensor:
    """Compute <phi_m | x | phi_n> for the infinite well sine basis.

    <m|x|n> = (2/L) * integral_0^L x * sin(m*pi*x/L) * sin(n*pi*x/L) dx

    For m == n: <n|x|n> = L/2 (center of the well, shifted by domain.left)
    For m != n: <m|x|n> = (2*L/pi^2) * [(-1)^{m+n} - 1] * m*n / (m^2-n^2)^2
               (nonzero only when m+n is odd)
    """
    N = basis.num_modes
    L = basis.domain.length
    a = basis.domain.left
    dtype = basis.domain.real_dtype
    device = basis.domain.device

    modes = torch.arange(1, N + 1, dtype=dtype, device=device)
    m = modes[:, None]
    n = modes[None, :]

    diff_sq = m ** 2 - n ** 2
    safe_diff_sq = torch.where(diff_sq != 0, diff_sq, torch.ones_like(diff_sq))

    m_int = torch.arange(1, N + 1, device=device)
    parity = (m_int[:, None] + m_int[None, :]) % 2
    sign_factor = parity.to(dtype=dtype) * (-2.0)

    off_diag = (4 * L / torch.pi ** 2) * sign_factor * m * n / safe_diff_sq ** 2
    off_diag = torch.where(diff_sq != 0, off_diag, torch.zeros_like(off_diag))

    # Diagonal: <n|x|n> = L/2 (relative to left boundary)
    diag = torch.full((N,), (L / 2).item(), dtype=dtype, device=device)
    X = off_diag + torch.diag(diag)

    # Shift to physical coordinates: <m|x_phys|n> = <m|x_rel|n> + a * delta_{mn}
    X = X + a * torch.eye(N, dtype=dtype, device=device)
    return X


def _position_variance_spectral(
    coefficients: Tensor,
    basis: InfiniteWellBasis,
) -> Tensor:
    """Estimate Var(x) from spectral coefficients and position matrix elements."""
    coeffs = coerce_tensor(coefficients, dtype=basis.domain.complex_dtype, device=basis.domain.device)
    X = _position_matrix_elements(basis).to(dtype=coeffs.dtype)

    if coeffs.ndim == 1:
        mean_x = (torch.conj(coeffs) @ X @ coeffs).real
        mean_x2 = (torch.conj(coeffs) @ (X @ X) @ coeffs).real
        return mean_x2 - mean_x ** 2
    if coeffs.ndim == 2:
        Xc = (X @ coeffs.T).T
        mean_x = torch.sum(torch.conj(coeffs) * Xc, dim=-1).real
        X2c = (X @ Xc.T).T
        mean_x2 = torch.sum(torch.conj(coeffs) * X2c, dim=-1).real
        return mean_x2 - mean_x ** 2
    raise ValueError("coefficients must be one- or two-dimensional")


__all__ = [
    "UncertaintyProduct",
    "expectation_position_spectral",
    "expectation_momentum_spectral",
    "expectation_momentum_squared_spectral",
    "heisenberg_uncertainty",
    "variance_position_spectral",
    "variance_momentum_spectral",
]
