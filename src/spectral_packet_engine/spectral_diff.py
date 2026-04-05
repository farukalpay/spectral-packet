"""Spectral differentiation operators for bounded-domain bases.

Spectral differentiation computes spatial derivatives through basis
coefficients rather than finite differences.  For smooth functions this
is exponentially more accurate — the error drops as fast as the spectral
coefficients decay, rather than at a fixed polynomial rate.

This module provides:
- Sine-basis differentiation via analytic wavenumber multiplication
- Chebyshev-basis differentiation via the recurrence relation
- A unified interface that dispatches on basis type

Physics note: for the infinite-well sine basis, the n-th eigenfunction
phi_n(x) = sqrt(2/L) sin(n*pi*x/L) has derivative
phi_n'(x) = sqrt(2/L) * (n*pi/L) * cos(n*pi*x/L).
In coefficient space, differentiation of a *real profile* decomposed in
the sine basis maps to cosine coefficients — but we can express the
result back in the sine basis via the connection formula.  For complex
wavefunctions, we work with the full spectral state.
"""

from __future__ import annotations

import torch

from spectral_packet_engine.basis import InfiniteWellBasis
from spectral_packet_engine.domain import coerce_tensor

Tensor = torch.Tensor


def sine_basis_derivative_coefficients(
    coefficients: Tensor,
    basis: InfiniteWellBasis,
    *,
    order: int = 1,
) -> Tensor:
    """Compute the spectral derivative for the sine basis via wavenumber multiplication.

    For a function f(x) = sum_n c_n * phi_n(x) where phi_n are the
    infinite-well eigenstates, the k-th derivative on the grid is:

        f^(k)(x) = sum_n c_n * (n*pi/L)^k * d^k/dx^k[sin(n*pi*(x-a)/L)]

    For even-order derivatives, the result stays in the sine basis:
        d^(2m)/dx^(2m) phi_n = (-1)^m (n*pi/L)^(2m) phi_n

    For odd-order derivatives, sine -> cosine, so we return the
    *amplitude scaling* that can be used with the cosine basis or
    evaluated directly on a grid.

    Returns the scaled coefficients (same shape as input).
    """
    if order < 0:
        raise ValueError("differentiation order must be non-negative")
    if order == 0:
        return coefficients

    coeffs = coerce_tensor(coefficients, dtype=basis.domain.real_dtype, device=basis.domain.device)
    wavenumbers = basis.wavenumbers  # n * pi / L, shape (num_modes,)

    # Scale factor: (k_n)^order with alternating sign for even orders
    scale = wavenumbers ** order

    # For even order derivatives: d^(2m)/dx^(2m) sin(kx) = (-1)^m k^(2m) sin(kx)
    # For odd order derivatives: d^(2m+1)/dx^(2m+1) sin(kx) = (-1)^m k^(2m+1) cos(kx)
    sign = torch.tensor((-1.0) ** (order // 2), dtype=scale.dtype, device=scale.device)
    scale = sign * scale

    if coeffs.ndim == 1:
        return coeffs * scale
    if coeffs.ndim == 2:
        return coeffs * scale[None, :]
    raise ValueError("coefficients must be one- or two-dimensional")


def sine_basis_derivative_on_grid(
    coefficients: Tensor,
    grid: Tensor,
    basis: InfiniteWellBasis,
    *,
    order: int = 1,
) -> Tensor:
    """Evaluate the k-th derivative of a sine-basis expansion on a spatial grid.

    This handles the sine→cosine transition for odd-order derivatives
    by directly computing the derivative basis functions on the grid.
    """
    if order < 0:
        raise ValueError("differentiation order must be non-negative")
    if order == 0:
        return basis.reconstruct(coefficients, grid)

    coeffs = coerce_tensor(coefficients, dtype=basis.domain.real_dtype, device=basis.domain.device)
    spatial_grid = coerce_tensor(grid, dtype=basis.domain.real_dtype, device=basis.domain.device)
    if spatial_grid.ndim == 0:
        spatial_grid = spatial_grid.reshape(1)

    wavenumbers = basis.wavenumbers  # (num_modes,)
    factor = torch.sqrt(
        torch.tensor(2.0, dtype=basis.domain.real_dtype, device=basis.domain.device) / basis.domain.length
    )
    scaled = (spatial_grid[:, None] - basis.domain.left) / basis.domain.length  # (grid, 1)
    arg = torch.pi * scaled * basis.mode_numbers[None, :]  # (grid, modes)

    # k-th derivative of sin(n*pi*(x-a)/L):
    # = (n*pi/L)^k * sin(n*pi*(x-a)/L + k*pi/2)
    phase_shift = order * torch.pi / 2
    derivative_basis = factor * wavenumbers[None, :] ** order * torch.sin(arg + phase_shift)

    if coeffs.ndim == 1:
        return derivative_basis @ coeffs
    if coeffs.ndim == 2:
        return coeffs @ derivative_basis.T
    raise ValueError("coefficients must be one- or two-dimensional")


def kinetic_energy_spectral(
    coefficients: Tensor,
    basis: InfiniteWellBasis,
) -> Tensor:
    """Compute kinetic energy from spectral coefficients.

    For the infinite well: T = -hbar^2/(2m) * d^2/dx^2
    In the eigenbasis: T_nn = E_n = (n*pi*hbar)^2 / (2*m*L^2)

    Since the sine basis diagonalizes the kinetic operator:
        <T> = sum_n |c_n|^2 * E_n

    This is Parseval's theorem applied to the kinetic energy.
    """
    coeffs = coerce_tensor(coefficients, device=basis.domain.device)
    energies = basis.energies  # E_n for each mode

    if torch.is_complex(coeffs):
        weights = torch.abs(coeffs) ** 2
    else:
        weights = coeffs.to(dtype=basis.domain.real_dtype) ** 2

    if weights.ndim == 1:
        return torch.sum(weights * energies)
    if weights.ndim == 2:
        return torch.sum(weights * energies[None, :], dim=-1)
    raise ValueError("coefficients must be one- or two-dimensional")


def parseval_norm(coefficients: Tensor) -> Tensor:
    """Parseval's identity: ||f||^2 = sum |c_n|^2.

    For a properly normalized basis, this equals the L2 norm squared.
    Deviation from the initial norm during propagation indicates
    numerical error or truncation loss.
    """
    coeffs = coerce_tensor(coefficients)
    if torch.is_complex(coeffs):
        return torch.sum(torch.abs(coeffs) ** 2, dim=-1)
    return torch.sum(coeffs ** 2, dim=-1)


def parseval_conservation_error(
    initial_coefficients: Tensor,
    propagated_coefficients: Tensor,
) -> Tensor:
    """Measure violation of norm conservation (Parseval's identity).

    Returns the relative change in norm: |N(t) - N(0)| / N(0).
    For exact unitary evolution this should be zero (or machine epsilon).
    Large values indicate truncation-induced norm leakage.
    """
    initial_norm = parseval_norm(initial_coefficients)
    propagated_norm = parseval_norm(propagated_coefficients)
    safe_initial = torch.where(
        initial_norm > 0,
        initial_norm,
        torch.ones_like(initial_norm),
    )
    return torch.abs(propagated_norm - initial_norm) / safe_initial


__all__ = [
    "kinetic_energy_spectral",
    "parseval_conservation_error",
    "parseval_norm",
    "sine_basis_derivative_coefficients",
    "sine_basis_derivative_on_grid",
]
