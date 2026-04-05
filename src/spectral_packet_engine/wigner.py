"""Wigner quasi-probability distribution for quantum phase-space analysis.

The Wigner function maps a quantum state to a real-valued distribution over
position and momentum:

    W(x, p) = (1 / pi hbar) integral psi*(x + y) psi(x - y) exp(2ipy / hbar) dy

Unlike a classical probability distribution, W(x, p) can take negative values.
The integrated negativity serves as a witness of non-classicality.

For a bounded domain [a, b] the displacement variable y is constrained so
that both x + y and x - y remain inside [a, b]:

    |y| <= min(x - a, b - x)

Implementation
--------------
For each x-point we sample the correlation function
C(x, y) = psi*(x + y) psi(x - y) on a uniform y-grid and Fourier-transform
in y to get W(x, p).  The wavefunction is interpolated to the required
(x +/- y) points via linear interpolation on the input grid.

The y-grid resolution is chosen to satisfy the Nyquist criterion for the
requested momentum range: dy <= pi * hbar / (2 * p_max).

Marginals
---------
    integral W(x, p) dp = |psi(x)|^2      (position marginal)
    integral W(x, p) dx = |phi(p)|^2      (momentum marginal)
    integral W(x, p) dx dp = 1            (normalisation)

Negativity
----------
    delta = integral |W(x, p)| dx dp - 1  >= 0

with equality if and only if the state is a Gaussian (Hudson's theorem).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from spectral_packet_engine.basis import InfiniteWellBasis, sine_basis_matrix
from spectral_packet_engine.domain import InfiniteWell1D, coerce_tensor, complex_dtype_for

Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class WignerResult:
    """Container for the Wigner function and derived quantities.

    Attributes
    ----------
    x_grid : Tensor
        Shape ``(nx,)`` -- position grid.
    p_grid : Tensor
        Shape ``(np,)`` -- momentum grid.
    W : Tensor
        Shape ``(nx, np)`` -- Wigner quasi-probability distribution.
    x_marginal : Tensor
        Shape ``(nx,)`` -- integral W dp, should equal |psi(x)|^2.
    p_marginal : Tensor
        Shape ``(np,)`` -- integral W dx, should equal |phi(p)|^2.
    negativity : Tensor
        Scalar -- integral |W| dx dp - 1, a non-classicality witness.
    total_integral : Tensor
        Scalar -- integral W dx dp, should be 1 for a normalised state.
    """
    x_grid: Tensor
    p_grid: Tensor
    W: Tensor
    x_marginal: Tensor
    p_marginal: Tensor
    negativity: Tensor
    total_integral: Tensor


def _interpolate_wavefunction(psi: Tensor, grid: Tensor, x_query: Tensor) -> Tensor:
    """Linear interpolation of a complex wavefunction onto arbitrary query points.

    Points outside the grid domain are assigned zero (Dirichlet BC).
    """
    cdtype = psi.dtype
    rdtype = grid.dtype
    x_min = grid[0]
    x_max = grid[-1]
    dx = grid[1] - grid[0]

    # Normalise query points to grid index coordinates
    idx_float = (x_query - x_min) / dx
    idx_lo = idx_float.floor().long()
    t = (idx_float - idx_lo.to(rdtype)).to(cdtype)

    n = grid.shape[0]

    # Clamp to valid range; out-of-bounds will be zeroed below
    idx_lo_clamped = idx_lo.clamp(0, n - 2)
    idx_hi_clamped = (idx_lo_clamped + 1).clamp(0, n - 1)

    val_lo = psi[idx_lo_clamped]
    val_hi = psi[idx_hi_clamped]

    result = val_lo * (1.0 - t) + val_hi * t

    # Zero out-of-bounds queries (Dirichlet)
    mask = (x_query < x_min) | (x_query > x_max)
    result = torch.where(mask, torch.zeros_like(result), result)
    return result


def compute_wigner(
    psi: Tensor,
    grid: Tensor,
    *,
    num_p_points: int = 128,
    p_max: float | None = None,
    hbar: float = 1.0,
) -> WignerResult:
    r"""Compute the Wigner function from a wavefunction on a spatial grid.

    Parameters
    ----------
    psi : Tensor
        Shape ``(num_points,)`` -- complex wavefunction.
    grid : Tensor
        Shape ``(num_points,)`` -- uniformly spaced position grid (ascending).
    num_p_points : int
        Number of momentum grid points.
    p_max : float or None
        Maximum momentum.  If None, estimated from the grid spacing as
        p_max = pi * hbar / dx.
    hbar : float
        Reduced Planck constant.

    Returns
    -------
    WignerResult
    """
    psi = coerce_tensor(psi)
    grid = coerce_tensor(grid, dtype=torch.float64)
    cdtype = complex_dtype_for(grid.dtype)
    dtype = grid.dtype
    device = grid.device

    if not torch.is_complex(psi):
        psi = psi.to(cdtype)
    else:
        psi = psi.to(cdtype)

    num_x = grid.shape[0]
    dx = (grid[-1] - grid[0]) / (num_x - 1)
    a = grid[0]
    b = grid[-1]

    # Momentum grid
    if p_max is None:
        p_max_val = torch.pi * hbar / dx.item()
    else:
        p_max_val = p_max
    p_grid = torch.linspace(-p_max_val, p_max_val, num_p_points, dtype=dtype, device=device)

    # Use the position grid as the x sampling grid
    x_grid = grid

    # Choose y-grid resolution to satisfy Nyquist: dy <= pi*hbar/(2*p_max)
    # The Fourier kernel is exp(2ipy/hbar), so the effective frequency in y
    # is 2*p_max/hbar, requiring dy <= pi / (2*p_max/hbar) = pi*hbar/(2*p_max).
    dy_nyquist = torch.pi * hbar / (2.0 * p_max_val)
    # Maximum possible y range is half the domain length
    y_max_global = (b - a).item() / 2.0
    num_y = max(int(2.0 * y_max_global / dy_nyquist) + 1, num_x)
    # Ensure odd so y=0 is included
    if num_y % 2 == 0:
        num_y += 1

    W = torch.zeros(num_x, num_p_points, dtype=dtype, device=device)

    for ix in range(num_x):
        x_val = x_grid[ix]
        # y range: limited by domain boundaries
        y_max = min((x_val - a).item(), (b - x_val).item())
        if y_max < dx.item() * 0.5:
            # Too close to boundary; skip (contributes zero)
            continue

        y = torch.linspace(-y_max, y_max, num_y, dtype=dtype, device=device)
        dy = y[1] - y[0] if num_y > 1 else torch.tensor(1.0, dtype=dtype, device=device)

        # Evaluate psi*(x + y) and psi(x - y) via interpolation
        x_plus_y = x_val + y
        x_minus_y = x_val - y

        psi_plus = _interpolate_wavefunction(psi, grid, x_plus_y)
        psi_minus = _interpolate_wavefunction(psi, grid, x_minus_y)

        # Correlation function C(y) = psi*(x+y) * psi(x-y)
        C = psi_plus.conj() * psi_minus  # (num_y,)

        # Fourier transform: W(x, p) = (1/pi*hbar) integral C(y) exp(2ipy/hbar) dy
        # Via matrix-vector product for the DFT at specified p values:
        phase_matrix = torch.exp(2j * p_grid.unsqueeze(1).to(cdtype) * y.unsqueeze(0).to(cdtype) / hbar)
        # (num_p, num_y) @ (num_y,) -> (num_p,)
        integral = (phase_matrix @ C.to(cdtype)) * dy
        W[ix, :] = (integral / (torch.pi * hbar)).real

    # Marginals
    x_marginal = torch.trapezoid(W, p_grid, dim=1)       # (num_x,)
    p_marginal = torch.trapezoid(W, x_grid, dim=0)       # (num_p,)

    # Total integral and negativity
    total_integral = torch.trapezoid(x_marginal, x_grid)
    abs_integral = torch.trapezoid(
        torch.trapezoid(torch.abs(W), p_grid, dim=1), x_grid,
    )
    negativity = abs_integral - 1.0

    return WignerResult(
        x_grid=x_grid,
        p_grid=p_grid,
        W=W,
        x_marginal=x_marginal,
        p_marginal=p_marginal,
        negativity=negativity,
        total_integral=total_integral,
    )


def wigner_from_spectral(
    coefficients: Tensor,
    basis: InfiniteWellBasis,
    *,
    num_x_points: int = 64,
    num_p_points: int = 64,
    p_max: float | None = None,
) -> WignerResult:
    """Compute the Wigner function from spectral coefficients in the sine basis.

    Reconstructs the real-space wavefunction on a uniform grid and
    delegates to :func:`compute_wigner`.

    Parameters
    ----------
    coefficients : Tensor
        Shape ``(num_modes,)`` -- complex or real expansion coefficients
        in the infinite-well sine basis.
    basis : InfiniteWellBasis
        The basis used for the expansion.
    num_x_points : int
        Number of spatial grid points for the reconstructed wavefunction.
    num_p_points : int
        Number of momentum grid points.
    p_max : float or None
        Maximum momentum; auto-estimated if None.

    Returns
    -------
    WignerResult
    """
    coeffs = coerce_tensor(coefficients, device=basis.domain.device)
    cdtype = basis.domain.complex_dtype
    if not torch.is_complex(coeffs):
        coeffs = coeffs.to(cdtype)

    grid = basis.domain.grid(num_x_points)
    psi = basis.reconstruct(coeffs, grid)

    hbar_val = basis.domain.hbar.item()
    return compute_wigner(
        psi,
        grid,
        num_p_points=num_p_points,
        p_max=p_max,
        hbar=hbar_val,
    )


__all__ = [
    "WignerResult",
    "compute_wigner",
    "wigner_from_spectral",
]
