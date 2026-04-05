"""Chebyshev spectral basis for bounded-domain computation.

Chebyshev polynomials of the first kind provide exponential convergence
for smooth functions on bounded domains — the gold standard in spectral
methods.  Unlike the sine basis (which assumes periodicity), Chebyshev
polynomials handle non-periodic boundary data gracefully and avoid the
Gibbs phenomenon for smooth profiles.

The implementation maps an arbitrary [left, right] physical domain to the
canonical Chebyshev interval [-1, 1], evaluates T_n(x) via the stable
three-term recurrence, and provides projection, reconstruction, and
differentiation through spectral coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from spectral_packet_engine.domain import InfiniteWell1D, coerce_tensor

Tensor = torch.Tensor


def _chebyshev_nodes(num_points: int, *, dtype: torch.dtype, device: torch.device | str | None) -> Tensor:
    """Chebyshev-Gauss-Lobatto nodes on [-1, 1].

    These are the extrema of T_{N-1}(x) plus the endpoints,
    yielding optimal interpolation points that avoid Runge's phenomenon.
    """
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    j = torch.arange(num_points, dtype=dtype, device=device)
    return torch.cos(torch.pi * j / (num_points - 1))


def chebyshev_nodes(domain: InfiniteWell1D, num_points: int) -> Tensor:
    """Chebyshev-Gauss-Lobatto nodes mapped to the physical domain."""
    canonical = _chebyshev_nodes(num_points, dtype=domain.real_dtype, device=domain.device)
    return domain.left + (canonical + 1) * domain.length / 2


def _to_canonical(x: Tensor, domain: InfiniteWell1D) -> Tensor:
    """Map physical coordinates to the canonical [-1, 1] interval."""
    return 2 * (x - domain.left) / domain.length - 1


def _from_canonical(xi: Tensor, domain: InfiniteWell1D) -> Tensor:
    """Map canonical [-1, 1] coordinates back to the physical domain."""
    return domain.left + (xi + 1) * domain.length / 2


def chebyshev_matrix(domain: InfiniteWell1D, modes, x) -> Tensor:
    """Evaluate Chebyshev polynomials T_0(x) ... T_{N-1}(x) on a grid.

    Uses the three-term recurrence:
        T_0(xi) = 1,  T_1(xi) = xi,  T_{n+1}(xi) = 2*xi*T_n(xi) - T_{n-1}(xi)

    Returns shape (len(x), num_modes).
    """
    grid = coerce_tensor(x, dtype=domain.real_dtype, device=domain.device)
    if grid.ndim == 0:
        grid = grid.reshape(1)
    mode_numbers = coerce_tensor(modes, dtype=torch.int64, device=domain.device)
    if mode_numbers.ndim == 0:
        mode_numbers = mode_numbers.reshape(1)
    num_modes = int(mode_numbers.max().item()) + 1

    xi = _to_canonical(grid, domain)
    # Build full Chebyshev matrix up to max mode via recurrence
    T = torch.zeros(grid.shape[0], num_modes, dtype=domain.real_dtype, device=domain.device)
    T[:, 0] = 1.0
    if num_modes > 1:
        T[:, 1] = xi
    for n in range(2, num_modes):
        T[:, n] = 2 * xi * T[:, n - 1] - T[:, n - 2]

    # Select requested modes
    return T[:, mode_numbers]


def chebyshev_quadrature_weights(num_points: int, *, dtype: torch.dtype, device: torch.device | str | None) -> Tensor:
    """Clenshaw-Curtis quadrature weights on [-1, 1].

    These integrate exactly for polynomials up to degree N-1 on the
    Chebyshev-Gauss-Lobatto grid, which is the correct choice for
    projecting onto Chebyshev coefficients.
    """
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    N = num_points - 1
    theta = torch.pi * torch.arange(num_points, dtype=dtype, device=device) / N
    w = torch.zeros(num_points, dtype=dtype, device=device)
    for k in range(num_points):
        s = torch.tensor(0.0, dtype=dtype, device=device)
        for j in range(1, N // 2 + 1):
            b = torch.tensor(1.0 if 2 * j == N else 2.0, dtype=dtype, device=device)
            s = s + b * torch.cos(2 * j * theta[k]) / (4 * j * j - 1)
        w[k] = (1 - s) * 2 / N
    # Endpoint correction
    w[0] = w[0] / 2
    w[-1] = w[-1] / 2
    return w


@dataclass(frozen=True, slots=True)
class ChebyshevBasis:
    """Chebyshev spectral basis on a bounded 1D domain.

    Provides the same interface as InfiniteWellBasis but uses Chebyshev
    polynomials of the first kind, enabling exponential convergence for
    smooth (non-periodic) functions.
    """

    domain: InfiniteWell1D
    num_modes: int

    def __post_init__(self) -> None:
        if self.num_modes <= 0:
            raise ValueError("num_modes must be positive")

    @property
    def mode_numbers(self) -> Tensor:
        return torch.arange(
            self.num_modes,
            dtype=torch.int64,
            device=self.domain.device,
        )

    def nodes(self, num_points: int | None = None) -> Tensor:
        """Chebyshev-Gauss-Lobatto nodes on the physical domain."""
        n = num_points or max(self.num_modes, 2)
        return chebyshev_nodes(self.domain, n)

    def evaluate(self, x) -> Tensor:
        """Evaluate T_0(x) ... T_{N-1}(x), returning shape (len(x), num_modes)."""
        return chebyshev_matrix(self.domain, self.mode_numbers, x)

    def project(self, values, grid) -> Tensor:
        """Project function values onto Chebyshev coefficients via least squares.

        Solves the overdetermined system  V @ a = f  where V is the
        Chebyshev Vandermonde matrix.  This is robust for any grid
        (not just Chebyshev nodes) and avoids quadrature weight issues.
        """
        spatial_grid = coerce_tensor(grid, dtype=self.domain.real_dtype, device=self.domain.device)
        vals = coerce_tensor(values, dtype=self.domain.real_dtype, device=self.domain.device)
        V = self.evaluate(spatial_grid)  # (num_points, num_modes)

        if vals.ndim == 1:
            result = torch.linalg.lstsq(V, vals)
            return result.solution
        # Batch: vals shape (batch, num_points)
        result = torch.linalg.lstsq(V, vals.T)
        return result.solution.T

    def reconstruct(self, coefficients, x) -> Tensor:
        """Reconstruct function values from Chebyshev coefficients.

        Uses Clenshaw summation for numerical stability when evaluating
        sum_n a_n T_n(x).
        """
        coeffs = coerce_tensor(coefficients, dtype=self.domain.real_dtype, device=self.domain.device)
        grid = coerce_tensor(x, dtype=self.domain.real_dtype, device=self.domain.device)
        basis = self.evaluate(grid)  # (len(x), num_modes)

        if coeffs.ndim == 1:
            if coeffs.shape[0] != self.num_modes:
                raise ValueError("coefficient vector length does not match the basis")
            return basis @ coeffs
        if coeffs.ndim == 2:
            if coeffs.shape[-1] != self.num_modes:
                raise ValueError("coefficient matrix width does not match the basis")
            return coeffs @ basis.transpose(0, 1)
        raise ValueError("coefficients must be one- or two-dimensional")

    def differentiation_matrix(self) -> Tensor:
        """Chebyshev spectral differentiation matrix in coefficient space.

        Given coefficients a_n of f(x) = sum a_n T_n(x), the derivative
        f'(x) = sum b_n T_n(x) where the b_n are related to a_n by the
        recurrence:

            b_{N-1} = 0
            b_{N-2} = 2(N-1) * a_{N-1}
            b_n = b_{n+2} + 2(n+1) * a_{n+1}   for n = N-3 ... 1
            b_0 = a_1 + b_2 / 2

        scaled by 2/L for the physical domain mapping.

        Returns a (num_modes, num_modes) matrix D such that b = D @ a.
        """
        N = self.num_modes
        D = torch.zeros(N, N, dtype=self.domain.real_dtype, device=self.domain.device)

        if N <= 1:
            return D

        # Build the recurrence relation as a matrix
        # b_{N-1} = 0 (already zero)
        # b_{N-2} = 2*(N-1) * a_{N-1}
        if N >= 2:
            D[N - 2, N - 1] = 2 * (N - 1)

        # b_n = b_{n+2} + 2*(n+1)*a_{n+1}, iterate downward
        for n in range(N - 3, 0, -1):
            D[n, :] = D[n + 2, :]
            D[n, n + 1] += 2 * (n + 1)

        # b_0 = a_1 + b_2/2
        if N >= 3:
            D[0, :] = D[2, :] / 2
        else:
            D[0, :] = 0
        D[0, 1] += 1

        # Scale for physical domain: d/dx_phys = (2/L) * d/d_xi
        D = D * (2 / self.domain.length)
        return D

    def differentiate(self, coefficients, *, order: int = 1) -> Tensor:
        """Compute the spectral derivative of a function given its Chebyshev coefficients.

        Returns the Chebyshev coefficients of the derivative.
        """
        if order < 0:
            raise ValueError("differentiation order must be non-negative")
        if order == 0:
            return coerce_tensor(coefficients)
        coeffs = coerce_tensor(coefficients, dtype=self.domain.real_dtype, device=self.domain.device)
        D = self.differentiation_matrix()
        for _ in range(order):
            if coeffs.ndim == 1:
                coeffs = D @ coeffs
            elif coeffs.ndim == 2:
                coeffs = (D @ coeffs.T).T
            else:
                raise ValueError("coefficients must be one- or two-dimensional")
        return coeffs


__all__ = [
    "ChebyshevBasis",
    "chebyshev_matrix",
    "chebyshev_nodes",
    "chebyshev_quadrature_weights",
]
