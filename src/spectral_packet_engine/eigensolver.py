"""Spectral Galerkin eigensolver for the 1D Schrodinger equation.

Solves the time-independent Schrodinger equation

    -hbar^2/(2m) d^2 psi/dx^2 + V(x) psi(x) = E psi(x)

on a bounded domain [a, b] with Dirichlet boundary conditions psi(a) = psi(b) = 0.

Method: Galerkin projection onto the sine basis
--------------------------------------------
1. Expand psi in the orthonormal sine basis of the infinite well:
       psi(x) = sum_n a_n phi_n(x),   phi_n(x) = sqrt(2/L) sin(n pi (x-a)/L)
2. The kinetic-energy matrix is diagonal in this basis:
       T_nn = E_n = (n pi hbar)^2 / (2 m L^2)
3. Compute the potential-energy matrix V_nm = <phi_n | V | phi_m> via
   high-order quadrature on a fine grid.
4. Assemble H = T + V and diagonalise with ``torch.linalg.eigh``.
5. Reconstruct eigenstates on a user-facing grid from the expansion
   coefficients.

This approach is spectrally accurate for smooth potentials and avoids
the well-known ill-conditioning of the Chebyshev D^2 collocation matrix.

Predefined potentials are provided for common textbook and research problems:
harmonic oscillator, symmetric double well, Morse oscillator, and Poschl-Teller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from spectral_packet_engine.basis import InfiniteWellBasis, eigenenergies, sine_basis_matrix
from spectral_packet_engine.domain import InfiniteWell1D, coerce_tensor

Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# Predefined potentials
# ---------------------------------------------------------------------------

def harmonic_potential(
    x: Tensor,
    omega: float,
    domain: InfiniteWell1D,
) -> Tensor:
    r"""Quantum harmonic oscillator potential.

    V(x) = \frac{1}{2} m \omega^2 (x - x_{\mathrm{center}})^2

    Centred at the domain midpoint so that the ground state is well
    contained within the domain boundaries.
    """
    x_center = domain.midpoint
    m = domain.mass
    return 0.5 * m * omega ** 2 * (x - x_center) ** 2


def double_well_potential(
    x: Tensor,
    a_param: float,
    b_param: float,
    domain: InfiniteWell1D,
) -> Tensor:
    r"""Symmetric double-well (quartic) potential.

    V(x) = a (x - x_{\mathrm{center}})^4 - b (x - x_{\mathrm{center}})^2

    Centred at the domain midpoint.  Barrier height is b^2 / (4a).
    """
    x_center = domain.midpoint
    u = x - x_center
    return a_param * u ** 4 - b_param * u ** 2


def morse_potential(
    x: Tensor,
    D_e: float,
    alpha: float,
    x_eq: float,
) -> Tensor:
    r"""Morse oscillator potential.

    V(x) = D_e \bigl(1 - e^{-\alpha (x - x_{eq})}\bigr)^2

    D_e is the well depth, alpha controls the width, and x_eq is the
    equilibrium bond length.
    """
    return D_e * (1.0 - torch.exp(-alpha * (x - x_eq))) ** 2


def poschl_teller_potential(
    x: Tensor,
    V0: float,
    alpha: float,
    domain: InfiniteWell1D,
) -> Tensor:
    r"""Poschl--Teller potential (reflectionless well).

    V(x) = -\frac{V_0}{\cosh^2\!\bigl(\alpha (x - x_{\mathrm{center}})\bigr)}

    Centred at the domain midpoint.  Admits a finite number of bound
    states determined by V0, alpha, and the particle mass.
    """
    x_center = domain.midpoint
    return -V0 / torch.cosh(alpha * (x - x_center)) ** 2


# ---------------------------------------------------------------------------
# Eigensolver result
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EigensolverResult:
    """Result container for the spectral Galerkin eigensolver.

    Attributes
    ----------
    eigenvalues : Tensor
        Shape ``(num_states,)`` -- the lowest eigenvalues in ascending order.
    eigenstates : Tensor
        Shape ``(num_states, num_points)`` -- corresponding normalised
        wavefunctions evaluated on *grid* (zero at boundaries).
    grid : Tensor
        Shape ``(num_points,)`` -- uniform evaluation grid on the
        physical domain.
    num_states : int
        Number of eigenstates returned.
    potential_on_grid : Tensor
        Shape ``(num_points,)`` -- the potential V(x) evaluated on *grid*.
    """
    eigenvalues: Tensor
    eigenstates: Tensor
    grid: Tensor
    num_states: int
    potential_on_grid: Tensor


# ---------------------------------------------------------------------------
# Main eigensolver
# ---------------------------------------------------------------------------

def solve_eigenproblem(
    potential_fn: Callable[[Tensor], Tensor],
    domain: InfiniteWell1D,
    *,
    num_points: int = 128,
    num_states: int = 10,
    num_quad: int | None = None,
) -> EigensolverResult:
    r"""Solve the 1D Schrodinger eigenproblem by spectral Galerkin projection.

    The wavefunction is expanded in the orthonormal sine basis
    {phi_n} of the infinite well.  The kinetic-energy matrix is diagonal
    (T_{nn} = E_n), and the potential-energy matrix elements

        V_{nm} = \int_a^b \phi_n(x)\,V(x)\,\phi_m(x)\,dx

    are computed via trapezoidal quadrature on a fine grid whose density
    is controlled by *num_quad*.  The resulting Hamiltonian H = T + V is
    dense, symmetric, and diagonalised with ``torch.linalg.eigh``.

    Parameters
    ----------
    potential_fn : callable
        V(x) accepting a Tensor of positions and returning a Tensor of
        potential values.
    domain : InfiniteWell1D
        The spatial domain, carrying mass and hbar.
    num_points : int
        Number of sine-basis functions **and** the number of points on
        the output evaluation grid.
    num_states : int
        Number of lowest eigenstates to return.
    num_quad : int or None
        Number of quadrature points for computing V_{nm}.  If None,
        defaults to ``max(4 * num_points, 512)`` for high accuracy.

    Returns
    -------
    EigensolverResult
    """
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    if num_states < 1:
        raise ValueError("num_states must be at least 1")

    dtype = domain.real_dtype
    device = domain.device
    N = num_points  # number of basis functions

    # Mode indices 1, 2, ..., N
    modes = torch.arange(1, N + 1, dtype=dtype, device=device)

    # Kinetic-energy matrix (diagonal): T_nn = E_n
    E_n = eigenenergies(domain, modes)  # (N,)

    # Quadrature grid for potential-energy matrix elements
    if num_quad is None:
        num_quad = max(4 * N, 512)
    quad_grid = domain.grid(num_quad)
    dx = quad_grid[1] - quad_grid[0]

    # Evaluate potential on quadrature grid
    V_vals = potential_fn(quad_grid)
    V_vals = coerce_tensor(V_vals, dtype=dtype, device=device)

    # Sine basis matrix on quadrature grid: B[i, n] = phi_n(x_i)
    # shape (num_quad, N)
    B = sine_basis_matrix(domain, modes, quad_grid)

    # Trapezoidal quadrature weights
    weights = torch.ones(num_quad, dtype=dtype, device=device) * dx
    weights[0] = weights[0] / 2.0
    weights[-1] = weights[-1] / 2.0

    # Potential-energy matrix: V_nm = sum_i phi_n(x_i) V(x_i) phi_m(x_i) w_i
    # = B^T @ diag(V * w) @ B
    V_weighted = V_vals * weights  # (num_quad,)
    V_mat = B.T @ (V_weighted.unsqueeze(1) * B)  # (N, N)

    # Full Hamiltonian
    H = torch.diag(E_n) + V_mat

    # Diagonalise
    eigenvalues_all, eigenvectors_all = torch.linalg.eigh(H)

    # Select the lowest num_states
    num_available = eigenvalues_all.shape[0]
    num_states = min(num_states, num_available)
    eigenvalues = eigenvalues_all[:num_states]
    coefficients = eigenvectors_all[:, :num_states]  # (N, num_states)

    # Reconstruct eigenstates on the output grid
    out_grid = domain.grid(num_points)
    V_on_grid = potential_fn(out_grid)
    V_on_grid = coerce_tensor(V_on_grid, dtype=dtype, device=device)

    B_out = sine_basis_matrix(domain, modes, out_grid)  # (num_points, N)
    # eigenstates_grid[j, s] = sum_n B_out[j, n] * coefficients[n, s]
    eigenstates_grid = B_out @ coefficients  # (num_points, num_states)

    # Normalise each eigenstate on the output grid via trapezoidal quadrature
    dx_out = out_grid[1] - out_grid[0]
    w_out = torch.ones(num_points, dtype=dtype, device=device) * dx_out
    w_out[0] = w_out[0] / 2.0
    w_out[-1] = w_out[-1] / 2.0

    for s in range(num_states):
        psi = eigenstates_grid[:, s]
        norm_sq = torch.sum(psi ** 2 * w_out)
        if norm_sq > 0:
            eigenstates_grid[:, s] = psi / torch.sqrt(norm_sq)

    # Return eigenstates as (num_states, num_points)
    return EigensolverResult(
        eigenvalues=eigenvalues,
        eigenstates=eigenstates_grid.T.contiguous(),
        grid=out_grid,
        num_states=num_states,
        potential_on_grid=V_on_grid,
    )


# ---------------------------------------------------------------------------
# Overlap and orthonormality checks
# ---------------------------------------------------------------------------

def eigenstate_overlap(state1: Tensor, state2: Tensor, grid: Tensor) -> Tensor:
    r"""Inner product of two real wavefunctions on a grid.

    .. math::
        \langle \psi_1 | \psi_2 \rangle
        = \int_a^b \psi_1(x)\,\psi_2(x)\,dx

    evaluated via the trapezoidal rule.
    """
    state1 = coerce_tensor(state1)
    state2 = coerce_tensor(state2)
    grid = coerce_tensor(grid)
    integrand = state1 * state2
    return torch.trapezoid(integrand, grid)


def verify_orthonormality(result: EigensolverResult, tolerance: float = 1e-6) -> dict:
    r"""Check that the computed eigenstates satisfy orthonormality.

    Computes the overlap matrix O_{ij} = <i|j> and reports the maximum
    deviation from the identity matrix, along with a boolean pass/fail
    flag.

    Returns
    -------
    dict
        ``overlap_matrix`` : Tensor (num_states, num_states)
        ``max_diagonal_error`` : float  -- max |O_{ii} - 1|
        ``max_offdiagonal_error`` : float  -- max |O_{ij}| for i != j
        ``is_orthonormal`` : bool
    """
    n = result.num_states
    grid = result.grid
    overlap = torch.zeros(n, n, dtype=result.eigenstates.dtype, device=result.eigenstates.device)

    for i in range(n):
        for j in range(i, n):
            o = eigenstate_overlap(result.eigenstates[i], result.eigenstates[j], grid)
            overlap[i, j] = o
            overlap[j, i] = o

    diag_error = torch.max(torch.abs(torch.diag(overlap) - 1.0)).item()

    mask = ~torch.eye(n, dtype=torch.bool, device=overlap.device)
    if mask.any():
        offdiag_error = torch.max(torch.abs(overlap[mask])).item()
    else:
        offdiag_error = 0.0

    return {
        "overlap_matrix": overlap,
        "max_diagonal_error": diag_error,
        "max_offdiagonal_error": offdiag_error,
        "is_orthonormal": (diag_error < tolerance) and (offdiag_error < tolerance),
    }


__all__ = [
    "EigensolverResult",
    "double_well_potential",
    "eigenstate_overlap",
    "harmonic_potential",
    "morse_potential",
    "poschl_teller_potential",
    "solve_eigenproblem",
    "verify_orthonormality",
]
