"""Density matrix formalism for pure and mixed quantum states.

The density operator rho encodes the full statistical description of a
quantum state, generalising the pure-state ket |psi> to statistical
mixtures.  In a discrete spectral basis {|n>} with N modes the density
matrix is an N x N positive-semidefinite Hermitian matrix satisfying
Tr(rho) = 1.

Pure states
    rho = |psi><psi|  <=>  rho_{mn} = c_m c_n*     Tr(rho^2) = 1

Mixed states
    rho = sum_i p_i |psi_i><psi_i|                  Tr(rho^2) < 1

Thermal (Gibbs) state
    rho = (1/Z) sum_n exp(-E_n / kT) |n><n|         Z = sum_n exp(-E_n / kT)

The module provides construction routines, standard measures (purity,
von Neumann entropy, linear entropy), distance measures (fidelity, trace
distance), and partial trace for bipartite systems.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from spectral_packet_engine.basis import InfiniteWellBasis
from spectral_packet_engine.domain import coerce_tensor, complex_dtype_for

Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DensityMatrixResult:
    """Full analysis of a density matrix.

    Attributes
    ----------
    rho : Tensor
        Shape ``(N, N)`` -- the density matrix (complex).
    eigenvalues : Tensor
        Shape ``(N,)`` -- eigenvalues of rho in ascending order.
    purity : Tensor
        Scalar -- Tr(rho^2), equals 1 for pure states.
    von_neumann_entropy : Tensor
        Scalar -- S = -Tr(rho ln rho).
    linear_entropy : Tensor
        Scalar -- S_L = 1 - Tr(rho^2).
    rank : int
        Effective rank (number of eigenvalues exceeding a threshold).
    is_pure : bool
        True if the purity is within tolerance of 1.
    """
    rho: Tensor
    eigenvalues: Tensor
    purity: Tensor
    von_neumann_entropy: Tensor
    linear_entropy: Tensor
    rank: int
    is_pure: bool


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def pure_state_density_matrix(coefficients: Tensor) -> Tensor:
    r"""Density matrix for a pure state.

    rho_{mn} = c_m c_n^*

    Parameters
    ----------
    coefficients : Tensor
        Shape ``(N,)`` -- complex expansion coefficients c_n.

    Returns
    -------
    Tensor
        Shape ``(N, N)`` -- the density matrix.
    """
    c = coerce_tensor(coefficients)
    if not torch.is_complex(c):
        c = c.to(torch.complex128)
    # Outer product |c><c|
    return c.unsqueeze(1) * c.conj().unsqueeze(0)


def mixed_state_density_matrix(
    states: list[Tensor],
    weights: Tensor,
) -> Tensor:
    r"""Density matrix for a statistical mixture.

    rho = sum_i p_i |psi_i><psi_i|

    Parameters
    ----------
    states : list of Tensor
        Each element has shape ``(N,)`` -- expansion coefficients.
    weights : Tensor
        Shape ``(K,)`` -- mixture probabilities, must be non-negative and
        sum to 1.

    Returns
    -------
    Tensor
        Shape ``(N, N)`` -- the mixed density matrix.
    """
    w = coerce_tensor(weights, dtype=torch.float64)
    if torch.any(w < 0):
        raise ValueError("mixture weights must be non-negative")
    w_sum = w.sum()
    if not torch.isclose(w_sum, torch.ones_like(w_sum)):
        raise ValueError("mixture weights must sum to 1")
    if len(states) != w.shape[0]:
        raise ValueError("number of states must match number of weights")

    N = states[0].shape[0]
    cdtype = torch.complex128
    rho = torch.zeros(N, N, dtype=cdtype, device=w.device)

    for i, state in enumerate(states):
        c = coerce_tensor(state, dtype=cdtype, device=w.device)
        rho = rho + w[i].to(cdtype) * (c.unsqueeze(1) * c.conj().unsqueeze(0))

    return rho


def thermal_density_matrix(
    basis: InfiniteWellBasis,
    temperature: float,
    *,
    k_boltzmann: float = 1.0,
) -> Tensor:
    r"""Thermal (Gibbs) density matrix in the energy eigenbasis.

    rho = (1/Z) sum_n exp(-E_n / kT) |n><n|
    Z   = sum_n exp(-E_n / kT)

    Parameters
    ----------
    basis : InfiniteWellBasis
        Provides the eigenenergies E_n.
    temperature : float
        Temperature T (in natural units determined by k_boltzmann).
    k_boltzmann : float
        Boltzmann constant.

    Returns
    -------
    Tensor
        Shape ``(N, N)`` -- diagonal thermal density matrix.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    E_n = basis.energies  # (num_modes,)
    dtype = basis.domain.real_dtype
    device = basis.domain.device
    cdtype = complex_dtype_for(dtype)
    N = basis.num_modes

    beta = 1.0 / (k_boltzmann * temperature)
    # Shift energies so the minimum is zero for numerical stability
    E_shifted = E_n - E_n.min()
    boltzmann = torch.exp(-beta * E_shifted)
    Z = boltzmann.sum()
    populations = boltzmann / Z

    rho = torch.zeros(N, N, dtype=cdtype, device=device)
    for n in range(N):
        rho[n, n] = populations[n].to(cdtype)

    return rho


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------

def von_neumann_entropy(rho: Tensor) -> Tensor:
    r"""Von Neumann entropy of a density matrix.

    S = -Tr(rho ln rho) = -sum_i lambda_i ln(lambda_i)

    where lambda_i are the eigenvalues of rho.  Eigenvalues that are zero
    or negative (numerical noise) are excluded from the sum.
    """
    rho = coerce_tensor(rho)
    # Use eigenvalues of the Hermitian density matrix
    eigenvalues = torch.linalg.eigvalsh(rho).real
    # Filter positive eigenvalues
    mask = eigenvalues > 0
    lam = eigenvalues[mask]
    return -torch.sum(lam * torch.log(lam))


def purity(rho: Tensor) -> Tensor:
    r"""Purity of a density matrix: Tr(rho^2).

    Equals 1 for pure states, strictly less than 1 for mixed states,
    and at least 1/N for a maximally mixed N-dimensional state.
    """
    rho = coerce_tensor(rho)
    rho2 = rho @ rho
    return torch.trace(rho2).real


def fidelity(rho: Tensor, sigma: Tensor) -> Tensor:
    r"""Quantum state fidelity between two density matrices.

    F(rho, sigma) = [Tr(sqrt(sqrt(rho) sigma sqrt(rho)))]^2

    For the square root of rho we use the eigendecomposition
    rho = U diag(lambda) U^dag  =>  sqrt(rho) = U diag(sqrt(lambda)) U^dag.

    Fidelity is 1 if and only if rho = sigma.
    """
    rho = coerce_tensor(rho)
    sigma = coerce_tensor(sigma)

    # Eigendecomposition of rho
    eigenvalues, U = torch.linalg.eigh(rho)
    # Clamp to avoid sqrt of negative numerical noise
    sqrt_lam = torch.sqrt(eigenvalues.real.clamp(min=0.0)).to(rho.dtype)
    sqrt_rho = U @ torch.diag(sqrt_lam) @ U.conj().T

    # Product: sqrt(rho) @ sigma @ sqrt(rho)
    M = sqrt_rho @ sigma @ sqrt_rho

    # Eigenvalues of M (should be non-negative)
    M_eigvals = torch.linalg.eigvalsh(M).real.clamp(min=0.0)
    sqrt_M_trace = torch.sum(torch.sqrt(M_eigvals))

    return sqrt_M_trace ** 2


def trace_distance(rho: Tensor, sigma: Tensor) -> Tensor:
    r"""Trace distance between two density matrices.

    D(rho, sigma) = (1/2) Tr|rho - sigma| = (1/2) sum_i |lambda_i|

    where lambda_i are the eigenvalues of (rho - sigma).
    """
    rho = coerce_tensor(rho)
    sigma = coerce_tensor(sigma)
    diff = rho - sigma
    eigenvalues = torch.linalg.eigvalsh(diff).real
    return 0.5 * torch.sum(torch.abs(eigenvalues))


# ---------------------------------------------------------------------------
# Partial trace
# ---------------------------------------------------------------------------

def partial_trace(rho: Tensor, dims: tuple[int, int], trace_out: int) -> Tensor:
    r"""Partial trace of a bipartite density matrix.

    Given a density matrix rho acting on a Hilbert space H_A (x) H_B of
    total dimension d_A * d_B, computes the reduced density matrix by
    tracing out one subsystem.

    Parameters
    ----------
    rho : Tensor
        Shape ``(d_A * d_B, d_A * d_B)`` -- the full density matrix.
    dims : tuple of int
        ``(d_A, d_B)`` -- dimensions of the two subsystems.
    trace_out : int
        Which subsystem to trace over: 0 traces out A (returns rho_B),
        1 traces out B (returns rho_A).

    Returns
    -------
    Tensor
        The reduced density matrix.
    """
    rho = coerce_tensor(rho)
    d_A, d_B = dims

    if rho.shape[0] != d_A * d_B or rho.shape[1] != d_A * d_B:
        raise ValueError(
            f"density matrix shape {rho.shape} is incompatible with "
            f"dims ({d_A}, {d_B})"
        )
    if trace_out not in (0, 1):
        raise ValueError("trace_out must be 0 or 1")

    # Reshape into a rank-4 tensor: rho_{a1 b1, a2 b2}
    rho_4 = rho.reshape(d_A, d_B, d_A, d_B)

    if trace_out == 0:
        # Trace over A: rho_B_{b1 b2} = sum_a rho_{a b1, a b2}
        return torch.einsum("abac->bc", rho_4)
    else:
        # Trace over B: rho_A_{a1 a2} = sum_b rho_{a1 b, a2 b}
        return torch.einsum("abcb->ac", rho_4)


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_density_matrix(
    rho: Tensor,
    *,
    purity_tolerance: float = 1e-6,
    rank_threshold: float = 1e-10,
) -> DensityMatrixResult:
    """Compute a comprehensive set of diagnostics for a density matrix.

    Parameters
    ----------
    rho : Tensor
        Shape ``(N, N)`` -- density matrix (Hermitian, positive semi-definite,
        trace 1).
    purity_tolerance : float
        Tolerance for declaring the state pure (|Tr(rho^2) - 1| < tol).
    rank_threshold : float
        Eigenvalues below this threshold are treated as zero for rank
        computation.

    Returns
    -------
    DensityMatrixResult
    """
    rho = coerce_tensor(rho)

    eigenvalues = torch.linalg.eigvalsh(rho).real
    p = purity(rho)
    S_vn = von_neumann_entropy(rho)
    S_lin = 1.0 - p
    effective_rank = int((eigenvalues > rank_threshold).sum().item())
    is_pure_flag = bool(torch.abs(p - 1.0).item() < purity_tolerance)

    return DensityMatrixResult(
        rho=rho,
        eigenvalues=eigenvalues,
        purity=p,
        von_neumann_entropy=S_vn,
        linear_entropy=S_lin,
        rank=effective_rank,
        is_pure=is_pure_flag,
    )


__all__ = [
    "DensityMatrixResult",
    "analyze_density_matrix",
    "fidelity",
    "mixed_state_density_matrix",
    "partial_trace",
    "pure_state_density_matrix",
    "purity",
    "thermal_density_matrix",
    "trace_distance",
    "von_neumann_entropy",
]
