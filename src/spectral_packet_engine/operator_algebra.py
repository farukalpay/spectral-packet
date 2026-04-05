"""Operator algebra in spectral basis -- commutators, uncertainty, BCH, ladder operators.

This module provides tools for working with quantum operators represented
as matrices in a finite spectral basis.

Commutators and anticommutators:

    [A, B]  = AB - BA
    {A, B}  = AB + BA

Robertson generalised uncertainty relation:

    Delta A * Delta B >= (1/2) |<[A, B]>|

Baker--Campbell--Hausdorff expansion:

    ln(e^A e^B) = A + B + (1/2)[A,B] + (1/12)([A,[A,B]] - [B,[A,B]]) + ...

Ladder operators for the harmonic oscillator:

    a   = sqrt(m*omega/(2*hbar)) X + i/sqrt(2*m*omega*hbar) P
    a^+ = sqrt(m*omega/(2*hbar)) X - i/sqrt(2*m*omega*hbar) P
    N   = a^+ a
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from spectral_packet_engine.basis import InfiniteWellBasis
from spectral_packet_engine.domain import coerce_tensor

Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# Commutators
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CommutatorResult:
    """Full commutator analysis of two operators."""

    commutator: Tensor
    anticommutator: Tensor
    trace_commutator: Tensor
    frobenius_norm: Tensor


def commutator(A: Tensor, B: Tensor) -> Tensor:
    """[A, B] = AB - BA."""
    return A @ B - B @ A


def anticommutator(A: Tensor, B: Tensor) -> Tensor:
    """{A, B} = AB + BA."""
    return A @ B + B @ A


def compute_commutator(A: Tensor, B: Tensor) -> CommutatorResult:
    """Full commutator analysis of two operator matrices."""
    A = coerce_tensor(A, dtype=torch.float64)
    B = coerce_tensor(B, dtype=torch.float64)

    comm = commutator(A, B)
    anti = anticommutator(A, B)
    tr = torch.trace(comm)
    frob = torch.linalg.norm(comm, ord="fro")

    return CommutatorResult(
        commutator=comm,
        anticommutator=anti,
        trace_commutator=tr,
        frobenius_norm=frob,
    )


# ---------------------------------------------------------------------------
# Expectation values and uncertainty
# ---------------------------------------------------------------------------

def expectation_value(operator: Tensor, state: Tensor) -> Tensor:
    r"""<psi|O|psi> = psi^+ O psi.

    Parameters
    ----------
    operator:
        Hermitian operator matrix of shape ``(N, N)``.
    state:
        Coefficient vector of shape ``(N,)``.
    """
    state = coerce_tensor(state, dtype=torch.complex128)
    O = coerce_tensor(operator).to(dtype=torch.complex128)
    return (torch.conj(state) @ O @ state).real


def variance(operator: Tensor, state: Tensor) -> Tensor:
    r"""Var(O) = <O^2> - <O>^2."""
    O = coerce_tensor(operator).to(dtype=torch.complex128)
    state = coerce_tensor(state, dtype=torch.complex128)

    mean = expectation_value(O, state)
    O2 = O @ O
    mean_sq = expectation_value(O2, state)
    return mean_sq - mean ** 2


@dataclass(frozen=True, slots=True)
class GeneralizedUncertainty:
    """Robertson uncertainty relation analysis."""

    delta_A: Tensor
    delta_B: Tensor
    product: Tensor
    lower_bound: Tensor
    satisfies_bound: bool


def generalized_uncertainty(
    A: Tensor,
    B: Tensor,
    state: Tensor,
) -> GeneralizedUncertainty:
    r"""Robertson uncertainty relation: Delta A * Delta B >= (1/2)|<[A,B]>|.

    Parameters
    ----------
    A, B:
        Hermitian operator matrices of shape ``(N, N)``.
    state:
        Coefficient vector of shape ``(N,)``.
    """
    A = coerce_tensor(A).to(dtype=torch.complex128)
    B = coerce_tensor(B).to(dtype=torch.complex128)
    state = coerce_tensor(state, dtype=torch.complex128)

    var_A = variance(A, state)
    var_B = variance(B, state)
    delta_A = torch.sqrt(torch.clamp(var_A, min=0.0))
    delta_B = torch.sqrt(torch.clamp(var_B, min=0.0))
    product = delta_A * delta_B

    comm_AB = commutator(A, B)
    # Use full complex expectation (the commutator of Hermitian operators
    # is anti-Hermitian, so its expectation is purely imaginary).
    exp_comm_complex = torch.conj(state) @ comm_AB @ state
    lower = 0.5 * torch.abs(exp_comm_complex)

    satisfies = bool((product >= lower - 1e-12).item())

    return GeneralizedUncertainty(
        delta_A=delta_A,
        delta_B=delta_B,
        product=product,
        lower_bound=lower,
        satisfies_bound=satisfies,
    )


# ---------------------------------------------------------------------------
# Baker--Campbell--Hausdorff
# ---------------------------------------------------------------------------

def baker_campbell_hausdorff(A: Tensor, B: Tensor, *, order: int = 4) -> Tensor:
    r"""BCH expansion: ln(e^A e^B) truncated at the given order.

    Order 1:  A + B
    Order 2:  + (1/2) [A, B]
    Order 3:  + (1/12)([A,[A,B]] - [B,[A,B]])
    Order 4:  - (1/24)[B,[A,[A,B]]]

    These follow from the Dynkin form of the BCH series.
    """
    A = coerce_tensor(A, dtype=torch.float64)
    B = coerce_tensor(B, dtype=torch.float64)

    if order < 1:
        raise ValueError("order must be at least 1")

    result = A + B

    if order >= 2:
        AB = commutator(A, B)
        result = result + 0.5 * AB

    if order >= 3:
        A_AB = commutator(A, AB)  # [A, [A, B]]
        B_AB = commutator(B, AB)  # [B, [A, B]]
        result = result + (1.0 / 12.0) * (A_AB - B_AB)

    if order >= 4:
        # -1/24 [B, [A, [A, B]]]
        B_A_AB = commutator(B, A_AB)
        result = result - (1.0 / 24.0) * B_A_AB

    return result


# ---------------------------------------------------------------------------
# Position and momentum operator matrices
# ---------------------------------------------------------------------------

def position_operator_matrix(basis: InfiniteWellBasis) -> Tensor:
    r"""X_{mn} = <m|x|n> position operator in the infinite-well sine basis.

    Diagonal: <n|x|n> = L/2 + a  (centre of the well in physical coords)
    Off-diagonal (m+n odd):
        <m|x|n> = (4L/pi^2) * (-1) * m*n / (m^2 - n^2)^2
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
    parity = (m_int[:, None] + m_int[None, :]) % 2  # 1 if odd
    sign_factor = parity.to(dtype=dtype) * (-2.0)

    off_diag = (4 * L / torch.pi ** 2) * sign_factor * m * n / safe_diff_sq ** 2
    off_diag = torch.where(diff_sq != 0, off_diag, torch.zeros_like(off_diag))

    diag = torch.full((N,), (L / 2).item(), dtype=dtype, device=device)
    X = off_diag + torch.diag(diag)
    # Shift to physical coordinates
    X = X + a * torch.eye(N, dtype=dtype, device=device)
    return X


def momentum_operator_matrix(basis: InfiniteWellBasis) -> Tensor:
    r"""P_{mn} = <m|p|n> = -i * P_real_{mn} momentum operator in the sine basis.

    The real-valued antisymmetric matrix P_real is:

        P_real_{mn} = hbar * 2*m*n / (L*(m^2 - n^2)) * [1-(-1)^{m+n}]

    which is nonzero only when m+n is odd.  The actual matrix element
    is <m|p|n> = -i * P_real_{mn}, so we return the full complex matrix.
    """
    N = basis.num_modes
    L = basis.domain.length
    hbar = basis.domain.hbar
    dtype = basis.domain.real_dtype
    device = basis.domain.device

    modes = torch.arange(1, N + 1, dtype=dtype, device=device)
    m = modes[:, None]
    n = modes[None, :]

    diff_sq = m ** 2 - n ** 2
    safe_diff = torch.where(diff_sq != 0, diff_sq, torch.ones_like(diff_sq))

    m_int = torch.arange(1, N + 1, device=device)
    parity = (m_int[:, None] + m_int[None, :]) % 2
    sign_factor = 2.0 * parity.to(dtype=dtype)

    P_real = hbar * 2 * m * n / (L * safe_diff) * sign_factor
    P_real = torch.where(diff_sq != 0, P_real, torch.zeros_like(P_real))

    # Full complex momentum matrix: P = -i * P_real
    P_complex = (-1j * P_real).to(dtype=torch.complex128)
    return P_complex


# ---------------------------------------------------------------------------
# Ladder operators
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LadderOperators:
    """Annihilation, creation, and number operators in spectral basis."""

    a: Tensor
    a_dag: Tensor
    number: Tensor


def harmonic_ladder_operators(
    basis: InfiniteWellBasis,
    omega: float = 1.0,
) -> LadderOperators:
    r"""Ladder operators for the harmonic oscillator in the sine basis.

    a   = sqrt(m*omega/(2*hbar)) X + i / sqrt(2*m*omega*hbar) P
    a^+ = sqrt(m*omega/(2*hbar)) X - i / sqrt(2*m*omega*hbar) P

    where X and P are the position and momentum operator matrices in
    the infinite-well basis.  Note that P is already the full complex
    matrix (P = -i * P_real).
    """
    m = basis.domain.mass.item()
    hbar = basis.domain.hbar.item()

    X = position_operator_matrix(basis).to(dtype=torch.complex128)
    P = momentum_operator_matrix(basis)  # already complex128

    coeff_x = (m * omega / (2.0 * hbar)) ** 0.5
    coeff_p = 1.0 / (2.0 * m * omega * hbar) ** 0.5

    a = coeff_x * X + 1j * coeff_p * P
    a_dag = coeff_x * X - 1j * coeff_p * P

    number = a_dag @ a

    return LadderOperators(
        a=a,
        a_dag=a_dag,
        number=number,
    )


__all__ = [
    "CommutatorResult",
    "GeneralizedUncertainty",
    "LadderOperators",
    "anticommutator",
    "baker_campbell_hausdorff",
    "commutator",
    "compute_commutator",
    "expectation_value",
    "generalized_uncertainty",
    "harmonic_ladder_operators",
    "momentum_operator_matrix",
    "position_operator_matrix",
    "variance",
]
