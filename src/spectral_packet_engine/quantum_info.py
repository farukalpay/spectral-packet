"""Quantum information measures -- Fisher information, entanglement, mutual information.

This module provides publishable-quality implementations of core quantum
information quantities:

Quantum Fisher Information (QFI):
    F_Q[rho, A] = 2 sum_{m,n: p_m+p_n>0} (p_m - p_n)^2 |<m|A|n>|^2 / (p_m + p_n)
    For pure states:  F_Q = 4 Var(A) = 4(<A^2> - <A>^2)
    Cramer-Rao bound: Var(theta) >= 1 / (nu * F_Q)

Entanglement entropy (Schmidt decomposition):
    S(rho_A) = -sum_k lambda_k ln(lambda_k)
    where lambda_k are the squared singular values of the coefficient matrix.

Concurrence (Wootters formula for 2-qubit systems):
    C(rho) = max(0, lambda_1 - lambda_2 - lambda_3 - lambda_4)

Quantum mutual information:
    I(A:B) = S(rho_A) + S(rho_B) - S(rho_AB)

Quantum channels (Kraus representation):
    rho_out = sum_i K_i rho K_i^dag
    with completeness: sum_i K_i^dag K_i = I

Relative entropy:
    S(rho || sigma) = Tr(rho (ln rho - ln sigma))
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from spectral_packet_engine.domain import coerce_tensor

Tensor = torch.Tensor

_CDTYPE = torch.complex128
_RDTYPE = torch.float64
_EPS = 1e-15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _von_neumann_entropy(rho: Tensor) -> Tensor:
    """S = -Tr(rho ln rho) via eigendecomposition."""
    eigenvalues = torch.linalg.eigvalsh(rho).real
    mask = eigenvalues > _EPS
    lam = eigenvalues[mask]
    return -torch.sum(lam * torch.log(lam))


def _partial_trace(rho: Tensor, dims: tuple[int, int], trace_out: int) -> Tensor:
    """Partial trace of a bipartite density matrix.

    Parameters
    ----------
    rho : Tensor
        Shape ``(d_A * d_B, d_A * d_B)``.
    dims : tuple[int, int]
        ``(d_A, d_B)``.
    trace_out : int
        0 to trace out A (returns rho_B), 1 to trace out B (returns rho_A).
    """
    d_A, d_B = dims
    rho_4 = rho.reshape(d_A, d_B, d_A, d_B)
    if trace_out == 0:
        return torch.einsum("abac->bc", rho_4)
    return torch.einsum("abcb->ac", rho_4)


def _state_fidelity(rho: Tensor, sigma: Tensor) -> Tensor:
    """Quantum state fidelity F(rho, sigma) = [Tr sqrt(sqrt(rho) sigma sqrt(rho))]^2."""
    eigenvalues, U = torch.linalg.eigh(rho)
    sqrt_lam = torch.sqrt(eigenvalues.real.clamp(min=0.0)).to(rho.dtype)
    sqrt_rho = U @ torch.diag(sqrt_lam) @ U.conj().T
    M = sqrt_rho @ sigma @ sqrt_rho
    M_eigvals = torch.linalg.eigvalsh(M).real.clamp(min=0.0)
    return torch.sum(torch.sqrt(M_eigvals)) ** 2


# ---------------------------------------------------------------------------
# Quantum Fisher Information
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class QuantumFisherResult:
    """Result of quantum Fisher information calculation.

    Attributes
    ----------
    fisher_information : Tensor
        Scalar -- F_Q[rho, A].
    cramer_rao_bound : Tensor
        Scalar -- 1 / F_Q, the minimum variance for unbiased estimation
        of the parameter conjugate to A (for a single measurement).
    generator : str
        Name or description of the generator observable.
    """

    fisher_information: Tensor
    cramer_rao_bound: Tensor
    generator: str


def quantum_fisher_information(
    rho: Tensor,
    observable: Tensor,
) -> QuantumFisherResult:
    """Quantum Fisher Information for parameter estimation.

    F_Q[rho, A] = 2 sum_{m,n: p_m + p_n > 0}
                  (p_m - p_n)^2 |<m|A|n>|^2 / (p_m + p_n)

    For pure states this reduces to F_Q = 4 Var(A).

    Parameters
    ----------
    rho : Tensor
        Shape ``(d, d)`` -- density matrix.
    observable : Tensor
        Shape ``(d, d)`` -- Hermitian observable (generator).

    Returns
    -------
    QuantumFisherResult
    """
    rho = coerce_tensor(rho, dtype=_CDTYPE)
    A = coerce_tensor(observable, dtype=_CDTYPE)
    d = rho.shape[0]

    if rho.shape != (d, d) or A.shape != (d, d):
        raise ValueError("rho and observable must be square matrices of the same size")

    # Eigendecomposition of rho
    eigenvalues, eigenvectors = torch.linalg.eigh(rho)
    p = eigenvalues.real  # (d,)

    # Matrix elements of A in the eigenbasis: A_mn = <m|A|n>
    A_eigenbasis = eigenvectors.conj().T @ A @ eigenvectors  # (d, d)

    F_Q = torch.tensor(0.0, dtype=_RDTYPE)
    for m in range(d):
        for n in range(d):
            denom = p[m] + p[n]
            if denom.item() < _EPS:
                continue
            diff = p[m] - p[n]
            A_mn_abs2 = (A_eigenbasis[m, n] * A_eigenbasis[m, n].conj()).real
            F_Q = F_Q + 2.0 * diff * diff * A_mn_abs2 / denom

    # Cramer-Rao bound
    safe_FQ = torch.where(F_Q > _EPS, F_Q, torch.ones_like(F_Q))
    crb = 1.0 / safe_FQ

    return QuantumFisherResult(
        fisher_information=F_Q,
        cramer_rao_bound=crb,
        generator="observable",
    )


# ---------------------------------------------------------------------------
# Entanglement entropy via Schmidt decomposition
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EntanglementResult:
    """Entanglement analysis of a bipartite pure state.

    Attributes
    ----------
    entanglement_entropy : Tensor
        Scalar -- von Neumann entropy of the reduced density matrix S(rho_A).
    schmidt_coefficients : Tensor
        Shape ``(min(d_A, d_B),)`` -- Schmidt coefficients (squared
        singular values, normalised to sum to 1).
    schmidt_rank : int
        Number of non-negligible Schmidt coefficients.
    concurrence : Tensor | None
        Concurrence for 2-qubit systems; None otherwise.
    is_entangled : bool
        True if the Schmidt rank exceeds 1.
    """

    entanglement_entropy: Tensor
    schmidt_coefficients: Tensor
    schmidt_rank: int
    concurrence: Tensor | None
    is_entangled: bool


def entanglement_entropy(
    state: Tensor,
    dims: tuple[int, int],
) -> EntanglementResult:
    """Entanglement entropy via Schmidt decomposition.

    For a bipartite pure state |psi> in H_A (x) H_B:
    1. Reshape the coefficient vector as a matrix C_{ij} of shape (d_A, d_B).
    2. SVD: C = U Sigma V^dag.
    3. Schmidt coefficients: lambda_k = sigma_k^2 (normalised).
    4. Entanglement entropy: S = -sum_k lambda_k ln(lambda_k).

    Parameters
    ----------
    state : Tensor
        Shape ``(d_A * d_B,)`` -- coefficient vector of the pure state.
    dims : tuple[int, int]
        ``(d_A, d_B)`` -- dimensions of the two subsystems.

    Returns
    -------
    EntanglementResult
    """
    psi = coerce_tensor(state, dtype=_CDTYPE)
    d_A, d_B = dims

    if psi.ndim != 1:
        raise ValueError("state must be a 1D tensor")
    if psi.shape[0] != d_A * d_B:
        raise ValueError(
            f"state length {psi.shape[0]} does not match dims ({d_A}, {d_B})"
        )

    # Reshape into coefficient matrix and compute SVD
    C = psi.reshape(d_A, d_B)
    U, sigma, Vh = torch.linalg.svd(C, full_matrices=False)

    # Schmidt coefficients: lambda_k = sigma_k^2, normalised
    sigma_sq = (sigma.real ** 2)
    total = sigma_sq.sum()
    if total.item() > _EPS:
        schmidt = sigma_sq / total
    else:
        schmidt = sigma_sq

    # Entanglement entropy
    mask = schmidt > _EPS
    lam = schmidt[mask]
    S = -torch.sum(lam * torch.log(lam))

    # Schmidt rank
    rank = int(mask.sum().item())

    # Concurrence for 2-qubit systems
    conc: Tensor | None = None
    if d_A == 2 and d_B == 2:
        # For a pure bipartite state, concurrence = 2|det(C)|
        # which equals 2 * sigma_0 * sigma_1 (the two singular values)
        if sigma.shape[0] >= 2:
            conc = 2.0 * sigma[0].real * sigma[1].real / total.sqrt()
        else:
            conc = torch.tensor(0.0, dtype=_RDTYPE)

    return EntanglementResult(
        entanglement_entropy=S,
        schmidt_coefficients=schmidt,
        schmidt_rank=rank,
        concurrence=conc,
        is_entangled=rank > 1,
    )


# ---------------------------------------------------------------------------
# Concurrence (Wootters formula for 2-qubit density matrices)
# ---------------------------------------------------------------------------

def concurrence(rho: Tensor) -> Tensor:
    """Concurrence for a 2-qubit density matrix (Wootters formula).

    C(rho) = max(0, lambda_1 - lambda_2 - lambda_3 - lambda_4)

    where lambda_i are the eigenvalues (in descending order) of the matrix
    R = sqrt(sqrt(rho) rho_tilde sqrt(rho))  and
    rho_tilde = (sigma_y (x) sigma_y) rho* (sigma_y (x) sigma_y).

    Parameters
    ----------
    rho : Tensor
        Shape ``(4, 4)`` -- density matrix of a two-qubit system.

    Returns
    -------
    Tensor
        Scalar -- concurrence in [0, 1].
    """
    rho = coerce_tensor(rho, dtype=_CDTYPE)
    if rho.shape != (4, 4):
        raise ValueError("concurrence requires a (4, 4) density matrix")

    # sigma_y
    sigma_y = torch.tensor([[0.0, -1j], [1j, 0.0]], dtype=_CDTYPE)
    # sigma_y (x) sigma_y
    yy = torch.kron(sigma_y, sigma_y)

    # rho_tilde = (sigma_y (x) sigma_y) rho* (sigma_y (x) sigma_y)
    rho_tilde = yy @ rho.conj() @ yy

    # sqrt(rho) via eigendecomposition
    eigenvalues, U = torch.linalg.eigh(rho)
    sqrt_lam = torch.sqrt(eigenvalues.real.clamp(min=0.0)).to(_CDTYPE)
    sqrt_rho = U @ torch.diag(sqrt_lam) @ U.conj().T

    # R_matrix = sqrt(rho) @ rho_tilde @ sqrt(rho)
    R_matrix = sqrt_rho @ rho_tilde @ sqrt_rho

    # Eigenvalues of R_matrix (should be non-negative real)
    R_eigvals = torch.linalg.eigvalsh(R_matrix).real.clamp(min=0.0)
    # lambda_i = sqrt(eigenvalues of R_matrix), sorted descending
    lambdas = torch.sqrt(R_eigvals).sort(descending=True).values

    # C = max(0, lambda_1 - lambda_2 - lambda_3 - lambda_4)
    C = lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3]
    return torch.clamp(C, min=0.0)


# ---------------------------------------------------------------------------
# Quantum mutual information
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MutualInformationResult:
    """Quantum mutual information for a bipartite system.

    Attributes
    ----------
    mutual_information : Tensor
        Scalar -- I(A:B) = S(rho_A) + S(rho_B) - S(rho_AB).
    entropy_A : Tensor
        Scalar -- von Neumann entropy of rho_A.
    entropy_B : Tensor
        Scalar -- von Neumann entropy of rho_B.
    entropy_AB : Tensor
        Scalar -- von Neumann entropy of rho_AB.
    classical_bound : Tensor
        Scalar -- min(2 S_A, 2 S_B).  Quantum mutual information can
        exceed the classical bound for entangled states.
    """

    mutual_information: Tensor
    entropy_A: Tensor
    entropy_B: Tensor
    entropy_AB: Tensor
    classical_bound: Tensor


def quantum_mutual_information(
    rho_AB: Tensor,
    dims: tuple[int, int],
) -> MutualInformationResult:
    """Quantum mutual information I(A:B) = S(rho_A) + S(rho_B) - S(rho_AB).

    Parameters
    ----------
    rho_AB : Tensor
        Shape ``(d_A * d_B, d_A * d_B)`` -- density matrix of the joint system.
    dims : tuple[int, int]
        ``(d_A, d_B)`` -- dimensions of subsystems A and B.

    Returns
    -------
    MutualInformationResult
    """
    rho_AB = coerce_tensor(rho_AB, dtype=_CDTYPE)
    d_A, d_B = dims

    if rho_AB.shape != (d_A * d_B, d_A * d_B):
        raise ValueError(
            f"rho_AB shape {rho_AB.shape} incompatible with dims ({d_A}, {d_B})"
        )

    rho_A = _partial_trace(rho_AB, dims, trace_out=1)
    rho_B = _partial_trace(rho_AB, dims, trace_out=0)

    S_A = _von_neumann_entropy(rho_A)
    S_B = _von_neumann_entropy(rho_B)
    S_AB = _von_neumann_entropy(rho_AB)

    I_AB = S_A + S_B - S_AB
    classical_bound = torch.minimum(2.0 * S_A, 2.0 * S_B)

    return MutualInformationResult(
        mutual_information=I_AB,
        entropy_A=S_A,
        entropy_B=S_B,
        entropy_AB=S_AB,
        classical_bound=classical_bound,
    )


# ---------------------------------------------------------------------------
# Simple entropy measures
# ---------------------------------------------------------------------------

def linear_entropy(rho: Tensor) -> Tensor:
    """Linear entropy: S_L = 1 - Tr(rho^2).

    A computationally cheaper approximation to the von Neumann entropy.
    S_L = 0 for pure states, S_L = 1 - 1/d for maximally mixed states.

    Parameters
    ----------
    rho : Tensor
        Shape ``(d, d)`` -- density matrix.

    Returns
    -------
    Tensor
        Scalar -- linear entropy.
    """
    rho = coerce_tensor(rho, dtype=_CDTYPE)
    rho2 = rho @ rho
    return (1.0 - torch.trace(rho2).real)


def relative_entropy(rho: Tensor, sigma: Tensor) -> Tensor:
    """Quantum relative entropy (Kullback-Leibler divergence).

    S(rho || sigma) = Tr(rho (ln rho - ln sigma))

    Uses eigendecomposition for the matrix logarithms.
    Requires the support of rho to be contained in the support of sigma
    (otherwise the relative entropy is +infinity).

    Parameters
    ----------
    rho : Tensor
        Shape ``(d, d)`` -- density matrix.
    sigma : Tensor
        Shape ``(d, d)`` -- density matrix.

    Returns
    -------
    Tensor
        Scalar -- relative entropy (non-negative).
    """
    rho = coerce_tensor(rho, dtype=_CDTYPE)
    sigma = coerce_tensor(sigma, dtype=_CDTYPE)

    if rho.shape != sigma.shape:
        raise ValueError("rho and sigma must have the same shape")

    # Eigendecomposition of rho
    evals_rho, evecs_rho = torch.linalg.eigh(rho)
    p = evals_rho.real
    mask_rho = p > _EPS
    ln_p = torch.zeros_like(p)
    ln_p[mask_rho] = torch.log(p[mask_rho])
    ln_rho = evecs_rho @ torch.diag(ln_p.to(_CDTYPE)) @ evecs_rho.conj().T

    # Eigendecomposition of sigma
    evals_sigma, evecs_sigma = torch.linalg.eigh(sigma)
    q = evals_sigma.real
    mask_sigma = q > _EPS
    ln_q = torch.zeros_like(q)
    ln_q[mask_sigma] = torch.log(q[mask_sigma])

    # Check support condition: if rho has support where sigma does not,
    # the relative entropy is infinite.
    for i in range(p.shape[0]):
        if p[i].item() > _EPS:
            # Project rho eigenstate onto sigma eigenbasis
            evec_rho_i = evecs_rho[:, i]
            overlaps = (evecs_sigma.conj().T @ evec_rho_i).abs() ** 2
            sigma_support_overlap = (overlaps[mask_sigma]).sum()
            if sigma_support_overlap.item() < 1.0 - 1e-6:
                return torch.tensor(float("inf"), dtype=_RDTYPE)

    ln_sigma = evecs_sigma @ torch.diag(ln_q.to(_CDTYPE)) @ evecs_sigma.conj().T

    # S(rho || sigma) = Tr(rho (ln_rho - ln_sigma))
    result = torch.trace(rho @ (ln_rho - ln_sigma)).real
    # Clamp to non-negative (may be slightly negative due to numerics)
    return torch.clamp(result, min=0.0)


def quantum_conditional_entropy(
    rho_AB: Tensor,
    dims: tuple[int, int],
) -> Tensor:
    """Quantum conditional entropy S(A|B) = S(rho_AB) - S(rho_B).

    Unlike the classical conditional entropy, this quantity can be negative
    for entangled states -- a signature of quantum correlations.

    Parameters
    ----------
    rho_AB : Tensor
        Shape ``(d_A * d_B, d_A * d_B)`` -- joint density matrix.
    dims : tuple[int, int]
        ``(d_A, d_B)`` -- subsystem dimensions.

    Returns
    -------
    Tensor
        Scalar -- conditional entropy.
    """
    rho_AB = coerce_tensor(rho_AB, dtype=_CDTYPE)
    d_A, d_B = dims

    if rho_AB.shape != (d_A * d_B, d_A * d_B):
        raise ValueError(
            f"rho_AB shape {rho_AB.shape} incompatible with dims ({d_A}, {d_B})"
        )

    rho_B = _partial_trace(rho_AB, dims, trace_out=0)
    S_AB = _von_neumann_entropy(rho_AB)
    S_B = _von_neumann_entropy(rho_B)

    return S_AB - S_B


# ---------------------------------------------------------------------------
# Quantum channels (Kraus representation)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class QuantumChannelResult:
    """Result of applying a quantum channel.

    Attributes
    ----------
    output_state : Tensor
        Shape ``(d, d)`` -- output density matrix rho_out.
    fidelity_with_input : Tensor
        Scalar -- fidelity F(rho_in, rho_out).
    entropy_change : Tensor
        Scalar -- S(rho_out) - S(rho_in).
    """

    output_state: Tensor
    fidelity_with_input: Tensor
    entropy_change: Tensor


def apply_quantum_channel(
    rho: Tensor,
    kraus_operators: list[Tensor],
) -> QuantumChannelResult:
    """Apply a quantum channel via the Kraus representation.

    rho_out = sum_i K_i rho K_i^dag

    Completeness condition: sum_i K_i^dag K_i = I.

    Parameters
    ----------
    rho : Tensor
        Shape ``(d, d)`` -- input density matrix.
    kraus_operators : list[Tensor]
        Each element has shape ``(d, d)`` -- Kraus operators.

    Returns
    -------
    QuantumChannelResult
    """
    rho = coerce_tensor(rho, dtype=_CDTYPE)
    d = rho.shape[0]

    if not kraus_operators:
        raise ValueError("at least one Kraus operator is required")

    rho_out = torch.zeros(d, d, dtype=_CDTYPE)
    for K in kraus_operators:
        K_t = coerce_tensor(K, dtype=_CDTYPE)
        rho_out = rho_out + K_t @ rho @ K_t.conj().T

    S_in = _von_neumann_entropy(rho)
    S_out = _von_neumann_entropy(rho_out)
    fid = _state_fidelity(rho, rho_out)

    return QuantumChannelResult(
        output_state=rho_out,
        fidelity_with_input=fid,
        entropy_change=S_out - S_in,
    )


def depolarizing_channel(dim: int, p: float) -> list[Tensor]:
    """Kraus operators for the depolarizing channel.

    rho -> (1 - p) rho + p I/d

    The channel replaces the state with the maximally mixed state with
    probability p.

    For a qubit (dim=2), the Kraus decomposition uses the identity and
    scaled Pauli matrices.  For general dimension d, we use:
        K_0 = sqrt(1 - p + p/d) I
        K_{ij} = sqrt(p/d) |i><j|   for i != j
        K_{ii} = sqrt(p/d) (|i><i| - delta_{i0}|0><0|)  ... (generalised Paulis)

    A simpler equivalent decomposition uses d^2 operators:
        K_0 = sqrt(1 - p(d^2-1)/d^2) I
        K_{a} = sqrt(p/d^2) G_a    (a = 1, ..., d^2-1)
    where {G_a} are the generalised Gell-Mann matrices plus identity.

    For simplicity we use the two-operator form valid for any dimension:
        K_0 = sqrt(1 - p) I
        rho_out = (1-p) rho + p I/d
    achieved by treating it as a mixture.  Here we use the standard form:
        K_0 = sqrt(1 - p + p/d) I  (shrunk identity)
    plus off-diagonal generators.

    For practical purposes we implement the exact qubit case and a
    general-dimension variant.

    Parameters
    ----------
    dim : int
        Hilbert space dimension.
    p : float
        Depolarizing probability in [0, 1].

    Returns
    -------
    list[Tensor]
        Kraus operators.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")

    if dim == 2:
        # Qubit depolarizing channel: standard Pauli form
        # rho -> (1-p)rho + (p/3)(sigma_x rho sigma_x + sigma_y rho sigma_y + sigma_z rho sigma_z)
        # K_0 = sqrt(1 - 3p/4) I,  K_i = sqrt(p/4) sigma_i
        I2 = torch.eye(2, dtype=_CDTYPE)
        sx = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=_CDTYPE)
        sy = torch.tensor([[0.0, -1j], [1j, 0.0]], dtype=_CDTYPE)
        sz = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=_CDTYPE)

        # Note: (1-p)rho + p*I/2 = (1-p)rho + (p/4)(rho + sx rho sx + sy rho sy + sz rho sz)
        # = (1 - 3p/4)rho + (p/4)(sx rho sx + sy rho sy + sz rho sz)
        coeff_0 = (1.0 - 3.0 * p / 4.0)
        if coeff_0 < 0:
            coeff_0 = 0.0
        K0 = torch.sqrt(torch.tensor(coeff_0, dtype=_RDTYPE)).to(_CDTYPE) * I2
        coeff_p = p / 4.0
        scale = torch.sqrt(torch.tensor(coeff_p, dtype=_RDTYPE)).to(_CDTYPE)
        return [K0, scale * sx, scale * sy, scale * sz]

    # General dimension: use the decomposition
    # rho -> (1-p) rho + (p/d) I
    # = (1 - p + p/d) |projection onto rho's diagonal> ... this is tricky.
    #
    # Correct Kraus decomposition for general d:
    # K_0 = sqrt(1 - p(1 - 1/d^2)) I    [note: this is for channel rho -> (1-p)rho + p*I/d]
    # Wait -- the standard form uses d^2 Kraus operators.
    #
    # Alternative: use operator-sum form with basis {E_{ij}} where E_{ij} = |i><j|.
    # rho -> (1-p) rho + p/d I = sum_ij K_{ij} rho K_{ij}^dag
    # with K_0 = sqrt(1-p) I and K_{ij} = sqrt(p/d) |i><j| for all i,j.
    # Check: sum K^dag K = (1-p)I + (p/d) sum_{ij} |j><i||i><j| = (1-p)I + (p/d)*d*I = I.  Correct.
    operators: list[Tensor] = []
    K0 = torch.sqrt(torch.tensor(1.0 - p, dtype=_RDTYPE)).to(_CDTYPE) * torch.eye(dim, dtype=_CDTYPE)
    operators.append(K0)

    scale = torch.sqrt(torch.tensor(p / dim, dtype=_RDTYPE)).to(_CDTYPE)
    for i in range(dim):
        for j in range(dim):
            E_ij = torch.zeros(dim, dim, dtype=_CDTYPE)
            E_ij[i, j] = 1.0
            operators.append(scale * E_ij)

    return operators


def amplitude_damping_channel(gamma: float) -> list[Tensor]:
    """Kraus operators for the amplitude damping channel (qubit).

    Models spontaneous emission / energy relaxation:
        K_0 = [[1, 0], [0, sqrt(1 - gamma)]]
        K_1 = [[0, sqrt(gamma)], [0, 0]]

    K_0^dag K_0 + K_1^dag K_1 = I (completeness).

    The ground state |0> is invariant; the excited state |1> decays to
    |0> with probability gamma.

    Parameters
    ----------
    gamma : float
        Decay probability in [0, 1].

    Returns
    -------
    list[Tensor]
        Two Kraus operators [K_0, K_1].
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    import math

    K0 = torch.tensor(
        [[1.0, 0.0], [0.0, math.sqrt(1.0 - gamma)]],
        dtype=_CDTYPE,
    )
    K1 = torch.tensor(
        [[0.0, math.sqrt(gamma)], [0.0, 0.0]],
        dtype=_CDTYPE,
    )
    return [K0, K1]


__all__ = [
    "QuantumFisherResult",
    "EntanglementResult",
    "MutualInformationResult",
    "QuantumChannelResult",
    "quantum_fisher_information",
    "entanglement_entropy",
    "concurrence",
    "quantum_mutual_information",
    "linear_entropy",
    "relative_entropy",
    "quantum_conditional_entropy",
    "apply_quantum_channel",
    "depolarizing_channel",
    "amplitude_damping_channel",
]
