"""Many-body quantum mechanics: Fock space, second quantization, and Hamiltonians.

Constructs the full configuration-interaction (CI) Hamiltonian for a system
of *N* identical fermions in *M* single-particle orbitals.  Everything is
built with **sparse matrices** so that the O(N·k·nnz) iterative solvers
from :mod:`sparse_eigensolver` can be applied directly.

Key components
--------------
- **Fock space enumeration**: all valid *N*-particle configurations
  |n₁, n₂, …, n_M⟩ with antisymmetry (Slater determinants).
- **Creation / annihilation operators**: â†_i, â_j in the full 2^M Fock space.
- **Number-conserving excitation operators**: E_ij = â†_i â_j and
  e_ijkl = â†_i â†_j â_l â_k directly in the N-particle sector.
- **One-body and two-body integrals**: ⟨i|h|j⟩ and ⟨ij|V|kl⟩.
- **Many-body Hamiltonian assembly**: H = Σ h_ij â†_i â_j + ½ Σ V_ijkl â†_i â†_j â_l â_k
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product as iterproduct
from math import comb
from typing import Sequence

import torch
import numpy as np
import scipy.sparse as sp

Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# Fock space
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FockSpace:
    """Enumerated Fock space for N fermions in M orbitals.

    Attributes
    ----------
    num_orbitals : int
        Number of single-particle orbitals (M).
    num_particles : int
        Number of fermions (N).
    dim : int
        Hilbert space dimension C(M, N).
    configs : np.ndarray
        Shape ``(dim, M)`` — each row is an occupation-number vector.
    config_to_index : dict[tuple[int, ...], int]
        Reverse lookup: occupation tuple → basis index.
    """

    num_orbitals: int
    num_particles: int
    dim: int
    configs: np.ndarray
    config_to_index: dict[tuple[int, ...], int]


def build_fock_space(num_orbitals: int, num_particles: int) -> FockSpace:
    """Enumerate all N-particle Slater determinants in M orbitals.

    Parameters
    ----------
    num_orbitals : int
        Number of single-particle orbitals M.
    num_particles : int
        Number of fermions N (must satisfy N ≤ M).

    Returns
    -------
    FockSpace
    """
    if num_particles > num_orbitals:
        raise ValueError(
            f"num_particles ({num_particles}) cannot exceed "
            f"num_orbitals ({num_orbitals})"
        )
    if num_particles < 0 or num_orbitals < 1:
        raise ValueError("Need num_orbitals >= 1 and num_particles >= 0")

    dim = comb(num_orbitals, num_particles)
    configs = np.zeros((dim, num_orbitals), dtype=np.int8)
    config_to_index: dict[tuple[int, ...], int] = {}

    for idx, occupied in enumerate(combinations(range(num_orbitals), num_particles)):
        occ = [0] * num_orbitals
        for o in occupied:
            occ[o] = 1
        key = tuple(occ)
        configs[idx] = occ
        config_to_index[key] = idx

    return FockSpace(
        num_orbitals=num_orbitals,
        num_particles=num_particles,
        dim=dim,
        configs=configs,
        config_to_index=config_to_index,
    )


# ---------------------------------------------------------------------------
# Full Fock space (all particle-number sectors, dim = 2^M)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FullFockSpace:
    """Full Fock space spanning all particle-number sectors (dim = 2^M).

    Required for individual â†, â operators which change particle number.
    """

    num_orbitals: int
    dim: int
    config_to_index: dict[tuple[int, ...], int]
    configs: np.ndarray


def build_full_fock_space(num_orbitals: int) -> FullFockSpace:
    """Build the full 2^M-dimensional Fock space."""
    dim = 2 ** num_orbitals
    configs = np.zeros((dim, num_orbitals), dtype=np.int8)
    config_to_index: dict[tuple[int, ...], int] = {}

    for idx in range(dim):
        occ = tuple((idx >> q) & 1 for q in range(num_orbitals))
        configs[idx] = occ
        config_to_index[occ] = idx

    return FullFockSpace(
        num_orbitals=num_orbitals,
        dim=dim,
        config_to_index=config_to_index,
        configs=configs,
    )


# ---------------------------------------------------------------------------
# Fermionic sign convention
# ---------------------------------------------------------------------------

def _fermionic_sign(config, orbital: int) -> int:
    """Compute (-1)^(number of occupied orbitals before *orbital*)."""
    return (-1) ** int(np.sum(config[:orbital]))


# ---------------------------------------------------------------------------
# Full-space creation / annihilation operators (2^M × 2^M)
# ---------------------------------------------------------------------------

def creation_operator(fock: FullFockSpace, orbital: int) -> sp.csr_matrix:
    r"""Build â†_orbital in the full 2^M Fock space.

    â†_i |…, 0_i, …⟩ = (-1)^{Σ_{j<i} n_j} |…, 1_i, …⟩
    â†_i |…, 1_i, …⟩ = 0   (Pauli exclusion)
    """
    if orbital < 0 or orbital >= fock.num_orbitals:
        raise ValueError(f"orbital {orbital} out of range [0, {fock.num_orbitals})")

    rows, cols, vals = [], [], []
    for idx, config in enumerate(fock.configs):
        if config[orbital] == 1:
            continue
        new_config = list(config)
        new_config[orbital] = 1
        new_key = tuple(new_config)
        new_idx = fock.config_to_index.get(new_key)
        if new_idx is not None:
            sign = _fermionic_sign(config, orbital)
            rows.append(new_idx)
            cols.append(idx)
            vals.append(sign)

    return sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(fock.dim, fock.dim),
        dtype=np.float64,
    )


def annihilation_operator(fock: FullFockSpace, orbital: int) -> sp.csr_matrix:
    r"""Build â_orbital in the full 2^M Fock space.  â_i = (â†_i)^T."""
    return creation_operator(fock, orbital).T.tocsr()


def number_operator(fock: FockSpace | FullFockSpace, orbital: int) -> sp.csr_matrix:
    r"""Build n̂_i = â†_i â_i.  Diagonal in any Fock-space basis."""
    diag = np.array([float(c[orbital]) for c in fock.configs])
    return sp.diags(diag, format="csr")


# ---------------------------------------------------------------------------
# Number-conserving excitation operators (N-particle sector, dim = C(M,N))
# ---------------------------------------------------------------------------

def one_body_excitation(fock: FockSpace, i: int, j: int) -> sp.csr_matrix:
    r"""Build E_ij = â†_i â_j within the N-particle sector.

    This operator conserves particle number and is a well-defined
    (dim × dim) matrix on the fixed-N Fock space.

    Action on |config⟩:
    - If n_j = 0 → 0
    - If i = j → n_i |config⟩
    - If n_i = 1 and i ≠ j → 0 (Pauli exclusion)
    - Otherwise: remove j, add i, with fermionic signs.
    """
    rows, cols, vals = [], [], []
    for idx, config in enumerate(fock.configs):
        if config[j] == 0:
            continue

        if i == j:
            rows.append(idx)
            cols.append(idx)
            vals.append(1.0)
            continue

        if config[i] == 1:
            continue

        # Remove particle from j
        sign_j = _fermionic_sign(config, j)
        intermediate = config.copy()
        intermediate[j] = 0

        # Add particle to i
        sign_i = _fermionic_sign(intermediate, i)
        new_config = intermediate.copy()
        new_config[i] = 1

        new_key = tuple(new_config)
        new_idx = fock.config_to_index.get(new_key)
        if new_idx is not None:
            rows.append(new_idx)
            cols.append(idx)
            vals.append(float(sign_j * sign_i))

    return sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(fock.dim, fock.dim),
        dtype=np.float64,
    )


def two_body_excitation(
    fock: FockSpace, i: int, j: int, k: int, l: int,
) -> sp.csr_matrix:
    r"""Build e_ijkl = â†_i â†_j â_l â_k within the N-particle sector.

    Action: remove k, then l, then add j, then add i, with signs.
    Conserves particle number (removes 2, adds 2).
    """
    rows, cols, vals = [], [], []
    for idx, config in enumerate(fock.configs):
        # Step 1: â_k — remove k
        if config[k] == 0:
            continue
        sign_k = _fermionic_sign(config, k)
        c1 = config.copy()
        c1[k] = 0

        # Step 2: â_l — remove l
        if c1[l] == 0:
            continue
        sign_l = _fermionic_sign(c1, l)
        c2 = c1.copy()
        c2[l] = 0

        # Step 3: â†_j — add j
        if c2[j] == 1:
            continue
        sign_j = _fermionic_sign(c2, j)
        c3 = c2.copy()
        c3[j] = 1

        # Step 4: â†_i — add i
        if c3[i] == 1:
            continue
        sign_i = _fermionic_sign(c3, i)
        c4 = c3.copy()
        c4[i] = 1

        new_key = tuple(c4)
        new_idx = fock.config_to_index.get(new_key)
        if new_idx is not None:
            total_sign = sign_k * sign_l * sign_j * sign_i
            rows.append(new_idx)
            cols.append(idx)
            vals.append(float(total_sign))

    return sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(fock.dim, fock.dim),
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# One-body and two-body integrals
# ---------------------------------------------------------------------------

def compute_one_body_integrals(
    fock: FockSpace,
    eigensolver_result,
    domain,
) -> np.ndarray:
    r"""Compute ⟨i|h|j⟩ from an eigensolver result.

    For the spectral Galerkin eigenbasis, h_ij = ε_i δ_ij.
    """
    M = fock.num_orbitals
    h = np.zeros((M, M), dtype=np.float64)
    eigenvalues = eigensolver_result.eigenvalues.detach().cpu().numpy()
    for i in range(min(M, len(eigenvalues))):
        h[i, i] = eigenvalues[i]
    return h


def compute_two_body_integrals(
    fock: FockSpace,
    eigenstates: Tensor,
    grid: Tensor,
    interaction_fn=None,
) -> np.ndarray:
    r"""Compute ⟨ij|V|kl⟩ two-body contact integrals.

    V_ijkl = ∫ ψ_i(x) ψ_j(x) ψ_k(x) ψ_l(x) dx  (delta interaction).
    """
    M = fock.num_orbitals
    states = eigenstates.detach().cpu().numpy()
    x = grid.detach().cpu().numpy()
    n_available = states.shape[0]
    V = np.zeros((M, M, M, M), dtype=np.float64)

    for i in range(min(M, n_available)):
        for j in range(min(M, n_available)):
            for k in range(min(M, n_available)):
                for l in range(min(M, n_available)):
                    integrand = states[i] * states[j] * states[k] * states[l]
                    V[i, j, k, l] = np.trapezoid(integrand, x)

    return V


# ---------------------------------------------------------------------------
# Many-body Hamiltonian assembly
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ManyBodyHamiltonian:
    """Assembled many-body Hamiltonian in the Fock basis.

    Attributes
    ----------
    matrix : scipy.sparse.csr_matrix
        The full CI Hamiltonian, sparse, dimension C(M,N) × C(M,N).
    fock : FockSpace
        The underlying N-particle Fock space.
    one_body_integrals : np.ndarray
    two_body_integrals : np.ndarray or None
    """

    matrix: sp.csr_matrix
    fock: FockSpace
    one_body_integrals: np.ndarray
    two_body_integrals: np.ndarray | None


def build_many_body_hamiltonian(
    fock: FockSpace,
    one_body: np.ndarray,
    two_body: np.ndarray | None = None,
    interaction_strength: float = 1.0,
) -> ManyBodyHamiltonian:
    r"""Assemble the many-body Hamiltonian using number-conserving operators.

    .. math::

        \hat{H} = \sum_{ij} h_{ij}\, E_{ij}
                + \frac{1}{2} \sum_{ijkl} V_{ijkl}\, e_{ijkl}

    where E_ij = â†_i â_j and e_ijkl = â†_i â†_j â_l â_k are built
    directly within the N-particle sector.
    """
    M = fock.num_orbitals
    dim = fock.dim

    H = sp.csr_matrix((dim, dim), dtype=np.float64)

    # One-body: Σ h_ij E_ij
    for i in range(M):
        for j in range(M):
            if abs(one_body[i, j]) > 1e-15:
                H = H + one_body[i, j] * one_body_excitation(fock, i, j)

    # Two-body: ½ Σ V_ijkl e_ijkl
    if two_body is not None:
        for i in range(M):
            for j in range(M):
                for k in range(M):
                    for l in range(M):
                        coeff = 0.5 * interaction_strength * two_body[i, j, k, l]
                        if abs(coeff) > 1e-15:
                            H = H + coeff * two_body_excitation(fock, i, j, k, l)

    H = H.tocsr()
    H.eliminate_zeros()

    return ManyBodyHamiltonian(
        matrix=H,
        fock=fock,
        one_body_integrals=one_body,
        two_body_integrals=two_body,
    )


def hamiltonian_to_torch(
    mb_hamiltonian: ManyBodyHamiltonian,
    *,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> Tensor:
    """Convert a ManyBodyHamiltonian to a torch sparse CSR tensor."""
    H_csr = mb_hamiltonian.matrix.tocsr()
    H_csr.sort_indices()
    crow = torch.tensor(H_csr.indptr, dtype=torch.int64, device=device)
    col = torch.tensor(H_csr.indices, dtype=torch.int64, device=device)
    vals = torch.tensor(H_csr.data, dtype=dtype, device=device)
    return torch.sparse_csr_tensor(crow, col, vals, size=H_csr.shape, device=device)


__all__ = [
    "FockSpace",
    "FullFockSpace",
    "ManyBodyHamiltonian",
    "annihilation_operator",
    "build_fock_space",
    "build_full_fock_space",
    "build_many_body_hamiltonian",
    "compute_one_body_integrals",
    "compute_two_body_integrals",
    "creation_operator",
    "hamiltonian_to_torch",
    "number_operator",
    "one_body_excitation",
    "two_body_excitation",
]
