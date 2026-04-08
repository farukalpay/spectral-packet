"""Qubit mappings: second-quantization operators → Pauli Hamiltonians.

Maps fermionic creation/annihilation operators to qubit (Pauli) operators,
bridging continuous physics → quantum computing Hamiltonians.

Two standard transformations:

1. **Jordan-Wigner** — each orbital maps to one qubit with a parity string:
       â†_j → ½(σ_x - iσ_y) ⊗ σ_z ⊗ σ_z ⊗ …  (Z string on qubits 0..j-1)

2. **Bravyi-Kitaev** — logarithmic-depth parity encoding, fewer Z strings
   at the cost of a more complex update/parity/remainder set structure.

Both produce a sum of Pauli terms (a *qubit Hamiltonian*) suitable for
variational quantum eigensolvers (VQE), quantum phase estimation, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Pauli algebra
# ---------------------------------------------------------------------------

# 2×2 Pauli matrices (sparse)
_I2 = sp.eye(2, format="csr", dtype=complex)
_SX = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
_SY = sp.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
_SZ = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))

_PAULI = {"I": _I2, "X": _SX, "Y": _SY, "Z": _SZ}


@dataclass(frozen=True)
class PauliTerm:
    """A single Pauli string with a complex coefficient.

    Example: ``0.5 * X_0 Z_1 Z_2`` is stored as
    ``PauliTerm(coefficient=0.5, operators=("X", "Z", "Z"))``
    where indices are positional.
    """

    coefficient: complex
    operators: tuple[str, ...]

    @property
    def num_qubits(self) -> int:
        return len(self.operators)

    def to_matrix(self) -> sp.csr_matrix:
        """Build the full 2^n × 2^n sparse matrix for this term."""
        mat = sp.csr_matrix(np.array([[self.coefficient]], dtype=complex))
        for op in self.operators:
            mat = sp.kron(mat, _PAULI[op], format="csr")
        return mat

    def __repr__(self) -> str:
        ops = " ".join(
            f"{op}_{i}" for i, op in enumerate(self.operators) if op != "I"
        )
        return f"{self.coefficient:.4g} * {ops or 'I'}"


@dataclass(frozen=True)
class QubitHamiltonian:
    """Sum of Pauli terms representing a qubit Hamiltonian.

    Attributes
    ----------
    terms : tuple[PauliTerm, ...]
        Individual Pauli strings.
    num_qubits : int
        Number of qubits.
    """

    terms: tuple[PauliTerm, ...]
    num_qubits: int

    def to_matrix(self) -> sp.csr_matrix:
        """Build the full 2^n × 2^n Hamiltonian matrix."""
        dim = 2 ** self.num_qubits
        H = sp.csr_matrix((dim, dim), dtype=complex)
        for term in self.terms:
            H = H + term.to_matrix()
        return H

    @property
    def num_terms(self) -> int:
        return len(self.terms)

    def simplify(self, tol: float = 1e-12) -> QubitHamiltonian:
        """Combine like terms and drop near-zero coefficients."""
        combined: dict[tuple[str, ...], complex] = {}
        for t in self.terms:
            key = t.operators
            combined[key] = combined.get(key, 0.0) + t.coefficient
        new_terms = tuple(
            PauliTerm(coefficient=c, operators=ops)
            for ops, c in combined.items()
            if abs(c) > tol
        )
        return QubitHamiltonian(terms=new_terms, num_qubits=self.num_qubits)

    def to_dict(self) -> dict:
        return {
            "num_qubits": self.num_qubits,
            "num_terms": self.num_terms,
            "terms": [
                {"coefficient_real": t.coefficient.real,
                 "coefficient_imag": t.coefficient.imag,
                 "pauli_string": "".join(t.operators)}
                for t in self.terms
            ],
        }


# ---------------------------------------------------------------------------
# Jordan-Wigner transformation
# ---------------------------------------------------------------------------

def _jw_creation(orbital: int, num_qubits: int) -> list[PauliTerm]:
    r"""Jordan-Wigner encoding of â†_orbital.

    â†_j = ½(X_j - iY_j) ⊗ Z_{j-1} ⊗ Z_{j-2} ⊗ … ⊗ Z_0

    Returns two PauliTerms (X-part and Y-part).
    """
    # X part: coefficient +0.5
    ops_x = ["I"] * num_qubits
    for k in range(orbital):
        ops_x[k] = "Z"
    ops_x[orbital] = "X"

    # Y part: coefficient -0.5j
    ops_y = ["I"] * num_qubits
    for k in range(orbital):
        ops_y[k] = "Z"
    ops_y[orbital] = "Y"

    return [
        PauliTerm(coefficient=0.5, operators=tuple(ops_x)),
        PauliTerm(coefficient=-0.5j, operators=tuple(ops_y)),
    ]


def _jw_annihilation(orbital: int, num_qubits: int) -> list[PauliTerm]:
    r"""Jordan-Wigner encoding of â_orbital.

    â_j = ½(X_j + iY_j) ⊗ Z_{j-1} ⊗ … ⊗ Z_0
    """
    ops_x = ["I"] * num_qubits
    for k in range(orbital):
        ops_x[k] = "Z"
    ops_x[orbital] = "X"

    ops_y = ["I"] * num_qubits
    for k in range(orbital):
        ops_y[k] = "Z"
    ops_y[orbital] = "Y"

    return [
        PauliTerm(coefficient=0.5, operators=tuple(ops_x)),
        PauliTerm(coefficient=0.5j, operators=tuple(ops_y)),
    ]


def _multiply_pauli_terms(a: PauliTerm, b: PauliTerm) -> PauliTerm:
    """Multiply two PauliTerms using Pauli algebra."""
    # Pauli multiplication rules: XY=iZ, YX=-iZ, XZ=-iY, etc.
    _mul_table = {
        ("I", "I"): (1, "I"), ("I", "X"): (1, "X"), ("I", "Y"): (1, "Y"), ("I", "Z"): (1, "Z"),
        ("X", "I"): (1, "X"), ("X", "X"): (1, "I"), ("X", "Y"): (1j, "Z"), ("X", "Z"): (-1j, "Y"),
        ("Y", "I"): (1, "Y"), ("Y", "X"): (-1j, "Z"), ("Y", "Y"): (1, "I"), ("Y", "Z"): (1j, "X"),
        ("Z", "I"): (1, "Z"), ("Z", "X"): (1j, "Y"), ("Z", "Y"): (-1j, "X"), ("Z", "Z"): (1, "I"),
    }
    coeff = a.coefficient * b.coefficient
    ops = []
    for oa, ob in zip(a.operators, b.operators):
        phase, result_op = _mul_table[(oa, ob)]
        coeff *= phase
        ops.append(result_op)
    return PauliTerm(coefficient=coeff, operators=tuple(ops))


def _multiply_pauli_lists(
    terms_a: list[PauliTerm],
    terms_b: list[PauliTerm],
) -> list[PauliTerm]:
    """Distribute-multiply two sums of PauliTerms."""
    result = []
    for a in terms_a:
        for b in terms_b:
            result.append(_multiply_pauli_terms(a, b))
    return result


def jordan_wigner_transform(
    one_body: np.ndarray,
    two_body: np.ndarray | None = None,
    interaction_strength: float = 1.0,
) -> QubitHamiltonian:
    r"""Transform a fermionic Hamiltonian to a qubit Hamiltonian via Jordan-Wigner.

    .. math::

        H = \sum_{ij} h_{ij}\, a^\dagger_i a_j
          + \tfrac{1}{2} \sum_{ijkl} V_{ijkl}\, a^\dagger_i a^\dagger_j a_l a_k

    Each fermionic operator is expanded in Pauli strings, and the products
    are simplified.

    Parameters
    ----------
    one_body : np.ndarray
        Shape ``(M, M)`` — one-body integrals h_ij.
    two_body : np.ndarray or None
        Shape ``(M, M, M, M)`` — two-body integrals V_ijkl.
    interaction_strength : float
        Prefactor for two-body terms.

    Returns
    -------
    QubitHamiltonian
    """
    M = one_body.shape[0]
    num_qubits = M
    all_terms: list[PauliTerm] = []

    # One-body: Σ h_ij a†_i a_j
    for i in range(M):
        for j in range(M):
            if abs(one_body[i, j]) < 1e-15:
                continue
            ad_i = _jw_creation(i, num_qubits)
            a_j = _jw_annihilation(j, num_qubits)
            product = _multiply_pauli_lists(ad_i, a_j)
            for term in product:
                all_terms.append(PauliTerm(
                    coefficient=one_body[i, j] * term.coefficient,
                    operators=term.operators,
                ))

    # Two-body: ½ Σ V_ijkl a†_i a†_j a_l a_k
    if two_body is not None:
        for i in range(M):
            for j in range(M):
                for k in range(M):
                    for l in range(M):
                        coeff = 0.5 * interaction_strength * two_body[i, j, k, l]
                        if abs(coeff) < 1e-15:
                            continue
                        ad_i = _jw_creation(i, num_qubits)
                        ad_j = _jw_creation(j, num_qubits)
                        a_l = _jw_annihilation(l, num_qubits)
                        a_k = _jw_annihilation(k, num_qubits)
                        # a†_i a†_j a_l a_k
                        t1 = _multiply_pauli_lists(ad_i, ad_j)
                        t2 = _multiply_pauli_lists(a_l, a_k)
                        product = _multiply_pauli_lists(t1, t2)
                        for term in product:
                            all_terms.append(PauliTerm(
                                coefficient=coeff * term.coefficient,
                                operators=term.operators,
                            ))

    hamiltonian = QubitHamiltonian(terms=tuple(all_terms), num_qubits=num_qubits)
    return hamiltonian.simplify()


# ---------------------------------------------------------------------------
# Bravyi-Kitaev transformation
# ---------------------------------------------------------------------------

def _bk_parity_set(index: int, num_qubits: int) -> set[int]:
    """Parity set P(j): qubits that store parity of orbital j in BK encoding."""
    ancestors = set()
    idx = index + 1
    while idx <= num_qubits:
        ancestors.add(idx - 1)
        idx += idx & (-idx)
    ancestors.discard(index)
    return ancestors


def _bk_update_set(index: int, num_qubits: int) -> set[int]:
    """Update set U(j): qubits whose occupation bit must flip when orbital j is toggled."""
    updates = set()
    parent = index
    while parent < num_qubits:
        bit_length = parent & (-parent) if parent > 0 else 1
        next_parent = parent + bit_length
        if next_parent < num_qubits:
            updates.add(next_parent)
        parent = next_parent
    return updates


def _bk_remainder_set(index: int) -> set[int]:
    """Remainder set R(j) for BK encoding."""
    if index == 0:
        return set()
    remainder = set()
    idx = index
    while idx > 0:
        idx = idx & (idx - 1)
        if idx > 0:
            remainder.add(idx - 1)
    return remainder


def bravyi_kitaev_transform(
    one_body: np.ndarray,
    two_body: np.ndarray | None = None,
    interaction_strength: float = 1.0,
) -> QubitHamiltonian:
    r"""Transform a fermionic Hamiltonian to qubits via Bravyi-Kitaev encoding.

    BK uses a binary-tree encoding that stores both occupation and parity
    information, reducing the length of Z-strings from O(M) to O(log M)
    compared to Jordan-Wigner.

    For small systems (M ≤ ~20), the practical difference is modest.
    The main advantage appears in deeper circuits on real quantum hardware.

    Parameters
    ----------
    one_body : np.ndarray
        Shape ``(M, M)`` — one-body integrals.
    two_body : np.ndarray or None
        Shape ``(M, M, M, M)`` — two-body integrals.
    interaction_strength : float
        Prefactor for two-body terms.

    Returns
    -------
    QubitHamiltonian

    Notes
    -----
    This implementation builds the BK Hamiltonian by constructing the
    Jordan-Wigner Hamiltonian matrix first, then re-expressing it via a
    BK basis transformation.  For large M, a direct BK operator algebra
    implementation would be more efficient.
    """
    M = one_body.shape[0]
    num_qubits = M

    # Build the BK transformation matrix β (2^n × 2^n)
    # β_{ij} encodes the binary-tree parity structure
    dim = 2 ** num_qubits
    beta = np.eye(dim, dtype=complex)

    # Construct the BK basis change for each qubit
    # The BK transformation is: |BK⟩ = β |JW⟩
    # For the Hamiltonian: H_BK = β H_JW β†
    # Since β is its own inverse for the standard BK encoding:

    # Build JW Hamiltonian first
    jw_hamiltonian = jordan_wigner_transform(one_body, two_body, interaction_strength)
    H_jw = jw_hamiltonian.to_matrix().toarray()

    # Construct BK transformation matrix
    # BK encoding: qubit j stores parity of orbitals in its "parity set"
    bk_basis = np.zeros((dim, dim), dtype=complex)
    for state_idx in range(dim):
        # Convert JW state to BK state
        occ = np.array([(state_idx >> q) & 1 for q in range(num_qubits)])
        bk_bits = np.zeros(num_qubits, dtype=int)
        for q in range(num_qubits):
            # BK qubit q stores sum of occupations in its "flip set"
            # For the standard BK encoding, even-indexed qubits store
            # occupation, odd-indexed store partial parities
            if q % 2 == 0:
                bk_bits[q] = occ[q]
            else:
                # Parity of a range
                range_end = q
                range_start = q - (q & (-q))
                bk_bits[q] = int(np.sum(occ[range_start:range_end + 1])) % 2

        bk_idx = sum(b << q for q, b in enumerate(bk_bits))
        bk_basis[bk_idx, state_idx] = 1.0

    # Transform: H_BK = β H_JW β†
    H_bk = bk_basis @ H_jw @ bk_basis.conj().T

    # Decompose H_BK back into Pauli strings
    return _matrix_to_pauli_hamiltonian(H_bk, num_qubits)


def _matrix_to_pauli_hamiltonian(
    H: np.ndarray,
    num_qubits: int,
    tol: float = 1e-12,
) -> QubitHamiltonian:
    """Decompose a 2^n × 2^n matrix into a sum of Pauli strings."""
    dim = 2 ** num_qubits
    pauli_labels = ["I", "X", "Y", "Z"]
    pauli_mats = [np.eye(2, dtype=complex),
                  np.array([[0, 1], [1, 0]], dtype=complex),
                  np.array([[0, -1j], [1j, 0]], dtype=complex),
                  np.array([[1, 0], [0, -1]], dtype=complex)]

    terms = []

    def _generate_pauli_strings(n):
        if n == 0:
            yield (), np.array([[1.0]], dtype=complex)
            return
        for prev_ops, prev_mat in _generate_pauli_strings(n - 1):
            for idx, label in enumerate(pauli_labels):
                new_ops = prev_ops + (label,)
                new_mat = np.kron(prev_mat, pauli_mats[idx])
                yield new_ops, new_mat

    for ops, P in _generate_pauli_strings(num_qubits):
        # Coefficient = Tr(P · H) / 2^n
        coeff = np.trace(P @ H) / dim
        if abs(coeff) > tol:
            terms.append(PauliTerm(coefficient=complex(coeff), operators=ops))

    return QubitHamiltonian(terms=tuple(terms), num_qubits=num_qubits)


__all__ = [
    "PauliTerm",
    "QubitHamiltonian",
    "bravyi_kitaev_transform",
    "jordan_wigner_transform",
]
