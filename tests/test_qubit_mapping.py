"""Tests for qubit mapping (Jordan-Wigner, Bravyi-Kitaev) module."""
from __future__ import annotations

import pytest
import numpy as np

from spectral_packet_engine.qubit_mapping import (
    PauliTerm,
    QubitHamiltonian,
    jordan_wigner_transform,
    bravyi_kitaev_transform,
)
from spectral_packet_engine.many_body import (
    build_fock_space,
    build_many_body_hamiltonian,
)


class TestPauliTerm:
    def test_identity(self):
        t = PauliTerm(coefficient=1.0, operators=("I", "I"))
        mat = t.to_matrix().toarray()
        assert np.allclose(mat, np.eye(4))

    def test_pauli_x(self):
        t = PauliTerm(coefficient=1.0, operators=("X",))
        mat = t.to_matrix().toarray()
        expected = np.array([[0, 1], [1, 0]], dtype=complex)
        assert np.allclose(mat, expected)

    def test_coefficient_scaling(self):
        t = PauliTerm(coefficient=0.5, operators=("Z",))
        mat = t.to_matrix().toarray()
        expected = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
        assert np.allclose(mat, expected)

    def test_tensor_product(self):
        """X⊗Z should be the Kronecker product."""
        t = PauliTerm(coefficient=1.0, operators=("X", "Z"))
        mat = t.to_matrix().toarray()
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        expected = np.kron(X, Z)
        assert np.allclose(mat, expected)


class TestQubitHamiltonian:
    def test_simplify_combines_like_terms(self):
        t1 = PauliTerm(coefficient=0.3, operators=("X", "I"))
        t2 = PauliTerm(coefficient=0.7, operators=("X", "I"))
        t3 = PauliTerm(coefficient=0.5, operators=("Z", "Z"))
        qh = QubitHamiltonian(terms=(t1, t2, t3), num_qubits=2).simplify()
        # X⊗I terms should combine to 1.0
        found = False
        for t in qh.terms:
            if t.operators == ("X", "I"):
                assert abs(t.coefficient - 1.0) < 1e-12
                found = True
        assert found

    def test_simplify_drops_zeros(self):
        t1 = PauliTerm(coefficient=0.5, operators=("X",))
        t2 = PauliTerm(coefficient=-0.5, operators=("X",))
        qh = QubitHamiltonian(terms=(t1, t2), num_qubits=1).simplify()
        assert qh.num_terms == 0

    def test_to_dict(self):
        t = PauliTerm(coefficient=1.0 + 0.5j, operators=("X", "Y"))
        qh = QubitHamiltonian(terms=(t,), num_qubits=2)
        d = qh.to_dict()
        assert d["num_qubits"] == 2
        assert d["num_terms"] == 1
        assert d["terms"][0]["pauli_string"] == "XY"


class TestJordanWigner:
    def test_single_orbital_number_operator(self):
        """h = [[1.0]] → JW should give ½(I - Z)."""
        h1 = np.array([[1.0]])
        qh = jordan_wigner_transform(h1)
        mat = qh.to_matrix().toarray()
        # n = a†a → ½(I - Z)
        expected = np.array([[0, 0], [0, 1]], dtype=complex)
        assert np.allclose(mat, expected, atol=1e-10)

    def test_two_orbital_diagonal(self):
        """Diagonal h → qubit H should have correct spectrum."""
        h1 = np.diag([1.0, 2.0])
        qh = jordan_wigner_transform(h1)
        mat = qh.to_matrix().toarray()
        evals = np.sort(np.linalg.eigvalsh(mat))
        # States: |00⟩→0, |10⟩→1, |01⟩→2, |11⟩→3
        expected = np.sort([0.0, 1.0, 2.0, 3.0])
        assert np.allclose(evals, expected, atol=1e-10)

    def test_hermiticity(self):
        """JW Hamiltonian must be Hermitian."""
        h1 = np.array([[1.0, 0.5], [0.5, 2.0]])
        qh = jordan_wigner_transform(h1)
        mat = qh.to_matrix().toarray()
        assert np.allclose(mat, mat.conj().T, atol=1e-10)

    def test_spectrum_matches_fock_space(self):
        """JW qubit Hamiltonian eigenvalues should match Fock-space diagonalization."""
        M = 4
        h1 = np.diag([1.0, 2.5, 4.0, 5.5])
        N = 2

        # Fock space reference
        fock = build_fock_space(M, N)
        mb = build_many_body_hamiltonian(fock, h1)
        evals_fock = np.sort(np.linalg.eigvalsh(mb.matrix.toarray()))

        # JW qubit Hamiltonian (full 2^M space, pick N-particle sector)
        qh = jordan_wigner_transform(h1)
        mat = qh.to_matrix().toarray()
        evals_qubit = np.sort(np.linalg.eigvalsh(mat.real))

        # The N-particle sector eigenvalues should appear in the qubit spectrum
        for e in evals_fock:
            assert np.any(np.abs(evals_qubit - e) < 1e-8), f"Missing eigenvalue {e}"

    def test_to_dict_structure(self):
        h1 = np.diag([1.0, 2.0])
        qh = jordan_wigner_transform(h1)
        d = qh.to_dict()
        assert "num_qubits" in d
        assert "terms" in d
        assert d["num_qubits"] == 2


class TestBravyiKitaev:
    def test_single_orbital(self):
        """BK for 1 orbital should match JW exactly."""
        h1 = np.array([[1.0]])
        qh_jw = jordan_wigner_transform(h1)
        qh_bk = bravyi_kitaev_transform(h1)
        mat_jw = qh_jw.to_matrix().toarray()
        mat_bk = qh_bk.to_matrix().toarray()
        # Same spectrum
        evals_jw = np.sort(np.linalg.eigvalsh(mat_jw.real))
        evals_bk = np.sort(np.linalg.eigvalsh(mat_bk.real))
        assert np.allclose(evals_jw, evals_bk, atol=1e-10)

    def test_spectrum_matches_jordan_wigner(self):
        """BK and JW should produce the same eigenvalue spectrum."""
        h1 = np.diag([1.0, 2.0, 3.0, 4.0])
        qh_jw = jordan_wigner_transform(h1)
        qh_bk = bravyi_kitaev_transform(h1)
        evals_jw = np.sort(np.linalg.eigvalsh(qh_jw.to_matrix().toarray().real))
        evals_bk = np.sort(np.linalg.eigvalsh(qh_bk.to_matrix().toarray().real))
        assert np.allclose(evals_jw, evals_bk, atol=1e-10)

    def test_hermiticity(self):
        h1 = np.array([[1.0, 0.3], [0.3, 2.0]])
        qh = bravyi_kitaev_transform(h1)
        mat = qh.to_matrix().toarray()
        assert np.allclose(mat, mat.conj().T, atol=1e-10)

    def test_with_two_body(self):
        """BK with interaction should still produce a Hermitian Hamiltonian."""
        h1 = np.diag([1.0, 2.0])
        h2 = np.zeros((2, 2, 2, 2))
        h2[0, 1, 0, 1] = 0.5
        h2[1, 0, 1, 0] = 0.5
        qh = bravyi_kitaev_transform(h1, h2, interaction_strength=0.1)
        mat = qh.to_matrix().toarray()
        assert np.allclose(mat, mat.conj().T, atol=1e-10)
