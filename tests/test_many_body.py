"""Tests for the many-body Fock space module."""
from __future__ import annotations

import pytest
import torch
import numpy as np
from math import comb

from spectral_packet_engine.many_body import (
    FockSpace,
    FullFockSpace,
    build_fock_space,
    build_full_fock_space,
    creation_operator,
    annihilation_operator,
    number_operator,
    one_body_excitation,
    two_body_excitation,
    compute_one_body_integrals,
    compute_two_body_integrals,
    build_many_body_hamiltonian,
    hamiltonian_to_torch,
)


class TestFockSpace:
    def test_dimension(self):
        fock = build_fock_space(6, 3)
        assert fock.dim == comb(6, 3) == 20

    def test_single_particle(self):
        fock = build_fock_space(4, 1)
        assert fock.dim == 4
        for config in fock.configs:
            assert sum(config) == 1

    def test_full_occupation(self):
        fock = build_fock_space(4, 4)
        assert fock.dim == 1
        assert list(fock.configs[0]) == [1, 1, 1, 1]

    def test_vacuum(self):
        fock = build_fock_space(4, 0)
        assert fock.dim == 1
        assert list(fock.configs[0]) == [0, 0, 0, 0]

    def test_reverse_lookup(self):
        fock = build_fock_space(5, 2)
        for idx, config in enumerate(fock.configs):
            assert fock.config_to_index[tuple(config)] == idx

    def test_invalid_particles(self):
        with pytest.raises(ValueError):
            build_fock_space(3, 5)

    def test_large_space(self):
        fock = build_fock_space(10, 5)
        assert fock.dim == comb(10, 5) == 252


class TestFullFockSpace:
    def test_dimension(self):
        full = build_full_fock_space(4)
        assert full.dim == 16

    def test_configs_cover_all_states(self):
        full = build_full_fock_space(3)
        assert full.dim == 8
        particle_counts = sorted([int(np.sum(c)) for c in full.configs])
        assert particle_counts == [0, 1, 1, 1, 2, 2, 2, 3]


class TestCreationAnnihilation:
    """Tests on the full 2^M Fock space where a† and a are well-defined."""

    @pytest.fixture
    def full4(self):
        return build_full_fock_space(4)

    def test_creation_annihilation_adjoint(self, full4):
        for orb in range(4):
            c = creation_operator(full4, orb)
            a = annihilation_operator(full4, orb)
            diff = (c - a.T).toarray()
            assert np.allclose(diff, 0)

    def test_anticommutation_same_orbital(self, full4):
        """{a_i, a†_i} = I."""
        dim = full4.dim
        for orb in range(4):
            c = creation_operator(full4, orb)
            a = annihilation_operator(full4, orb)
            anticomm = (a @ c + c @ a).toarray()
            assert np.allclose(anticomm, np.eye(dim)), f"orbital {orb}"

    def test_anticommutation_different_orbitals(self, full4):
        """{a_i, a†_j} = 0 for i ≠ j."""
        dim = full4.dim
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                ci = creation_operator(full4, i)
                aj = annihilation_operator(full4, j)
                anticomm = (aj @ ci + ci @ aj).toarray()
                assert np.allclose(anticomm, 0), f"orbitals {i}, {j}"

    def test_pauli_exclusion(self, full4):
        """(a†_i)² = 0."""
        for orb in range(4):
            c = creation_operator(full4, orb)
            c_sq = (c @ c).toarray()
            assert np.allclose(c_sq, 0)

    def test_number_from_creation_annihilation(self, full4):
        """n_i = a†_i a_i in full space."""
        for orb in range(4):
            c = creation_operator(full4, orb)
            a = annihilation_operator(full4, orb)
            n_computed = (c @ a).toarray()
            n_direct = number_operator(full4, orb).toarray()
            assert np.allclose(n_computed, n_direct)


class TestExcitationOperators:
    """Tests for number-conserving operators in the N-particle sector."""

    @pytest.fixture
    def fock4_2(self):
        return build_fock_space(4, 2)

    def test_diagonal_one_body(self, fock4_2):
        """E_ii = n_i in the N-particle sector."""
        for orb in range(4):
            E_ii = one_body_excitation(fock4_2, orb, orb).toarray()
            n_i = number_operator(fock4_2, orb).toarray()
            assert np.allclose(E_ii, n_i)

    def test_total_number(self, fock4_2):
        """Σ E_ii = N · I."""
        dim = fock4_2.dim
        N_total = sum(
            one_body_excitation(fock4_2, i, i) for i in range(4)
        ).toarray()
        expected = np.eye(dim) * fock4_2.num_particles
        assert np.allclose(N_total, expected)

    def test_one_body_matches_full_space(self, fock4_2):
        """E_ij in N-particle sector should match a†_i a_j restricted from full space."""
        full = build_full_fock_space(4)

        # Build projection from full space to N-particle sector
        n_sector_indices = []
        for full_idx, config in enumerate(full.configs):
            if int(np.sum(config)) == 2:
                n_sector_indices.append(full_idx)

        for i in range(4):
            for j in range(4):
                # Full-space operator
                c_i = creation_operator(full, i)
                a_j = annihilation_operator(full, j)
                E_full = (c_i @ a_j).toarray()
                # Extract N-particle block
                E_block = E_full[np.ix_(n_sector_indices, n_sector_indices)]

                # N-particle operator
                E_sector = one_body_excitation(fock4_2, i, j).toarray()

                # Map indices: full configs to sector configs
                # Need to reorder since fock4_2 and full may order configs differently
                sector_configs = [tuple(full.configs[fi]) for fi in n_sector_indices]
                perm = [fock4_2.config_to_index[c] for c in sector_configs]
                E_block_reordered = E_block[np.ix_(perm, perm)]

                assert np.allclose(E_sector, E_block_reordered), f"i={i}, j={j}"


class TestManyBodyHamiltonian:
    def test_non_interacting_eigenvalues(self):
        """Without interaction, many-body E = sum of single-particle E."""
        fock = build_fock_space(4, 2)
        h1 = np.diag([1.0, 2.0, 3.0, 4.0])
        mb = build_many_body_hamiltonian(fock, h1, two_body=None)

        H_dense = mb.matrix.toarray()
        evals = np.sort(np.linalg.eigvalsh(H_dense))

        # Ground state: particles in orbitals 0,1 → E = 1+2 = 3
        assert abs(evals[0] - 3.0) < 1e-10

    def test_non_interacting_all_energies(self):
        """All energies should be sums of pairs of single-particle energies."""
        fock = build_fock_space(4, 2)
        h1 = np.diag([1.0, 2.0, 3.0, 4.0])
        mb = build_many_body_hamiltonian(fock, h1, two_body=None)
        evals = np.sort(np.linalg.eigvalsh(mb.matrix.toarray()))

        # All C(4,2)=6 pair sums
        expected = sorted([1+2, 1+3, 1+4, 2+3, 2+4, 3+4])
        assert np.allclose(evals, expected, atol=1e-10)

    def test_hamiltonian_hermitian(self):
        fock = build_fock_space(4, 2)
        h1 = np.diag([1.0, 2.0, 3.0, 4.0])
        rng = np.random.default_rng(42)
        h2 = rng.standard_normal((4, 4, 4, 4))
        h2 = (h2 + h2.transpose(2, 3, 0, 1)) / 2

        mb = build_many_body_hamiltonian(fock, h1, h2, interaction_strength=0.1)
        H_dense = mb.matrix.toarray()
        assert np.allclose(H_dense, H_dense.T, atol=1e-10)

    def test_torch_conversion(self):
        fock = build_fock_space(4, 2)
        h1 = np.diag([1.0, 2.0, 3.0, 4.0])
        mb = build_many_body_hamiltonian(fock, h1)
        H_torch = hamiltonian_to_torch(mb)
        assert H_torch.is_sparse_csr
        assert H_torch.shape == (fock.dim, fock.dim)

    def test_sparse_solve_matches_dense(self):
        """Sparse solver on many-body H should match dense diag."""
        from spectral_packet_engine.sparse_eigensolver import solve_eigenproblem_sparse

        fock = build_fock_space(6, 3)  # dim = 20
        h1 = np.diag([1.0, 2.5, 4.0, 5.5, 7.0, 8.5])
        mb = build_many_body_hamiltonian(fock, h1)

        H_dense = mb.matrix.toarray()
        evals_dense = np.sort(np.linalg.eigvalsh(H_dense))[:4]

        H_torch = hamiltonian_to_torch(mb)
        result = solve_eigenproblem_sparse(H_torch, num_states=4, method="lanczos")

        assert torch.allclose(
            result.eigenvalues,
            torch.tensor(evals_dense, dtype=torch.float64),
            atol=1e-8,
        )
