"""Quantum perturbation theory -- first and second order corrections.

Given an unperturbed Hamiltonian H_0 with known eigenstates |n> and
eigenvalues E_n^(0), and a perturbation V', the corrected energies and
states are expanded order-by-order:

    E_n = E_n^(0) + E_n^(1) + E_n^(2) + ...
    |n> = |n^(0)> + |n^(1)> + ...

First-order energy correction:

    E_n^(1) = <n|V'|n>

Second-order energy correction:

    E_n^(2) = sum_{m != n} |<m|V'|n>|^2 / (E_n^(0) - E_m^(0))

First-order state correction coefficients:

    C_{mn} = <m|V'|n> / (E_n^(0) - E_m^(0))    for m != n

When the unperturbed spectrum contains degeneracies, standard perturbation
theory breaks down.  Degenerate perturbation theory diagonalises V' within
each degenerate subspace to lift the degeneracy and identify the "good"
linear combinations that remain eigenstates of the full Hamiltonian.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from spectral_packet_engine.domain import coerce_tensor

Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PerturbationResult:
    """Complete non-degenerate perturbation theory analysis."""

    unperturbed_energies: Tensor
    first_order_energies: Tensor
    second_order_energies: Tensor
    corrected_energies: Tensor
    first_order_states: Tensor
    perturbation_matrix: Tensor
    convergence_parameter: Tensor


@dataclass(frozen=True, slots=True)
class DegeneratePerturbationResult:
    """Degenerate perturbation theory analysis.

    ``degenerate_indices`` groups eigenstate indices that share a common
    unperturbed energy.  ``lifted_energies`` are the eigenvalues of V'
    restricted to each degenerate subspace, and ``good_states`` are the
    corresponding eigenvectors (expressed as coefficients in the original
    basis).
    """

    degenerate_indices: list[list[int]]
    lifted_energies: Tensor
    good_states: Tensor


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def compute_perturbation_matrix(
    perturbation_fn: Callable[[Tensor], Tensor],
    eigenstates: Tensor,
    grid: Tensor,
) -> Tensor:
    r"""Compute V'_{mn} = \int \psi_m(x) V'(x) \psi_n(x) dx.

    The integral is evaluated numerically via the trapezoidal rule on the
    supplied ``grid``.

    Parameters
    ----------
    perturbation_fn:
        Callable mapping a position tensor to the perturbation potential.
    eigenstates:
        Tensor of shape ``(num_states, num_points)`` containing the
        wavefunctions evaluated on the grid.
    grid:
        One-dimensional tensor of grid points.

    Returns
    -------
    Tensor of shape ``(num_states, num_states)`` with matrix elements.
    """
    grid = coerce_tensor(grid, dtype=torch.float64)
    eigenstates = coerce_tensor(eigenstates, dtype=torch.float64)
    V_prime = perturbation_fn(grid)  # (num_points,)
    V_prime = coerce_tensor(V_prime, dtype=torch.float64, device=grid.device)

    # Integrand: psi_m(x) * V'(x) * psi_n(x) for all m, n
    # eigenstates: (N, G), V_prime: (G,)
    weighted = eigenstates * V_prime.unsqueeze(0)  # (N, G)

    # Trapezoidal weights
    dx = grid[1:] - grid[:-1]  # (G-1,)
    trap_weights = torch.zeros_like(grid)
    trap_weights[0] = dx[0] / 2
    trap_weights[-1] = dx[-1] / 2
    trap_weights[1:-1] = (dx[:-1] + dx[1:]) / 2  # (G-2,)

    # V'_{mn} = sum_x psi_m(x) * V'(x) * psi_n(x) * w(x)
    # = (eigenstates * sqrt(w)) @ (weighted * sqrt(w))^T  -- but simpler:
    integrand = weighted * trap_weights.unsqueeze(0)  # (N, G)
    matrix = integrand @ eigenstates.T  # (N, N)
    return matrix


def first_order_energy(perturbation_matrix: Tensor) -> Tensor:
    r"""E_n^{(1)} = V'_{nn}, the diagonal elements of the perturbation matrix."""
    return torch.diagonal(perturbation_matrix)


def second_order_energy(
    perturbation_matrix: Tensor,
    eigenvalues: Tensor,
) -> Tensor:
    r"""E_n^{(2)} = \sum_{m \ne n} |V'_{mn}|^2 / (E_n^{(0)} - E_m^{(0)}).

    The diagonal (m == n) is excluded via masking to avoid division by zero.
    """
    perturbation_matrix = coerce_tensor(perturbation_matrix, dtype=torch.float64)
    eigenvalues = coerce_tensor(eigenvalues, dtype=torch.float64)

    N = perturbation_matrix.shape[0]
    # Energy denominator: E_n - E_m  for column index n, row index m
    # denom[m, n] = E_n - E_m
    E_n = eigenvalues.unsqueeze(0)  # (1, N)
    E_m = eigenvalues.unsqueeze(1)  # (N, 1)
    denom = E_n - E_m  # (N, N) -- denom[m, n] = E_n - E_m

    # Mask diagonal to avoid 0/0
    mask = ~torch.eye(N, dtype=torch.bool, device=perturbation_matrix.device)
    safe_denom = torch.where(mask, denom, torch.ones_like(denom))

    Vmn_sq = perturbation_matrix ** 2  # |V'_{mn}|^2 (real basis)
    contributions = Vmn_sq / safe_denom  # (N, N)
    contributions = torch.where(mask, contributions, torch.zeros_like(contributions))

    # Sum over m (rows) for each n (column)
    return contributions.sum(dim=0)


def first_order_states(
    perturbation_matrix: Tensor,
    eigenvalues: Tensor,
) -> Tensor:
    r"""Coefficient matrix C_{mn} = V'_{mn} / (E_n^{(0)} - E_m^{(0)}).

    Diagonal entries are set to zero (no self-mixing).  The corrected
    state is |n^{(1)}> = \sum_{m \ne n} C_{mn} |m>.
    """
    perturbation_matrix = coerce_tensor(perturbation_matrix, dtype=torch.float64)
    eigenvalues = coerce_tensor(eigenvalues, dtype=torch.float64)

    N = perturbation_matrix.shape[0]
    E_n = eigenvalues.unsqueeze(0)  # (1, N)
    E_m = eigenvalues.unsqueeze(1)  # (N, 1)
    denom = E_n - E_m  # (N, N)

    mask = ~torch.eye(N, dtype=torch.bool, device=perturbation_matrix.device)
    safe_denom = torch.where(mask, denom, torch.ones_like(denom))

    coeffs = perturbation_matrix / safe_denom
    coeffs = torch.where(mask, coeffs, torch.zeros_like(coeffs))
    return coeffs


def analyze_perturbation(
    perturbation_fn: Callable[[Tensor], Tensor],
    eigenstates: Tensor,
    eigenvalues: Tensor,
    grid: Tensor,
) -> PerturbationResult:
    """Run a full non-degenerate perturbation theory analysis.

    Parameters
    ----------
    perturbation_fn:
        V'(x) -- maps position tensor to perturbation potential.
    eigenstates:
        Shape ``(num_states, num_points)`` -- unperturbed wavefunctions on
        the grid.
    eigenvalues:
        Shape ``(num_states,)`` -- unperturbed energies E_n^{(0)}.
    grid:
        Shape ``(num_points,)`` -- spatial grid.

    Returns
    -------
    PerturbationResult with all corrections through second order.
    """
    eigenvalues = coerce_tensor(eigenvalues, dtype=torch.float64)
    V_mat = compute_perturbation_matrix(perturbation_fn, eigenstates, grid)

    E1 = first_order_energy(V_mat)
    E2 = second_order_energy(V_mat, eigenvalues)
    C = first_order_states(V_mat, eigenvalues)

    corrected = eigenvalues + E1 + E2

    # Convergence parameter: max|V'_{mn}| / min|E_m - E_n| for m != n
    N = V_mat.shape[0]
    mask = ~torch.eye(N, dtype=torch.bool, device=V_mat.device)
    off_diag_V = torch.abs(V_mat[mask])
    denom_full = eigenvalues.unsqueeze(0) - eigenvalues.unsqueeze(1)
    off_diag_denom = torch.abs(denom_full[mask])

    max_V = off_diag_V.max() if off_diag_V.numel() > 0 else torch.tensor(0.0, dtype=torch.float64)
    min_dE = off_diag_denom.min() if off_diag_denom.numel() > 0 else torch.tensor(1.0, dtype=torch.float64)
    safe_min_dE = torch.where(min_dE > 0, min_dE, torch.ones_like(min_dE))
    convergence = max_V / safe_min_dE

    return PerturbationResult(
        unperturbed_energies=eigenvalues,
        first_order_energies=E1,
        second_order_energies=E2,
        corrected_energies=corrected,
        first_order_states=C,
        perturbation_matrix=V_mat,
        convergence_parameter=convergence,
    )


# ---------------------------------------------------------------------------
# Degenerate perturbation theory
# ---------------------------------------------------------------------------

def degenerate_perturbation(
    perturbation_matrix: Tensor,
    eigenvalues: Tensor,
    *,
    degeneracy_threshold: float = 1e-8,
) -> DegeneratePerturbationResult:
    r"""Degenerate perturbation theory.

    Groups eigenstates whose unperturbed energies differ by less than
    ``degeneracy_threshold``, then diagonalises V' within each degenerate
    subspace.  The eigenvalues of V' restricted to a degenerate block give
    the first-order energy splittings, and the eigenvectors are the "good"
    basis states.

    Parameters
    ----------
    perturbation_matrix:
        V'_{mn} matrix, shape ``(N, N)``.
    eigenvalues:
        Unperturbed energies, shape ``(N,)``.
    degeneracy_threshold:
        Energies closer than this value are deemed degenerate.

    Returns
    -------
    DegeneratePerturbationResult
    """
    perturbation_matrix = coerce_tensor(perturbation_matrix, dtype=torch.float64)
    eigenvalues = coerce_tensor(eigenvalues, dtype=torch.float64)

    N = eigenvalues.shape[0]
    device = eigenvalues.device

    # --- identify degenerate subspaces via union-find style grouping ---
    visited = [False] * N
    groups: list[list[int]] = []
    for i in range(N):
        if visited[i]:
            continue
        group = [i]
        visited[i] = True
        for j in range(i + 1, N):
            if not visited[j] and torch.abs(eigenvalues[i] - eigenvalues[j]).item() < degeneracy_threshold:
                group.append(j)
                visited[j] = True
        groups.append(group)

    # --- diagonalise V' within each subspace ---
    lifted_list: list[Tensor] = []
    good_states_rows: list[Tensor] = []

    for group in groups:
        idx = torch.tensor(group, dtype=torch.long, device=device)
        if len(group) == 1:
            # Non-degenerate: first-order energy is just the diagonal element
            lifted_list.append(perturbation_matrix[idx[0], idx[0]].unsqueeze(0))
            row = torch.zeros(N, dtype=torch.float64, device=device)
            row[group[0]] = 1.0
            good_states_rows.append(row.unsqueeze(0))
        else:
            # Extract sub-block of V'
            sub = perturbation_matrix[idx][:, idx]  # (d, d)
            evals, evecs = torch.linalg.eigh(sub)
            lifted_list.append(evals)
            # Express good states in full basis
            block = torch.zeros(len(group), N, dtype=torch.float64, device=device)
            for local_i, global_i in enumerate(group):
                block[:, global_i] = evecs[local_i, :]
            # Transpose: each row of block is a good state
            block = block.T  # now (N, d) -- columns are good states?
            # We want (d, N) where each row is a good state
            full_vecs = torch.zeros(len(group), N, dtype=torch.float64, device=device)
            for a in range(len(group)):
                for local_i, global_i in enumerate(group):
                    full_vecs[a, global_i] = evecs[local_i, a]
            good_states_rows.append(full_vecs)

    lifted_energies = torch.cat(lifted_list)
    good_states = torch.cat(good_states_rows, dim=0)

    return DegeneratePerturbationResult(
        degenerate_indices=groups,
        lifted_energies=lifted_energies,
        good_states=good_states,
    )


__all__ = [
    "DegeneratePerturbationResult",
    "PerturbationResult",
    "analyze_perturbation",
    "compute_perturbation_matrix",
    "degenerate_perturbation",
    "first_order_energy",
    "first_order_states",
    "second_order_energy",
]
