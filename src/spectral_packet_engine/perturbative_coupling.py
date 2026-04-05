"""Weakly nonseparable perturbative coupling analysis.

This module bridges the gap between the separable tensor-product spectra
(``reduced_models.py``) and full coupled-channel dynamics.  It provides:

1. **Landau-Zener transition probabilities** at avoided crossings.
2. **Fano resonance profile fitting** for asymmetric line shapes.
3. **Multi-channel coupling analysis** extending the two-channel model.
4. **Perturbative energy shifts** from weak inter-channel coupling.

All functions compose existing infrastructure (``perturbation.py``,
``reduced_models.py``, ``scattering.py``) rather than reimplementing
core algorithms.  The perturbative picture is honest: it applies when
the coupling is small relative to the level spacing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from spectral_packet_engine.domain import InfiniteWell1D, coerce_tensor
from spectral_packet_engine.eigensolver import solve_eigenproblem
from spectral_packet_engine.perturbation import (
    compute_perturbation_matrix,
    first_order_energy,
    second_order_energy,
)
from spectral_packet_engine.runtime import TorchRuntime, inspect_torch_runtime

Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LandauZenerResult:
    """Landau-Zener transition probability at an avoided crossing.

    The formula P = exp(-2π δ²/ (ℏ v |ΔF|)) gives the probability of
    a diabatic transition, where δ is the minimum gap, v the traversal
    velocity, and ΔF the slope difference of the diabatic surfaces.
    """

    minimum_gap: float
    slope_difference: float
    transition_probability: Tensor
    adiabatic_probability: Tensor
    velocity: float
    coupling_regime: str  # "adiabatic", "diabatic", or "intermediate"
    assumptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FanoProfile:
    """Fano resonance line-shape parameters.

    T(E) = σ_bg * (q + ε)² / (1 + ε²)

    where ε = (E - E_0) / (Γ/2) is the reduced energy, q is the
    asymmetry parameter, and σ_bg is the background cross section.
    """

    resonance_energy: float
    width: float         # Γ (FWHM)
    q_parameter: float   # Fano asymmetry (|q|→∞ = Lorentzian, q=0 = antiresonance)
    background: float    # σ_bg
    fitted_profile: Tensor
    residual_rms: float
    assumptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MultiChannelCouplingResult:
    """Multi-channel weak-coupling analysis.

    Extends the two-channel avoided-crossing model to N channels
    with perturbative energy shifts.
    """

    num_channels: int
    unperturbed_energies: Tensor        # (num_channels,)
    coupling_matrix: Tensor             # (num_channels, num_channels)
    first_order_shifts: Tensor          # (num_channels,)
    second_order_shifts: Tensor         # (num_channels,)
    corrected_energies: Tensor          # (num_channels,)
    mixing_coefficients: Tensor         # (num_channels, num_channels)
    convergence_parameter: float
    coupling_regime: str                # "weak", "intermediate", "strong"
    avoided_crossings: tuple[tuple[int, int, float], ...]  # (i, j, gap)
    assumptions: tuple[str, ...]


# ---------------------------------------------------------------------------
# Landau-Zener
# ---------------------------------------------------------------------------


def landau_zener_transition(
    *,
    minimum_gap: float,
    slope_difference: float,
    velocity: float = 1.0,
    hbar: float = 1.0,
) -> LandauZenerResult:
    """Compute the Landau-Zener diabatic transition probability.

    Parameters
    ----------
    minimum_gap : float
        The minimum energy gap δ at the avoided crossing.
    slope_difference : float
        |ΔF| = |dE₁/dx - dE₂/dx| at the crossing point.
    velocity : float
        Traversal velocity v through the crossing region.
    hbar : float
        Reduced Planck constant.

    Returns
    -------
    LandauZenerResult
    """
    if minimum_gap < 0:
        raise ValueError("minimum_gap must be non-negative")
    if slope_difference <= 0:
        raise ValueError("slope_difference must be positive")
    if velocity <= 0:
        raise ValueError("velocity must be positive")

    exponent = 2 * math.pi * minimum_gap ** 2 / (hbar * velocity * abs(slope_difference))
    p_diabatic = torch.tensor(math.exp(-exponent))
    p_adiabatic = 1.0 - p_diabatic

    if exponent > 5.0:
        regime = "adiabatic"
    elif exponent < 0.2:
        regime = "diabatic"
    else:
        regime = "intermediate"

    return LandauZenerResult(
        minimum_gap=minimum_gap,
        slope_difference=slope_difference,
        transition_probability=p_diabatic,
        adiabatic_probability=p_adiabatic,
        velocity=velocity,
        coupling_regime=regime,
        assumptions=(
            "The Landau-Zener formula assumes a linear diabatic crossing with constant velocity.",
            f"Coupling regime: {regime} (exponent = {exponent:.3f}).",
        ),
    )


# ---------------------------------------------------------------------------
# Fano profile fitting
# ---------------------------------------------------------------------------


def fit_fano_profile(
    *,
    energies: Tensor,
    transmission: Tensor,
    resonance_energy: float,
    resonance_width: float,
) -> FanoProfile:
    """Fit a Fano line shape to a transmission resonance.

    Parameters
    ----------
    energies : Tensor
        Energy grid.
    transmission : Tensor
        T(E) values.
    resonance_energy : float
        Approximate resonance center.
    resonance_width : float
        Approximate FWHM.
    """
    E = energies.detach().to(dtype=torch.float64)
    T = transmission.detach().to(dtype=torch.float64)

    half_gamma = max(abs(resonance_width) / 2.0, 1e-30)
    epsilon = (E - resonance_energy) / half_gamma

    # Least-squares fit for q and σ_bg
    # T(ε) = σ_bg * (q + ε)² / (1 + ε²)
    # Linearize: T * (1 + ε²) = σ_bg * (q² + 2qε + ε²)
    # Let a = σ_bg * q², b = 2 * σ_bg * q, c = σ_bg
    # T * (1 + ε²) = a + b*ε + c*ε²
    lhs = T * (1 + epsilon ** 2)
    A = torch.stack([torch.ones_like(epsilon), epsilon, epsilon ** 2], dim=-1)
    # Normal equations
    result = torch.linalg.lstsq(A, lhs)
    coeffs = result.solution
    a, b, c = float(coeffs[0].item()), float(coeffs[1].item()), float(coeffs[2].item())

    sigma_bg = max(abs(c), 1e-30)
    q = b / (2 * sigma_bg) if sigma_bg > 1e-30 else 0.0

    fitted = torch.tensor(sigma_bg, dtype=E.dtype) * (q + epsilon) ** 2 / (1 + epsilon ** 2)
    residual = T - fitted
    residual_rms = float(torch.sqrt(torch.mean(residual ** 2)).item())

    return FanoProfile(
        resonance_energy=resonance_energy,
        width=resonance_width,
        q_parameter=q,
        background=sigma_bg,
        fitted_profile=fitted.detach(),
        residual_rms=residual_rms,
        assumptions=(
            "The Fano profile assumes a single discrete state coupled to a flat continuum.",
            f"Asymmetry parameter q = {q:.3f}: "
            + ("nearly symmetric Lorentzian." if abs(q) > 5 else "significant asymmetry." if abs(q) > 0.5 else "near antiresonance."),
        ),
    )


# ---------------------------------------------------------------------------
# Multi-channel coupling
# ---------------------------------------------------------------------------


def analyze_multichannel_coupling(
    *,
    potential_fns: tuple[..., ...],
    coupling_fn: ...,
    domain: InfiniteWell1D,
    num_points: int = 128,
    num_states: int = 6,
    coupling_strength: float = 1.0,
    device: str | torch.device = "cpu",
) -> MultiChannelCouplingResult:
    """Perturbative coupling analysis for N channels sharing a domain.

    Each channel has its own potential V_i(x).  The coupling between
    channels i and j is ``coupling_strength * coupling_fn(x)`` applied
    perturbatively to the uncoupled eigenstates.

    Parameters
    ----------
    potential_fns : tuple of callables
        V_i(x) for each channel.  Length N determines channel count.
    coupling_fn : callable
        W(x) — spatial profile of the inter-channel coupling.
    domain : InfiniteWell1D
        Shared domain for all channels.
    num_points : int
        Grid resolution.
    num_states : int
        Number of states per channel to retain.
    coupling_strength : float
        Multiplicative scale λ for the coupling.
    device : str
        Torch device.
    """
    runtime = inspect_torch_runtime(device)
    num_channels = len(potential_fns)
    if num_channels < 2:
        raise ValueError("Need at least 2 channels")

    # Solve each channel independently
    channel_results = []
    for pot_fn in potential_fns:
        result = solve_eigenproblem(
            pot_fn, domain,
            num_points=num_points,
            num_states=num_states,
        )
        channel_results.append(result)

    total_states = num_channels * num_states
    grid = channel_results[0].grid

    # Build unperturbed energies (block diagonal)
    unperturbed = torch.zeros(total_states, dtype=runtime.preferred_real_dtype, device=runtime.device)
    for ch, result in enumerate(channel_results):
        unperturbed[ch * num_states:(ch + 1) * num_states] = result.eigenvalues

    # Build coupling matrix V'_{mn} between ALL states
    coupling_values = coupling_fn(grid) * coupling_strength
    coupling_matrix = torch.zeros(
        total_states, total_states,
        dtype=runtime.preferred_real_dtype, device=runtime.device,
    )

    dx = float((grid[-1] - grid[0]).item()) / (grid.shape[0] - 1)
    for ch_i in range(num_channels):
        for ch_j in range(ch_i + 1, num_channels):
            states_i = channel_results[ch_i].eigenstates
            states_j = channel_results[ch_j].eigenstates
            # V'_{mn} = ∫ ψ_m(x) W(x) ψ_n(x) dx  (trapezoidal)
            for m in range(num_states):
                for n in range(num_states):
                    integrand = states_i[m] * coupling_values * states_j[n]
                    element = torch.trapezoid(integrand, grid)
                    idx_m = ch_i * num_states + m
                    idx_n = ch_j * num_states + n
                    coupling_matrix[idx_m, idx_n] = element
                    coupling_matrix[idx_n, idx_m] = element

    # First-order energy shifts
    first_shifts = torch.diagonal(coupling_matrix)

    # Second-order energy shifts
    second_shifts = torch.zeros_like(unperturbed)
    for n in range(total_states):
        for m in range(total_states):
            if m == n:
                continue
            denom = unperturbed[n] - unperturbed[m]
            if abs(float(denom.item())) < 1e-12:
                continue
            second_shifts[n] += coupling_matrix[m, n] ** 2 / denom

    corrected = unperturbed + first_shifts + second_shifts

    # Mixing coefficients (first-order state corrections)
    mixing = torch.eye(total_states, dtype=unperturbed.dtype, device=unperturbed.device)
    for n in range(total_states):
        for m in range(total_states):
            if m == n:
                continue
            denom = unperturbed[n] - unperturbed[m]
            if abs(float(denom.item())) < 1e-12:
                continue
            mixing[m, n] = coupling_matrix[m, n] / denom

    # Convergence parameter
    off_diag = coupling_matrix.clone()
    off_diag.fill_diagonal_(0)
    max_coupling = float(torch.max(torch.abs(off_diag)).item())
    energy_diffs = unperturbed.unsqueeze(0) - unperturbed.unsqueeze(1)
    energy_diffs.fill_diagonal_(float('inf'))
    min_spacing = float(torch.min(torch.abs(energy_diffs)).item())
    convergence = max_coupling / max(min_spacing, 1e-30)

    if convergence < 0.1:
        regime = "weak"
    elif convergence < 1.0:
        regime = "intermediate"
    else:
        regime = "strong"

    # Detect avoided crossings (pairs with gap < 2 * coupling)
    crossings: list[tuple[int, int, float]] = []
    for i in range(total_states):
        for j in range(i + 1, total_states):
            gap = abs(float((corrected[i] - corrected[j]).item()))
            coupling_ij = abs(float(coupling_matrix[i, j].item()))
            if coupling_ij > 0 and gap < 4 * coupling_ij:
                crossings.append((i, j, gap))

    return MultiChannelCouplingResult(
        num_channels=num_channels,
        unperturbed_energies=unperturbed.detach(),
        coupling_matrix=coupling_matrix.detach(),
        first_order_shifts=first_shifts.detach(),
        second_order_shifts=second_shifts.detach(),
        corrected_energies=corrected.detach(),
        mixing_coefficients=mixing.detach(),
        convergence_parameter=convergence,
        coupling_regime=regime,
        avoided_crossings=tuple(crossings),
        assumptions=(
            "Perturbative coupling is valid when the convergence parameter is small (< 0.1).",
            f"Convergence parameter = {convergence:.4f} → regime: {regime}.",
            f"Detected {len(crossings)} avoided crossing(s) among {total_states} states.",
            "Off-diagonal coupling uses trapezoidal quadrature on the shared grid.",
        ),
    )


__all__ = [
    "FanoProfile",
    "LandauZenerResult",
    "MultiChannelCouplingResult",
    "analyze_multichannel_coupling",
    "fit_fano_profile",
    "landau_zener_transition",
]
