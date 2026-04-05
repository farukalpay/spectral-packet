"""Energy functionals and conservation checks for spectral states.

The total energy of a quantum state in the infinite well is purely kinetic
(potential is zero inside the well, infinite outside).  For profile data
(real-valued density functions), we define an analogous "spectral energy"
that measures the smoothness cost of the profile — useful for regularization
and quality assessment.

Conservation analysis:
- For quantum propagation: E(t) = E(0) exactly under unitary evolution.
  Any deviation is numerical error from truncation or time-stepping.
- For profile workflows: the spectral energy budget tells you whether
  your compression is preserving the right physics.

This module also provides the energy spectrum (energy per mode) which is
the fundamental diagnostic for understanding where the physics lives in
mode space.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from spectral_packet_engine.basis import InfiniteWellBasis
from spectral_packet_engine.domain import coerce_tensor

Tensor = torch.Tensor


def mode_energies(basis: InfiniteWellBasis) -> Tensor:
    """Return the eigenenergies E_n = (n*pi*hbar)^2 / (2*m*L^2) for each mode."""
    return basis.energies


def kinetic_energy(coefficients: Tensor, basis: InfiniteWellBasis) -> Tensor:
    """Total kinetic energy from spectral coefficients.

    <T> = sum_n |c_n|^2 * E_n

    For the infinite well, this is also the total energy since V=0 inside.
    """
    coeffs = coerce_tensor(coefficients, device=basis.domain.device)
    E_n = basis.energies

    if torch.is_complex(coeffs):
        weights = torch.abs(coeffs) ** 2
    else:
        weights = coeffs.to(dtype=basis.domain.real_dtype) ** 2

    if weights.ndim == 1:
        return torch.sum(weights * E_n)
    if weights.ndim == 2:
        return torch.sum(weights * E_n[None, :], dim=-1)
    raise ValueError("coefficients must be one- or two-dimensional")


def energy_spectrum(coefficients: Tensor, basis: InfiniteWellBasis) -> Tensor:
    """Energy contributed by each mode: |c_n|^2 * E_n.

    The energy spectrum reveals which modes carry the physics.
    Modes with negligible energy contribution can be safely truncated.
    """
    coeffs = coerce_tensor(coefficients, device=basis.domain.device)
    E_n = basis.energies

    if torch.is_complex(coeffs):
        weights = torch.abs(coeffs) ** 2
    else:
        weights = coeffs.to(dtype=basis.domain.real_dtype) ** 2

    if weights.ndim == 1:
        return weights * E_n
    if weights.ndim == 2:
        return weights * E_n[None, :]
    raise ValueError("coefficients must be one- or two-dimensional")


def energy_per_mode_fraction(coefficients: Tensor, basis: InfiniteWellBasis) -> Tensor:
    """Fractional energy contribution per mode: |c_n|^2 * E_n / <T>.

    Sums to 1 across modes. This is the energy-weighted version of the
    modal weight distribution — more physically meaningful than raw
    |c_n|^2 because it accounts for the increasing energy cost of
    higher modes.
    """
    spectrum = energy_spectrum(coefficients, basis)
    total = torch.sum(spectrum, dim=-1, keepdim=True)
    safe_total = torch.where(total > 0, total, torch.ones_like(total))
    return spectrum / safe_total


@dataclass(frozen=True, slots=True)
class EnergyConservationReport:
    """Detailed energy conservation analysis during propagation."""
    initial_energy: Tensor
    final_energy: Tensor
    absolute_error: Tensor
    relative_error: Tensor
    max_deviation: Tensor         # max deviation across all time steps (if time series)
    norm_initial: Tensor
    norm_final: Tensor
    norm_relative_error: Tensor
    is_conserved: bool            # True if relative error < tolerance


def check_energy_conservation(
    initial_coefficients: Tensor,
    propagated_coefficients: Tensor,
    basis: InfiniteWellBasis,
    *,
    tolerance: float = 1e-6,
) -> EnergyConservationReport:
    """Check energy and norm conservation during spectral propagation.

    For exact unitary time evolution, both the total energy and the norm
    are conserved.  Deviations indicate:
    - Mode truncation effects (losing energy to unresolved modes)
    - Numerical precision loss (catastrophic cancellation)
    - Time-stepping errors (if not using exact phase factors)
    """
    init = coerce_tensor(initial_coefficients, device=basis.domain.device)
    prop = coerce_tensor(propagated_coefficients, device=basis.domain.device)

    E_init = kinetic_energy(init, basis)
    E_prop = kinetic_energy(prop, basis)

    # Handle time series: prop may be (num_times, num_modes)
    if prop.ndim == 2 and init.ndim == 1:
        E_all = E_prop
        abs_error = torch.abs(E_all - E_init)
        max_dev = torch.max(abs_error)
        E_final = E_all[-1]
    else:
        E_final = E_prop
        abs_error = torch.abs(E_final - E_init)
        max_dev = abs_error

    safe_E_init = torch.where(E_init > 0, E_init, torch.ones_like(E_init))
    rel_error_energy = torch.abs(E_final - E_init) / safe_E_init

    # Norm conservation
    if torch.is_complex(init):
        norm_init = torch.sum(torch.abs(init) ** 2, dim=-1)
    else:
        norm_init = torch.sum(init ** 2, dim=-1)

    if torch.is_complex(prop):
        norm_prop = torch.sum(torch.abs(prop) ** 2, dim=-1)
    else:
        norm_prop = torch.sum(prop ** 2, dim=-1)

    if prop.ndim == 2 and init.ndim == 1:
        norm_final = norm_prop[-1]
    else:
        norm_final = norm_prop

    safe_norm = torch.where(norm_init > 0, norm_init, torch.ones_like(norm_init))
    norm_rel_error = torch.abs(norm_final - norm_init) / safe_norm

    conserved = bool(rel_error_energy.item() < tolerance and norm_rel_error.item() < tolerance)

    return EnergyConservationReport(
        initial_energy=E_init,
        final_energy=E_final,
        absolute_error=torch.abs(E_final - E_init),
        relative_error=rel_error_energy,
        max_deviation=max_dev,
        norm_initial=norm_init,
        norm_final=norm_final,
        norm_relative_error=norm_rel_error,
        is_conserved=conserved,
    )


@dataclass(frozen=True, slots=True)
class SpectralEnergyBudget:
    """Energy budget analysis for profile compression.

    Shows how the spectral energy is distributed across modes and
    what fraction is retained after truncation to N modes.
    """
    total_energy: Tensor
    mode_energies: Tensor        # E_n for each mode
    mode_contributions: Tensor   # |c_n|^2 * E_n
    mode_fractions: Tensor       # fractional contribution
    cumulative_fraction: Tensor  # cumulative energy captured
    energy_tail: Tensor          # 1 - cumulative (energy lost by truncation)


def compute_energy_budget(
    coefficients: Tensor,
    basis: InfiniteWellBasis,
) -> SpectralEnergyBudget:
    """Compute the full energy budget for a spectral expansion.

    This tells you exactly how much physics you're capturing at each
    truncation level, weighted by energy rather than raw amplitude.
    """
    coeffs = coerce_tensor(coefficients, device=basis.domain.device)
    if coeffs.ndim != 1:
        raise ValueError("expects a one-dimensional coefficient vector")

    E_n = basis.energies
    contributions = energy_spectrum(coeffs, basis)
    total = torch.sum(contributions)
    safe_total = torch.where(total > 0, total, torch.ones_like(total))
    fractions = contributions / safe_total
    cumulative = torch.cumsum(fractions, dim=0)
    tail = 1 - cumulative

    return SpectralEnergyBudget(
        total_energy=total,
        mode_energies=E_n,
        mode_contributions=contributions,
        mode_fractions=fractions,
        cumulative_fraction=cumulative,
        energy_tail=tail,
    )


__all__ = [
    "EnergyConservationReport",
    "SpectralEnergyBudget",
    "check_energy_conservation",
    "compute_energy_budget",
    "energy_per_mode_fraction",
    "energy_spectrum",
    "kinetic_energy",
    "mode_energies",
]
