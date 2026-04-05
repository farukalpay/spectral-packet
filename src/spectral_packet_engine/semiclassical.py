"""WKB approximation and semiclassical methods.

The Wentzel--Kramers--Brillouin (WKB) approximation provides an
asymptotic solution to the one-dimensional Schrodinger equation in
the limit where the de Broglie wavelength varies slowly compared to
the scale on which the potential changes.

In the classically allowed region (E > V):

    psi(x) ~ A / sqrt(|p(x)|) exp( +/- i/hbar int p dx )

where p(x) = sqrt(2m(E - V(x))) is the classical momentum.

In the classically forbidden region (E < V):

    psi(x) ~ B / sqrt(|kappa(x)|) exp( -1/hbar int kappa dx )

where kappa(x) = sqrt(2m(V(x) - E)).

The Bohr--Sommerfeld quantization condition is:

    oint p dx = (n + 1/2) 2*pi*hbar       (full loop)
    int_{x1}^{x2} p dx = (n + 1/2) pi*hbar   (half loop)

WKB tunneling through a barrier of height V > E gives a transmission
coefficient:

    T ~ exp( -2/hbar int_{x1}^{x2} kappa(x) dx )
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
class WKBResult:
    """Result of the WKB approximation on a grid."""

    x_grid: Tensor
    classical_momentum: Tensor
    wkb_wavefunction: Tensor
    turning_points: Tensor
    phase_integral: Tensor
    is_classically_allowed: Tensor


@dataclass(frozen=True, slots=True)
class BohrSommerfeldResult:
    """Quantised energies from Bohr--Sommerfeld."""

    quantum_numbers: Tensor
    energies: Tensor
    action_integrals: Tensor
    exact_energies: Tensor | None


@dataclass(frozen=True, slots=True)
class TunnelingResult:
    """WKB tunneling probability through a barrier."""

    transmission: Tensor
    reflection: Tensor
    kappa_integral: Tensor
    barrier_width: Tensor


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def classical_momentum(
    energy: float,
    potential: Tensor,
    grid: Tensor,
    *,
    mass: float = 1.0,
) -> Tensor:
    r"""p(x) = sqrt(2m(E - V(x))).

    Returns a *complex* tensor.  In the classically allowed region the
    result is real; in the forbidden region the result is purely imaginary.
    """
    potential = coerce_tensor(potential, dtype=torch.float64)
    grid = coerce_tensor(grid, dtype=torch.float64)
    diff = 2.0 * mass * (energy - potential)
    # Use complex square root so forbidden regions yield imaginary values
    diff_complex = diff.to(dtype=torch.complex128)
    return torch.sqrt(diff_complex)


def find_turning_points(
    energy: float,
    potential: Tensor,
    grid: Tensor,
) -> Tensor:
    """Find classical turning points where E = V(x).

    Turning points are located by detecting sign changes in (E - V(x))
    and linearly interpolating to find the zero crossings.
    """
    potential = coerce_tensor(potential, dtype=torch.float64)
    grid = coerce_tensor(grid, dtype=torch.float64)
    diff = energy - potential
    sign_changes = diff[:-1] * diff[1:]
    idx = torch.where(sign_changes < 0)[0]

    if idx.numel() == 0:
        return torch.tensor([], dtype=torch.float64, device=grid.device)

    # Linear interpolation between adjacent grid points
    d0 = diff[idx]
    d1 = diff[idx + 1]
    fraction = d0 / (d0 - d1)
    turning = grid[idx] + fraction * (grid[idx + 1] - grid[idx])
    return turning


def wkb_wavefunction(
    energy: float,
    potential: Tensor,
    grid: Tensor,
    *,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> WKBResult:
    r"""Compute the WKB wavefunction approximation.

    In the classically allowed region:

        psi(x) ~ A / sqrt(|p(x)|) exp(i/hbar int p dx)

    In the classically forbidden region:

        psi(x) ~ B / sqrt(|kappa(x)|) exp(-1/hbar int kappa dx)

    Near turning points a small floor is applied to |p| and |kappa| to
    avoid the 1/sqrt(p) singularity.
    """
    potential = coerce_tensor(potential, dtype=torch.float64)
    grid = coerce_tensor(grid, dtype=torch.float64)

    p_complex = classical_momentum(energy, potential, grid, mass=mass)
    allowed = (energy - potential) >= 0  # bool mask

    turning = find_turning_points(energy, potential, grid)

    # Real momentum in allowed region, real kappa in forbidden region
    p_real = torch.sqrt(torch.clamp(2.0 * mass * (energy - potential), min=0.0))
    kappa_real = torch.sqrt(torch.clamp(2.0 * mass * (potential - energy), min=0.0))

    # Cumulative phase integral (allowed) or decay integral (forbidden)
    dx = grid[1:] - grid[:-1]
    # Trapezoidal cumulative integrals
    p_avg = (p_real[:-1] + p_real[1:]) / 2.0
    kappa_avg = (kappa_real[:-1] + kappa_real[1:]) / 2.0
    phase_increments = p_avg * dx / hbar
    decay_increments = kappa_avg * dx / hbar

    phase_cumulative = torch.zeros_like(grid)
    decay_cumulative = torch.zeros_like(grid)
    phase_cumulative[1:] = torch.cumsum(phase_increments, dim=0)
    decay_cumulative[1:] = torch.cumsum(decay_increments, dim=0)

    # Build wavefunction (complex)
    psi = torch.zeros(grid.shape[0], dtype=torch.complex128, device=grid.device)
    # Small floor to avoid division by zero at turning points
    eps = 1e-12
    safe_p = torch.clamp(p_real, min=eps)
    safe_kappa = torch.clamp(kappa_real, min=eps)

    psi_allowed = (1.0 / torch.sqrt(safe_p)) * torch.exp(1j * phase_cumulative)
    psi_forbidden = (1.0 / torch.sqrt(safe_kappa)) * torch.exp(-decay_cumulative)

    psi = torch.where(allowed, psi_allowed, psi_forbidden.to(dtype=torch.complex128))

    # Normalise
    dx_all = grid[1] - grid[0]  # uniform grid assumed for normalisation
    norm = torch.sqrt(torch.trapezoid(torch.abs(psi) ** 2, grid))
    safe_norm = torch.where(norm > 0, norm, torch.ones_like(norm))
    psi = psi / safe_norm

    # Total phase integral across entire allowed region
    phase_total = torch.sum(
        torch.where(allowed[:-1] & allowed[1:], phase_increments, torch.zeros_like(phase_increments))
    ) * hbar  # undo the /hbar to get actual int p dx

    return WKBResult(
        x_grid=grid,
        classical_momentum=p_complex,
        wkb_wavefunction=psi,
        turning_points=turning,
        phase_integral=phase_total,
        is_classically_allowed=allowed,
    )


def bohr_sommerfeld_quantization(
    potential_fn: Callable[[Tensor], Tensor],
    domain_left: float,
    domain_right: float,
    *,
    mass: float = 1.0,
    hbar: float = 1.0,
    num_states: int = 10,
    energy_search_points: int = 1000,
) -> BohrSommerfeldResult:
    r"""Find quantised energies from the Bohr--Sommerfeld condition.

    The half-loop quantisation condition is:

        \int_{x_1}^{x_2} \sqrt{2m(E - V(x))} dx = (n + 1/2) \pi \hbar

    Algorithm
    ---------
    1. Build a fine spatial grid and evaluate V(x).
    2. Create an energy grid from V_min to near V_max.
    3. For each trial energy, compute the action integral J(E) across the
       classically allowed region.
    4. Find energies where J(E) = (n + 1/2) pi*hbar by interpolation.
    """
    x_grid = torch.linspace(domain_left, domain_right, 2000, dtype=torch.float64)
    V = potential_fn(x_grid)
    V = coerce_tensor(V, dtype=torch.float64)

    V_min = V.min().item()
    V_max = V.max().item()
    # Search energies slightly above V_min up to V_max
    E_grid = torch.linspace(V_min + 1e-10, V_max - 1e-10, energy_search_points, dtype=torch.float64)

    dx = x_grid[1:] - x_grid[:-1]

    # Compute action integral for each trial energy
    J_values = torch.zeros(energy_search_points, dtype=torch.float64)
    for i in range(energy_search_points):
        E_trial = E_grid[i].item()
        diff = 2.0 * mass * (E_trial - V)
        p = torch.sqrt(torch.clamp(diff, min=0.0))
        # Trapezoidal integration
        p_avg = (p[:-1] + p[1:]) / 2.0
        J_values[i] = torch.sum(p_avg * dx)

    # Find quantised energies where J(E) = (n + 0.5) * pi * hbar
    found_n: list[int] = []
    found_E: list[float] = []
    found_J: list[float] = []

    for n in range(num_states):
        target_val = float((n + 0.5) * torch.pi * hbar)
        # Find first crossing of J_values through target_val
        below = J_values < target_val
        above = J_values >= target_val
        if not below.any() or not above.any():
            continue
        crossings = below[:-1] & above[1:]
        cross_idx = torch.where(crossings)[0]
        if cross_idx.numel() == 0:
            continue
        idx = cross_idx[0].item()
        # Linear interpolation
        J0 = J_values[idx].item()
        J1 = J_values[idx + 1].item()
        E0 = E_grid[idx].item()
        E1 = E_grid[idx + 1].item()
        if abs(J1 - J0) < 1e-30:
            continue
        frac = (target_val - J0) / (J1 - J0)
        E_quantised = E0 + frac * (E1 - E0)
        found_n.append(n)
        found_E.append(E_quantised)
        found_J.append(target_val)

    if len(found_n) == 0:
        return BohrSommerfeldResult(
            quantum_numbers=torch.tensor([], dtype=torch.float64),
            energies=torch.tensor([], dtype=torch.float64),
            action_integrals=torch.tensor([], dtype=torch.float64),
            exact_energies=None,
        )

    return BohrSommerfeldResult(
        quantum_numbers=torch.tensor(found_n, dtype=torch.float64),
        energies=torch.tensor(found_E, dtype=torch.float64),
        action_integrals=torch.tensor(found_J, dtype=torch.float64),
        exact_energies=None,
    )


def tunneling_probability(
    energy: float,
    potential: Tensor,
    grid: Tensor,
    *,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> TunnelingResult:
    r"""WKB tunneling probability through a potential barrier.

    T ~ exp(-2/hbar \int_{x_1}^{x_2} kappa(x) dx)

    where kappa(x) = sqrt(2m(V(x) - E)) inside the classically forbidden
    region (V > E).
    """
    potential = coerce_tensor(potential, dtype=torch.float64)
    grid = coerce_tensor(grid, dtype=torch.float64)

    forbidden = potential > energy
    kappa = torch.sqrt(torch.clamp(2.0 * mass * (potential - energy), min=0.0))

    dx = grid[1:] - grid[:-1]
    kappa_avg = (kappa[:-1] + kappa[1:]) / 2.0
    # Only integrate through the forbidden region
    in_barrier = forbidden[:-1] & forbidden[1:]
    kappa_dx = torch.where(in_barrier, kappa_avg * dx, torch.zeros_like(dx))
    kappa_integral = torch.sum(kappa_dx)

    # Barrier width
    forbidden_float = forbidden.to(dtype=torch.float64)
    barrier_dx = torch.where(in_barrier, dx, torch.zeros_like(dx))
    width = torch.sum(barrier_dx)

    T = torch.exp(-2.0 * kappa_integral / hbar)
    R = 1.0 - T

    return TunnelingResult(
        transmission=T,
        reflection=R,
        kappa_integral=kappa_integral,
        barrier_width=width,
    )


__all__ = [
    "BohrSommerfeldResult",
    "TunnelingResult",
    "WKBResult",
    "bohr_sommerfeld_quantization",
    "classical_momentum",
    "find_turning_points",
    "tunneling_probability",
    "wkb_wavefunction",
]
