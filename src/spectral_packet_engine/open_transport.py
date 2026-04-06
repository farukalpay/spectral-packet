"""Open-boundary transport analysis.

This module extends the bounded-domain spectral core to open-boundary
physics where particles can enter and leave the computational region.
It provides:

1. **Complex absorbing potentials (CAPs)** — smooth imaginary potentials
   at domain edges that absorb outgoing waves without spurious reflection.
2. **CAP-augmented propagation** — split-operator propagation with CAP
   damping, yielding survival probability and decay rates.
3. **Resonance pole extraction** — Breit-Wigner fitting to S-matrix
   eigenphases for precise resonance positions and widths.
4. **Decay width and lifetime** — Γ and τ = ℏ/Γ from resonance widths.
5. **Wigner-Smith delay time** — energy derivative of S-matrix phase.

All functions compose existing infrastructure (``scattering.py``,
``split_operator.py``, ``eigensolver.py``) rather than reimplementing
core algorithms.  The open-boundary picture is honest: CAPs are an
approximation whose quality depends on the absorber profile and strength.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from spectral_packet_engine.domain import InfiniteWell1D, coerce_tensor, complex_dtype_for
from spectral_packet_engine.scattering import (
    PotentialSegment,
    ScatteringResult,
    SMatrixResult,
    scattering_spectrum,
    s_matrix_from_transfer,
    total_transfer_matrix,
)
from spectral_packet_engine.split_operator import (
    gaussian_wavepacket_on_grid,
    split_operator_propagate,
)

Tensor = torch.Tensor

_RDTYPE = torch.float64
_CDTYPE = torch.complex128


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CAPProfile:
    """Complex absorbing potential profile on a grid.

    The CAP is a smooth imaginary potential W(x) that absorbs outgoing
    waves.  It is zero in the physical region and ramps up at the edges.

    W(x) = -i * eta * f(x)

    where f(x) is the spatial profile (polynomial, sine-squared, etc.)
    and eta is the absorbing strength.
    """

    grid: Tensor           # (num_points,)
    values: Tensor         # (num_points,) — real envelope f(x), >=0
    strength: float        # eta
    cap_width: float       # width of the absorbing region on each side
    cap_order: int         # polynomial order n for power-law CAPs
    cap_type: str          # "polynomial", "sine_squared"
    assumptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CAPPropagationResult:
    """Result of CAP-augmented wavepacket propagation.

    The wavepacket is propagated under H = T + V_real - i*W where W is
    the complex absorbing potential.  The norm decays as the wavepacket
    is absorbed at the boundaries.
    """

    times: Tensor              # (num_saved,)
    survival_probability: Tensor  # (num_saved,) — ||ψ(t)||² / ||ψ(0)||²
    densities: Tensor          # (num_saved, num_points)
    grid: Tensor               # (num_points,)
    decay_rate: float          # fitted exponential decay rate Γ/ℏ
    half_life: float           # t_{1/2} = ℏ ln(2) / Γ
    initial_norm: float
    final_norm: float
    cap_profile: CAPProfile
    assumptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResonancePole:
    """A single resonance pole extracted from scattering data.

    E_res = E_0 - i Γ/2

    where E_0 is the resonance position and Γ is the full width.
    """

    energy: float         # E_0 — resonance centre
    width: float          # Γ — full width at half maximum
    lifetime: float       # τ = ℏ / Γ
    quality_factor: float  # Q = E_0 / Γ
    fit_residual: float   # RMS residual of Breit-Wigner fit
    assumptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResonanceExtractionResult:
    """Collection of resonance poles from a scattering spectrum."""

    poles: tuple[ResonancePole, ...]
    scattering: ScatteringResult
    num_resonances: int
    assumptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DelayTimeResult:
    """Wigner-Smith time delay from S-matrix phase derivative.

    τ_W(E) = ℏ dδ/dE

    where δ(E) is the scattering phase shift extracted from the S-matrix.
    """

    energies: Tensor       # (num_E,)
    phase_shift: Tensor    # (num_E,) — δ(E)
    delay_time: Tensor     # (num_E,) — τ_W(E)
    peak_delay_energy: float
    peak_delay_value: float
    assumptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OpenTransportSummary:
    """Combined open-boundary transport analysis.

    Bundles resonance extraction, delay times, and optional CAP
    propagation into a single summary.
    """

    resonances: ResonanceExtractionResult
    delay_times: DelayTimeResult
    cap_propagation: CAPPropagationResult | None
    num_resonances: int
    narrowest_resonance: ResonancePole | None
    widest_resonance: ResonancePole | None
    assumptions: tuple[str, ...]


# ---------------------------------------------------------------------------
# Complex absorbing potential
# ---------------------------------------------------------------------------


def build_cap_profile(
    domain: InfiniteWell1D,
    num_points: int = 128,
    *,
    cap_width_fraction: float = 0.15,
    strength: float = 5.0,
    order: int = 3,
    cap_type: str = "polynomial",
) -> CAPProfile:
    """Build a complex absorbing potential profile on the domain grid.

    Parameters
    ----------
    domain : InfiniteWell1D
        Spatial domain.
    num_points : int
        Grid resolution.
    cap_width_fraction : float
        Fraction of domain length for the absorbing region on each side.
    strength : float
        Absorbing strength eta (imaginary prefactor).
    order : int
        Polynomial order n.  Higher order = sharper ramp.
    cap_type : str
        "polynomial" for W ~ (x/d)^n, or "sine_squared" for W ~ sin²(πx/2d).

    Returns
    -------
    CAPProfile
    """
    if cap_width_fraction <= 0 or cap_width_fraction >= 0.5:
        raise ValueError("cap_width_fraction must be in (0, 0.5)")
    if strength <= 0:
        raise ValueError("strength must be positive")
    if order < 1:
        raise ValueError("order must be >= 1")
    if cap_type not in ("polynomial", "sine_squared"):
        raise ValueError("cap_type must be 'polynomial' or 'sine_squared'")

    grid = domain.grid(num_points)
    x_min = float(grid[0].item())
    x_max = float(grid[-1].item())
    L = x_max - x_min
    cap_width = cap_width_fraction * L

    values = torch.zeros(num_points, dtype=_RDTYPE, device=domain.device)

    for i in range(num_points):
        x = float(grid[i].item())
        # Left absorber
        if x < x_min + cap_width:
            xi = (x_min + cap_width - x) / cap_width  # 1 at edge, 0 at boundary
            if cap_type == "polynomial":
                values[i] = xi ** order
            else:
                values[i] = math.sin(math.pi * xi / 2.0) ** 2
        # Right absorber
        elif x > x_max - cap_width:
            xi = (x - (x_max - cap_width)) / cap_width
            if cap_type == "polynomial":
                values[i] = xi ** order
            else:
                values[i] = math.sin(math.pi * xi / 2.0) ** 2

    return CAPProfile(
        grid=grid,
        values=values,
        strength=strength,
        cap_width=cap_width,
        cap_order=order,
        cap_type=cap_type,
        assumptions=(
            f"CAP type: {cap_type} with order {order}.",
            f"Absorber width: {cap_width:.4f} ({cap_width_fraction*100:.0f}% of domain on each side).",
            f"Absorbing strength eta = {strength:.2f}.",
            "CAP quality depends on the absorber being wide enough relative to the de Broglie wavelength.",
        ),
    )


# ---------------------------------------------------------------------------
# CAP-augmented propagation
# ---------------------------------------------------------------------------


def cap_augmented_propagation(
    *,
    potential_fn: Callable[[Tensor], Tensor],
    domain: InfiniteWell1D,
    center: float,
    width: float,
    wavenumber: float,
    total_time: float,
    num_points: int = 128,
    num_steps: int = 2000,
    save_every: int = 20,
    cap_width_fraction: float = 0.15,
    cap_strength: float = 5.0,
    cap_order: int = 3,
    hbar: float = 1.0,
) -> CAPPropagationResult:
    """Propagate a wavepacket with CAP-augmented Hamiltonian.

    The effective Hamiltonian is H_eff = T + V(x) - i*eta*W(x), where
    W(x) is the absorbing profile.  The non-Hermitian term causes the
    norm to decay, giving the survival probability.

    Parameters
    ----------
    potential_fn : callable
        Real potential V(x).
    domain : InfiniteWell1D
        Spatial domain.
    center, width, wavenumber : float
        Initial Gaussian wavepacket parameters.
    total_time : float
        Total propagation time.
    num_points, num_steps, save_every : int
        Grid and time-stepping parameters.
    cap_width_fraction, cap_strength, cap_order : float
        CAP parameters.
    hbar : float
        Reduced Planck constant.

    Returns
    -------
    CAPPropagationResult
    """
    cap = build_cap_profile(
        domain, num_points,
        cap_width_fraction=cap_width_fraction,
        strength=cap_strength,
        order=cap_order,
    )

    grid = domain.grid(num_points)
    cdtype = complex_dtype_for(domain.real_dtype)

    # Build complex potential: V_real - i * eta * W
    V_real = potential_fn(grid).to(domain.real_dtype)
    V_complex = V_real.to(cdtype) - 1j * cap.strength * cap.values.to(cdtype)

    # Initial wavepacket
    psi0 = gaussian_wavepacket_on_grid(grid, center, width, wavenumber, hbar=hbar)

    # Manual split-operator propagation with complex potential
    # We can't use the standard split_operator_propagate directly because
    # it expects a real potential.  Instead we do the Trotter splitting
    # manually with the complex potential.
    from spectral_packet_engine.basis import eigenenergies, sine_basis_matrix

    num_modes = max(num_points - 2, 1)
    modes = torch.arange(1, num_modes + 1, dtype=domain.real_dtype, device=domain.device)

    # Sine transform matrices
    B = sine_basis_matrix(domain, modes, grid).to(cdtype)
    S_fwd = torch.linalg.pinv(B)
    S_inv = B

    E_n = eigenenergies(domain, modes)
    dt = total_time / num_steps

    # Phase operators
    V_half = torch.exp(-1j * V_complex * dt / (2.0 * hbar))
    kinetic_phase = torch.exp(-1j * E_n.to(cdtype) * dt / hbar)

    psi = psi0.to(cdtype).clone()
    initial_norm = float(torch.trapezoid((psi.conj() * psi).real, grid).item())

    num_saved = num_steps // save_every + 1
    survival = torch.zeros(num_saved, dtype=_RDTYPE, device=domain.device)
    densities = torch.zeros(num_saved, num_points, dtype=_RDTYPE, device=domain.device)
    times = torch.zeros(num_saved, dtype=_RDTYPE, device=domain.device)

    # Save initial
    save_idx = 0
    survival[0] = 1.0
    densities[0] = (psi.conj() * psi).real
    times[0] = 0.0
    save_idx = 1

    for step in range(1, num_steps + 1):
        # Trotter step: V/2 -> T -> V/2
        psi = V_half * psi
        coeffs = S_fwd @ psi
        coeffs = kinetic_phase * coeffs
        psi = S_inv @ coeffs
        psi = V_half * psi

        if step % save_every == 0 and save_idx < num_saved:
            dens = (psi.conj() * psi).real
            norm = float(torch.trapezoid(dens, grid).item())
            survival[save_idx] = norm / max(initial_norm, 1e-30)
            densities[save_idx] = dens
            times[save_idx] = step * dt
            save_idx += 1

    survival = survival[:save_idx]
    densities = densities[:save_idx]
    times = times[:save_idx]

    final_norm = float(torch.trapezoid((psi.conj() * psi).real, grid).item())

    # Fit exponential decay: P(t) ~ exp(-Γt/ℏ)
    # Use log-linear fit on survival > threshold to avoid log(0)
    mask = survival > 0.01
    if mask.sum() > 2:
        t_fit = times[mask]
        log_p = torch.log(survival[mask])
        # Linear fit: log(P) = a + b*t => b = -Γ/ℏ
        t_mean = t_fit.mean()
        lp_mean = log_p.mean()
        b = ((t_fit - t_mean) * (log_p - lp_mean)).sum() / max(((t_fit - t_mean) ** 2).sum().item(), 1e-30)
        decay_rate = max(-float(b.item()) / hbar, 0.0)
    else:
        decay_rate = 0.0

    half_life = hbar * math.log(2.0) / max(decay_rate, 1e-30) if decay_rate > 0 else float('inf')

    return CAPPropagationResult(
        times=times.detach(),
        survival_probability=survival.detach(),
        densities=densities.detach(),
        grid=grid.detach(),
        decay_rate=decay_rate,
        half_life=half_life,
        initial_norm=initial_norm,
        final_norm=final_norm,
        cap_profile=cap,
        assumptions=(
            "CAP propagation is non-unitary: norm decay tracks absorption at boundaries.",
            f"Fitted exponential decay rate: {decay_rate:.6f} (Γ/ℏ units).",
            f"Half-life: {half_life:.6f}.",
            "The decay rate is physical only if the CAP is wide and smooth enough "
            "that spurious reflections are negligible.",
        ),
    )


# ---------------------------------------------------------------------------
# Resonance pole extraction
# ---------------------------------------------------------------------------


def _breit_wigner_fit(
    energies: Tensor,
    transmission: Tensor,
    peak_idx: int,
) -> tuple[float, float, float]:
    """Fit a Breit-Wigner profile to a transmission peak.

    T(E) = Γ² / (4(E - E_0)² + Γ²)

    Returns (E_0, Γ, rms_residual).
    """
    E = energies.to(_RDTYPE)
    T = transmission.to(_RDTYPE)

    E_0 = float(E[peak_idx].item())
    T_peak = float(T[peak_idx].item())

    # Estimate Γ from FWHM: find where T drops to T_peak/2
    half_max = T_peak / 2.0
    dE = float((E[1] - E[0]).item())

    left = peak_idx
    while left > 0 and float(T[left].item()) > half_max:
        left -= 1
    right = peak_idx
    while right < len(T) - 1 and float(T[right].item()) > half_max:
        right += 1

    gamma = float((E[right] - E[left]).item())
    gamma = max(gamma, 2.0 * dE)  # at least 2 grid spacings

    # Compute Breit-Wigner with these parameters
    fitted = gamma ** 2 / (4.0 * (E - E_0) ** 2 + gamma ** 2)

    # Only compute residual near the peak (within 3Γ)
    mask = torch.abs(E - E_0) < 3.0 * gamma
    if mask.sum() > 0:
        residual = float(torch.sqrt(torch.mean((T[mask] - fitted[mask]) ** 2)).item())
    else:
        residual = 0.0

    return E_0, gamma, residual


def extract_resonance_poles(
    segments: list[PotentialSegment],
    *,
    energy_min: float,
    energy_max: float,
    num_energies: int = 500,
    mass: float = 1.0,
    hbar: float = 1.0,
    peak_threshold: float = 0.5,
) -> ResonanceExtractionResult:
    """Extract resonance poles from a scattering spectrum.

    Computes T(E), detects transmission peaks, and fits Breit-Wigner
    profiles to extract precise resonance positions and widths.

    Parameters
    ----------
    segments : list[PotentialSegment]
        The potential structure.
    energy_min, energy_max : float
        Energy scan range.
    num_energies : int
        Number of energy sample points.
    mass, hbar : float
        Physical constants.
    peak_threshold : float
        Minimum T value for a peak to qualify as a resonance.

    Returns
    -------
    ResonanceExtractionResult
    """
    spectrum = scattering_spectrum(
        segments,
        energy_min=energy_min,
        energy_max=energy_max,
        num_energies=num_energies,
        mass=mass,
        hbar=hbar,
    )

    E = spectrum.energies
    T = spectrum.transmission

    # Detect peaks
    peak_indices: list[int] = []
    for i in range(1, num_energies - 1):
        if (
            float(T[i].item()) > float(T[i - 1].item())
            and float(T[i].item()) > float(T[i + 1].item())
            and float(T[i].item()) > peak_threshold
        ):
            peak_indices.append(i)

    poles: list[ResonancePole] = []
    for idx in peak_indices:
        E_0, gamma, residual = _breit_wigner_fit(E, T, idx)
        lifetime = hbar / max(gamma, 1e-30)
        Q = E_0 / max(gamma, 1e-30)

        poles.append(ResonancePole(
            energy=E_0,
            width=gamma,
            lifetime=lifetime,
            quality_factor=Q,
            fit_residual=residual,
            assumptions=(
                f"Breit-Wigner fit: E_0 = {E_0:.6f}, Γ = {gamma:.6f}.",
                f"Lifetime τ = ℏ/Γ = {lifetime:.6f}.",
                f"Quality factor Q = E_0/Γ = {Q:.2f}.",
            ),
        ))

    return ResonanceExtractionResult(
        poles=tuple(poles),
        scattering=spectrum,
        num_resonances=len(poles),
        assumptions=(
            f"Found {len(poles)} resonance(s) with T > {peak_threshold} in [{energy_min}, {energy_max}].",
            "Breit-Wigner fitting assumes isolated resonances (non-overlapping widths).",
            "Resonance positions are approximate; accuracy depends on energy grid resolution.",
        ),
    )


# ---------------------------------------------------------------------------
# Wigner-Smith delay time
# ---------------------------------------------------------------------------


def compute_delay_time(
    segments: list[PotentialSegment],
    *,
    energy_min: float,
    energy_max: float,
    num_energies: int = 500,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> DelayTimeResult:
    """Compute the Wigner-Smith time delay from S-matrix eigenphases.

    τ_W(E) = ℏ dδ/dE

    where δ(E) is the transmission phase shift.

    Parameters
    ----------
    segments : list[PotentialSegment]
        The potential structure.
    energy_min, energy_max : float
        Energy scan range (both must be positive).
    num_energies : int
        Number of energy sample points.
    mass, hbar : float
        Physical constants.

    Returns
    -------
    DelayTimeResult
    """
    if energy_min <= 0:
        raise ValueError("energy_min must be positive")
    if energy_max <= energy_min:
        raise ValueError("energy_max must exceed energy_min")

    energies = torch.linspace(energy_min, energy_max, num_energies, dtype=_RDTYPE)
    phases = torch.zeros(num_energies, dtype=_RDTYPE)

    for i in range(num_energies):
        E_i = float(energies[i].item())
        tm = total_transfer_matrix(E_i, segments, mass=mass, hbar=hbar)
        sm = s_matrix_from_transfer(tm.M)
        # Transmission phase: arg(S_10) = arg(t)
        t_element = sm.S[1, 0]
        phases[i] = float(torch.angle(t_element).item())

    # Unwrap phase for smooth derivative
    phases_np = phases.clone()
    for i in range(1, num_energies):
        diff = float((phases_np[i] - phases_np[i - 1]).item())
        if diff > math.pi:
            phases_np[i:] -= 2 * math.pi
        elif diff < -math.pi:
            phases_np[i:] += 2 * math.pi

    # Numerical derivative: dδ/dE via central differences
    dE = float((energies[1] - energies[0]).item())
    delay = torch.zeros(num_energies, dtype=_RDTYPE)
    # Central differences for interior points
    delay[1:-1] = hbar * (phases_np[2:] - phases_np[:-2]) / (2.0 * dE)
    # Forward/backward for endpoints
    delay[0] = hbar * (phases_np[1] - phases_np[0]) / dE
    delay[-1] = hbar * (phases_np[-1] - phases_np[-2]) / dE

    peak_idx = int(torch.argmax(torch.abs(delay)).item())
    peak_energy = float(energies[peak_idx].item())
    peak_value = float(delay[peak_idx].item())

    return DelayTimeResult(
        energies=energies.detach(),
        phase_shift=phases_np.detach(),
        delay_time=delay.detach(),
        peak_delay_energy=peak_energy,
        peak_delay_value=peak_value,
        assumptions=(
            "Wigner-Smith delay time τ_W = ℏ dδ/dE from transmission phase.",
            "Phase unwrapping assumes smooth variation; rapid jumps may introduce artifacts.",
            f"Peak delay: τ = {peak_value:.6f} at E = {peak_energy:.6f}.",
        ),
    )


# ---------------------------------------------------------------------------
# Combined open-transport analysis
# ---------------------------------------------------------------------------


def analyze_open_transport(
    *,
    segments: list[PotentialSegment],
    energy_min: float,
    energy_max: float,
    num_energies: int = 500,
    mass: float = 1.0,
    hbar: float = 1.0,
    peak_threshold: float = 0.5,
    cap_domain: InfiniteWell1D | None = None,
    cap_potential_fn: Callable[[Tensor], Tensor] | None = None,
    cap_center: float | None = None,
    cap_width: float | None = None,
    cap_wavenumber: float | None = None,
    cap_total_time: float | None = None,
    cap_num_points: int = 128,
    cap_strength: float = 5.0,
) -> OpenTransportSummary:
    """Full open-boundary transport analysis.

    Combines resonance extraction, delay time computation, and optional
    CAP-augmented propagation into a single summary.

    Parameters
    ----------
    segments : list[PotentialSegment]
        The potential structure for scattering analysis.
    energy_min, energy_max : float
        Energy scan range.
    num_energies : int
        Number of energy sample points.
    mass, hbar : float
        Physical constants.
    peak_threshold : float
        Minimum T for resonance detection.
    cap_domain : InfiniteWell1D, optional
        Domain for CAP propagation.  If None, CAP is skipped.
    cap_potential_fn : callable, optional
        Real potential for CAP propagation.
    cap_center, cap_width, cap_wavenumber : float, optional
        Initial wavepacket parameters for CAP propagation.
    cap_total_time : float, optional
        Propagation time for CAP.
    cap_num_points : int
        Grid resolution for CAP.
    cap_strength : float
        CAP absorbing strength.

    Returns
    -------
    OpenTransportSummary
    """
    resonances = extract_resonance_poles(
        segments,
        energy_min=energy_min,
        energy_max=energy_max,
        num_energies=num_energies,
        mass=mass,
        hbar=hbar,
        peak_threshold=peak_threshold,
    )

    delay_times = compute_delay_time(
        segments,
        energy_min=energy_min,
        energy_max=energy_max,
        num_energies=num_energies,
        mass=mass,
        hbar=hbar,
    )

    # Optional CAP propagation
    cap_result = None
    if (
        cap_domain is not None
        and cap_potential_fn is not None
        and cap_center is not None
        and cap_width is not None
        and cap_wavenumber is not None
        and cap_total_time is not None
    ):
        cap_result = cap_augmented_propagation(
            potential_fn=cap_potential_fn,
            domain=cap_domain,
            center=cap_center,
            width=cap_width,
            wavenumber=cap_wavenumber,
            total_time=cap_total_time,
            num_points=cap_num_points,
            cap_strength=cap_strength,
            hbar=hbar,
        )

    poles = resonances.poles
    narrowest = min(poles, key=lambda p: p.width) if poles else None
    widest = max(poles, key=lambda p: p.width) if poles else None

    assumptions_list = [
        f"Open-transport analysis over E in [{energy_min}, {energy_max}].",
        f"Detected {len(poles)} resonance(s).",
    ]
    if narrowest:
        assumptions_list.append(
            f"Narrowest resonance: E = {narrowest.energy:.6f}, Γ = {narrowest.width:.6f}, "
            f"Q = {narrowest.quality_factor:.1f}."
        )
    if cap_result:
        assumptions_list.append(
            f"CAP propagation: decay rate = {cap_result.decay_rate:.6f}, "
            f"half-life = {cap_result.half_life:.6f}."
        )
    else:
        assumptions_list.append("CAP propagation was not requested.")

    return OpenTransportSummary(
        resonances=resonances,
        delay_times=delay_times,
        cap_propagation=cap_result,
        num_resonances=len(poles),
        narrowest_resonance=narrowest,
        widest_resonance=widest,
        assumptions=tuple(assumptions_list),
    )


__all__ = [
    "CAPProfile",
    "CAPPropagationResult",
    "DelayTimeResult",
    "OpenTransportSummary",
    "ResonanceExtractionResult",
    "ResonancePole",
    "analyze_open_transport",
    "build_cap_profile",
    "cap_augmented_propagation",
    "compute_delay_time",
    "extract_resonance_poles",
]
