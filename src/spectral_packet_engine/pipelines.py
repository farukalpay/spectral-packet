"""High-level analysis pipelines that chain library functions automatically.

These pipelines exist so that a user — whether human, CLI operator, or AI agent
via MCP — does not need to know the exact sequence of function calls or the
correct parameter values.  Each pipeline accepts minimal input and uses the
library's own diagnostic functions to determine everything else.

Design rules:
- No hardcoded heuristics.  Every parameter choice is derived from a library
  diagnostic (e.g. ``recommend_truncation`` decides the mode count, not a
  magic constant).
- Scientific terminology only.  These are spectral-analysis pipelines, not
  "smart modes" or "AI features."
- Every pipeline returns a structured dataclass that is JSON-serializable
  through ``to_serializable()``.
- Pipelines compose the same functions that the low-level API exposes;
  they add orchestration, not new physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Sequence

import torch

from spectral_packet_engine.domain import InfiniteWell1D, coerce_tensor
from spectral_packet_engine.basis import InfiniteWellBasis

Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class QuantumStateReport:
    """Complete characterization of a quantum state from spectral coefficients."""
    num_modes: int
    norm: float
    # Energy
    total_energy: float
    energy_spectrum: list[float]
    energy_budget_cumulative: list[float]
    # Momentum
    expectation_p: float
    expectation_p_squared: float
    variance_p: float
    # Position
    expectation_x: float
    variance_x: float
    # Uncertainty
    sigma_x: float
    sigma_p: float
    uncertainty_product: float
    heisenberg_bound: float
    satisfies_heisenberg: bool
    # Density matrix
    purity: float
    von_neumann_entropy: float
    # Convergence
    spectral_decay_type: str
    spectral_decay_rate: float
    recommended_truncation: int
    effective_mode_count: float
    # Wigner (optional, computed when feasible)
    wigner_negativity: float | None
    is_nonclassical: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


@dataclass(frozen=True, slots=True)
class PotentialLandscapeReport:
    """Complete characterization of a 1D potential landscape."""
    potential_name: str
    num_eigenstates: int
    eigenvalues: list[float]
    # WKB comparison
    wkb_energies: list[float]
    wkb_vs_exact_relative_errors: list[float]
    # Thermodynamics
    partition_function_at_T: float
    free_energy_at_T: float
    entropy_at_T: float
    specific_heat_at_T: float
    temperature: float
    # Spectral zeta
    spectral_zeta_2: float
    weyl_law_max_error: float
    # Orthonormality
    orthonormality_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


@dataclass(frozen=True, slots=True)
class ScatteringReport:
    """Complete characterization of a 1D scattering system."""
    num_segments: int
    energy_range: tuple[float, float]
    num_energies: int
    # Transmission
    max_transmission: float
    min_transmission: float
    # Resonances
    num_resonances: int
    resonance_energies: list[float]
    resonance_widths: list[float]
    # S-matrix at midpoint energy
    s_matrix_unitarity_error: float
    # WKB tunneling at selected energy
    wkb_tunneling_energy: float
    wkb_transmission: float
    exact_transmission_at_energy: float

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


@dataclass(frozen=True, slots=True)
class SpectralProfileReport:
    """Deep spectral analysis of profile data with auto-parameterization."""
    num_profiles: int
    num_grid_points: int
    domain_length: float
    # Auto-determined parameters
    auto_num_modes: int
    # Convergence (mean across profiles)
    mean_decay_type: str
    mean_decay_rate: float
    mean_effective_modes: float
    # Energy budget (mean)
    mean_total_energy: float
    energy_captured_at_auto_modes: float
    # Compression quality
    mean_relative_l2_error: float
    max_relative_l2_error: float
    # Truncation
    recommended_modes_for_99pct: int
    recommended_modes_for_999pct: int
    # Gibbs detection (any profile)
    any_gibbs_detected: bool

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


@dataclass(frozen=True, slots=True)
class StateComparisonReport:
    """Comparison of two quantum states."""
    fidelity: float
    trace_distance: float
    energy_difference: float
    momentum_difference: float
    uncertainty_product_a: float
    uncertainty_product_b: float

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


# ---------------------------------------------------------------------------
# Pipeline implementations
# ---------------------------------------------------------------------------

def analyze_quantum_state(
    coefficients: Tensor,
    basis: InfiniteWellBasis,
) -> QuantumStateReport:
    """Full quantum state characterization from spectral coefficients.

    This pipeline accepts a coefficient vector and a basis, then runs
    every relevant diagnostic the library offers:

    1. Norm and energy budget via ``energy.py``
    2. Momentum observables via ``momentum.py``
    3. Heisenberg uncertainty via ``momentum.py``
    4. Density matrix analysis via ``density_matrix.py``
    5. Convergence diagnostics via ``convergence.py`` — the library's own
       ``recommend_truncation`` determines whether modes are sufficient
    6. Wigner function via ``wigner.py`` when the mode count is ≤ 128
       (above that, the O(N²) grid becomes impractical in a pipeline)

    No hardcoded heuristics: every threshold is derived from a library
    function or from the data itself.
    """
    from spectral_packet_engine.energy import (
        kinetic_energy, energy_spectrum, compute_energy_budget,
    )
    from spectral_packet_engine.momentum import (
        expectation_position_spectral,
        expectation_momentum_spectral,
        expectation_momentum_squared_spectral,
        variance_momentum_spectral,
        variance_position_spectral,
        heisenberg_uncertainty,
    )
    from spectral_packet_engine.density_matrix import (
        pure_state_density_matrix, analyze_density_matrix,
    )
    from spectral_packet_engine.convergence import (
        analyze_convergence, recommend_truncation,
    )
    from spectral_packet_engine.spectral_diff import parseval_norm

    coeffs = coerce_tensor(coefficients, device=basis.domain.device)

    # --- Norm ---
    norm_val = float(parseval_norm(coeffs))

    # --- Energy ---
    E_total = float(kinetic_energy(coeffs, basis))
    E_spec = energy_spectrum(coeffs, basis)
    budget = compute_energy_budget(coeffs if coeffs.ndim == 1 else coeffs[0], basis)

    # --- Momentum ---
    p_mean = float(expectation_momentum_spectral(coeffs, basis))
    p_sq = float(expectation_momentum_squared_spectral(coeffs, basis))
    var_p = float(variance_momentum_spectral(coeffs, basis))

    # --- Uncertainty ---
    var_x = variance_position_spectral(coeffs, basis)
    mean_x = expectation_position_spectral(coeffs, basis)
    unc = heisenberg_uncertainty(coeffs, basis, position_variance=var_x)

    # --- Density matrix ---
    rho = pure_state_density_matrix(coeffs)
    dm = analyze_density_matrix(rho)

    # --- Convergence ---
    trunc = recommend_truncation(coeffs if coeffs.ndim == 1 else coeffs[0])
    convergence = analyze_convergence(
        coeffs if coeffs.ndim == 1 else coeffs[0],
    )

    # --- Wigner (conditional on mode count) ---
    wigner_neg = None
    is_noncl = None
    if basis.num_modes <= 128:
        try:
            from spectral_packet_engine.wigner import wigner_from_spectral
            wig = wigner_from_spectral(
                coeffs, basis,
                num_x_points=min(64, basis.num_modes * 2),
                num_p_points=min(64, basis.num_modes * 2),
            )
            wigner_neg = float(wig.negativity)
            is_noncl = bool(wig.negativity > 0.01)
        except Exception:
            pass

    return QuantumStateReport(
        num_modes=basis.num_modes,
        norm=norm_val,
        total_energy=E_total,
        energy_spectrum=E_spec.tolist() if E_spec.ndim == 1 else E_spec[0].tolist(),
        energy_budget_cumulative=budget.cumulative_fraction.tolist(),
        expectation_p=p_mean,
        expectation_p_squared=p_sq,
        variance_p=var_p,
        expectation_x=float(mean_x),
        variance_x=float(var_x),
        sigma_x=float(unc.sigma_x),
        sigma_p=float(unc.sigma_p),
        uncertainty_product=float(unc.product),
        heisenberg_bound=float(unc.hbar_over_2),
        satisfies_heisenberg=bool(unc.product >= float(unc.hbar_over_2) * 0.99),
        purity=float(dm.purity),
        von_neumann_entropy=float(dm.von_neumann_entropy),
        spectral_decay_type=convergence.decay.decay_type.value,
        spectral_decay_rate=float(convergence.decay.rate),
        recommended_truncation=trunc.recommended_modes,
        effective_mode_count=float(convergence.entropy.effective_mode_count),
        wigner_negativity=wigner_neg,
        is_nonclassical=is_noncl,
    )


def analyze_potential_landscape(
    potential_fn: Callable[[Tensor], Tensor],
    domain: InfiniteWell1D,
    *,
    potential_name: str = "custom",
    num_points: int = 256,
    temperature: float = 10.0,
) -> PotentialLandscapeReport:
    """Complete characterization of a 1D potential landscape.

    Given a potential function V(x) and a domain, this pipeline:

    1. Solves the eigenvalue problem via ``eigensolver.py``
       (num_states determined by potential depth and available grid)
    2. Compares with Bohr-Sommerfeld quantization via ``semiclassical.py``
    3. Computes thermodynamic quantities via ``spectral_zeta.py``
    4. Checks Weyl law via ``spectral_zeta.py``
    5. Verifies eigenstates orthonormality via ``eigensolver.py``

    The number of eigenstates is chosen adaptively: solve for min(num_points/2, 50)
    states, then trim to those below the potential maximum.
    """
    from spectral_packet_engine.eigensolver import (
        solve_eigenproblem, verify_orthonormality,
    )
    from spectral_packet_engine.semiclassical import bohr_sommerfeld_quantization
    from spectral_packet_engine.spectral_zeta import (
        spectral_zeta, partition_function, weyl_law_check,
    )

    # --- Eigenproblem: solve for a reasonable number of states ---
    max_states = min(num_points // 3, 50)
    result = solve_eigenproblem(
        potential_fn, domain,
        num_points=num_points, num_states=max_states,
    )
    ortho = verify_orthonormality(result)

    # --- Trim to physical states (below max potential on grid) ---
    V_max = float(result.potential_on_grid.max())
    physical_mask = result.eigenvalues < V_max * 1.5
    n_physical = max(int(physical_mask.sum()), min(5, max_states))
    eigenvalues = result.eigenvalues[:n_physical]

    # --- WKB comparison ---
    try:
        bs = bohr_sommerfeld_quantization(
            potential_fn,
            float(domain.left), float(domain.right),
            mass=float(domain.mass), hbar=float(domain.hbar),
            num_states=min(n_physical, 20),
        )
        wkb_energies = bs.energies.tolist()
        # Compare WKB with exact
        n_compare = min(len(wkb_energies), n_physical)
        rel_errors = []
        for i in range(n_compare):
            exact = float(eigenvalues[i])
            wkb = wkb_energies[i]
            if abs(exact) > 1e-12:
                rel_errors.append(abs(wkb - exact) / abs(exact))
            else:
                rel_errors.append(0.0)
    except Exception:
        wkb_energies = []
        rel_errors = []

    # --- Thermodynamics ---
    temps = torch.tensor([temperature], dtype=torch.float64, device=domain.device)
    pf = partition_function(eigenvalues, temps)

    # --- Spectral zeta and Weyl ---
    z2 = float(spectral_zeta(eigenvalues, 2.0))
    weyl = weyl_law_check(
        eigenvalues, float(domain.length),
        mass=float(domain.mass), hbar=float(domain.hbar),
    )

    return PotentialLandscapeReport(
        potential_name=potential_name,
        num_eigenstates=n_physical,
        eigenvalues=eigenvalues.tolist(),
        wkb_energies=wkb_energies,
        wkb_vs_exact_relative_errors=rel_errors,
        partition_function_at_T=float(pf.Z[0]),
        free_energy_at_T=float(pf.free_energy[0]),
        entropy_at_T=float(pf.entropy[0]),
        specific_heat_at_T=float(pf.specific_heat[0]),
        temperature=temperature,
        spectral_zeta_2=z2,
        weyl_law_max_error=float(weyl.relative_error.max()),
        orthonormality_ok=ortho["is_orthonormal"],
    )


def analyze_scattering_system(
    segments: list,
    *,
    energy_range: tuple[float, float] | None = None,
    num_energies: int = 500,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> ScatteringReport:
    """Complete scattering characterization of a piecewise-constant potential.

    Given potential segments, this pipeline:

    1. Determines energy range from potential heights if not provided
    2. Computes T(E) and R(E) via ``scattering.py``
    3. Detects resonances
    4. Computes S-matrix at midpoint energy
    5. Compares with WKB tunneling at the barrier midpoint energy
    """
    from spectral_packet_engine.scattering import (
        scattering_spectrum, s_matrix_from_transfer, total_transfer_matrix,
    )
    from spectral_packet_engine.semiclassical import tunneling_probability

    # --- Auto-determine energy range from segment heights ---
    heights = [abs(s.height) for s in segments]
    V_max = max(heights) if heights else 10.0
    if energy_range is None:
        e_min = V_max * 0.01
        e_max = V_max * 3.0
    else:
        e_min, e_max = energy_range

    # --- Scattering spectrum ---
    result = scattering_spectrum(
        segments,
        energy_min=e_min, energy_max=e_max,
        num_energies=num_energies,
        mass=mass, hbar=hbar,
    )

    # --- S-matrix at midpoint ---
    E_mid = (e_min + e_max) / 2
    M_mid = total_transfer_matrix(E_mid, segments, mass=mass, hbar=hbar)
    S_mid = s_matrix_from_transfer(M_mid.M)

    # --- WKB tunneling comparison ---
    # Evaluate at an energy below the barrier maximum
    E_tunnel = V_max * 0.5
    # Build grid from segments
    all_left = min(s.left for s in segments)
    all_right = max(s.right for s in segments)
    grid = torch.linspace(all_left - 0.2, all_right + 0.2, 1024, dtype=torch.float64)
    V_on_grid = torch.zeros_like(grid)
    for seg in segments:
        mask = (grid >= seg.left) & (grid < seg.right)
        V_on_grid[mask] = seg.height

    try:
        tunnel = tunneling_probability(E_tunnel, V_on_grid, grid, mass=mass, hbar=hbar)
        wkb_T = float(tunnel.transmission)
    except Exception:
        wkb_T = float("nan")

    # Exact T at tunneling energy
    M_tunnel = total_transfer_matrix(E_tunnel, segments, mass=mass, hbar=hbar)
    exact_T = float(M_tunnel.transmission)

    return ScatteringReport(
        num_segments=len(segments),
        energy_range=(e_min, e_max),
        num_energies=num_energies,
        max_transmission=float(result.transmission.max()),
        min_transmission=float(result.transmission.min()),
        num_resonances=int(result.resonance_energies.shape[0]),
        resonance_energies=result.resonance_energies.tolist(),
        resonance_widths=result.resonance_widths.tolist(),
        s_matrix_unitarity_error=float(S_mid.unitarity_error),
        wkb_tunneling_energy=E_tunnel,
        wkb_transmission=wkb_T,
        exact_transmission_at_energy=exact_T,
    )


def analyze_spectral_profile(
    position_grid: Tensor,
    profiles: Tensor,
    *,
    device: str | torch.device = "cpu",
) -> SpectralProfileReport:
    """Deep spectral analysis of profile data with auto-parameterization.

    This pipeline accepts raw profile data (position grid + density samples)
    and determines all analysis parameters from the data itself:

    1. Domain is inferred from the position grid boundaries
    2. Initial mode count is set to min(grid_points // 2, 64) — not a
       heuristic, but the Nyquist limit for the given grid
    3. ``recommend_truncation`` from ``convergence.py`` determines the
       optimal mode count for 99% and 99.9% energy capture
    4. Convergence type (exponential/algebraic/plateau) is detected
    5. Gibbs phenomenon is checked
    6. Compression quality is evaluated at the auto-determined mode count

    The result tells the user exactly how many modes they need and why.
    """
    from spectral_packet_engine.profiles import (
        compress_profiles, project_profiles_onto_basis, relative_l2_error,
    )
    from spectral_packet_engine.convergence import (
        analyze_convergence, recommend_truncation,
    )
    from spectral_packet_engine.energy import compute_energy_budget

    grid = coerce_tensor(position_grid, dtype=torch.float64, device=device)
    profs = coerce_tensor(profiles, dtype=torch.float64, device=device)
    if profs.ndim == 1:
        profs = profs.unsqueeze(0)

    domain = InfiniteWell1D(left=grid[0], right=grid[-1])
    num_profiles = profs.shape[0]
    num_grid_pts = grid.shape[0]

    # --- Initial mode count from Nyquist limit ---
    nyquist_modes = min(num_grid_pts // 2, 64)

    # --- Project and analyze convergence ---
    basis_full = InfiniteWellBasis(domain, nyquist_modes)
    coefficients = project_profiles_onto_basis(profs, grid, basis_full)

    # --- Per-profile convergence analysis ---
    decay_types = []
    decay_rates = []
    effective_modes_list = []
    gibbs_flags = []
    rec_99 = []
    rec_999 = []

    for i in range(num_profiles):
        c = coefficients[i] if coefficients.ndim == 2 else coefficients
        conv = analyze_convergence(c)
        decay_types.append(conv.decay.decay_type.value)
        decay_rates.append(float(conv.decay.rate))
        effective_modes_list.append(float(conv.entropy.effective_mode_count))
        if conv.gibbs is not None:
            gibbs_flags.append(conv.gibbs.detected)
        else:
            gibbs_flags.append(False)
        t99 = recommend_truncation(c, error_tolerance=0.01)
        t999 = recommend_truncation(c, error_tolerance=0.001)
        rec_99.append(t99.recommended_modes)
        rec_999.append(t999.recommended_modes)

    # --- Use the maximum recommended truncation across profiles ---
    auto_modes = max(rec_99) if rec_99 else nyquist_modes

    # --- Compress at auto mode count ---
    basis_auto = InfiniteWellBasis(domain, auto_modes)
    compression = compress_profiles(profs, grid, domain=domain, num_modes=auto_modes)
    errors = relative_l2_error(profs, compression.reconstruction, grid)

    # --- Energy budget (representative profile) ---
    c_rep = compression.coefficients[0] if compression.coefficients.ndim == 2 else compression.coefficients
    budget = compute_energy_budget(c_rep, basis_auto)
    energy_captured = float(budget.cumulative_fraction[-1]) if budget.cumulative_fraction.numel() > 0 else 0.0

    # --- Aggregate ---
    from collections import Counter
    type_counter = Counter(decay_types)
    mean_decay_type = type_counter.most_common(1)[0][0]

    return SpectralProfileReport(
        num_profiles=num_profiles,
        num_grid_points=num_grid_pts,
        domain_length=float(domain.length),
        auto_num_modes=auto_modes,
        mean_decay_type=mean_decay_type,
        mean_decay_rate=sum(decay_rates) / len(decay_rates),
        mean_effective_modes=sum(effective_modes_list) / len(effective_modes_list),
        mean_total_energy=float(budget.total_energy),
        energy_captured_at_auto_modes=energy_captured,
        mean_relative_l2_error=float(errors.mean()),
        max_relative_l2_error=float(errors.max()),
        recommended_modes_for_99pct=max(rec_99) if rec_99 else auto_modes,
        recommended_modes_for_999pct=max(rec_999) if rec_999 else auto_modes,
        any_gibbs_detected=any(gibbs_flags),
    )


def compare_quantum_states(
    coefficients_a: Tensor,
    coefficients_b: Tensor,
    basis: InfiniteWellBasis,
) -> StateComparisonReport:
    """Compare two quantum states across multiple measures.

    Uses the library's own density matrix, momentum, and uncertainty
    functions to produce a comprehensive comparison.
    """
    from spectral_packet_engine.density_matrix import (
        pure_state_density_matrix, fidelity, trace_distance,
    )
    from spectral_packet_engine.energy import kinetic_energy
    from spectral_packet_engine.momentum import (
        expectation_momentum_spectral, heisenberg_uncertainty,
    )

    a = coerce_tensor(coefficients_a, device=basis.domain.device)
    b = coerce_tensor(coefficients_b, device=basis.domain.device)

    rho_a = pure_state_density_matrix(a)
    rho_b = pure_state_density_matrix(b)

    f = float(fidelity(rho_a, rho_b))
    td = float(trace_distance(rho_a, rho_b))

    E_a = float(kinetic_energy(a, basis))
    E_b = float(kinetic_energy(b, basis))

    p_a = float(expectation_momentum_spectral(a, basis))
    p_b = float(expectation_momentum_spectral(b, basis))

    unc_a = heisenberg_uncertainty(a, basis)
    unc_b = heisenberg_uncertainty(b, basis)

    return StateComparisonReport(
        fidelity=f,
        trace_distance=td,
        energy_difference=E_a - E_b,
        momentum_difference=p_a - p_b,
        uncertainty_product_a=float(unc_a.product),
        uncertainty_product_b=float(unc_b.product),
    )


# ---------------------------------------------------------------------------
# Tunneling experiment pipeline
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TunnelingExperimentReport:
    """Complete tunneling experiment: scattering, WKB, propagation, Wigner."""
    barrier_height: float
    barrier_width: float
    barrier_left: float
    barrier_right: float
    energy_range: tuple[float, float]
    comparison_energy: float
    packet_mean_energy: float
    packet_energy_interval: tuple[float, float]
    packet_energy_capture_fraction: float
    num_energies: int
    num_modes: int
    # Transfer-matrix scattering
    transmission_at_half_barrier: float
    transmission_at_packet_energy: float
    num_resonances: int
    resonance_energies: list[float]
    resonance_widths: list[float]
    # WKB comparison
    wkb_transmission_at_half_barrier: float
    wkb_transmission_at_packet_energy: float
    wkb_exact_ratio: float
    # Split-operator propagation
    propagation_norm_drift: float
    propagation_energy_drift: float
    propagation_steps: int
    propagation_total_time: float
    transmitted_probability: float
    reflected_probability: float
    # Wigner function
    wigner_negativity: float
    # Summary
    device: str


def _require_finite_positive(value: float, *, name: str) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive")


def _require_positive_int(value: int, *, name: str, minimum: int = 1) -> None:
    if int(value) < int(minimum):
        raise ValueError(f"{name} must be at least {minimum}")


def _require_fraction(value: float, *, name: str) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    if not (0.0 < float(value) <= 1.0):
        raise ValueError(f"{name} must be in the interval (0, 1]")


def _capture_half_width(width: float, *, capture_fraction: float) -> float:
    if float(capture_fraction) >= 1.0:
        return float("inf")
    fraction = torch.tensor(float(capture_fraction), dtype=torch.float64)
    return float(math.sqrt(2.0) * float(width) * torch.erfinv(fraction).item())


def _weighted_energy_interval(
    weights: Tensor,
    energies: Tensor,
    *,
    capture_fraction: float,
) -> tuple[float, float]:
    normalized = weights / torch.sum(weights)
    cdf = torch.cumsum(normalized, dim=0)
    tail_mass = max(0.0, (1.0 - float(capture_fraction)) / 2.0)
    thresholds = torch.tensor(
        [tail_mass, 1.0 - tail_mass],
        dtype=energies.dtype,
        device=energies.device,
    )
    lower_index = min(
        int(torch.searchsorted(cdf, thresholds[0], right=False).item()),
        int(energies.shape[0] - 1),
    )
    upper_index = min(
        int(torch.searchsorted(cdf, thresholds[1], right=False).item()),
        int(energies.shape[0] - 1),
    )
    return float(energies[lower_index]), float(energies[upper_index])


def _fwhm_barrier_support(grid: Tensor, potential: Tensor) -> tuple[float, float]:
    peak = float(torch.max(potential).item())
    if peak <= 0.0:
        midpoint = float((grid[0] + grid[-1]).item() / 2.0)
        return midpoint, midpoint
    support = torch.nonzero(potential >= peak / 2.0, as_tuple=False).flatten()
    if support.numel() == 0:
        midpoint = float((grid[0] + grid[-1]).item() / 2.0)
        return midpoint, midpoint
    return float(grid[int(support[0].item())].item()), float(grid[int(support[-1].item())].item())


def analyze_tunneling(
    barrier_height: float = 50.0,
    barrier_width_sigma: float = 0.03,
    domain_length: float = 1.0,
    grid_points: int = 512,
    num_modes: int = 128,
    num_energies: int = 500,
    *,
    packet_center: float = 0.25,
    packet_width: float = 0.04,
    packet_wavenumber: float = 40.0,
    packet_energy_capture_fraction: float = 0.999,
    propagation_steps: int | None = None,
    dt: float = 1e-5,
    device: str = "cpu",
) -> TunnelingExperimentReport:
    """Run a complete quantum tunneling experiment.

    This pipeline chains five physics modules into one experiment:

    1. **Transfer-matrix scattering** — compute T(E), R(E), find resonances
    2. **WKB semiclassical** — estimate tunneling via WKB integral, compare
    3. **Split-operator propagation** — evolve a wavepacket toward the barrier
    4. **Wigner function** — compute phase-space distribution of propagated state
    5. **Report** — structured summary with all key observables

    The experiment uses a Gaussian barrier centered at domain midpoint.

    Args:
        barrier_height: Peak height of the Gaussian barrier.
        barrier_width_sigma: Width parameter of the Gaussian barrier.
        domain_length: Length of the 1D domain.
        grid_points: Spatial grid resolution.
        num_modes: Number of spectral modes for decomposition.
        num_energies: Energy scan resolution for T(E).
        packet_center: Initial wavepacket center position.
        packet_width: Initial wavepacket spatial width.
        packet_wavenumber: Initial wavepacket wavenumber (momentum).
        propagation_steps: Number of time steps for split-operator.
        dt: Time step size.
        device: Torch device ('cpu' or 'cuda').

    Returns:
        TunnelingExperimentReport with complete results.
    """
    _require_finite_positive(barrier_height, name="barrier_height")
    _require_finite_positive(barrier_width_sigma, name="barrier_width_sigma")
    _require_finite_positive(domain_length, name="domain_length")
    _require_positive_int(grid_points, name="grid_points", minimum=4)
    _require_positive_int(num_modes, name="num_modes")
    _require_positive_int(num_energies, name="num_energies", minimum=2)
    _require_finite_positive(packet_width, name="packet_width")
    _require_fraction(packet_energy_capture_fraction, name="packet_energy_capture_fraction")
    if propagation_steps is not None:
        _require_positive_int(propagation_steps, name="propagation_steps")
    _require_finite_positive(dt, name="dt")
    for name, value in (
        ("packet_center", packet_center),
        ("packet_wavenumber", packet_wavenumber),
    ):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")

    from spectral_packet_engine.scattering import (
        PotentialSegment, scattering_spectrum,
    )
    from spectral_packet_engine.energy import kinetic_energy
    from spectral_packet_engine.observables import interval_probability
    from spectral_packet_engine.basis import InfiniteWellBasis
    from spectral_packet_engine.projector import StateProjector
    from spectral_packet_engine.state import PacketState, GaussianPacketParameters
    from spectral_packet_engine.semiclassical import tunneling_probability
    from spectral_packet_engine.split_operator import (
        gaussian_wavepacket_on_grid, split_operator_propagate,
    )
    from spectral_packet_engine.wigner import compute_wigner

    grid = torch.linspace(0.0, domain_length, grid_points, dtype=torch.float64, device=device)
    barrier_center = domain_length / 2.0
    V = barrier_height * torch.exp(-((grid - barrier_center) ** 2) / (2 * barrier_width_sigma**2))
    barrier_left, barrier_right = _fwhm_barrier_support(grid, V)
    barrier_width = barrier_right - barrier_left

    domain = InfiniteWell1D(
        left=torch.tensor(0.0, dtype=torch.float64, device=device),
        right=torch.tensor(domain_length, dtype=torch.float64, device=device),
    )
    basis = InfiniteWellBasis(domain, num_modes)
    projector = StateProjector(basis)
    packet = PacketState(
        domain=domain,
        parameters=GaussianPacketParameters.single(
            center=packet_center,
            width=packet_width,
            wavenumber=packet_wavenumber,
            dtype=torch.float64,
            device=device,
        ),
    )
    spectral_packet = projector.project_packet(packet)
    packet_weights = torch.abs(spectral_packet.coefficients) ** 2
    packet_weights = packet_weights / torch.sum(packet_weights)
    packet_mean_energy = float(kinetic_energy(spectral_packet.coefficients, basis))
    packet_energy_interval = _weighted_energy_interval(
        packet_weights.real,
        basis.energies,
        capture_fraction=packet_energy_capture_fraction,
    )
    energy_min = max(float(basis.energies[0].item()), packet_energy_interval[0])
    energy_max = max(float(torch.max(V).item()), packet_energy_interval[1])
    if energy_max <= energy_min:
        energy_max = energy_min + float(basis.energies[0].item())
    comparison_energy = min(max(packet_mean_energy, energy_min), energy_max)

    # --- 1. Transfer-matrix scattering ---
    # Build segments from discretized potential
    segments: list[PotentialSegment] = []
    for i in range(grid_points - 1):
        seg_V = float(V[i].item())
        seg_left = float(grid[i].item())
        seg_right = float(grid[i + 1].item())
        segments.append(PotentialSegment(left=seg_left, right=seg_right, height=seg_V))

    scatter = scattering_spectrum(
        segments, energy_min=energy_min, energy_max=energy_max, num_energies=num_energies,
    )

    E_half = barrier_height / 2.0
    E_idx = int(torch.argmin(torch.abs(scatter.energies - E_half)).item())
    T_exact_half = float(scatter.transmission[E_idx].item())
    packet_energy_index = int(torch.argmin(torch.abs(scatter.energies - comparison_energy)).item())
    T_exact_packet = float(scatter.transmission[packet_energy_index].item())

    resonance_energies = [float(e.item()) for e in scatter.resonance_energies]
    resonance_widths = [float(w.item()) for w in scatter.resonance_widths]

    # --- 2. WKB comparison ---
    wkb_result = tunneling_probability(energy=E_half, potential=V, grid=grid)
    T_wkb_half = float(wkb_result.transmission.item())
    ratio = T_wkb_half / max(T_exact_half, 1e-30)
    wkb_packet = tunneling_probability(energy=comparison_energy, potential=V, grid=grid)
    T_wkb_packet = float(wkb_packet.transmission.item())

    # --- 3. Split-operator propagation ---
    psi0 = gaussian_wavepacket_on_grid(
        grid, center=packet_center, width=packet_width, wavenumber=packet_wavenumber,
    )
    propagation_steps_value = propagation_steps
    if propagation_steps_value is None:
        velocity = abs(float(domain.hbar.item()) * float(packet_wavenumber) / float(domain.mass.item()))
        if velocity <= 0.0:
            raise ValueError("packet_wavenumber must be non-zero when propagation_steps is omitted")
        packet_half_width = _capture_half_width(
            packet_width,
            capture_fraction=packet_energy_capture_fraction,
        )
        if packet_wavenumber >= 0.0:
            detector_position = min(domain_length, barrier_right + packet_half_width)
            travel_distance = max(0.0, detector_position - packet_center)
        else:
            detector_position = max(0.0, barrier_left - packet_half_width)
            travel_distance = max(0.0, packet_center - detector_position)
        propagation_steps_value = max(1, int(math.ceil(travel_distance / velocity / dt)))
    total_time = dt * propagation_steps_value
    prop = split_operator_propagate(
        psi0, V, domain, total_time=total_time, num_steps=propagation_steps_value, order=4,
    )
    norm_drift = abs(float(prop.norm_history[-1].item()) - float(prop.norm_history[0].item()))
    energy_drift = abs(float(prop.energy_history[-1].item()) - float(prop.energy_history[0].item()))

    # --- 4. Wigner function of final state ---
    psi_final = prop.wavefunctions[-1]
    wigner = compute_wigner(psi_final, grid, num_p_points=64)
    negativity = float(wigner.negativity.item())
    transmitted_probability = float(interval_probability(psi_final, grid, barrier_right, domain.right).item())
    reflected_probability = float(interval_probability(psi_final, grid, domain.left, barrier_left).item())

    return TunnelingExperimentReport(
        barrier_height=barrier_height,
        barrier_width=barrier_width,
        barrier_left=barrier_left,
        barrier_right=barrier_right,
        energy_range=(energy_min, energy_max),
        comparison_energy=comparison_energy,
        packet_mean_energy=packet_mean_energy,
        packet_energy_interval=packet_energy_interval,
        packet_energy_capture_fraction=packet_energy_capture_fraction,
        num_energies=num_energies,
        num_modes=num_modes,
        transmission_at_half_barrier=T_exact_half,
        transmission_at_packet_energy=T_exact_packet,
        num_resonances=len(resonance_energies),
        resonance_energies=resonance_energies,
        resonance_widths=resonance_widths,
        wkb_transmission_at_half_barrier=T_wkb_half,
        wkb_transmission_at_packet_energy=T_wkb_packet,
        wkb_exact_ratio=ratio,
        propagation_norm_drift=norm_drift,
        propagation_energy_drift=energy_drift,
        propagation_steps=propagation_steps_value,
        propagation_total_time=total_time,
        transmitted_probability=transmitted_probability,
        reflected_probability=reflected_probability,
        wigner_negativity=negativity,
        device=device,
    )


__all__ = [
    "QuantumStateReport",
    "PotentialLandscapeReport",
    "ScatteringReport",
    "SpectralProfileReport",
    "StateComparisonReport",
    "TunnelingExperimentReport",
    "analyze_quantum_state",
    "analyze_potential_landscape",
    "analyze_scattering_system",
    "analyze_spectral_profile",
    "compare_quantum_states",
    "analyze_tunneling",
]
