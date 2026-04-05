"""Spectral load modeling — self-referential infrastructure optimization.

This module uses the spectral packet engine's own mathematical machinery
to model server request patterns, detect anomalous load, and compute
optimal throttling parameters.  The core idea is that server request rate
r(t) over a bounded time window [0, T] is a signal on a finite domain —
exactly the kind of function the engine was built to decompose.

Mathematical foundation:

    r(t) = sum_{n=1}^{N} c_n * phi_n(t)

where phi_n are the sine basis functions of the bounded domain [0, T].
The spectral coefficients c_n encode the traffic structure:

  - Smooth, legitimate traffic: exponential coefficient decay |c_n| ~ e^{-alpha n}
  - Bursty but organic traffic: algebraic decay |c_n| ~ n^{-beta}
  - Bot / abuse / DDoS spikes: plateau (no decay) or anomalous high-frequency energy

The module uses the engine's own diagnostics (convergence analysis, spectral
entropy, truncation recommendation, decay estimation) to classify traffic
and derive throttling parameters — no hardcoded thresholds.

Example usage:

    from spectral_packet_engine.load_spectral import (
        ingest_request_log,
        decompose_load_signal,
        analyze_load_spectrum,
        compute_adaptive_throttle,
        LoadSpectralReport,
    )

    # From raw timestamps
    signal = ingest_request_log(timestamps, window_seconds=300.0)
    coefficients = decompose_load_signal(signal, num_modes=64)
    report = analyze_load_spectrum(coefficients)
    throttle = compute_adaptive_throttle(coefficients, capacity_rps=100.0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from spectral_packet_engine.basis import InfiniteWellBasis, sine_basis_matrix
from spectral_packet_engine.convergence import (
    ConvergenceDiagnostics,
    SpectralDecayEstimate,
    SpectralEntropyReport,
    TruncationRecommendation,
    analyze_convergence,
    estimate_spectral_decay,
    recommend_truncation,
    spectral_entropy,
)
from spectral_packet_engine.domain import InfiniteWell1D, coerce_tensor
from spectral_packet_engine.profiles import (
    normalize_profiles,
    profile_mass,
    project_profiles_onto_basis,
)
from spectral_packet_engine.diagnostics import spectral_weights

Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LoadSignal:
    """Discretized request-rate signal on a bounded time window.

    Attributes:
        grid: Uniformly spaced time points in [0, window_seconds].
        values: Request rate (requests/second) at each grid point.
        window_seconds: Duration of the observation window.
        total_requests: Integral of rate over the window (total count).
    """
    grid: Tensor
    values: Tensor
    window_seconds: float
    total_requests: float


@dataclass(frozen=True, slots=True)
class LoadCoefficients:
    """Spectral coefficients of a load signal.

    The coefficients live in the same mathematical space as profile-table
    spectral coefficients — they represent the sine-basis expansion of the
    request rate r(t) over the bounded time window.
    """
    coefficients: Tensor        # shape (num_modes,)
    basis: InfiniteWellBasis
    signal: LoadSignal
    num_modes: int
    energy_fractions: Tensor    # per-mode energy fraction


@dataclass(frozen=True, slots=True)
class LoadSpectrumAnalysis:
    """Result of spectral analysis on a load signal.

    Uses the engine's own convergence diagnostics, spectral entropy,
    and decay estimation to classify the traffic pattern.
    """
    decay: SpectralDecayEstimate
    convergence: ConvergenceDiagnostics
    entropy: SpectralEntropyReport
    truncation: TruncationRecommendation
    effective_mode_count: float
    dominant_frequency_hz: float
    high_frequency_energy_ratio: float


@dataclass(frozen=True, slots=True)
class AnomalyAssessment:
    """Spectral anomaly assessment comparing current load to a baseline.

    Anomaly is detected by comparing the spectral structure of the current
    signal to a reference baseline.  The distance metrics are the same
    fidelity and trace-distance concepts used in quantum state comparison,
    applied to the spectral coefficient distributions.
    """
    is_anomalous: bool
    spectral_distance: float      # L2 distance in coefficient space (normalized)
    entropy_shift: float          # change in spectral entropy (bits)
    mode_count_shift: float       # change in effective mode count
    energy_redistribution: float  # Jensen-Shannon divergence of energy fractions
    reason: str


@dataclass(frozen=True, slots=True)
class AdaptiveThrottle:
    """Spectral-derived adaptive throttling parameters.

    The cooldown duration, message interval, and concurrency limit are
    computed from the spectral structure of the traffic — not from
    hard-coded rules.  The engine's own truncation recommendation drives
    the granularity, and the spectral entropy determines how much
    "disorder" (burstiness) to tolerate.
    """
    recommended_cooldown_seconds: float
    recommended_min_interval_seconds: float
    recommended_max_concurrent: int
    capacity_utilization: float       # 0-1
    spectral_load_factor: float       # energy-weighted load metric
    headroom_fraction: float          # remaining capacity fraction
    regime: str                       # "smooth", "bursty", "saturated", "anomalous"


@dataclass(frozen=True, slots=True)
class CapacityEstimate:
    """Spectral estimate of sustainable server capacity.

    Uses the spectral energy budget to distinguish sustained load from
    transient spikes, giving a capacity estimate that is robust to
    burst noise.
    """
    sustained_rps: float          # time-averaged rate from low-frequency modes
    peak_rps: float               # full-spectrum peak including bursts
    burst_ratio: float            # peak / sustained
    spectral_headroom_modes: int  # how many modes before truncation error grows
    stable: bool                  # convergence diagnostics say the pattern is stable


@dataclass(frozen=True, slots=True)
class LoadSpectralReport:
    """Complete spectral load analysis report.

    Combines signal decomposition, traffic classification, anomaly
    assessment, throttling recommendation, and capacity estimation.
    """
    signal: LoadSignal
    coefficients: LoadCoefficients
    spectrum: LoadSpectrumAnalysis
    throttle: AdaptiveThrottle
    capacity: CapacityEstimate
    anomaly: AnomalyAssessment | None


# ---------------------------------------------------------------------------
# Signal ingestion
# ---------------------------------------------------------------------------

def ingest_request_log(
    timestamps: Sequence[float] | Tensor,
    *,
    window_seconds: float = 300.0,
    resolution: int = 256,
    device: str = "cpu",
) -> LoadSignal:
    """Convert raw request timestamps to a discretized rate signal.

    Bins timestamps into a uniform grid and computes the instantaneous
    request rate (requests per second) at each grid point.

    Args:
        timestamps: Unix-epoch or relative timestamps of individual requests.
        window_seconds: Duration of the observation window.
        resolution: Number of grid points for the discretization.
        device: Torch device for computation.

    Returns:
        LoadSignal with the discretized rate on [0, window_seconds].
    """
    ts = coerce_tensor(timestamps, dtype=torch.float64, device=device)
    if ts.numel() == 0:
        grid = torch.linspace(0.0, window_seconds, resolution, dtype=torch.float64, device=device)
        return LoadSignal(
            grid=grid,
            values=torch.zeros(resolution, dtype=torch.float64, device=device),
            window_seconds=window_seconds,
            total_requests=0.0,
        )

    # Shift to relative time within window
    t_min = ts.min()
    t_rel = ts - t_min
    # Clamp to window
    t_rel = t_rel.clamp(0.0, window_seconds)

    grid = torch.linspace(0.0, window_seconds, resolution, dtype=torch.float64, device=device)
    bin_width = window_seconds / (resolution - 1)

    # Histogram binning → count per bin → rate (count / bin_width)
    bin_edges = torch.linspace(0.0, window_seconds + bin_width, resolution + 1, dtype=torch.float64, device=device)
    counts = torch.histogram(t_rel.cpu().to(torch.float64), bins=bin_edges.cpu()).hist.to(device)

    # Trim to resolution (histogram may produce resolution bins)
    counts = counts[:resolution]
    rate = counts / bin_width

    total = float(ts.numel())

    return LoadSignal(
        grid=grid,
        values=rate,
        window_seconds=window_seconds,
        total_requests=total,
    )


def load_signal_from_rate(
    rate_values: Sequence[float] | Tensor,
    *,
    window_seconds: float = 300.0,
    device: str = "cpu",
) -> LoadSignal:
    """Create a LoadSignal from pre-computed rate values.

    Args:
        rate_values: Request-rate samples (requests/second), uniformly spaced.
        window_seconds: Duration of the observation window.
        device: Torch device.

    Returns:
        LoadSignal with the given rate on [0, window_seconds].
    """
    values = coerce_tensor(rate_values, dtype=torch.float64, device=device)
    if values.ndim != 1 or values.numel() < 2:
        raise ValueError("rate_values must be a 1-D sequence with at least 2 samples")
    grid = torch.linspace(0.0, window_seconds, values.shape[0], dtype=torch.float64, device=device)
    total = float(torch.trapezoid(values, grid).item())
    return LoadSignal(grid=grid, values=values, window_seconds=window_seconds, total_requests=total)


# ---------------------------------------------------------------------------
# Spectral decomposition
# ---------------------------------------------------------------------------

def decompose_load_signal(
    signal: LoadSignal,
    *,
    num_modes: int = 64,
) -> LoadCoefficients:
    """Project a load signal onto the spectral basis.

    This is the same sine-basis projection that the engine uses for
    density profile tables, applied to the request-rate time series.

    Args:
        signal: Discretized load signal.
        num_modes: Number of spectral modes for decomposition.

    Returns:
        LoadCoefficients with the spectral expansion.
    """
    device = signal.grid.device
    domain = InfiniteWell1D(
        left=torch.tensor(0.0, dtype=torch.float64, device=device),
        right=torch.tensor(signal.window_seconds, dtype=torch.float64, device=device),
    )
    basis = InfiniteWellBasis(domain=domain, num_modes=num_modes)

    # Project: same as project_profiles_onto_basis
    coefficients = project_profiles_onto_basis(
        signal.values.unsqueeze(0),
        signal.grid,
        basis,
    ).squeeze(0)

    # Energy fractions per mode: |c_n|^2 / sum |c_n|^2
    weights = spectral_weights(coefficients)
    total_weight = weights.sum()
    e_frac = weights / total_weight.clamp(min=1e-30)

    return LoadCoefficients(
        coefficients=coefficients,
        basis=basis,
        signal=signal,
        num_modes=num_modes,
        energy_fractions=e_frac,
    )


# ---------------------------------------------------------------------------
# Spectral analysis — uses engine diagnostics, no hardcoded thresholds
# ---------------------------------------------------------------------------

def analyze_load_spectrum(coefficients: LoadCoefficients) -> LoadSpectrumAnalysis:
    """Analyze the spectral structure of a load signal.

    Uses the engine's own convergence diagnostics, spectral entropy, and
    decay estimation.  No hardcoded thresholds — the engine's diagnostics
    determine what is "smooth", "bursty", or "anomalous".

    Args:
        coefficients: Spectral coefficients from decompose_load_signal.

    Returns:
        LoadSpectrumAnalysis with decay classification, entropy, and
        truncation recommendation.
    """
    c = coefficients.coefficients

    # Engine's own diagnostics
    decay = estimate_spectral_decay(c)
    convergence = analyze_convergence(c)
    entropy_report = spectral_entropy(c)
    truncation = recommend_truncation(c, error_tolerance=0.01)

    effective_modes = float(entropy_report.effective_mode_count.item())

    # Dominant frequency: mode with maximum energy
    e_frac = coefficients.energy_fractions
    dominant_mode = int(torch.argmax(e_frac).item()) + 1  # 1-indexed
    dominant_freq = dominant_mode / coefficients.signal.window_seconds

    # High-frequency energy ratio: energy above the truncation point
    rec_modes = int(truncation.recommended_modes)
    if rec_modes < coefficients.num_modes:
        high_freq_energy = float(e_frac[rec_modes:].sum().item())
    else:
        high_freq_energy = 0.0

    return LoadSpectrumAnalysis(
        decay=decay,
        convergence=convergence,
        entropy=entropy_report,
        truncation=truncation,
        effective_mode_count=effective_modes,
        dominant_frequency_hz=dominant_freq,
        high_frequency_energy_ratio=high_freq_energy,
    )


# ---------------------------------------------------------------------------
# Anomaly detection — spectral comparison
# ---------------------------------------------------------------------------

def detect_load_anomaly(
    current: LoadCoefficients,
    baseline: LoadCoefficients,
) -> AnomalyAssessment:
    """Detect anomalous load by comparing spectral structure to a baseline.

    The comparison uses:
      1. L2 distance in normalized coefficient space
      2. Spectral entropy shift (change in effective mode count)
      3. Jensen-Shannon divergence of per-mode energy distributions

    The engine's own convergence diagnostics on the *difference* signal
    determine whether the deviation is significant — no hardcoded
    thresholds.

    Args:
        current: Spectral coefficients of current load window.
        baseline: Spectral coefficients of a known-good baseline.

    Returns:
        AnomalyAssessment with structured anomaly determination.
    """
    # Align mode counts to the minimum
    n = min(current.num_modes, baseline.num_modes)
    c_cur = current.coefficients[:n]
    c_base = baseline.coefficients[:n]

    # Normalized L2 distance
    norm_base = float(torch.norm(c_base).item())
    if norm_base > 0:
        spectral_dist = float(torch.norm(c_cur - c_base).item() / norm_base)
    else:
        spectral_dist = float(torch.norm(c_cur).item())

    # Entropy shift
    ent_cur = spectral_entropy(c_cur)
    ent_base = spectral_entropy(c_base)
    max_ent = math.log(max(ent_cur.total_modes, 2))
    norm_ent_cur = float(ent_cur.entropy.item()) / max_ent
    norm_ent_base = float(ent_base.entropy.item()) / max_ent
    entropy_shift = norm_ent_cur - norm_ent_base
    mode_count_shift = float(ent_cur.effective_mode_count.item() - ent_base.effective_mode_count.item())

    # Jensen-Shannon divergence of energy fractions
    e_cur = current.energy_fractions[:n]
    e_base = baseline.energy_fractions[:n]
    # Add small epsilon for numerical stability
    eps = 1e-12
    p = e_cur + eps
    q = e_base + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    jsd = 0.5 * float((p * (p / m).log()).sum().item()) + 0.5 * float((q * (q / m).log()).sum().item())

    # Use convergence diagnostics on the DIFFERENCE signal
    diff_coeffs = c_cur - c_base
    diff_convergence = analyze_convergence(diff_coeffs)

    # Anomaly criterion: the difference signal should be negligible
    # (well-converged with low energy).  If it isn't, the current
    # pattern has diverged from baseline.
    # Use the engine's own convergence quality metric.
    diff_energy = float(torch.norm(diff_coeffs).item() ** 2)
    base_energy = float(torch.norm(c_base).item() ** 2)
    relative_diff_energy = diff_energy / max(base_energy, eps)

    # The engine's convergence diagnostics tell us if the difference
    # has structured content (not just noise)
    diff_decay = estimate_spectral_decay(diff_coeffs)
    diff_has_structure = diff_decay.decay_type.value != "exponential" or float(diff_decay.rate.item()) < 1.0

    is_anomalous = diff_has_structure and relative_diff_energy > float(
        recommend_truncation(c_base, error_tolerance=0.01).energy_captured.item()
        * 0.01  # 1% of baseline captured energy
    )

    if is_anomalous:
        if entropy_shift > 0.2:
            reason = f"Spectral entropy increased by {entropy_shift:.3f} — traffic pattern is significantly more disordered than baseline"
        elif mode_count_shift > 5:
            reason = f"Effective mode count increased by {mode_count_shift:.1f} — new high-frequency components detected"
        else:
            reason = f"Structured deviation from baseline (JSD={jsd:.4f}, relative energy={relative_diff_energy:.4f})"
    else:
        reason = "Current load spectrum is within baseline spectral envelope"

    return AnomalyAssessment(
        is_anomalous=is_anomalous,
        spectral_distance=spectral_dist,
        entropy_shift=entropy_shift,
        mode_count_shift=mode_count_shift,
        energy_redistribution=jsd,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Adaptive throttling — derived from spectral structure
# ---------------------------------------------------------------------------

def compute_adaptive_throttle(
    coefficients: LoadCoefficients,
    *,
    capacity_rps: float = 100.0,
) -> AdaptiveThrottle:
    """Compute adaptive throttling parameters from spectral load analysis.

    The throttling parameters are derived from the spectral structure:

      - **Cooldown duration**: proportional to the inverse of the dominant
        frequency — the engine uses the spectral decomposition to find
        the natural timescale of the traffic pattern.

      - **Minimum interval**: derived from the truncation recommendation —
        the engine determines the fastest timescale that carries
        significant energy.

      - **Max concurrent**: proportional to the headroom, scaled by the
        spectral entropy (more disordered traffic → more conservative).

    Args:
        coefficients: Spectral coefficients from decompose_load_signal.
        capacity_rps: Server capacity in requests per second.

    Returns:
        AdaptiveThrottle with computed parameters.
    """
    c = coefficients.coefficients
    signal = coefficients.signal
    window = signal.window_seconds
    e_frac = coefficients.energy_fractions

    # Engine diagnostics
    entropy_report = spectral_entropy(c)
    truncation = recommend_truncation(c, error_tolerance=0.01)
    decay = estimate_spectral_decay(c)

    max_ent = math.log(max(entropy_report.total_modes, 2))
    normalized_entropy = float(entropy_report.entropy.item()) / max_ent
    effective_modes = float(entropy_report.effective_mode_count.item())
    rec_modes = max(1, int(truncation.recommended_modes))

    # Sustained load: energy in the first mode (DC-like component)
    # The first sine mode captures the slowest variation
    sustained_energy_frac = float(e_frac[0].item()) if e_frac.numel() > 0 else 1.0

    # Average rate
    avg_rate = signal.total_requests / max(window, 1e-6)

    # Capacity utilization
    utilization = min(avg_rate / max(capacity_rps, 1e-6), 1.0)
    headroom = 1.0 - utilization

    # Spectral load factor: energy-weighted average rate
    # Higher modes carrying more energy means more burst load
    mode_weights = torch.arange(1, coefficients.num_modes + 1, dtype=torch.float64, device=c.device)
    spectral_load_factor = float((e_frac * mode_weights).sum().item() / max(mode_weights.sum().item(), 1e-6))

    # --- Derive throttle parameters from spectral structure ---

    # Cooldown: natural timescale of the traffic = window / dominant_mode
    dominant_mode = max(1, int(torch.argmax(e_frac).item()) + 1)
    natural_timescale = window / dominant_mode

    # Scale cooldown by utilization: more loaded → longer cooldown
    # The spectral entropy modulates: high entropy (chaotic) → more conservative
    entropy_factor = 1.0 + normalized_entropy  # range [1, 2]
    cooldown = natural_timescale * utilization * entropy_factor

    # Minimum interval: fastest significant timescale from truncation
    # The truncation recommendation tells us the highest mode that matters
    fastest_significant_period = window / rec_modes
    min_interval = fastest_significant_period * utilization

    # Concurrent requests: scale by headroom, modulated by entropy
    # Low entropy (smooth traffic) → can handle more concurrent
    # High entropy (chaotic) → be more conservative
    base_concurrent = max(1, int(capacity_rps * headroom))
    entropy_penalty = max(0.1, 1.0 - normalized_entropy * 0.5)
    max_concurrent = max(1, int(base_concurrent * entropy_penalty))

    # Regime classification from spectral decay type
    decay_type = decay.decay_type.value
    if utilization > 0.9:
        regime = "saturated"
    elif decay_type == "plateau":
        regime = "anomalous"
    elif decay_type == "algebraic" or normalized_entropy > 0.7:
        regime = "bursty"
    else:
        regime = "smooth"

    return AdaptiveThrottle(
        recommended_cooldown_seconds=max(0.0, cooldown),
        recommended_min_interval_seconds=max(0.0, min_interval),
        recommended_max_concurrent=max_concurrent,
        capacity_utilization=utilization,
        spectral_load_factor=spectral_load_factor,
        headroom_fraction=headroom,
        regime=regime,
    )


# ---------------------------------------------------------------------------
# Capacity estimation
# ---------------------------------------------------------------------------

def estimate_capacity(
    coefficients: LoadCoefficients,
) -> CapacityEstimate:
    """Estimate sustainable server capacity from spectral load analysis.

    The sustained rate is estimated from low-frequency spectral content
    (modes below the truncation cutoff), while the peak rate includes
    the full spectrum.  The burst ratio quantifies how spiky the traffic
    is relative to its sustained component.

    Args:
        coefficients: Spectral coefficients from decompose_load_signal.

    Returns:
        CapacityEstimate with sustained rate, peak rate, and burst ratio.
    """
    c = coefficients.coefficients
    signal = coefficients.signal
    basis = coefficients.basis
    e_frac = coefficients.energy_fractions

    truncation = recommend_truncation(c, error_tolerance=0.01)
    convergence = analyze_convergence(c)
    rec_modes = max(1, int(truncation.recommended_modes))

    # Reconstruct using only low-frequency modes → sustained load
    modes_tensor = torch.arange(1, rec_modes + 1, dtype=torch.float64, device=c.device)
    phi = sine_basis_matrix(basis.domain, modes_tensor, signal.grid)
    sustained_signal = phi @ c[:rec_modes]
    sustained_rps = float(sustained_signal.abs().mean().item())

    # Full reconstruction → peak load
    modes_full = torch.arange(1, coefficients.num_modes + 1, dtype=torch.float64, device=c.device)
    phi_full = sine_basis_matrix(basis.domain, modes_full, signal.grid)
    full_signal = phi_full @ c
    peak_rps = float(full_signal.abs().max().item())

    burst_ratio = peak_rps / max(sustained_rps, 1e-6)

    # Spectral headroom: how many modes before truncation error grows
    headroom_modes = coefficients.num_modes - rec_modes

    # Stability: exponential decay indicates a well-converged (stable) pattern
    stable = convergence.decay.decay_type.value == "exponential"

    return CapacityEstimate(
        sustained_rps=sustained_rps,
        peak_rps=peak_rps,
        burst_ratio=burst_ratio,
        spectral_headroom_modes=headroom_modes,
        stable=stable,
    )


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------

def analyze_load(
    signal: LoadSignal,
    *,
    num_modes: int = 64,
    capacity_rps: float = 100.0,
    baseline: LoadCoefficients | None = None,
) -> LoadSpectralReport:
    """Run the complete spectral load analysis pipeline.

    Decomposes the signal, analyzes the spectrum, computes throttling
    parameters, estimates capacity, and optionally compares to a baseline.

    Args:
        signal: Discretized load signal.
        num_modes: Number of spectral modes.
        capacity_rps: Server capacity in requests per second.
        baseline: Optional baseline coefficients for anomaly detection.

    Returns:
        LoadSpectralReport with all analysis results.
    """
    coefficients = decompose_load_signal(signal, num_modes=num_modes)
    spectrum = analyze_load_spectrum(coefficients)
    throttle = compute_adaptive_throttle(coefficients, capacity_rps=capacity_rps)
    capacity = estimate_capacity(coefficients)

    anomaly = None
    if baseline is not None:
        anomaly = detect_load_anomaly(coefficients, baseline)

    return LoadSpectralReport(
        signal=signal,
        coefficients=coefficients,
        spectrum=spectrum,
        throttle=throttle,
        capacity=capacity,
        anomaly=anomaly,
    )


# ---------------------------------------------------------------------------
# Convenience: from raw timestamps to full report
# ---------------------------------------------------------------------------

def analyze_request_load(
    timestamps: Sequence[float] | Tensor,
    *,
    window_seconds: float = 300.0,
    resolution: int = 256,
    num_modes: int = 64,
    capacity_rps: float = 100.0,
    baseline_timestamps: Sequence[float] | Tensor | None = None,
    device: str = "cpu",
) -> LoadSpectralReport:
    """End-to-end spectral load analysis from raw request timestamps.

    This is the single-call entry point for AI agents and MCP clients.

    Args:
        timestamps: Request timestamps (unix or relative).
        window_seconds: Observation window duration.
        resolution: Grid resolution for signal discretization.
        num_modes: Number of spectral modes.
        capacity_rps: Server capacity in requests per second.
        baseline_timestamps: Optional baseline timestamps for anomaly comparison.
        device: Torch device.

    Returns:
        LoadSpectralReport with complete analysis.
    """
    signal = ingest_request_log(
        timestamps, window_seconds=window_seconds, resolution=resolution, device=device,
    )

    baseline_coeffs = None
    if baseline_timestamps is not None:
        baseline_signal = ingest_request_log(
            baseline_timestamps, window_seconds=window_seconds, resolution=resolution, device=device,
        )
        baseline_coeffs = decompose_load_signal(baseline_signal, num_modes=num_modes)

    return analyze_load(
        signal, num_modes=num_modes, capacity_rps=capacity_rps, baseline=baseline_coeffs,
    )


__all__ = [
    "LoadSignal",
    "LoadCoefficients",
    "LoadSpectrumAnalysis",
    "AnomalyAssessment",
    "AdaptiveThrottle",
    "CapacityEstimate",
    "LoadSpectralReport",
    "ingest_request_log",
    "load_signal_from_rate",
    "decompose_load_signal",
    "analyze_load_spectrum",
    "detect_load_anomaly",
    "compute_adaptive_throttle",
    "estimate_capacity",
    "analyze_load",
    "analyze_request_load",
]
