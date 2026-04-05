"""Spectral convergence diagnostics.

In spectral methods, the rate at which coefficients decay determines how
well the truncated expansion approximates the original function:

- **Exponential decay** (|c_n| ~ exp(-alpha*n)):  The function is analytic
  (infinitely differentiable).  Truncation is safe; a modest number of
  modes captures essentially all the information.

- **Algebraic decay** (|c_n| ~ n^{-beta}):  The function has finite
  regularity (beta-1 continuous derivatives).  More modes are needed,
  and the convergence is slower.

- **No decay / plateau**:  The function has a discontinuity or the
  data is noisy.  The Gibbs phenomenon may be present, and spectral
  methods need special treatment (filtering, Gegenbauer reconstruction).

This module estimates:
1. Spectral decay rate (exponential or algebraic)
2. Gibbs phenomenon detection
3. Spectral entropy (effective mode count)
4. Optimal truncation point recommendation
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from spectral_packet_engine.domain import coerce_tensor

Tensor = torch.Tensor


class DecayType(Enum):
    """Classification of spectral coefficient decay."""
    EXPONENTIAL = "exponential"
    ALGEBRAIC = "algebraic"
    PLATEAU = "plateau"


@dataclass(frozen=True, slots=True)
class SpectralDecayEstimate:
    """Result of spectral decay rate estimation."""
    decay_type: DecayType
    rate: Tensor           # alpha for exponential, beta for algebraic
    r_squared: Tensor      # goodness of fit (0 to 1)
    log_coefficients: Tensor  # log |c_n| used for fitting
    fit_residual: Tensor   # RMS residual of the fit


@dataclass(frozen=True, slots=True)
class GibbsAnalysis:
    """Gibbs phenomenon detection results."""
    detected: bool
    overshoot_ratio: Tensor    # max(reconstruction) / max(original) near discontinuity
    ringing_energy: Tensor     # fraction of energy in high-mode oscillations
    affected_region_fraction: Tensor  # what fraction of domain shows Gibbs ringing


@dataclass(frozen=True, slots=True)
class SpectralEntropyReport:
    """Spectral entropy analysis — how many modes are "effectively active"."""
    entropy: Tensor              # Shannon entropy of normalized spectral weights
    effective_mode_count: Tensor  # exp(entropy) — the effective number of modes
    total_modes: int
    concentration_ratio: Tensor  # fraction of energy in top-k modes
    sparsity: Tensor             # 1 - effective_mode_count/total_modes


@dataclass(frozen=True, slots=True)
class TruncationRecommendation:
    """Optimal truncation point recommendation."""
    recommended_modes: int
    estimated_error: Tensor      # relative L2 error estimate from tail mass
    energy_captured: Tensor      # fraction of total energy in kept modes
    safety_margin_modes: int     # extra modes for safety (2x error tolerance)


@dataclass(frozen=True, slots=True)
class ConvergenceDiagnostics:
    """Complete spectral convergence analysis."""
    decay: SpectralDecayEstimate
    entropy: SpectralEntropyReport
    truncation: TruncationRecommendation
    gibbs: GibbsAnalysis | None  # None if Gibbs analysis not requested


def estimate_spectral_decay(coefficients: Tensor, *, min_modes: int = 4) -> SpectralDecayEstimate:
    """Estimate the spectral decay rate from coefficient magnitudes.

    Fits both exponential (log|c_n| ~ -alpha*n) and algebraic
    (log|c_n| ~ -beta*log(n)) models and picks the better fit.
    """
    coeffs = coerce_tensor(coefficients)
    if coeffs.ndim != 1:
        raise ValueError("expects a one-dimensional coefficient vector")

    # Compute log magnitudes, filtering near-zero coefficients
    magnitudes = torch.abs(coeffs) if torch.is_complex(coeffs) else torch.abs(coeffs)
    eps = torch.finfo(magnitudes.dtype).tiny
    nonzero_mask = magnitudes > eps * 1000
    if nonzero_mask.sum() < min_modes:
        return SpectralDecayEstimate(
            decay_type=DecayType.PLATEAU,
            rate=torch.tensor(0.0, dtype=magnitudes.dtype, device=magnitudes.device),
            r_squared=torch.tensor(0.0, dtype=magnitudes.dtype, device=magnitudes.device),
            log_coefficients=torch.log(magnitudes.clamp(min=eps)),
            fit_residual=torch.tensor(float("inf"), dtype=magnitudes.dtype, device=magnitudes.device),
        )

    log_mag = torch.log(magnitudes.clamp(min=eps))
    n = torch.arange(1, len(coeffs) + 1, dtype=magnitudes.dtype, device=magnitudes.device)

    # Fit exponential: log|c_n| = a - alpha*n
    exp_fit = _linear_fit(n[nonzero_mask], log_mag[nonzero_mask])

    # Fit algebraic: log|c_n| = a - beta*log(n)
    alg_fit = _linear_fit(torch.log(n[nonzero_mask]), log_mag[nonzero_mask])

    # Pick better fit based on R^2
    if exp_fit.r_squared > alg_fit.r_squared and exp_fit.r_squared > 0.7:
        return SpectralDecayEstimate(
            decay_type=DecayType.EXPONENTIAL,
            rate=-exp_fit.slope,  # alpha > 0 for decay
            r_squared=exp_fit.r_squared,
            log_coefficients=log_mag,
            fit_residual=exp_fit.residual,
        )
    elif alg_fit.r_squared > 0.5:
        return SpectralDecayEstimate(
            decay_type=DecayType.ALGEBRAIC,
            rate=-alg_fit.slope,  # beta > 0 for decay
            r_squared=alg_fit.r_squared,
            log_coefficients=log_mag,
            fit_residual=alg_fit.residual,
        )
    else:
        # Neither fits well — plateau or noisy
        rate = -exp_fit.slope if exp_fit.r_squared > alg_fit.r_squared else -alg_fit.slope
        r2 = max(exp_fit.r_squared, alg_fit.r_squared)
        return SpectralDecayEstimate(
            decay_type=DecayType.PLATEAU,
            rate=rate,
            r_squared=r2,
            log_coefficients=log_mag,
            fit_residual=min(exp_fit.residual, alg_fit.residual),
        )


@dataclass(frozen=True, slots=True)
class _LinearFitResult:
    slope: Tensor
    intercept: Tensor
    r_squared: Tensor
    residual: Tensor


def _linear_fit(x: Tensor, y: Tensor) -> _LinearFitResult:
    """Simple least-squares linear regression y = a + b*x."""
    n = x.shape[0]
    if n < 2:
        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        return _LinearFitResult(slope=zero, intercept=zero, r_squared=zero, residual=zero)

    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    dx = x - x_mean
    dy = y - y_mean

    ss_xx = torch.sum(dx ** 2)
    ss_xy = torch.sum(dx * dy)
    ss_yy = torch.sum(dy ** 2)

    slope = ss_xy / ss_xx.clamp(min=torch.finfo(x.dtype).tiny)
    intercept = y_mean - slope * x_mean

    y_pred = intercept + slope * x
    ss_res = torch.sum((y - y_pred) ** 2)
    r_squared = (1 - ss_res / ss_yy.clamp(min=torch.finfo(x.dtype).tiny)).clamp(min=0, max=1)
    residual = torch.sqrt(ss_res / n)

    return _LinearFitResult(slope=slope, intercept=intercept, r_squared=r_squared, residual=residual)


def spectral_entropy(coefficients: Tensor) -> SpectralEntropyReport:
    """Compute the spectral entropy — a measure of effective mode count.

    The Shannon entropy of the normalized spectral weight distribution
    p_n = |c_n|^2 / sum|c_k|^2 quantifies how many modes are
    effectively contributing.  exp(H) gives the effective mode count:

    - H = 0 (1 effective mode): all energy in a single mode
    - H = log(N) (N effective modes): energy equally distributed

    The sparsity = 1 - exp(H)/N measures how concentrated the spectrum is.
    """
    coeffs = coerce_tensor(coefficients)
    if coeffs.ndim != 1:
        raise ValueError("expects a one-dimensional coefficient vector")

    N = coeffs.shape[0]
    if torch.is_complex(coeffs):
        weights = torch.abs(coeffs) ** 2
    else:
        weights = coeffs ** 2

    total = torch.sum(weights)
    if total <= 0:
        zero = torch.tensor(0.0, dtype=weights.dtype, device=weights.device)
        return SpectralEntropyReport(
            entropy=zero,
            effective_mode_count=torch.tensor(1.0, dtype=weights.dtype, device=weights.device),
            total_modes=N,
            concentration_ratio=zero,
            sparsity=torch.tensor(1.0, dtype=weights.dtype, device=weights.device),
        )

    p = weights / total
    # Shannon entropy: H = -sum p_n log(p_n), handling p=0
    log_p = torch.log(p.clamp(min=torch.finfo(p.dtype).tiny))
    H = -torch.sum(p * log_p)
    effective = torch.exp(H)

    # Concentration: fraction of energy in modes with above-average weight
    threshold = 1.0 / N
    concentration = torch.sum(p[p > threshold])

    return SpectralEntropyReport(
        entropy=H,
        effective_mode_count=effective,
        total_modes=N,
        concentration_ratio=concentration,
        sparsity=(1 - effective / N).clamp(min=0, max=1),
    )


def detect_gibbs(
    coefficients: Tensor,
    reconstruction: Tensor,
    original: Tensor,
    grid: Tensor,
) -> GibbsAnalysis:
    """Detect Gibbs phenomenon by comparing reconstruction overshoot to original.

    The Gibbs phenomenon manifests as ~9% overshoot near discontinuities
    and oscillatory "ringing" that doesn't diminish with more modes.
    """
    coeffs = coerce_tensor(coefficients)
    recon = coerce_tensor(reconstruction)
    orig = coerce_tensor(original)
    x = coerce_tensor(grid)

    if recon.ndim > 1 or orig.ndim > 1:
        raise ValueError("Gibbs detection expects single profiles, not batches")

    orig_max = torch.max(torch.abs(orig))
    recon_max = torch.max(torch.abs(recon))

    overshoot = recon_max / orig_max.clamp(min=torch.finfo(orig.dtype).tiny)

    # Ringing energy: high-frequency content relative to total
    N = coeffs.shape[-1]
    if torch.is_complex(coeffs):
        weights = torch.abs(coeffs) ** 2
    else:
        weights = coeffs ** 2
    total_energy = torch.sum(weights)
    high_mode_start = max(1, N // 2)
    ringing_energy = torch.sum(weights[high_mode_start:]) / total_energy.clamp(
        min=torch.finfo(weights.dtype).tiny
    )

    # Region affected: where |recon - orig| > 5% of orig_max
    residual = torch.abs(recon - orig)
    affected = (residual > 0.05 * orig_max).to(dtype=x.dtype)
    affected_fraction = torch.mean(affected)

    # Gibbs detected if overshoot > 5% AND ringing energy is significant
    detected = bool(overshoot > 1.05 and ringing_energy > 0.01)

    return GibbsAnalysis(
        detected=detected,
        overshoot_ratio=overshoot,
        ringing_energy=ringing_energy,
        affected_region_fraction=affected_fraction,
    )


def recommend_truncation(
    coefficients: Tensor,
    *,
    error_tolerance: float = 0.01,
) -> TruncationRecommendation:
    """Recommend the optimal number of modes to keep.

    Uses the tail mass (1 - cumulative energy) as an estimate of
    relative L2 error: error ≈ sqrt(tail_mass).
    """
    coeffs = coerce_tensor(coefficients)
    if coeffs.ndim != 1:
        raise ValueError("expects a one-dimensional coefficient vector")

    if torch.is_complex(coeffs):
        weights = torch.abs(coeffs) ** 2
    else:
        weights = coeffs ** 2

    total = torch.sum(weights)
    if total <= 0:
        return TruncationRecommendation(
            recommended_modes=1,
            estimated_error=torch.tensor(1.0, dtype=weights.dtype, device=weights.device),
            energy_captured=torch.tensor(0.0, dtype=weights.dtype, device=weights.device),
            safety_margin_modes=2,
        )

    cumulative = torch.cumsum(weights, dim=0) / total
    tail_mass = 1 - cumulative
    estimated_error = torch.sqrt(tail_mass)

    # Find first mode where estimated error < tolerance
    meets_tolerance = estimated_error < error_tolerance
    if torch.any(meets_tolerance):
        recommended = int(torch.argmax(meets_tolerance.to(torch.int64)).item()) + 1
    else:
        recommended = coeffs.shape[0]

    # Safety margin: use 1.5x recommended modes
    safety = min(int(recommended * 1.5) + 1, coeffs.shape[0])

    return TruncationRecommendation(
        recommended_modes=recommended,
        estimated_error=estimated_error[recommended - 1],
        energy_captured=cumulative[recommended - 1],
        safety_margin_modes=safety,
    )


def analyze_convergence(
    coefficients: Tensor,
    *,
    error_tolerance: float = 0.01,
    reconstruction: Tensor | None = None,
    original: Tensor | None = None,
    grid: Tensor | None = None,
) -> ConvergenceDiagnostics:
    """Run the full convergence diagnostic suite.

    Estimates decay rate, spectral entropy, and optimal truncation.
    Optionally detects Gibbs phenomenon if reconstruction data is provided.
    """
    decay = estimate_spectral_decay(coefficients)
    entropy = spectral_entropy(coefficients)
    truncation = recommend_truncation(coefficients, error_tolerance=error_tolerance)

    gibbs = None
    if reconstruction is not None and original is not None and grid is not None:
        gibbs = detect_gibbs(coefficients, reconstruction, original, grid)

    return ConvergenceDiagnostics(
        decay=decay,
        entropy=entropy,
        truncation=truncation,
        gibbs=gibbs,
    )


__all__ = [
    "ConvergenceDiagnostics",
    "DecayType",
    "GibbsAnalysis",
    "SpectralDecayEstimate",
    "SpectralEntropyReport",
    "TruncationRecommendation",
    "analyze_convergence",
    "detect_gibbs",
    "estimate_spectral_decay",
    "recommend_truncation",
    "spectral_entropy",
]
