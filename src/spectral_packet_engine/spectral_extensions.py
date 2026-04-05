"""Extended spectral analysis methods for time-series and profile data.

This module provides mathematical tools that complement the core
infinite-well basis decomposition with signal-processing and
approximation-theoretic methods:

- **Discrete Fourier decomposition**: FFT-based frequency analysis for
  periodic or quasi-periodic profiles.
- **Padé approximants**: Rational function extrapolation from truncated
  spectral coefficients — converges where Taylor series diverge.
- **Hilbert transform**: Compute the analytic signal to extract
  instantaneous amplitude and phase envelopes.
- **Correlation spectral analysis**: Eigenvalue decomposition of the
  cross-correlation matrix among multiple profiles.
- **Richardson extrapolation**: Accelerate convergence of any sequence
  of numerical approximations.
- **Kramers-Kronig relations**: Recover imaginary part of a response
  function from its real part (or vice versa) via dispersion integrals.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch

from spectral_packet_engine.domain import coerce_tensor

Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FourierDecomposition:
    """Result of discrete Fourier decomposition."""
    frequencies: list[float]
    amplitudes: list[float]
    phases: list[float]
    power_spectrum: list[float]
    dominant_frequencies: list[float]
    dominant_amplitudes: list[float]
    total_power: float
    nyquist_frequency: float

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


@dataclass(frozen=True, slots=True)
class PadeApproximant:
    """Result of Padé approximant construction."""
    numerator_coefficients: list[float]
    denominator_coefficients: list[float]
    order_m: int
    order_n: int
    poles: list[float]
    evaluation_points: list[float]
    evaluation_values: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


@dataclass(frozen=True, slots=True)
class HilbertAnalysis:
    """Result of Hilbert transform analysis."""
    analytic_signal_real: list[float]
    analytic_signal_imag: list[float]
    instantaneous_amplitude: list[float]
    instantaneous_phase: list[float]
    instantaneous_frequency: list[float]
    mean_frequency: float

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


@dataclass(frozen=True, slots=True)
class CorrelationSpectralAnalysis:
    """Result of correlation matrix spectral analysis."""
    eigenvalues: list[float]
    explained_variance_ratio: list[float]
    cumulative_variance: list[float]
    num_significant_components: int
    condition_number: float
    principal_components: list[list[float]]
    correlation_matrix: list[list[float]]

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


@dataclass(frozen=True, slots=True)
class RichardsonResult:
    """Result of Richardson extrapolation."""
    estimates: list[float]
    extrapolated_value: float
    convergence_order: float
    error_estimate: float

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


@dataclass(frozen=True, slots=True)
class KramersKronigResult:
    """Result of Kramers-Kronig transform."""
    input_frequencies: list[float]
    input_values: list[float]
    output_values: list[float]
    direction: str

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


# ---------------------------------------------------------------------------
# Fourier decomposition
# ---------------------------------------------------------------------------

def fourier_decomposition(
    signal,
    *,
    sample_spacing: float = 1.0,
    num_dominant: int = 5,
) -> FourierDecomposition:
    """Compute the discrete Fourier decomposition of a 1-D signal.

    Parameters
    ----------
    signal : array-like
        Real-valued 1-D signal.
    sample_spacing : float
        Spacing between samples (inverse of sampling rate).
    num_dominant : int
        Number of dominant frequency components to report.
    """
    x = coerce_tensor(signal).double()
    if x.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    n = x.shape[0]
    if n < 2:
        raise ValueError("signal must have at least 2 samples")

    fft_result = torch.fft.rfft(x)
    freqs = torch.fft.rfftfreq(n, d=sample_spacing)
    amplitudes = (2.0 / n) * torch.abs(fft_result)
    amplitudes[0] /= 2.0  # DC component
    phases = torch.angle(fft_result)
    power = amplitudes ** 2

    # Sort by amplitude descending (skip DC)
    positive_mask = freqs > 0
    positive_amps = amplitudes[positive_mask]
    positive_freqs = freqs[positive_mask]
    sorted_indices = torch.argsort(positive_amps, descending=True)
    k = min(num_dominant, sorted_indices.shape[0])
    top_indices = sorted_indices[:k]

    return FourierDecomposition(
        frequencies=freqs.tolist(),
        amplitudes=amplitudes.tolist(),
        phases=phases.tolist(),
        power_spectrum=power.tolist(),
        dominant_frequencies=positive_freqs[top_indices].tolist(),
        dominant_amplitudes=positive_amps[top_indices].tolist(),
        total_power=float(power.sum()),
        nyquist_frequency=float(freqs[-1]),
    )


# ---------------------------------------------------------------------------
# Padé approximant
# ---------------------------------------------------------------------------

def pade_approximant(
    coefficients,
    *,
    order_m: int = 3,
    order_n: int = 3,
    evaluation_points: Sequence[float] | None = None,
) -> PadeApproximant:
    r"""Construct a [m/n] Padé approximant from power series coefficients.

    Given a truncated Taylor series  f(x) = c_0 + c_1 x + ... + c_{m+n} x^{m+n},
    the Padé approximant is the rational function  P_m(x)/Q_n(x)  that
    matches the series through order m+n.

    This often converges where the Taylor series diverges, making it
    valuable for extrapolating spectral coefficient sequences.
    """
    c = coerce_tensor(coefficients).double()
    if c.ndim != 1:
        raise ValueError("coefficients must be one-dimensional")
    needed = order_m + order_n + 1
    if c.shape[0] < needed:
        raise ValueError(
            f"need at least {needed} coefficients for [{order_m}/{order_n}] Padé, "
            f"got {c.shape[0]}"
        )
    c = c[:needed]

    # Solve the Padé equations: build the linear system for denominator coefficients
    # Q(x) = 1 + q_1 x + ... + q_n x^n
    # P(x) = p_0 + p_1 x + ... + p_m x^m
    # such that P(x) - Q(x) * f(x) = O(x^{m+n+1})
    if order_n == 0:
        p_coeffs = c[:order_m + 1].tolist()
        q_coeffs = [1.0]
    else:
        # Build the system for q coefficients
        A = torch.zeros(order_n, order_n, dtype=torch.float64)
        b = torch.zeros(order_n, dtype=torch.float64)
        for i in range(order_n):
            row = order_m + 1 + i
            b[i] = -c[row]
            for j in range(order_n):
                idx = row - j - 1
                if 0 <= idx < needed:
                    A[i, j] = c[idx]

        q = torch.linalg.solve(A, b)
        q_coeffs = [1.0] + q.tolist()

        # Compute numerator: p_k = c_k + sum_{j=1}^{min(k,n)} q_j * c_{k-j}
        q_full = torch.tensor(q_coeffs, dtype=torch.float64)
        p_coeffs = []
        for k in range(order_m + 1):
            pk = float(c[k])
            for j in range(1, min(k, order_n) + 1):
                pk += float(q_full[j]) * float(c[k - j])
            p_coeffs.append(pk)

    # Find poles (roots of denominator)
    poles: list[float] = []
    if order_n > 0:
        q_tensor = torch.tensor(q_coeffs, dtype=torch.float64)
        # Reverse to match numpy polynomial convention (highest power first)
        q_reversed = torch.flip(q_tensor, dims=[0])
        companion = torch.zeros(order_n, order_n, dtype=torch.float64)
        for i in range(order_n - 1):
            companion[i + 1, i] = 1.0
        for i in range(order_n):
            companion[i, -1] = -q_reversed[i + 1] / q_reversed[0] if q_reversed[0] != 0 else 0.0
        try:
            eigvals = torch.linalg.eigvals(companion)
            poles = [float(e.real) for e in eigvals if abs(float(e.imag)) < 1e-10]
        except Exception:
            poles = []

    # Evaluate at requested points
    eval_pts = evaluation_points or []
    eval_vals = []
    p_t = torch.tensor(p_coeffs, dtype=torch.float64)
    q_t = torch.tensor(q_coeffs, dtype=torch.float64)
    for x in eval_pts:
        x_powers = torch.tensor([x ** k for k in range(max(len(p_coeffs), len(q_coeffs)))], dtype=torch.float64)
        num = float(torch.dot(p_t, x_powers[:len(p_coeffs)]))
        den = float(torch.dot(q_t, x_powers[:len(q_coeffs)]))
        eval_vals.append(num / den if abs(den) > 1e-15 else float("nan"))

    return PadeApproximant(
        numerator_coefficients=p_coeffs,
        denominator_coefficients=q_coeffs,
        order_m=order_m,
        order_n=order_n,
        poles=poles,
        evaluation_points=list(eval_pts),
        evaluation_values=eval_vals,
    )


# ---------------------------------------------------------------------------
# Hilbert transform
# ---------------------------------------------------------------------------

def hilbert_transform(signal) -> HilbertAnalysis:
    r"""Compute the analytic signal via the Hilbert transform.

    The analytic signal z(t) = x(t) + i H[x](t) encodes the
    instantaneous amplitude A(t) = |z(t)| and phase φ(t) = arg(z(t)).
    """
    x = coerce_tensor(signal).double()
    if x.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    n = x.shape[0]
    if n < 2:
        raise ValueError("signal must have at least 2 samples")

    # FFT-based Hilbert transform
    X = torch.fft.fft(x)
    h = torch.zeros(n, dtype=torch.float64)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[1:(n + 1) // 2] = 2.0

    analytic = torch.fft.ifft(X * h)
    amplitude = torch.abs(analytic)
    phase = torch.angle(analytic)

    # Instantaneous frequency = d(phase)/dt / (2*pi)
    inst_freq = torch.diff(torch.unwrap(phase)) / (2.0 * math.pi)
    # Pad to match length
    inst_freq = torch.cat([inst_freq, inst_freq[-1:]])

    return HilbertAnalysis(
        analytic_signal_real=analytic.real.tolist(),
        analytic_signal_imag=analytic.imag.tolist(),
        instantaneous_amplitude=amplitude.tolist(),
        instantaneous_phase=phase.tolist(),
        instantaneous_frequency=inst_freq.tolist(),
        mean_frequency=float(inst_freq.mean()),
    )


def _torch_unwrap(phase: Tensor) -> Tensor:
    """Unwrap phase angles (torch doesn't have built-in unwrap in all versions)."""
    diff = torch.diff(phase)
    diff = diff - 2 * math.pi * torch.round(diff / (2 * math.pi))
    return torch.cat([phase[:1], phase[:1] + torch.cumsum(diff, dim=0)])


# Monkey-patch if needed
if not hasattr(torch, "unwrap"):
    torch.unwrap = _torch_unwrap


# ---------------------------------------------------------------------------
# Correlation spectral analysis
# ---------------------------------------------------------------------------

def correlation_spectral_analysis(
    profiles,
    *,
    significance_threshold: float = 0.05,
) -> CorrelationSpectralAnalysis:
    """Eigenvalue decomposition of the cross-correlation matrix among profiles.

    This is a spectral (PCA-style) analysis of how multiple
    time-series/profiles co-vary.  The eigenvalues reveal the number
    of independent driving modes.

    Parameters
    ----------
    profiles : array-like  (num_profiles, num_points)
        Each row is a separate profile or time series.
    significance_threshold : float
        Eigenvalues below this fraction of total variance are insignificant.
    """
    X = coerce_tensor(profiles).double()
    if X.ndim == 1:
        X = X.unsqueeze(0)
    if X.ndim != 2:
        raise ValueError("profiles must be 2-D (num_profiles, num_points)")

    # Center each profile
    X_centered = X - X.mean(dim=1, keepdim=True)

    # Correlation matrix (normalised covariance)
    stds = X_centered.std(dim=1, keepdim=True).clamp(min=1e-15)
    X_normed = X_centered / stds
    n = X.shape[0]
    corr = (X_normed @ X_normed.T) / X.shape[1]

    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(corr)
    # Sort descending
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    total = eigenvalues.sum().clamp(min=1e-15)
    explained = eigenvalues / total
    cumulative = torch.cumsum(explained, dim=0)

    num_significant = int((explained > significance_threshold).sum().item())
    num_significant = max(1, num_significant)

    cond = float(eigenvalues[0] / eigenvalues[-1].clamp(min=1e-15))

    return CorrelationSpectralAnalysis(
        eigenvalues=eigenvalues.tolist(),
        explained_variance_ratio=explained.tolist(),
        cumulative_variance=cumulative.tolist(),
        num_significant_components=num_significant,
        condition_number=cond,
        principal_components=eigenvectors.T.tolist(),
        correlation_matrix=corr.tolist(),
    )


# ---------------------------------------------------------------------------
# Richardson extrapolation
# ---------------------------------------------------------------------------

def richardson_extrapolation(
    estimates,
    *,
    step_ratio: float = 2.0,
    convergence_order: float | None = None,
) -> RichardsonResult:
    r"""Accelerate convergence of a sequence of numerical estimates.

    Given a sequence A(h), A(h/r), A(h/r²), ... where r is the step
    ratio, Richardson extrapolation removes the leading error term to
    produce a higher-order estimate.

    If the convergence order p is not known, it is estimated from
    three consecutive values.
    """
    vals = coerce_tensor(estimates).double()
    if vals.ndim != 1 or vals.shape[0] < 2:
        raise ValueError("need at least 2 estimates")

    n = vals.shape[0]

    # Estimate convergence order if not given
    if convergence_order is None and n >= 3:
        a0, a1, a2 = float(vals[0]), float(vals[1]), float(vals[2])
        num = a2 - a1
        den = a1 - a0
        if abs(den) > 1e-15 and abs(num) > 1e-15:
            ratio = abs(den / num)
            convergence_order = math.log(ratio) / math.log(step_ratio)
        else:
            convergence_order = 1.0
    elif convergence_order is None:
        convergence_order = 1.0

    # Apply Richardson tableau
    tableau = vals.clone()
    for level in range(1, n):
        r_p = step_ratio ** (convergence_order * level)
        new_tableau = torch.zeros(n - level, dtype=torch.float64)
        for i in range(n - level):
            new_tableau[i] = (r_p * tableau[i + 1] - tableau[i]) / (r_p - 1.0)
        tableau = new_tableau

    extrapolated = float(tableau[0])
    error = abs(extrapolated - float(vals[-1])) if n >= 2 else 0.0

    return RichardsonResult(
        estimates=vals.tolist(),
        extrapolated_value=extrapolated,
        convergence_order=convergence_order,
        error_estimate=error,
    )


# ---------------------------------------------------------------------------
# Kramers-Kronig
# ---------------------------------------------------------------------------

def kramers_kronig(
    frequencies,
    values,
    *,
    direction: str = "real_to_imag",
) -> KramersKronigResult:
    r"""Compute Kramers-Kronig dispersion relation.

    For a causal response function χ(ω) = χ'(ω) + i χ''(ω), the
    Kramers-Kronig relations connect real and imaginary parts:

        χ''(ω) = -(1/π) P ∫ χ'(ω') / (ω' - ω) dω'
        χ'(ω)  =  (1/π) P ∫ χ''(ω') / (ω' - ω) dω'

    where P denotes the Cauchy principal value.

    Parameters
    ----------
    direction : str
        "real_to_imag" or "imag_to_real".
    """
    w = coerce_tensor(frequencies).double()
    v = coerce_tensor(values).double()
    if w.shape != v.shape or w.ndim != 1:
        raise ValueError("frequencies and values must be 1-D with matching length")
    if direction not in ("real_to_imag", "imag_to_real"):
        raise ValueError("direction must be 'real_to_imag' or 'imag_to_real'")

    n = w.shape[0]
    result = torch.zeros(n, dtype=torch.float64)

    for i in range(n):
        # Principal value integral via trapezoidal rule, skipping the pole
        integrand = torch.zeros(n, dtype=torch.float64)
        for j in range(n):
            if i == j:
                continue
            integrand[j] = v[j] / (w[j] - w[i])
        pv = float(torch.trapezoid(integrand, w))

        if direction == "real_to_imag":
            result[i] = -pv / math.pi
        else:
            result[i] = pv / math.pi

    return KramersKronigResult(
        input_frequencies=w.tolist(),
        input_values=v.tolist(),
        output_values=result.tolist(),
        direction=direction,
    )


__all__ = [
    "CorrelationSpectralAnalysis",
    "FourierDecomposition",
    "HilbertAnalysis",
    "KramersKronigResult",
    "PadeApproximant",
    "RichardsonResult",
    "correlation_spectral_analysis",
    "fourier_decomposition",
    "hilbert_transform",
    "kramers_kronig",
    "pade_approximant",
    "richardson_extrapolation",
]
