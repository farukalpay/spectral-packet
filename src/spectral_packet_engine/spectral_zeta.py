"""Spectral zeta function, heat kernel, partition function, and Casimir energy.

The spectral zeta function of a positive operator H with eigenvalues
{E_n} is defined as:

    zeta_H(s) = sum_n E_n^{-s}

which converges for Re(s) sufficiently large.  It is intimately related
to the heat kernel trace:

    K(t) = Tr(e^{-tH}) = sum_n e^{-E_n t}

via the Mellin transform:

    zeta_H(s) = (1/Gamma(s)) int_0^infty t^{s-1} K(t) dt

The partition function of statistical mechanics is the heat kernel
evaluated at inverse temperature:

    Z(beta) = sum_n e^{-beta E_n}

from which all thermodynamic quantities follow.

Weyl's law gives the asymptotic density of eigenvalues.  In 1-D:

    N(E) ~ (L / (pi * hbar)) sqrt(2mE)

The Casimir energy is the regularised sum of zero-point energies:

    E_Cas = (1/2) zeta_H(-1)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from spectral_packet_engine.domain import coerce_tensor

Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# Spectral zeta function
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SpectralZetaResult:
    """Spectral zeta function evaluated at one or more s values."""

    s_values: Tensor
    zeta_values: Tensor
    residue_at_pole: Tensor | None


def spectral_zeta(
    eigenvalues: Tensor,
    s: Tensor | float,
    *,
    regularization: str = "direct",
) -> Tensor:
    r"""Compute zeta_H(s) = sum_n E_n^{-s}.

    Parameters
    ----------
    eigenvalues:
        Positive eigenvalues of the operator.
    s:
        Complex parameter(s).  May be a scalar or a 1-D tensor.
    regularization:
        ``"direct"`` -- straightforward sum (converges for Re(s) > d/2).
        ``"heat_kernel"`` -- compute via numerical Mellin transform of
        the heat kernel trace.

    Returns
    -------
    Tensor of zeta values with the same shape as *s*.
    """
    eigenvalues = coerce_tensor(eigenvalues, dtype=torch.float64)
    if not isinstance(s, Tensor):
        s = torch.tensor(s, dtype=torch.complex128)
    s = coerce_tensor(s, dtype=torch.complex128)
    scalar_input = s.ndim == 0
    if scalar_input:
        s = s.unsqueeze(0)

    # Filter to positive eigenvalues
    pos = eigenvalues[eigenvalues > 0]

    if regularization == "direct":
        # E_n^{-s} = exp(-s * ln(E_n))
        log_E = torch.log(pos).to(dtype=torch.complex128)  # (M,)
        # Outer: (num_s, M)
        exponents = -s.unsqueeze(-1) * log_E.unsqueeze(0)
        terms = torch.exp(exponents)
        zeta_vals = terms.sum(dim=-1)
    elif regularization == "heat_kernel":
        # Numerical Mellin: zeta(s) = 1/Gamma(s) * int_0^inf t^{s-1} K(t) dt
        # Use log-spaced quadrature for the integral
        num_t = 500
        t_min, t_max = 1e-4, 50.0
        log_t = torch.linspace(
            torch.log(torch.tensor(t_min)).item(),
            torch.log(torch.tensor(t_max)).item(),
            num_t,
            dtype=torch.float64,
        )
        t = torch.exp(log_t)  # (num_t,)
        dt = t[1:] - t[:-1]

        # K(t) = sum_n exp(-E_n * t)
        K = torch.exp(-pos.unsqueeze(0) * t.unsqueeze(1)).sum(dim=1)  # (num_t,)

        zeta_vals_list = []
        for si in range(s.shape[0]):
            s_val = s[si]
            # Integrand: t^{s-1} * K(t)
            integrand = torch.exp((s_val - 1) * log_t.to(dtype=torch.complex128)) * K.to(dtype=torch.complex128)
            # Trapezoidal rule
            avg = (integrand[:-1] + integrand[1:]) / 2.0
            integral = (avg * dt.to(dtype=torch.complex128)).sum()
            gamma_s = torch.exp(torch.lgamma(s_val.real.to(dtype=torch.float64))).to(dtype=torch.complex128)
            safe_gamma = torch.where(torch.abs(gamma_s) > 1e-30, gamma_s, torch.ones_like(gamma_s))
            zeta_vals_list.append(integral / safe_gamma)
        zeta_vals = torch.stack(zeta_vals_list)
    else:
        raise ValueError(f"unknown regularization method: {regularization!r}")

    if scalar_input:
        return zeta_vals.squeeze(0)
    return zeta_vals


# ---------------------------------------------------------------------------
# Heat kernel
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class HeatKernelResult:
    """Heat kernel trace analysis."""

    times: Tensor
    trace: Tensor
    short_time_coeffs: Tensor | None


def heat_kernel_trace(
    eigenvalues: Tensor,
    times: Tensor,
) -> Tensor:
    r"""K(t) = Tr(e^{-tH}) = sum_n e^{-E_n t}.

    Parameters
    ----------
    eigenvalues:
        Shape ``(N,)`` -- positive eigenvalues.
    times:
        Shape ``(T,)`` -- time points at which to evaluate the trace.

    Returns
    -------
    Tensor of shape ``(T,)`` with the heat kernel trace.
    """
    eigenvalues = coerce_tensor(eigenvalues, dtype=torch.float64)
    times = coerce_tensor(times, dtype=torch.float64)
    # (T, N) = exp(-t_i * E_n)
    exponents = -times.unsqueeze(-1) * eigenvalues.unsqueeze(0)
    return torch.exp(exponents).sum(dim=-1)


def heat_kernel_analysis(
    eigenvalues: Tensor,
    *,
    num_time_points: int = 100,
    t_min: float = 0.001,
    t_max: float = 10.0,
) -> HeatKernelResult:
    """Compute the heat kernel trace over a logarithmically spaced time grid.

    Optionally fits the short-time asymptotic expansion:

        K(t) ~ (4*pi*t)^{-d/2} (a_0 + a_1*t + a_2*t^2 + ...)

    In 1-D this becomes K(t) ~ a_0/sqrt(t) + a_1*sqrt(t) + ...
    The first few Seeley-DeWitt coefficients are extracted by a
    least-squares fit to the short-time data.
    """
    eigenvalues = coerce_tensor(eigenvalues, dtype=torch.float64)
    log_t = torch.linspace(
        torch.log(torch.tensor(t_min)).item(),
        torch.log(torch.tensor(t_max)).item(),
        num_time_points,
        dtype=torch.float64,
    )
    times = torch.exp(log_t)
    trace = heat_kernel_trace(eigenvalues, times)

    # Fit short-time coefficients: K(t) ~ a0/sqrt(t) + a1*sqrt(t) + a2*t^{3/2}
    # Use the first third of time points for fitting
    n_fit = max(num_time_points // 3, 4)
    t_fit = times[:n_fit]
    K_fit = trace[:n_fit]

    # Design matrix: columns are t^{-1/2}, t^{1/2}, t^{3/2}
    num_coeffs = 3
    A = torch.zeros(n_fit, num_coeffs, dtype=torch.float64)
    A[:, 0] = t_fit ** (-0.5)
    A[:, 1] = t_fit ** 0.5
    A[:, 2] = t_fit ** 1.5

    # Least squares: A @ coeffs = K_fit
    try:
        result = torch.linalg.lstsq(A, K_fit.unsqueeze(-1))
        coeffs = result.solution.squeeze(-1)
    except Exception:
        coeffs = None

    return HeatKernelResult(
        times=times,
        trace=trace,
        short_time_coeffs=coeffs,
    )


# ---------------------------------------------------------------------------
# Partition function and thermodynamics
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PartitionFunctionResult:
    """Full thermodynamic analysis from the canonical partition function."""

    temperatures: Tensor
    Z: Tensor
    free_energy: Tensor
    internal_energy: Tensor
    entropy: Tensor
    specific_heat: Tensor


def partition_function(
    eigenvalues: Tensor,
    temperatures: Tensor,
    *,
    k_boltzmann: float = 1.0,
) -> PartitionFunctionResult:
    r"""Compute the canonical partition function and derived thermodynamic quantities.

    Z(beta) = sum_n exp(-beta * E_n)

    U = <E> = sum_n E_n exp(-beta E_n) / Z

    F = -k_B T ln Z

    S = (U - F) / T = k_B (ln Z + beta U)

    C_V = k_B beta^2 (<E^2> - <E>^2)

    Parameters
    ----------
    eigenvalues:
        Shape ``(N,)`` -- energy eigenvalues.
    temperatures:
        Shape ``(T,)`` -- temperature values (must be positive).
    k_boltzmann:
        Boltzmann constant (default 1.0 for natural units).
    """
    eigenvalues = coerce_tensor(eigenvalues, dtype=torch.float64)
    temperatures = coerce_tensor(temperatures, dtype=torch.float64)

    # beta = 1 / (k_B T)
    beta = 1.0 / (k_boltzmann * temperatures)  # (T,)

    # Boltzmann factors: exp(-beta_i * E_n)  -> (T, N)
    boltzmann = torch.exp(-beta.unsqueeze(-1) * eigenvalues.unsqueeze(0))

    Z = boltzmann.sum(dim=-1)  # (T,)

    # Internal energy: U = sum_n E_n exp(-beta E_n) / Z
    U = (boltzmann * eigenvalues.unsqueeze(0)).sum(dim=-1) / Z

    # Free energy: F = -k_B T ln Z
    F = -k_boltzmann * temperatures * torch.log(Z)

    # Entropy: S = (U - F) / T
    S = (U - F) / temperatures

    # Specific heat: C_V = k_B * beta^2 * (<E^2> - <E>^2)
    E_sq_mean = (boltzmann * eigenvalues.unsqueeze(0) ** 2).sum(dim=-1) / Z
    C_V = k_boltzmann * beta ** 2 * (E_sq_mean - U ** 2)

    return PartitionFunctionResult(
        temperatures=temperatures,
        Z=Z,
        free_energy=F,
        internal_energy=U,
        entropy=S,
        specific_heat=C_V,
    )


# ---------------------------------------------------------------------------
# Weyl's law
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class WeylLawResult:
    """Comparison of actual eigenvalue counting function with Weyl prediction."""

    eigenvalue_indices: Tensor
    eigenvalues: Tensor
    weyl_prediction: Tensor
    actual_count: Tensor
    relative_error: Tensor


def weyl_law_check(
    eigenvalues: Tensor,
    domain_length: float,
    *,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> WeylLawResult:
    r"""Check Weyl's asymptotic law for the eigenvalue counting function.

    In 1-D for a domain of length L:

        N(E) ~ (L / (pi * hbar)) sqrt(2 m E)

    where N(E) is the number of eigenvalues <= E.

    Parameters
    ----------
    eigenvalues:
        Sorted eigenvalues of the Hamiltonian.
    domain_length:
        Length L of the spatial domain.
    mass:
        Particle mass.
    hbar:
        Reduced Planck constant.
    """
    eigenvalues = coerce_tensor(eigenvalues, dtype=torch.float64)
    sorted_E, _ = torch.sort(eigenvalues)

    N = sorted_E.shape[0]
    indices = torch.arange(1, N + 1, dtype=torch.float64)

    # Actual count: N(E_n) = n (since eigenvalues are sorted)
    actual = indices

    # Weyl prediction at each eigenvalue
    weyl = (domain_length / (torch.pi * hbar)) * torch.sqrt(2.0 * mass * sorted_E)

    safe_actual = torch.where(actual > 0, actual, torch.ones_like(actual))
    rel_error = torch.abs(weyl - actual) / safe_actual

    return WeylLawResult(
        eigenvalue_indices=indices,
        eigenvalues=sorted_E,
        weyl_prediction=weyl,
        actual_count=actual,
        relative_error=rel_error,
    )


# ---------------------------------------------------------------------------
# Casimir energy
# ---------------------------------------------------------------------------

def casimir_energy(
    eigenvalues: Tensor,
    *,
    regularization: str = "zeta",
) -> Tensor:
    r"""Casimir energy via regularised summation of zero-point energies.

    E_Cas = (1/2) sum_n E_n  (regularised)

    Zeta function regularization:
        E_Cas = (1/2) zeta_H(-1)

    For the 1-D infinite well with E_n = n^2 pi^2 hbar^2 / (2mL^2):
        E_Cas = (1/2) * (pi^2 hbar^2 / (2mL^2)) * zeta_Riemann(-2)
    Since zeta_Riemann(-2) = 0, the Casimir energy vanishes.

    Exponential cutoff regularization:
        E_Cas = (1/2) lim_{eps->0} [sum_n E_n exp(-eps E_n) - divergent terms]
    Numerically we use a small eps and subtract the 1/eps^2 divergence.

    Parameters
    ----------
    eigenvalues:
        Positive eigenvalues.
    regularization:
        ``"zeta"`` -- evaluate zeta_H at s = -1.
        ``"exponential"`` -- exponential cutoff with divergence subtraction.
    """
    eigenvalues = coerce_tensor(eigenvalues, dtype=torch.float64)
    pos = eigenvalues[eigenvalues > 0]

    if regularization == "zeta":
        # zeta_H(-1) = sum_n E_n^{+1} = sum E_n ... but that diverges.
        # We use analytic continuation: compute zeta for several Re(s) > 1
        # values and extrapolate, or use the direct formula for known spectra.
        # For numerical work, use exponential regularization as a fallback.
        # Direct evaluation at s = -1 via the series:
        # E_n^{-(-1)} = E_n^1 = E_n
        # This diverges, so we use the heat-kernel Mellin approach
        # which provides the analytic continuation.
        s_val = torch.tensor(-1.0, dtype=torch.complex128)
        zeta_val = spectral_zeta(pos, s_val, regularization="heat_kernel")
        return 0.5 * zeta_val.real

    if regularization == "exponential":
        # Sum_n E_n exp(-eps * E_n) for decreasing eps,
        # then subtract the leading divergence ~ 1/eps^2
        eps_values = torch.tensor([0.1, 0.05, 0.02, 0.01, 0.005], dtype=torch.float64)
        sums = torch.zeros_like(eps_values)
        for i, eps in enumerate(eps_values):
            sums[i] = 0.5 * torch.sum(pos * torch.exp(-eps * pos))

        # Richardson extrapolation: assume S(eps) = E_cas + a/eps^2 + b/eps + ...
        # Fit E_cas from the last few points
        # Use pairs: S(eps1) - S(eps2) to cancel leading divergence
        # Simple approach: polynomial fit in eps^2 and extrapolate to eps=0
        x = eps_values ** 2
        # Linear fit: sums = a + b * x
        n_pts = eps_values.shape[0]
        x_mean = x.mean()
        s_mean = sums.mean()
        slope = ((x - x_mean) * (sums - s_mean)).sum() / ((x - x_mean) ** 2).sum()
        intercept = s_mean - slope * x_mean
        return intercept

    raise ValueError(f"unknown regularization method: {regularization!r}")


__all__ = [
    "HeatKernelResult",
    "PartitionFunctionResult",
    "SpectralZetaResult",
    "WeylLawResult",
    "casimir_energy",
    "heat_kernel_analysis",
    "heat_kernel_trace",
    "partition_function",
    "spectral_zeta",
    "weyl_law_check",
]
