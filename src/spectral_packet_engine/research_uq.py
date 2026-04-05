from __future__ import annotations

"""Optional MCMC posterior inference for research workflows (Layer 3).

Requires the ``research`` extra::

    pip install spectral-packet-engine[research]

This module wraps NumPyro NUTS sampling around the same log-likelihood
functions used by the local Laplace approximation in ``uq.py``.  The
torch-to-JAX bridge is handled transparently.
"""

import importlib.util
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from spectral_packet_engine.uq import _central_normal_quantile


# ---------------------------------------------------------------------------
# Availability checks (follows tf_surrogate.py pattern)
# ---------------------------------------------------------------------------


def numpyro_is_available() -> bool:
    return importlib.util.find_spec("numpyro") is not None


def arviz_is_available() -> bool:
    return importlib.util.find_spec("arviz") is not None


def _require_numpyro():
    try:
        import numpyro
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "NumPyro is not installed.  Install the 'research' extra: "
            "pip install spectral-packet-engine[research]"
        ) from exc
    return numpyro


def _require_jax():
    try:
        import jax
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "JAX is not installed.  Install the 'research' extra: "
            "pip install spectral-packet-engine[research]"
        ) from exc
    return jax


# ---------------------------------------------------------------------------
# MCMC posterior summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MCMCPosteriorSummary:
    """Full posterior from MCMC sampling."""

    parameter_names: tuple[str, ...]
    chains: torch.Tensor
    mean: torch.Tensor
    standard_deviation: torch.Tensor
    confidence_level: float
    confidence_interval_low: torch.Tensor
    confidence_interval_high: torch.Tensor
    r_hat: torch.Tensor
    effective_sample_size: torch.Tensor
    num_chains: int
    num_samples: int
    num_warmup: int
    divergences: int
    sampler: str
    noise_scale: float
    assumptions: tuple[str, ...]


# ---------------------------------------------------------------------------
# Diagnostics (minimal R-hat and ESS in pure torch)
# ---------------------------------------------------------------------------


def _split_r_hat(chains: torch.Tensor) -> torch.Tensor:
    """Compute split-R-hat for convergence diagnostics.

    ``chains`` has shape (num_chains, num_samples, num_params).
    """
    num_chains, num_samples, num_params = chains.shape
    half = num_samples // 2
    split = torch.cat([chains[:, :half], chains[:, half:half * 2]], dim=0)
    m = split.shape[0]
    n = split.shape[1]
    chain_means = split.mean(dim=1)
    grand_mean = chain_means.mean(dim=0)
    b = n / (m - 1) * torch.sum((chain_means - grand_mean) ** 2, dim=0)
    chain_vars = split.var(dim=1)
    w = chain_vars.mean(dim=0)
    var_hat = (n - 1) / n * w + b / n
    return torch.sqrt(var_hat / torch.clamp(w, min=1e-30))


def _bulk_ess(chains: torch.Tensor) -> torch.Tensor:
    """Approximate effective sample size per parameter."""
    num_chains, num_samples, num_params = chains.shape
    r_hat = _split_r_hat(chains)
    total_samples = num_chains * num_samples
    return torch.clamp(
        torch.tensor(total_samples, dtype=chains.dtype) / (r_hat ** 2),
        max=float(total_samples),
    ).expand(num_params)


# ---------------------------------------------------------------------------
# NUTS posterior
# ---------------------------------------------------------------------------


def run_nuts_posterior(
    *,
    log_likelihood_fn: Callable[[torch.Tensor], float],
    parameter_names: tuple[str, ...],
    initial_position: torch.Tensor,
    parameter_bounds: Sequence[tuple[float | None, float | None]],
    noise_scale: float,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 2,
    confidence_level: float = 0.95,
    rng_seed: int = 0,
) -> MCMCPosteriorSummary:
    """Run NumPyro NUTS on the given log-likelihood.

    The ``log_likelihood_fn`` accepts a 1-D torch parameter vector and returns
    a scalar log-likelihood value.  It is wrapped internally to bridge torch
    and JAX.
    """
    numpyro = _require_numpyro()
    jax = _require_jax()
    import jax.numpy as jnp
    import numpy as np

    num_params = len(parameter_names)
    init_np = initial_position.detach().cpu().numpy()

    # Build the torch→JAX bridge
    def jax_log_likelihood(params_jax):
        params_np = np.array(params_jax, dtype=np.float64)
        params_torch = torch.tensor(params_np, dtype=torch.float64)
        ll = log_likelihood_fn(params_torch)
        return jnp.array(float(ll))

    # NumPyro model
    def model():
        params = []
        for i, (name, (lb, ub)) in enumerate(zip(parameter_names, parameter_bounds)):
            if lb is not None and ub is not None:
                p = numpyro.sample(name, numpyro.distributions.Uniform(lb, ub))
            elif lb is not None:
                p = numpyro.sample(name, numpyro.distributions.HalfNormal(10.0)) + lb
            else:
                p = numpyro.sample(name, numpyro.distributions.Normal(float(init_np[i]), 10.0))
            params.append(p)
        param_vector = jnp.stack(params)
        ll = jax_log_likelihood(param_vector)
        numpyro.factor("log_likelihood", ll)

    kernel = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False,
    )
    rng_key = jax.random.PRNGKey(rng_seed)
    mcmc.run(rng_key)

    # Extract samples
    samples_dict = mcmc.get_samples(group_by_chain=True)
    chains_list = [np.array(samples_dict[name]) for name in parameter_names]
    chains_np = np.stack(chains_list, axis=-1)  # (num_chains, num_samples, num_params)
    chains_tensor = torch.tensor(chains_np, dtype=torch.float64)

    # Diagnostics
    r_hat = _split_r_hat(chains_tensor)
    ess = _bulk_ess(chains_tensor)

    flat_samples = chains_tensor.reshape(-1, num_params)
    mean = flat_samples.mean(dim=0)
    std = flat_samples.std(dim=0)

    z = _central_normal_quantile(confidence_level, dtype=torch.float64, device=None)
    alpha = (1 - confidence_level) / 2
    ci_low = torch.quantile(flat_samples, alpha, dim=0)
    ci_high = torch.quantile(flat_samples, 1 - alpha, dim=0)

    # Count divergences
    try:
        extra_fields = mcmc.get_extra_fields()
        divergences = int(np.sum(np.array(extra_fields.get("diverging", 0))))
    except Exception:
        divergences = 0

    return MCMCPosteriorSummary(
        parameter_names=parameter_names,
        chains=chains_tensor,
        mean=mean,
        standard_deviation=std,
        confidence_level=confidence_level,
        confidence_interval_low=ci_low,
        confidence_interval_high=ci_high,
        r_hat=r_hat,
        effective_sample_size=ess,
        num_chains=num_chains,
        num_samples=num_samples,
        num_warmup=num_warmup,
        divergences=divergences,
        sampler="NUTS",
        noise_scale=noise_scale,
        assumptions=(
            "Posterior is sampled via NumPyro NUTS (No U-Turn Sampler) with the same log-likelihood used by the local Laplace approximation.",
            "Priors are uniform within parameter bounds, or half-normal/normal for unbounded parameters.",
            "Convergence should be assessed via R-hat (target < 1.05) and effective sample size.",
        ),
    )


__all__ = [
    "MCMCPosteriorSummary",
    "arviz_is_available",
    "numpyro_is_available",
    "run_nuts_posterior",
]
