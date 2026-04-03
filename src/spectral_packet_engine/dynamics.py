from __future__ import annotations

from dataclasses import dataclass

import torch

from spectral_packet_engine.basis import InfiniteWellBasis
from spectral_packet_engine.domain import coerce_tensor
from spectral_packet_engine.state import SpectralState

Tensor = torch.Tensor


def phase_factors(energies, times, *, hbar) -> Tensor:
    spectrum = coerce_tensor(energies)
    evaluation_times = coerce_tensor(times, dtype=spectrum.dtype, device=spectrum.device)
    if evaluation_times.ndim == 0:
        evaluation_times = evaluation_times.reshape(1)
    planck = coerce_tensor(hbar, dtype=spectrum.dtype, device=spectrum.device)
    return torch.exp((-1j / planck) * evaluation_times[:, None] * spectrum[None, :])


@dataclass(frozen=True, slots=True)
class SpectralPropagator:
    basis: InfiniteWellBasis

    def phase_factors(self, times) -> Tensor:
        return phase_factors(self.basis.energies, times, hbar=self.basis.domain.hbar)

    def propagate_many(self, state: SpectralState, times) -> Tensor:
        phases = self.phase_factors(times).to(dtype=state.coefficients.dtype, device=state.coefficients.device)
        return phases * state.coefficients[None, :]

    def propagate(self, state: SpectralState, time) -> SpectralState:
        propagated = self.propagate_many(state, coerce_tensor(time, dtype=self.basis.domain.real_dtype, device=self.basis.domain.device))
        return SpectralState(domain=state.domain, coefficients=propagated.reshape(-1))


__all__ = [
    "SpectralPropagator",
    "phase_factors",
]
