from __future__ import annotations

from dataclasses import dataclass

import torch

from spectral_packet_engine.dynamics import SpectralPropagator
from spectral_packet_engine.observables import (
    expectation_position as expectation_position_fn,
    interval_probability as interval_probability_fn,
    total_probability as total_probability_fn,
    variance_position as variance_position_fn,
)
from spectral_packet_engine.projector import StateProjector
from spectral_packet_engine.state import PacketState, SpectralState

Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class SimulationRecord:
    times: Tensor
    coefficients: Tensor
    grid: Tensor | None = None
    wavefunctions: Tensor | None = None
    densities: Tensor | None = None

    def total_probability(self) -> Tensor:
        if self.wavefunctions is None or self.grid is None:
            raise ValueError("simulation record does not contain reconstructed wavefunctions")
        return total_probability_fn(self.wavefunctions, self.grid)

    def expectation_position(self) -> Tensor:
        if self.wavefunctions is None or self.grid is None:
            raise ValueError("simulation record does not contain reconstructed wavefunctions")
        return expectation_position_fn(self.wavefunctions, self.grid)

    def variance_position(self) -> Tensor:
        if self.wavefunctions is None or self.grid is None:
            raise ValueError("simulation record does not contain reconstructed wavefunctions")
        return variance_position_fn(self.wavefunctions, self.grid)

    def interval_probability(self, left, right) -> Tensor:
        if self.wavefunctions is None or self.grid is None:
            raise ValueError("simulation record does not contain reconstructed wavefunctions")
        return interval_probability_fn(self.wavefunctions, self.grid, left, right)


def simulate(
    initial_state: PacketState | SpectralState,
    times,
    *,
    projector: StateProjector,
    propagator: SpectralPropagator,
    grid=None,
) -> SimulationRecord:
    evaluation_times = torch.atleast_1d(
        torch.as_tensor(
            times,
            dtype=projector.domain.real_dtype,
            device=projector.domain.device,
        )
    )
    if isinstance(initial_state, PacketState):
        spectral_state = projector.project_packet(initial_state)
    elif isinstance(initial_state, SpectralState):
        spectral_state = initial_state
    else:
        raise TypeError("initial_state must be a PacketState or SpectralState")

    propagated_coefficients = propagator.propagate_many(spectral_state, evaluation_times)
    reconstructed_grid = None
    wavefunctions = None
    densities = None
    if grid is not None:
        reconstructed_grid = torch.as_tensor(
            grid,
            dtype=projector.domain.real_dtype,
            device=projector.domain.device,
        )
        wavefunctions = projector.reconstruct_coefficients(propagated_coefficients, reconstructed_grid)
        densities = torch.abs(wavefunctions) ** 2

    return SimulationRecord(
        times=evaluation_times,
        coefficients=propagated_coefficients,
        grid=reconstructed_grid,
        wavefunctions=wavefunctions,
        densities=densities,
    )


__all__ = [
    "SimulationRecord",
    "simulate",
]
