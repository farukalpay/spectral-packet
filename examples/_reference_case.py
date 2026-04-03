from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spectral_packet_engine import (
    InfiniteWell1D,
    InfiniteWellBasis,
    PacketState,
    ProjectionConfig,
    SpectralPropagator,
    StateProjector,
    make_truncated_gaussian_packet,
)


REFERENCE_TIMES = (0.0, 1e-3, 3e-3, 5e-3, 1e-2)


@dataclass(frozen=True, slots=True)
class ReferenceProblem:
    domain: InfiniteWell1D
    basis: InfiniteWellBasis
    projector: StateProjector
    propagator: SpectralPropagator
    packet: PacketState


def build_reference_problem(
    *,
    num_modes: int = 200,
    quadrature_points: int = 8192,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> ReferenceProblem:
    domain = InfiniteWell1D.from_length(1.0, mass=1.0, hbar=1.0, dtype=dtype, device=device)
    basis = InfiniteWellBasis(domain, num_modes=num_modes)
    projector = StateProjector(basis, ProjectionConfig(quadrature_points=quadrature_points))
    propagator = SpectralPropagator(basis)
    packet = make_truncated_gaussian_packet(
        domain,
        center=0.30,
        width=0.07,
        wavenumber=25.0,
    )
    return ReferenceProblem(
        domain=domain,
        basis=basis,
        projector=projector,
        propagator=propagator,
        packet=packet,
    )


def reference_times(
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    return torch.tensor(REFERENCE_TIMES, dtype=dtype, device=device)
