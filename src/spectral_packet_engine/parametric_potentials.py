from __future__ import annotations

"""Parameterized potential families for inverse and design workflows.

The goal of this module is not to be a generic potential zoo. It provides a
small set of explicit, differentiable, bounded-domain model families that the
shared spectral workflows can calibrate, compare, and optimize.
"""

from dataclasses import dataclass
from typing import Callable, Mapping

import torch

from spectral_packet_engine.domain import InfiniteWell1D
from spectral_packet_engine.eigensolver import (
    double_well_potential,
    harmonic_potential,
    morse_potential,
)

Tensor = torch.Tensor


def gaussian_barrier_potential(
    x: Tensor,
    *,
    height,
    width,
    center,
) -> Tensor:
    displacement = x - center
    return height * torch.exp(-(displacement**2) / (2 * width**2))


@dataclass(frozen=True, slots=True)
class PotentialParameterSpec:
    name: str
    description: str
    lower_bound: float | None = None
    upper_bound: float | None = None
    bind_to_domain: bool = False

    def resolved_bounds(self, domain: InfiniteWell1D | None = None) -> tuple[Tensor | None, Tensor | None]:
        lower = None if self.lower_bound is None else torch.as_tensor(self.lower_bound)
        upper = None if self.upper_bound is None else torch.as_tensor(self.upper_bound)
        if self.bind_to_domain:
            if domain is None:
                raise ValueError(f"{self.name} requires a domain to resolve its bounds")
            lower = domain.left.detach().cpu()
            upper = domain.right.detach().cpu()
        return lower, upper

    def validate_tensor(self, value: Tensor, *, domain: InfiniteWell1D | None = None) -> Tensor:
        lower_bound, upper_bound = self.resolved_bounds(domain)
        if lower_bound is not None and bool(torch.any(value < lower_bound.to(device=value.device, dtype=value.dtype)).item()):
            raise ValueError(f"{self.name} must be >= {float(lower_bound)}")
        if upper_bound is not None and bool(torch.any(value > upper_bound.to(device=value.device, dtype=value.dtype)).item()):
            raise ValueError(f"{self.name} must be <= {float(upper_bound)}")
        return value


PotentialBuilder = Callable[[InfiniteWell1D, Tensor], Callable[[Tensor], Tensor]]


@dataclass(frozen=True, slots=True)
class PotentialFamilyDefinition:
    name: str
    description: str
    parameter_specs: tuple[PotentialParameterSpec, ...]
    build_from_vector: PotentialBuilder

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.parameter_specs)

    def vector_from_mapping(
        self,
        parameters: Mapping[str, float | Tensor],
        *,
        domain: InfiniteWell1D | None = None,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> Tensor:
        values: list[Tensor] = []
        for spec in self.parameter_specs:
            if spec.name not in parameters:
                raise KeyError(f"missing parameter '{spec.name}' for potential family '{self.name}'")
            tensor = torch.as_tensor(parameters[spec.name], dtype=dtype, device=device).reshape(())
            values.append(spec.validate_tensor(tensor, domain=domain))
        return torch.stack(values)

    def mapping_from_vector(
        self,
        vector: Tensor,
        *,
        domain: InfiniteWell1D | None = None,
    ) -> dict[str, Tensor]:
        tensor = torch.as_tensor(vector)
        if tensor.ndim != 1 or tensor.shape[0] != len(self.parameter_specs):
            raise ValueError(
                f"expected a parameter vector of shape ({len(self.parameter_specs)},) for '{self.name}'"
            )
        mapping = {
            spec.name: spec.validate_tensor(tensor[index].reshape(()), domain=domain)
            for index, spec in enumerate(self.parameter_specs)
        }
        return mapping

    def build_from_mapping(
        self,
        domain: InfiniteWell1D,
        parameters: Mapping[str, float | Tensor],
    ) -> Callable[[Tensor], Tensor]:
        vector = self.vector_from_mapping(
            parameters,
            domain=domain,
            dtype=domain.real_dtype,
            device=domain.device,
        )
        return self.build_from_vector(domain, vector)


def _harmonic_builder(domain: InfiniteWell1D, vector: Tensor) -> Callable[[Tensor], Tensor]:
    omega = vector[0]
    return lambda x: harmonic_potential(x, omega=omega, domain=domain)


def _double_well_builder(domain: InfiniteWell1D, vector: Tensor) -> Callable[[Tensor], Tensor]:
    a_param, b_param = vector
    return lambda x: double_well_potential(x, a_param=a_param, b_param=b_param, domain=domain)


def _morse_builder(domain: InfiniteWell1D, vector: Tensor) -> Callable[[Tensor], Tensor]:
    D_e, alpha, x_eq = vector
    return lambda x: morse_potential(x, D_e=D_e, alpha=alpha, x_eq=x_eq)


def _gaussian_barrier_builder(domain: InfiniteWell1D, vector: Tensor) -> Callable[[Tensor], Tensor]:
    height, width, center = vector
    return lambda x: gaussian_barrier_potential(x, height=height, width=width, center=center)


_POTENTIAL_FAMILIES: tuple[PotentialFamilyDefinition, ...] = (
    PotentialFamilyDefinition(
        name="harmonic",
        description="Bounded harmonic well centered at the domain midpoint.",
        parameter_specs=(
            PotentialParameterSpec("omega", "Oscillator frequency.", lower_bound=1e-8),
        ),
        build_from_vector=_harmonic_builder,
    ),
    PotentialFamilyDefinition(
        name="double-well",
        description="Symmetric quartic double-well centered at the domain midpoint.",
        parameter_specs=(
            PotentialParameterSpec("a_param", "Quartic confinement strength.", lower_bound=1e-8),
            PotentialParameterSpec("b_param", "Quadratic barrier strength.", lower_bound=1e-8),
        ),
        build_from_vector=_double_well_builder,
    ),
    PotentialFamilyDefinition(
        name="morse",
        description="Morse oscillator on a bounded interval.",
        parameter_specs=(
            PotentialParameterSpec("D_e", "Well depth.", lower_bound=1e-8),
            PotentialParameterSpec("alpha", "Inverse width parameter.", lower_bound=1e-8),
            PotentialParameterSpec("x_eq", "Equilibrium position.", bind_to_domain=True),
        ),
        build_from_vector=_morse_builder,
    ),
    PotentialFamilyDefinition(
        name="gaussian-barrier",
        description="Smooth Gaussian barrier suitable for transport and design workflows.",
        parameter_specs=(
            PotentialParameterSpec("height", "Barrier height."),
            PotentialParameterSpec("width", "Barrier width.", lower_bound=1e-8),
            PotentialParameterSpec("center", "Barrier center.", bind_to_domain=True),
        ),
        build_from_vector=_gaussian_barrier_builder,
    ),
)


def available_potential_families() -> tuple[str, ...]:
    return tuple(family.name for family in _POTENTIAL_FAMILIES)


def describe_potential_families() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "name": family.name,
            "description": family.description,
            "parameters": [
                {
                    "name": spec.name,
                    "description": spec.description,
                    "lower_bound": spec.lower_bound,
                    "upper_bound": spec.upper_bound,
                }
                for spec in family.parameter_specs
            ],
        }
        for family in _POTENTIAL_FAMILIES
    )


def default_parameter_mapping(
    family: str | PotentialFamilyDefinition,
    *,
    domain: InfiniteWell1D,
) -> dict[str, float]:
    """Return a minimal, inspectable initialization policy for a family.

    This is intentionally not a physical prior. It exists so CLI and MCP
    surfaces can expose the differentiable family workflows without forcing
    every caller to construct seed values manually.
    """
    resolved = family if isinstance(family, PotentialFamilyDefinition) else resolve_potential_family(family)
    mapping: dict[str, float] = {}
    domain_midpoint = float(((domain.left + domain.right) / 2).detach().cpu().item())
    for spec in resolved.parameter_specs:
        lower_bound, upper_bound = spec.resolved_bounds(domain)
        if lower_bound is not None and upper_bound is not None:
            lower = float(lower_bound)
            upper = float(upper_bound)
            mapping[spec.name] = 0.5 * (lower + upper)
        elif lower_bound is not None:
            lower = float(lower_bound)
            mapping[spec.name] = max(1.0, lower * 2.0)
        elif upper_bound is not None:
            upper = float(upper_bound)
            mapping[spec.name] = min(-1.0, upper - 1.0)
        elif spec.bind_to_domain:
            mapping[spec.name] = domain_midpoint
        else:
            mapping[spec.name] = 1.0
    return mapping


def resolve_potential_family(name: str) -> PotentialFamilyDefinition:
    normalized = str(name).strip().lower()
    for family in _POTENTIAL_FAMILIES:
        if family.name == normalized:
            return family
    supported = ", ".join(available_potential_families())
    raise ValueError(f"unknown potential family '{name}'. Supported families: {supported}")


__all__ = [
    "PotentialFamilyDefinition",
    "PotentialParameterSpec",
    "available_potential_families",
    "default_parameter_mapping",
    "describe_potential_families",
    "gaussian_barrier_potential",
    "resolve_potential_family",
]
