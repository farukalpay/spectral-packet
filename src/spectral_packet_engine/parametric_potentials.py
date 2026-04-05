from __future__ import annotations

"""Parameterized potential families for inverse and design workflows.

The goal of this module is not to be a generic potential zoo. It provides a
small set of explicit, differentiable, bounded-domain model families that the
shared spectral workflows can calibrate, compare, and optimize.

Each family is a complete scientific object: it declares its potential builder,
parameter specifications and priors, boundary condition, observable map,
differentiability guarantees, supported workflows, analytical properties, and
academic citations.  Adding a new family means populating all of these fields,
not just writing a function.
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


# ---------------------------------------------------------------------------
# Supporting contract dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParameterPrior:
    """Structured prior distribution for a single potential parameter.

    Used by ``default_parameter_mapping`` when available and exposed in
    ``describe_potential_families`` for scientific documentation.
    """

    distribution: str  # "normal", "log_normal", "uniform", "half_normal"
    center: float
    scale: float
    rationale: str = ""


@dataclass(frozen=True, slots=True)
class DifferentiabilityInfo:
    """Declares whether gradient-based optimization is valid for a family."""

    supports_gradient: bool = True
    caveats: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ArtifactField:
    """Describes one extra artifact a family produces beyond the standard set."""

    key: str
    description: str
    dtype: str = "float"  # "float", "tensor", "string"


@dataclass(frozen=True, slots=True)
class Citation:
    """Academic reference for a potential family."""

    label: str  # short key, e.g. "morse1929"
    text: str   # full reference string


# ---------------------------------------------------------------------------
# Parameter specification (unchanged)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Family definition — the unified physics model contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PotentialFamilyDefinition:
    """Complete scientific description of a parametric potential family.

    Every field beyond the builder is metadata that the library's inference,
    design, and reporting workflows consume automatically.  New families must
    populate all fields — the defaults exist only for backward compatibility
    during the transition period.
    """

    # ── identity ──
    name: str
    description: str

    # ── physics ──
    parameter_specs: tuple[PotentialParameterSpec, ...]
    build_from_vector: PotentialBuilder

    # ── scientific contract (new) ──
    boundary_condition: str = "dirichlet"
    default_priors: tuple[ParameterPrior, ...] | None = None
    observable_map: tuple[str, ...] = ("eigenvalues", "transition_energies")
    differentiability_info: DifferentiabilityInfo = DifferentiabilityInfo()
    artifact_schema: tuple[ArtifactField, ...] = ()
    supported_workflows: frozenset[str] = frozenset({"spectroscopy", "calibration", "design"})
    citations: tuple[Citation, ...] = ()
    analytical_properties: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if self.default_priors is not None and len(self.default_priors) != len(self.parameter_specs):
            raise ValueError(
                f"default_priors length ({len(self.default_priors)}) must match "
                f"parameter_specs length ({len(self.parameter_specs)})"
            )

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


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Family registry — each entry is a complete scientific object
# ---------------------------------------------------------------------------


_POTENTIAL_FAMILIES: tuple[PotentialFamilyDefinition, ...] = (
    PotentialFamilyDefinition(
        name="harmonic",
        description="Bounded harmonic well centered at the domain midpoint.",
        parameter_specs=(
            PotentialParameterSpec("omega", "Oscillator frequency.", lower_bound=1e-8),
        ),
        build_from_vector=_harmonic_builder,
        boundary_condition="dirichlet",
        default_priors=(
            ParameterPrior("log_normal", center=5.0, scale=3.0,
                           rationale="Typical bounded-domain frequency scale."),
        ),
        observable_map=("eigenvalues", "transition_energies"),
        differentiability_info=DifferentiabilityInfo(supports_gradient=True),
        supported_workflows=frozenset({"spectroscopy", "calibration", "design"}),
        citations=(
            Citation("griffiths2018",
                     "Griffiths & Schroeter, Introduction to Quantum Mechanics, 3rd ed., Cambridge (2018)."),
        ),
        analytical_properties={
            "spectrum_type": "discrete",
            "symmetry": "even",
            "bound_states": "infinite_in_principle",
            "classical_limit": "equally_spaced",
        },
    ),
    PotentialFamilyDefinition(
        name="double-well",
        description="Symmetric quartic double-well centered at the domain midpoint.",
        parameter_specs=(
            PotentialParameterSpec("a_param", "Quartic confinement strength.", lower_bound=1e-8),
            PotentialParameterSpec("b_param", "Quadratic barrier strength.", lower_bound=1e-8),
        ),
        build_from_vector=_double_well_builder,
        boundary_condition="dirichlet",
        default_priors=(
            ParameterPrior("log_normal", center=2.0, scale=2.0,
                           rationale="Quartic coefficient for bounded domain."),
            ParameterPrior("log_normal", center=1.5, scale=1.5,
                           rationale="Barrier strength relative to domain width."),
        ),
        observable_map=("eigenvalues", "transition_energies", "tunnel_splitting"),
        differentiability_info=DifferentiabilityInfo(
            supports_gradient=True,
            caveats=("Near-degenerate tunnel-split doublets may cause gradient instability.",),
        ),
        supported_workflows=frozenset({"spectroscopy", "calibration", "design"}),
        citations=(
            Citation("razavy2003",
                     "Razavy, Quantum Theory of Tunneling, World Scientific (2003)."),
        ),
        analytical_properties={
            "spectrum_type": "discrete",
            "symmetry": "even",
            "bound_states": "finite",
            "tunnel_splitting": True,
        },
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
        boundary_condition="dirichlet",
        default_priors=(
            ParameterPrior("log_normal", center=10.0, scale=5.0,
                           rationale="Typical molecular well depth."),
            ParameterPrior("log_normal", center=3.0, scale=2.0,
                           rationale="Inverse width for molecular-scale potentials."),
            ParameterPrior("normal", center=0.5, scale=0.2,
                           rationale="Equilibrium near domain center."),
        ),
        observable_map=("eigenvalues", "transition_energies", "anharmonicity"),
        differentiability_info=DifferentiabilityInfo(
            supports_gradient=True,
            caveats=("Exponential growth near domain edges can produce large gradients.",),
        ),
        supported_workflows=frozenset({"spectroscopy", "calibration", "design"}),
        citations=(
            Citation("morse1929", "Morse, Phys. Rev. 34, 57 (1929)."),
        ),
        analytical_properties={
            "spectrum_type": "discrete",
            "symmetry": "asymmetric",
            "bound_states": "finite",
            "anharmonic": True,
        },
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
        boundary_condition="dirichlet",
        default_priors=(
            ParameterPrior("normal", center=50.0, scale=20.0,
                           rationale="Typical barrier height for tunneling experiments."),
            ParameterPrior("log_normal", center=0.03, scale=0.02,
                           rationale="Narrow barrier width relative to domain."),
            ParameterPrior("normal", center=0.5, scale=0.1,
                           rationale="Barrier center near domain midpoint."),
        ),
        observable_map=("eigenvalues", "transition_energies", "transmission_coefficient", "tunneling_probability"),
        differentiability_info=DifferentiabilityInfo(supports_gradient=True),
        supported_workflows=frozenset({"spectroscopy", "calibration", "design", "transport"}),
        analytical_properties={
            "spectrum_type": "continuous_above_barrier",
            "symmetry": "even",
            "bound_states": "quasi_bound",
            "scattering": True,
        },
    ),
)


# ---------------------------------------------------------------------------
# Public query API
# ---------------------------------------------------------------------------


def available_potential_families() -> tuple[str, ...]:
    return tuple(family.name for family in _POTENTIAL_FAMILIES)


def families_for_workflow(workflow: str) -> tuple[str, ...]:
    """Return family names that declare support for the given workflow."""
    return tuple(
        family.name for family in _POTENTIAL_FAMILIES
        if workflow in family.supported_workflows
    )


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
            "boundary_condition": family.boundary_condition,
            "default_priors": [
                {
                    "parameter": family.parameter_specs[i].name,
                    "distribution": p.distribution,
                    "center": p.center,
                    "scale": p.scale,
                    "rationale": p.rationale,
                }
                for i, p in enumerate(family.default_priors)
            ] if family.default_priors is not None else None,
            "observable_map": list(family.observable_map),
            "differentiability": {
                "supports_gradient": family.differentiability_info.supports_gradient,
                "caveats": list(family.differentiability_info.caveats),
            },
            "artifact_schema": [
                {"key": a.key, "description": a.description, "dtype": a.dtype}
                for a in family.artifact_schema
            ],
            "supported_workflows": sorted(family.supported_workflows),
            "citations": [
                {"label": c.label, "text": c.text}
                for c in family.citations
            ],
            "analytical_properties": family.analytical_properties,
        }
        for family in _POTENTIAL_FAMILIES
    )


def default_parameter_mapping(
    family: str | PotentialFamilyDefinition,
    *,
    domain: InfiniteWell1D,
) -> dict[str, float]:
    """Return a minimal, inspectable initialization policy for a family.

    When the family declares ``default_priors``, the prior center is used
    directly.  Otherwise falls back to a heuristic based on parameter bounds.
    """
    resolved = family if isinstance(family, PotentialFamilyDefinition) else resolve_potential_family(family)
    mapping: dict[str, float] = {}
    domain_midpoint = float(((domain.left + domain.right) / 2).detach().cpu().item())
    for index, spec in enumerate(resolved.parameter_specs):
        # Prefer prior center when available
        if resolved.default_priors is not None:
            mapping[spec.name] = resolved.default_priors[index].center
            continue
        # Heuristic fallback
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
    "ArtifactField",
    "Citation",
    "DifferentiabilityInfo",
    "ParameterPrior",
    "PotentialFamilyDefinition",
    "PotentialParameterSpec",
    "available_potential_families",
    "default_parameter_mapping",
    "describe_potential_families",
    "families_for_workflow",
    "gaussian_barrier_potential",
    "resolve_potential_family",
]
