from __future__ import annotations

"""First-class physics contracts shared by workflows and interfaces.

These objects give forward simulation, inverse fitting, reduced models, and
surrogate training a common mathematical vocabulary without moving research
logic into CLI/MCP/API wrappers.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from spectral_packet_engine.basis import InfiniteWellBasis
from spectral_packet_engine.domain import InfiniteWell1D
from spectral_packet_engine.eigensolver import EigensolverResult, solve_eigenproblem
from spectral_packet_engine.parametric_potentials import (
    PotentialFamilyDefinition,
    resolve_potential_family,
)

Tensor = torch.Tensor

_SINE_BASIS_FAMILY = "infinite-well-sine"
_DIRICHLET_KIND = "dirichlet"


@dataclass(frozen=True, slots=True)
class BoundaryCondition:
    """Boundary condition for a bounded-domain spectral problem."""

    kind: str = _DIRICHLET_KIND
    left_value: float = 0.0
    right_value: float = 0.0

    def __post_init__(self) -> None:
        kind = self.kind.strip().lower()
        object.__setattr__(self, "kind", kind)
        if kind != _DIRICHLET_KIND:
            raise ValueError("only Dirichlet boundary conditions are currently implemented")
        if self.left_value != 0.0 or self.right_value != 0.0:
            raise ValueError("the current sine-basis solver requires homogeneous Dirichlet boundaries")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "left_value": self.left_value,
            "right_value": self.right_value,
        }

    @classmethod
    def from_family_definition(cls, family: PotentialFamilyDefinition) -> "BoundaryCondition":
        return cls(kind=family.boundary_condition)


@dataclass(frozen=True, slots=True)
class BasisSpec:
    """Basis contract for spectral workflows."""

    family: str = _SINE_BASIS_FAMILY
    num_modes: int = 64
    quadrature_points: int | None = None

    def __post_init__(self) -> None:
        family = self.family.strip().lower()
        object.__setattr__(self, "family", family)
        if family != _SINE_BASIS_FAMILY:
            raise ValueError(f"unsupported basis family {self.family!r}; supported: {_SINE_BASIS_FAMILY}")
        if self.num_modes <= 0:
            raise ValueError("num_modes must be positive")
        if self.quadrature_points is not None and self.quadrature_points < 2:
            raise ValueError("quadrature_points must be at least 2 when provided")

    def build(self, domain: InfiniteWell1D) -> InfiniteWellBasis:
        return InfiniteWellBasis(domain, self.num_modes)

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "num_modes": self.num_modes,
            "quadrature_points": self.quadrature_points,
        }


@dataclass(frozen=True, slots=True)
class PotentialFamily:
    """First-class potential family contract over the existing registry."""

    definition: PotentialFamilyDefinition

    @classmethod
    def from_name(cls, name: str) -> "PotentialFamily":
        return cls(resolve_potential_family(name))

    @property
    def name(self) -> str:
        return self.definition.name

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self.definition.parameter_names

    @property
    def supported_workflows(self) -> frozenset[str]:
        return self.definition.supported_workflows

    @property
    def observable_names(self) -> tuple[str, ...]:
        return self.definition.observable_map

    def default_parameters(self, domain: InfiniteWell1D) -> dict[str, float]:
        from spectral_packet_engine.parametric_potentials import default_parameter_mapping

        return default_parameter_mapping(self.definition, domain=domain)

    def build_potential(
        self,
        domain: InfiniteWell1D,
        parameters: Mapping[str, float | Tensor],
    ):
        return self.definition.build_from_mapping(domain, parameters)

    def vector_from_mapping(
        self,
        domain: InfiniteWell1D,
        parameters: Mapping[str, float | Tensor],
    ) -> Tensor:
        return self.definition.vector_from_mapping(
            parameters,
            domain=domain,
            dtype=domain.real_dtype,
            device=domain.device,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.definition.description,
            "parameters": list(self.parameter_names),
            "boundary_condition": self.definition.boundary_condition,
            "observables": list(self.observable_names),
            "supported_workflows": sorted(self.supported_workflows),
        }


@dataclass(frozen=True, slots=True)
class HamiltonianOperator:
    """Hamiltonian contract H = T + V over a domain, basis, and potential family."""

    domain: InfiniteWell1D
    potential_family: PotentialFamily
    parameters: Mapping[str, float | Tensor]
    basis_spec: BasisSpec
    boundary_condition: BoundaryCondition

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", dict(self.parameters))
        self.potential_family.vector_from_mapping(self.domain, self.parameters)
        if self.boundary_condition.kind != self.potential_family.definition.boundary_condition:
            raise ValueError(
                "Hamiltonian boundary condition must match the potential-family contract "
                f"({self.potential_family.definition.boundary_condition!r})"
            )

    @classmethod
    def from_family(
        cls,
        family: str | PotentialFamily | PotentialFamilyDefinition,
        *,
        domain: InfiniteWell1D,
        parameters: Mapping[str, float | Tensor] | None = None,
        basis_spec: BasisSpec | None = None,
        boundary_condition: BoundaryCondition | None = None,
    ) -> "HamiltonianOperator":
        if isinstance(family, PotentialFamily):
            potential_family = family
        elif isinstance(family, PotentialFamilyDefinition):
            potential_family = PotentialFamily(family)
        else:
            potential_family = PotentialFamily.from_name(family)
        resolved_parameters = (
            potential_family.default_parameters(domain)
            if parameters is None
            else dict(parameters)
        )
        return cls(
            domain=domain,
            potential_family=potential_family,
            parameters=resolved_parameters,
            basis_spec=basis_spec or BasisSpec(),
            boundary_condition=boundary_condition or BoundaryCondition.from_family_definition(
                potential_family.definition
            ),
        )

    def potential_fn(self):
        return self.potential_family.build_potential(self.domain, self.parameters)

    def solve(
        self,
        *,
        num_states: int = 10,
        num_points: int | None = None,
        num_quad: int | None = None,
    ) -> EigensolverResult:
        return solve_eigenproblem(
            self.potential_fn(),
            self.domain,
            num_points=num_points or self.basis_spec.num_modes,
            num_states=num_states,
            num_quad=num_quad if num_quad is not None else self.basis_spec.quadrature_points,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "domain": {
                "left": float(self.domain.left.detach().cpu().item()),
                "right": float(self.domain.right.detach().cpu().item()),
                "mass": float(self.domain.mass.detach().cpu().item()),
                "hbar": float(self.domain.hbar.detach().cpu().item()),
                "dtype": str(self.domain.real_dtype).replace("torch.", ""),
                "device": str(self.domain.device),
            },
            "potential_family": self.potential_family.to_dict(),
            "parameters": {
                key: float(torch.as_tensor(value).detach().cpu().item())
                for key, value in self.parameters.items()
            },
            "basis": self.basis_spec.to_dict(),
            "boundary_condition": self.boundary_condition.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ObservableSpec:
    """Named observable contract."""

    name: str
    description: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("observable name must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description}


@dataclass(frozen=True, slots=True)
class SpectrumObservableEvaluator:
    spec: ObservableSpec
    evaluate: Callable[[EigensolverResult], Tensor]


def _evaluate_eigenvalues(result: EigensolverResult) -> Tensor:
    return result.eigenvalues


def _evaluate_transition_energies(result: EigensolverResult) -> Tensor:
    return result.eigenvalues[1:] - result.eigenvalues[:-1]


def _evaluate_tunnel_splitting(result: EigensolverResult) -> Tensor:
    return result.eigenvalues[1:2] - result.eigenvalues[0:1]


def _evaluate_anharmonicity(result: EigensolverResult) -> Tensor:
    gaps = _evaluate_transition_energies(result)
    return gaps[1:2] - gaps[0:1]


def _evaluate_potential_on_grid(result: EigensolverResult) -> Tensor:
    return result.potential_on_grid


def _evaluate_ground_density(result: EigensolverResult) -> Tensor:
    return torch.abs(result.eigenstates[0]) ** 2


def _spectrum_observable(
    name: str,
    description: str,
    evaluator: Callable[[EigensolverResult], Tensor],
) -> SpectrumObservableEvaluator:
    return SpectrumObservableEvaluator(ObservableSpec(name, description), evaluator)


_SPECTRUM_OBSERVABLES: dict[str, SpectrumObservableEvaluator] = {
    "eigenvalues": _spectrum_observable(
        "eigenvalues",
        "Lowest eigenvalues of the Hamiltonian.",
        _evaluate_eigenvalues,
    ),
    "transition_energies": _spectrum_observable(
        "transition_energies",
        "Adjacent energy gaps E[n+1] - E[n].",
        _evaluate_transition_energies,
    ),
    "tunnel_splitting": _spectrum_observable(
        "tunnel_splitting",
        "Lowest doublet gap E[1] - E[0] when at least two states are available.",
        _evaluate_tunnel_splitting,
    ),
    "anharmonicity": _spectrum_observable(
        "anharmonicity",
        "Difference between the first two adjacent gaps.",
        _evaluate_anharmonicity,
    ),
    "potential_on_grid": _spectrum_observable(
        "potential_on_grid",
        "Potential values evaluated on the solver grid.",
        _evaluate_potential_on_grid,
    ),
    "ground_density": _spectrum_observable(
        "ground_density",
        "Ground-state probability density evaluated on the solver grid.",
        _evaluate_ground_density,
    ),
}


@dataclass(frozen=True, slots=True)
class ObservableSet:
    """Ordered set of observables requested from a physical problem."""

    observables: tuple[ObservableSpec, ...]

    def __post_init__(self) -> None:
        names = [observable.name for observable in self.observables]
        if len(names) != len(set(names)):
            raise ValueError("observable names must be unique")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(observable.name for observable in self.observables)

    @classmethod
    def from_names(cls, names: tuple[str, ...] | list[str]) -> "ObservableSet":
        observables: list[ObservableSpec] = []
        for name in names:
            evaluator = _SPECTRUM_OBSERVABLES.get(name)
            if evaluator is not None:
                observables.append(evaluator.spec)
            else:
                observables.append(
                    ObservableSpec(
                        name,
                        "Declared by a family or workflow; no built-in spectrum evaluator is registered.",
                    )
                )
        return cls(tuple(observables))

    @classmethod
    def from_family(cls, family: PotentialFamily | PotentialFamilyDefinition | str) -> "ObservableSet":
        if isinstance(family, PotentialFamily):
            observable_names = family.observable_names
        elif isinstance(family, PotentialFamilyDefinition):
            observable_names = family.observable_map
        else:
            observable_names = resolve_potential_family(family).observable_map
        return cls.from_names(list(observable_names))

    def evaluate_spectrum(
        self,
        result: EigensolverResult,
        *,
        strict: bool = True,
    ) -> dict[str, Tensor]:
        payload: dict[str, Tensor] = {}
        unsupported: list[str] = []
        for name in self.names:
            evaluator = _SPECTRUM_OBSERVABLES.get(name)
            if evaluator is None:
                unsupported.append(name)
                continue
            payload[name] = evaluator.evaluate(result)
        if unsupported and strict:
            raise ValueError(
                "no spectrum evaluator registered for observables: "
                + ", ".join(sorted(unsupported))
            )
        return payload

    def to_dict(self) -> dict[str, object]:
        return {"observables": [observable.to_dict() for observable in self.observables]}


@dataclass(frozen=True, slots=True)
class MeasurementModel:
    """Measurement contract over an observable set and a noise model."""

    observable_set: ObservableSet
    noise_scale: float = 1.0
    noise_model: str = "independent-gaussian"

    def __post_init__(self) -> None:
        noise_model = self.noise_model.strip().lower()
        object.__setattr__(self, "noise_model", noise_model)
        if noise_model != "independent-gaussian":
            raise ValueError("only independent-gaussian measurement noise is currently implemented")
        if self.noise_scale <= 0:
            raise ValueError("noise_scale must be positive")

    def predict(self, result: EigensolverResult, *, strict: bool = True) -> dict[str, Tensor]:
        return self.observable_set.evaluate_spectrum(result, strict=strict)

    def residual(
        self,
        result: EigensolverResult,
        observations: Mapping[str, Any],
    ) -> dict[str, Tensor]:
        prediction = self.predict(result, strict=True)
        residuals: dict[str, Tensor] = {}
        for name, observed in observations.items():
            if name not in prediction:
                raise KeyError(f"measurement {name!r} is not predicted by this observable set")
            observed_tensor = torch.as_tensor(
                observed,
                dtype=prediction[name].dtype,
                device=prediction[name].device,
            )
            residuals[name] = prediction[name] - observed_tensor
        return residuals

    def negative_log_likelihood(
        self,
        result: EigensolverResult,
        observations: Mapping[str, Any],
    ) -> Tensor:
        residuals = self.residual(result, observations)
        total = torch.zeros((), dtype=result.eigenvalues.dtype, device=result.eigenvalues.device)
        sigma = torch.as_tensor(
            self.noise_scale,
            dtype=result.eigenvalues.dtype,
            device=result.eigenvalues.device,
        )
        log_normalizer = torch.log(2 * torch.pi * sigma**2)
        for residual in residuals.values():
            total = total + 0.5 * torch.sum((residual / sigma) ** 2 + log_normalizer)
        return total

    def to_dict(self) -> dict[str, object]:
        return {
            "noise_model": self.noise_model,
            "noise_scale": self.noise_scale,
            "observable_set": self.observable_set.to_dict(),
        }


def build_hamiltonian_operator(
    family: str | PotentialFamily | PotentialFamilyDefinition,
    *,
    domain: InfiniteWell1D,
    parameters: Mapping[str, float | Tensor] | None = None,
    basis_spec: BasisSpec | None = None,
    boundary_condition: BoundaryCondition | None = None,
) -> HamiltonianOperator:
    """Build the shared Hamiltonian contract used by spectral workflows."""

    return HamiltonianOperator.from_family(
        family,
        domain=domain,
        parameters=parameters,
        basis_spec=basis_spec,
        boundary_condition=boundary_condition,
    )


__all__ = [
    "BasisSpec",
    "BoundaryCondition",
    "HamiltonianOperator",
    "MeasurementModel",
    "ObservableSet",
    "ObservableSpec",
    "PotentialFamily",
    "SpectrumObservableEvaluator",
    "build_hamiltonian_operator",
]
