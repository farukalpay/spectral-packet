from __future__ import annotations

import torch

from spectral_packet_engine import (
    BasisSpec,
    HamiltonianOperator,
    InfiniteWell1D,
    MeasurementModel,
    ObservableSet,
    PotentialFamily,
    build_hamiltonian_operator,
)


def test_hamiltonian_contract_solves_through_shared_potential_family() -> None:
    domain = InfiniteWell1D.from_length(1.0, dtype=torch.float64)
    family = PotentialFamily.from_name("harmonic")
    operator = build_hamiltonian_operator(
        family,
        domain=domain,
        parameters={"omega": 4.0},
        basis_spec=BasisSpec(num_modes=16, quadrature_points=128),
    )

    assert isinstance(operator, HamiltonianOperator)
    assert operator.potential_family.name == "harmonic"
    assert operator.boundary_condition.kind == "dirichlet"

    result = operator.solve(num_states=4)
    assert result.eigenvalues.shape == (4,)
    assert torch.all(result.eigenvalues[1:] > result.eigenvalues[:-1])

    contract = operator.to_dict()
    assert contract["potential_family"]["name"] == "harmonic"
    assert contract["basis"]["num_modes"] == 16


def test_observable_set_and_measurement_model_share_spectrum_contract() -> None:
    domain = InfiniteWell1D.from_length(1.0, dtype=torch.float64)
    operator = build_hamiltonian_operator(
        "double-well",
        domain=domain,
        parameters={"a_param": 10.0, "b_param": 4.0},
        basis_spec=BasisSpec(num_modes=18, quadrature_points=144),
    )
    result = operator.solve(num_states=5)

    observables = ObservableSet.from_names(["eigenvalues", "transition_energies", "tunnel_splitting"])
    payload = observables.evaluate_spectrum(result)

    assert payload["eigenvalues"].shape == (5,)
    assert payload["transition_energies"].shape == (4,)
    assert payload["tunnel_splitting"].shape == (1,)

    measurement = MeasurementModel(ObservableSet.from_names(["eigenvalues"]), noise_scale=0.05)
    residual = measurement.residual(result, {"eigenvalues": result.eigenvalues.detach().clone()})
    assert torch.allclose(residual["eigenvalues"], torch.zeros_like(result.eigenvalues))
    assert torch.isfinite(measurement.negative_log_likelihood(result, {"eigenvalues": result.eigenvalues}))


def test_family_declared_observables_are_inspectable_without_fake_evaluators() -> None:
    domain = InfiniteWell1D.from_length(1.0, dtype=torch.float64)
    operator = build_hamiltonian_operator(
        "gaussian-barrier",
        domain=domain,
        parameters={"height": 20.0, "width": 0.08, "center": 0.5},
        basis_spec=BasisSpec(num_modes=14, quadrature_points=112),
    )
    result = operator.solve(num_states=3)
    observables = ObservableSet.from_family("gaussian-barrier")

    assert "transmission_coefficient" in observables.names
    evaluated = observables.evaluate_spectrum(result, strict=False)
    assert "eigenvalues" in evaluated
    assert "transmission_coefficient" not in evaluated
