from __future__ import annotations

"""Structured reduced models built on the bounded-domain spectral core.

This module is intentionally narrow. It does not claim arbitrary 2D/3D
capabilities. It implements restricted, inspectable reductions that preserve
the repository's spectral center of gravity:

- separable tensor-product spectra from independent 1D axes,
- a phase-1 structured dimensional lift for separable 2D bounded problems,
- coupled-channel adiabatic surface analysis for avoided crossings,
- radial effective-coordinate reductions,
- low-rank matrix approximations for structured coefficient objects.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from spectral_packet_engine.domain import InfiniteWell1D
from spectral_packet_engine.eigensolver import solve_eigenproblem
from spectral_packet_engine.parametric_potentials import resolve_potential_family
from spectral_packet_engine.runtime import TorchRuntime, inspect_torch_runtime
from spectral_packet_engine.tensor_product import (
    KroneckerSumOperator2D,
    TensorAxisModes,
    TensorModeBudget2D,
    TensorProductBasis2D,
    TensorProductBasisSummary2D,
    TensorTruncationDiagnostics2D,
    build_tensor_mode_budget_2d,
    make_infinite_well_axis_modes,
    make_tensor_axis_modes,
    summarize_tensor_product_basis_2d,
    summarize_tensor_truncation_2d,
)

Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class LowRankFactorizationSummary:
    input_shape: tuple[int, int]
    retained_rank: int
    singular_values: Tensor
    energy_capture: float
    reconstruction_error: float
    left_factors: Tensor
    right_factors: Tensor


@dataclass(frozen=True, slots=True)
class CouplingStructureSummary:
    tensor_shape: tuple[int, int]
    matrix_shape: tuple[int, int]
    total_mode_count: int
    frobenius_norm: float
    diagonal_energy_fraction: float
    additive_diagonal_score: float | None
    low_rank_rank: int
    low_rank_energy_capture: float
    low_rank_relative_error: float
    block_count: int
    within_block_energy_fraction: float | None
    off_block_energy_fraction: float | None
    assumptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StructuredOperatorSummary:
    kind: str
    tensor_shape: tuple[int, int]
    matrix_shape: tuple[int, int]
    coupling_kind: str
    coupling_nonzero_count: int
    diagonal_coupling_l2_norm: float | None


@dataclass(frozen=True, slots=True)
class SeparableSpectrumSummary:
    family_x: str
    family_y: str
    parameters_x: dict[str, float]
    parameters_y: dict[str, float]
    domain_lengths: tuple[float, float]
    basis: TensorProductBasisSummary2D
    mode_budget: TensorModeBudget2D
    truncation: TensorTruncationDiagnostics2D
    operator: StructuredOperatorSummary
    axis_eigenvalues_x: Tensor
    axis_eigenvalues_y: Tensor
    combined_eigenvalues: Tensor
    state_index_pairs: tuple[tuple[int, int], ...]
    transition_energies_from_ground: Tensor
    ground_density_low_rank: LowRankFactorizationSummary
    assumptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Separable2DReportOverview:
    example_name: str
    axis_models: tuple[str, str]
    domain_lengths: tuple[float, float]
    tensor_shape: tuple[int, int]
    total_tensor_modes: int
    retained_combined_state_count: int
    retained_energy_ceiling: float | None
    analytic_reference: str
    coupling_kind: str
    max_absolute_reference_error: float


@dataclass(frozen=True, slots=True)
class Separable2DReport:
    runtime: TorchRuntime
    overview: Separable2DReportOverview
    spectrum: SeparableSpectrumSummary
    analytic_reference_eigenvalues: Tensor
    absolute_error_to_reference: Tensor
    assumptions: tuple[str, ...]

    def write_artifacts(
        self,
        output_dir: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ):
        from spectral_packet_engine.artifacts import inspect_artifact_directory, write_reduced_model_artifacts

        write_reduced_model_artifacts(output_dir, self, metadata=metadata)
        return inspect_artifact_directory(output_dir)


@dataclass(frozen=True, slots=True)
class CoupledChannelSurfaceSummary:
    grid: Tensor
    diabatic_potentials: Tensor
    adiabatic_potentials: Tensor
    coupling_profile: Tensor
    derivative_couplings: Tensor
    minimum_gap: float
    crossing_position: float
    mean_derivative_coupling: float
    max_derivative_coupling: float
    assumptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RadialReductionSummary:
    family: str
    parameters: dict[str, float]
    angular_momentum: int
    radial_interval: tuple[float, float]
    eigenvalues: Tensor
    radial_grid: Tensor
    effective_potential: Tensor
    assumptions: tuple[str, ...]


def low_rank_factorize_matrix(
    matrix,
    *,
    rank: int | None = None,
    capture_fraction: float = 0.99,
) -> LowRankFactorizationSummary:
    values = torch.as_tensor(matrix, dtype=torch.float64)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if rank is not None and rank <= 0:
        raise ValueError("rank must be positive when provided")
    if not (0.0 < capture_fraction <= 1.0):
        raise ValueError("capture_fraction must lie in (0, 1]")

    U, S, Vh = torch.linalg.svd(values, full_matrices=False)
    squared = S**2
    total = float(torch.sum(squared).item())
    if total <= 0.0:
        retained_rank = 1 if rank is None else min(rank, int(S.shape[0]))
        retained_rank = max(retained_rank, 1)
    elif rank is None:
        cumulative = torch.cumsum(squared, dim=0) / torch.sum(squared)
        retained_rank = int(torch.searchsorted(cumulative, torch.tensor(capture_fraction, dtype=S.dtype)).item()) + 1
    else:
        retained_rank = min(rank, int(S.shape[0]))

    retained_rank = max(1, retained_rank)
    truncated = (U[:, :retained_rank] * S[:retained_rank]) @ Vh[:retained_rank, :]
    residual = values - truncated
    return LowRankFactorizationSummary(
        input_shape=(int(values.shape[0]), int(values.shape[1])),
        retained_rank=retained_rank,
        singular_values=S.detach(),
        energy_capture=0.0 if total <= 0.0 else float(torch.sum(squared[:retained_rank]).item() / total),
        reconstruction_error=float(torch.linalg.norm(residual).item() / max(float(torch.linalg.norm(values).item()), 1e-12)),
        left_factors=U[:, :retained_rank].detach(),
        right_factors=Vh[:retained_rank, :].detach(),
    )


def _coerce_coupling_matrix(coupling, *, tensor_shape: tuple[int, int]) -> Tensor:
    num_x, num_y = tensor_shape
    if num_x <= 0 or num_y <= 0:
        raise ValueError("tensor_shape entries must be positive")
    total = int(num_x * num_y)
    values = torch.as_tensor(coupling)
    if values.ndim == 1:
        if int(values.shape[0]) != total:
            raise ValueError("one-dimensional coupling must have tensor_shape[0] * tensor_shape[1] entries")
        values = torch.diag(values)
    elif values.ndim == 2:
        if tuple(values.shape) != (total, total):
            raise ValueError("two-dimensional coupling must have shape (total_modes, total_modes)")
    elif values.ndim == 4:
        if tuple(values.shape) != (num_x, num_y, num_x, num_y):
            raise ValueError("four-dimensional coupling must have shape (x, y, x, y)")
        values = values.reshape(total, total)
    else:
        raise ValueError("coupling must be a vector, matrix, or rank-4 tensor over the tensor-product basis")
    if not torch.isfinite(values).all().item():
        raise ValueError("coupling values must be finite")
    dtype = values.dtype if torch.is_complex(values) else torch.float64
    return values.to(dtype=dtype)


def _low_rank_energy_metrics(matrix: Tensor, capture_fraction: float) -> tuple[int, Tensor, float, float]:
    if not (0.0 < capture_fraction <= 1.0):
        raise ValueError("capture_fraction must lie in (0, 1]")
    singular_values = torch.linalg.svdvals(matrix)
    squared = torch.real(singular_values * singular_values.conj())
    total = float(torch.sum(squared).item())
    if total <= 0.0:
        return 0, singular_values.detach(), 0.0, 0.0
    cumulative = torch.cumsum(squared, dim=0) / torch.sum(squared)
    retained_rank = int(torch.searchsorted(cumulative, torch.tensor(capture_fraction, dtype=cumulative.dtype)).item()) + 1
    retained_rank = min(max(retained_rank, 1), int(singular_values.numel()))
    energy_capture = float(torch.sum(squared[:retained_rank]).item() / total)
    relative_error = float(torch.sqrt(torch.clamp(1.0 - torch.as_tensor(energy_capture), min=0.0)).item())
    return retained_rank, singular_values.detach(), energy_capture, relative_error


def _block_energy_fraction(
    matrix: Tensor,
    block_partitions: Sequence[Sequence[int]] | None,
) -> tuple[int, float | None, float | None]:
    if block_partitions is None:
        return 0, None, None
    total = int(matrix.shape[0])
    mask = torch.zeros(total, total, dtype=torch.bool, device=matrix.device)
    seen: set[int] = set()
    block_count = 0
    for block in block_partitions:
        indices = tuple(int(index) for index in block)
        if not indices:
            continue
        if any(index < 0 or index >= total for index in indices):
            raise ValueError("block_partitions contain an out-of-range index")
        if seen.intersection(indices):
            raise ValueError("block_partitions must not overlap")
        seen.update(indices)
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=matrix.device)
        mask[index_tensor[:, None], index_tensor[None, :]] = True
        block_count += 1
    energy = torch.abs(matrix) ** 2
    total_energy = float(torch.sum(energy).item())
    if total_energy <= 0.0:
        return block_count, 0.0, 0.0
    within = float(torch.sum(energy[mask]).item() / total_energy)
    return block_count, within, 1.0 - within


def _additive_diagonal_score(matrix: Tensor, tensor_shape: tuple[int, int]) -> float | None:
    diagonal = torch.diagonal(matrix)
    if torch.is_complex(diagonal):
        diagonal_values = diagonal.real
    else:
        diagonal_values = diagonal
    surface = diagonal_values.reshape(tensor_shape)
    denominator = torch.linalg.norm(surface)
    if float(denominator.item()) <= 0.0:
        return None
    row_mean = torch.mean(surface, dim=1, keepdim=True)
    col_mean = torch.mean(surface, dim=0, keepdim=True)
    additive = row_mean + col_mean - torch.mean(surface)
    residual = torch.linalg.norm(surface - additive) / denominator
    return float(torch.clamp(1.0 - residual, min=0.0, max=1.0).item())


def analyze_structured_coupling(
    coupling,
    *,
    tensor_shape: tuple[int, int],
    block_partitions: Sequence[Sequence[int]] | None = None,
    capture_fraction: float = 0.99,
) -> CouplingStructureSummary:
    """Summarize near-separable, block-coupled, and low-rank structure.

    The function analyzes an explicit coupling over an existing tensor-product
    basis.  It does not introduce a generic multidimensional mesh or solver.
    """

    matrix = _coerce_coupling_matrix(coupling, tensor_shape=tensor_shape)
    total = int(matrix.shape[0])
    frobenius = torch.linalg.norm(matrix)
    energy = torch.abs(matrix) ** 2
    total_energy = float(torch.sum(energy).item())
    diagonal_energy = float(torch.sum(torch.abs(torch.diagonal(matrix)) ** 2).item())
    diagonal_energy_fraction = 0.0 if total_energy <= 0.0 else diagonal_energy / total_energy
    retained_rank, _, low_rank_capture, low_rank_error = _low_rank_energy_metrics(
        matrix,
        capture_fraction=capture_fraction,
    )
    block_count, within_block, off_block = _block_energy_fraction(matrix, block_partitions)
    return CouplingStructureSummary(
        tensor_shape=(int(tensor_shape[0]), int(tensor_shape[1])),
        matrix_shape=(total, total),
        total_mode_count=total,
        frobenius_norm=float(torch.real(frobenius).item()),
        diagonal_energy_fraction=diagonal_energy_fraction,
        additive_diagonal_score=_additive_diagonal_score(matrix, tensor_shape),
        low_rank_rank=retained_rank,
        low_rank_energy_capture=low_rank_capture,
        low_rank_relative_error=low_rank_error,
        block_count=block_count,
        within_block_energy_fraction=within_block,
        off_block_energy_fraction=off_block,
        assumptions=(
            "The coupling is analyzed over an explicit retained tensor-product basis.",
            "Low-rank structure is measured by singular-value energy capture, not by a generic multidimensional discretization.",
            "Block-coupling structure is reported only when explicit block partitions are supplied.",
        ),
    )


def _domain_length_from_interval(interval: tuple[float, float]) -> float:
    return float(interval[1] - interval[0])


def _coerce_coupling_diagonal(
    coupling_diagonal: Sequence[float] | Tensor | None,
    *,
    tensor_shape: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor | None:
    if coupling_diagonal is None:
        return None
    coupling = torch.as_tensor(coupling_diagonal, dtype=dtype, device=device)
    if coupling.ndim == 2:
        if tuple(coupling.shape) != tensor_shape:
            raise ValueError("two-dimensional coupling_diagonal must match the tensor basis shape")
        coupling = coupling.reshape(-1)
    elif coupling.ndim != 1:
        raise ValueError("coupling_diagonal must be one- or two-dimensional")
    if int(coupling.shape[0]) != tensor_shape[0] * tensor_shape[1]:
        raise ValueError("coupling_diagonal length must match the tensor basis size")
    return coupling.detach()


def _structured_operator_summary(operator: KroneckerSumOperator2D) -> StructuredOperatorSummary:
    coupling_nonzero_count = 0
    diagonal_coupling_l2_norm = None
    if operator.coupling_diagonal is not None:
        coupling_nonzero_count = int(torch.count_nonzero(operator.coupling_diagonal).item())
        diagonal_coupling_l2_norm = float(torch.linalg.norm(operator.coupling_diagonal).item())
    return StructuredOperatorSummary(
        kind="kronecker-sum",
        tensor_shape=operator.tensor_shape,
        matrix_shape=(operator.size, operator.size),
        coupling_kind=operator.coupling_kind,
        coupling_nonzero_count=coupling_nonzero_count,
        diagonal_coupling_l2_norm=diagonal_coupling_l2_norm,
    )


def _make_family_axis_modes(
    *,
    axis_name: str,
    family: str,
    parameters: Mapping[str, float],
    domain_length: float,
    basis_resolution_points: int,
    requested_mode_count: int,
    runtime: TorchRuntime,
) -> tuple[str, TensorAxisModes]:
    domain = InfiniteWell1D.from_length(
        domain_length,
        dtype=runtime.preferred_real_dtype,
        device=runtime.device,
    )
    family_def = resolve_potential_family(family)
    result = solve_eigenproblem(
        family_def.build_from_mapping(domain, parameters),
        domain,
        num_points=basis_resolution_points,
        num_states=requested_mode_count,
    )
    return (
        family_def.name,
        make_tensor_axis_modes(
            axis_name,
            kind="parametric-family",
            requested_mode_count=requested_mode_count,
            basis_resolution_points=basis_resolution_points,
            domain_interval=(float(domain.left), float(domain.right)),
            grid=result.grid.detach(),
            eigenvalues=result.eigenvalues.detach(),
            eigenstates=result.eigenstates.detach(),
        ),
    )


def _build_separable_spectrum_summary(
    *,
    family_x: str,
    parameters_x: Mapping[str, float],
    axis_x: TensorAxisModes,
    family_y: str,
    parameters_y: Mapping[str, float],
    axis_y: TensorAxisModes,
    num_combined_states: int,
    low_rank_rank: int,
    coupling_diagonal: Sequence[float] | Tensor | None = None,
    assumptions: tuple[str, ...],
) -> SeparableSpectrumSummary:
    if num_combined_states <= 0:
        raise ValueError("num_combined_states must be positive")

    basis = TensorProductBasis2D(axis_x=axis_x, axis_y=axis_y)
    coupling = _coerce_coupling_diagonal(
        coupling_diagonal,
        tensor_shape=basis.shape,
        dtype=axis_x.eigenvalues.dtype,
        device=axis_x.eigenvalues.device,
    )
    operator = KroneckerSumOperator2D(
        axis_energies_x=axis_x.eigenvalues,
        axis_energies_y=axis_y.eigenvalues,
        coupling_diagonal=coupling,
    )
    sorted_full_spectrum = torch.sort(operator.diagonal_entries())[0]
    keep = min(int(num_combined_states), int(sorted_full_spectrum.shape[0]))
    combined_eigenvalues, state_index_pairs = operator.lowest_states(keep)
    transition_energies = (
        combined_eigenvalues[1:] - combined_eigenvalues[0]
        if keep > 1
        else torch.empty(0, dtype=combined_eigenvalues.dtype, device=combined_eigenvalues.device)
    )
    ground_density = basis.ground_density()

    return SeparableSpectrumSummary(
        family_x=family_x,
        family_y=family_y,
        parameters_x={name: float(value) for name, value in parameters_x.items()},
        parameters_y={name: float(value) for name, value in parameters_y.items()},
        domain_lengths=(
            _domain_length_from_interval(axis_x.domain_interval),
            _domain_length_from_interval(axis_y.domain_interval),
        ),
        basis=summarize_tensor_product_basis_2d(basis),
        mode_budget=build_tensor_mode_budget_2d(
            basis,
            requested_combined_state_count=num_combined_states,
            retained_combined_state_count=keep,
        ),
        truncation=summarize_tensor_truncation_2d(
            basis,
            retained_combined_state_count=keep,
            sorted_diagonal=sorted_full_spectrum,
        ),
        operator=_structured_operator_summary(operator),
        axis_eigenvalues_x=axis_x.eigenvalues.detach(),
        axis_eigenvalues_y=axis_y.eigenvalues.detach(),
        combined_eigenvalues=combined_eigenvalues.detach(),
        state_index_pairs=state_index_pairs,
        transition_energies_from_ground=transition_energies.detach(),
        ground_density_low_rank=low_rank_factorize_matrix(ground_density, rank=low_rank_rank),
        assumptions=assumptions,
    )


def analyze_separable_tensor_product_spectrum(
    *,
    family_x: str,
    parameters_x: Mapping[str, float],
    family_y: str,
    parameters_y: Mapping[str, float],
    domain_length_x: float = 1.0,
    domain_length_y: float = 1.0,
    num_points_x: int = 96,
    num_points_y: int = 96,
    num_states_x: int = 6,
    num_states_y: int = 6,
    num_combined_states: int = 12,
    low_rank_rank: int = 1,
    coupling_diagonal: Sequence[float] | Tensor | None = None,
    device: str | torch.device = "auto",
) -> SeparableSpectrumSummary:
    runtime = inspect_torch_runtime(device)
    resolved_family_x, axis_x = _make_family_axis_modes(
        axis_name="x",
        family=family_x,
        parameters=parameters_x,
        domain_length=domain_length_x,
        basis_resolution_points=num_points_x,
        requested_mode_count=num_states_x,
        runtime=runtime,
    )
    resolved_family_y, axis_y = _make_family_axis_modes(
        axis_name="y",
        family=family_y,
        parameters=parameters_y,
        domain_length=domain_length_y,
        basis_resolution_points=num_points_y,
        requested_mode_count=num_states_y,
        runtime=runtime,
    )
    coupling_assumption = (
        "A diagonal retained-basis coupling is included explicitly and reported in the structured operator summary."
        if coupling_diagonal is not None
        else "No retained-basis coupling is applied beyond the Kronecker-sum separable operator."
    )
    return _build_separable_spectrum_summary(
        family_x=resolved_family_x,
        parameters_x=parameters_x,
        axis_x=axis_x,
        family_y=resolved_family_y,
        parameters_y=parameters_y,
        axis_y=axis_y,
        num_combined_states=num_combined_states,
        low_rank_rank=low_rank_rank,
        coupling_diagonal=coupling_diagonal,
        assumptions=(
            "The total Hamiltonian is represented on a retained 2D tensor-product basis with x-major/y-minor indexing.",
            "The structured operator is Hx ⊗ I + I ⊗ Hy on the retained modal axes, not a generic dense 2D discretization.",
            coupling_assumption,
            "The low-rank summary applies to the separable ground-state density on the reported tensor grid, not to a general non-separable 2D state.",
        ),
    )


def _box_reference_eigenvalues(
    *,
    domain_length_x: float,
    domain_length_y: float,
    num_modes_x: int,
    num_modes_y: int,
    num_combined_states: int,
    mass: float,
    hbar: float,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    mode_x = torch.arange(1, num_modes_x + 1, dtype=dtype, device=device)
    mode_y = torch.arange(1, num_modes_y + 1, dtype=dtype, device=device)
    prefactor = (torch.pi**2 * (hbar**2)) / (2.0 * mass)
    spectrum = prefactor * (
        (mode_x[:, None] ** 2) / (domain_length_x**2)
        + (mode_y[None, :] ** 2) / (domain_length_y**2)
    )
    sorted_values = torch.sort(spectrum.reshape(-1))[0]
    return sorted_values[: min(int(num_combined_states), int(sorted_values.shape[0]))].detach()


def build_separable_2d_report(
    *,
    domain_length_x: float = 1.0,
    domain_length_y: float = 1.0,
    num_modes_x: int = 6,
    num_modes_y: int = 6,
    grid_points_x: int = 96,
    grid_points_y: int = 96,
    num_combined_states: int = 12,
    low_rank_rank: int = 1,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "auto",
) -> Separable2DReport:
    runtime = inspect_torch_runtime(device)
    axis_x = make_infinite_well_axis_modes(
        "x",
        domain_length=domain_length_x,
        num_modes=num_modes_x,
        evaluation_grid_points=grid_points_x,
        mass=mass,
        hbar=hbar,
        dtype=runtime.preferred_real_dtype,
        device=runtime.device,
    )
    axis_y = make_infinite_well_axis_modes(
        "y",
        domain_length=domain_length_y,
        num_modes=num_modes_y,
        evaluation_grid_points=grid_points_y,
        mass=mass,
        hbar=hbar,
        dtype=runtime.preferred_real_dtype,
        device=runtime.device,
    )
    spectrum = _build_separable_spectrum_summary(
        family_x="infinite-well",
        parameters_x={},
        axis_x=axis_x,
        family_y="infinite-well",
        parameters_y={},
        axis_y=axis_y,
        num_combined_states=num_combined_states,
        low_rank_rank=low_rank_rank,
        assumptions=(
            "This report is a phase-1 structured dimensional lift for the separable box-plus-box problem on a bounded domain.",
            "Each axis is an explicit retained 1D infinite-well modal basis; no arbitrary geometry or mesh layer is introduced.",
            "The report validates a separable 2D additive spectrum against the closed-form box-plus-box reference on the same retained mode budget.",
        ),
    )
    analytic_reference = _box_reference_eigenvalues(
        domain_length_x=domain_length_x,
        domain_length_y=domain_length_y,
        num_modes_x=spectrum.basis.tensor_shape[0],
        num_modes_y=spectrum.basis.tensor_shape[1],
        num_combined_states=spectrum.mode_budget.retained_combined_state_count,
        mass=mass,
        hbar=hbar,
        dtype=runtime.preferred_real_dtype,
        device=runtime.device,
    )
    absolute_error = torch.abs(spectrum.combined_eigenvalues - analytic_reference)
    retained_energy_ceiling = spectrum.truncation.retained_energy_ceiling
    return Separable2DReport(
        runtime=runtime,
        overview=Separable2DReportOverview(
            example_name="box-plus-box",
            axis_models=spectrum.basis.axis_kinds,
            domain_lengths=spectrum.domain_lengths,
            tensor_shape=spectrum.basis.tensor_shape,
            total_tensor_modes=spectrum.mode_budget.total_tensor_mode_count,
            retained_combined_state_count=spectrum.mode_budget.retained_combined_state_count,
            retained_energy_ceiling=retained_energy_ceiling,
            analytic_reference="closed-form additive box-plus-box spectrum on the retained mode budget",
            coupling_kind=spectrum.operator.coupling_kind,
            max_absolute_reference_error=float(torch.max(absolute_error).item()) if absolute_error.numel() else 0.0,
        ),
        spectrum=spectrum,
        analytic_reference_eigenvalues=analytic_reference.detach(),
        absolute_error_to_reference=absolute_error.detach(),
        assumptions=(
            "This workflow is intentionally narrow: it reports one separable 2D bounded-domain path rather than a generic multidimensional solver.",
            "Artifact outputs expose the retained tensor basis, mode budget, structured operator contract, and truncation cutoff explicitly.",
            "The example is box-plus-box because it is analytically interpretable and deterministic within the bounded-domain product identity.",
        ),
    )


def _finite_difference_along_grid(values: Tensor, grid: Tensor) -> Tensor:
    derivative = torch.zeros_like(values)
    derivative[1:-1] = (values[2:] - values[:-2]) / (grid[2:, None, None] - grid[:-2, None, None])
    derivative[0] = (values[1] - values[0]) / (grid[1] - grid[0])
    derivative[-1] = (values[-1] - values[-2]) / (grid[-1] - grid[-2])
    return derivative


def analyze_coupled_channel_surfaces(
    *,
    domain_length: float = 1.0,
    grid_points: int = 256,
    slope: float = 30.0,
    bias: float = 0.0,
    coupling: float = 2.0,
    coupling_width: float = 0.12,
    device: str | torch.device = "cpu",
) -> CoupledChannelSurfaceSummary:
    if grid_points < 4:
        raise ValueError("grid_points must be at least 4")
    if coupling_width <= 0:
        raise ValueError("coupling_width must be positive")

    runtime = inspect_torch_runtime(device)
    grid = torch.linspace(0.0, domain_length, grid_points, dtype=runtime.preferred_real_dtype, device=runtime.device)
    center = torch.tensor(domain_length / 2.0, dtype=grid.dtype, device=grid.device)
    displacement = grid - center
    diabatic_1 = slope * displacement + bias / 2.0
    diabatic_2 = -slope * displacement - bias / 2.0
    coupling_profile = coupling * torch.exp(-(displacement**2) / (2 * coupling_width**2))

    hamiltonian = torch.zeros(grid_points, 2, 2, dtype=grid.dtype, device=grid.device)
    hamiltonian[:, 0, 0] = diabatic_1
    hamiltonian[:, 1, 1] = diabatic_2
    hamiltonian[:, 0, 1] = coupling_profile
    hamiltonian[:, 1, 0] = coupling_profile

    adiabatic_potentials, eigenvectors = torch.linalg.eigh(hamiltonian)
    eigenvector_derivative = _finite_difference_along_grid(eigenvectors, grid)
    derivative_couplings = torch.einsum("xai,xaj->xij", eigenvectors, eigenvector_derivative).permute(1, 2, 0)
    gap = adiabatic_potentials[:, 1] - adiabatic_potentials[:, 0]
    gap_index = int(torch.argmin(gap).item())

    return CoupledChannelSurfaceSummary(
        grid=grid.detach(),
        diabatic_potentials=torch.stack((diabatic_1, diabatic_2)).detach(),
        adiabatic_potentials=adiabatic_potentials.transpose(0, 1).detach(),
        coupling_profile=coupling_profile.detach(),
        derivative_couplings=derivative_couplings.detach(),
        minimum_gap=float(gap[gap_index].item()),
        crossing_position=float(grid[gap_index].item()),
        mean_derivative_coupling=float(torch.mean(torch.abs(derivative_couplings[0, 1])).item()),
        max_derivative_coupling=float(torch.max(torch.abs(derivative_couplings[0, 1])).item()),
        assumptions=(
            "This workflow models a reduced two-channel avoided crossing rather than a full multidimensional electronic-structure problem.",
            "Derivative couplings are estimated by finite differences of local adiabatic eigenvectors on the reported 1D grid.",
        ),
    )


def solve_radial_reduction(
    *,
    family: str,
    parameters: Mapping[str, float],
    angular_momentum: int = 0,
    radial_min: float = 0.05,
    radial_max: float = 3.0,
    num_points: int = 128,
    num_states: int = 6,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> RadialReductionSummary:
    if radial_min <= 0.0:
        raise ValueError("radial_min must be strictly positive for the centrifugal term")
    if radial_max <= radial_min:
        raise ValueError("radial_max must be greater than radial_min")
    if angular_momentum < 0:
        raise ValueError("angular_momentum must be non-negative")

    runtime = inspect_torch_runtime(device)
    domain = InfiniteWell1D(
        left=torch.tensor(radial_min, dtype=runtime.preferred_real_dtype, device=runtime.device),
        right=torch.tensor(radial_max, dtype=runtime.preferred_real_dtype, device=runtime.device),
        mass=torch.tensor(mass, dtype=runtime.preferred_real_dtype, device=runtime.device),
        hbar=torch.tensor(hbar, dtype=runtime.preferred_real_dtype, device=runtime.device),
    )
    family_def = resolve_potential_family(family)
    base_potential = family_def.build_from_mapping(domain, parameters)

    def effective_potential(radius: Tensor) -> Tensor:
        centrifugal = (domain.hbar**2 * angular_momentum * (angular_momentum + 1)) / (2 * domain.mass * radius**2)
        return base_potential(radius) + centrifugal

    result = solve_eigenproblem(
        effective_potential,
        domain,
        num_points=num_points,
        num_states=num_states,
    )

    return RadialReductionSummary(
        family=family_def.name,
        parameters={name: float(value) for name, value in parameters.items()},
        angular_momentum=int(angular_momentum),
        radial_interval=(float(domain.left), float(domain.right)),
        eigenvalues=result.eigenvalues.detach(),
        radial_grid=result.grid.detach(),
        effective_potential=result.potential_on_grid.detach(),
        assumptions=(
            "This is a 1D radial effective-coordinate reduction with Dirichlet boundaries on a finite interval.",
            "The centrifugal term l(l+1)hbar^2/(2mr^2) is included explicitly; no claim is made for full arbitrary 3D geometry.",
        ),
    )


__all__ = [
    "CoupledChannelSurfaceSummary",
    "CouplingStructureSummary",
    "LowRankFactorizationSummary",
    "RadialReductionSummary",
    "Separable2DReport",
    "Separable2DReportOverview",
    "SeparableSpectrumSummary",
    "StructuredOperatorSummary",
    "analyze_coupled_channel_surfaces",
    "analyze_separable_tensor_product_spectrum",
    "analyze_structured_coupling",
    "build_separable_2d_report",
    "low_rank_factorize_matrix",
    "solve_radial_reduction",
]
