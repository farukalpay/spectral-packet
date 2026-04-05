from __future__ import annotations

"""Structured 2D tensor-product primitives for bounded-domain spectral lifts.

This module is intentionally narrow. It supports explicit 2D tensor-product
bases built from bounded 1D modal axes and Kronecker-sum operators of the form
Hx ⊗ I + I ⊗ Hy plus an optional diagonal coupling in the retained tensor basis.
It does not claim arbitrary geometry, mesh handling, or unrestricted
high-dimensional solvers.
"""

from dataclasses import dataclass

import torch

from spectral_packet_engine.basis import InfiniteWellBasis
from spectral_packet_engine.domain import InfiniteWell1D, coerce_tensor

Tensor = torch.Tensor

_FLATTENING_ORDER = "x-major-then-y"


def _coerce_real_vector(values, *, name: str, dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> Tensor:
    tensor = coerce_tensor(values, dtype=dtype, device=device)
    if tensor.ndim != 1 or tensor.numel() == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional tensor or sequence")
    if torch.is_complex(tensor):
        raise TypeError(f"{name} must be real-valued")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must contain only finite values")
    return tensor


@dataclass(frozen=True, slots=True)
class TensorAxisModes:
    axis_name: str
    kind: str
    requested_mode_count: int
    basis_resolution_points: int
    domain_interval: tuple[float, float]
    grid: Tensor
    eigenvalues: Tensor
    eigenstates: Tensor

    def __post_init__(self) -> None:
        if self.requested_mode_count <= 0:
            raise ValueError("requested_mode_count must be positive")
        if self.basis_resolution_points <= 0:
            raise ValueError("basis_resolution_points must be positive")

        grid = _coerce_real_vector(self.grid, name="grid")
        eigenvalues = _coerce_real_vector(
            self.eigenvalues,
            name="eigenvalues",
            dtype=grid.dtype,
            device=grid.device,
        )
        eigenstates = coerce_tensor(self.eigenstates, device=grid.device)
        if eigenstates.ndim != 2:
            raise ValueError("eigenstates must be two-dimensional")
        if tuple(eigenstates.shape) != (int(eigenvalues.shape[0]), int(grid.shape[0])):
            raise ValueError("eigenstates must have shape (num_modes, num_grid_points)")
        left, right = self.domain_interval
        if not right > left:
            raise ValueError("domain_interval requires right > left")

        object.__setattr__(self, "grid", grid.detach())
        object.__setattr__(self, "eigenvalues", eigenvalues.detach())
        object.__setattr__(self, "eigenstates", eigenstates.detach())

    @property
    def retained_mode_count(self) -> int:
        return int(self.eigenvalues.shape[0])

    @property
    def evaluation_grid_points(self) -> int:
        return int(self.grid.shape[0])


@dataclass(frozen=True, slots=True)
class TensorProductBasis2D:
    axis_x: TensorAxisModes
    axis_y: TensorAxisModes

    def __post_init__(self) -> None:
        if self.axis_x.eigenvalues.device != self.axis_y.eigenvalues.device:
            raise ValueError("tensor-product axes must live on the same device")
        if self.axis_x.eigenvalues.dtype != self.axis_y.eigenvalues.dtype:
            raise ValueError("tensor-product axes must use the same real dtype")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.axis_x.retained_mode_count, self.axis_y.retained_mode_count)

    @property
    def total_mode_count(self) -> int:
        shape = self.shape
        return int(shape[0] * shape[1])

    @property
    def flattening_order(self) -> str:
        return _FLATTENING_ORDER

    def flatten_index(self, state_index_x: int, state_index_y: int) -> int:
        num_x, num_y = self.shape
        if not (0 <= state_index_x < num_x):
            raise IndexError("state_index_x is out of range")
        if not (0 <= state_index_y < num_y):
            raise IndexError("state_index_y is out of range")
        return int(state_index_x * num_y + state_index_y)

    def unflatten_index(self, flat_index: int) -> tuple[int, int]:
        if not (0 <= flat_index < self.total_mode_count):
            raise IndexError("flat_index is out of range")
        num_y = self.shape[1]
        return (int(flat_index // num_y), int(flat_index % num_y))

    def energy_grid(self) -> Tensor:
        return self.axis_x.eigenvalues[:, None] + self.axis_y.eigenvalues[None, :]

    def ground_density(self) -> Tensor:
        density_x = torch.abs(self.axis_x.eigenstates[0]) ** 2
        density_y = torch.abs(self.axis_y.eigenstates[0]) ** 2
        return torch.outer(density_x, density_y)


@dataclass(frozen=True, slots=True)
class TensorProductBasisSummary2D:
    axis_names: tuple[str, str]
    axis_kinds: tuple[str, str]
    tensor_shape: tuple[int, int]
    total_mode_count: int
    domain_intervals: tuple[tuple[float, float], tuple[float, float]]
    evaluation_grid_shape: tuple[int, int]
    flattening_order: str


@dataclass(frozen=True, slots=True)
class TensorAxisModeBudget:
    axis_name: str
    kind: str
    requested_mode_count: int
    retained_mode_count: int
    basis_resolution_points: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class TensorModeBudget2D:
    axis_x: TensorAxisModeBudget
    axis_y: TensorAxisModeBudget
    tensor_shape: tuple[int, int]
    total_tensor_mode_count: int
    requested_combined_state_count: int
    retained_combined_state_count: int


@dataclass(frozen=True, slots=True)
class TensorTruncationDiagnostics2D:
    axis_truncation_applied: bool
    combined_state_truncation_applied: bool
    discarded_tensor_mode_count: int
    retained_tensor_fraction: float
    retained_energy_ceiling: float | None
    first_discarded_eigenvalue: float | None


@dataclass(frozen=True, slots=True)
class KroneckerSumOperator2D:
    axis_energies_x: Tensor
    axis_energies_y: Tensor
    coupling_diagonal: Tensor | None = None

    def __post_init__(self) -> None:
        axis_energies_x = _coerce_real_vector(self.axis_energies_x, name="axis_energies_x")
        axis_energies_y = _coerce_real_vector(
            self.axis_energies_y,
            name="axis_energies_y",
            dtype=axis_energies_x.dtype,
            device=axis_energies_x.device,
        )
        object.__setattr__(self, "axis_energies_x", axis_energies_x.detach())
        object.__setattr__(self, "axis_energies_y", axis_energies_y.detach())

        if self.coupling_diagonal is None:
            return

        coupling = coerce_tensor(
            self.coupling_diagonal,
            dtype=axis_energies_x.dtype,
            device=axis_energies_x.device,
        )
        tensor_shape = (int(axis_energies_x.shape[0]), int(axis_energies_y.shape[0]))
        if coupling.ndim == 2:
            if tuple(coupling.shape) != tensor_shape:
                raise ValueError("two-dimensional coupling_diagonal must match the tensor basis shape")
            coupling = coupling.reshape(-1)
        elif coupling.ndim != 1:
            raise ValueError("coupling_diagonal must be one- or two-dimensional")
        if int(coupling.shape[0]) != tensor_shape[0] * tensor_shape[1]:
            raise ValueError("coupling_diagonal length must match the tensor basis size")
        if torch.is_complex(coupling):
            raise TypeError("coupling_diagonal must be real-valued")
        if not torch.isfinite(coupling).all().item():
            raise ValueError("coupling_diagonal must contain only finite values")
        object.__setattr__(self, "coupling_diagonal", coupling.detach())

    @property
    def tensor_shape(self) -> tuple[int, int]:
        return (int(self.axis_energies_x.shape[0]), int(self.axis_energies_y.shape[0]))

    @property
    def size(self) -> int:
        tensor_shape = self.tensor_shape
        return int(tensor_shape[0] * tensor_shape[1])

    @property
    def coupling_kind(self) -> str:
        return "diagonal" if self.coupling_diagonal is not None else "none"

    def diagonal_entries(self) -> Tensor:
        diagonal = (self.axis_energies_x[:, None] + self.axis_energies_y[None, :]).reshape(-1)
        if self.coupling_diagonal is not None:
            diagonal = diagonal + self.coupling_diagonal
        return diagonal

    def to_matrix(self) -> Tensor:
        num_x, num_y = self.tensor_shape
        dtype = self.axis_energies_x.dtype
        device = self.axis_energies_x.device
        identity_x = torch.eye(num_x, dtype=dtype, device=device)
        identity_y = torch.eye(num_y, dtype=dtype, device=device)
        operator_x = torch.diag(self.axis_energies_x)
        operator_y = torch.diag(self.axis_energies_y)
        matrix = torch.kron(operator_x, identity_y) + torch.kron(identity_x, operator_y)
        if self.coupling_diagonal is not None:
            matrix = matrix + torch.diag(self.coupling_diagonal)
        return matrix

    def lowest_states(self, num_states: int) -> tuple[Tensor, tuple[tuple[int, int], ...]]:
        if num_states <= 0:
            raise ValueError("num_states must be positive")
        diagonal = self.diagonal_entries()
        sorted_values, sorted_indices = torch.sort(diagonal)
        keep = min(int(num_states), int(sorted_values.shape[0]))
        sorted_values = sorted_values[:keep]
        sorted_indices = sorted_indices[:keep]
        state_pairs = tuple(self._unflatten_index(int(index.item())) for index in sorted_indices)
        return sorted_values.detach(), state_pairs

    def _unflatten_index(self, flat_index: int) -> tuple[int, int]:
        num_y = self.tensor_shape[1]
        return (int(flat_index // num_y), int(flat_index % num_y))


def make_tensor_axis_modes(
    axis_name: str,
    *,
    kind: str,
    requested_mode_count: int,
    basis_resolution_points: int,
    domain_interval: tuple[float, float],
    grid,
    eigenvalues,
    eigenstates,
) -> TensorAxisModes:
    return TensorAxisModes(
        axis_name=axis_name,
        kind=kind,
        requested_mode_count=requested_mode_count,
        basis_resolution_points=basis_resolution_points,
        domain_interval=domain_interval,
        grid=grid,
        eigenvalues=eigenvalues,
        eigenstates=eigenstates,
    )


def make_infinite_well_axis_modes(
    axis_name: str,
    *,
    domain_length: float = 1.0,
    left: float = 0.0,
    num_modes: int = 6,
    evaluation_grid_points: int | None = None,
    mass: float = 1.0,
    hbar: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> TensorAxisModes:
    if evaluation_grid_points is None:
        evaluation_grid_points = max(64, int(num_modes))
    if evaluation_grid_points < 2:
        raise ValueError("evaluation_grid_points must be at least 2")

    domain = InfiniteWell1D.from_length(
        domain_length,
        left=left,
        mass=mass,
        hbar=hbar,
        dtype=dtype,
        device=device,
    )
    basis = InfiniteWellBasis(domain, num_modes=num_modes)
    grid = domain.grid(evaluation_grid_points)
    eigenstates = basis.evaluate(grid).transpose(0, 1)
    return make_tensor_axis_modes(
        axis_name,
        kind="infinite-well",
        requested_mode_count=num_modes,
        basis_resolution_points=num_modes,
        domain_interval=(float(domain.left), float(domain.right)),
        grid=grid,
        eigenvalues=basis.energies,
        eigenstates=eigenstates,
    )


def summarize_tensor_product_basis_2d(basis: TensorProductBasis2D) -> TensorProductBasisSummary2D:
    return TensorProductBasisSummary2D(
        axis_names=(basis.axis_x.axis_name, basis.axis_y.axis_name),
        axis_kinds=(basis.axis_x.kind, basis.axis_y.kind),
        tensor_shape=basis.shape,
        total_mode_count=basis.total_mode_count,
        domain_intervals=(basis.axis_x.domain_interval, basis.axis_y.domain_interval),
        evaluation_grid_shape=(basis.axis_x.evaluation_grid_points, basis.axis_y.evaluation_grid_points),
        flattening_order=basis.flattening_order,
    )


def build_tensor_mode_budget_2d(
    basis: TensorProductBasis2D,
    *,
    requested_combined_state_count: int,
    retained_combined_state_count: int,
) -> TensorModeBudget2D:
    if requested_combined_state_count <= 0:
        raise ValueError("requested_combined_state_count must be positive")
    if retained_combined_state_count <= 0:
        raise ValueError("retained_combined_state_count must be positive")
    return TensorModeBudget2D(
        axis_x=TensorAxisModeBudget(
            axis_name=basis.axis_x.axis_name,
            kind=basis.axis_x.kind,
            requested_mode_count=basis.axis_x.requested_mode_count,
            retained_mode_count=basis.axis_x.retained_mode_count,
            basis_resolution_points=basis.axis_x.basis_resolution_points,
            truncated=basis.axis_x.retained_mode_count < basis.axis_x.requested_mode_count,
        ),
        axis_y=TensorAxisModeBudget(
            axis_name=basis.axis_y.axis_name,
            kind=basis.axis_y.kind,
            requested_mode_count=basis.axis_y.requested_mode_count,
            retained_mode_count=basis.axis_y.retained_mode_count,
            basis_resolution_points=basis.axis_y.basis_resolution_points,
            truncated=basis.axis_y.retained_mode_count < basis.axis_y.requested_mode_count,
        ),
        tensor_shape=basis.shape,
        total_tensor_mode_count=basis.total_mode_count,
        requested_combined_state_count=requested_combined_state_count,
        retained_combined_state_count=retained_combined_state_count,
    )


def summarize_tensor_truncation_2d(
    basis: TensorProductBasis2D,
    *,
    retained_combined_state_count: int,
    sorted_diagonal: Tensor,
) -> TensorTruncationDiagnostics2D:
    diagonal = _coerce_real_vector(
        sorted_diagonal,
        name="sorted_diagonal",
        dtype=basis.axis_x.eigenvalues.dtype,
        device=basis.axis_x.eigenvalues.device,
    )
    if retained_combined_state_count <= 0:
        raise ValueError("retained_combined_state_count must be positive")
    if retained_combined_state_count > basis.total_mode_count:
        raise ValueError("retained_combined_state_count cannot exceed the tensor basis size")
    discarded = int(basis.total_mode_count - retained_combined_state_count)
    first_discarded = None
    if discarded > 0:
        first_discarded = float(diagonal[retained_combined_state_count].item())
    return TensorTruncationDiagnostics2D(
        axis_truncation_applied=(
            basis.axis_x.retained_mode_count < basis.axis_x.requested_mode_count
            or basis.axis_y.retained_mode_count < basis.axis_y.requested_mode_count
        ),
        combined_state_truncation_applied=discarded > 0,
        discarded_tensor_mode_count=discarded,
        retained_tensor_fraction=float(retained_combined_state_count / basis.total_mode_count),
        retained_energy_ceiling=float(diagonal[retained_combined_state_count - 1].item()),
        first_discarded_eigenvalue=first_discarded,
    )


__all__ = [
    "KroneckerSumOperator2D",
    "TensorAxisModeBudget",
    "TensorAxisModes",
    "TensorModeBudget2D",
    "TensorProductBasis2D",
    "TensorProductBasisSummary2D",
    "TensorTruncationDiagnostics2D",
    "build_tensor_mode_budget_2d",
    "make_infinite_well_axis_modes",
    "make_tensor_axis_modes",
    "summarize_tensor_product_basis_2d",
    "summarize_tensor_truncation_2d",
]
