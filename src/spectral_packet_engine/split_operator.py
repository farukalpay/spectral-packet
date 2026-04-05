"""Split-operator propagation for the time-dependent Schrodinger equation.

Implements symmetric Trotter (2nd order) and Forest-Ruth (4th order)
split-operator time stepping for the Hamiltonian

    H = T + V = -hbar^2/(2m) d^2/dx^2 + V(x)

on a bounded domain with Dirichlet (infinite-well) boundary conditions.

Because PyTorch lacks a built-in discrete sine transform, the kinetic
propagation is performed via explicit matrix multiplication with the
sine basis matrix:

    position space  --S-->  spectral coefficients  --exp(-i T dt)-->  --S^{-1}-->  position space

where S is the sine basis matrix and T_n = (n pi hbar)^2 / (2 m L^2) are
the mode energies.

A single second-order Trotter step reads

    psi(t + dt) = e^{-i V dt/2}  S^{-1}  e^{-i T dt}  S  e^{-i V dt/2}  psi(t)

and the fourth-order Forest-Ruth integrator composes three such steps
with appropriately scaled time increments.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from spectral_packet_engine.basis import InfiniteWellBasis, eigenenergies, sine_basis_matrix
from spectral_packet_engine.domain import InfiniteWell1D, coerce_tensor, complex_dtype_for

Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# Configuration and result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SplitOperatorConfig:
    """Configuration knobs for the split-operator propagator."""
    num_steps: int = 1000
    order: int = 2  # Trotter splitting order: 2 or 4


@dataclass(frozen=True, slots=True)
class PropagationResult:
    """Output of a split-operator time propagation.

    Attributes
    ----------
    times : Tensor
        Shape ``(num_saved,)`` -- the time stamps at which snapshots were
        recorded.
    wavefunctions : Tensor
        Shape ``(num_saved, num_points)`` -- complex wavefunction snapshots.
    densities : Tensor
        Shape ``(num_saved, num_points)`` -- probability densities
        |psi(x,t)|^2.
    norm_history : Tensor
        Shape ``(num_saved,)`` -- total norm at each saved time (should
        remain close to 1 for unitary propagation).
    energy_history : Tensor
        Shape ``(num_saved,)`` -- expectation value of H at each saved time.
    grid : Tensor
        Shape ``(num_points,)`` -- the spatial grid.
    """
    times: Tensor
    wavefunctions: Tensor
    densities: Tensor
    norm_history: Tensor
    energy_history: Tensor
    grid: Tensor


# ---------------------------------------------------------------------------
# Gaussian wavepacket initialiser
# ---------------------------------------------------------------------------

def gaussian_wavepacket_on_grid(
    grid: Tensor,
    center: float,
    width: float,
    wavenumber: float,
    *,
    hbar: float = 1.0,
) -> Tensor:
    r"""Create a normalised Gaussian wavepacket on *grid*.

    .. math::
        \psi(x) = (2\pi\sigma^2)^{-1/4}
                   \exp\!\Bigl(-\frac{(x - x_0)^2}{4\sigma^2}
                               + i k_0 x\Bigr)

    Parameters
    ----------
    grid : Tensor
        Spatial grid points.
    center : float
        Centre position x_0.
    width : float
        Gaussian width sigma.
    wavenumber : float
        Central wavenumber k_0.
    hbar : float
        Reduced Planck constant (only enters the docstring context;
        the spatial phase is exp(i k_0 x) by convention).

    Returns
    -------
    Tensor
        Complex wavepacket values on *grid*.
    """
    grid = coerce_tensor(grid, dtype=torch.float64)
    cdtype = complex_dtype_for(grid.dtype)

    norm_factor = (2.0 * torch.pi * width ** 2) ** (-0.25)
    envelope = torch.exp(-((grid - center) ** 2) / (4.0 * width ** 2))
    phase = torch.exp(1j * wavenumber * grid.to(cdtype))

    psi = norm_factor * envelope.to(cdtype) * phase
    return psi


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_sine_transform_matrices(
    domain: InfiniteWell1D,
    grid: Tensor,
    num_modes: int,
) -> tuple[Tensor, Tensor]:
    """Build the forward (position -> spectral) and inverse sine basis matrices.

    The forward transform is the pseudo-inverse of the basis matrix B
    (shape num_points x num_modes).  For an interior grid (excluding boundary
    nodes where the sine basis vanishes), the least-squares solution is
    used via the normal equations.

    Returns (S_fwd, S_inv) where:
        coeffs = S_fwd @ psi       (num_modes,)
        psi    = S_inv @ coeffs    (num_points,)
    """
    modes = torch.arange(1, num_modes + 1, dtype=domain.real_dtype, device=domain.device)
    # B has shape (num_points, num_modes)
    B = sine_basis_matrix(domain, modes, grid)

    # Forward transform: pseudo-inverse via B^T B (orthonormal in the
    # continuous limit, but for a discrete grid we use lstsq).
    # For well-resolved grids this is numerically stable.
    cdtype = complex_dtype_for(domain.real_dtype)
    B_c = B.to(cdtype)
    S_inv = B_c  # (num_points, num_modes)
    # S_fwd via least-squares: S_fwd = pinv(B)
    S_fwd = torch.linalg.pinv(B_c)  # (num_modes, num_points)

    return S_fwd, S_inv


def _compute_energy_expectation(
    psi: Tensor,
    potential: Tensor,
    grid: Tensor,
    S_fwd: Tensor,
    S_inv: Tensor,
    mode_energies: Tensor,
) -> Tensor:
    """Evaluate <H> = <T> + <V> for a wavefunction snapshot."""
    cdtype = psi.dtype

    # <V> = integral psi* V psi dx  (trapezoidal)
    density = (psi.conj() * psi).real
    V_expect = torch.trapezoid((psi.conj() * potential.to(cdtype) * psi).real, grid)

    # <T> = sum_n |c_n|^2 E_n
    coeffs = S_fwd @ psi
    T_expect = torch.sum((coeffs.conj() * coeffs).real * mode_energies.to(coeffs.real.dtype))

    return T_expect + V_expect


def _trotter2_step(
    psi: Tensor,
    V_half: Tensor,
    kinetic_phase: Tensor,
    S_fwd: Tensor,
    S_inv: Tensor,
) -> Tensor:
    """One symmetric (second-order) Trotter step.

    psi(t+dt) = exp(-i V dt/2)  S^{-1}  exp(-i T dt)  S  exp(-i V dt/2) psi(t)
    """
    # Half potential kick
    psi = V_half * psi
    # Forward transform to spectral space
    coeffs = S_fwd @ psi
    # Full kinetic kick in spectral space
    coeffs = kinetic_phase * coeffs
    # Inverse transform back to position space
    psi = S_inv @ coeffs
    # Half potential kick
    psi = V_half * psi
    return psi


# ---------------------------------------------------------------------------
# Main propagator
# ---------------------------------------------------------------------------

def split_operator_propagate(
    psi_initial: Tensor,
    potential: Tensor,
    domain: InfiniteWell1D,
    *,
    total_time: float,
    num_steps: int = 1000,
    save_every: int = 10,
    order: int = 2,
) -> PropagationResult:
    """Propagate a wavefunction under H = T + V using the split-operator method.

    Parameters
    ----------
    psi_initial : Tensor
        Shape ``(num_points,)`` -- initial complex wavefunction on a
        uniform spatial grid (excluding boundary nodes where psi = 0,
        or including them -- the grid is inferred from the domain).
    potential : Tensor
        Shape ``(num_points,)`` -- the potential V(x) evaluated on the
        same grid.
    domain : InfiniteWell1D
        The spatial domain with mass and hbar.
    total_time : float
        Total propagation time.
    num_steps : int
        Number of time steps.
    save_every : int
        Save a snapshot every *save_every* steps.
    order : int
        Splitting order: 2 (symmetric Trotter) or 4 (Forest-Ruth).

    Returns
    -------
    PropagationResult
    """
    if order not in (2, 4):
        raise ValueError("order must be 2 or 4")
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    if save_every < 1:
        raise ValueError("save_every must be at least 1")

    dtype = domain.real_dtype
    device = domain.device
    cdtype = complex_dtype_for(dtype)

    num_points = psi_initial.shape[0]
    grid = domain.grid(num_points)
    dt = total_time / num_steps
    hbar = domain.hbar.item()

    psi = coerce_tensor(psi_initial, dtype=cdtype, device=device).clone()
    V = coerce_tensor(potential, dtype=dtype, device=device)

    # Number of sine modes: use num_points - 2 interior modes (Dirichlet BC)
    # but cap at num_points to avoid under-determination.
    num_modes = max(num_points - 2, 1)

    # Build transform matrices and mode energies
    S_fwd, S_inv = _build_sine_transform_matrices(domain, grid, num_modes)
    modes = torch.arange(1, num_modes + 1, dtype=dtype, device=device)
    E_n = eigenenergies(domain, modes)  # (num_modes,)

    # Precompute phase operators for second-order Trotter
    if order == 2:
        V_half = torch.exp(-1j * V.to(cdtype) * dt / (2.0 * hbar))
        kinetic_phase = torch.exp(-1j * E_n.to(cdtype) * dt / hbar)
    # For 4th order Forest-Ruth, we build phases on the fly with scaled dt.

    # Forest-Ruth coefficients:
    # Three second-order steps with time increments c1*dt, c2*dt, c1*dt
    # where c1 = 1/(2 - 2^{1/3}),  c2 = 1 - 2*c1
    if order == 4:
        theta = 2.0 ** (1.0 / 3.0)
        c1 = 1.0 / (2.0 - theta)
        c2 = 1.0 - 2.0 * c1
        sub_dts = [c1 * dt, c2 * dt, c1 * dt]
        V_halfs = [torch.exp(-1j * V.to(cdtype) * sdt / (2.0 * hbar)) for sdt in sub_dts]
        kinetic_phases = [torch.exp(-1j * E_n.to(cdtype) * sdt / hbar) for sdt in sub_dts]

    # Storage for snapshots
    num_saved = num_steps // save_every + 1
    wavefunctions = torch.zeros(num_saved, num_points, dtype=cdtype, device=device)
    densities = torch.zeros(num_saved, num_points, dtype=dtype, device=device)
    norm_history = torch.zeros(num_saved, dtype=dtype, device=device)
    energy_history = torch.zeros(num_saved, dtype=dtype, device=device)
    times = torch.zeros(num_saved, dtype=dtype, device=device)

    # Save initial state
    save_idx = 0
    wavefunctions[0] = psi
    densities[0] = (psi.conj() * psi).real
    norm_history[0] = torch.trapezoid(densities[0], grid)
    energy_history[0] = _compute_energy_expectation(psi, V, grid, S_fwd, S_inv, E_n)
    times[0] = 0.0
    save_idx = 1

    # Time stepping
    for step in range(1, num_steps + 1):
        if order == 2:
            psi = _trotter2_step(psi, V_half, kinetic_phase, S_fwd, S_inv)
        else:
            # 4th-order: three Trotter-2 sub-steps
            for k in range(3):
                psi = _trotter2_step(psi, V_halfs[k], kinetic_phases[k], S_fwd, S_inv)

        if step % save_every == 0 and save_idx < num_saved:
            wavefunctions[save_idx] = psi
            dens = (psi.conj() * psi).real
            densities[save_idx] = dens
            norm_history[save_idx] = torch.trapezoid(dens, grid)
            energy_history[save_idx] = _compute_energy_expectation(psi, V, grid, S_fwd, S_inv, E_n)
            times[save_idx] = step * dt
            save_idx += 1

    # Trim if fewer snapshots were taken than allocated
    wavefunctions = wavefunctions[:save_idx]
    densities = densities[:save_idx]
    norm_history = norm_history[:save_idx]
    energy_history = energy_history[:save_idx]
    times = times[:save_idx]

    return PropagationResult(
        times=times,
        wavefunctions=wavefunctions,
        densities=densities,
        norm_history=norm_history,
        energy_history=energy_history,
        grid=grid,
    )


__all__ = [
    "PropagationResult",
    "SplitOperatorConfig",
    "gaussian_wavepacket_on_grid",
    "split_operator_propagate",
]
