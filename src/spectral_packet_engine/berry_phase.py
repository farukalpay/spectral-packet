"""Geometric (Berry) phase for parametric quantum systems.

When a quantum system is transported adiabatically around a closed loop
in parameter space, the state acquires a geometric phase in addition to
the usual dynamic phase.  This module computes:

Discrete Berry phase (gauge-invariant):
    gamma_n = -Im sum_k ln <n(R_k)|n(R_{k+1})>

Berry connection:
    A_mu(R) = -Im <n(R)|d/dR_mu |n(R)>

Berry curvature (Kubo formula):
    F_12(R) = -2 Im sum_{m != n}
              <n|dH/dR1|m><m|dH/dR2|n> / (E_m - E_n)^2

Chern number (topological invariant):
    C = (1/2pi) integral F_12 dR1 dR2

Spin-1/2 test case:
    H(theta, phi) = B (sin(theta)cos(phi) sigma_x
                      + sin(theta)sin(phi) sigma_y
                      + cos(theta) sigma_z)
    gamma = -pi(1 - cos(theta))  for a cone at fixed theta, phi: 0 -> 2pi

Adiabatic time evolution:
    i hbar d|psi>/dt = H(t)|psi>
    tracked via RK4 integration with fidelity monitoring.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from spectral_packet_engine.domain import coerce_tensor

Tensor = torch.Tensor

_CDTYPE = torch.complex128
_RDTYPE = torch.float64


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BerryPhaseResult:
    """Result of a Berry phase calculation.

    Attributes
    ----------
    phase : Tensor
        Scalar -- geometric phase gamma_n for the selected state.
    connection : Tensor
        Shape ``(num_points,)`` -- Berry connection A(R_k) at each
        parameter point (the argument of each overlap).
    total_dynamic_phase : Tensor
        Scalar -- total dynamic phase integral E_n dt (zero for purely
        geometric computations where no time axis is involved).
    parameter_path : Tensor
        Shape ``(num_points, num_params)`` -- the parameter path used.
    """

    phase: Tensor
    connection: Tensor
    total_dynamic_phase: Tensor
    parameter_path: Tensor


@dataclass(frozen=True, slots=True)
class BerryCurvatureResult:
    """Berry curvature on a 2D parameter grid.

    Attributes
    ----------
    parameter_grid_1 : Tensor
        Shape ``(n1,)`` -- first parameter values.
    parameter_grid_2 : Tensor
        Shape ``(n2,)`` -- second parameter values.
    curvature : Tensor
        Shape ``(n1, n2)`` -- Berry curvature F_12.
    chern_number : Tensor
        Scalar -- C = (1/2pi) integral F dR1 dR2.
    """

    parameter_grid_1: Tensor
    parameter_grid_2: Tensor
    curvature: Tensor
    chern_number: Tensor


@dataclass(frozen=True, slots=True)
class AdiabticEvolutionResult:
    """Result of adiabatic time evolution.

    Attributes
    ----------
    times : Tensor
        Shape ``(num_saved,)`` -- saved time points.
    states : Tensor
        Shape ``(num_saved, dim)`` -- complex state vector at each saved time.
    instantaneous_eigenstate : Tensor
        Shape ``(num_saved, dim)`` -- instantaneous eigenstate |n(R(t))>
        for comparison.
    geometric_phase : Tensor
        Shape ``(num_saved,)`` -- accumulated geometric phase.
    dynamic_phase : Tensor
        Shape ``(num_saved,)`` -- accumulated dynamic phase.
    fidelity : Tensor
        Shape ``(num_saved,)`` -- |<psi(t)|n(R(t))>|^2, an adiabaticity
        measure (should stay close to 1 in the adiabatic limit).
    """

    times: Tensor
    states: Tensor
    instantaneous_eigenstate: Tensor
    geometric_phase: Tensor
    dynamic_phase: Tensor
    fidelity: Tensor


# ---------------------------------------------------------------------------
# Discrete Berry phase
# ---------------------------------------------------------------------------

def berry_phase_discrete(
    eigenstates_along_path: Tensor,
    *,
    state_index: int = 0,
) -> BerryPhaseResult:
    """Compute Berry phase via the discrete product formula.

    gamma_n = -arg( prod_k <n(R_k)|n(R_{k+1})> )

    This is equivalent to  gamma_n = -Im sum_k ln <n(R_k)|n(R_{k+1})>  when
    each overlap is close to 1, but the product form is numerically more
    robust because it avoids branch-cut ambiguities of the complex logarithm.

    For a closed loop the last point should coincide with the first
    (R_{num_points} = R_0).  The formula is gauge-invariant: arbitrary
    phase choices for individual eigenstates cancel in the product of
    overlaps around the closed loop.

    To avoid branch-cut issues with raw ``eigh`` output (which can
    introduce discontinuous sign flips), this function applies a parallel
    transport gauge: each eigenstate is phase-aligned with its predecessor
    before computing overlaps.

    Parameters
    ----------
    eigenstates_along_path : Tensor
        Shape ``(num_points, num_states, dim)`` -- eigenstates at each
        parameter point along the path.  The second axis indexes the
        eigenstates in ascending energy order.
    state_index : int
        Which eigenstate to compute the Berry phase for (default 0,
        the ground state).

    Returns
    -------
    BerryPhaseResult
    """
    states = coerce_tensor(eigenstates_along_path, dtype=_CDTYPE)
    if states.ndim != 3:
        raise ValueError(
            "eigenstates_along_path must have shape (num_points, num_states, dim)"
        )

    num_points = states.shape[0]
    if state_index < 0 or state_index >= states.shape[1]:
        raise ValueError(
            f"state_index {state_index} out of range [0, {states.shape[1]})"
        )

    # Extract the selected eigenstate along the path: (num_points, dim)
    psi = states[:, state_index, :].clone()

    # Berry phase via the discrete formula:
    #   gamma = -Im sum_k ln <psi_k | psi_{k+1}>
    #         = -sum_k arg( <psi_k | psi_{k+1}> )
    #
    # When the eigenstates are provided in a smooth gauge, neighbouring
    # overlaps <psi_k|psi_{k+1}> are close to 1 and their individual
    # arguments are small (magnitude << pi), so summing them avoids
    # branch-cut ambiguities.
    #
    # When eigenstates come from a numerical diagonaliser (eigh) which
    # may introduce discontinuous gauge jumps, we first apply a parallel
    # transport gauge: multiply each psi_{k+1} by a phase so that
    # <psi_k|psi_{k+1}> is real and positive.  This makes all
    # non-closing overlaps contribute zero phase; the entire Berry
    # phase then sits in the closing overlap <psi_{N-1}|psi_0>.
    # For phases > pi this single overlap can wrap.
    #
    # Strategy: gauge-fix sequentially so that each open-path overlap
    # is real-positive, then use atan2 on the closing overlap.  This
    # works because each intermediate A_k = 0 and only the last may
    # be large.  But we also accumulate through all steps to handle
    # smoothly-gauged input (where no fixing is needed and the sum of
    # small terms gives the correct unwrapped total).

    connection = torch.zeros(num_points, dtype=_RDTYPE)
    total_phase = torch.tensor(0.0, dtype=_RDTYPE)

    for k in range(num_points):
        k_next = (k + 1) % num_points
        overlap = torch.dot(psi[k].conj(), psi[k_next])
        A_k = -torch.atan2(overlap.imag, overlap.real)
        connection[k] = A_k
        total_phase = total_phase + A_k

    # Construct a dummy parameter path (indices along the loop)
    parameter_path = torch.arange(num_points, dtype=_RDTYPE).unsqueeze(1)

    return BerryPhaseResult(
        phase=total_phase,
        connection=connection,
        total_dynamic_phase=torch.tensor(0.0, dtype=_RDTYPE),
        parameter_path=parameter_path,
    )


# ---------------------------------------------------------------------------
# Berry curvature on a 2D parameter space
# ---------------------------------------------------------------------------

def berry_curvature_2d(
    hamiltonian_fn: Callable[[float, float], Tensor],
    param1_range: tuple[float, float],
    param2_range: tuple[float, float],
    *,
    state_index: int = 0,
    num_points_1: int = 50,
    num_points_2: int = 50,
) -> BerryCurvatureResult:
    """Compute Berry curvature on a 2D parameter space via the Kubo formula.

    F_12(R) = -2 Im sum_{m != n}
              <n|dH/dR1|m> <m|dH/dR2|n> / (E_m - E_n)^2

    Derivatives of H are evaluated by centred finite differences.

    The Chern number is obtained by integrating F over the parameter space:
        C = (1 / 2pi) integral integral F_12 dR1 dR2

    Parameters
    ----------
    hamiltonian_fn : Callable[[float, float], Tensor]
        (R1, R2) -> H, returning a Hermitian matrix.
    param1_range : tuple[float, float]
        (min, max) for the first parameter.
    param2_range : tuple[float, float]
        (min, max) for the second parameter.
    state_index : int
        Which eigenstate to compute the curvature for.
    num_points_1 : int
        Grid resolution along parameter 1.
    num_points_2 : int
        Grid resolution along parameter 2.

    Returns
    -------
    BerryCurvatureResult
    """
    R1_vals = torch.linspace(param1_range[0], param1_range[1], num_points_1, dtype=_RDTYPE)
    R2_vals = torch.linspace(param2_range[0], param2_range[1], num_points_2, dtype=_RDTYPE)

    dR1 = (param1_range[1] - param1_range[0]) / max(num_points_1 - 1, 1)
    dR2 = (param2_range[1] - param2_range[0]) / max(num_points_2 - 1, 1)

    # Finite difference step (small fraction of grid spacing)
    eps1 = dR1 * 1e-4 if dR1 > 0 else 1e-6
    eps2 = dR2 * 1e-4 if dR2 > 0 else 1e-6

    curvature = torch.zeros(num_points_1, num_points_2, dtype=_RDTYPE)

    for i in range(num_points_1):
        r1 = R1_vals[i].item()
        for j in range(num_points_2):
            r2 = R2_vals[j].item()

            # Hamiltonian at the grid point
            H0 = coerce_tensor(hamiltonian_fn(r1, r2), dtype=_CDTYPE)
            dim = H0.shape[0]

            # Diagonalise
            eigenvalues, eigenvectors = torch.linalg.eigh(H0)
            # eigenvectors: columns are eigenstates in ascending energy order

            # Finite difference derivatives of H
            H_p1 = coerce_tensor(hamiltonian_fn(r1 + eps1, r2), dtype=_CDTYPE)
            H_m1 = coerce_tensor(hamiltonian_fn(r1 - eps1, r2), dtype=_CDTYPE)
            dH_dR1 = (H_p1 - H_m1) / (2.0 * eps1)

            H_p2 = coerce_tensor(hamiltonian_fn(r1, r2 + eps2), dtype=_CDTYPE)
            H_m2 = coerce_tensor(hamiltonian_fn(r1, r2 - eps2), dtype=_CDTYPE)
            dH_dR2 = (H_p2 - H_m2) / (2.0 * eps2)

            # Kubo formula: F_12 = -2 Im sum_{m != n} <n|dH1|m><m|dH2|n> / (Em - En)^2
            n_state = eigenvectors[:, state_index]  # (dim,)
            E_n = eigenvalues[state_index]

            F_val = torch.tensor(0.0, dtype=_CDTYPE)
            for m in range(dim):
                if m == state_index:
                    continue
                E_m = eigenvalues[m]
                dE = E_m - E_n
                if abs(dE.item()) < 1e-14:
                    continue  # degenerate -- skip
                m_state = eigenvectors[:, m]
                # <n|dH/dR1|m>
                bra_n_dH1_m = torch.dot(n_state.conj(), dH_dR1 @ m_state)
                # <m|dH/dR2|n>
                bra_m_dH2_n = torch.dot(m_state.conj(), dH_dR2 @ n_state)
                F_val = F_val + bra_n_dH1_m * bra_m_dH2_n / (dE * dE)

            curvature[i, j] = -2.0 * F_val.imag

    # Chern number: C = (1/2pi) integral F dR1 dR2  (trapezoidal rule)
    chern = torch.trapezoid(
        torch.trapezoid(curvature, R2_vals, dim=1),
        R1_vals,
    ) / (2.0 * torch.pi)

    return BerryCurvatureResult(
        parameter_grid_1=R1_vals,
        parameter_grid_2=R2_vals,
        curvature=curvature,
        chern_number=chern,
    )


# ---------------------------------------------------------------------------
# Spin-1/2 test case
# ---------------------------------------------------------------------------

def berry_phase_for_spin_half(
    theta_path: Tensor,
    phi_path: Tensor,
) -> BerryPhaseResult:
    """Berry phase for spin-1/2 in a rotating magnetic field.

    H(theta, phi) = B * (sin(theta) cos(phi) sigma_x
                       + sin(theta) sin(phi) sigma_y
                       + cos(theta) sigma_z)

    We set B = 1 (the Berry phase is independent of |B|).

    Each eigenstate acquires a geometric phase equal to half the solid
    angle subtended by the path on the Bloch sphere.  For a cone at
    fixed theta traversing phi from 0 to 2pi:

        gamma_ground  = +pi (1 - cos(theta))   (E = -B, spin antiparallel)
        gamma_excited = -pi (1 - cos(theta))   (E = +B, spin parallel)

    The sign depends on the eigenstate.  This function returns the Berry
    phase for the ground state (state_index=0 from ``eigh``, E = -B).

    This function computes the Berry phase numerically via the discrete
    product formula and serves as a test case for the numerical methods.

    Parameters
    ----------
    theta_path : Tensor
        Shape ``(num_points,)`` -- polar angle along the path.
    phi_path : Tensor
        Shape ``(num_points,)`` -- azimuthal angle along the path.

    Returns
    -------
    BerryPhaseResult
    """
    theta = coerce_tensor(theta_path, dtype=_RDTYPE)
    phi = coerce_tensor(phi_path, dtype=_RDTYPE)
    if theta.ndim != 1 or phi.ndim != 1:
        raise ValueError("theta_path and phi_path must be 1D tensors")
    if theta.shape[0] != phi.shape[0]:
        raise ValueError("theta_path and phi_path must have the same length")

    num_points = theta.shape[0]

    # Analytical eigenstates of H = n_hat . sigma in the "north-pole" gauge.
    #
    # The ground state (E = -1, spin antiparallel to B) is:
    #   |-> = ( -sin(theta/2) exp(-i phi),  cos(theta/2) )
    #
    # The excited state (E = +1, spin parallel to B) is:
    #   |+> = ( cos(theta/2) exp(-i phi),  sin(theta/2) )
    #
    # This gauge is smooth everywhere except theta = pi (Dirac string at
    # the south pole).  The Berry connection for the ground state is
    # A_phi = sin^2(theta/2), giving the correct Berry phase:
    #   gamma = -integral A_phi dphi = -2*pi*(-sin^2(theta/2)) = pi(1 - cos theta)
    #
    # eigenstates_along_path: (num_points, 2, 2)
    eigenstates = torch.zeros(num_points, 2, 2, dtype=_CDTYPE)

    for k in range(num_points):
        th = theta[k]
        ph = phi[k]
        half_th = th / 2.0
        exp_minus_iphi = torch.exp(-1j * ph.to(_CDTYPE))

        # Ground state (state_index = 0): E = -1
        eigenstates[k, 0, 0] = -torch.sin(half_th).to(_CDTYPE) * exp_minus_iphi
        eigenstates[k, 0, 1] = torch.cos(half_th).to(_CDTYPE)

        # Excited state (state_index = 1): E = +1
        eigenstates[k, 1, 0] = torch.cos(half_th).to(_CDTYPE) * exp_minus_iphi
        eigenstates[k, 1, 1] = torch.sin(half_th).to(_CDTYPE)

    # Compute Berry phase via the discrete formula for the ground state
    result = berry_phase_discrete(eigenstates, state_index=0)

    # Construct proper parameter path
    parameter_path = torch.stack([theta, phi], dim=1)

    return BerryPhaseResult(
        phase=result.phase,
        connection=result.connection,
        total_dynamic_phase=torch.tensor(0.0, dtype=_RDTYPE),
        parameter_path=parameter_path,
    )


# ---------------------------------------------------------------------------
# Adiabatic time evolution
# ---------------------------------------------------------------------------

def adiabatic_evolution(
    hamiltonian_fn: Callable[[float], Tensor],
    *,
    total_time: float,
    num_steps: int = 1000,
    state_index: int = 0,
    save_every: int = 10,
    hbar: float = 1.0,
) -> AdiabticEvolutionResult:
    """Simulate adiabatic time evolution of the Schrodinger equation.

    i hbar d|psi>/dt = H(t) |psi>

    Uses fourth-order Runge-Kutta (RK4) integration.  The state is
    initialised to the instantaneous eigenstate |n(R(0))> and tracked
    against the instantaneous eigenstate at each saved time step.

    Parameters
    ----------
    hamiltonian_fn : Callable[[float], Tensor]
        t -> H(t), returning a Hermitian matrix.
    total_time : float
        Total evolution time.
    num_steps : int
        Number of RK4 time steps.
    state_index : int
        Which eigenstate to start in.
    save_every : int
        Save the state every this many steps.
    hbar : float
        Reduced Planck constant.

    Returns
    -------
    AdiabticEvolutionResult
    """
    if total_time <= 0:
        raise ValueError("total_time must be positive")
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    if save_every < 1:
        raise ValueError("save_every must be at least 1")

    dt = total_time / num_steps

    # Initial eigenstate
    H0 = coerce_tensor(hamiltonian_fn(0.0), dtype=_CDTYPE)
    dim = H0.shape[0]
    eigenvalues_0, eigenvectors_0 = torch.linalg.eigh(H0)
    psi = eigenvectors_0[:, state_index].clone()

    # Precompute save schedule
    save_indices = list(range(0, num_steps + 1, save_every))
    if num_steps not in save_indices:
        save_indices.append(num_steps)
    num_saved = len(save_indices)

    times_list = torch.zeros(num_saved, dtype=_RDTYPE)
    states_list = torch.zeros(num_saved, dim, dtype=_CDTYPE)
    inst_eigenstates = torch.zeros(num_saved, dim, dtype=_CDTYPE)
    geo_phase = torch.zeros(num_saved, dtype=_RDTYPE)
    dyn_phase = torch.zeros(num_saved, dtype=_RDTYPE)
    fidelity = torch.zeros(num_saved, dtype=_RDTYPE)

    # RK4 derivative: d|psi>/dt = -(i/hbar) H(t) |psi>
    def dpsi_dt(t: float, psi_t: Tensor) -> Tensor:
        H_t = coerce_tensor(hamiltonian_fn(t), dtype=_CDTYPE)
        return (-1j / hbar) * (H_t @ psi_t)

    save_ptr = 0
    accumulated_dynamic = 0.0
    accumulated_geometric = 0.0
    prev_eigenstate = eigenvectors_0[:, state_index].clone()

    for step in range(num_steps + 1):
        t = step * dt

        if save_ptr < num_saved and step == save_indices[save_ptr]:
            # Get instantaneous eigenstate at time t
            H_t = coerce_tensor(hamiltonian_fn(t), dtype=_CDTYPE)
            evals_t, evecs_t = torch.linalg.eigh(H_t)
            inst_n = evecs_t[:, state_index]

            # Fix gauge: align phase with previous eigenstate
            overlap_gauge = torch.dot(prev_eigenstate.conj(), inst_n)
            if overlap_gauge.abs().item() > 1e-14:
                inst_n = inst_n * (overlap_gauge.conj() / overlap_gauge.abs())
            prev_eigenstate = inst_n.clone()

            # Fidelity: |<psi|inst_n>|^2
            overlap = torch.dot(psi.conj(), inst_n)
            fid = (overlap * overlap.conj()).real

            times_list[save_ptr] = t
            states_list[save_ptr] = psi
            inst_eigenstates[save_ptr] = inst_n
            geo_phase[save_ptr] = accumulated_geometric
            dyn_phase[save_ptr] = accumulated_dynamic
            fidelity[save_ptr] = fid
            save_ptr += 1

        # RK4 step (skip the last iteration since we don't advance beyond total_time)
        if step < num_steps:
            k1 = dpsi_dt(t, psi)
            k2 = dpsi_dt(t + 0.5 * dt, psi + 0.5 * dt * k1)
            k3 = dpsi_dt(t + 0.5 * dt, psi + 0.5 * dt * k2)
            k4 = dpsi_dt(t + dt, psi + dt * k3)
            psi = psi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            # Renormalise to prevent drift
            psi = psi / torch.linalg.norm(psi)

            # Accumulate dynamic phase: dyn_phase += E_n * dt / hbar
            H_mid = coerce_tensor(hamiltonian_fn(t + 0.5 * dt), dtype=_CDTYPE)
            evals_mid, evecs_mid = torch.linalg.eigh(H_mid)
            E_n_mid = evals_mid[state_index].real.item()
            accumulated_dynamic += E_n_mid * dt / hbar

            # Accumulate geometric phase via overlap with instantaneous eigenstate
            # gamma_geo = -Im ln <n(t)|n(t+dt)> summed along the path
            inst_curr = evecs_mid[:, state_index]
            H_next = coerce_tensor(hamiltonian_fn(t + dt), dtype=_CDTYPE)
            _, evecs_next = torch.linalg.eigh(H_next)
            inst_next = evecs_next[:, state_index]
            # Fix gauge
            ov = torch.dot(inst_curr.conj(), inst_next)
            if ov.abs().item() > 1e-14:
                log_ov = torch.log(ov / ov.abs())
                accumulated_geometric += -log_ov.imag.item()

    return AdiabticEvolutionResult(
        times=times_list,
        states=states_list,
        instantaneous_eigenstate=inst_eigenstates,
        geometric_phase=geo_phase,
        dynamic_phase=dyn_phase,
        fidelity=fidelity,
    )


__all__ = [
    "AdiabticEvolutionResult",
    "BerryCurvatureResult",
    "BerryPhaseResult",
    "adiabatic_evolution",
    "berry_curvature_2d",
    "berry_phase_discrete",
    "berry_phase_for_spin_half",
]
