import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

from pulses import *

def chi_hamiltonian_simulation_pi2(
    H, state, device_params, exp_params, finite_pulses=False, pulse_params=None
):
    num_iter = exp_params["num_iter"]
    pulse_interval = np.linspace(0, exp_params["pulse_interval"], 20)
    # jump operators qubit-cavity
    Tphi = -1 / (1 / 2 / device_params["T1"] - 1 / device_params["T2"])
    c_ops = [
        # Qubit Relaxation
        np.sqrt(1 / device_params["T1"]) * Q,
        # Qubit Dephasing, changed
        np.sqrt(2 / Tphi) * Qd * Q,
        # Cavity Relaxation
        np.sqrt((1 + device_params["nbar_cav"]) / device_params["cavT1"]) * C,
        # Cavity Thermal Excitations
        np.sqrt(device_params["nbar_cav"] / device_params["cavT1"]) * Cd,
    ]
    for i in range(num_iter):
        if finite_pulses:            
            H_ry_pi2 = get_ry_pi2_hamiltonian(amp=pulse_params["amp"])
            
            H_pulse = [H, [H_ry_pi2, pulse_params["pulse"]]]
            psi_flip = qt.mesolve(H_pulse, state, pulse_params["timesteps"], c_ops).states[-1]

        else:
            psi_flip = Ry(np.pi / 2) * state * Ry(np.pi / 2).dag()
        results = qt.mesolve(H, psi_flip, pulse_interval, c_ops=c_ops)
        state = results.states[-1]
    return state



def kerr_hamiltonian_simulation(H, state, device_params, experiment_params):
    wait_timesteps = np.linspace(0, experiment_params["wait_time"], 50)
    # jump operators qubit-cavity
    Tphi = -1 / (1 / 2 / device_params["T1"] - 1 / device_params["T2"])
    c_ops = [
        # Qubit Relaxation
        np.sqrt(1 / device_params["T1"]) * Q,
        # Qubit Dephasing, changed
        np.sqrt(2 / Tphi) * Qd * Q,
        # Cavity Relaxation
        np.sqrt((1 + device_params["nbar_cav"]) / device_params["cavT1"]) * C,
        # Cavity Thermal Excitations
        np.sqrt(device_params["nbar_cav"] / device_params["cavT1"]) * Cd,
    ]
    results = qt.mesolve(
        H, state, wait_timesteps, options=qt.Options(nsteps=5000), c_ops=c_ops
    )
    state = results.states[-1]
    return state


def char_func_ideal_2d(state, xvec, scale):
    """Calculate the Characteristic function as a 2Dgrid (xvec, xvec) for a given state.

    Args:
        state (Qobject): State of which we want to calc the charfunc
        xvec (_type_): array of displacements. The char func will be calculated for the grid (xvec, xvec)

    Returns:
        tuple(ndarray, ndarray): Re(char func), Im(char func)
    """
    cfReal = np.empty((len(xvec), len(xvec)))
    cfImag = np.empty((len(xvec), len(xvec)))
    N = state.dims[0][1]
    # num_points = len(xvec) ** 2
    # curr_point = 0
    for i, alpha_x in enumerate(xvec):
        for j, alpha_p in enumerate(xvec):
            # print(f"calculating point {curr_point} of {num_points}")
            # curr_point += 1
            expect_value = qt.expect(
                qt.displace(N, alpha_x*scale + 1j * alpha_p*scale), qt.ptrace(state, 1)
            )
            cfReal[j, i] = np.real(expect_value)
            cfImag[j, i] = np.imag(expect_value)

    return cfReal, cfImag


def char_func_ideal_1d(state, xvec, axis, cut_point=0):
    """Calculate the Characteristic function as a 1D line xvec for a given state.

    Args:
        state (Qobject): State of which we want to calc the charfunc
        xvec (_type_): array of displacements. The char func will be calculated for the grid (xvec, xvec)
        axis (string): axis of which to make the 1d cut (either x or y).
        cut_point (int): the point on the axiss on which to make the 1d cut (defaults to 0).

    Returns:
        tuple(ndarray, ndarray): Re(char func), Im(char func)
    """
    cfReal = np.empty(len(xvec))
    cfImag = np.empty(len(xvec))
    N = state.dims[0][1]

    if axis == "x":
        for i, alpha_x in enumerate(xvec):
            expect_value = qt.expect(
                qt.displace(N, alpha_x + 1j * cut_point), qt.ptrace(state, 1)
            )
            cfReal[i] = np.real(expect_value)
            cfImag[i] = np.imag(expect_value)

    else:
        for j, alpha_p in enumerate(xvec):
            expect_value = qt.expect(
                qt.displace(N, cut_point + 1j * alpha_p), qt.ptrace(state, 1)
            )
            cfReal[j] = np.real(expect_value)
            cfImag[j] = np.imag(expect_value)

    return cfReal, cfImag
