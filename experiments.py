import numpy as np
import qutip as qt

from utils.operators import *

def chi_hamiltonian_simulation(
    H, state, device_params, exp_params, finite_pulses=False, pulse_params=None
):
    num_iter = exp_params["num_iter"]
    pulse_interval = np.linspace(0, exp_params["pulse_interval"], 100)
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
    
    # Define pulse hamiltonian
    amp = pulse_params["rabi_freq"]*exp_params["amp_scale"]
    H_ry_pi = 2 * np.pi * amp * 1j * (Qd - Q)
    # Include pulse hamiltonian in the system hamiltonian
    H_pulse = [H, [H_ry_pi, pulse_params["pulse"]]]
    
    # Simulate pulses and wait time
    for i in range(num_iter):
        if finite_pulses:
            psi_flip = qt.mesolve(H_pulse, state, pulse_params["timesteps"], c_ops).states[-1]

        else:
            psi_flip = Ry(np.pi / 2) * state * Ry(np.pi / 2).dag()
        results = qt.mesolve(H, psi_flip, pulse_interval, c_ops=c_ops)
        state = results.states[-1]
        
    return state

def kerr_hamiltonian_simulation(H, state, device_params, experiment_params):
    wait_timesteps = np.linspace(0, experiment_params["wait_time"], 1000)
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

