import numpy as np
import qutip as qt
from qutip.qip.operations import rx, ry, rz

#######################################
########## System Operators ###########
#######################################

# for cavity
cdim = 100
c = qt.destroy(cdim)
u_1 = (qt.basis(cdim, 0) + 0 * 1j * qt.basis(cdim, 1)).unit()
rho = u_1 * u_1.dag()  # the to-be measured state

# for qubit
qdim = 2
u_g = qt.basis(qdim, 0)
u_e = qt.basis(qdim, 1)
q = u_g * u_e.dag()
qd = q.dag()

# qubit-cavity
Q = qt.tensor(qt.destroy(qdim), qt.qeye(cdim))
C = qt.tensor(qt.qeye(qdim), qt.destroy(cdim))
Cd, Qd = C.dag(), Q.dag()

# Identity Operators
Ic = qt.qeye(cdim)
Iq = qt.qeye(2)

sz = qt.tensor(qt.sigmaz(), Ic)
sy = qt.tensor(qt.sigmay(), Ic)

# Projector for qubit in g
Pg = qt.tensor(u_g * u_g.dag(), Ic)


def Rx(theta):
    return qt.tensor(rx(theta), Ic)


def Ry(theta):
    return qt.tensor(ry(theta), Ic)


def Rz(theta):
    return qt.tensor(rz(theta), Ic)


def get_dispersive_hamiltonian(device_params):
    chi = device_params["chi"]
    anharm = device_params["anharm"]
    H_disp = -2 * np.pi * chi * Cd * C * Qd * Q - 2 * np.pi * anharm / 2 * Qd * Qd * Q * Q #-2 * np.pi * chi * Cd * C * Qd * Q
    return H_disp


def get_dispersive_hamiltonian_with_kerr(device_params):
    chi = device_params["chi"]
    kappa = device_params["anharm"]
    kerr_override = device_params["kerr"]
    if kerr_override is not None:
        kerr = kerr_override
    else:
        kerr = (chi**2) / (4 * kappa)

    H_dis = get_dispersive_hamiltonian(device_params=device_params)
    H_kerr = H_dis - 2 * np.pi * kerr * 0.5 * Cd * Cd * C * C
    print(f"Kerr at this point is {kerr * 1e6:.3f} kHz")
    return H_kerr


def get_ry_pi2_hamiltonian(amp):
    H_ry_pi2 = 2 * np.pi * amp * 1j * (Qd - Q) / 2  # 1/2 factor for Ry(pi/2)
    return H_ry_pi2


def get_ry_amp_phase_hamiltonian(theta, phase ):
    H_ry_amp =  2 * np.pi * np.exp((phase * 1j * sz)) * (theta * 1j * sy) * (-phase * 1j * sz)  # 1/2 factor for Ry(pi/2)
    return H_ry_amp

def get_rz_phase_hamiltonian(phase):
    H_rz_phase =  2 * np.pi * phase * 1j * sz  # 1/2 factor for Ry(pi/2)
    return H_rz_phase

def get_ry_pi_hamiltonian(amp):
    H_ry_pi = 2 * np.pi * amp * 1j * (Qd - Q)  # 1/2 factor for Ry(pi/2)
    return H_ry_pi

#######################################
############### Pulses ################
#######################################

def RY_pi_exp_phase_amp(H, state, pulse_params, device_params):
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
    
    H_ry_theta = get_ry_amp_hamiltonian(amp=pulse_params["theta"])
    H_rz_phase = get_rz_phase_hamiltonian(amp=pulse_params["phase"])
    H_rz_phase_m = get_rz_phase_hamiltonian(amp=pulse_params["phase"])
    # print(pulse_params["timesteps"])
    H = [H, [H_ry_theta, pulse_params["pulse"]], [H_rz_phase, pulse_params["pulse"]]]
    return qt.mesolve(H, state, pulse_params["timesteps"], c_ops)

