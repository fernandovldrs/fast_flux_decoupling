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
    kerr_override = device_params["kerr"]
    if kerr_override is not None:
        kerr = kerr_override
    else:
        kerr = (chi**2) / (4 * anharm)

    H_dis = -2 * np.pi * chi * Cd * C * Qd * Q - 2 * np.pi * anharm / 2 * Qd * Qd * Q * Q 
    
    H_kerr = H_dis - 2 * np.pi * kerr * 0.5 * Cd * Cd * C * C
    print(f"Kerr at this point is {kerr * 1e6:.3f} kHz")
    return H_kerr


def get_ry_pi2_hamiltonian(amp):
    H_ry_pi2 = 2 * np.pi * amp * 1j * (Qd - Q) / 2  # 1/2 factor for Ry(pi/2)
    return H_ry_pi2

def get_ry_pi_hamiltonian(amp):
    H_ry_pi = 2 * np.pi * amp * 1j * (Qd - Q)  # 1/2 factor for Ry(pi/2)
    return H_ry_pi
