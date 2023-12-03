import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

from utils.pulses import configure_pi_pulse
from utils.plotting import plot_2d_cmap

from experiments import (
    chi_hamiltonian_simulation,
    kerr_hamiltonian_simulation,
    char_func_ideal_2d,
)
from utils.operators import (
    cdim,
    rho_vacuum,
    u_g,
    u_e,
    Pg,
    get_dispersive_hamiltonian,
)


device_params = {
    "chi": 0.679e-3, #1e-3, #0.94e-3, #0.679e-3,  # in GHz   0.679e-3
    "T1": 15e3,  # 15e3,  # in ns
    "T2": 400, # in ns
    "cavT1": 30e3,  # in ns
    "nbar_cav": 0.03,  # thermal state population of cavity
    "anharm": 0.200,  # anharmonicity of qubit in GHz
    "nbar_qubit": 0.05,  # thermal state population of qubit
    "kerr": 0.000006,  # overwrites analytical value of kerr (in GHz) if not None
}

pi_pulse_params = {
    "pulse_type": "gaussian",  # can be constant or gaussian
    # "pulse_length": 100,  # in ns, only for constant pulses
    "sigma": 10,  # in ns, only for gaussian pulses
    "chop": 4,  # only for gaussian pulses
    "rabi_freq": None,  # to be set during runtime
    "pulse": None,  # to be set during runtime
    "timesteps": None,  # to be set during runtime
}
experiment_params = {
    "alpha": 2.5,  # choose the coherent state you want to create
    "num_iter":10,  # only for chi experiments
    "pulse_interval": None,  # in ns, only for chi experiments
    "amp_scale": 0.5,  # Use 1 for pi, 0.5 for pi2. Only for chi long experiments
    "wait_time": 10e3,  # in ns, only for kerr experiments
}

system_params = {
    "device_params": device_params,
    "pi_pulse_params": pi_pulse_params,
    "experiment_params": experiment_params,
}

"""
Configure sweep here. Format of sweep fields should be of the form <subcategory>:<param_name>. For example to sweep T1, you use 
sweep_field = "device_params:T1"
sweep_points = [10e3, 15e3]
"""

sweep_field = "experiment_params:pulse_interval"
sweep_points = [400,] #ns

# choose between kerr or chi experiment
experiment = "chi"
# choose if finite pulses are to be used (only applicable for chi exp)
finite_pulses = True

project_to_ground = True

# Measurement params
plot_type = "char" # char/ wigner
plot_data = False
max_alpha = 4
npts = 61
xvec = np.linspace(-max_alpha, max_alpha, npts)


if __name__ == "__main__":
    # configure sweep
    field_split = sweep_field.split(":")
    category = field_split[0]
    param = field_split[1]
    result_states = []
    print(f"Sweeping over {category}:{param}")

    # if we are not sweeping pulse params, we can just configure the pulse once here.
    if category != "pi_pulse_params":
        system_params = configure_pi_pulse(system_params=system_params)

    for point in sweep_points:
        # set current sweep point value
        system_params[category][param] = point
        print(f"{sweep_field} = {point}")

        if category == "pi_pulse_params":
            system_params = configure_pi_pulse(system_params=system_params)

        d_params = system_params["device_params"]
        p_params = system_params["pi_pulse_params"]
        e_params = system_params["experiment_params"]
        
        # setting system hamiltonian
        ham = get_dispersive_hamiltonian(device_params=d_params)

        # state preparation
        Ds = qt.displace(cdim, e_params["alpha"])
        rho_d = Ds.dag() * rho_vacuum * Ds
        
        state = qt.tensor(
            (1 - d_params["nbar_qubit"]) * u_g * u_g.dag()
            + d_params["nbar_qubit"] * u_e * u_e.dag(),
            rho_d,
        )

        # experiment
        if experiment == "chi":
            state = chi_hamiltonian_simulation(
                H=ham,
                state=state,
                device_params=d_params,
                exp_params=e_params,
                finite_pulses=finite_pulses,
                pulse_params=p_params,
            )
        elif experiment == "kerr":
            state = kerr_hamiltonian_simulation(
                H=ham,
                state=state,
                device_params=d_params,
                experiment_params=e_params,
            )
        result_states.append(state)

    if project_to_ground:
        state = Pg * state * Pg # this should be right 
        
        
    # Plotting
    for i in range(len(result_states)):
        state = result_states[i]
        name = sweep_points[i]
        cf_real, cf_imag = char_func_ideal_2d(state=state, xvec=xvec, scale = 1)
        wigner = qt.wigner(qt.ptrace(state, 1), xvec, xvec, g = 2 )

        fig, axes = plt.subplots(1, 3)
        plot_2d_cmap(xvec, wigner, axes[0], title = "Wigner")
        plot_2d_cmap(xvec, cf_real, axes[1], title = "Char function Re")
        plot_2d_cmap(xvec, cf_imag, axes[2], title = "Char function Im")
        plt.show()