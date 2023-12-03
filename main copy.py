import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

from experiments import (
    chi_hamiltonian_simulation,
    kerr_hamiltonian_simulation,
    char_func_ideal_1d,
    char_func_ideal_2d,
)
from pulses import (
    cdim,
    rho,
    u_g,
    u_e,
    Pg,
    get_dispersive_hamiltonian,
)

from pulse_utils import (
    configure_pi_pulse,
    gaussian,
    gaussian_fit,
    gaussian_2d,
    get_2d_guassian_sigma,
)

from plotting import (
    plot_double_2d_cmap,
    plot_double_1d_graphs,
    plot_2d_cmap,
    do_fitting,
    plot_wigner,
)

device_params = {
    "chi": 0.679e-3, #1e-3, #0.94e-3, #0.679e-3,  # in GHz   0.679e-3
    "T1": 15e3,  # 15e3,  # in ns
    "T2": 400, #400,  # in ns
    # "T2": 400, #400,  # in ns
    "cavT1": 30e3,  # in ns
    "nbar_cav": 0.03,  # thermal state population of cavity
    "anharm": 0.200,  # anharmonicity of qubit in GHz
    "g": 7.3e-3,  # coupling strength in GHz
    "nbar_qubit": 0.05,  # thermal state population of qubit
    "kerr": 0.000006,  # overwrites analytical value of kerr (in GHz) if not None
}

pi_pulse_params = {
    "pulse_type": "gaussian",  # can be constant or gaussian
    "pulse_length": 100,  # in ns, only for constant pulses
    "sigma": 10,  # in ns, only for gaussian pulses
    "chop": 4,  # only for gaussian pulses
    "rabi_freq": None,  # to be set during runtime
    "pulse": None,  # to be set during runtime
    "timesteps": None,  # to be set during runtime
}
experiment_params = {
    "alpha": 1.5,  # choose the coherent state you want to create
    "num_iter":10,  # only for chi experiments
    "pulse_interval": None,  # in ns, only for chi experiments
    "amp_scale": 1,  # Use 1 for pi, 0.5 for pi2. Only for chi long experiments
    "pulse_time": 1e3,  # only for chi long experiments
    "wait_time": 10e3,  # in ns, only for kerr experiments
    "evolve_time": 10e3,  # in ns, only for old chi exp
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
# sweep_field = "experiment_params:num_iter"
# sweep_points = [0, 2, 4, 6, 8, 10, 12]

sweep_field = "experiment_params:pulse_interval"
sweep_points = [400,] #ns
# units = "us"  # For figure labels
# label_scale_factor = 1e-3  # For figure labels

# sweep_field = "device_params:nbar_qubit"
# sweep_points = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14]
units = "ns"  # For figure labels
label_scale_factor = 1  # For figure labels

# choose between kerr or chi experiment
experiment = "chi_pi2" 
# choose if finite pulses are to be used (only applicable for chi exp)
finite_pulses = True # False #True

project_to_ground = True

# Measurement params
plot_type = "char" # char/ wigner
plot_data = False
max_alpha = 4
npts = 101
xvec = np.linspace(-max_alpha, max_alpha, npts)

# Plotting params for 2D plots
vmin = -1
vmax = 1

plot_imag = False
plot_real = True

# Plotting params for 1D plots
axis = "x"
cut_point = 0
do_fits = (False, False)


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
        rho_d = Ds.dag() * rho * Ds
        rho_f1 = (qt.fock(cdim, 1)) * (qt.fock(cdim, 1)).dag()
        state = qt.tensor(
            (1 - d_params["nbar_qubit"]) * u_g * u_g.dag()
            + d_params["nbar_qubit"] * u_e * u_e.dag(),
            rho_d,
        )
        # state = qt.tensor(u_g * u_g.dag(), rho_d)

        if d_params["chi"] > 0:
            detuning = (d_params["g"] ** 2) / (d_params["chi"])
            print(f"Detuning here should be {detuning * 1e3} MHz")

        # experiment
        
        if experiment == "chi_pi2": #pi2 chi
            state = chi_hamiltonian_simulation(
                H=ham,
                state=state,
                device_params=d_params,
                exp_params=e_params,
                finite_pulses=finite_pulses,
                pulse_params=p_params,
            )
        else:
            state = kerr_hamiltonian_simulation(
                H=ham,
                state=state,
                device_params=d_params,
                experiment_params=e_params,
            )
        result_states.append(state)

    if project_to_ground:
        state = Pg * state * Pg # this should be right 
    # measurement and plotting
    x_sig = []
    y_sig = []
    sig_ratio = []
    if plot_type == "wigner":
        plot_wigner(state)
    elif plot_type == "char":
        for i in range(len(result_states)):
            state = result_states[i]
            name = sweep_points[i] * label_scale_factor
            cf_real, cf_imag = char_func_ideal_2d(state=state, xvec=xvec, scale =1)

            real_title = f"{param} = {name} {units} real"
            imag_title = f"{param} = {name} {units} imag"
            if plot_imag and plot_real:
                _ = plot_double_2d_cmap(
                    xvecs=(xvec, xvec),
                    yvecs=(xvec, xvec),
                    zs=(cf_real, cf_imag),
                    vmin=vmin,
                    vmax=vmax,
                    titles=(real_title, imag_title),
                )
            elif plot_imag:
                fig, axes = plt.subplots()
                _ = plot_2d_cmap(
                    xvec=xvec,
                    yvec=xvec,
                    z=(cf_imag),
                    ax=axes,
                    vmin=vmin,
                    vmax=vmax,
                    title=(imag_title),
                )
            elif plot_real:
                fig, axes = plt.subplots()
                _ = plot_2d_cmap(
                    xvec=xvec,
                    yvec=xvec,
                    z=(cf_real),
                    ax=axes,
                    vmin=vmin,
                    vmax=vmax,
                    title=(real_title),
                )
            else:
                pass

            if plot_data:
                (sig1, sig2, v1, v2, opt) = do_fitting(x=xvec, y=xvec, z=cf_real)
                x_sig.append(sig1)
                y_sig.append(sig2)
                sig_ratio.append(min(sig1, sig2) / max(sig1, sig2))
            plt.show()
            # plt.savefig(
            #     f"new_chi_values/{(sweep_points[i])}_kerr_exp_wait_10us_alpha_5.png"
            # )
        if plot_data:
            x_axis = sweep_points
            fig, axes = plt.subplots()
            axes.plot(x_axis, sig_ratio)

            axes.set_xlabel("kerr factor")
            axes.set_ylabel("sigma ratio")

            plt.show()

    else:
        fit_text = [[], []]
        real_title = f"{axis}-axis cut real"
        imag_title = f"{axis}-axis cut imag"
        fig, axes = plt.subplots(1, 2)
        for i in range(len(result_states)):
            name = sweep_points[i] * label_scale_factor
            label = f"{name} {units}"
            state = result_states[i]
            name = sweep_points[i] * label_scale_factor
            cf_real, cf_imag = char_func_ideal_1d(
                state=state, xvec=xvec, axis=axis, cut_point=cut_point
            )
            _, axes, opts = plot_double_1d_graphs(
                fig=fig,
                axes=axes,
                xvecs=(xvec, xvec),
                yvecs=(cf_real, cf_imag),
                do_fits=do_fits,
                label=label,
            )

            for opt in opts:
                if opt:
                    sigma = opt[2]
                    fit_text[0].append(f"{name} R-sigma = {sigma:.5f}")

        text_real = "\n".join(fit_text[0])
        axes[0].set_title(real_title)
        axes[0].legend()
        axes[0].text(
            0,
            -0.06,
            text_real,
            horizontalalignment="left",
            verticalalignment="top",
            transform=axes[0].transAxes,
            fontsize=6,
        )

        text_imag = "\n".join(fit_text[1])
        axes[1].set_title(imag_title)
        axes[1].legend()
        axes[1].text(
            0,
            -0.06,
            text_imag,
            horizontalalignment="left",
            verticalalignment="top",
            transform=axes[1].transAxes,
            fontsize=6,
        )
        plt.show()