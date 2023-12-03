import numpy as np
import scipy as sc
import qutip as qt

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy import linalg as LA

from pulses import qd, q


erf = sc.special.erf  # error function


def build_guassian_pulse(sigma, chop):
    t0 = sigma * chop / 2

    def pulse(t, *args):
        return np.exp(-1 / 2 * (t - t0) ** 2 / sigma**2)

    return pulse


def build_constant_pulse():
    def pulse(t, *args):
        return 1.0

    return pulse


def configure_pi_pulse(system_params):
    pulse_type = system_params["pi_pulse_params"]["pulse_type"]
    if pulse_type == "gaussian":
        chop = system_params["pi_pulse_params"]["chop"]
        sigma = system_params["pi_pulse_params"]["sigma"]
        pulse = build_guassian_pulse(sigma=sigma, chop=chop)
        timesteps = np.linspace(0, sigma * chop, 50)
    else:
        pulse_length = system_params["pi_pulse_params"]["pulse_length"]
        timesteps = np.linspace(0, pulse_length, 50)
        pulse = build_constant_pulse()
    system_params["pi_pulse_params"]["pulse"] = pulse
    system_params["pi_pulse_params"]["amp"] = pi_pulse_amp_calibrator(
        device_params=system_params["device_params"],
        pulse_params=system_params["pi_pulse_params"],
        axis="y",
        plot=False,
    )
    system_params["pi_pulse_params"]["timesteps"] = timesteps
    return system_params


def pi_pulse_amp_calibrator(device_params, pulse_params, axis, plot=False):
    """
    Power Rabi: Use this to calibrate the amplitude needed to drive a qubit pi pulse
    """
    amp = np.linspace(0.0, 1.5, 199)
    output = []

    T1 = device_params["T1"]
    Tphi = -1 / (1 / 2 / device_params["T1"] - 1 / device_params["T2"])

    pulse_type = pulse_params["pulse_type"]
    pulse = pulse_params["pulse"]
    if pulse_type == "gaussian":
        sigma = pulse_params["sigma"]
        chop = pulse_params["chop"]
        A0 = (
            np.sqrt(2 / np.pi) / erf(np.sqrt(2)) * np.pi / (4 * sigma) / 2 / np.pi
        )  # initial guess
        tlist = np.linspace(0, sigma * chop, 50)  # in ns
    else:
        pulse_length = pulse_params["pulse_length"]
        A0 = (
            np.sqrt(2 / np.pi) / erf(np.sqrt(2)) * np.pi / pulse_length / 2 / np.pi
        )  # initial guess
        tlist = np.linspace(0, pulse_length, 50)  # in ns

    for Ax in amp:
        # A = (
        #     np.sqrt(2 / np.pi) / erf(np.sqrt(2)) * np.pi / (4 * sigma) / 2 / np.pi
        # )  # initial guess
        # A0 = A  # keep it for later
        A = A0
        freq = 0  # resonant driving

        A *= Ax  # coefficient for the Gaussian pulse

        H0 = 2 * np.pi * freq * qd * q

        if axis == "y":
            Hd = 2 * np.pi * A * 1j * (qd - q)
        else:
            Hd = -2 * np.pi * A * (qd + q)

        H = [H0, [Hd, pulse]]

        psi = qt.basis(2, 0)  # initial state
        rhoq = qt.ket2dm(psi)

        c_ops = [np.sqrt(1 / T1) * q, np.sqrt(2 / Tphi) * qd * q]  # changed

        e_ops = [
            qd * q,
        ]

        # options = Options(max_step=1, nsteps=1e6)

        results = qt.mesolve(
            H, rhoq, tlist, c_ops=c_ops, e_ops=e_ops
        )  # , options=options)  # , progress_bar = True)

        output += [
            results.expect[0][-1],
        ]
    if plot:
        # for checking
        plt.plot(amp, output)
        plt.ylabel(r"pe")
        plt.xlabel("Amplitude Scale")
        plt.title("Power Rabi")
        plt.grid()
        plt.show()

    print(max(output), output.index(max(output)), amp[output.index(max(output))])
    # print(A0)
    return A0 * amp[output.index(max(output))]  # this is the correct coeff


# Gaussian function for curve fitting
def gaussian(x, a, b, sigma, c):
    return a * np.exp(-((x - b) ** 2) / (2 * sigma**2)) + c


def gaussian_fit(y, x):
    mean_arg = np.argmax(y)  # np.argmin(y)
    mean = x[mean_arg]
    sigma = 0.5  # np.sqrt(sum(y * (x - mean) ** 2) / sum(y))

    popt, pcov = curve_fit(
        gaussian,
        x,
        y,
        bounds=(
            (0, min(x), -np.inf, -np.inf),
            (np.inf, max(y), np.inf, np.inf),
        ),
        p0=[max(y) - min(y), mean, sigma, min(y)],
    )
    return popt


def correct_ofs_and_amp(data_array, amp, ofs):
    return (data_array - ofs) / amp


def gaussian_2d(xy, A, B, C):
    x = xy[:, 0]
    y = xy[:, 1]
    envelope = np.exp(-(A * x**2 + 2 * B * x * y + C * y**2))
    return envelope


def gaussian_2d_fit(x, y, z, bounds, guess):
    x_mesh, y_mesh = np.meshgrid(x, y)
    xdata = np.c_[x_mesh.flatten(), y_mesh.flatten()]
    popt, pcov = curve_fit(
        gaussian_2d,
        xdata,
        z.flatten(),
        bounds=bounds,
        p0=guess,
    )
    return popt


def get_2d_guassian_sigma(x, y, z):
    bounds = [
        (0, -np.inf, 0),
        (
            np.inf,
            np.inf,
            np.inf,
        ),
    ]
    guess = (20, -50, 40)
    # z = correct_ofs_and_amp(z, 1.0, 1 * min(z.flatten()))
    opt = gaussian_2d_fit(x, y, z, bounds, guess)
    A, B, C = opt[:3]
    cov_matrix = np.array([[A, B], [B, C]])
    results = LA.eig(cov_matrix)
    lamb1, lamb2 = results[0]
    vec1, vec2 = results[1]
    sig1, sig2 = 1 / (2 * lamb1) ** 0.5, 1 / (2 * lamb2) ** 0.5
    return (sig1, sig2, vec1, vec2, opt)
