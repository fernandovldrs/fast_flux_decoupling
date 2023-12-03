import numpy as np
import scipy as sc
import qutip as qt

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import integrate
from numpy import linalg as LA

from pulses import qd, q

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
        pulse_length = chop*sigma
        pulse = build_guassian_pulse(sigma=sigma, chop=chop)
    else:
        pulse_length = system_params["pi_pulse_params"]["pulse_length"]
        pulse = build_constant_pulse()

    # Calculate rabi frequency of the pulse
    ### Calculate rabi frequency for normalized square pi pulse
    square_area = pulse_length*1
    rabi_frequency_square = 1/4/square_area # Derived from rabi flip equation
    
    ### Calculate actual rabi frequency for input pulse 
    pi_pulse_area, _ = integrate.quad(pulse, 0, pulse_length)
    rabi_frequency = rabi_frequency_square*square_area/pi_pulse_area
    
    ### Update parameters
    timesteps = np.linspace(0, pulse_length, 50)
    system_params["pi_pulse_params"]["pulse"] = pulse
    system_params["pi_pulse_params"]["rabi_freq"] = rabi_frequency
    system_params["pi_pulse_params"]["timesteps"] = timesteps
    
    return system_params


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
