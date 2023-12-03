import numpy as np
from scipy import integrate

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
