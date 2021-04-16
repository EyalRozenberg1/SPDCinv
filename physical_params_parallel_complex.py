from jax import numpy as np
import jax.random as random
from jax.ops import index_update
from spdc_helper_parallel_complex import nz_MgCLN_Gayer

"Interaction Initialization"
# Structure arrays - initialize crystal and structure arrays
d33 = 16.9e-12  # in meter/Volt.
dx = 4e-6
dy = 4e-6
dz = 10e-6  # 2um works with batch size of 150
MaxX = 120e-6  # was 180 for 2um #Worked for learning: 4um and 120um
MaxY = 120e-6
MaxZ = 1e-3
R = 0.1  # distance to far-field screen in meters
Temperature = 50
dk_offset = 1  # delta_k offset

# initialize interacting beams' parameters

projection_type = 'LG'  # type of projection + pump modes

# LG number of modes for the pump + projections
max_mode_p = 1
max_mode_l = 4

# HG number of modes for the pump + projections
max_mode_x = 10
max_mode_y = 1

# pump
lam_pump = 532e-9
delta = 1
k = 2 * np.pi * nz_MgCLN_Gayer(lam_pump * 1e6, Temperature) / lam_pump
waist_pump = 40e-6 # delta * np.sqrt(MaxZ / k)  # according to L_crystal = 2*pi*(w0)^2*n/lambda, we get w_p = sqrt(L/k) -> w_s =sqrt(2).
power_pump = 1e-3
# signal
lam_signal = 2 * lam_pump
power_signal = 1
# Idler
power_idler = 1

# coincidence window
tau = 1e-9  # [nanosec]

# Experiment parameters
coeffs_str = "LG00"
poling_str = "no_tr_phase"
targert_folder = 'LG_target/'  # for loading targets for training

# Poling learning parameters
phi_scale = 1  # scale for transverse phase
r_scale = waist_pump  # was np.srt(2)*waist_pump # scale for radial variance of the poling


def projection_crystal_modes():
    # define two pump's function

    """
    * define two pump's function (for now n_coeff must be 2) to define the pump *
    * this should be later changed to the definition given by Sivan *
    """

    if projection_type == 'HG':
        max_mode1 = max_mode_x
        max_mode2 = max_mode_y
    else:
        max_mode1 = 2 * max_mode_l + 1
        max_mode2 = max_mode_p

    n_coeff = max_mode1 * max_mode2  # Total number of pump/projection modes

    # set the number of modes (radial for LG or y for HG) to allow the crystal to learn
    max_mode_crystal = 5

    return n_coeff, max_mode1, max_mode2, max_mode_crystal


def HG_coeff_array(coeff_str, n_coeff):  # TODO: change HG_coeff_array name to a genereal name
    if (coeff_str == "rand_real"):
        coeffs = random.normal(random.PRNGKey(0), [n_coeff])
    elif (coeff_str == "random"):
        coeffs_rand = random.normal(random.PRNGKey(0), (n_coeff, 2))
        coeffs = np.array(coeffs_rand[:, 0] + 1j * coeffs_rand[:, 1])
    elif (coeff_str == "HG00"):
        coeffs = np.zeros(n_coeff, dtype=np.float32)
        coeffs = index_update(coeffs, 0, 1.0)
    elif (coeff_str == "LG00"):  # index of LG_lp = p(2*max_mode_l+1) + max_mode_l + l
        coeffs_real = np.zeros(n_coeff, dtype=np.float32)
        coeffs_real = index_update(coeffs_real, max_mode_l - 4, 0)
        coeffs_real = index_update(coeffs_real, max_mode_l - 3, 0)
        coeffs_real = index_update(coeffs_real, max_mode_l - 2, np.sqrt(2))
        coeffs_real = index_update(coeffs_real, max_mode_l - 1, 0)
        coeffs_real = index_update(coeffs_real, max_mode_l, 1)
        coeffs_real = index_update(coeffs_real, max_mode_l + 1, 0)
        coeffs_real = index_update(coeffs_real, max_mode_l + 2, np.sqrt(2))
        coeffs_real = index_update(coeffs_real, max_mode_l + 3, 0)
        coeffs_real = index_update(coeffs_real, max_mode_l + 4, 0)

        coeffs_imag = np.zeros(n_coeff, dtype=np.float32)
        coeffs_imag = index_update(coeffs_imag, max_mode_l - 4, 0)
        coeffs_imag = index_update(coeffs_imag, max_mode_l - 3, 0)
        coeffs_imag = index_update(coeffs_imag, max_mode_l - 2, 0)
        coeffs_imag = index_update(coeffs_imag, max_mode_l - 1, 0)
        coeffs_imag = index_update(coeffs_imag, max_mode_l, 0)
        coeffs_imag = index_update(coeffs_imag, max_mode_l + 1, 0)
        coeffs_imag = index_update(coeffs_imag, max_mode_l + 2, 0)
        coeffs_imag = index_update(coeffs_imag, max_mode_l + 3, 0)
        coeffs_imag = index_update(coeffs_imag, max_mode_l + 4, 0)

    elif (coeff_str == "HG01"):
        coeffs = np.zeros(n_coeff, dtype=complex)
        coeffs = index_update(coeffs, 1, 1.0)
    elif (coeff_str == "HG02"):
        coeffs = np.zeros(n_coeff, dtype=complex)
        coeffs = index_update(coeffs, 2, 1.0)
    elif (coeff_str == "my_custom"):
        coeffs = np.zeros(n_coeff)
        coeffs = index_update(coeffs, 2, 1.0)
        coeffs = index_update(coeffs, 3, 1.0)
        coeffs = index_update(coeffs, 14, 1.0)
        coeffs = index_update(coeffs, 35, 1.0)
    else:
        assert "ERROR: incompatible HG coefficients-string"

    normalization = np.sqrt(np.sum(np.abs(coeffs_real) ** 2 + np.abs(coeffs_imag) ** 2))
    coeffs_real = coeffs_real / normalization
    coeffs_imag = coeffs_imag / normalization

    return coeffs_real, coeffs_imag


def poling_array_init(poling_str,
                      n_coeff):  # TODO: these are the initial conditions for the poliung profile. Let's rename it and remove what's not necessary
    if (poling_str == "no_tr_phase"):
        # phi_parameters = random.normal(random.PRNGKey(0), [n_coeff])
        phi_parameters = np.zeros(n_coeff, dtype=complex)
        phi_parameters = index_update(phi_parameters, 0, 0)
        phi_parameters = index_update(phi_parameters, max_mode_l, 1)
        phi_parameters = index_update(phi_parameters, max_mode_l - 2, 0)
        phi_parameters = index_update(phi_parameters, max_mode_l + 2, 0)
    elif (poling_str == "linear_shift"):
        phi_parameters = [0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif (poling_str == "lens"):
        phi_parameters = [0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0]
    elif (poling_str == "cube"):  # ??
        phi_parameters = [0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0]
    elif (poling_str == "Hermite4"):
        phi_parameters = [12, 0, -48, 0, 16, 0, 0, 0, 0, 0, 0]
    elif (poling_str == "Hermite6"):
        phi_parameters = [-120, 0, 720, 0, -480, 0, 64, 0, 0, 0, 0]
    elif (poling_str == "my_custom_22/07/2020"):
        phi_parameters = [30, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        phi_parameters = []

    phi_parameters = np.array(phi_parameters, dtype=np.float32)  # complex

    return phi_parameters

