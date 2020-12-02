from jax import numpy as np
import jax.random as random
from jax.ops import index_update

"Interaction Initialization"

# Structure arrays - initialize crystal and structure arrays
d33         = 23.4e-12  # in meter/Volt.[LiNbO3]
dx          = 10e-6
dy          = 10e-6
dz          = 1e-5
MaxX        = 200e-6
MaxY        = 200e-6
MaxZ        = 5e-3
R           = 0.1  # distance to far-field screen in meters
Temperature = 50
dk_offset   = 1  # delta_k offset
phi_scale   = 1  # scale for transverse phase


# initialize interacting beams' parameters
max_mode     = 10  # coefficients of beam-basis functions
# pump
lam_pump     = 532e-9
waist_pump   = 50e-6
power_pump   = 1e-3
#signal
lam_signal   = 1064e-9
power_signal = 1
# Idler
power_idler  = 1

# coincidence window
tau          = 1e-9  # [nanosec]

# Experiment parameters
coeffs_str = "my_custom"
poling_str = "no_tr_phase"
targert_folder = '2020-11-13_Nb500_Nx30Ny30_z0.02_steps400_2/'  # for loading targets for training


def HG_coeff_array(coeff_str, n_coeff):
    if (coeff_str == "rand_real"):
        coeffs = random.normal(random.PRNGKey(0), [n_coeff])
    elif (coeff_str == "random"):
        coeffs_rand = random.normal(random.PRNGKey(0), (n_coeff, 2))
        coeffs      = np.array(coeffs_rand[:, 0] + 1j*coeffs_rand[:, 1])
    elif (coeff_str == "HG00"):
        coeffs = np.zeros(n_coeff)
        coeffs = index_update(coeffs, 0, 1.0)
    elif (coeff_str == "my_custom"):
        coeffs = np.zeros(n_coeff)
        coeffs = index_update(coeffs, 2, 1.0)
        coeffs = index_update(coeffs, 3, 1.0)
        coeffs = index_update(coeffs, 14, 1.0)
        coeffs = index_update(coeffs, 35, 1.0)
    else:
        assert "ERROR: incompatible HG coefficients-string"

    coeffs    = coeffs / np.sqrt(np.sum(np.abs(coeffs)**2))

    return coeffs


def poling_array(poling_str):
    if (poling_str=="no_tr_phase"):
        phi_parameters = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif(poling_str == "linear_shift"):
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

    phi_parameters    = np.array(phi_parameters, dtype=np.float32)

    return phi_parameters
