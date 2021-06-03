from jax import numpy as np
import jax.random as random
from jax.ops import index_update
from spdc_inv.data.utils import n_KTP_Kato


"Interaction Initialization"
# Structure arrays - initialize crystal and structure arrays
d33  = 16.9e-12  # in meter/Volt.
dx   = 4e-6
dy   = 4e-6
dz   = 10e-6  # 2um works with batch size of 150
MaxX = 120e-6  # was 180 for 2um #Worked for learning: 4um and 120um
MaxY = 120e-6
MaxZ = 1e-4
R    = 0.1  # distance to far-field screen in meters

Temperature = 50
dk_offset = 1  # delta_k offset

" initialize interacting beams' parameters "

projection_type = 'LG'  # type of projection
# LG number of modes for projections
max_mode_p = 1
max_mode_l = 4

# HG number of modes for projections
max_mode_x = 10
max_mode_y = 1

pump_basis = 'LG'  # pump construction method
# LG number of modes for pump basis
max_angular_mode_pump = 2
max_radial_mode_pump  = 5

# HG number of modes for pump basis
max_mode_x_pump = 5
max_mode_y_pump = 1

crystal_basis = 'LG'
# FT (Fourier-Taylor) / FB (Fourier-Bessel) / LG (Laguerre-Gauss) number of modes for crystal basis
max_mode_crystal_1 = 2
max_mode_crystal_2 = 7

# HG (Hermite-Gauss) number of modes for crystal basis
max_mode_x_crystal = 5
max_mode_y_crystal = 5


# pump physical parameters
lam_pump        = 405e-9
delta           = 1
k               = 2 * np.pi * n_KTP_Kato(lam_pump * 1e6, Temperature, 'y') / lam_pump
power_pump      = 1e-3
waist_pump0     = np.sqrt(MaxZ / k)  # pump waists #40e-6
waist_pump_proj = np.sqrt(2)*waist_pump0  # 1/np.sqrt(1/r_scale0**2 + 1/waist_pump0**2)

# crystal physical parameter
r_scale0        = 40e-6  # np.sqrt(MaxZ / k) # pump waists


# Signal & Idler
lam_signal   = 2 * lam_pump
power_signal = 1
power_idler  = 1

# coincidence window
tau = 1e-9  # [nanosec]

# Experiment parameters
coeffs_str     = "random"
crystal_str    = "random"
targert_folder = 'LG_target/'  # for loading targets for training
path_for_read  = '2021-04-20_N_infer4000_Nx75Ny75_z0.03_steps1000_#devices8_N_learn96_loss_l1_epochs100_complex/'

# Poling learning parameters
phi_scale = 1  # scale for transverse phase

def projection_crystal_modes():
    """
    * define two pump's function (for now n_coeff must be 2) to define the pump *
    * this should be later changed to the definition given by Sivan *
    """

    if projection_type == 'LG':
        max_mode1 = 2 * max_mode_l + 1
        max_mode2 = max_mode_p
    else:
        max_mode1 = max_mode_x
        max_mode2 = max_mode_y

    if pump_basis == 'LG':
        max_mode1_pump = 2 * max_angular_mode_pump + 1
        max_mode2_pump = max_radial_mode_pump
    else:
        max_mode1_pump = max_mode_x_pump
        max_mode2_pump = max_mode_y_pump

    if crystal_basis in ['FT', 'FB', 'LG']:
        max_mode1_crystal = 2 * max_mode_crystal_1 + 1
        max_mode2_crystal = max_mode_crystal_2
    else:
        max_mode1_crystal = max_mode_x_crystal
        max_mode2_crystal = max_mode_y_crystal

    return max_mode1, max_mode2,\
           max_mode1_pump, max_mode2_pump,\
           max_mode1_crystal, max_mode2_crystal


def Pump_coeff_array(coeff_str, n_coeff):
    if (coeff_str == "uniform"):
        coeffs_real = np.ones(n_coeff, dtype=np.float32)
        coeffs_imag = np.ones(n_coeff, dtype=np.float32)
    elif (coeff_str == "random"):
        seed_real = 111
        seed_imag = 222
        coeffs_real = random.normal(random.PRNGKey(seed_real), (n_coeff,))
        coeffs_imag = random.normal(random.PRNGKey(seed_imag), (n_coeff,))
    elif (coeff_str == "LG00"):
        # index of LG_lp = p(2 * max_angular_mode_pump + 1) + max_angular_mode_pump + l
        coeffs_real = np.zeros(n_coeff, dtype=np.float32)
        coeffs_real = index_update(coeffs_real, 0*(2*max_angular_mode_pump+1) + max_angular_mode_pump - 2, 0)
        coeffs_real = index_update(coeffs_real, 0*(2*max_angular_mode_pump+1) + max_angular_mode_pump, 1)
        coeffs_real = index_update(coeffs_real, 0*(2*max_angular_mode_pump+1) + max_angular_mode_pump + 2, 0)

        coeffs_real = index_update(coeffs_real, 1*(2*max_angular_mode_pump+1) + max_angular_mode_pump - 2, 0)
        coeffs_real = index_update(coeffs_real, 1*(2*max_angular_mode_pump+1) + max_angular_mode_pump, 0)
        coeffs_real = index_update(coeffs_real, 1*(2*max_angular_mode_pump+1) + max_angular_mode_pump + 2, 0)

        coeffs_imag = np.zeros(n_coeff, dtype=np.float32)
        coeffs_imag = index_update(coeffs_imag, 0*(2*max_angular_mode_pump+1) + max_angular_mode_pump - 2, 0)
        coeffs_imag = index_update(coeffs_imag, 0*(2*max_angular_mode_pump+1) + max_angular_mode_pump, 1)
        coeffs_imag = index_update(coeffs_imag, 0*(2*max_angular_mode_pump+1) + max_angular_mode_pump + 2, 0)

        coeffs_imag = index_update(coeffs_imag, 1*(2*max_angular_mode_pump+1) + max_angular_mode_pump - 2, 0)
        coeffs_imag = index_update(coeffs_imag, 1*(2*max_angular_mode_pump+1) + max_angular_mode_pump, 0)
        coeffs_imag = index_update(coeffs_imag, 1*(2*max_angular_mode_pump+1) + max_angular_mode_pump + 2, 0)
    elif (coeff_str == "read"):
        coeffs_real = np.load('results/' + path_for_read + 'PumpCoeffs_real.npy')
        coeffs_imag = np.load('results/' + path_for_read + 'PumpCoeffs_imag.npy')
    else:
        assert "ERROR: incompatible pump coefficients-string"

    normalization = np.sqrt(np.sum(np.abs(coeffs_real) ** 2 + np.abs(coeffs_imag) ** 2))
    coeffs_real = coeffs_real / normalization
    coeffs_imag = coeffs_imag / normalization

    if coeffs_str == "read":
        waist_pump = np.load("results/" + path_for_read + "PumpWaistCoeffs.npy") * 1e-1
    else:  # initial pump waists w.r.t the waist basis
        waist_pump = np.ones(n_coeff, dtype=np.float32) * waist_pump0 * 1e5

    return coeffs_real, coeffs_imag, waist_pump


def Crystal_coeff_array(crystal_str, n_coeff):
    if (crystal_str == "uniform"):
        crystal_coeffs_real = np.ones(n_coeff, dtype=np.float32)
        crystal_coeffs_imag = np.ones(n_coeff, dtype=np.float32)
    elif (crystal_str == "random"):
        seed_real = 1110
        seed_imag = 2220
        crystal_coeffs_real = random.normal(random.PRNGKey(seed_real), (n_coeff,))
        crystal_coeffs_imag = random.normal(random.PRNGKey(seed_imag), (n_coeff,))
    elif (crystal_str == "FB00"):
        crystal_coeffs_real = np.zeros(n_coeff, dtype=np.float32)
        crystal_coeffs_real = index_update(crystal_coeffs_real,
                                           0 * (2 * max_mode_crystal_1 + 1) + max_mode_crystal_1 - 2, 0)
        crystal_coeffs_real = index_update(crystal_coeffs_real,
                                           0 * (2 * max_mode_crystal_1 + 1) + max_mode_crystal_1, 1)
        crystal_coeffs_real = index_update(crystal_coeffs_real,
                                           0 * (2 * max_mode_crystal_1 + 1) + max_mode_crystal_1 + 2, 0)

        crystal_coeffs_real = index_update(crystal_coeffs_real,
                                           1 * (2 * max_mode_crystal_1 + 1) + max_mode_crystal_1 - 2, 0)
        crystal_coeffs_real = index_update(crystal_coeffs_real,
                                           1 * (2 * max_mode_crystal_1 + 1) + max_mode_crystal_1, 0)
        crystal_coeffs_real = index_update(crystal_coeffs_real,
                                           1 * (2 * max_mode_crystal_1 + 1) + max_mode_crystal_1 + 2, 0)


        crystal_coeffs_imag = np.zeros(n_coeff, dtype=np.float32)
        crystal_coeffs_imag = index_update(crystal_coeffs_imag,
                                           0 * (2 * max_mode_crystal_1 + 1) + max_mode_crystal_1 - 2, 0)
        crystal_coeffs_imag = index_update(crystal_coeffs_imag,
                                           0 * (2 * max_mode_crystal_1 + 1) + max_mode_crystal_1, 1)
        crystal_coeffs_imag = index_update(crystal_coeffs_imag,
                                           0 * (2 * max_mode_crystal_1 + 1) + max_mode_crystal_1 + 2, 0)

        crystal_coeffs_imag = index_update(crystal_coeffs_imag,
                                           1 * (2 * max_mode_crystal_1 + 1) + max_mode_crystal_1 - 2, 0)
        crystal_coeffs_imag = index_update(crystal_coeffs_imag,
                                           1 * (2 * max_mode_crystal_1 + 1) + max_mode_crystal_1, 0)
        crystal_coeffs_imag = index_update(crystal_coeffs_imag,
                                           1 * (2 * max_mode_crystal_1 + 1) + max_mode_crystal_1 + 2, 0)
    elif (crystal_str == "HG00"):
        crystal_coeffs_real = np.zeros(n_coeff, dtype=np.float32)
        crystal_coeffs_real = index_update(crystal_coeffs_real, 0, 1)
        crystal_coeffs_imag = np.zeros(n_coeff, dtype=np.float32)
    else:
        assert "ERROR: incompatible crystal coefficients-string"

    normalization = np.sqrt(np.sum(np.abs(crystal_coeffs_real) ** 2 + np.abs(crystal_coeffs_imag) ** 2))
    crystal_coeffs_real = crystal_coeffs_real / normalization
    crystal_coeffs_imag = crystal_coeffs_imag / normalization

    # scale for radial variance of the poling
    r_scale = np.ones(n_coeff, dtype=np.float32) * r_scale0 * 1e5

    return crystal_coeffs_real, crystal_coeffs_imag, r_scale
