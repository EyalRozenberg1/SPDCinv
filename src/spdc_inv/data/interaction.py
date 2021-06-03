import jax.random as random
import os

from abc import ABC
from jax import numpy as np
from jax.ops import index_update
from spdc_inv.data.utils import n_KTP_Kato, nz_MgCLN_Gayer
from spdc_inv.utils.utils import PP_crystal_slab
from spdc_inv.utils.utils import SFG_idler_wavelength
from spdc_inv.utils.defaults import REAL, IMAG
from typing import Dict


class Interaction(ABC):
    """
    A class that represents the SPDC interaction process,
    on all of its physical parameters.
    """

    def __init__(
            self,
            pump_basis: str = 'LG',
            pump_max_mode1: int = 5,
            pump_max_mode2: int = 2,
            initial_pump_coefficient: str = 'uniform',
            custom_pump_coefficient: Dict[str, Dict[int, int]] = None,
            pump_coefficient_path: str = None,
            initial_pump_waist: str = 'waist_pump',
            pump_waists_path: str = None,
            crystal_basis: str = 'LG',
            crystal_max_mode1: int = 5,
            crystal_max_mode2: int = 2,
            initial_crystal_coefficient: str = 'uniform',
            custom_crystal_coefficient: Dict[str, Dict[int, int]] = None,
            crystal_coefficient_path: str = None,
            initial_crystal_waist: str = 'r_scale0',
            crystal_waists_path: str = None,
            lam_pump: float = 405e-9,
            crystal_str: str = 'ktp',
            power_pump: float = 1e-3,
            waist_pump0: float = None,
            r_scale0: float = 40e-6,
            dx: float = 4e-6,
            dy: float = 4e-6,
            dz: float = 10e-6,
            maxX: float = 120e-6,
            maxY: float = 120e-6,
            maxZ: float = 1e-4,
            R: float = 0.1,
            Temperature: float = 50,
            pump_polarization: str = 'y',
            signal_polarization: str = 'y',
            idler_polarization: str = 'z',
            dk_offset: float = 1,
            power_signal: float = 1,
            power_idler: float = 1,
            key: np.array = None,

    ):
        """

        Parameters
        ----------
        pump_basis: Pump's construction basis method
                    Can be: LG (Laguerre-Gauss) / HG (Hermite-Gauss)
        pump_max_mode1: Maximum value of first mode of the 2D pump basis
        pump_max_mode2: Maximum value of second mode of the 2D pump basis
        initial_pump_coefficient: initial distribution of coefficient-amplitudes for pump basis function
        pump_coefficient_path: path for loading waists for pump basis function
        custom_pump_coefficient: (dictionary) used only if initial_pump_coefficient=='custom'
                                 {'real': {indexes:coeffs}, 'imag': {indexes:coeffs}}.
        initial_pump_waist: initial values of waists for pump basis function
        pump_waists_path: path for loading coefficient-amplitudes for pump basis function
        crystal_basis: Crystal's construction basis method
                       Can be:
                       None / FT (Fourier-Taylor) / FB (Fourier-Bessel) / LG (Laguerre-Gauss) / HG (Hermite-Gauss)

        crystal_max_mode1: Maximum value of first mode of the 2D crystal basis
        crystal_max_mode2: Maximum value of second mode of the 2D crystal basis
        initial_crystal_coefficient: initial distribution of coefficient-amplitudes for crystal basis function
        custom_crystal_coefficient: (dictionary) used only if initial_pump_coefficient=='custom'
                                 {'real': {indexes:coeffs}, 'imag': {indexes:coeffs}}.
        crystal_coefficient_path: path for loading coefficient-amplitudes for pump basis function
        initial_crystal_waist: initial values of waists for crystal basis function
        crystal_waists_path: path for loading waists for crystal basis function
        lam_pump: Pump wavelength
        crystal_str: Crystal type. Can be: KTP or MgCLN
        power_pump: Pump power [watt]
        waist_pump0: waists of the pump basis functions
        r_scale0: effective waists of the crystal basis functions
        dx: transverse resolution in x [m]
        dy: transverse resolution in y [m]
        dz: longitudinal resolution in z [m]
        maxX: Transverse cross-sectional size from the center of the crystal in x [m]
        maxY: Transverse cross-sectional size from the center of the crystal in y [m]
        maxZ: Crystal's length in z [m]
        R: distance to far-field screen [m]
        Temperature: crystal's temperature [Celsius Degrees]
        pump_polarization: Polarization of the pump beam
        signal_polarization: Polarization of the signal beam
        idler_polarization: Polarization of the idler beam
        dk_offset: delta_k offset
        power_signal: Signal power [watt]
        power_idler: Idler power [watt]
        key: Random key
        """

        self.lam_pump = lam_pump
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.x = np.arange(-maxX, maxX, dx)  # x axis, length 2*MaxX (transverse)
        self.y = np.arange(-maxY, maxY, dy)  # y axis, length 2*MaxY  (transverse)
        self.Nx = len(self.x)
        self.Ny = len(self.y)
        self.z = np.arange(-maxZ / 2, maxZ / 2, dz)  # z axis, length MaxZ (propagation)
        self.maxX = maxX
        self.maxY = maxY
        self.maxZ = maxZ
        self.R = R
        self.Temperature = Temperature
        self.dk_offset = dk_offset
        self.power_pump = power_pump
        self.power_signal = power_signal
        self.power_idler = power_idler
        self.lam_signal = 2 * lam_pump
        self.lam_idler = SFG_idler_wavelength(self.lam_pump, self.lam_signal)
        self.pump_polarization = pump_polarization
        self.signal_polarization = signal_polarization
        self.idler_polarization = idler_polarization
        self.key = key

        assert crystal_str.lower() in ['ktp', 'mgcln'], 'crystal must be either KTP or MgCLN'
        if crystal_str.lower() == 'ktp':
            self.ctype = n_KTP_Kato  # refractive index function
            self.pump_k = 2 * np.pi * n_KTP_Kato(lam_pump * 1e6, Temperature, pump_polarization) / lam_pump
            self.d33 = 16.9e-12  # nonlinear coefficient [meter/Volt]
        else:
            self.ctype = nz_MgCLN_Gayer  # refractive index function
            self.pump_k = 2 * np.pi * nz_MgCLN_Gayer(lam_pump * 1e6, Temperature) / lam_pump
            self.d33 = 23.4e-12  # [meter/Volt]
        self.slab = PP_crystal_slab

        if waist_pump0 is None:
            self.waist_pump0 = np.sqrt(maxZ / self.pump_k)
        else:
            self.waist_pump0 = waist_pump0

        if r_scale0 is None:
            self.r_scale0 = self.waist_pump0
        else:
            self.r_scale0 = r_scale0

        assert pump_basis.lower() in ['lg', 'hg'], 'The beam structure is constructed as a combination ' \
                                                   'of LG or HG basis functions only'
        self.pump_basis = pump_basis
        self.pump_max_mode1 = pump_max_mode1
        self.pump_max_mode2 = pump_max_mode2
        self.initial_pump_coefficient = initial_pump_coefficient
        self.custom_pump_coefficient = custom_pump_coefficient
        self.pump_coefficient_path = pump_coefficient_path

        # number of modes for pump basis
        if pump_basis.lower() == 'lg':
            self.pump_n_modes1 = pump_max_mode1
            self.pump_n_modes2 = 2 * pump_max_mode2 + 1
        else:
            self.pump_n_modes1 = pump_max_mode1
            self.pump_n_modes2 = pump_max_mode2

        # Total number of pump modes
        self.pump_n_modes = self.pump_n_modes1 * self.pump_n_modes2

        self.initial_pump_waist = initial_pump_waist
        self.pump_waists_path = pump_waists_path

        self.crystal_basis = crystal_basis
        if crystal_basis:
            assert crystal_basis.lower() in ['ft', 'fb', 'lg', 'hg'], 'The crystal structure was constructed ' \
                                                                      'as a combination of FT, FB, LG or HG ' \
                                                                      'basis functions only'

            self.crystal_max_mode1 = crystal_max_mode1
            self.crystal_max_mode2 = crystal_max_mode2
            self.initial_crystal_coefficient = initial_crystal_coefficient
            self.custom_crystal_coefficient = custom_crystal_coefficient
            self.crystal_coefficient_path = crystal_coefficient_path

            # number of modes for crystal basis
            if crystal_basis.lower() in ['ft', 'fb', 'lg']:
                self.crystal_n_modes1 = crystal_max_mode1
                self.crystal_n_modes2 = 2 * crystal_max_mode2 + 1
            else:
                self.crystal_n_modes1 = crystal_max_mode1
                self.crystal_n_modes2 = crystal_max_mode2

            # Total number of crystal modes
            self.crystal_n_modes = self.crystal_n_modes1 * self.crystal_n_modes2

            self.initial_crystal_waist = initial_crystal_waist
            self.crystal_waists_path = crystal_waists_path


    def initial_pump_coefficients(
            self,
    ):

        if self.initial_pump_coefficient == "uniform":
            coeffs_real = np.ones(self.pump_n_modes, dtype=np.float32)
            coeffs_imag = np.ones(self.pump_n_modes, dtype=np.float32)

        elif self.initial_pump_coefficient == "random":

            self.key, pump_coeff_key = random.split(self.key)
            rand_real, rand_imag = random.split(pump_coeff_key)
            coeffs_real = random.normal(rand_real, (self.pump_n_modes,))
            coeffs_imag = random.normal(rand_imag, (self.pump_n_modes,))

        elif self.initial_pump_coefficient == "LG00":
            coeffs_real = np.zeros(self.pump_n_modes, dtype=np.float32)
            coeffs_imag = np.zeros(self.pump_n_modes, dtype=np.float32)
            coeffs_real = index_update(coeffs_real, 0 * (2 * self.pump_max_mode2 + 1) + self.pump_max_mode2, 1)

        elif self.initial_pump_coefficient == "custom":
            assert self.custom_pump_coefficient, 'for custom method, pump basis coefficients and ' \
                                                 'indexes must be selected'
            coeffs_real = np.zeros(self.pump_n_modes, dtype=np.float32)
            coeffs_imag = np.zeros(self.pump_n_modes, dtype=np.float32)
            for index, coeff in self.custom_pump_coefficient[REAL].items():
                coeffs_real = index_update(coeffs_real, index, coeff)

            for index, coeff in self.custom_pump_coefficient[IMAG].items():
                coeffs_real = coeffs_imag(coeffs_real, index, coeff)


        elif self.initial_pump_coefficient == "load":
            assert self.pump_coefficient_path, 'Path to pump coefficients must be defined'

            coeffs_real = np.load(os.path.join(self.pump_coefficient_path, 'PumpCoeffs_real.npy'))
            coeffs_imag = np.load(os.path.join(self.pump_coefficient_path, 'PumpCoeffs_imag.npy'))

        else:
            coeffs_real, coeffs_imag = None, None
            assert "ERROR: incompatible pump basis coefficients"

        normalization = np.sqrt(np.sum(np.abs(coeffs_real) ** 2 + np.abs(coeffs_imag) ** 2))
        coeffs_real = coeffs_real / normalization
        coeffs_imag = coeffs_imag / normalization

        return coeffs_real, coeffs_imag


    def initial_pump_waists(self):
        if self.initial_pump_waist == "waist_pump":
            waist_pump = np.ones(self.pump_n_modes, dtype=np.float32) * self.waist_pump0 * 1e5

        elif self.initial_pump_waist == "load":
            assert self.pump_waists_path, 'Path to pump waists must be defined'

            waist_pump = np.load(os.path.join(self.pump_coefficient_path, "PumpWaistCoeffs.npy")) * 1e-1

        else:
            waist_pump = None
            assert "ERROR: incompatible pump basis waists"

        return waist_pump


    def initial_crystal_coefficients(
            self,
    ):
        if not self.crystal_basis:
            return None, None

        elif self.initial_crystal_coefficient == "uniform":
            coeffs_real = np.ones(self.crystal_n_modes, dtype=np.float32)
            coeffs_imag = np.ones(self.crystal_n_modes, dtype=np.float32)

        elif self.initial_crystal_coefficient == "random":

            self.key, crystal_coeff_key = random.split(self.key)
            rand_real, rand_imag = random.split(crystal_coeff_key)
            coeffs_real = random.normal(rand_real, (self.crystal_n_modes,))
            coeffs_imag = random.normal(rand_imag, (self.crystal_n_modes,))

        elif self.initial_crystal_coefficient == "LG00":
            coeffs_real = np.zeros(self.crystal_n_modes, dtype=np.float32)
            coeffs_imag = np.zeros(self.crystal_n_modes, dtype=np.float32)
            coeffs_real = index_update(coeffs_real, 0 * (2 * self.crystal_max_mode2 + 1) + self.crystal_max_mode2, 1)

        elif self.initial_crystal_coefficient == "custom":
            assert self.custom_crystal_coefficient, 'for custom method, crystal basis coefficients and ' \
                                                    'indexes must be selected'
            coeffs_real = np.zeros(self.crystal_n_modes, dtype=np.float32)
            coeffs_imag = np.zeros(self.crystal_n_modes, dtype=np.float32)
            for index, coeff in self.custom_crystal_coefficient[REAL].items():
                coeffs_real = index_update(coeffs_real, index, coeff)

            for index, coeff in self.custom_crystal_coefficient[IMAG].items():
                coeffs_real = coeffs_imag(coeffs_real, index, coeff)

        elif self.initial_crystal_coefficient == "load":
            assert self.crystal_coefficient_path, 'Path to crystal coefficients must be defined'

            coeffs_real = np.load(os.path.join(self.crystal_coefficient_path, 'CrystalCoeffs_real.npy'))
            coeffs_imag = np.load(os.path.join(self.crystal_coefficient_path, 'CrystalCoeffs_imag.npy'))

        else:
            coeffs_real, coeffs_imag = None, None
            assert "ERROR: incompatible crystal basis coefficients"

        normalization = np.sqrt(np.sum(np.abs(coeffs_real) ** 2 + np.abs(coeffs_imag) ** 2))
        coeffs_real = coeffs_real / normalization
        coeffs_imag = coeffs_imag / normalization

        return coeffs_real, coeffs_imag


    def initial_crystal_waists(self):

        if not self.crystal_basis:
            return None

        if self.initial_crystal_waist == "r_scale0":
            r_scale = np.ones(self.crystal_n_modes, dtype=np.float32) * self.r_scale0 * 1e5

        elif self.initial_crystal_waist == "load":
            assert self.crystal_waists_path, 'Path to crystal waists must be defined'

            r_scale = np.load(os.path.join(self.crystal_coefficient_path, "CrystalWaistCoeffs.npy")) * 1e-1

        else:
            r_scale = None
            assert "ERROR: incompatible crystal basis waists"

        return r_scale
