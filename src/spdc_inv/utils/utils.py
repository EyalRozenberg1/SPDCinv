from abc import ABC
from jax.ops import index_update, index_add, index
from typing import List, Union, Any
from spdc_inv.utils.defaults import QUBIT

import scipy.special as sp
import jax.numpy as np
import math


# Constants:
pi      = np.pi
c       = 2.99792458e8  # speed of light [meter/sec]
eps0    = 8.854187817e-12  # vacuum permittivity [Farad/meter]
h_bar   = 1.054571800e-34  # [m^2 kg / s], taken from http://physics.nist.gov/cgi-bin/cuu/Value?hbar|search_for=planck

# lambda functions:
G1_Normalization        = lambda w: h_bar * w / (2 * eps0 * c)
I                       = lambda A, n: 2 * n * eps0 * c * np.abs(A) ** 2  # Intensity
Power2D                 = lambda A, n, dx, dy: np.sum(I(A, n)) * dx * dy

# Compute the idler wavelength given pump and signal
SFG_idler_wavelength    = lambda lambda_p, lambda_s: lambda_p * lambda_s / (lambda_s - lambda_p)


def PP_crystal_slab(
        delta_k,
        z,
        crystal_profile,
        inference=None
):
    """
    Periodically poled crystal slab.
    create the crystal slab at point z in the crystal, for poling period 2pi/delta_k

    Parameters
    ----------
    delta_k: k mismatch
    z: longitudinal point for generating poling pattern
    crystal_profile: Crystal 3D hologram (if None, ignore)
    inference: (True/False) if in inference mode, we include more coefficients in the poling
                description for better validation

    Returns Periodically poled crystal slab at point z
    -------

    """
    if crystal_profile is None:
        return np.sign(np.cos(np.abs(delta_k) * z))
    else:
        magnitude = np.abs(crystal_profile)
        phase = np.angle(crystal_profile)
        if inference is not None:
            max_order_fourier = 20
            poling = 0
            magnitude = magnitude / magnitude.max()
            DutyCycle = np.arcsin(magnitude) / np.pi
            for m in range(max_order_fourier):
                if m == 0:
                    poling = poling + 2 * DutyCycle - 1
                else:
                    poling = poling + (2 / (m * np.pi)) * \
                             np.sin(m * pi * DutyCycle) * 2 * np.cos(m * phase + m * np.abs(delta_k) * z)
            return poling
        else:
            return (2 / np.pi) * np.exp(1j * (np.abs(delta_k) * z)) * magnitude * np.exp(1j * phase)


def HermiteBank(
        lam,
        refractive_index,
        W0,
        max_mode_x,
        max_mode_y,
        x,
        y,
        z=0
):
    """
    generates a dictionary of Hermite Gauss basis functions

    Parameters
    ----------
    lam; wavelength
    refractive_index: refractive index
    W0: beam waist
    max_mode_x: maximum projection mode 1st axis
    max_mode_y: maximum projection mode 2nd axis
    x: transverse points, x axis
    y: transverse points, y axis
    z: projection longitudinal position

    Returns
    -------
    dictionary of Hermite Gauss basis functions
    """
    Hermite_dict = {}
    for nx in range(max_mode_x):
        for ny in range(max_mode_y):
            Hermite_dict[f'|HG{nx}{ny}>'] = Hermite_gauss(lam, refractive_index, W0, nx, ny, z, x, y)
    return np.array(list(Hermite_dict.values())), [*Hermite_dict]


def LaguerreBank(
        lam,
        refractive_index,
        W0,
        max_mode_p,
        max_mode_l,
        x,
        y,
        z=0,
        get_dict: bool = False,
):
    """
    generates a dictionary of Laguerre Gauss basis functions

    Parameters
    ----------
    lam; wavelength
    refractive_index: refractive index
    W0: beam waist
    max_mode_p: maximum projection mode 1st axis
    max_mode_l: maximum projection mode 2nd axis
    x: transverse points, x axis
    y: transverse points, y axis
    z: projection longitudinal position
    get_dict: (True/False) if True, the function will return a dictionary,
              else the dictionary is splitted to basis functions np.array and list of dictionary keys.

    Returns
    -------
    dictionary of Laguerre Gauss basis functions
    """
    Laguerre_dict = {}
    for p in range(max_mode_p):
        for l in range(-max_mode_l, max_mode_l + 1):
            Laguerre_dict[f'|LG{p}{l}>'] = Laguerre_gauss(lam, refractive_index, W0, l, p, z, x, y)
    if get_dict:
        return Laguerre_dict

    return np.array(list(Laguerre_dict.values())), [*Laguerre_dict]


def TomographyBank(
        lam,
        refractive_index,
        W0,
        max_mode_p,
        max_mode_l,
        x,
        y,
        z=0,
        relative_phase: List[Union[Union[int, float], Any]] = None,
        tomography_quantum_state: str = None
):
    """
    generates a dictionary of basis function with projections into two orthogonal LG bases and mutually unbiased
    bases (MUBs). The MUBs are constructed from superpositions of the two orthogonal LG bases.
    according to: https://doi.org/10.1364/AOP.11.000067

    Parameters
    ----------
    lam; wavelength
    refractive_index: refractive index
    W0: beam waist
    max_mode_p: maximum projection mode 1st axis
    max_mode_l: maximum projection mode 2nd axis
    x: transverse points, x axis
    y: transverse points, y axis
    z: projection longitudinal position
    relative_phase: The relative phase between the mutually unbiased bases (MUBs) states
    tomography_quantum_state: the current quantum state we calculate it tomography matrix.
                              currently we support: qubit/qutrit

    Returns
    -------
    dictionary of bases functions used for constructing the tomography matrix
    """

    TOMO_dict = \
        LaguerreBank(
            lam,
            refractive_index,
            W0,
            max_mode_p,
            max_mode_l,
            x, y, z,
            get_dict=True)

    if tomography_quantum_state is QUBIT:
        del TOMO_dict['|LG00>']

    LG_modes, LG_string = np.array(list(TOMO_dict.values())), [*TOMO_dict]

    for m in range(len(TOMO_dict) - 1, -1, -1):
        for n in range(m - 1, -1, -1):
            for k in range(len(relative_phase)):
                TOMO_dict[f'{LG_string[m]}+e^j{str(relative_phase[k]/np.pi)}π{LG_string[n]}'] = \
                    (1 / np.sqrt(2)) * (LG_modes[m] + np.exp(1j * relative_phase[k]) * LG_modes[n])

    return np.array(list(TOMO_dict.values())), [*TOMO_dict]


def Hermite_gauss(lam, refractive_index, W0, nx, ny, z, X, Y, coef=None):
    """
    Hermite Gauss in 2D

    Parameters
    ----------
    lam: wavelength
    refractive_index: refractive index
    W0: beam waists
    n, m: order of the HG beam
    z: the place in z to calculate for
    x,y: matrices of x and y
    coef

    Returns
    -------
    Hermite-Gaussian beam of order n,m in 2D
    """
    k = 2 * np.pi * refractive_index / lam
    z0 = np.pi * W0 ** 2 * refractive_index / lam  # Rayleigh range
    Wz = W0 * np.sqrt(1 + (z / z0) ** 2)  # w(z), the variation of the spot size

    invR = z / ((z ** 2) + (z0 ** 2))  # radius of curvature
    gouy = (nx + ny + 1)*np.arctan(z/z0)
    if coef is None:
        coefx = np.sqrt(np.sqrt(2/pi) / (2**nx * math.factorial(nx)))
        coefy = np.sqrt(np.sqrt(2/pi) / (2**ny * math.factorial(ny)))
        coef = coefx * coefy
    U = coef * \
        (W0/Wz) * np.exp(-(X**2 + Y**2) / Wz**2) * \
        HermiteP(nx, np.sqrt(2) * X / Wz) * \
        HermiteP(ny, np.sqrt(2) * Y / Wz) * \
        np.exp(-1j * (k * (X**2 + Y**2) / 2) * invR) * \
        np.exp(1j * gouy)

    return U


def Laguerre_gauss(lam, refractive_index, W0, l, p, z, x, y, coef=None):
    """
    Laguerre Gauss in 2D

    Parameters
    ----------
    lam: wavelength
    refractive_index: refractive index
    W0: beam waists
    l, p: order of the LG beam
    z: the place in z to calculate for
    x,y: matrices of x and y
    coef

    Returns
    -------
    Laguerre-Gaussian beam of order l,p in 2D
    """
    k = 2 * np.pi * refractive_index / lam
    z0 = np.pi * W0 ** 2 * refractive_index / lam  # Rayleigh range
    Wz = W0 * np.sqrt(1 + (z / z0) ** 2)  # w(z), the variation of the spot size
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    invR = z / ((z ** 2) + (z0 ** 2))  # radius of curvature
    gouy = (np.abs(l)+2*p+1)*np.arctan(z/z0)
    if coef is None:
        coef = np.sqrt(2*math.factorial(p)/(np.pi * math.factorial(p + np.abs(l))))

    U = coef * \
        (W0/Wz)*(r*np.sqrt(2)/Wz)**(np.abs(l)) * \
        np.exp(-r**2 / Wz**2) * \
        LaguerreP(p, l, 2 * r**2 / Wz**2) * \
        np.exp(-1j * (k * r**2 / 2) * invR) * \
        np.exp(-1j * l * phi) * \
        np.exp(1j * gouy)
    return U


def HermiteP(n, x):
    """
    Hermite polynomial of rank n Hn(x)

    Parameters
    ----------
    n: order of the LG beam
    x: matrix of x

    Returns
    -------
    Hermite polynomial
    """
    if n == 0:
        return 1
    elif n == 1:
        return 2 * x
    else:
        return 2 * x * HermiteP(n - 1, x) - 2 * (n - 1) * HermiteP(n - 2, x)


def LaguerreP(p, l, x):
    """
    Generalized Laguerre polynomial of rank p,l L_p^|l|(x)

    Parameters
    ----------
    l, p: order of the LG beam
    x: matrix of x

    Returns
    -------
    Generalized Laguerre polynomial
    """
    if p == 0:
        return 1
    elif p == 1:
        return 1 + np.abs(l)-x
    else:
        return ((2*p-1+np.abs(l)-x)*LaguerreP(p-1, l, x) - (p-1+np.abs(l))*LaguerreP(p-2, l, x))/p


class Beam(ABC):
    """
    A class that holds everything to do with a beam
    """
    def __init__(self,
                 lam: float,
                 ctype,
                 polarization: str,
                 T: float,
                 power: float = 0):

        """

        Parameters
        ----------
        lam: beam's wavelength
        ctype: function that holds crystal type fo calculating refractive index
        polarization: Polarization of the beam
        T: crystal's temperature [Celsius Degrees]
        power: beam power [watt]
        """

        self.lam          = lam
        self.n            = ctype(lam * 1e6, T, polarization)  # refractive index
        self.w            = 2 * np.pi * c / lam  # frequency
        self.k            = 2 * np.pi * ctype(lam * 1e6, T, polarization) / lam  # wave vector
        self.power        = power  # beam power


class Beam_profile(ABC):
    def __init__(
            self,
            pump_coeffs_real,
            pump_coeffs_imag,
            waist_pump,
            power_pump,
            x,
            y,
            dx,
            dy,
            max_mode1,
            max_mode2,
            pump_basis: str,
            lam_pump,
            refractive_index,
            learn_pump: bool = False,
            z: float = 0.,
    ):

        self.x = x
        self.y = y
        self.z = z
        self.learn_pump = learn_pump
        self.lam_pump   = lam_pump
        self.pump_basis = pump_basis
        self.max_mode1  = max_mode1
        self.max_mode2  = max_mode2
        self.power      = power_pump
        self.crystal_dx = dx
        self.crystal_dy = dy
        self.refractive_index = refractive_index

        if self.pump_basis.lower() == 'lg':  # Laguerre-Gauss
            self.coef = np.zeros(len(waist_pump), dtype=np.float32)
            idx = 0
            for p in range(self.max_mode1):
                for l in range(-self.max_mode2, self.max_mode2 + 1):

                    self.coef = index_update(
                        self.coef, idx,
                        np.sqrt(2 * math.factorial(p) / (np.pi * math.factorial(p + np.abs(l))))
                    )

                    idx += 1

            if not learn_pump:
                self.E = self._profile_laguerre_gauss(pump_coeffs_real, pump_coeffs_imag, waist_pump)

        elif self.pump_basis.lower() == "hg":  # Hermite-Gauss
            self.coef = np.zeros(len(waist_pump), dtype=np.float32)
            idx = 0
            for nx in range(self.max_mode1):
                for ny in range(self.max_mode2):
                    self.coef = index_update(
                        self.coef, idx,
                        np.sqrt(np.sqrt(2 / pi) / (2 ** nx * math.factorial(nx))) *
                        np.sqrt(np.sqrt(2 / pi) / (2 ** ny * math.factorial(ny))))

                    idx += 1

            if not learn_pump:
                self.E = self._profile_hermite_gauss(pump_coeffs_real, pump_coeffs_imag, waist_pump)


    def create_profile(self, pump_coeffs_real, pump_coeffs_imag, waist_pump):
        if self.learn_pump:
            if self.pump_basis.lower() == 'lg':  # Laguerre-Gauss
                self.E = self._profile_laguerre_gauss(pump_coeffs_real, pump_coeffs_imag, waist_pump)

            elif self.pump_basis.lower() == 'hg':  # Hermite-Gauss
                self.E = self._profile_hermite_gauss(pump_coeffs_real, pump_coeffs_imag, waist_pump)

    def _profile_laguerre_gauss(
            self,
            pump_coeffs_real,
            pump_coeffs_imag,
            waist_pump
    ):
        coeffs = pump_coeffs_real + 1j * pump_coeffs_imag
        [X, Y] = np.meshgrid(self.x, self.y)
        pump_profile = 0.
        idx = 0
        for p in range(self.max_mode1):
            for l in range(-self.max_mode2, self.max_mode2 + 1):
                pump_profile += coeffs[idx] * \
                                Laguerre_gauss(self.lam_pump, self.refractive_index,
                                               waist_pump[idx] * 1e-5, l, p, self.z, X, Y, self.coef[idx])
                idx += 1

        pump_profile = fix_power(pump_profile, self.power, self.refractive_index,
                                 self.crystal_dx, self.crystal_dy)[np.newaxis, :, :]
        return pump_profile

    def _profile_hermite_gauss(
            self,
            pump_coeffs_real,
            pump_coeffs_imag,
            waist_pump
    ):

        coeffs = pump_coeffs_real + 1j * pump_coeffs_imag
        [X, Y] = np.meshgrid(self.x, self.y)
        pump_profile = 0.
        idx = 0
        for nx in range(self.max_mode1):
            for ny in range(self.max_mode2):
                pump_profile += coeffs[idx] * \
                                Hermite_gauss(self.lam_pump, self.refractive_index,
                                              waist_pump[idx] * 1e-5, nx, ny, self.z, X, Y, self.coef[idx])
                idx += 1

        pump_profile = fix_power(pump_profile, self.power, self.refractive_index,
                                 self.crystal_dx, self.crystal_dy)[np.newaxis, :, :]
        return pump_profile


class Crystal_hologram(ABC):
    def __init__(
            self,
            crystal_coeffs_real,
            crystal_coeffs_imag,
            r_scale,
            x,
            y,
            max_mode1,
            max_mode2,
            crystal_basis,
            lam_signal,
            refractive_index,
            learn_crystal: bool = False,
            z: float = 0.,
    ):

        self.x = x
        self.y = y
        self.z = z
        self.learn_crystal        = learn_crystal
        self.refractive_index     = refractive_index
        self.lam_signal           = lam_signal
        self.crystal_basis        = crystal_basis
        self.max_mode1 = max_mode1
        self.max_mode2 = max_mode2


        if crystal_basis.lower() == 'ft':  # Fourier-Taylor
            if not learn_crystal:
                self.crystal_profile = self._profile_fourier_taylor(crystal_coeffs_real, crystal_coeffs_imag, r_scale)

        elif crystal_basis.lower() == 'fb':  # Fourier-Bessel

            [X, Y] = np.meshgrid(self.x, self.y)
            self.coef = np.zeros(len(r_scale), dtype=np.float32)
            idx = 0
            for p in range(self.max_mode1):
                for l in range(-self.max_mode2, self.max_mode2 + 1):
                    rad = np.sqrt(X ** 2 + Y ** 2) / (r_scale[idx] * 1e-5)
                    self.coef = index_update(
                        self.coef, idx,
                        sp.jv(0, sp.jn_zeros(0, p + 1)[-1] * rad)
                    )
                    idx += 1

            if not learn_crystal:
                self.crystal_profile = self._profile_fourier_bessel(crystal_coeffs_real, crystal_coeffs_imag)

        elif crystal_basis.lower() == 'lg':  # Laguerre-Gauss

            self.coef = np.zeros(len(r_scale), dtype=np.float32)
            idx = 0
            for p in range(self.max_mode1):
                for l in range(-self.max_mode2, self.max_mode2 + 1):
                    self.coef = index_update(
                        self.coef, idx,
                        np.sqrt(2 * math.factorial(p) / (np.pi * math.factorial(p + np.abs(l))))
                    )
                    idx += 1

            if not learn_crystal:
                self.crystal_profile = self._profile_laguerre_gauss(crystal_coeffs_real, crystal_coeffs_imag, r_scale)

        elif crystal_basis.lower() == 'hg':  # Hermite-Gauss

            self.coef = np.zeros(len(r_scale), dtype=np.float32)
            idx = 0
            for m in range(self.max_mode1):
                for n in range(self.max_mode2):
                    self.coef = index_update(
                        self.coef, idx,
                        np.sqrt(np.sqrt(2 / pi) / (2 ** m * math.factorial(m))) *
                        np.sqrt(np.sqrt(2 / pi) / (2 ** n * math.factorial(n)))
                    )

                    idx += 1

            if not learn_crystal:
                self.crystal_profile = self._profile_hermite_gauss(crystal_coeffs_real, crystal_coeffs_imag, r_scale)

    def create_profile(
            self,
            crystal_coeffs_real,
            crystal_coeffs_imag,
            r_scale,
    ):
        if self.learn_crystal:
            if self.crystal_basis.lower() == 'ft':  # Fourier-Taylor
                self.crystal_profile = self._profile_fourier_taylor(crystal_coeffs_real, crystal_coeffs_imag, r_scale)

            elif self.crystal_basis.lower() == 'fb':  # Fourier-Bessel
                self.crystal_profile = self._profile_fourier_bessel(crystal_coeffs_real, crystal_coeffs_imag)

            elif self.crystal_basis.lower() == 'lg':  # Laguerre-Gauss
                self.crystal_profile = self._profile_laguerre_gauss(crystal_coeffs_real, crystal_coeffs_imag, r_scale)

            elif self.crystal_basis.lower() == 'hg':  # Hermite-Gauss
                self.crystal_profile = self._profile_hermite_gauss(crystal_coeffs_real, crystal_coeffs_imag, r_scale)

    def _profile_fourier_taylor(
            self,
            crystal_coeffs_real,
            crystal_coeffs_imag,
            r_scale,
    ):
        coeffs = crystal_coeffs_real + 1j * crystal_coeffs_imag
        [X, Y] = np.meshgrid(self.x, self.y)
        phi_angle = np.arctan2(Y, X)
        crystal_profile = 0.
        idx = 0
        for p in range(self.max_mode1):
            for l in range(-self.max_mode2, self.max_mode2 + 1):
                rad = np.sqrt(X**2 + Y**2) / (r_scale[idx] * 1e-5)
                crystal_profile += coeffs[idx] * rad**p * np.exp(-rad**2) * np.exp(-1j * l * phi_angle)
                idx += 1

        return crystal_profile

    def _profile_fourier_bessel(
            self,
            crystal_coeffs_real,
            crystal_coeffs_imag,
    ):
        coeffs = crystal_coeffs_real + 1j * crystal_coeffs_imag
        [X, Y] = np.meshgrid(self.x, self.y)
        phi_angle = np.arctan2(Y, X)
        crystal_profile = 0.
        idx = 0
        for p in range(self.max_mode1):
            for l in range(-self.max_mode2, self.max_mode2 + 1):
                crystal_profile += coeffs[idx] * self.coef[idx] * np.exp(-1j * l * phi_angle)
                idx += 1

        return crystal_profile

    def _profile_laguerre_gauss(
            self,
            crystal_coeffs_real,
            crystal_coeffs_imag,
            r_scale,
    ):
        coeffs = crystal_coeffs_real + 1j * crystal_coeffs_imag
        [X, Y] = np.meshgrid(self.x, self.y)
        idx = 0
        crystal_profile = 0.
        for p in range(self.max_mode1):
            for l in range(-self.max_mode2, self.max_mode2 + 1):
                crystal_profile += coeffs[idx] * \
                                   Laguerre_gauss(self.lam_signal, self.refractive_index,
                                                  r_scale[idx] * 1e-5, l, p, self.z, X, Y, self.coef[idx])
                idx += 1

        return crystal_profile

    def _profile_hermite_gauss(
            self,
            crystal_coeffs_real,
            crystal_coeffs_imag,
            r_scale,
    ):
        coeffs = crystal_coeffs_real + 1j * crystal_coeffs_imag
        [X, Y] = np.meshgrid(self.x, self.y)
        idx = 0
        crystal_profile = 0.
        for m in range(self.max_mode1):
            for n in range(self.max_mode2):
                crystal_profile += coeffs[idx] * \
                                   Hermite_gauss(self.lam_signal, self.refractive_index,
                                                 r_scale[idx] * 1e-5, m, n, self.z, X, Y, self.coef[idx])

                idx += 1

        return crystal_profile


def fix_power(
        A,
        power,
        n,
        dx,
        dy
):
    """
    The function takes a field A and normalizes in to have the power indicated

    Parameters
    ----------
    A
    power
    n
    dx
    dy

    Returns
    -------

    """
    output = A * np.sqrt(power) / np.sqrt(Power2D(A, n, dx, dy))
    return output


class DensMat(ABC):
    """
    A class that holds tomography dimensions and
    tensors used for calculating qubit and qutrit tomography
    """

    def __init__(
            self,
            projection_n_state2,
            tomography_dimension
    ):
        assert tomography_dimension in [2, 3], "tomography_dimension must be 2 or 3, " \
                                               f"got {tomography_dimension}"

        self.projection_n_state2 = projection_n_state2
        self.tomography_dimension = tomography_dimension
        self.rotation_mats, self.masks = self.dens_mat_tensors()

    def dens_mat_tensors(
            self
    ):
        rot_mats_tensor = np.zeros([self.tomography_dimension ** 4,
                                    self.tomography_dimension ** 2,
                                    self.tomography_dimension ** 2],
                                   dtype='complex64')

        masks_tensor = np.zeros([self.tomography_dimension ** 4,
                                 self.projection_n_state2,
                                 self.projection_n_state2],
                                dtype='complex64')

        if self.tomography_dimension == 2:
            mats = (
                np.eye(2, dtype='complex64'),
                (1 / np.sqrt(2)) * np.array([[0, 1], [1, 0]]),
                (1 / np.sqrt(2)) * np.array([[0, -1j], [1j, 0]]),
                np.array([[1, 0], [0, -1]])
            )

            vecs = (
                np.array([1, 1, 0, 0, 0, 0]),
                (1 / np.sqrt(2)) * np.array([0, 0, 1, -1, 0, 0]),
                (1 / np.sqrt(2)) * np.array([0, 0, 0, 0, 1, -1]),
                np.array([1, -1, 0, 0, 0, 0])
            )

        else:  # tomography_dimension == 3
            mats = (
                np.eye(3, dtype='complex64'),
                np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
                np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
                np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]]),
                np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
                np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]]),
                np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
                np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]]),
                (1 / np.sqrt(3)) * np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]])
            )

            vecs = (
                np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
               np.array([1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
               (1 / np.sqrt(2)) * np.array([0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
               (1 / np.sqrt(2)) * np.array([0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0]),
               (1 / np.sqrt(2)) * np.array([0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0]),
               (1 / np.sqrt(2)) * np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0]),
               (1 / np.sqrt(2)) * np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0]),
               (1 / np.sqrt(2)) * np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1]),
               (np.sqrt(3) / 3) * np.array([1, 1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            )

        counter = 0

        for m in range(self.tomography_dimension ** 2):
            for n in range(self.tomography_dimension ** 2):
                norm1 = np.trace(mats[m] @ mats[m])
                norm2 = np.trace(mats[n] @ mats[n])
                mat1 = mats[m] / norm1
                mat2 = mats[n] / norm2
                rot_mats_tensor = index_add(rot_mats_tensor, index[counter, :, :], np.kron(mat1, mat2))
                mask = np.dot(vecs[m].reshape(self.projection_n_state2, 1),
                              np.transpose((vecs[n]).reshape(self.projection_n_state2, 1)))
                masks_tensor = index_add(masks_tensor, index[counter, :, :], mask)
                counter = counter + 1

        return rot_mats_tensor, masks_tensor
