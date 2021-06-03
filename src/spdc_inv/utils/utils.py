from abc import ABC
import scipy.special as sp
import jax.numpy as np
import math
from jax.ops import index_update


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
        inference: bool = False
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
        if inference:
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
            Hermite_dict[str(nx) + str(ny)] = Hermite_gauss(lam, refractive_index, W0, nx, ny, z, x, y)
    return np.array(list(Hermite_dict.values())), [*Hermite_dict]


def LaguerreBank(
        lam,
        refractive_index,
        W0,
        max_mode_p,
        max_mode_l,
        x,
        y,
        z=0
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

    Returns
    -------
    dictionary of Laguerre Gauss basis functions
    """
    Laguerre_dict = {}
    for p in range(max_mode_p):
        for l in range(-max_mode_l, max_mode_l+1):
            Laguerre_dict[str(p) + str(l)] = Laguerre_gauss(lam, refractive_index, W0, l, p, z, x, y)
    return np.array(list(Laguerre_dict.values())), [*Laguerre_dict]


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
                self.E = self._profile_laguerre_gauss(pump_coeffs_real, pump_coeffs_imag)

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





'''
make a beam from basis modes
pump_basis_arr  - a dictionary of basis modes. ordered in dictionary order (00,01,10,11)
pump_coeffs - the wieght of each mode in pump_basis_arr
these two have to have the same length!
'''
#@jit
def make_beam_from_coeffs(pump_basis_arr, pump_coeffs):
    if len(pump_coeffs) != len(pump_basis_arr):
        print('WRONG NUMBER OF PARAMETERS!!!')
        return
    return (pump_coeffs[:, None, None] * pump_basis_arr).sum(0)


def type_beam_from_pump_str(type, pump_basis_str, pump_coeffs, coeffs_str, pump_coeffs_gt=None):
    print_str = f'initial {type} pump coefficients string: {coeffs_str}\n\n'
    if len(pump_coeffs.shape) > 1:
        pump_coeffs = pump_coeffs[0]
    if pump_coeffs_gt is None:
        print_str += f'{type} coefficients:\n'
        for n, mode_x in enumerate(pump_basis_str):
            coeffs_str = ' : {:.4}\n'.format(pump_coeffs[n])
            print_str += type + mode_x + coeffs_str
    else:
        print_str += f'{type} coefficients \t ground truth \n'
        for n, mode_x in enumerate(pump_basis_str):
            coeffs_str = ' : {:.4}\t\t\t {:.4}\n'.format(pump_coeffs[n], pump_coeffs_gt[n])
            print_str += type + mode_x + coeffs_str
    return print_str


def type_waists_from_pump_str(type, pump_basis_str, waist_coeffs):
    print_str = '\n\n'

    if len(waist_coeffs.shape) > 1:
        waist_coeffs = waist_coeffs[0]

    print_str += f'{type} waist coefficients string:\n'
    for n, mode_x in enumerate(pump_basis_str):
        coeffs_str = ' : {:.4}um\n'.format(waist_coeffs[n])
        print_str += type + mode_x + coeffs_str
    return print_str


def type_poling_from_crystal_str(type, crystal_coeffs, crystal_str):
    print_str = f'\n\ninitial {type} crystal coefficients string: {crystal_str}\n\n'

    print_str += f'{type} coefficients:\n'
    for n in range(len(crystal_coeffs)):
        coeffs_str = ' : {:.4}\n'.format(crystal_coeffs[n])
        print_str += str(n) + coeffs_str

    return print_str


def type_waists_from_crystal_str(type, waist_coeffs):
    print_str = '\n\n'

    print_str += f'{type} r_scale coefficients string:\n'
    for n in range(len(waist_coeffs)):
        coeffs_str = ' : {:.4}um\n'.format(waist_coeffs[n])
        print_str += str(n) + coeffs_str

    return print_str


'''
unwrap_kron takes a Kronicker product of size M^2 x M^2 and turns is into an
M x M x M x M array. It is used only for illustration and not during the learning
'''
def unwrap_kron(G, C, M1, M2):
    for i in range(M1):
        for j in range(M2):
            for k in range(M1):
                for l in range(M2):
                    G[i, j, k, l] = C[k + M1 * i, l + M2 * j]
    return G


'''
TRACE_IT takes an M x M x M x M array representing a Kronecker product, 
and traces over 2 of its dimensions
'''
def trace_it(G, dim1, dim2):
    C = np.sum(G, axis=dim1)  # trace over dimesnion dim1
    C = np.sum(C, axis=dim2 - 1)  # trace over dimesnion dim2
    return C
