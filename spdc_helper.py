from jax import numpy as np, jit, nn, random
import math as math


###########################################
# Constants
###########################################
pi = np.pi
c = 2.99792458e8  # the speed of light in meter/sec
eps0 = 8.854187817e-12  # the vacuum permittivity, in Farad/meter.
h_bar = 1.054571800e-34  # Units are m^2 kg / s, taken from http://physics.nist.gov/cgi-bin/cuu/Value?hbar|search_for=planck

###########################################
# lambda funtions:
###########################################
SFG_idler_wavelength    = lambda lambda_p, lambda_s: lambda_p * lambda_s / (lambda_s - lambda_p)  # Compute the idler wavelength given pump and signal
E0                      = lambda P, n, W0: np.sqrt(P / (n * c * eps0 * np.pi * W0 ** 2))  # Calc amplitude
Fourier                 = lambda A: (np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(A))))  # Fourier
G1_Normalization        = lambda w: h_bar * w / (2 * eps0 * c)

#################################################################
# Create classes:
#  - beam: all parameters of a laser beam
#  - field: initializes E_out and E_vac
#  - crystal: all structure arrays
################################################################
'''
Class Crystal:
----------
initialize all strcutres of crystal
    - dx,dy,dz : resolution in x y and z axis
    - MaxX,MaxY : half the crystal length in x y 
    - MaxZ: crystal length in the z axis
    - Ref_ind : function for calcualting the refracive index of a crystal, accepts lambda in microns and temperature
    - slab_function: function for calculating the crystal slab at each z
    - d : half the second order nonlinear suscpetiblity
    - poliong period (optional): the poling period of the crystal
'''
class Crystal:
    def __init__(self, dx, dy, dz, MaxX, MaxY, MaxZ, Ref_ind, slab_function, d, period=0):
        self.dz = dz  # resolution of z axis
        self.dx = dx  # resolution of x axis
        self.dy = dy  # resolution of y axis
        self.MaxX = MaxX
        self.MaxY = MaxY
        self.MaxZ = MaxZ
        self.x = np.arange(-MaxX, MaxX, dx)  # x axis, length 2*MaxX (transverse)
        self.y = np.arange(-MaxY, MaxY, dy)  # y axis, length 2*MaxY  (transverse)
        self.z = np.arange(-MaxZ / 2, MaxZ / 2, dz)  # z axis, length MaxZ (propagation)
        self.ctype = Ref_ind  # refractive index function
        self.slab = slab_function
        self.d = d
        self.poling_period = period



'''
Class BEAM:
----------
compute everything to do with a beam.
    -lam - wavelength, in m
    -crystal -  class Crystal with refractive index function
    -T - temperature, in C
    -waist- beam waist in meters. optional
    -power - peak power of the beam, in W. optional
'''
class Beam:
    def __init__(self, lam, crystal, T, waist=0, power=0, max_mode=0):
        self.lam = lam  # wavelength
        self.waist = waist  # waist
        self.n = crystal.ctype(lam * 1e6, T)  # refractive index
        self.w = 2 * np.pi * c / lam  # frequency
        self.k = 2 * np.pi * crystal.ctype(lam * 1e6, T) / lam  # wave vector
        self.b = waist ** 2 * self.k  #
        self.power = power  # beam power
        if max_mode:
            self.hemite_dict = HermiteBank(lam, self.waist, self.waist, max_mode, crystal.x, crystal.y)
            self.profile     = []

    def create_profile(self, HG_parameters):
        self.profile = HG_parameters
        self.E = make_beam_from_HG(self.hemite_dict, HG_parameters)

'''
Class Field:
----------
initialize E_out and E_vac, for a given beam (class Beam) and crystal (class Crystal)
    E_out - as in the paper, the output field, initilized as 0
    E_vac - as in the paper, the vacuum field, initilized as gaussian noise
    kappa - coupling constant 
'''


class Field:
    def __init__(self, beam, crystal, vac_rnd):
        Nx = len(crystal.x)
        Ny = len(crystal.y)
        self.E_out = np.zeros([Nx, Ny], dtype=complex)
        vac = np.sqrt(h_bar * beam.w / (2 * eps0 * beam.n ** 2 * crystal.dx * crystal.dy * crystal.MaxZ))
        self.E_vac = vac * (vac_rnd[0] + 1j * vac_rnd[1]) / np.sqrt(2)
        self.kappa = 2 * 1j * beam.w ** 2 / (beam.k * c ** 2)  # we leave d_33 out and add it in the propagation function.
        self.k = beam.k


#######################################################
# Functions
######################################################
'''
A helper function to randomly initialize weights
for the coefficients
* m - number of coefficients
* scale - all weights should sum to 'scale'
'''
def random_params(m, key, scale=1):
  return scale * nn.softmax(random.normal(key, (m,)))

'''
Periodically poled crystal slab
create the crystal slab at point z in the crystal, for poling period 2pi/delta_k
'''
@jit
def PP_crystal_slab(delta_k, z):
    return np.sign(np.cos(np.abs(delta_k) * z))


'''
Refractive index for MgCLN, based on Gayer et al, APB 2008
lambda is in microns, T in celsius
'''
@jit
def nz_MgCLN_Gayer(lam, T):
    a = np.array([5.756, 0.0983, 0.2020, 189.32, 12.52, 1.32 * 10 ** (-2)])
    b = np.array([2.860 * 10 ** (-6), 4.700 * 10 ** (-8), 6.113 * 10 ** (-8), 1.516 * 10 ** (-4)])
    f = (T - 24.5) * (T + 570.82)

    n1 = a[0]
    n2 = b[0] * f
    n3 = (a[1] + b[1] * f) / (lam ** 2 - (a[2] + b[2] * f) ** 2)
    n4 = (a[3] + b[3] * f) / (lam ** 2 - (a[4]) ** 2)
    n5 = -a[5] * lam ** 2

    nz = np.sqrt(n1 + n2 + n3 + n4 + n5)
    return nz


'''
Crystal propagation
propagate through crystal using split step Fourier for 4 fields: e_out and E_vac, signal and idler.

'''
def crystal_prop(Pump, Siganl_field, Idler_field, crystal):
    for z in crystal.z:
        # propagate
        x  = crystal.x
        y  = crystal.y
        dz = crystal.dz

        # pump beam:
        E_pump = propagate(Pump.E, x ,y, Pump.k, z) * np.exp(1j * Pump.k * z)

        # crystal slab:
        PP = crystal.slab(crystal.poling_period, z)

        # cooupled wave equations - split step
        # signal:
        dEs_out_dz = Siganl_field.kappa * crystal.d * PP * E_pump * np.conj(Idler_field.E_vac)
        dEs_vac_dz = Siganl_field.kappa * crystal.d * PP * E_pump * np.conj(Idler_field.E_out)

        Siganl_field.E_out = Siganl_field.E_out + dEs_out_dz * dz
        Siganl_field.E_vac = Siganl_field.E_vac + dEs_vac_dz * dz

        # idler:
        dEi_out_dz = Idler_field.kappa * crystal.d * PP * E_pump * np.conj(Siganl_field.E_vac)
        dEi_vac_dz = Idler_field.kappa * crystal.d * PP * E_pump * np.conj(Siganl_field.E_out)

        Idler_field.E_out = Idler_field.E_out + dEi_out_dz * dz
        Idler_field.E_vac = Idler_field.E_vac + dEi_vac_dz * dz

        # propagate
        Siganl_field.E_out = propagate(Siganl_field.E_out, x, y, Siganl_field.k, dz) * np.exp(1j * Siganl_field.k * dz)
        Siganl_field.E_vac = propagate(Siganl_field.E_vac, x, y, Siganl_field.k, dz) * np.exp(1j * Siganl_field.k * dz)
        Idler_field.E_out  = propagate(Idler_field.E_out, x, y, Idler_field.k, dz) * np.exp(1j * Idler_field.k * dz)
        Idler_field.E_vac  = propagate(Idler_field.E_vac, x, y, Idler_field.k, dz) * np.exp(1j * Idler_field.k * dz)

    return


'''
Free space propagation of a Gaussian beam
Calcualtes the 2-D distribution of a gaussian beam at point z_point
calculation according to Boyd, second edition, 2008
inputs:
    - beam: class beam
    - crystal: crystal class
    - z_point: the z location of the beam
Boyd 2nd eddition       
'''

def Gaussian_beam_calc(beam, crystal, z_point):
    X, Y = np.meshgrid(crystal.x, crystal.y, indexing='ij')
    xi = 2 * (z_point) / beam.b
    tau = 1 / (1 + 1j * xi)
    E = E0(beam.power, beam.n, beam.waist) * tau * np.exp(-((X) ** 2 + (Y) ** 2) / (beam.waist ** 2) * tau) * np.exp(
        1j * beam.k * (z_point))
    return E


'''
Free Space propagation using the free space transfer function 
(two  dimensional), according to Saleh
 Function reciveing:        
        -Beam: class beam
        -x,y : spatial vectors
       -d: The distance to propagate
The output is the propagated field.
Using CGS, or MKS, Boyd 2nd eddition       
'''

@jit
def propagate(A, x, y, k, dz):
    dx = np.abs(x[1] - x[0])
    dy = np.abs(y[1] - y[0])
    # define the fourier vectors
    X, Y = np.meshgrid(x, y, indexing='ij')
    KX = 2 * np.pi * (X / dx) / (np.size(X, 1) * dx)
    KY = 2 * np.pi * (Y / dy) / (np.size(Y, 1) * dy)
    # The Free space transfer function of propagation, using the Fresnel approximation
    # (from "Engineering optics with matlab"/ing-ChungPoon&TaegeunKim):
    H_w = np.exp(-1j * dz * (np.square(KX) + np.square(KY)) / (2 * k))
    # (inverse fast Fourier transform shift). For matrices, ifftshift(X) swaps the
    # first quadrant with the third and the second quadrant with the fourth.
    H_w = np.fft.ifftshift(H_w)
    # Fourier Transform: move to k-space
    G = np.fft.fft2(A)  # The two-dimensional discrete Fourier transform (DFT) of A.
    # propoagte in the fourier space
    F = np.multiply(G, H_w)
    # inverse Fourier Transform: go back to real space
    Eout = np.fft.ifft2(F)  # [in real space]. E1 is the two-dimensional INVERSE discrete Fourier transform (DFT) of F1
    return Eout


'''
Hermite polynomial of rank n Hn(x)
'''


def HermiteP(n, x):
    if n == 0:
        return 1
    elif n == 1:
        return 2 * x
    else:
        return 2 * x * HermiteP(n - 1, x) - 2 * (n - 1) * HermiteP(n - 2, x)


'''
Hermite Gausse in 2-D
return the Hermite-Gaussian beam of order n in 1 d
recives:
      - lambda =  the wavelength
      - W0x, W0y = beam waist in x and y axis
      - n , m = order of the H-G beam
      - z = the place in z to calculate for
      - x,y  = matrixes of x and y to calculate Un(x,z) for.
      - P  = the total power of the beam. If this is not given then it is
       set to 1W (all in Air).
'''


def Hermite_gause2Dxy(lam, W0x, W0y, n, m, z, x, y):
    W0 = np.sqrt(2 * W0x ** 2)
    k = 2 * np.pi / lam
    z0 = np.pi * W0 ** 2 / lam  # Rayleigh range
    Wx = W0x * np.sqrt(1 + (z / z0) ** 2)  # w(z), the variation of the spot size
    invR = z / ((z ** 2) + (z0 ** 2))  # radius of curvature
    qx = 1 / (invR - 1j * lam / (np.pi * (Wx ** 2)))  # the complex beam parameter
    q0 = 1j * z0
    coefx = (2 / np.pi) ** 0.25 * np.sqrt(1 / (2 ** n * math.factorial(n) * W0x)) * np.sqrt(q0 / qx) * (
                q0 * np.conj(qx) / (np.conj(q0) * qx)) ** (n / 2)
    Unx = coefx * HermiteP(n, np.sqrt(2) * x / Wx) * np.exp(-1j * k * ((x ** 2) * (1 / (2 * qx))))

    W0 = np.sqrt(2 * W0y ** 2);
    z0 = np.pi * W0 ** 2 / lam;  # Rayleigh range
    Wy = W0y * np.sqrt(1 + (z / z0) ** 2);  # w(z), the variation of the spot size
    invR = z / ((z ** 2) + (z0 ** 2));  # radius of curvature
    qy = 1 / (invR - 1j * lam / (np.pi * (Wy ** 2)));  # the complex beam parameter
    q0 = 1j * z0;
    coefy = (2 / np.pi) ** 0.25 * np.sqrt(1 / (2 ** m * math.factorial(m) * W0y)) * np.sqrt(q0 / qy) * (
                q0 * np.conj(qy) / (np.conj(q0) * qy)) ** (m / 2);
    Uny = coefy * HermiteP(m, np.sqrt(2) * y / Wy) * np.exp(-1j * k * ((y ** 2) * (1 / (2 * qy))))

    # plt.imshow(np.abs(np.dot(Uny.reshape(len(Unx),1),Unx.reshape(1,len(Unx)))))

    return Unx, Uny


'''
HemiteBank returns a dictionary of Hermite gausee
'''


def HermiteBank(lam, W0x, W0y, max_mode, x, y):
    hermite_dictx = {}
    hermite_dicty = {}
    hermite_dict = {}

    for n in range(max_mode):
        hermite_dictx[str(n)], hermite_dicty[str(n)] = Hermite_gause2Dxy(lam, W0x, W0y, n, n, 0, x, y)

    for n in range(max_mode):
        for m in range(max_mode):
            Uny = hermite_dicty[str(m)]
            Unx = hermite_dictx[str(n)]
            hermite_dict[str(n) + str(m)] = np.dot(Uny.reshape(len(Unx), 1), Unx.reshape(1, len(Unx)))

    return hermite_dict


'''
make a beam from HG modes
hermite_dict  - a dictionary of HG modes. ordered in dictionary order (00,01,10,11)
HG_parameters - the wieght of each mode in hermite_dict
these two have to have the same length!
'''


def make_beam_from_HG(hermite_dict, HG_parameters, PRINT=0):
    final = 0
    # print_str = str([])
    if len(HG_parameters) != len(hermite_dict):
        print('WRONG NUMBER OF PARAMETERS!!!')
        return
    for n, (mode_x, HG) in enumerate(hermite_dict.items()):
        final = final + HG_parameters[n] * HG
        # if HG_parameters[n]:
        #     print_str = print_str + ' + ' + str(HG_parameters[n]) + 'HG' + mode_x
    # if PRINT:
    #     print(print_str)
    return final
