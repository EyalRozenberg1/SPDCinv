import jax.random as random
from jax import value_and_grad, pmap, lax
from jax.experimental import optimizers
import matplotlib.pyplot as plt
from functools import partial
import os, time
from datetime import datetime
import numpy as onp
import scipy.special as sp
import jax.numpy as np
import math
from jax import jit
from jax.ops import index_update


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
G1_Normalization        = lambda w: h_bar * w / (2 * eps0 * c)
I                       = lambda A, n: 2 * n * eps0 * c * np.abs(A) ** 2  # Intensity
Power2D                 = lambda A, n, dx, dy: np.sum(I(A, n)) * dx * dy

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
    def __init__(self, dx, dy, dz, MaxX, MaxY, MaxZ, d, period=0, learn_crystal_coeffs=False):
        self.dz   = dz  # resolution of z axis
        self.dx   = dx  # resolution of x axis
        self.dy   = dy  # resolution of y axis
        self.MaxX = MaxX
        self.MaxY = MaxY
        self.MaxZ = MaxZ
        self.x    = np.arange(-MaxX, MaxX, dx)  # x axis, length 2*MaxX (transverse)
        self.y    = np.arange(-MaxY, MaxY, dy)  # y axis, length 2*MaxY  (transverse)
        self.z    = np.arange(-MaxZ / 2, MaxZ / 2, dz)  # z axis, length MaxZ (propagation)
        self.ctype = n_KTP_Kato  # refractive index function
        if learn_crystal_coeffs:
            self.slab = PP_crystal_slab_2D
        else:
            self.slab = PP_crystal_slab
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
    def __init__(self, lam, crystal, ax, T, waist, power=0,
                 type='LG', max_mode1=0, max_mode2=0, z=0, n_coeff_pump=None, learn_pump_coeffs=True, learn_pump_waists=True):

        self.lam          = lam
        self.n            = crystal.ctype(lam * 1e6, T, ax)  # refractive index
        self.w            = 2 * np.pi * c / lam  # frequency
        self.k            = 2 * np.pi * crystal.ctype(lam * 1e6, T, ax) / lam  # wave vector
        self.power        = power  # beam power
        self.crystal_dx   = crystal.dx
        self.crystal_dy   = crystal.dy
        self.type         = type

        self.learn_pump_coeffs   = learn_pump_coeffs
        self.pump_coeffs         = None
        self.learn_pump_waists   = learn_pump_waists
        self.waist               = waist


        if n_coeff_pump is not None:
            self.X, self.Y    = np.meshgrid(crystal.x, crystal.y)
            self.z            = z
            self.max_mode1    = max_mode1
            self.max_mode2    = max_mode2
            self.lam          = lam
            self.n_coeff_pump = n_coeff_pump


        if max_mode1 and max_mode2:
            if self.type == 'LG':
                if n_coeff_pump is None:
                    [X, Y] = np.meshgrid(crystal.x, crystal.y)
                    self.pump_basis_arr, self.pump_basis_str = LaguerreBank(lam, self.n, waist,
                                                                            max_mode1, max_mode2, X, Y, z)
                else:
                    self.max_mode_l     = int((self.max_mode1 - 1) / 2)
                    self.max_mode_p     = self.max_mode2

                    self.coef = np.zeros(self.n_coeff_pump, dtype=np.float32)
                    idx = 0
                    for p in range(self.max_mode_p):
                        for l in range(-self.max_mode_l, self.max_mode_l + 1):

                            self.coef = index_update(
                                self.coef, idx,
                                np.sqrt(2 * math.factorial(p) / (np.pi * math.factorial(p + np.abs(l))))
                            )

                            idx += 1

                    self.pump_basis_arr, self.pump_basis_str = LaguerreBank(lam, self.n, waist[0] * 1e-5,
                                                                            max_mode1, max_mode2, self.X, self.Y, z)

            elif self.type == "HG":
                if n_coeff_pump is None:
                    [X, Y] = np.meshgrid(crystal.x, crystal.y)
                    self.pump_basis_arr, self.pump_basis_str = HermiteBank(lam, self.n, waist,
                                                                           max_mode1, max_mode2, X, Y, z)

                else:
                    self.max_mode_x     = self.max_mode1
                    self.max_mode_y     = self.max_mode2

                    self.coef = np.zeros(self.n_coeff_pump, dtype=np.float32)
                    idx = 0
                    for ny in range(self.max_mode_y):
                        for nx in range(self.max_mode_x):
                            coefx = np.sqrt(np.sqrt(2 / pi) / (2 ** nx * math.factorial(nx)))
                            coefy = np.sqrt(np.sqrt(2 / pi) / (2 ** ny * math.factorial(ny)))
                            self.coef = index_update(
                                self.coef, idx,
                                coefx * coefy
                            )

                            idx += 1

                    self.pump_basis_arr, self.pump_basis_str = HermiteBank(lam, self.n, waist[0] * 1e-5,
                                                                            max_mode1, max_mode2, self.X, self.Y, z)

    def create_profile(self, pump_coeffs, w0=None):
        if w0 is not None:
            E_temp = 0.
            idx    = 0
            if self.type == 'LG':
                for p in range(self.max_mode_p):
                    for l in range(-self.max_mode_l, self.max_mode_l + 1):
                        if self.learn_pump_waists:
                            if self.learn_pump_coeffs:
                                E_temp += pump_coeffs[idx] * Laguerre_gauss(self.lam, self.n, w0[idx] * 1e-5,
                                                                            l, p, self.z, self.X, self.Y, self.coef[idx])
                            else:
                                E_temp += self.pump_coeffs[idx] * Laguerre_gauss(self.lam, self.n, w0[idx] * 1e-5,
                                                                            l, p, self.z, self.X, self.Y,
                                                                            self.coef[idx])
                        else:
                            if self.learn_pump_coeffs:
                                E_temp += pump_coeffs[idx] * Laguerre_gauss(self.lam, self.n, self.waist[idx] * 1e-5,
                                                                            l, p, self.z, self.X, self.Y, self.coef[idx])
                            else:
                                E_temp += self.pump_coeffs[idx] * Laguerre_gauss(self.lam, self.n, self.waist[idx] * 1e-5,
                                                                            l, p, self.z, self.X, self.Y,
                                                                            self.coef[idx])
                        idx += 1

            elif self.type == "HG":
                for ny in range(self.max_mode_y):
                    for nx in range(self.max_mode_x):
                        if self.learn_pump_waists:
                            if self.learn_pump_coeffs:
                                E_temp += pump_coeffs[idx] * Hermite_gauss(self.lam, self.n, w0[idx]*1e-5,
                                                                           nx, ny, self.z, self.X, self.Y, self.coef[idx])
                            else:
                                E_temp += self.pump_coeffs[idx] * Hermite_gauss(self.lam, self.n, w0[idx] * 1e-5,
                                                                           nx, ny, self.z, self.X, self.Y,
                                                                           self.coef[idx])
                        else:
                            if self.learn_pump_coeffs:
                                E_temp += pump_coeffs[idx] * Hermite_gauss(self.lam, self.n, self.waist[idx] * 1e-5,
                                                                           nx, ny, self.z, self.X, self.Y, self.coef[idx])
                            else:
                                E_temp += self.pump_coeffs[idx] * Hermite_gauss(self.lam, self.n, self.waist[idx] * 1e-5,
                                                                           nx, ny, self.z, self.X, self.Y,
                                                                           self.coef[idx])

                        idx += 1
        else:
            if self.learn_pump_coeffs:
                E_temp = make_beam_from_coeffs(self.pump_basis_arr, pump_coeffs)
            else:
                E_temp = make_beam_from_coeffs(self.pump_basis_arr, self.pump_coeffs)

        self.E = fix_power(E_temp, self.power, self.n, self.crystal_dx, self.crystal_dy)[np.newaxis, :, :]


'''
Class Field:
----------
initialize E_out and E_vac, for a given beam (class Beam) and crystal (class Crystal)
    E_out - as in the paper, the output field, initilized as 0
    E_vac - as in the paper, the vacuum field, initilized as gaussian noise
    kappa - coupling constant 
'''
class Field:
    def __init__(self, beam, crystal, vac_rnd, N):
        Nx          = len(crystal.x)
        Ny          = len(crystal.y)
        self.E_out  = np.zeros([N, Nx, Ny])
        vac         = np.sqrt(h_bar * beam.w / (2 * eps0 * beam.n ** 2 * crystal.dx * crystal.dy * crystal.MaxZ))
        self.E_vac  = vac * (vac_rnd[:, 0] + 1j * vac_rnd[:, 1]) / np.sqrt(2)
        self.kappa  = 2 * 1j * beam.w ** 2 / (beam.k * c ** 2)
        # we leave d_33 out and add it in the propagation function.
        self.k      = beam.k


#######################################################
# Functions
######################################################
'''
Periodically poled crystal slab
create the crystal slab at point z in the crystal, for poling period 2pi/delta_k
'''
def PP_crystal_slab(delta_k, z, crystal_profile, infer = 0):
    return np.sign(np.cos(np.abs(delta_k) * z))


def PP_crystal_slab_2D(delta_k, z, crystal_profile, infer=0):
    magnitude = np.abs(crystal_profile)
    phase = np.angle(crystal_profile)
    if infer:
        MaxOrderFourier = 20
        poling = 0
        magnitude = magnitude/magnitude.max()
        DutyCycle = np.arcsin(magnitude)/np.pi
        for m in range(MaxOrderFourier):
            if m == 0:
                poling = poling + 2*DutyCycle - 1
            else:
                poling = poling + (2 / (m * np.pi)) * np.sin(m * pi * DutyCycle) * 2 * np.cos(m * phase + m * np.abs(delta_k) * z)
        #print("Crystal built with {} Fourier series orders \n".format(MaxOrderFourier))
        return poling
    else:
        return (2 / np.pi) * np.exp(1j * (np.abs(delta_k) * z)) * magnitude * np.exp(1j * phase)


class Poling_profile:
    def __init__(self, r_scale, x, y, length1, length2, series_type, lam_signal, ind_ref, learn_crystal_waists=False):
        self.learn_crystal_waists = learn_crystal_waists
        self.length2              = length2
        self.ind_ref              = ind_ref
        self.lam_signal           = lam_signal
        self.series_type          = series_type

        if series_type == 'FT':  # Fourier-Taylor
            self.length1    = int((length1 - 1) / 2)
            [X,Y]           = np.meshgrid(x, y)
            if learn_crystal_waists:
                self.X    = X
                self.Y    = Y
            self.phi_angle  = np.arctan2(Y, X)
            rad             = np.sqrt(X ** 2 + Y ** 2) / (r_scale[0] * 1e-5)
            self.series     = np.array([rad**p * np.exp(-rad**2) *
                                        np.exp(-1j * l * self.phi_angle)
                                        for p in range(self.length2) for l in range(-self.length1, self.length1 + 1)])




        elif series_type == 'FB':  # Fourier-Bessel
            self.length1    = int((length1 - 1) / 2)
            [X,Y]           = np.meshgrid(x, y)
            self.phi_angle  = np.arctan2(Y, X)
            rad             = np.sqrt(X ** 2 + Y ** 2) / (r_scale[0] * 1e-5)

            self.series     = np.array([sp.jv(0, sp.jn_zeros(0, p + 1)[-1] * rad) *
                                        np.exp(-1j * l * self.phi_angle)
                                        for p in range(self.length2) for l in range(-self.length1, self.length1 + 1)])

        elif series_type == 'LG':  # Laguerre-Gauss

            self.length1 = int((length1 - 1) / 2)
            [X, Y]       = np.meshgrid(x, y)

            if learn_crystal_waists:
                self.X    = X
                self.Y    = Y
                self.coef = np.zeros(len(r_scale), dtype=np.float32)
                idx = 0
                for p in range(self.length2):
                    for l in range(-self.length1, self.length1 + 1):
                        self.coef = index_update(
                            self.coef, idx,
                            np.sqrt(2 * math.factorial(p) / (np.pi * math.factorial(p + np.abs(l))))
                        )

                        idx += 1

            self.series  = np.array([Laguerre_gauss(lam_signal, ind_ref, r_scale[0] * 1e-5, l, p, 0, X, Y)
                                    for p in range(self.length2) for l in range(-self.length1, self.length1 + 1)])

        elif series_type == 'HG':  # Hermite-Gauss
            self.length1 = length1
            [X,Y]       = np.meshgrid(x, y)
            if learn_crystal_waists:
                self.X    = X
                self.Y    = Y
                self.coef = np.zeros(len(r_scale), dtype=np.float32)
                idx = 0
                for m in range(self.length2):
                    for n in range(self.length1):
                        self.coef = index_update(
                            self.coef, idx,
                            np.sqrt(np.sqrt(2 / pi) / (2 ** m * math.factorial(m))) *
                            np.sqrt(np.sqrt(2 / pi) / (2 ** n * math.factorial(n)))
                        )

                        idx += 1


            self.series = np.array([np.sqrt(np.sqrt(2 / pi) / (2 ** m * math.factorial(m))) *
                                    np.sqrt(np.sqrt(2 / pi) / (2 ** n * math.factorial(n))) *
                                    np.exp(-(X ** 2 + Y ** 2) / r_scale ** 2) * HermiteP(m, np.sqrt(2) * Y / r_scale) *
                                    HermiteP(n, np.sqrt(2) * X / r_scale)
                                    for m in range(self.length2) for n in range(self.length1)])



    def create_profile(self, poling_parameters, r_scale):
        if self.learn_crystal_waists:

            if self.series_type == 'FT':  # Fourier-Taylor
                crystal_profile = 0.
                idx = 0
                for p in range(self.length2):
                    for l in range(-self.length1, self.length1 + 1):
                        rad = np.sqrt(self.X ** 2 + self.Y ** 2) / (r_scale[idx] * 1e-5)
                        crystal_profile += rad ** p * np.exp(-rad ** 2) * np.exp(-1j * l * self.phi_angle)
                        idx += 1

                self.crystal_profile = crystal_profile


            elif self.series_type == 'FB':  # Fourier-Bessel
                crystal_profile = 0.
                idx = 0
                for p in range(self.length2):
                    for l in range(-self.length1, self.length1 + 1):
                        rad = np.sqrt(self.X ** 2 + self.Y ** 2) / (r_scale[idx] * 1e-5)
                        crystal_profile += sp.jv(0, sp.jn_zeros(0, p + 1)[-1] * rad) * \
                                           np.exp(-1j * l * self.phi_angle)
                        idx += 1

                self.crystal_profile = crystal_profile


            elif self.series_type == 'LG':  # Laguerre-Gauss
                idx = 0
                crystal_profile = 0.
                for p in range(self.length2):
                    for l in range(-self.length1, self.length1 + 1):
                        crystal_profile += poling_parameters[idx] * \
                                           Laguerre_gauss(self.lam_signal, self.ind_ref,
                                                          r_scale[idx] * 1e-5, l, p, 0, self.X, self.Y, self.coef[idx])
                        idx += 1

                self.crystal_profile = crystal_profile


            elif self.series_type == 'HG':  # Hermite-Gauss
                idx = 0
                crystal_profile = 0.
                for m in range(self.length2):
                    for n in range(self.length1):
                        crystal_profile += self.coef[idx] * \
                                           np.exp(-(self.X ** 2 + self.Y ** 2) / (r_scale[idx] * 1e-5) ** 2) * \
                                           HermiteP(m, np.sqrt(2) * self.Y / (r_scale[idx] * 1e-5)) * \
                                           HermiteP(n, np.sqrt(2) * self.X / (r_scale[idx] * 1e-5))

                        idx += 1

                self.crystal_profile = crystal_profile

        else:
            self.crystal_profile = (poling_parameters[:, None, None] * self.series).sum(0)


'''
Refractive index for MgCLN, based on Gayer et al, APB 2008
lambda is in microns, T in celsius
'''
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


def n_KTP_Kato(lam, T, ax):
    dT = (T - 20)

    if ax == "z":
        n_no_T_dep = np.sqrt(4.59423 + 0.06206 / (lam ** 2 - 0.04763) + 110.80672 / (lam ** 2 - 86.12171))
        dn         = (0.9221 / lam ** 3 - 2.9220 / lam ** 2 + 3.6677 / lam - 0.1897) * 1e-5 * dT
    if ax == "y":
        n_no_T_dep = np.sqrt(3.45018 + 0.04341 / (lam ** 2 - 0.04597) + 16.98825 / (lam ** 2 - 39.43799))
        dn         = (0.1997 / lam ** 3 - 0.4063 / lam ** 2 + 0.5154 / lam + 0.5425) * 1e-5 * dT
    n           = n_no_T_dep + dn
    return n


'''
Crystal propagation
propagate through crystal using split step Fourier for 4 fields: e_out and E_vac, signal and idler.
'''
def crystal_prop(Pump, Siganl_field, Idler_field, crystal, Poling_crystal_profile=None, infer = 0):
    # propagate
    x  = crystal.x
    y  = crystal.y
    dz = crystal.dz

    for z in crystal.z:
        # pump beam:
        E_pump = propagate(Pump.E, x, y, Pump.k, z) * np.exp(-1j * Pump.k * z)

        # crystal slab:
        PP = crystal.slab(crystal.poling_period, z, Poling_crystal_profile, infer)

        # coupled wave equations - split step
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
        Siganl_field.E_out = propagate(Siganl_field.E_out, x, y, Siganl_field.k, dz) * np.exp(-1j * Siganl_field.k * dz)
        Siganl_field.E_vac = propagate(Siganl_field.E_vac, x, y, Siganl_field.k, dz) * np.exp(-1j * Siganl_field.k * dz)
        Idler_field.E_out  = propagate(Idler_field.E_out, x, y, Idler_field.k, dz) * np.exp(-1j * Idler_field.k * dz)
        Idler_field.E_vac  = propagate(Idler_field.E_vac, x, y, Idler_field.k, dz) * np.exp(-1j * Idler_field.k * dz)
    return


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
# @jit
def propagate(A, x, y, k, dz):
    dx      = np.abs(x[1] - x[0])
    dy      = np.abs(y[1] - y[0])

    # define the fourier vectors
    X, Y    = np.meshgrid(x, y, indexing='ij')
    KX      = 2 * np.pi * (X / dx) / (np.size(X, 1) * dx)
    KY      = 2 * np.pi * (Y / dy) / (np.size(Y, 1) * dy)

    # The Free space transfer function of propagation, using the Fresnel approximation
    # (from "Engineering optics with matlab"/ing-ChungPoon&TaegeunKim):
    H_w     = np.exp(-1j * dz * (np.square(KX) + np.square(KY)) / (2 * k))
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
Generalized Laguerre polynomial of rank p,l L_p^|l|(x)
'''
def LaguerreP(p,l,x):
    if p==0:
        return 1
    elif p==1:
        return 1+np.abs(l)-x
    else:
        return ((2*p-1+np.abs(l)-x)*LaguerreP(p-1,l,x)-(p-1+np.abs(l))*LaguerreP(p-2,l,x))/p

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
#def Hermite_gause2Dxy(lam, ind_ref, W0x, W0y, n, m, z, x, y):
#    W0 = np.sqrt(W0x ** 2)
#    k = 2 * np.pi * ind_ref/ lam
#    z0 = np.pi * W0 ** 2 * ind_ref/ lam  # Rayleigh range
#    Wx = W0x * np.sqrt(1 + (z / z0) ** 2)  # w(z), the variation of the spot size
#    invR = z / ((z ** 2) + (z0 ** 2))  # radius of curvature
#    qx = 1 / (invR - 1j * (lam / ind_ref) / (np.pi * (Wx ** 2)))  # the complex beam parameter
#    q0 = 1j * z0
#    coefx = (2 / np.pi) ** 0.25 * np.sqrt(1 / (2 ** n * math.factorial(n) * W0x)) * np.sqrt(q0 / qx) * (
#                q0 * np.conj(qx) / (np.conj(q0) * qx)) ** (n / 2)
#    Unx = coefx * HermiteP(n, np.sqrt(2) * x / Wx) * np.exp(-1j * k * ((x ** 2) * (1 / (2 * qx))))

#    W0 = np.sqrt(W0y ** 2)
#    z0 = np.pi * W0 ** 2 * ind_ref/ lam;  # Rayleigh range
#    Wy = W0y * np.sqrt(1 + (z / z0) ** 2);  # w(z), the variation of the spot size
#    invR = z / ((z ** 2) + (z0 ** 2));  # radius of curvature
#    qy = 1 / (invR - 1j * (lam / ind_ref) / (np.pi * (Wy ** 2)));  # the complex beam parameter
#    q0 = 1j * z0;
#    coefy = (2 / np.pi) ** 0.25 * np.sqrt(1 / (2 ** m * math.factorial(m) * W0y)) * np.sqrt(q0 / qy) * (
#                q0 * np.conj(qy) / (np.conj(q0) * qy)) ** (m / 2);
#    Uny = coefy * HermiteP(m, np.sqrt(2) * y / Wy) * np.exp(-1j * k * ((y ** 2) * (1 / (2 * qy))))

    # plt.imshow(np.abs(np.dot(Uny.reshape(len(Unx),1),Unx.reshape(1,len(Unx)))))

#    return Unx, Uny

def Hermite_gauss(lam, ind_ref, W0, nx, ny, z, X, Y, coef=None):
    k = 2 * np.pi * ind_ref / lam
    z0 = np.pi * W0 ** 2 * ind_ref / lam  # Rayleigh range
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
        np.exp(- 1j * (k * (X**2 + Y**2) / 2) * invR) * \
        np.exp(1j * gouy)

    # plt.imshow(np.abs(np.dot(Uny.reshape(len(Unx),1),Unx.reshape(1,len(Unx)))))

    return U


'''
Laguerre Gauss in 2-D
return the Laguerre-Gaussian beam of order l,p in 2 d
receives:
      - lambda =  the wavelength
      - W0 = beam waist 
      - l , p = order of the L-G beam
      - z = the place in z to calculate for
      - x,y  = matrices of x and y 
      - P  = the total power of the beam. If this is not given then it is
       set to 1W (all in Air).
'''
def Laguerre_gauss(lam, ind_ref, W0, l, p, z, x, y, coef=None):
    k = 2 * np.pi * ind_ref / lam
    z0 = np.pi * W0 ** 2 * ind_ref/ lam  # Rayleigh range
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
        np.exp(- 1j * (k * r **2 / 2) * invR ) * \
        np.exp(- 1j* l * phi) * \
        np.exp(1j * gouy)

    # plt.imshow(np.abs(np.dot(Uny.reshape(len(Unx),1),Unx.reshape(1,len(Unx)))))

    return U



'''
HemiteBank returns a dictionary of Hermite gausee
'''
def HermiteBank(lam, ind_ref, W0, max_mode1, max_mode2, x, y, z=0):
    Hermite_dict = {}
    max_mode_x = max_mode1
    max_mode_y = max_mode2
    for ny in range(max_mode_y):
        for nx in range(max_mode_x):
            Hermite_dict[str(nx) + str(ny)] = Hermite_gauss(lam, ind_ref, W0, nx, ny, z, x, y)

    return np.array(list(Hermite_dict.values())), [*Hermite_dict]


'''
LaguerreBank returns a dictionary of Laguerre gauss
'''
def LaguerreBank(lam, ind_ref, W0, max_mode1, max_mode2, x, y, z=0):
    Laguerre_dict = {}
    max_mode_l = int((max_mode1 - 1)/2)
    max_mode_p = max_mode2
    for p in range(max_mode_p):
        for l in range(-max_mode_l, max_mode_l+1):
            Laguerre_dict[str(p) + str(l)] = Laguerre_gauss(lam, ind_ref, W0, l, p, z, x, y)

    return np.array(list(Laguerre_dict.values())), [*Laguerre_dict]

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


'''
project: projects some state A to projected_state
Both are matrices of the same size
'''
#@jit
def project(projected_state, A, minval=0):
    Nxx2           = A.shape[1] ** 2
    N              = A.shape[0]
    Nh             = projected_state.shape[0]
    projection     = (np.conj(projected_state) * A).reshape(Nh, N, Nxx2).sum(2)
    normalization1 = np.abs(A ** 2).reshape(N, Nxx2).sum(1)
    normalization2 = np.abs(projected_state**2).reshape(Nh, Nxx2).sum(1)
    projection     = projection / np.sqrt(normalization1[None, :] * normalization2[:, None])
    #cond1          = np.abs(np.real(projection)) <= np.abs(minval)
    #cond2          = np.abs(np.imag(projection)) <= np.abs(minval)
    #projection     = cond1*(cond2*0+(1-cond2)*1j*np.imag(projection)) + (1-cond1)*(np.real(projection))
    return projection
'''
Decompose a state A into modes defined in the dictionary
'''
#@jit
def decompose(A, pump_basis_arr):
        # minval1 = np.abs(project(pump_basis_arr[0][None, :], pump_basis_arr[2][None, :]))
        # minval = np.abs(project(pump_basis_arr[0][None, :], pump_basis_arr[1][None, :]))
        # minval = min(minval1, minval2) * 1.1
        basis         = pump_basis_arr[:, None]
        projection = project(basis, A)
        return np.transpose(projection)

''' 
This function takes a field A and normalizes in to have the power indicated 
'''
#@jit
def fix_power(A, power, n, dx, dy):
    output = A * np.sqrt(power) / np.sqrt(Power2D(A, n, dx, dy))
    return output

'''
'''
def fix_power1(E_fix, E_original, beam, crystal):
   # n  = beam.n
   # dx = crystal.dx
   # dy = crystal.dy
   # power = Power2D(E_original, n, dx, dy)
   scale = np.sqrt(np.sum(E_original * np.conj(E_original), (1, 2))) / np.sqrt(np.sum(E_fix * np.conj(E_fix), (1, 2)))
   return E_fix * scale[:, None, None]
   # return fix_power(E_fix, power, n, dx, dy)


def kron1(a, b):
   return (a[:, :, None, :, None] * b[:, None, :, None, :]).sum(0)

def kron(a, b):
    # return (a[:, :, None, :, None] * b[:, None, :, None, :]).sum(0)
    return lax.psum((a[:, :, None, :, None] * b[:, None, :, None, :]).sum(0), 'device')

# @jit
def Fourier(A):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(A)))