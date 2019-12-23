# -*- coding: utf-8 -*-
"""
ALL FUNCTIONS FOR SPDC SIMULATIONS
---------------------------------
All units in MKS
Requirements: python modules: numpy, matplotlib, scipy, polarTransform
According to Simulating correlations of structured spontaneously down-converted photon pairs / LPR, 2020


constants:
    - pi 
    - c - the speed ofl light
    - eps0 - vacuum permetitivity
    - h_bar - Plancks constant

lambda functions:
    - I - calculate the inteneisty of a field A with refractive index n
    - E0 - calcualte the constant amplitude size for power P, refractive index n and beam waist W0
    - Fourier - perform fourier transform using FFT
    - Power2D - calcualte the power of a 2D field amplitude with refrative index n and dx,dy resolution
    - SFG_idler_wavelength - calcualte the idler wavelength given signal and idler wavelength, so that energy is conserved
    - G1_Normalization - normalization factor for G1 with frequnecy w
    - Fourier_axis - calcualte the Fourier axis for dx,MaxX resolution and maximal length, and wave-vector k
    - FF_position_axis - far field position axis - calcualte the axis for far field where dx,MaxX are the near field resolution and maximal length, k the wave-vector and R the distance to the FF screan
    
Classes:
    - Beam: saves all conatants of a beam
    - Field: initializes E_out, E_vac and kappa (coupling) for a beam
    - Crystal: creates all strcuturs arrays for a crystal
    - G1_mat: stores all first order correlation G1's: ss = signal-signal, ii=idler-idler, is = idler-signal, updates all values given the propagated fields
    - Q_mat: stores all Q matrixes (according to the paper): ss = signal-signal, ii=idler-idler, is = idler-signal, updates all values given the propagated fields
    
Functions:
    - PP_crystal_slab - returns the crystal slab for a periodically poled crystal at z
    - Gaussian_beam_calc - calculates the 2D distribution of a Gaussian beam at location z_point 
    - propagate - free space propagation
    - crystal_prop - propagation through non linear crystal using splot step fourier
    - kron - calcualte the kroneker product of two M X N matrices
    - trace_it - traces over dimensions dim1 and dim2 of a (0<dim<3) for the product of two matrices
    - car2pol - converts a matrix from cartesian to polar coordinates
    - nz_MgCLN_Gayer - temperature and wavelength dependant refractive index of SLT 
    
Please acknowledge this work if it is used for academic research   
@author: Sivan Trajtenberg-Mills*, Aviv Karnieli, Noa Voloch-Bloch, Eli Megidish, Hagai S. Eisenberg and Ady Arie
"""
import numpy as np
import scipy.interpolate as interp
import polarTransform

###########################################
# Constants
###########################################
pi            = np.pi
c             = 2.99792458e8;#the speed of light in meter/sec 
eps0          = 8.854187817e-12; # the vacuum permittivity, in Farad/meter.
h_bar         = 1.054571800e-34; # Units are m^2 kg / s, taken from http://physics.nist.gov/cgi-bin/cuu/Value?hbar|search_for=planck

###########################################
# lambda funtions:
###########################################
I                    = lambda A,n: 2*n*eps0*c*np.abs(A)**2                            # Intensity
E0                   = lambda P,n,W0: np.sqrt(P/(n*c*eps0*np.pi*W0**2))               #Calc amplitude
Fourier              = lambda A: (np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(A))))  #Fourier
Power2D              = lambda A,n,dx,dy: np.sum(I(A,n))*dx*dy;
SFG_idler_wavelength = lambda lambda_p,lambda_s: lambda_p * lambda_s / ( lambda_s - lambda_p ) #Compute the idler wavelength given pump and signal
G1_Normalization     = lambda w:  h_bar * w / ( 2 * eps0 * c )
Fourier_axis         = lambda dx,MaxX,k: np.arctan(np.pi*np.arange(-1/dx,1/dx,1/MaxX)/k) * 180/np.pi
FF_position_axis     = lambda dx,MaxX,k,R: np.arange(-1/dx,1/dx,1/MaxX)*(np.pi*R/k)

#################################################################
# Create classes: 
#  - beam: all parameters of a laser beam
#  - field: initializes E_out and E_vac
#  - crystal: all structure arrays
################################################################

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
  def __init__(self, lam,crystal,T,waist=0,power=0):
    self.lam    = lam                                     #wavelength
    self.waist  = waist                                   #waist
    self.n      = crystal.ctype(lam*1e6,T)                #refractive index
    self.w      = 2*np.pi*c/lam                           #frequency
    self.k      = 2*np.pi*crystal.ctype(lam*1e6,T)/lam    #wave vector
    self.b      = waist**2*self.k                         #
    self.power  = power                                   #beam power

'''
Class Field:
----------
initialize E_out and E_vac, for a given beam (class Beam) and crystal (class Crystal)
    E_out - as in the paper, the output field, initilized as 0
    E_vac - as in the paper, the vacuum field, initilized as gaussian noise
    kappa - coupling constant 
'''
class Field:
  def __init__(self, beam, crystal):
    Nx          = len(crystal.x)
    Ny          = len(crystal.y)
    self.E_out  = np.zeros([Nx,Ny],dtype=complex)
    vac         = np.sqrt(h_bar*beam.w/(2*eps0 * beam.n**2 * crystal.dx * crystal.dy * crystal.MaxZ))
    self.E_vac  = vac*(np.random.normal(loc = 0, scale = 1, size = (Nx,Ny))+1j*np.random.normal(loc = 0, scale = 1, size = (Nx,Ny)))/np.sqrt(2)
    self.kappa  = 2*1j*beam.w**2/(beam.k*c**2) #we leave d_33 out and add it in the propagation function.
    self.k      = beam.k

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
  def __init__(self, dx,dy,dz,MaxX,MaxY,MaxZ,Ref_ind,slab_function,d,period=0):     
    self.dz             = dz                              #resolution of z axis
    self.dx             = dx                              #resolution of x axis
    self.dy             = dy                              #resolution of y axis
    self.MaxX           = MaxX
    self.MaxY           = MaxY
    self.MaxZ           = MaxZ
    self.x              = np.arange(-MaxX,MaxX,dx)        # x axis, length 2*MaxX (transverse)
    self.y              = np.arange(-MaxY,MaxY,dy)        # y axis, length 2*MaxY  (transverse)
    self.z              = np.arange(-MaxZ/2,MaxZ/2,dz)    # z axis, length MaxZ (propagation)
    self.ctype          = Ref_ind                         #refractive index function
    self.slab           = slab_function
    self.d              = d
    self.poling_period  = period


class G1_mat:
  def __init__(self):     
    self.ii             = 0
    self.ss             = 0
    self.si             = 0
    self.si_dagger      = 0
    
  def update(self,E_s_out_k,E_s_vac_k,E_i_out_k,E_i_vac_k,N):
    self.ii             = self.ii        + kron(np.conj(E_i_out_k), E_i_out_k) / N
    self.ss             = self.ss        + kron(np.conj(E_s_out_k), E_s_out_k) / N
    self.si             = self.si        + kron(np.conj(E_i_out_k), E_s_out_k) / N
    self.si_dagger      = self.si_dagger + kron(np.conj(E_s_out_k), E_i_out_k) / N
    
class Q_mat:
  def __init__(self):     
    self.ii             = 0
    self.ii_dagger      = 0
    self.ss             = 0
    self.ss_dagger      = 0
    self.si             = 0
    self.si_dagger      = 0

  def update(self,E_s_out_k,E_s_vac_k,E_i_out_k,E_i_vac_k,N):
    self.ii             = self.ii        + kron(E_i_vac_k, E_i_out_k) / N
    self.ii_dagger      = self.ii_dagger + kron(np.conj(E_i_out_k), np.conj(E_i_vac_k)) / N

    self.ss             = self.ss        + kron(E_s_vac_k, E_s_out_k) / N
    self.ss_dagger      = self.ss_dagger + kron(np.conj(E_s_out_k), np.conj(E_s_vac_k)) / N

    self.si             = self.si        + kron(E_i_vac_k, E_s_out_k) / N
    self.si_dagger      = self.si_dagger + kron(np.conj(E_s_out_k), np.conj(E_i_vac_k)) / N
    
#######################################################
# Functions
######################################################
'''
Periodically poled crystal slab
create the crystal slab at point z in the crystal, for poling period 2pi/delta_k
'''
def PP_crystal_slab(delta_k,z):
    return np.sign(np.cos(np.abs(delta_k)*z))

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
def Gaussian_beam_calc(beam,crystal,z_point): 
    X, Y = np.meshgrid(crystal.x,crystal.y, indexing='ij')
    xi   = 2*(z_point)/beam.b
    tau  = 1/(1+1j*xi)    
    E    = E0(beam.power, beam.n, beam.waist)*tau*np.exp(-((X)**2+(Y)**2)/(beam.waist**2)*tau)*np.exp(1j*beam.k*(z_point))
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
def propagate (A, x, y, k, dz):
    dx = np.abs(x[1] - x[0])
    dy = np.abs(y[1] - y[0])   
    # define the fourier vectors
    X, Y = np.meshgrid(x,y, indexing='ij')
    KX = 2*np.pi*(X/dx)/(np.size(X,1)*dx)
    KY = 2*np.pi*(Y/dy)/(np.size(Y,1)*dy)
    #The Free space transfer function of propagation, using the Fresnel approximation
    # (from "Engineering optics with matlab"/ing-ChungPoon&TaegeunKim):
    H_w = np.exp(-1j*dz*(np.square(KX)+np.square(KY))/(2*k))
    #(inverse fast Fourier transform shift). For matrices, ifftshift(X) swaps the 
    #first quadrant with the third and the second quadrant with the fourth.
    H_w = np.fft.ifftshift(H_w)
    #Fourier Transform: move to k-space 
    G = np.fft.fft2(A) #The two-dimensional discrete Fourier transform (DFT) of A.
    #propoagte in the fourier space    
    F = np.multiply(G, H_w)
    #inverse Fourier Transform: go back to real space
    Eout = np.fft.ifft2(F); #[in real space]. E1 is the two-dimensional INVERSE discrete Fourier transform (DFT) of F1
    return Eout    

'''
Crystal propagation
propagate through crystal using split step Fourier for 4 fields: e_out and E_vac, signal and idler.

'''
def crystal_prop(Pump,Siganl_field,Idler_field,crystal):
    
    for z in crystal.z: 
        #pump beam:
        E_pump   = Gaussian_beam_calc(Pump,crystal,z) 
        #crystal slab:
        PP           = crystal.slab(crystal.poling_period,z)
        
        #cooupled wave equations - split step
        #signal:
        dEs_out_dz   = Siganl_field.kappa*crystal.d * PP * E_pump*np.conj(Idler_field.E_vac)
        dEs_vac_dz   = Siganl_field.kappa*crystal.d * PP * E_pump*np.conj(Idler_field.E_out)
        
        Siganl_field.E_out = Siganl_field.E_out + dEs_out_dz*crystal.dz
        Siganl_field.E_vac = Siganl_field.E_vac + dEs_vac_dz*crystal.dz
        
        #idler:
        dEi_out_dz   = Idler_field.kappa*crystal.d * PP * E_pump*np.conj(Siganl_field.E_vac)
        dEi_vac_dz   = Idler_field.kappa*crystal.d * PP * E_pump*np.conj(Siganl_field.E_out)
        
        Idler_field.E_out  = Idler_field.E_out + dEi_out_dz*crystal.dz
        Idler_field.E_vac  = Idler_field.E_vac + dEi_vac_dz*crystal.dz       

        #propagate
        Siganl_field.E_out = propagate(Siganl_field.E_out, crystal.x, crystal.y, Siganl_field.k, crystal.dz) * np.exp(1j*Siganl_field.k*crystal.dz)
        Siganl_field.E_vac = propagate(Siganl_field.E_vac, crystal.x, crystal.y, Siganl_field.k, crystal.dz) * np.exp(1j*Siganl_field.k*crystal.dz)
        Idler_field.E_out  = propagate(Idler_field.E_out,  crystal.x, crystal.y, Idler_field.k,  crystal.dz) * np.exp(1j*Idler_field.k*crystal.dz)
        Idler_field.E_vac  = propagate(Idler_field.E_vac,  crystal.x, crystal.y, Idler_field.k,  crystal.dz) * np.exp(1j*Idler_field.k*crystal.dz)        
        
        
    return 

'''
Crystal propagation for nondistiguishable case
propagate through crystal using split step Fourier for 4 fields: e_out and E_vac, signal and idler.

'''
def crystal_prop_indistuigishable(Pump,Siganl_field,crystal):
    
    for z in crystal.z: 
        #pump beam:
        E_pump   = Gaussian_beam_calc(Pump,crystal,z) 
        #crystal slab:
        PP           = crystal.slab(crystal.poling_period,z)
        
        #cooupled wave equations - split step
        #signal:
        dEs_out_dz   = Siganl_field.kappa*crystal.d * PP * E_pump*np.conj(Siganl_field.E_vac)
        dEs_vac_dz   = Siganl_field.kappa*crystal.d * PP * E_pump*np.conj(Siganl_field.E_out)
        
        Siganl_field.E_out = Siganl_field.E_out + dEs_out_dz*crystal.dz
        Siganl_field.E_vac = Siganl_field.E_vac + dEs_vac_dz*crystal.dz
        
        #propagate
        Siganl_field.E_out = propagate(Siganl_field.E_out, crystal.x, crystal.y, Siganl_field.k, crystal.dz) * np.exp(1j*Siganl_field.k*crystal.dz)
        Siganl_field.E_vac = propagate(Siganl_field.E_vac, crystal.x, crystal.y, Siganl_field.k, crystal.dz) * np.exp(1j*Siganl_field.k*crystal.dz)
        
    return 


'''
Calculate the Kroneker outer product of A (X) B
'''
def kron(A,B):
    s1   = np.shape(A)
    s2   = np.shape(B)
    C    = np.zeros([s1[0], s1[1], s2[0], s2[1]],dtype=complex)
    for i in range(s1[0]):
        for j in range(s1[1]):
            C[i,j,:,:] = A[i,j] * B

    return C

'''
TRACE_IT takes a matrix (that has more than 2 dimesnions), 
squares it (A)*B and traces over 2 dimensions
'''
def trace_it(A, B, dim1, dim2):
   C    = np.sum(((A)*B),axis = dim1)    #square and trace over dimesnion dim1
   C    = np.sum(C,axis  = dim2-1)       #trace over over dimesnion dim2
   return C

'''
For a matrix A, convert it to polar coordinates 
returns:
    - output - the output matrix in polar coordinates
    - r - the radius extents
    - th - the angle extents
'''
def car2pol(A):

    polar_real, ptSettings = polarTransform.convertToPolarImage(np.real(A))
    polar_imag, ptSettings = polarTransform.convertToPolarImage(np.imag(A))
    
    output                 = polar_real + 1j*polar_imag
    r                      = [ptSettings.initialRadius , ptSettings.finalRadius]
    th                     = [ptSettings.initialAngle , ptSettings.finalAngle]

    return output, r, th           #interpolate 

     
'''
Refractive index for MgCLN, based on Gayer et al, APB 2008
lambda is in microns, T in celsius

'''
def nz_MgCLN_Gayer(lam,T):
    
    a  = np.array([5.756, 0.0983, 0.2020, 189.32, 12.52, 1.32*10**(-2)])
    b  = np.array([2.860*10**(-6), 4.700*10**(-8), 6.113*10**(-8), 1.516*10**(-4)])
    f  = (T-24.5)*(T+570.82);

    n1 = a[0] 
    n2 = b[0]*f 
    n3 = (a[1]+b[1]*f)/(lam**2-(a[2]+b[2]*f)**2);
    n4 = (a[3]+b[3]*f)/(lam**2-(a[4])**2); 
    n5 = -a[5]*lam**2;
    
    nz = np.sqrt(n1+n2+n3+n4+n5)
    return nz
