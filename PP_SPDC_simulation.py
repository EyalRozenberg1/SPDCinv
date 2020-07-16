# -*- coding: utf-8 -*-
"""
Simulation tool for simulating Spontaneaus parametric down covnersion 
----------------------------------------------------------------------
This file simulates a periodically poled LiNbO3, with a 532 nm pump and 1064.5 nm signal and idler
plots the first order correlation G1, and the second order correlation G2, for N iterations

Requirements:
    - python modules: numpy, matplotlib, scipy, polarTransform

Options:
    - IS_INDISTIGUISHABLE: for a truly degenerate process, there should be only 
    one field propagated since the two are completely indistiguishable. For this case, set flag to 1.
    This is the case shown in the paper.
    - DO_POLAR: show the correlations in polar coordinates as well as far field x-y coordinates.

Please acknowledge this work if it is used for academic research
According to Simulating correlations of structured spontaneously down-converted photon pairs / LPR, 2020
@authors: Sivan Trajtenberg-Mills*, Aviv Karnieli, Noa Voloch-Bloch, Eli Megidish, Hagai S. Eisenberg and Ady Arie

!!!!!!!!Changes made by AK, Feb23:

    1. Kronicker product now uses the numpy function.
    2. We use M^2 x M^2 matrices to describe all quantities: G1 is sparse and G2 is full.
    3. We "sample" the diagonal elements of G1 (photodetection probability) by multiplying element wise
       with an M^2 x M^2 indicator matrix (ones for relevant elements; zero otherwise) which we prepare in advance.
    4. We can learn G1 and G2 matrices directly on GPU now - compare them with the corresponding targets
       with similar matrix shape.
    5. OPTIONAL: As before we can plot both G1 and G2 - we squeeze G1 and trace over two of the dimensions in G2.
       To do this I added a Kronicker unwrapping function to bring the M^2 x M^2 matrix back to M x M x M x M. And
       changed "trace_it" a little bit.
    6. TODO: small modification so that the polar scheme also works (currently the polar transform uses nonsquare
       matrices)

"""

from All_SPDC_funcs import *
import matplotlib.pyplot as plt

plt.close('all');

###########################################
# Structure arrays
###########################################
# initialize crystal and structure arrays
d33 = 23.4e-12  # in meter/Volt.[LiNbO3]
PP_SLT = Crystal(5e-6, 5e-6, 1e-6, 300e-6, 300e-6, 20e-3, nz_MgCLN_Gayer, PP_crystal_slab_2D,
                 d33)  # dx,dy,dz,MaxX,MaxY,MaxZ,Ref_ind,slab_function
R = 0.1  # distance to far-field screenin meters
Temperature = 50
M = len(PP_SLT.x)  # simulation size
###########################################
# Interacting wavelengths
##########################################
# Initiialize the interacting beams
max_mode  = 10
Pump      = Beam(532e-9, PP_SLT, Temperature, 50e-6, 1e-3, max_mode)  # wavelength, crystal, tmperature,waist,power, maxmode
Signal    = Beam(1064e-9, PP_SLT, Temperature, np.sqrt(2)*Pump.waist, 1, max_mode)
Idler     = Beam(SFG_idler_wavelength(Pump.lam, Signal.lam), PP_SLT, Temperature, np.sqrt(2)*Pump.waist, 1, max_mode)

pump_parameters = np.zeros(max_mode**2)
pump_parameters[1] = 1
Pump.create_profile(pump_parameters, PP_SLT)

#phi_parameters = [0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0] #linear shift
#phi_parameters = [0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0] #Lens
#phi_parameters = [0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0] #cube?
phi_parameters = [12, 0, -48, 0, 16, 0, 0, 0, 0, 0, 0] #Hermite4
#phi_parameters = [-120, 0, 720, 0, -480, 0, 64, 0, 0, 0, 0] #Hermite6

phi_scale = 1
phi = Poling_profile(phi_parameters, phi_scale, PP_SLT.x, PP_SLT.MaxX)

# phase mismatch
delta_k = Pump.k - Signal.k - Idler.k
PP_SLT.poling_period = 1.004 * delta_k


#plot crystal pattern
XX, ZZ = np.meshgrid(PP_SLT.x, PP_SLT.z)
Phi = Poling_profile(phi_parameters, phi_scale, XX, PP_SLT.MaxX)
plt.figure()
plt.imshow(np.sign(np.cos(PP_SLT.poling_period*ZZ+Phi)), aspect='auto')
plt.xlabel(' x [mm]')
plt.ylabel(' z [mm]')
plt.title('Crystal')
plt.colorbar()
plt.show()

# flags
IS_INDISTIGUISHABLE = 0  # Flag for being indistiguishable or not
IS_GAUSSIAN         = 0  #Flag for being Gaussian or not. This will change the propagation 
DO_POLAR            = 0  # Flag for moving to polar coordinates
DO_HG               = 1  # Flag for projecting to HG modes
DO_FF               = 0


# Build Diagonal-Element indicator matrix for the Kronicker products
Kron_diag = np.zeros((M ** 2, M ** 2))
for i in range(M):
    for j in range(M):
        Kron_diag[i + i * M, j + j * M] = 1

if DO_HG: 
    Kron_diag_HG = np.zeros((max_mode ** 2, max_mode** 2))
    for i in range(max_mode):
        for j in range(max_mode):
            Kron_diag_HG[i + i * max_mode, j + j * max_mode] = 1       

##########################################
# Run N simulations through crystal
##########################################
N = 5 # number of iterations
# seed vacuum samples
np.random.seed(seed=1)

# initialize
G1 = G1_mat()
Q = Q_mat()

if DO_POLAR:
    G1_pol = G1_mat()
    Q_pol = Q_mat()
    
if DO_HG:
    G1_HG = G1_mat()
    Q_HG = Q_mat()

# ----------------------------------------
# run the degenerate (idistuigshable) case
# ----------------------------------------
if IS_INDISTIGUISHABLE:
    for n in range(N):

        print('running number ', n + 1)

        # initialize the vacuum and output fields:
        Siganl_field = Field(Signal, PP_SLT)

        # Propagate through the crystal:
        crystal_prop_indistuigishable(Pump, Siganl_field, PP_SLT,IS_GAUSSIAN)

        # Coumpute k-space far field using FFT:
        # normalization factors
        FarFieldNorm_signal = (2 * PP_SLT.MaxX) ** 2 / (np.size(Siganl_field.E_out) * Signal.lam * R)

        # FFT:
        E_s_out_k = FarFieldNorm_signal * Fourier(Siganl_field.E_out)
        E_s_vac_k = FarFieldNorm_signal * Fourier(Siganl_field.E_vac)

        if DO_FF:
            # Compute G1 for idler-idler, signal-signal and idler-signal correlations.
            # this results in M X M X M X M matrixes
            # compute got far field (g1)
            G1.update(E_s_out_k, E_s_vac_k, E_s_out_k, E_s_vac_k, N)

            # Compute Q for idler-idler, signal-signal and idler-signal correlations.
            # this results in M X M X M X M matrixes
            # compute got far field (g1) 
            Q.update(E_s_out_k, E_s_vac_k, E_s_out_k, E_s_vac_k, N)

        if DO_POLAR:
            # move to polar coordinates
            E_s_out_k_pol, r, th = car2pol(E_s_out_k)
            E_s_vac_k_pol, r, th = car2pol(E_s_vac_k)

            G1_pol.update(E_s_out_k_pol, E_s_vac_k_pol, E_s_out_k_pol, E_s_vac_k_pol, N)
            Q_pol.update(E_s_out_k_pol, E_s_vac_k_pol, E_s_out_k_pol, E_s_vac_k_pol, N)

    Idler = Signal

# ----------------------------------------
# run the distuigshable case, with all 4 fields
# ----------------------------------------
else:
    for n in range(N):

        print('running number ', n + 1)

        # initialize the vacuum and output fields:
        Siganl_field = Field(Signal, PP_SLT)
        Idler_field = Field(Idler, PP_SLT)

        # Propagate through the crystal:
        crystal_prop(Pump, Siganl_field, Idler_field, PP_SLT, phi)

        # Coumpute k-space far field using FFT:
        # normalization factors
        FarFieldNorm_signal = (2 * PP_SLT.MaxX) ** 2 / (np.size(Siganl_field.E_out) * Signal.lam * R)
        FarFieldNorm_idler = (2 * PP_SLT.MaxX) ** 2 / (np.size(Idler_field.E_out) * Idler.lam * R)

        # FFT:
        E_s_out_k = FarFieldNorm_signal * Fourier(Siganl_field.E_out)
        E_i_out_k = FarFieldNorm_idler * Fourier(Idler_field.E_out)
        E_s_vac_k = FarFieldNorm_signal * Fourier(Siganl_field.E_vac)
        E_i_vac_k = FarFieldNorm_idler * Fourier(Idler_field.E_vac)

        if DO_FF:
    
            # Compute G1 for idler-idler, signal-signal and idler-signal correlations.
            # this results in M X M X M X M matrixes
            # compute got far field (g1) 
            G1.update(E_s_out_k, E_s_vac_k, E_i_out_k, E_i_vac_k, N)
    
            # Compute Q for idler-idler, signal-signal and idler-signal correlations.
            # this results in M X M X M X M matrixes
            # compute got far field (g1) 
            Q.update(E_s_out_k, E_s_vac_k, E_i_out_k, E_i_vac_k, N)

        if DO_POLAR:
            # move to polar coordinates
            E_s_out_k_pol, r, th = car2pol(E_s_out_k)
            E_i_out_k_pol, r, th = car2pol(E_i_out_k)
            E_s_vac_k_pol, r, th = car2pol(E_s_vac_k)
            E_i_vac_k_pol, r, th = car2pol(E_i_vac_k)

            G1_pol.update(E_s_out_k_pol, E_s_vac_k_pol, E_i_out_k_pol, E_i_vac_k_pol, N)
            Q_pol.update(E_s_out_k_pol, E_s_vac_k_pol, E_i_out_k_pol, E_i_vac_k_pol, N)
            
        if DO_HG:
            E_s_out_HG = np.reshape(decompose(Siganl_field.E_out, Signal.hemite_dict),[max_mode,max_mode])
            E_i_out_HG = np.reshape(decompose(Idler_field.E_out, Signal.hemite_dict),[max_mode,max_mode])
            E_s_vac_HG = np.reshape(decompose(Siganl_field.E_vac, Signal.hemite_dict),[max_mode,max_mode])
            E_i_vac_HG = np.reshape(decompose(Idler_field.E_vac, Signal.hemite_dict),[max_mode,max_mode])

            #say there are no higher modes by normalizing the power
            E_s_out_HG = fix_power1(E_s_out_HG,Siganl_field.E_out,Signal,PP_SLT)
            E_i_out_HG = fix_power1(E_i_out_HG,Idler_field.E_out,Signal,PP_SLT)
            E_s_vac_HG = fix_power1(E_s_vac_HG,Siganl_field.E_vac,Signal,PP_SLT)
            E_i_vac_HG = fix_power1(E_i_vac_HG,Idler_field.E_vac,Signal,PP_SLT)

            
            G1_HG.update(E_s_out_HG, E_s_vac_HG, E_i_out_HG, E_i_vac_HG, N)
            Q_HG.update(E_s_out_HG, E_s_vac_HG, E_i_out_HG, E_i_vac_HG, N)
            
            
#########################################################################
# COMPUTE THE SINGLE PHOTODETECTION PROBABILITY P1(k) = G1(k ; k)
# for finding a photon in mode k = (r, theta) or k = (kx,ky)
# This is done by taking the diagonal elements of the G1 array: i.e.
# requiring (kx,ky) = (kx',ky')
########################################################################


# These are M^2 x M^2 sparse matrices containing only the diagonal elements of G1.
# These matrices can be computed directly on the the GPU and used for the learning (comparison with target)
if DO_FF:
    G1_ii_diag = G1.ii * Kron_diag / G1_Normalization(Idler.w)
    G1_ss_diag = G1.ss * Kron_diag / G1_Normalization(Signal.w)

if DO_POLAR:
    G1_pol_ii_diag = G1_pol.ii * Kron_diag / G1_Normalization(Idler.w)
    G1_pol_ss_diag = G1_pol.ss * Kron_diag / G1_Normalization(Signal.w)

if DO_HG:
    G1_HG_ii_diag = G1_HG.ii * Kron_diag_HG / G1_Normalization(Idler.w)
    G1_HG_ss_diag = G1_HG.ss * Kron_diag_HG / G1_Normalization(Signal.w)

#########################################################################
# Plot G1(k,k) for illustration only - need to do only once, if at all.
########################################################################

if DO_FF:
    ##For illustration purpose only: reduce the sparse matrices to an M x M probability density
    P_ii = np.reshape(G1_ii_diag[G1_ii_diag != 0], (M, M))
    P_ss = np.reshape(G1_ss_diag[G1_ss_diag != 0], (M, M))
    
    # Far field coordinates for distance R, in free space propagation
    FFcoordinate_axis_Idler = 1e3 * FF_position_axis(PP_SLT.dx, PP_SLT.MaxX, Idler.k / Idler.n, R)
    FFcoordinate_axis_Signal = 1e3 * FF_position_axis(PP_SLT.dx, PP_SLT.MaxX, Signal.k / Signal.n, R)
    
    # AK, NOV24: I added a far-field position axis extents, in mm.
    extents_FFcoordinates_signal = [min(FFcoordinate_axis_Signal), max(FFcoordinate_axis_Signal),
                                    min(FFcoordinate_axis_Signal), max(FFcoordinate_axis_Signal)]
    extents_FFcoordinates_idler = [min(FFcoordinate_axis_Idler), max(FFcoordinate_axis_Idler), min(FFcoordinate_axis_Idler),
                                   max(FFcoordinate_axis_Idler)]
    
    # calculate theoretical angle for signal
    theoretical_angle = np.arccos((Pump.k - PP_SLT.poling_period) / 2 / Signal.k)
    theoretical_angle = np.arcsin(Signal.n * np.sin(theoretical_angle) / 1)  # Snell's law
    
    plt.figure()
    plt.imshow(np.real(P_ss * 1e-6), extent=extents_FFcoordinates_signal)  # AK, Dec08: Units of counts/mm^2*sec
    plt.plot(1e3 * R * np.tan(theoretical_angle), 0, 'xw')
    plt.xlabel(' x [mm]')
    plt.ylabel(' y [mm]')
    plt.title('Single photo-detection probability, Far field')
    plt.colorbar()
    plt.show()

#########################################################################
# Compute the exact G2 via
# G2 (k, k',k',k) = P1(k) * P1(k') + |G1(k,k')|^2 + |Q(k,k')|^2
# We don't trace anything. The resulting matrix is M^2 x M^2 and we can compare
# it to the target function.
#########################################################################
if DO_FF:
    G2 = np.real(G1.ii * G1.ss + Q.si_dagger * Q.si + G1.si_dagger * G1.si)

if DO_POLAR:
    G2_pol = np.real(G1_pol.ii * G1_pol.ss + Q_pol.si_dagger * Q_pol.si + G1_pol.si_dagger * G1_pol.si)

if DO_HG:
    G2_HG = np.real(G1_HG.ii * G1_HG.ss + Q_HG.si_dagger * Q_HG.si + G1_HG.si_dagger * G1_HG.si)

#########################################################################
# For illustration only:
# Compute the reduced representation of G2 via
# G2 (theta, theta') = P1(theta) * P1(theta') + |G1(theta,theta')|^2 + |Q(theta,theta')|^2
# In our case of degenerate SPDC, we trace over the radial coordinates (r , r')
# for |G1|^2 and |Q|^2, and trace over r for P1, such that the reduced P1*P1, |G1|^2 and |Q|^2 now
# depend only on (theta, theta') of the photon pair.
#########################################################################
if DO_FF:
    G2_unwrapped = unwrap_kron(G2, M)
    
    # Fourier coordiantes
    # Add far-field resolution, in meters
    dx_farfield_idler = 1e-3 * (FFcoordinate_axis_Idler[1] - FFcoordinate_axis_Idler[0])
    dx_farfield_signal = 1e-3 * (FFcoordinate_axis_Signal[1] - FFcoordinate_axis_Signal[0])
        
    # add coincidence window
    tau = 1e-9  # nanosec
    
    # Compute and plot reduced G2
    #trace over the y index
    G2_reduced = trace_it(G2_unwrapped, 1, 3) 
    G2_reduced = G2_reduced * tau / (G1_Normalization(Idler.w) * G1_Normalization(Signal.w))
    
    # plot
    plt.figure()
    plt.imshow(1e-6 * G2_reduced, extent=extents_FFcoordinates_signal) 
    plt.title(r'$G^{(2)}$ (coincidences)')
    plt.xlabel(r'$x_i$ [mm]')
    plt.ylabel(r'$x_s$ [mm]')
    plt.colorbar()
    plt.show()

if DO_HG:
    G2_HG = G2_HG/np.sum(G2_HG)
    G2_unwrapped_HG = unwrap_kron(G2_HG, max_mode)


    # add coincidence window
    tau = 1e-9  # nanosec
    
    # Compute and plot reduced G2
    G2_reduced = trace_it(G2_unwrapped_HG, 0,2)
    G2_reduced = G2_reduced * tau / (G1_Normalization(Idler.w) * G1_Normalization(Signal.w))
    
    # plot
    plt.figure()
    plt.imshow((G2_reduced))  # AK, Dec08: G2 in counts/sec/mm^2
    plt.title(r'$G^{(2)}$ (coincidences)')
    plt.xlabel(r'signal mode i')
    plt.ylabel(r'idle mode j')

    plt.colorbar()
    plt.show()


if DO_POLAR:
    # Polar coordiantes
    # Need to trace over the r coordinate
    # First find dr in the far-field, in meters
    dr = 1e-3 * (FFcoordinate_axis_Idler[1] - FFcoordinate_axis_Idler[0])
    # Calculate theoretical ring radius for approximate Jacobian rdr
    r_th = R * np.tan(theoretical_angle)

    # add coincidence window
    tau = 1e-9  # nanosec

    # Compute and plot reduced G2
    G2_pol_unwrapped = unwrap_kron(G2_pol, M)
    G2_pol_reduced = trace_it(G2_pol_unwrapped, 1, 3) * (r_th * dr) ** 2
    G2_pol_reduced = G2_pol_reduced * tau / (G1_Normalization(Idler.w) * G1_Normalization(Signal.w))

    # plot
    extents = np.array([th[0], th[1], th[0], th[1]])
    plt.figure()
    plt.imshow(G2_pol_reduced, extent=extents)
    plt.title(r'$G^{(2)}$ (coincidences)')
    plt.xlabel(r'$\theta$ [rad]')
    plt.ylabel(r'$\theta$ [rad]')
    plt.colorbar()
    plt.show()
    

