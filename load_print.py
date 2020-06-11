
import numpy as np
from spdc_helper import Beam, Crystal, nz_MgCLN_Gayer, PP_crystal_slab, SFG_idler_wavelength
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '6'

###########################################
# Structure arrays
###########################################
# initialize crystal and structure arrays
d33         = 23.4e-12  # in meter/Volt.[LiNbO3]
PP_SLT      = Crystal(10e-6, 10e-6, 1e-6, 200e-6, 200e-6, 5e-3, nz_MgCLN_Gayer, PP_crystal_slab, d33)
R           = 0.1  # distance to far-field screenin meters
Temperature = 50

###########################################
# Interacting wavelengths
##########################################
# Initiialize the interacting beams

# * define two pump's function (for now n_coeff must be 2) to define the pump *
# * this should be later changed to the definition given by Sivan *
Pump    = Beam(532e-9, PP_SLT, Temperature, 100e-6, 0.03)  # wavelength, crystal, tmperature,waist,power
Signal  = Beam(1064e-9, PP_SLT, Temperature)
Idler   = Beam(SFG_idler_wavelength(Pump.lam, Signal.lam), PP_SLT, Temperature)

res_path    = 'results/'        # should be given as a user-parameter
Pss_path    = res_path+'P_ss.npy' # should be given as a user-parameter
P_ss        = np.load(Pss_path)  # result of signal probability-density
M = 4

FF_position_axis = lambda dx, MaxX, k, R: np.arange(-1 / dx, 1 / dx, 1 / MaxX) * (np.pi * R / k)
FFcoordinate_axis_Idler = 1e3 * FF_position_axis(PP_SLT.dx, PP_SLT.MaxX, Idler.k / Idler.n, R)
FFcoordinate_axis_Signal = 1e3 * FF_position_axis(PP_SLT.dx, PP_SLT.MaxX, Signal.k / Signal.n, R)

# AK, NOV24: I added a far-field position axis extents, in mm.
extents_FFcoordinates_signal = [min(FFcoordinate_axis_Signal), max(FFcoordinate_axis_Signal),
                                min(FFcoordinate_axis_Signal), max(FFcoordinate_axis_Signal)]
# extents_FFcoordinates_idler = [min(FFcoordinate_axis_Idler), max(FFcoordinate_axis_Idler), min(FFcoordinate_axis_Idler),
#                                max(FFcoordinate_axis_Idler)]

# calculate theoretical angle for signal
# theoretical_angle = np.arccos((Pump.k - PP_SLT.poling_period) / 2 / Signal.k)
# theoretical_angle = np.arcsin(Signal.n * np.sin(theoretical_angle) / 1)  # Snell's law
# theoretical_angle = np.arcsin(Signal.n * np.sin(theoretical_angle) / 1)  # Snell's law

plt.figure()
plt.imshow(np.real(P_ss * 1e-6), extent=extents_FFcoordinates_signal)  # AK, Dec08: Units of counts/mm^2*sec
# plt.plot(1e3 * R * np.tan(theoretical_angle), 0, 'xw')
plt.xlabel(' x [mm]')
plt.ylabel(' y [mm]')
plt.title('Single photo-detection probability, Far field')
plt.colorbar()
plt.show()
exit()