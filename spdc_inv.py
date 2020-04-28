from __future__ import print_function, division, absolute_import
from jax import grad, vmap, ops
from jax.numpy import kron, linalg as la
import numpy as onp

from spdc_helper import *
import matplotlib.pyplot as plt

import os, time
os.environ["CUDA_VISIBLE_DEVICES"] = '6'

start_time_initialization = time.time()

# are we in learning mode or infering from simulation
learn_mode  = False
# pick loss
loss_type = 'kl'  # 'l2', 'kl', 'wasserstein'

###############################
# Target function and results #
###############################
show_res      = True  # display and show results
# saving results
save_res      = False
res_path      = 'results/'          # path to folder. should be given as a user-parameter
res_name_pss      = 'P_ss_HG00_inference_N100_xy180e-6'
res_name_g2      = 'G2_HG00_inference_N100_xy180e-6'

# saving as target
save_tgt      = False
Pt_path       = 'targets/'          # path to folder. should be given as a user-parameter
Pss_t_name    = 'P_ss_HG00'

# load target P, G2
if learn_mode:
    P_ss_t     = np.load(Pt_path+Pss_t_name+'.npy')  # target signal probability-density
    G2t        = np.load(Pt_path+G2_t_name+'.npy')


n_coeff     = 16  # coefficients of beam-basis functions
param_scale = 1
step_size   = 0.01
num_epochs  = 500
batch_size  = 100  # 20, 20, 50, 100
N           = 10000  # 100, 500, 1000  number of iterations / dataset size
alpha       = 0  # weight for loss: (1-alpha)Pss + alpha G2

num_batches = N/batch_size
assert N % batch_size == 0, "num_batches should be 'signed integer'"


###########################################
# Structure arrays
###########################################
# initialize crystal and structure arrays
d33         = 23.4e-12  # in meter/Volt.[LiNbO3]
PP_SLT      = Crystal(10e-6, 10e-6, 1e-5, 200e-6, 200e-6, 5e-3, nz_MgCLN_Gayer, PP_crystal_slab, d33)
R           = 0.1  # distance to far-field screen in meters
Temperature = 50
M           = len(PP_SLT.x)  # simulation size

###########################################
# Interacting wavelengths
##########################################
# Initiialize the interacting beams

# * define two pump's function (for now n_coeff must be 2) to define the pump *
# * this should be later changed to the definition given by Sivan *
max_mode = 4
Pump    = Beam(532e-9, PP_SLT, Temperature, 100e-6, 0.03,max_mode)  # wavelength, crystal, tmperature,waist,power, maxmode
Signal  = Beam(1064e-9, PP_SLT, Temperature)
Idler   = Beam(SFG_idler_wavelength(Pump.lam, Signal.lam), PP_SLT, Temperature)
# phase mismatch
delta_k = Pump.k - Signal.k - Idler.k
PP_SLT.poling_period = 1.003 * delta_k

# add coincidence window
tau = 1e-9  # nanosec

###########################################
# Set dataset
##########################################
# Build a dataset of pairs Ai_vac, As_vac
# seed vacuum samples
key = random.PRNGKey(1986)

Nx = len(PP_SLT.x)
Ny = len(PP_SLT.y)
if learn_mode:
    vac_rnd = random.normal(key, (N, 2, 2, Nx, Ny)) # N iteration, 2-for vac states for signal and idler, 2 - real and imag, Nx X Ny for beam size)
    # seed shuffle batches in epochs
    key_batch_epoch = random.split(random.PRNGKey(1989), num_epochs)

# Unwrap G2 indices
G2_unwrapped_idx = onp.ndarray.tolist(unwrap_kron(onp.arange(0, M * M * M * M).reshape(M * M, M * M), M).reshape(M*M*M*M).astype(int))

# normalization factor
g1_ss_normalization = G1_Normalization(Signal.w)
g1_ii_normalization = G1_Normalization(Idler.w)

# params_coef = random.normal(random.PRNGKey(0), (n_coeff, 2))
# params      = np.array(params_coef[:, 0] + 1j*params_coef[:, 1])/np.sqrt(2)

params = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # random_params(n_coeff, random.PRNGKey(0), param_scale)
params = np.divide(params, la.norm(params))

print("--- the parameters initiated are: {} ---".format(params))
print("--- initialization time: %s seconds ---" % (time.time() - start_time_initialization))
start_time = time.time()

# forward model
def predict(params, vac_): # vac_ = vac_s, vac_i
    # initialize the vacuum and output fields:
    Siganl_field    = Field(Signal, PP_SLT, vac_[0])
    Idler_field     = Field(Idler, PP_SLT, vac_[1])
    # current pump structure
    Pump.create_profile(params)
    # Propagate through the crystal:
    crystal_prop(Pump, Siganl_field, Idler_field, PP_SLT)
    # Coumpute k-space far field using FFT:
    # normalization factors
    FarFieldNorm_signal = (2 * PP_SLT.MaxX) ** 2 / (np.size(Siganl_field.E_out) * Signal.lam * R)
    FarFieldNorm_idler  = (2 * PP_SLT.MaxX) ** 2 / (np.size(Idler_field.E_out) * Idler.lam * R)
    # FFT:
    E_s_out_k = FarFieldNorm_signal * Fourier(Siganl_field.E_out)
    E_i_out_k = FarFieldNorm_idler  * Fourier(Idler_field.E_out)
    # E_s_vac_k = FarFieldNorm_signal * Fourier(Siganl_field.E_vac)
    E_i_vac_k = FarFieldNorm_idler  * Fourier(Idler_field.E_vac)

    G1_ss        = kron(np.conj(E_s_out_k), E_s_out_k) / N
    G1_ii        = kron(np.conj(E_i_out_k), E_i_out_k) / N
    G1_si        = kron(np.conj(E_i_out_k), E_s_out_k) / N
    G1_si_dagger = kron(np.conj(E_s_out_k), E_i_out_k) / N

    # Q_ss         = kron(E_s_vac_k, E_s_out_k) / N
    # Q_ss_dagger  = kron(np.conj(E_s_out_k), np.conj(E_s_vac_k)) / N
    # Q_ii         = kron(E_i_vac_k, E_i_out_k) / N
    # Q_ii_dagger  = kron(np.conj(E_i_out_k), np.conj(E_i_vac_k)) / N
    Q_si         = kron(E_i_vac_k, E_s_out_k) / N
    Q_si_dagger  = kron(np.conj(E_s_out_k), np.conj(E_i_vac_k)) / N

    P_ss         = G1_ss[::M + 1, ::M + 1] / g1_ss_normalization
    # P_ii         = G1_ii[::M + 1, ::M + 1] / g1_ii_normalization

    return P_ss, G1_ii, G1_ss, Q_si_dagger, Q_si, G1_si_dagger, G1_si

# Make a batched version of the `predict` function
batched_predict = vmap(predict, in_axes=(None, 0))

def loss(params, vac_, P_ss_t, G2t): # vac_ = vac_s, vac_i, G2t = P and G2 target corrletion matrices
    batched_preds   = batched_predict(params, vac_)
    P_ss, G1_ii, G1_ss, Q_si_dagger, Q_si, G1_si_dagger, G1_si = batched_preds

    P_ss = P_ss.sum(0).real
    P_ss = P_ss / la.norm(P_ss)

    G1_ii = G1_ii.sum(0)
    G1_ss = G1_ss.sum(0)
    Q_si_dagger = Q_si_dagger.sum(0)
    Q_si = Q_si.sum(0)
    G1_si_dagger = G1_si_dagger.sum(0)
    G1_si = G1_si.sum(0)

    G2 = (G1_ii * G1_ss + Q_si_dagger * Q_si + G1_si_dagger * G1_si).real
    G2 = G2.reshape(M * M * M * M)[G2_unwrapped_idx].reshape(M, M, M, M)

    if loss_type is 'l2':
        return la.norm(P_ss - P_ss_t)
    if loss_type is 'kl':
        # KL Divergence #
        """ Epsilon is used here to avoid conditional code for
        checking that neither P nor Q is equal to 0. """
        epsilon = 0
        Q_pss = P_ss + epsilon
        P_pss = P_ss_t + epsilon
        P_ss_loss = np.abs(np.sum(P_pss * np.log(P_pss / Q_pss)))

        epsilon = 1e-7
        Q_g2 = G2 + epsilon
        P_g2 = G2t + epsilon
        G2_loss = np.abs(np.sum(P_g2 * np.log(P_g2 / Q_g2)))

        return (1 - alpha) * P_ss_loss + alpha * G2_loss
    if loss_type is 'wasserstein':
        raise Exception('not implemented yet')
    else:
        raise Exception('Nonstandard loss choice')

@jit
def update(params, x, y1, y2=None):
  grads = grad(loss)(params, x, y1, y2)
  params = [(w - step_size * dw)
          for (w), (dw) in zip(params, grads)]
  return np.divide(np.array(params[:]), la.norm(params))


def get_train_batches(vac_, key_):
    vac_shuff = random.shuffle(key_, vac_, axis=0)
    batch_arr = np.split(vac_shuff, num_batches, axis=0)
    return batch_arr

if learn_mode:
    for epoch in range(num_epochs):
      start_time_epoch = time.time()
      batch_set = get_train_batches(vac_rnd, key_batch_epoch[epoch])

      # batched_preds_ep = batched_predict(params, vac_rnd)
      # P_ss_ep = batched_preds_ep.sum(0).real
      # P_ss_ep = P_ss_ep / la.norm(P_ss_ep)
      # print(f'l2 loss {la.norm(P_ss_ep - P_ss_t)}')

      for x in batch_set:
          params = update(params, x, P_ss_t, G2t)
          # params = ops.index_update(params, ops.index[:], param_scale * nn.softmax(np.array(params[:])))  # normalize to param_scale (positive numbers)
      epoch_time = time.time() - start_time_epoch
      print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
      print("--- the parameters optimized are: {} ---".format(params))


################
# Plot G1(k,k) #
################

# show last epoch result
if save_res or save_tgt or show_res:
    N_res       = 100
    vac_rnd_res = random.normal(key, (N_res, 2, 2, Nx, Ny))
    batched_preds   = batched_predict(params, vac_rnd_res)
    P_ss, G1_ii, G1_ss, Q_si_dagger, Q_si, G1_si_dagger, G1_si = batched_preds

    P_ss            = P_ss.sum(0).real
    P_ss            = P_ss/la.norm(P_ss)

    G1_ii        = G1_ii.sum(0)
    G1_ss        = G1_ss.sum(0)
    Q_si_dagger  = Q_si_dagger.sum(0)
    Q_si         = Q_si.sum(0)
    G1_si_dagger = G1_si_dagger.sum(0)
    G1_si        = G1_si.sum(0)

    G2           = (G1_ii * G1_ss + Q_si_dagger * Q_si + G1_si_dagger * G1_si).real
    G2           = G2.reshape(M * M * M * M)[G2_unwrapped_idx].reshape(M, M, M, M)

if save_res:
    np.save(res_path + res_name_pss, P_ss)
    np.save(res_path + res_name_g2, G2)

if save_tgt:
    np.save(Pt_path + Pss_t_name, P_ss)
    np.save(Pt_path + G2_t_name, G2)

if show_res:
    FF_position_axis = lambda dx, MaxX, k, R: np.arange(-1 / dx, 1 / dx, 1 / MaxX) * (np.pi * R / k)
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
    plt.imshow(P_ss * 1e-6, extent=extents_FFcoordinates_signal)  # AK, Dec08: Units of counts/mm^2*sec
    plt.plot(1e3 * R * np.tan(theoretical_angle), 0, 'xw')
    plt.xlabel(' x [mm]')
    plt.ylabel(' y [mm]')
    plt.title('Single photo-detection probability, Far field')
    plt.colorbar()
    plt.show()


    # Fourier coordiantes
    dx_farfield_idler = 1e-3 * (FFcoordinate_axis_Idler[1] - FFcoordinate_axis_Idler[0])
    dx_farfield_signal = 1e-3 * (FFcoordinate_axis_Signal[1] - FFcoordinate_axis_Signal[0])

    # add coincidence window
    tau = 1e-9  # nanosec

    # Compute and plot reduced G2
    G2_reduced = trace_it(G2, 1, 3) * dx_farfield_idler * dx_farfield_signal
    G2_reduced = G2_reduced * tau / (g1_ii_normalization * g1_ss_normalization)

    # plot
    plt.figure()
    plt.imshow(1e-6 * G2_reduced, extent=extents_FFcoordinates_signal)  # AK, Dec08: G2 in counts/sec/mm^2
    plt.title(r'$G^{(2)}$ (coincidences)')
    plt.xlabel(r'$x_i$ [mm]')
    plt.ylabel(r'$x_s$ [mm]')
    plt.colorbar()
    plt.show()

print("--- the parameters optimized are: {} ---".format(params))
print("--- running time: %s seconds ---" % (time.time() - start_time))
exit()