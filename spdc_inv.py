from __future__ import print_function, division, absolute_import
from jax import value_and_grad, pmap, lax
from jax.numpy import kron
from jax.numpy import linalg as la
from spdc_helper import *
from jax.experimental import optimizers
import matplotlib.pyplot as plt
from jax.lib import xla_bridge
from functools import partial
import os, time
from datetime import datetime
import jax.numpy as np
import numpy as onp
from loss_funcs import l1_loss, kl_loss, sinkhorn_loss

# datetime object containing current date and time
now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)

JAX_ENABLE_X64 = False

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

num_devices = xla_bridge.device_count()
start_time_initialization = time.time()

learn_mode = False  # learn/infer
show_res   = True   # display results 0/1
save_res   = False  # save results
save_tgt   = False  # save targets
DO_HG      = True   # Flag for projecting to HG modes

res_path      = 'results/'  # path to results folder
Pt_path       = 'targets/'  # path to targets folder

"Learning Hyperparameters"
loss_type   = 'l1'  # l1:L1 Norm, kl:Kullback–Leibler Divergence, wass: Wasserstein (Sinkhorn) Distance", kll1: kl+gamma*l1
step_size   = 0.01
num_epochs  = 500
batch_size  = 100   # 10, 20, 50, 100 - number of iterations
N           = 10000  # 100, 500, 1000  - number of total-iterations (dataset size)
alpha       = 0.5   # in [0,1]; weight for loss: (1-alpha)Pss + alpha G2
gamma       = 1.0  # in [0, inf]; balance loss kll1: kl+gamma*l1
num_batches = int(N/batch_size)
Ndevice     = int(N/num_devices)
batch_device= int(batch_size/num_devices)

assert N % batch_size == 0, "num_batches should be 'signed integer'"
assert N % num_devices == 0, "The number of examples should be divisible by the number of devices"
assert batch_size % num_devices == 0, "The number of examples within a batch should be divisible by the number of devices"



"Interaction Initialization"
# Structure arrays - initialize crystal and structure arrays
d33         = 23.4e-12  # in meter/Volt.[LiNbO3]
PP_SLT      = Crystal(10e-6, 10e-6, 1e-5, 200e-6, 200e-6, 5e-3, d33)
R           = 0.1  # distance to far-field screen in meters
Temperature = 50
M           = len(PP_SLT.x)  # simulation size

# Interacting wavelengths - initialize the interacting beams
"""
    * define two pump's function (for now n_coeff must be 2) to define the pump *
    * this should be later changed to the definition given by Sivan *
"""
max_mode = 4
n_coeff  = 2 ** max_mode  # coefficients of beam-basis functions
Pump     = Beam(532e-9, PP_SLT, Temperature, 50e-6, 1e-3, max_mode)  # wavelength, crystal, tmperature,waist,power, maxmode
Signal   = Beam(1064e-9, PP_SLT, Temperature, np.sqrt(2)*Pump.waist, 1, max_mode)
Idler    = Beam(SFG_idler_wavelength(Pump.lam, Signal.lam), PP_SLT, Temperature, np.sqrt(2)*Pump.waist, 1, max_mode)

# phase mismatch
delta_k = Pump.k - Signal.k - Idler.k
PP_SLT.poling_period = 1.0 * delta_k
# set coincidence window
tau = 1e-9  # [nanosec]


"Interaction Parameters"
Nx = len(PP_SLT.x)
Ny = len(PP_SLT.y)


# normalization factor
g1_ss_normalization = G1_Normalization(Signal.w)
g1_ii_normalization = G1_Normalization(Idler.w)

# params_coef = random.normal(random.PRNGKey(0), (n_coeff, 2))
# params      = np.array(params_coef[:, 0] + 1j*params_coef[:, 1])
params = np.array([1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
params = np.divide(params, la.norm(params))
params_gt = np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# replicate parameters for gpus
params = pmap(lambda x: params)(np.arange(num_devices))

print("--- the parameters initiated are: {} ---".format(params[0]))
print("--- initialization time: %s seconds ---" % (time.time() - start_time_initialization))
start_time = time.time()

# forward model
def forward(params, vac_): # vac_ = vac_s, vac_i
    N = vac_.shape[0]

    # initialize the vacuum and output fields:
    Siganl_field    = Field(Signal, PP_SLT, vac_[:, 0], N)
    Idler_field     = Field(Idler, PP_SLT, vac_[:, 1], N)

    # current pump structure
    Pump.create_profile(params, N)

    # Propagate through the crystal:
    crystal_prop(Pump, Siganl_field, Idler_field, PP_SLT)

    if DO_HG:
        E_s_out_HG = np.reshape(decompose(Siganl_field.E_out, Signal.hemite_dict, N), (N, max_mode, max_mode))
        E_i_out_HG = np.reshape(decompose(Idler_field.E_out, Signal.hemite_dict, N), (N, max_mode, max_mode))
        E_i_vac_HG = np.reshape(decompose(Idler_field.E_vac, Signal.hemite_dict, N), (N, max_mode, max_mode))

        # say there are no higher modes by normalizing the power
        E_s_out = fix_power1(E_s_out_HG, Siganl_field.E_out, Signal, PP_SLT)
        E_i_out = fix_power1(E_i_out_HG, Idler_field.E_out, Signal, PP_SLT)
        E_i_vac = fix_power1(E_i_vac_HG, Idler_field.E_vac, Signal, PP_SLT)
    else:
        "Coumpute k-space far field using FFT:"
        # normalization factors
        FarFieldNorm_signal = (2 * PP_SLT.MaxX) ** 2 / (np.size(Siganl_field.E_out) * Signal.lam * R)
        FarFieldNorm_idler  = (2 * PP_SLT.MaxX) ** 2 / (np.size(Idler_field.E_out) * Idler.lam * R)
        # FFT:
        E_s_out = FarFieldNorm_signal * Fourier(Siganl_field.E_out)
        E_i_out = FarFieldNorm_idler  * Fourier(Idler_field.E_out)
        E_i_vac = FarFieldNorm_idler  * Fourier(Idler_field.E_vac)

    G1_ss        = (np.array([kron(np.conj(E_s_out[i]), E_s_out[i]) for i in range(N)]) / batch_size).sum(0)
    G1_ii        = (np.array([kron(np.conj(E_i_out[i]), E_i_out[i]) for i in range(N)]) / batch_size).sum(0)
    G1_si        = (np.array([kron(np.conj(E_i_out[i]), E_s_out[i]) for i in range(N)]) / batch_size).sum(0)
    G1_si_dagger = (np.array([kron(np.conj(E_s_out[i]), E_i_out[i]) for i in range(N)]) / batch_size).sum(0)
    Q_si         = (np.array([kron(E_i_vac[i], E_s_out[i]) for i in range(N)]) / batch_size).sum(0)
    Q_si_dagger  = (np.array([kron(np.conj(E_s_out[i]), np.conj(E_i_vac[i])) for i in range(N)]) / batch_size).sum(0)

    return G1_ii, G1_ss, Q_si_dagger, Q_si, G1_si_dagger, G1_si

def loss(params, vac_, P_ss_t, G2t):  # vac_ = vac_s, vac_i, G2t = P and G2 target correlation matrices
    params = np.divide(params, la.norm(params))
    batched_preds   = forward(params, vac_)
    G1_ii, G1_ss, Q_si_dagger, Q_si, G1_si_dagger, G1_si = batched_preds

    if DO_HG:
        P_ss = G1_ss[::max_mode + 1, ::max_mode + 1].real
    else:
        P_ss     = G1_ss[::M + 1, ::M + 1].real
    G2           = (G1_ii * G1_ss + Q_si_dagger * Q_si + G1_si_dagger * G1_si).real

    P_ss    = P_ss / np.sum(np.abs(P_ss))
    G2      = G2 / np.sum(np.abs(G2))

    if loss_type is 'l1':
        return (1-alpha)*l1_loss(P_ss, P_ss_t) + alpha*l1_loss(G2, G2t)
    if loss_type is 'kl':
        return (1-alpha)*kl_loss(P_ss_t, P_ss, eps=1e-7)+alpha*kl_loss(G2t, G2, eps=1)
    if loss_type is 'kll1':
        l1_l = (1-alpha)*l1_loss(P_ss, P_ss_t) + alpha*l1_loss(G2, G2t)
        kl_l = (1-alpha)*kl_loss(P_ss_t, P_ss, eps=1e-7)+alpha*kl_loss(G2t, G2, eps=1)
        return kl_l + gamma * l1_l
    if loss_type is 'wass':
        return 1e7*sinkhorn_loss(P_ss, P_ss_t, M, eps=1e-3, max_iters=100, stop_thresh=None)
    if loss_type is 'wass_kl':
        return 1e7*sinkhorn_loss(P_ss, P_ss_t, M, eps=1e-3, max_iters=100, stop_thresh=None) + kl_loss(G2t, G2, eps=1)
    else:
        raise Exception('Nonstandard loss choice')

@partial(pmap, axis_name='device')
def update(opt_state, i, x, P_ss_t, G2t):
    params              = get_params(opt_state)
    batch_loss, grads   = value_and_grad(loss)(params, x, P_ss_t, G2t)
    grads               = np.array([lax.psum(dw, 'device') for dw in grads])
    return lax.pmean(batch_loss, 'device'), opt_update(i, grads, opt_state)


if learn_mode:
    # load target P, G2
    Pss_t_load = 'P_ss_HG00_N{}_Nx{}Ny{}'.format(batch_size, Nx, Ny)
    G2_t_load  = 'G2_HG00_N{}_Nx{}Ny{}'.format(batch_size, Nx, Ny)
    P_ss_t  = pmap(lambda x: np.load(Pt_path + Pss_t_load + '.npy'))(np.arange(num_devices))
    G2t     = pmap(lambda x: np.load(Pt_path + G2_t_load + '.npy'))(np.arange(num_devices))

    """Set dataset - Build a dataset of pairs Ai_vac, As_vac"""
    # seed vacuum samples
    keys = random.split(random.PRNGKey(1986), num_devices)
    # generate dataset for each gpu
    vac_rnd = pmap(lambda key: random.normal(key, (Ndevice, 2, 2, Nx, Ny)))(keys) # N iteration for device, 2-for vac states for signal and idler, 2 - real and imag, Nx X Ny for beam size)

    # split to batches
    def get_train_batches(vac_, key_):
        vac_shuff = random.permutation(key_, vac_)
        batch_arr = np.split(vac_shuff, num_batches, axis=0)
        return batch_arr

    # seed shuffle batches in epochs
    key_batch_epoch = pmap(lambda i: random.split(random.PRNGKey(1989), num_epochs))(np.arange(num_devices))

    # Use optimizers to set optimizer initialization and update functions
    opt_init, opt_update, get_params = optimizers.adam(step_size, b1=0.9, b2=0.999, eps=1e-08)
    opt_state = opt_init(params)
    losses = []
    for epoch in range(num_epochs):
        losses.append(0.0)
        start_time_epoch = time.time()
        batch_set = pmap(get_train_batches)(vac_rnd, key_batch_epoch[:, epoch])
        print("Epoch {} is running".format(epoch))
        for i, x in enumerate(batch_set):
            idx = pmap(lambda x: np.array(epoch+i))(np.arange(num_devices))
            batch_loss, opt_state = update(opt_state, idx, x, P_ss_t, G2t)
            losses[epoch] += batch_loss[0].item()
        params_ = get_params(opt_state)[0]
        losses[epoch] /= (i+1)
        epoch_time = time.time() - start_time_epoch
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        ''' print loss value'''
        print("optimized parameters: {}".format(params_))
        print("objective loss:{:0.6f}".format(losses[epoch]))
        print("l1 loss:{:0.6f}".format(np.sum(np.abs(params_-params_gt))))
        print("l2 loss:{:0.6f}".format(np.sum((params_-params_gt)**2)))

# show last epoch result
if save_res or save_tgt or show_res:
    N_res = batch_size
    ###########################################
    # Set dataset
    ##########################################
    # Build a dataset of pairs Ai_vac, As_vac

        # seed vacuum samples
        keys = random.split(random.PRNGKey(1986), num_devices)
        # generate dataset for each gpu
        vac_rnd = pmap(lambda key: random.normal(key, (batch_device, 2, 2, Nx, Ny)))(keys)  # number of devices, N iteration, 2-for vac states for signal and idler, 2 - real and imag, Nx X Ny for beam size)

    batched_preds = pmap(forward)(params, vac_rnd)

    G1_ii, G1_ss, Q_si_dagger, Q_si, G1_si_dagger, G1_si = batched_preds

    G1_ii        = G1_ii.sum(0)
    G1_ss        = G1_ss.sum(0)
    Q_si_dagger  = Q_si_dagger.sum(0)
    Q_si         = Q_si.sum(0)
    G1_si_dagger = G1_si_dagger.sum(0)
    G1_si        = G1_si.sum(0)

    P_ss         = G1_ss[::M + 1, ::M + 1].real
    G2           = (G1_ii * G1_ss + Q_si_dagger * Q_si + G1_si_dagger * G1_si).real

if save_res:
    res_name_pss = '2020_05_07_P_ss_HG00PHG33_to_HG00_N120'
    res_name_g2 = '2020_05_07_G2_HG00PHG33_to_HG00_N120'
    np.save(res_path + res_name_pss, P_ss)
    np.save(res_path + res_name_g2, G2)

if save_tgt:
    Pss_t_name = 'P_ss_HG00_N{}_Nx{}Ny{}'.format(batch_size,Nx,Ny)
    G2_t_name = 'G2_HG00_N{}_Nx{}Ny{}'.format(batch_size,Nx,Ny)
    # save normalized version
    np.save(Pt_path + Pss_t_name, P_ss/np.sum(np.abs(P_ss)))
    np.save(Pt_path + G2_t_name, G2/np.sum(np.abs(G2)))

if show_res:
    ################
    # Plot G1 #
    ################
    if not DO_HG:
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

    P_ss /= g1_ss_normalization

    plt.figure()
    plt.imshow(P_ss * 1e-6, extent=extents_FFcoordinates_signal)  # AK, Dec08: Units of counts/mm^2*sec
    plt.plot(1e3 * R * np.tan(theoretical_angle), 0, 'xw')
    plt.xlabel(' x [mm]')
    plt.ylabel(' y [mm]')
    plt.title('Single photo-detection probability, Far field')
    plt.colorbar()
    plt.show()

    ################
    # Plot G2 #
    ################
    if DO_HG:
        G2_unwrapped_idx_np = onp.zeros((max_mode, max_mode, max_mode, max_mode), dtype='int32')
        G2_unwrapped_idx_np = \
            unwrap_kron(G2_unwrapped_idx_np,
                        onp.arange(0, max_mode * max_mode * max_mode * max_mode, dtype='int32').reshape(max_mode * max_mode, max_mode * max_mode),
                        max_mode).reshape(max_mode * max_mode * max_mode * max_mode).astype(int)

        G2_unwrapped_idx = onp.ndarray.tolist(G2_unwrapped_idx_np)
        G2 = G2.reshape(max_mode * max_mode * max_mode * max_mode)[G2_unwrapped_idx].reshape(max_mode, max_mode, max_mode, max_mode)

        # add coincidence window
        tau = 1e-9  # nanosec

        # Compute and plot reduced G2
        G2_reduced = trace_it(G2, 0, 2)
        G2_reduced = G2_reduced * tau / (g1_ii_normalization * g1_ss_normalization)

        # plot
        plt.figure()
        plt.imshow(G2_reduced)
        plt.title(r'$G^{(2)}$ (coincidences)')
        plt.xlabel(r'signal mode i')
        plt.ylabel(r'idle mode j')
        plt.colorbar()
        plt.show()
    else:
        # Unwrap G2 indices
        G2_unwrap_idx_str = 'G2_unwarp_idx/G2_unwrap_M{}.npy'.format(M)
        if not os.path.exists(G2_unwrap_idx_str):
        G2_unwrapped_idx_np = onp.zeros((M, M, M, M), dtype='int32')
        G2_unwrapped_idx_np = \
            unwrap_kron(G2_unwrapped_idx_np,
                        onp.arange(0, M * M * M * M, dtype='int32').reshape(M * M, M * M),
                        M).reshape(M * M * M * M).astype(int)

        np.save(G2_unwrap_idx_str, G2_unwrapped_idx_np)

    else:
        G2_unwrapped_idx_np = np.load(G2_unwrap_idx_str)
    G2_unwrapped_idx = onp.ndarray.tolist(G2_unwrapped_idx_np)
    del G2_unwrapped_idx_np

    G2 = G2.reshape(M * M * M * M)[G2_unwrapped_idx].reshape(M, M, M, M)
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
        plt.imshow(1e-6 * G2_reduced, extent=extents_FFcoordinates_signal)
        plt.title(r'$G^{(2)}$ (coincidences)')
        plt.xlabel(r'$x_i$ [mm]')
        plt.ylabel(r'$x_s$ [mm]')
    plt.colorbar()
    plt.show()

print("--- running time: %s seconds ---" % (time.time() - start_time))
exit()
