from __future__ import print_function, division, absolute_import
from jax import value_and_grad, pmap, lax
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
from jax.ops import index_update
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
max_mode = 10
n_coeff  = max_mode**2  # coefficients of beam-basis functions
Pump     = Beam(532e-9, PP_SLT, Temperature, 50e-6, 1e-3, max_mode)  # wavelength, crystal, tmperature,waist,power, maxmode
Signal   = Beam(1064e-9, PP_SLT, Temperature, np.sqrt(2)*Pump.waist, 1, max_mode)
Idler    = Beam(SFG_idler_wavelength(Pump.lam, Signal.lam), PP_SLT, Temperature, np.sqrt(2)*Pump.waist, 1)

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

# coeffs_rand = random.normal(random.PRNGKey(0), (n_coeff, 2))
# coeffs      = np.array(coeffs_rand[:, 0] + 1j*coeffs_rand[:, 1])
coeffs = np.zeros(n_coeff)
coeffs = index_update(coeffs, 2, 1.0)
coeffs = index_update(coeffs, 3, 1.0)
coeffs = index_update(coeffs, 14, 1.0)
coeffs = index_update(coeffs, 35, 1.0)
coeffs = np.divide(coeffs, la.norm(coeffs))

coeffs_gt = np.zeros(n_coeff)
coeffs_gt = index_update(coeffs_gt, 0, 1.0)
coeffs_gt = np.divide(coeffs_gt, la.norm(coeffs_gt))
# replicate parameters for gpus
# coeffs = pmap(lambda x: coeffs)(np.arange(num_devices))

poling_str = 'no_tr_phase'
phi_parameters = np.array([-120.0, 0, 720, 0, -480, 0, 64, 0, 0, 0, 0])  # no_tr_phase
# phi_parameters = np.array([0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # linear shift
# phi_parameters = [0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0]  # Lens
# phi_parameters = [0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0]  # cube?
# phi_parameters = [12, 0, -48, 0, 16, 0, 0, 0, 0, 0, 0]  # Hermite4
# phi_parameters = np.array([-120, 0, 720, 0, -480, 0, 64, 0, 0, 0, 0])  # Hermite6
phi_parameters_gt = np.array([0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # no_tr_phase
phi_scale      = 1
NormX          = PP_SLT.x / PP_SLT.MaxX
taylor_series  = np.array([NormX**i for i in range(len(phi_parameters))])
Poling = Poling_profile(taylor_series)

# phi_parameters = pmap(lambda x: phi_parameters)(np.arange(num_devices))
params = np.concatenate((coeffs, phi_parameters))
params = pmap(lambda x: params)(np.arange(num_devices))

print("--- the HG coefficients initiated are: {} ---\n".format(coeffs))
print("--- the Taylor coefficients initiated are: {} ---\n".format(phi_parameters))
print("--- initialization time: %s seconds ---" % (time.time() - start_time_initialization))
start_time = time.time()

# forward model
def forward(params, vac_): # vac_ = vac_s, vac_i
    N = vac_.shape[0]
    coeffs, phi_parameters = params[:n_coeff], params[n_coeff:]

    # initialize the vacuum and output fields:
    Siganl_field    = Field(Signal, PP_SLT, vac_[:, 0], N)
    Idler_field     = Field(Idler, PP_SLT, vac_[:, 1], N)

    # current pump structure
    Pump.create_profile(coeffs)

    # current poling profile
    Poling.create_profile(phi_parameters)

    # Propagate through the crystal:
    crystal_prop(Pump, Siganl_field, Idler_field, PP_SLT, Poling)


    E_s_out = decompose(Siganl_field.E_out, Signal.hermite_arr, N, max_mode)
    E_i_out = decompose(Idler_field.E_out, Signal.hermite_arr, N, max_mode)
    E_i_vac = decompose(Idler_field.E_vac, Signal.hermite_arr, N, max_mode)
    # say there are no higher modes by normalizing the power
    E_s_out = fix_power1(E_s_out, Siganl_field.E_out, Signal, PP_SLT)
    E_i_out = fix_power1(E_i_out, Idler_field.E_out, Signal, PP_SLT)
    E_i_vac = fix_power1(E_i_vac, Idler_field.E_vac, Signal, PP_SLT)

    "Coumpute k-space far field using FFT:"
    # normalization factors
    FarFieldNorm_signal = (2 * PP_SLT.MaxX) ** 2 / (np.size(Siganl_field.E_out) * Signal.lam * R)
    # FFT:
    E_s_out_k = FarFieldNorm_signal * Fourier(Siganl_field.E_out)

    P_ss         = ((np.conj(E_s_out_k)[:,:, None, :, None] * E_s_out_k[:,None, :, None, :]).real.sum(0).reshape(Nx**2, Ny**2))[::M + 1, ::M + 1] / batch_size
    G1_ss        = (np.conj(E_s_out)[:,:, None, :, None] * E_s_out[:,None, :, None, :]).sum(0).reshape(max_mode**2, max_mode**2) / batch_size
    G1_ii        = (np.conj(E_i_out)[:,:, None, :, None] * E_i_out[:,None, :, None, :]).sum(0).reshape(max_mode**2, max_mode**2) / batch_size
    G1_si        = (np.conj(E_i_out)[:,:, None, :, None] * E_s_out[:,None, :, None, :]).sum(0).reshape(max_mode**2, max_mode**2) / batch_size
    G1_si_dagger = (np.conj(E_s_out)[:,:, None, :, None] * E_i_out[:,None, :, None, :]).sum(0).reshape(max_mode**2, max_mode**2) / batch_size
    Q_si         = (E_i_vac[:,:, None, :, None] * E_s_out[:,None, :, None, :]).sum(0).reshape(max_mode**2, max_mode**2) / batch_size
    Q_si_dagger  = (np.conj(E_s_out)[:,:, None, :, None] * np.conj(E_i_vac)[:,None, :, None, :]).sum(0).reshape(max_mode**2, max_mode**2) / batch_size
    G2           = (G1_ii * G1_ss + Q_si_dagger * Q_si + G1_si_dagger * G1_si).real
    return P_ss, G2

def loss(params, vac_, P_ss_t, G2t):  # vac_ = vac_s, vac_i, G2t = P and G2 target correlation matrices
    coeffs, phi_parameters = params[:n_coeff], params[n_coeff:]
    coeffs = np.divide(coeffs, la.norm(coeffs))
    params = np.concatenate((coeffs, phi_parameters))

    P_ss, G2 = forward(params, vac_)
    P_ss     = P_ss / np.sum(np.abs(P_ss))
    G2       = G2 / np.sum(np.abs(G2))

    if loss_type is 'l1':
        return (1-alpha)*l1_loss(P_ss, P_ss_t) + alpha*l1_loss(G2, G2t)
    if loss_type is 'kl':
        return (1-alpha)*kl_loss(P_ss_t, P_ss, eps=1e-7)+alpha*kl_loss(G2t, G2, eps=1)
    if loss_type is 'kll1':
        l1_l = (1-alpha)*l1_loss(P_ss, P_ss_t) + alpha*l1_loss(G2, G2t)
        kl_l = (1-alpha)*kl_loss(P_ss_t, P_ss, eps=1e-7)+alpha*kl_loss(G2t, G2, eps=1)
        return kl_l + gamma * l1_l
    if loss_type is 'wass':
        return (1 - alpha) * sinkhorn_loss(P_ss, P_ss_t, M, eps=1e-3, max_iters=100, stop_thresh=None) + \
               alpha * sinkhorn_loss(G2, G2t, M, eps=1e-3, max_iters=100, stop_thresh=None)
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
    Pss_t_load = 'P_ss_1.0HG00_no_tr_phase_N100_Nx40Ny40_z0.005_steps500'
    G2_t_load  = 'G2_1.0HG00_no_tr_phase_N100_Nx40Ny40_z0.005_steps500'
    P_ss_t     = pmap(lambda x: np.load(Pt_path + Pss_t_load + '.npy'))(np.arange(num_devices))
    G2t        = pmap(lambda x: np.load(Pt_path + G2_t_load + '.npy'))(np.arange(num_devices))

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
        params = get_params(opt_state)
        losses[epoch] /= (i+1)
        epoch_time = time.time() - start_time_epoch
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        ''' print loss value'''
        coeffs, phi_parameters = params[0][:n_coeff], params[0][n_coeff:]
        print("optimized HG coefficients: {}".format(coeffs))
        print("optimized Taylor coefficients: {}".format(phi_parameters))
        print("objective loss:{:0.6f}".format(losses[epoch]))
        print("l1 HG coeffs loss:{:0.6f}".format(np.sum(np.abs(coeffs-coeffs_gt))))
        print("l2 HG coeffs loss:{:0.6f}".format(np.sum((coeffs-coeffs_gt)**2)))
        print("l1 Taylor coeffs loss:{:0.6f}".format(np.sum(np.abs(phi_parameters - phi_parameters_gt))))
        print("l2 Taylor coeffs loss:{:0.6f}".format(np.sum((phi_parameters - phi_parameters_gt) ** 2)))

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

    P_ss, G2 = pmap(forward)(params, vac_rnd)
    P_ss     = P_ss.sum(0)
    G2       = G2.sum(0)

    if save_tgt:
        HG_str = make_beam_from_HG_str(Pump.hermite_str, coeffs)
        Pss_t_name = 'P_ss' + HG_str + '_{}_N{}_Nx{}Ny{}_z{}_steps{}'.format(poling_str, batch_size, Nx, Ny, PP_SLT.MaxZ, len(PP_SLT.z))
        G2_t_name = 'G2' + HG_str + '_{}_N{}_Nx{}Ny{}_z{}_steps{}'.format(poling_str, batch_size, Nx, Ny, PP_SLT.MaxZ, len(PP_SLT.z))
        # save normalized version
        np.save(Pt_path + Pss_t_name, P_ss/np.sum(np.abs(P_ss)))
        np.save(Pt_path + G2_t_name, G2/np.sum(np.abs(G2)))

    if show_res or save_res:
        ################
        # Plot G1 #
        ################
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

        plt.imshow(P_ss * 1e-6, extent=extents_FFcoordinates_signal)  # AK, Dec08: Units of counts/mm^2*sec
        plt.plot(1e3 * R * np.tan(theoretical_angle), 0, 'xw')
        plt.xlabel(' x [mm]')
    plt.ylabel(' y [mm]')
        plt.title('Single photo-detection probability, Far field')
        plt.colorbar()
        if save_res:
            res_name_pss = 'P_ss'
            HG_str = make_beam_from_HG_str(Pump.hermite_str, coeffs)
            if learn_mode:
                plt.savefig(res_path + now.strftime("%_Y-%m-%d_") + res_name_pss + HG_str + '_{}_N{}_Nx{}Ny{}_z{}_steps{}_{}.png'.format(poling_str, batch_size, Nx, Ny, PP_SLT.MaxZ, len(PP_SLT.z), loss_type))
            else:
                plt.savefig(res_path + now.strftime("%_Y-%m-%d_") + res_name_pss + HG_str + '_{}_N{}_Nx{}Ny{}_z{}_steps{}.png'.format(poling_str, batch_size, Nx, Ny, PP_SLT.MaxZ, len(PP_SLT.z)))
        if show_res:
            plt.show()

        ################
        # Plot G2 #
        ################
        # Unwrap G2 indices
        G2_unwrap_idx_str = 'G2_unwarp_idx/G2_unwrap_max_mode{}.npy'.format(max_mode)
        if not os.path.exists(G2_unwrap_idx_str):
            G2_unwrapped_idx_np = onp.zeros((max_mode, max_mode, max_mode, max_mode), dtype='int32')
            G2_unwrapped_idx_np = \
                unwrap_kron(G2_unwrapped_idx_np,
                            onp.arange(0, max_mode * max_mode * max_mode * max_mode, dtype='int32').reshape(max_mode * max_mode, max_mode * max_mode),
                            max_mode).reshape(max_mode * max_mode * max_mode * max_mode).astype(int)

            np.save(G2_unwrap_idx_str, G2_unwrapped_idx_np)

        else:
        G2_unwrapped_idx_np = np.load(G2_unwrap_idx_str)
        G2_unwrapped_idx = onp.ndarray.tolist(G2_unwrapped_idx_np)
        del G2_unwrapped_idx_np

        G2 = G2.reshape(max_mode * max_mode * max_mode * max_mode)[G2_unwrapped_idx].reshape(max_mode, max_mode, max_mode, max_mode)

        # Compute and plot reduced G2
        G2_reduced = trace_it(G2, 0, 2)
        G2_reduced = G2_reduced * tau / (g1_ii_normalization * g1_ss_normalization)

        # plot
        plt.imshow(G2_reduced)
        plt.title(r'$G^{(2)}$ (coincidences)')
        plt.xlabel(r'signal mode i')
        plt.ylabel(r'idle mode j')
        plt.colorbar()
        if save_res:
            res_name_g2 = 'G2'
            HG_str = make_beam_from_HG_str(Pump.hermite_str, coeffs)
            if learn_mode:
                plt.savefig(res_path + now.strftime("%_Y-%m-%d_") + res_name_g2 + HG_str + '_{}_N{}_Nx{}Ny{}_z{}_steps{}_{}.png'.format(poling_str, batch_size, Nx, Ny, PP_SLT.MaxZ, len(PP_SLT.z), loss_type))
            else:
                plt.savefig(res_path + now.strftime("%_Y-%m-%d_") + res_name_g2 + HG_str + '_{}_N{}_Nx{}Ny{}_z{}_steps{}.png'.format(poling_str, batch_size, Nx, Ny, PP_SLT.MaxZ, len(PP_SLT.z)))
        if show_res:
            plt.show()

        # crystal's pattern
        XX, ZZ = np.meshgrid(PP_SLT.x, PP_SLT.z)
        Poling.create_profile(phi_parameters)
        plt.imshow(np.sign(np.cos(PP_SLT.poling_period * ZZ + Poling.phi)), aspect='auto')
        plt.xlabel(' x [mm]')
        plt.ylabel(' z [mm]')
        plt.title('Crystal\'s poling pattern')
        plt.colorbar()
        if learn_mode:
            plt.savefig(res_path + now.strftime("%_Y-%m-%d_") + 'poling_{}_N{}_Nx{}Ny{}_z{}_steps{}_{}.png'.format(poling_str, batch_size, Nx, Ny, PP_SLT.MaxZ, len(PP_SLT.z), loss_type))
        else:
            plt.savefig(res_path + now.strftime("%_Y-%m-%d_") + 'poling_{}_N{}_Nx{}Ny{}_z{}_steps{}.png'.format(poling_str, batch_size, Nx, Ny, PP_SLT.MaxZ, len(PP_SLT.z)))
        if show_res:
            plt.show()

print("--- running time: %s seconds ---" % (time.time() - start_time))
exit()
