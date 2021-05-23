from __future__ import print_function, division, absolute_import

import os
os.environ["JAX_ENABLE_X64"] = 'True'
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = 'platform'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,3'

from loss import l1_loss, kl_loss, sinkhorn_loss, l2_loss
from utils import *
from funcs import *
from physical_params import *

# datetime object containing current date and time
now = datetime.now()
print("\n date and time =%s \n" %now.strftime("%d/%m/%Y %H:%M:%S"))
start_time_initialization = time.time()

learn_mode = False  # learn/infer
save_stats = True
show_res = False  # display results 0/1
save_res = True  # save results
save_tgt = False  # save targets

res_path = 'results/'  # path to results folder
Pt_path = 'targets/'  # path to targets folder
stats_path = 'stats/'

seed = 1800

"Learning Hyperparameters"
loss_type = 'kl_sparse_balanced'  # l1:L1 Norm, kl:Kullback Leibler Divergence, wass: Wasserstein (Sinkhorn) Distance"
step_size = 0.05
num_epochs = 200
batch_size = 50  # 10, 20, 50, 100 - number of iterations
N          = 200  # batch_size   # 100, 500, 1000  - number of total-iterations (dataset size)

num_devices = xla_bridge.device_count()
num_batches = int(N / batch_size / num_devices)

print(f'Number of GPU devices:                     num_devices = {num_devices}')
print(f'Total number of vacuum states:             N           = {N}')
print(f'Number of vacuum states per single device: batch_size  = {batch_size}')
print(f'Number batches per single device:          num_batches = {num_batches}')
print(f'\nThe dimensions of the problem should agree with: \n'
      f'N = batch_size*num_devices*num_batches ->  {N} = {batch_size}*{num_devices}*{num_batches} \n')

assert N % (batch_size * num_devices) == 0, "num_batches should be 'signed integer'"

"Interaction Initialization"
# Structure arrays - initialize crystal and structure arrays
PP_crystal = Crystal(dx, dy, dz, MaxX, MaxY, MaxZ, d33)
M = len(PP_crystal.x)  # simulation size

n_coeff, max_mode1, max_mode2, max_mode_crystal = projection_crystal_modes()

Pump = Beam(lam_pump, PP_crystal, Temperature, waist_pump, power_pump, projection_type, max_mode1,
            max_mode2)  # wavelength, crystal, tmperature,waist,power, maxmode
Signal = Beam(lam_signal, PP_crystal, Temperature, np.sqrt(2) * Pump.waist, power_signal, projection_type, max_mode1,
              max_mode2, z=MaxZ / 2)
Idler = Beam(SFG_idler_wavelength(Pump.lam, Signal.lam), PP_crystal, Temperature, np.sqrt(2) * Pump.waist, power_signal,
             projection_type)

# phase mismatch
delta_k = Pump.k - Signal.k - Idler.k
PP_crystal.poling_period = dk_offset * delta_k
"Interaction Parameters"
Nx = len(PP_crystal.x)
Ny = len(PP_crystal.y)

# normalization factor
g1_ss_normalization = G1_Normalization(Signal.w)
g1_ii_normalization = G1_Normalization(Idler.w)

# Initialize pump and crystal coefficients
coeffs_real, coeffs_imag = Pump_coeff_array(coeffs_str, max_mode1 * max_mode2)

# replicate parameters for gpus
coeffs = pmap(lambda x: np.concatenate((coeffs_real, coeffs_imag)))(np.arange(num_devices))

print("--- Pump waist {:.2f}um for Crystal length {}m ---\n".format(waist_pump * 1e6, MaxZ))
print("--- the LG coefficients initiated are: {} ---\n".format(coeffs_real + 1j * coeffs_imag))
print("--- initialization time: %s seconds ---" % (time.time() - start_time_initialization))
start_time = time.time()

topic = now.strftime("%_Y-%m-%d") + "_Nb{}_N{}_Nx{}Ny{}_z{}_steps{}_#devices{}_new_version".format(
    batch_size, N, Nx, Ny, PP_crystal.MaxZ, len(PP_crystal.z), num_devices)

if learn_mode:
    topic += "_loss_{}".format(loss_type) + "_epochs{}".format(num_epochs) + "_complex"


def forward(coeffs, key):
    coeffs_real, coeffs_imag = coeffs[:n_coeff], coeffs[n_coeff:2 * n_coeff]
    G1_ii, G1_ss, Q_si_dagger, Q_si, G1_si_dagger, G1_si = init_corr_mats(n_coeff)

    for b in range(num_batches):
        vac_ = random.normal(key + b, (batch_size, 2, 2, Nx, Ny))

        # initialize the vacuum and output fields:
        # shape: batch_size, 2-for vac states for signal and idler, 2 - real and imag, Nx X Ny for beam size)
        Siganl_field = Field(Signal, PP_crystal, vac_[:, 0], batch_size)
        Idler_field = Field(Idler, PP_crystal, vac_[:, 1], batch_size)

        # current pump structure
        Pump.create_profile(coeffs_real + 1j * coeffs_imag)

        # Propagate through the crystal:
        crystal_prop(Pump, Siganl_field, Idler_field, PP_crystal)

        E_s_out = decompose(Siganl_field.E_out, Signal.hermite_arr).reshape(batch_size, max_mode2, max_mode1)
        E_i_out = decompose(Idler_field.E_out, Signal.hermite_arr).reshape(batch_size, max_mode2, max_mode1)
        E_i_vac = decompose(Idler_field.E_vac, Signal.hermite_arr).reshape(batch_size, max_mode2, max_mode1)

        # say there are no higher modes by normalizing the power
        E_s_out = fix_power1(E_s_out, Siganl_field.E_out, Signal, PP_crystal)
        E_i_out = fix_power1(E_i_out, Idler_field.E_out, Signal, PP_crystal)
        E_i_vac = fix_power1(E_i_vac, Idler_field.E_vac, Signal, PP_crystal)

        G1_ii, \
        G1_ss, \
        Q_si_dagger, \
        Q_si, \
        G1_si_dagger, \
        G1_si = \
            G2_calc_batch(E_s_out, E_i_out, E_i_vac, N,
                          G1_ii, G1_ss, Q_si_dagger, Q_si, G1_si_dagger, G1_si)


    G2 = ((
                  lax.psum(G1_ii, 'device') * lax.psum(G1_ss, 'device') +
                  lax.psum(Q_si_dagger, 'device') * lax.psum(Q_si, 'device') +
                  lax.psum(G1_si_dagger, 'device') * lax.psum(G1_si, 'device')
          ).real).reshape(max_mode2 * max_mode2, max_mode1 * max_mode1)
    return G2


def loss(coeffs, key, G2t):  # vac_ = vac_s, vac_i, G2t = P and G2 target correlation matrices
    coeffs_real, coeffs_imag = coeffs[:n_coeff], coeffs[n_coeff:2 * n_coeff]
    normalization = np.sqrt(np.sum(np.abs(coeffs_real) ** 2 + np.abs(coeffs_imag) ** 2))
    coeffs_real = coeffs_real / normalization
    coeffs_imag = coeffs_imag / normalization

    G2 = forward(np.concatenate((coeffs_real, coeffs_imag)), key)
    G2 = G2 / np.sum(np.abs(G2))

    if loss_type is 'l1':
        return l1_loss(G2, G2t)
    if loss_type is 'l2':
        return l2_loss(G2, G2t)
    if loss_type is 'kl':
        return kl_loss(G2, G2t, eps=1e-2)
    if loss_type is 'kl_sparse':
        return 0.5 * kl_loss(G2, G2t, eps=1e-2) + 0.5 * l1_loss(
            G2[..., onp.delete(onp.arange(n_coeff ** 2), [30, 40, 50])])
    if loss_type is 'kl_sparse_balanced':
        return 0.5 * kl_loss(G2, G2t, eps=1e-2) + \
               0.25 * l1_loss(G2[..., onp.delete(onp.arange(n_coeff ** 2), [30, 40, 50])]) + \
               0.25 * (
                       np.sum(np.abs(G2[..., 30] - G2[..., 40])) +
                       np.sum(np.abs(G2[..., 30] - G2[..., 50])) +
                       np.sum(np.abs(G2[..., 40] - G2[..., 50])))
    if loss_type is 'sparse_balanced':
        return 0.5 * l1_loss(G2[..., onp.delete(onp.arange(n_coeff ** 2), [30, 40, 50])]) + \
               0.5 * (
                       np.sum(np.abs(G2[..., 30] - G2[..., 40])) +
                       np.sum(np.abs(G2[..., 30] - G2[..., 50])) +
                       np.sum(np.abs(G2[..., 40] - G2[..., 50])))
    if loss_type is 'kl_l1':
        return 0.5 * kl_loss(G2, G2t, eps=1e-2) + 0.5 * l1_loss(G2, G2t)
    if loss_type is 'kl_l2':
        return 0.5 * kl_loss(G2, G2t, eps=1e-2) + 0.5 * l2_loss(G2, G2t)
    if loss_type is 'wass':
        return sinkhorn_loss(G2, G2t, n_coeff, eps=1e-3, max_iters=100, stop_thresh=None)
    else:
        raise Exception('Nonstandard loss choice')


@partial(pmap, axis_name='device')
def update(opt_state, i, key, G2t):
    coeffs = get_params(opt_state)
    batch_loss, grads = value_and_grad(loss)(coeffs, key, G2t)
    grads = np.array([lax.psum(dw, 'device') for dw in grads])
    return lax.pmean(batch_loss, 'device'), opt_update(i, grads, opt_state)


if learn_mode:
    print("--- training mode ---")
    # load target P, G2
    G2t = pmap(lambda x: np.load(Pt_path + targert_folder + 'G2.npy'))(np.arange(num_devices))
    coeffs_gt = np.load(Pt_path + targert_folder + 'HG_coeffs.npy')

    assert len(coeffs_real + 1j * coeffs_imag) == len(
        coeffs_gt), "HG parameters and its ground truth must contain same length"

    # Use optimizers to set optimizer initialization and update functions
    opt_init, opt_update, get_params = optimizers.adam(step_size, b1=0.9, b2=0.999, eps=1e-08)
    opt_state = opt_init(coeffs)
    obj_loss, best_obj_loss = [], None
    epochs_without_improvement = 0
    l1_HG_loss, l2_HG_loss = [], []
    for epoch in range(num_epochs):
        start_time_epoch = time.time()
        print("Epoch {}/{} is running".format(epoch, num_epochs))
        # seed vacuum samples
        keys = random.split(random.PRNGKey(seed), num_devices)
        idx = np.array([epoch]).repeat(num_devices)
        batch_loss, opt_state = update(opt_state, idx, keys, G2t)
        obj_loss_epoch = batch_loss[0].item()
        curr_coeffs = get_params(opt_state)
        obj_loss.append(obj_loss_epoch)
        epoch_time = time.time() - start_time_epoch
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        ''' print loss value'''

        coeffs_real, coeffs_imag = curr_coeffs[:, :n_coeff], curr_coeffs[:, n_coeff:2 * n_coeff]
        normalization = np.sqrt(np.sum(np.abs(coeffs_real) ** 2 + np.abs(coeffs_imag) ** 2, 1, keepdims=True))
        coeffs_real = coeffs_real / normalization
        coeffs_imag = coeffs_imag / normalization
        curr_coeffs = np.concatenate((coeffs_real, coeffs_imag), 1)

        coeffs_ = coeffs_real[0] + 1j * coeffs_imag[0]

        l1_HG_loss.append(np.sum(np.abs(coeffs_ - coeffs_gt)))
        l2_HG_loss.append(np.sum(np.abs(coeffs_ - coeffs_gt) ** 2))
        print("optimized LG coefficients: {}".format(coeffs_))
        print("Norm of LG coefficients: {}".format(np.sum((np.abs(coeffs_)) ** 2)))

        print("objective loss:{:0.6f}".format(obj_loss[epoch]))

        if best_obj_loss is None or obj_loss[epoch] < best_obj_loss:
            best_obj_loss = obj_loss[epoch]
            epochs_without_improvement = 0
            coeffs = curr_coeffs
            print(f'\n*** best objective loss updated at epoch {epoch}')
        else:
            epochs_without_improvement += 1
            print(f'\n*** Number of epochs without improvement {epochs_without_improvement}')

    print("--- training time: %s seconds ---" % (time.time() - start_time))

    curr_dir = stats_path + topic
    if os.path.isdir(curr_dir):
        for filename in os.listdir(curr_dir):
            os.remove(curr_dir + '/' + filename)
    else:
        os.makedirs(curr_dir)
    exp_details = open(curr_dir + '/' + "exp_details.txt", "w")
    exp_details.write(
        type_beam_from_pump_str(Pump.hermite_str, coeffs[0, :n_coeff] + 1j * coeffs[0, n_coeff:2 * n_coeff], coeffs_str,
                                coeffs_gt))
    exp_details.close()

    plt.plot(obj_loss, 'r')
    plt.title('loss(G2), loss type:{}'.format(loss_type))
    plt.ylabel('objective loss')
    plt.xlabel('#epoch')
    # plt.ylim(0, 1)
    if save_stats:
        plt.savefig(curr_dir + '/objective_loss')
    plt.show()
    plt.close()

    plt.plot(l1_HG_loss, 'r', label='L1')
    plt.plot(l2_HG_loss, 'b', label='L2')
    plt.title('HG coefficients')
    plt.ylabel('measure loss')
    plt.xlabel('#epoch')
    plt.legend()
    if save_stats:
        plt.savefig(curr_dir + '/HG_coeffs_losses')
    plt.show()
    plt.close()

# show last epoch result
if save_res or save_tgt or show_res:
    print("--- inference mode ---")
    ###########################################
    # Set dataset
    ##########################################
    # Build a dataset of pairs Ai_vac, As_vac

    # seed vacuum samples for each gpu
    keys = random.split(random.PRNGKey(seed), num_devices)

    G2 = pmap(forward, axis_name='device')(coeffs, keys)
    G2 = G2[0]
    coeffs = coeffs[0, :n_coeff] + 1j * coeffs[0, n_coeff:2 * n_coeff]

    if save_tgt:
        print("--- saving targets ---")
        curr_dir = Pt_path + topic
        if os.path.isdir(curr_dir):
            for filename in os.listdir(curr_dir):
                os.remove(curr_dir + '/' + filename)
        else:
            os.makedirs(curr_dir)

        G2_t_name = 'G2'
        # save normalized version
        np.save(curr_dir + '/' + G2_t_name, G2 / np.sum(np.abs(G2)))
        # save pump coeffs version
        np.save(curr_dir + '/HG_coeffs', coeffs)

        exp_details = open(curr_dir + '/' + "exp_details.txt", "w")
        if learn_mode:
            exp_details.write(type_beam_from_pump_str(Pump.hermite_str, coeffs, coeffs_str, coeffs_gt))
        else:
            exp_details.write(type_beam_from_pump_str(Pump.hermite_str, coeffs, coeffs_str))
        exp_details.close()

    if show_res or save_res:
        print("--- saving/plotting results ---")

        curr_dir = res_path + topic
        if os.path.isdir(curr_dir):
            for filename in os.listdir(curr_dir):
                os.remove(curr_dir + '/' + filename)
        else:
            os.makedirs(curr_dir)

        exp_details = open(curr_dir + '/' + "exp_details.txt", "w")
        if learn_mode:
            exp_details.write(type_beam_from_pump_str(Pump.hermite_str, coeffs, coeffs_str, coeffs_gt))
        else:
            exp_details.write(type_beam_from_pump_str(Pump.hermite_str, coeffs, coeffs_str))
        exp_details.close()

        ################
        # Plot G2 #
        ################
        # Unwrap G2 indices
        G2_unwrap_idx_str = 'G2_unwarp_idx/G2_unwrap_max_mode{}.npy'.format(max_mode1 * max_mode2)
        savetime_flag = 0
        if savetime_flag:
            if not os.path.exists(G2_unwrap_idx_str):
                G2_unwrapped_idx_np = onp.zeros((max_mode1, max_mode2, max_mode1, max_mode2), dtype=np.float32)
                print(np.shape(onp.arange(0, max_mode1 * max_mode2 * max_mode1 * max_mode2, dtype=np.float32).reshape(
                    max_mode1 * max_mode2, max_mode1 * max_mode2)))
                print(np.shape(G2_unwrapped_idx_np))
                G2_unwrapped_idx_np = \
                    unwrap_kron(G2_unwrapped_idx_np,
                                onp.arange(0, max_mode1 * max_mode2 * max_mode1 * max_mode2, dtype=np.float32).reshape(
                                    max_mode1 * max_mode1, max_mode2 * max_mode2),
                                max_mode1, max_mode2).reshape(max_mode1 * max_mode2 * max_mode1 * max_mode2)

                np.save(G2_unwrap_idx_str, G2_unwrapped_idx_np)

            else:
                G2_unwrapped_idx_np = np.load(G2_unwrap_idx_str)
            G2_unwrapped_idx = onp.ndarray.tolist(G2_unwrapped_idx_np)
            del G2_unwrapped_idx_np

            G2 = G2.reshape(max_mode1 * max_mode2 * max_mode1 * max_mode2)[G2_unwrapped_idx].reshape(max_mode1,
                                                                                                     max_mode2,
                                                                                                     max_mode1,
                                                                                                     max_mode2)
        else:
            G2_tensor = onp.zeros((max_mode2, max_mode1, max_mode2, max_mode1), dtype=np.float32)
            G2 = unwrap_kron(G2_tensor, G2, max_mode2, max_mode1)

        # Compute and plot reduced G2
        G2_reduced = G2[0, :, 0, :]
        G2_reduced = G2_reduced * tau / (g1_ii_normalization * g1_ss_normalization)

        # plot
        plt.imshow(G2_reduced)
        plt.title(r'$G^{(2)}$ (coincidences)')
        plt.xlabel(r'signal mode i')
        plt.ylabel(r'idle mode j')
        plt.colorbar()
        if save_res:
            plt.savefig(curr_dir + '/' + 'G2')
        if show_res:
            plt.show()

        # Save arrays
        np.save(curr_dir + '/' + 'PumpCoeffs_real.npy', coeffs.real)
        np.save(curr_dir + '/' + 'PumpCoeffs_imag.npy', coeffs.imag)
        np.save(curr_dir + '/' + 'G2.npy', G2)

print("\n--- Done: %s seconds ---" % (time.time() - start_time))
exit()
