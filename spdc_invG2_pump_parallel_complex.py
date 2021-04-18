from __future__ import print_function, division, absolute_import

import os
os.environ["JAX_ENABLE_X64"] = 'True'
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = 'platform'
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'

from loss_funcs_parallel_complex import l1_loss, kl_loss, sinkhorn_loss, l2_loss, bhattacharyya_loss, loss_func
from spdc_helper_parallel_complex import *
from spdc_funcs_parallel_complex import *
from physical_params_parallel_complex import *

# datetime object containing current date and time
now = datetime.now()
print("\n date and time =", now.strftime("%d/%m/%Y %H:%M:%S"))
start_time_initialization = time.time()

learn_mode = True  # learn/infer
save_stats = True
show_res = True  # display results 0/1
save_res = True  # save results
save_tgt = False  # save targets

res_path = 'results/'  # path to results folder
Pt_path = 'targets/'  # path to targets folder
stats_path = 'stats/'

seed = 1989

"Learning Hyperparameters"
loss_type = 'kl_sparse_balanced'  # l1:L1 Norm, kl:Kullback Leibler Divergence, wass: Wasserstein (Sinkhorn) Distance"
step_size = 0.05
num_epochs = 75
N           = 1200  # 100, 500, 1000  - number of total-iterations for learning (dataset size)
N_inference = 4000  # 100, 500, 1000  - number of total-iterations for inference (dataset size)

batch_device, num_devices = calc_and_asserts(N)

"Interaction Initialization"
# Structure arrays - initialize crystal and structure arrays
PP_crystal = Crystal(dx, dy, dz, MaxX, MaxY, MaxZ, d33)
M = len(PP_crystal.x)  # simulation size

n_coeff_projections, n_coeff_pump, max_mode1, max_mode2, \
max_radial_mode_crystal, max_radial_mode_pump, max_angular_mode_pump = projection_crystal_modes()

Pump = Beam(lam_pump, PP_crystal, Temperature, waist_pump, power_pump, projection_type, 2*max_angular_mode_pump + 1,
            max_radial_mode_pump)  # wavelength, crystal, tmperature,waist,power, maxmode
Signal = Beam(lam_signal, PP_crystal, Temperature, np.sqrt(2) * Pump.waist, power_signal, projection_type, max_mode1,
              max_mode2, z=0)
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
coeffs_real, coeffs_imag = HG_coeff_array(coeffs_str, n_coeff_pump)

# replicate parameters for gpus
coeffs = pmap(lambda x: np.concatenate((coeffs_real, coeffs_imag)))(np.arange(num_devices))

print("--- Pump waist {:.2f}um for Crystal length {}m ---\n".format(waist_pump * 1e6, MaxZ))
print("--- the LG coefficients initiated are: {} ---\n".format(coeffs_real + 1j * coeffs_imag))
print("--- initialization time: %s seconds ---" % (time.time() - start_time_initialization))
start_time = time.time()

topic = now.strftime("%_Y-%m-%d") + "_N_infer{}_Nx{}Ny{}_z{}_steps{}_#devices{}".format(
    N_inference, Nx, Ny, PP_crystal.MaxZ, len(PP_crystal.z), num_devices)

if learn_mode:
    topic += "_N_learn{}".format(N) + "_loss_{}".format(loss_type) + "_epochs{}".format(num_epochs) + "_complex"


def forward(coeffs, key):
    coeffs_real, coeffs_imag = coeffs[:n_coeff_pump], coeffs[n_coeff_pump:2 * n_coeff_pump]
    # batch_device iteration, 2-for vac states for signal and idler, 2 - real and imag, Nx X Ny for beam size)
    vac_ = random.normal(key, (batch_device, 2, 2, Nx, Ny))

    # initialize the vacuum and output fields:
    Siganl_field = Field(Signal, PP_crystal, vac_[:, 0], batch_device)
    Idler_field = Field(Idler, PP_crystal, vac_[:, 1], batch_device)

    # current pump structure
    Pump.create_profile(coeffs_real + 1j * coeffs_imag)

    # Propagate through the crystal:
    crystal_prop(Pump, Siganl_field, Idler_field, PP_crystal)

    ## Propagate generated fields back to the middle of the crystal
    DeltaZ = -MaxZ / 2

    E_s_out_prop = propagate(Siganl_field.E_out, PP_crystal.x, PP_crystal.y, Siganl_field.k, DeltaZ) * np.exp(
        -1j * Siganl_field.k * DeltaZ)
    E_i_out_prop = propagate(Idler_field.E_out, PP_crystal.x, PP_crystal.y, Idler_field.k, DeltaZ) * np.exp(
        -1j * Idler_field.k * DeltaZ)
    E_i_vac_prop = propagate(Idler_field.E_vac, PP_crystal.x, PP_crystal.y, Idler_field.k, DeltaZ) * np.exp(
        -1j * Idler_field.k * DeltaZ)

    E_s_out = decompose(E_s_out_prop, Signal.hermite_arr).reshape(batch_device, max_mode2, max_mode1)
    E_i_out = decompose(E_i_out_prop, Signal.hermite_arr).reshape(batch_device, max_mode2, max_mode1)
    E_i_vac = decompose(E_i_vac_prop, Signal.hermite_arr).reshape(batch_device, max_mode2, max_mode1)

    # say there are no higher modes by normalizing the power
    E_s_out = fix_power1(E_s_out, Siganl_field.E_out, Signal, PP_crystal)
    E_i_out = fix_power1(E_i_out, Idler_field.E_out, Signal, PP_crystal)
    E_i_vac = fix_power1(E_i_vac, Idler_field.E_vac, Signal, PP_crystal)

    G2 = G2_calc(E_s_out, E_i_out, E_i_vac, N).reshape(max_mode2 * max_mode2, max_mode1 * max_mode1)
    return G2


def loss(coeffs, key, G2t):  # vac_ = vac_s, vac_i, G2t = P and G2 target correlation matrices
    coeffs_real, coeffs_imag = coeffs[:n_coeff_pump], coeffs[n_coeff_pump:2 * n_coeff_pump]
    normalization = np.sqrt(np.sum(np.abs(coeffs_real) ** 2 + np.abs(coeffs_imag) ** 2))
    coeffs_real = coeffs_real / normalization
    coeffs_imag = coeffs_imag / normalization

    G2 = forward(np.concatenate((coeffs_real, coeffs_imag)), key)
    G2 = G2 / np.sum(np.abs(G2))

    coeffs_ = coeffs_real + 1j * coeffs_imag
    if loss_type is 'l1':
        return l1_loss(G2, G2t)
    if loss_type is 'l2':
        return l2_loss(G2, G2t)
    if loss_type is 'kl':
        return kl_loss(G2, G2t, eps=1e-2)
    if loss_type is 'bhattacharyya':
        return bhattacharyya_loss(G2, G2t)
    if loss_type is 'kl_sparse':
        return 0.5 * kl_loss(G2, G2t, eps=1e-2) + 0.5 * l1_loss(
            G2[..., onp.delete(onp.arange(n_coeff_projections ** 2), [30, 40, 50])])
    if loss_type is 'kl_sparse_balanced':
        return kl_loss(G2, G2t, eps=1e-2) + \
               4 * l1_loss(G2[..., onp.delete(onp.arange(n_coeff_projections ** 2), [30, 40, 50])]) + \
               1e2 * (
                       np.sum(np.abs(G2[..., 30] - G2[..., 40])) +
                       np.sum(np.abs(G2[..., 30] - G2[..., 50])) +
                       np.sum(np.abs(G2[..., 40] - G2[..., 50]))) + \
               1e-4 * np.sum(np.abs(coeffs_)) + \
               10e3 * np.sum(np.abs(coeffs_[np.array([1, 3, 6, 8, 11, 13])])) + \
               10 * np.sum(np.abs(G2[..., np.array([8, 72, 34, 66, 14, 46, 42, 58, 22, 38])]))
    if loss_type is 'sparse_balanced':
        return 0.5 * l1_loss(G2[..., onp.delete(onp.arange(n_coeff_projections ** 2), [30, 40, 50])]) + \
               0.5 * (
                       np.sum(np.abs(G2[..., 30] - G2[..., 40])) +
                       np.sum(np.abs(G2[..., 30] - G2[..., 50])) +
                       np.sum(np.abs(G2[..., 40] - G2[..., 50])))
    if loss_type is 'kl_l1':
        return 0.5 * kl_loss(G2, G2t, eps=1e-2) + 0.5 * l1_loss(G2, G2t)
    if loss_type is 'kl_l2':
        return 0.5 * kl_loss(G2, G2t, eps=1e-2) + 0.5 * l2_loss(G2, G2t)
    if loss_type is 'wass':
        return sinkhorn_loss(G2, G2t, n_coeff_projections, eps=1e-3, max_iters=100, stop_thresh=None)
    else:
        raise Exception('Nonstandard loss choice')


@partial(pmap, axis_name='device')
def update(opt_state, i, key, G2t):
    coeffs = get_params(opt_state)
    batch_loss, grads = value_and_grad(loss)(coeffs, key, G2t)
    grads = np.array([lax.psum(dw, 'device') for dw in grads])
    return lax.pmean(batch_loss, 'device'), opt_update(i, grads, opt_state)


@partial(pmap, axis_name='device')
def validate(opt_state, key, G2t):
    coeffs = get_params(opt_state)
    batch_loss, grads = value_and_grad(loss)(coeffs, key, G2t)
    return lax.pmean(batch_loss, 'device')


if learn_mode:
    print("--- training mode ---")
    # load target P, G2
    G2t = pmap(lambda x: np.load(Pt_path + targert_folder + 'G2.npy'))(np.arange(num_devices))

    # loss_func(G2t=G2t[0, 0],
    #           n_coeff_projections=n_coeff_projections,
    #           loss_type=loss_type)
              # abuse_pump_coeffs_idx=[1, 3, 6, 8, 11, 13])

    # Use optimizers to set optimizer initialization and update functions
    opt_init, opt_update, get_params = optimizers.adam(step_size, b1=0.9, b2=0.999, eps=1e-08)
    opt_state = opt_init(coeffs)
    obj_loss_trn, obj_loss_vld, best_obj_loss = [], [], None
    epochs_without_improvement = 0
    validation_flag = np.array([1]).repeat(num_devices)
    for epoch in range(num_epochs):
        start_time_epoch = time.time()
        print("Epoch {}/{} is running".format(epoch, num_epochs))
        # seed vacuum samples
        keys = random.split(random.PRNGKey(seed + epoch), num_devices)
        idx = np.array([epoch]).repeat(num_devices)
        batch_loss, opt_state = update(opt_state, idx, keys, G2t)
        curr_coeffs = get_params(opt_state)
        obj_loss_trn.append(batch_loss[0].item())

        # validate training parameters
        keys = random.split(random.PRNGKey(seed + num_epochs + epoch), num_devices)
        batch_loss_vld = validate(opt_state, keys, G2t)
        obj_loss_vld.append(batch_loss_vld[0].item())

        epoch_time = time.time() - start_time_epoch
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        ''' print loss value'''

        coeffs_real, coeffs_imag = curr_coeffs[:, :n_coeff_pump], curr_coeffs[:, n_coeff_pump:2 * n_coeff_pump]
        normalization = np.sqrt(np.sum(np.abs(coeffs_real) ** 2 + np.abs(coeffs_imag) ** 2, 1, keepdims=True))
        coeffs_real = coeffs_real / normalization
        coeffs_imag = coeffs_imag / normalization
        curr_coeffs = np.concatenate((coeffs_real, coeffs_imag), 1)

        coeffs_ = coeffs_real[0] + 1j * coeffs_imag[0]

        print("optimized LG coefficients: {}".format(coeffs_))
        print("Norm of LG coefficients: {}".format(np.sum((np.abs(coeffs_)) ** 2)))

        print("training   objective loss:{:0.6f}".format(obj_loss_trn[epoch]))
        print("validation objective loss:{:0.6f}".format(obj_loss_vld[epoch]))

        if best_obj_loss is None or obj_loss_vld[epoch] < best_obj_loss:
            best_obj_loss = obj_loss_vld[epoch]
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
        make_beam_from_HG_str(Pump.hermite_str,
                              coeffs[0, :n_coeff_pump] + 1j * coeffs[0, n_coeff_pump:2 * n_coeff_pump], coeffs_str))
    exp_details.close()

    plt.plot(obj_loss_trn, 'r', label='training')
    plt.plot(obj_loss_vld, 'b', label='validation')
    plt.title('loss(G2), loss type:{}'.format(loss_type))
    plt.ylabel('objective loss')
    plt.xlabel('#epoch')
    plt.legend()
    if save_stats:
        plt.savefig(curr_dir + '/objective_loss')
    plt.show()
    plt.close()


# show last epoch result
if save_res or save_tgt or show_res:
    print("--- inference mode ---")
    N          = N_inference  # number of total-iterations (dataset size)

    batch_device, num_devices = calc_and_asserts(N)
    ###########################################
    # Set dataset
    ##########################################
    # Build a dataset of pairs Ai_vac, As_vac

    # seed vacuum samples for each gpu
    keys = random.split(random.PRNGKey(seed * 1986), num_devices)

    G2 = pmap(forward, axis_name='device')(coeffs, keys)
    G2 = G2[0]
    coeffs = coeffs[0, :n_coeff_pump] + 1j * coeffs[0, n_coeff_pump:2 * n_coeff_pump]

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
            exp_details.write(make_beam_from_HG_str(Pump.hermite_str, coeffs, coeffs_str))
        else:
            exp_details.write(make_beam_from_HG_str(Pump.hermite_str, coeffs, coeffs_str))
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
            exp_details.write(make_beam_from_HG_str(Pump.hermite_str, coeffs, coeffs_str))
        else:
            exp_details.write(make_beam_from_HG_str(Pump.hermite_str, coeffs, coeffs_str))
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

        # plot G2 2d
        plt.imshow(G2_reduced)
        plt.title(r'$G^{(2)}$ (coincidences)')
        plt.xlabel(r'signal mode i')
        plt.ylabel(r'idle mode j')
        plt.colorbar()
        plt.xticks(np.arange(n_coeff_projections), np.arange(n_coeff_projections) - int(n_coeff_projections/2))
        plt.yticks(np.arange(n_coeff_projections), np.arange(n_coeff_projections) - int(n_coeff_projections/2))
        if save_res:
            plt.savefig(curr_dir + '/' + 'G2')
        if show_res:
            plt.show()


        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xpos, ypos = np.meshgrid(np.arange(n_coeff_projections) - int(n_coeff_projections / 2),
                                 np.arange(n_coeff_projections) - int(n_coeff_projections / 2))
        xpos = xpos.flatten('F')
        ypos = ypos.flatten('F')
        zpos = np.zeros_like(xpos)
        dx = 0.8 * np.ones_like(zpos)
        dy = dx.copy()
        dz = G2_reduced.flatten()
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True)
        ax.set_title(r'$G^{(2)}$ (coincidences)')
        ax.set_xlabel(r'signal mode i')
        ax.set_ylabel(r'signal mode i')
        # ax.set_xticks(np.arange(n_coeff_projections), np.arange(n_coeff_projections) - int(n_coeff_projections / 2))
        # ax.set_yticks(np.arange(n_coeff_projections), np.arange(n_coeff_projections) - int(n_coeff_projections / 2))
        if save_res:
            plt.savefig(curr_dir + '/' + 'G2_3d')
        if show_res:
            plt.show()

        # Save arrays
        np.save(curr_dir + '/' + 'PumpCoeffs_real.npy', coeffs.real)
        np.save(curr_dir + '/' + 'PumpCoeffs_imag.npy', coeffs.imag)
        np.save(curr_dir + '/' + 'G2.npy', G2)

print("\n--- Done: %s seconds ---" % (time.time() - start_time))
exit()
