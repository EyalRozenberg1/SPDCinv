import os

from learning_params_parallel_complex import *
from loss_funcs_parallel_complex import l1_loss, kl_loss, l2_loss, bhattacharyya_loss
from spdc_helper_parallel_complex import *
from spdc_funcs_parallel_complex import *
from physical_params_parallel_complex import *

batch_device, num_devices = calc_and_asserts(N)

# datetime object containing current date and time
now = datetime.now()
print("date and time =", now.strftime("%d/%m/%Y %H:%M:%S"))
start_time_initialization = time.time()


"Interaction Initialization"
# Structure arrays - initialize crystal and structure arrays
PP_crystal = Crystal(dx, dy, dz, MaxX, MaxY, MaxZ, d33, learn_crystal_coeffs=learn_crystal_coeffs)

max_mode1, max_mode2,\
max_mode1_pump, max_mode2_pump,\
max_mode1_crystal, max_mode2_crystal = projection_crystal_modes()

n_coeff_projections = max_mode1 * max_mode2  # Total number of projection modes
n_coeff_pump        = max_mode1_pump * max_mode2_pump  # Total number of pump modes
n_coeff_crystal     = max_mode1_crystal * max_mode2_crystal  # Total number of crystal modes


# Initialize pump and crystal coefficients
pump_coeffs_real, pump_coeffs_imag, waist_pump    = Pump_coeff_array(coeffs_str, n_coeff_pump)
crystal_coeffs_real, crystal_coeffs_imag, r_scale = Crystal_coeff_array(crystal_str, n_coeff_crystal)


Signal = Beam(lam_signal, PP_crystal, 'y', Temperature,
              waist_pump_proj, power_signal, projection_type,
              max_mode1, max_mode2, z=0)

Idler  = Beam(SFG_idler_wavelength(lam_pump, lam_signal), PP_crystal, 'z', Temperature,
              waist_pump_proj, power_signal, projection_type)

Pump = Beam(lam_pump, PP_crystal, 'y', Temperature,
            waist_pump, power_pump, projection_type,
            max_mode1_pump, max_mode2_pump,
            n_coeff_pump=n_coeff_pump, learn_pump_coeffs=learn_pump_coeffs, learn_pump_waists=learn_pump_waists)


# save initial pump coefficients
Pump.pump_coeffs = pump_coeffs_real + 1j * pump_coeffs_imag

# Settings for Fourier-Taylor / Fourier-Bessel / Laguerre-Gauss / Hermite-Gauss crystal hologram
Poling = Poling_profile(r_scale, PP_crystal.x, PP_crystal.y,
                        max_mode1_crystal, max_mode2_crystal,
                        crystal_basis, Signal.lam, Signal.n, learn_crystal_waists)


# phase mismatch
delta_k = Pump.k - Signal.k - Idler.k
PP_crystal.poling_period = dk_offset * delta_k


# Interaction Parameters
Nx     = len(PP_crystal.x)
Ny     = len(PP_crystal.y)
DeltaZ = - MaxZ / 2


# normalization factor
g1_ss_normalization = G1_Normalization(Signal.w)
g1_ii_normalization = G1_Normalization(Idler.w)

# replicate parameters for gpus
coeffs = pmap(lambda x: np.concatenate((pump_coeffs_real, pump_coeffs_imag,
                                        crystal_coeffs_real, crystal_coeffs_imag,
                                        waist_pump,
                                        r_scale)
                                       ))(np.arange(num_devices))

print("--- the pump coefficients initiated are: {} ---\n".format(pump_coeffs_real + 1j * pump_coeffs_imag))
print("--- the crystal coefficients initiated are: {} ---\n".format(crystal_coeffs_real + 1j * crystal_coeffs_imag))
print("--- Pump waist initiated are {}um\n".format(waist_pump * 10))
print("--- r_scale is initiated are {}um\n--- for Crystal length {}m\n".format(r_scale * 10, MaxZ))

print("--- initialization time: %s seconds ---" % (time.time() - start_time_initialization))

start_time = time.time()

topic = f'{now.strftime("%_Y-%m-%d")}' \
        f'_Ninfer{N_inference}' \
        f'_Nx{Nx}Ny{Ny}' \
        f'_z{PP_crystal.MaxZ}' \
        f'_steps{len(PP_crystal.z)}' \
        f'_pump_basis_{pump_basis}' \
        f'_crystal_basis{crystal_basis}' \
        f'_gpus{num_devices}'

if learn_mode:
    topic += f'_Nlearn{N}' \
             f'_loss_{loss_type}' \
             f'_epochs{num_epochs}' \
             f'_lr{step_size}' \
             f'_optimizer_{optimizer}'


def forward(coeffs, key):
    pump_coeffs_real    = coeffs[:n_coeff_pump]
    pump_coeffs_imag    = coeffs[n_coeff_pump:2 * n_coeff_pump]
    crystal_coeffs_real = coeffs[2 * n_coeff_pump: 2 * n_coeff_pump + n_coeff_crystal]
    crystal_coeffs_imag = coeffs[2 * n_coeff_pump + n_coeff_crystal: 2 * n_coeff_pump + 2 * n_coeff_crystal]
    w0                  = coeffs[2 * n_coeff_pump + 2 * n_coeff_crystal: 2 * n_coeff_pump + 2 * n_coeff_crystal + n_coeff_pump]
    r_scale             = coeffs[2 * n_coeff_pump + 2 * n_coeff_crystal + n_coeff_pump:]

    # batch_device iteration, 2-for vac states for signal and idler, 2 - real and imag, Nx X Ny for beam size)
    vac_ = random.normal(key, (batch_device, 2, 2, Nx, Ny))

    # initialize the vacuum and output fields:
    Siganl_field = Field(Signal, PP_crystal, vac_[:, 0], batch_device)
    Idler_field  = Field(Idler, PP_crystal, vac_[:, 1], batch_device)

    # current pump structure
    Pump.create_profile(pump_coeffs_real + 1j * pump_coeffs_imag, w0=w0)

    # current crystal structure
    Poling.create_profile(crystal_coeffs_real + 1j * crystal_coeffs_imag, r_scale)

    # Propagate through the crystal
    crystal_prop(Pump, Siganl_field, Idler_field, PP_crystal, Poling.crystal_profile)

    # Propagate generated fields back to the middle of the crystal
    E_s_out_prop = propagate(Siganl_field.E_out, PP_crystal.x, PP_crystal.y, Siganl_field.k, DeltaZ) * np.exp(
        -1j * Siganl_field.k * DeltaZ)
    E_i_out_prop = propagate(Idler_field.E_out, PP_crystal.x, PP_crystal.y, Idler_field.k, DeltaZ) * np.exp(
        -1j * Idler_field.k * DeltaZ)
    E_i_vac_prop = propagate(Idler_field.E_vac, PP_crystal.x, PP_crystal.y, Idler_field.k, DeltaZ) * np.exp(
        -1j * Idler_field.k * DeltaZ)

    E_s_out = decompose(E_s_out_prop, Signal.pump_basis_arr).reshape(batch_device, max_mode2, max_mode1)
    E_i_out = decompose(E_i_out_prop, Signal.pump_basis_arr).reshape(batch_device, max_mode2, max_mode1)
    E_i_vac = decompose(E_i_vac_prop, Signal.pump_basis_arr).reshape(batch_device, max_mode2, max_mode1)

    # say there are no higher modes by normalizing the power
    E_s_out = fix_power1(E_s_out, Siganl_field.E_out, Signal, PP_crystal)
    E_i_out = fix_power1(E_i_out, Idler_field.E_out, Signal, PP_crystal)
    E_i_vac = fix_power1(E_i_vac, Idler_field.E_vac, Signal, PP_crystal)

    G2 = G2_calc(E_s_out, E_i_out, E_i_vac, N).reshape(max_mode2 ** 2, max_mode1 ** 2)
    return G2


def loss(coeffs, key, G2t):
    pump_coeffs_real    = coeffs[:n_coeff_pump]
    pump_coeffs_imag    = coeffs[n_coeff_pump:2 * n_coeff_pump]
    crystal_coeffs_real = coeffs[2 * n_coeff_pump: 2 * n_coeff_pump + n_coeff_crystal]
    crystal_coeffs_imag = coeffs[2 * n_coeff_pump + n_coeff_crystal: 2 * n_coeff_pump + 2 * n_coeff_crystal]
    w0                  = coeffs[2 * n_coeff_pump + 2 * n_coeff_crystal: 2 * n_coeff_pump + 2 * n_coeff_crystal + n_coeff_pump]
    r_scale             = coeffs[2 * n_coeff_pump + 2 * n_coeff_crystal + n_coeff_pump:]

    normalization = np.sqrt(np.sum(np.abs(pump_coeffs_real) ** 2 + np.abs(pump_coeffs_imag) ** 2))
    pump_coeffs_real = pump_coeffs_real / normalization
    pump_coeffs_imag = pump_coeffs_imag / normalization

    normalization = np.sqrt(np.sum(np.abs(crystal_coeffs_real) ** 2 + np.abs(crystal_coeffs_imag) ** 2))
    crystal_coeffs_real = crystal_coeffs_real / normalization
    crystal_coeffs_imag = crystal_coeffs_imag / normalization

    G2 = forward(np.concatenate(
        (pump_coeffs_real, pump_coeffs_imag, crystal_coeffs_real, crystal_coeffs_imag, w0, r_scale)), key)

    G2 = G2 / np.sum(np.abs(G2))

    coeffs_ = pump_coeffs_real + 1j * pump_coeffs_imag
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
               1e3 * (
                       np.sum(np.abs(G2[..., 30] - G2[..., 40])) +
                       np.sum(np.abs(G2[..., 30] - G2[..., 50])) +
                       np.sum(np.abs(G2[..., 40] - G2[..., 50]))) + \
               1e-4 * np.sum(np.abs(coeffs_)) + \
               10e3 * np.sum(np.abs(coeffs_[np.array([1, 3, 6, 8, 11, 13])])) + \
               10 * np.sum(np.abs(G2[..., np.array([8, 72, 34, 66, 14, 46, 42, 58, 22, 38])]))
    if loss_type is 'sparse_balanced':
        return 1. * l1_loss(G2[..., onp.delete(onp.arange(n_coeff_projections ** 2), [30, 40, 50])]) + \
               0.5 * (
                       np.sum(np.abs(G2[..., 30] - G2[..., 40])) +
                       np.sum(np.abs(G2[..., 30] - G2[..., 50])) +
                       np.sum(np.abs(G2[..., 40] - G2[..., 50])))
    if loss_type is 'kl_l1':
        return 0.5 * kl_loss(G2, G2t, eps=1e-2) + 0.5 * l1_loss(G2, G2t)
    if loss_type is 'kl_l2':
        return 0.5 * kl_loss(G2, G2t, eps=1e-2) + 0.5 * l2_loss(G2, G2t)
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
    # load target G2
    G2t = pmap(lambda x: np.load(Pt_path + targert_folder + 'G2.npy'))(np.arange(num_devices))

    if exp_decay_lr:
        step_schedule = optimizers.exponential_decay(step_size=step_size, decay_steps=decay_steps,
                                                     decay_rate=decay_rate)
    else:
        step_schedule = step_size

    # Use optimizers to set optimizer initialization and update functions
    if optimizer == 'adam':
        opt_init, opt_update, get_params = optimizers.adam(step_schedule, b1=0.9, b2=0.999, eps=1e-08)
    elif optimizer == 'sgd':
        opt_init, opt_update, get_params = optimizers.sgd(step_schedule)
    elif optimizer == 'adagrad':
        opt_init, opt_update, get_params = optimizers.adagrad(step_schedule)
    elif optimizer == 'adamax':
        opt_init, opt_update, get_params = optimizers.adamax(step_schedule, b1=0.9, b2=0.999, eps=1e-08)
    elif optimizer == 'momentum':
        opt_init, opt_update, get_params = optimizers.momentum(step_schedule, mass=1e-02)
    elif optimizer == 'nesterov':
        opt_init, opt_update, get_params = optimizers.nesterov(step_schedule, mass=1e-02)
    elif optimizer == 'rmsprop':
        opt_init, opt_update, get_params = optimizers.rmsprop(step_schedule, gamma=0.9, eps=1e-08)
    elif optimizer == 'rmsprop_momentum':
        opt_init, opt_update, get_params = optimizers.rmsprop_momentum(step_schedule, gamma=0.9, eps=1e-08,
                                                                       momentum=0.9)
    else:
        raise Exception('Nonstandard Optimizer choice')

    opt_state = opt_init(coeffs)

    obj_loss_trn, obj_loss_vld, best_obj_loss = [], [], None
    epochs_without_improvement = 0
    for epoch in range(num_epochs):
        start_time_epoch = time.time()
        print("Epoch {}/{} is running".format(epoch, num_epochs))
        # seed vacuum samples for training
        keys = random.split(random.PRNGKey(seed + epoch), num_devices)
        idx = np.array([epoch]).repeat(num_devices)
        batch_loss, opt_state = update(opt_state, idx, keys, G2t)
        curr_coeffs = get_params(opt_state)
        obj_loss_trn.append(batch_loss[0].item())

        "validate training parameters"
        # seed vacuum samples for validation
        keys = random.split(random.PRNGKey(seed + num_epochs + epoch), num_devices)
        batch_loss_vld = validate(opt_state, keys, G2t)
        obj_loss_vld.append(batch_loss_vld[0].item())

        epoch_time = time.time() - start_time_epoch
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        ''' print loss value'''

        pump_coeffs_real    = curr_coeffs[:, :n_coeff_pump]
        pump_coeffs_imag    = curr_coeffs[:, n_coeff_pump:2 * n_coeff_pump]
        crystal_coeffs_real = curr_coeffs[:, 2 * n_coeff_pump: 2 * n_coeff_pump + n_coeff_crystal]
        crystal_coeffs_imag = curr_coeffs[:, 2 * n_coeff_pump + n_coeff_crystal: 2 * n_coeff_pump + 2 * n_coeff_crystal]
        w0                  = curr_coeffs[:, 2 * n_coeff_pump + 2 * n_coeff_crystal: 2 * n_coeff_pump + 2 * n_coeff_crystal + n_coeff_pump]
        r_scale             = curr_coeffs[:, 2 * n_coeff_pump + 2 * n_coeff_crystal + n_coeff_pump:]


        normalization = np.sqrt(np.sum(np.abs(pump_coeffs_real) ** 2 +
                                       np.abs(pump_coeffs_imag) ** 2, 1, keepdims=True))
        pump_coeffs_real = pump_coeffs_real / normalization
        pump_coeffs_imag = pump_coeffs_imag / normalization

        normalization = np.sqrt(np.sum(np.abs(crystal_coeffs_real) ** 2 +
                                       np.abs(crystal_coeffs_imag) ** 2, 1, keepdims=True))
        crystal_coeffs_real = crystal_coeffs_real / normalization
        crystal_coeffs_imag = crystal_coeffs_imag / normalization

        curr_coeffs = np.concatenate(
            (pump_coeffs_real, pump_coeffs_imag, crystal_coeffs_real, crystal_coeffs_imag, w0, r_scale), 1)

        coeffs_         = pump_coeffs_real[0] + 1j * pump_coeffs_imag[0]
        crystal_coeffs_ = crystal_coeffs_real[0] + 1j * crystal_coeffs_imag[0]

        print("optimized pump coefficients: {}".format(coeffs_))
        print("Norm of pump coefficients: {}\n".format(np.sum((np.abs(coeffs_)) ** 2)))
        print("optimized crystal coefficients: {}".format(crystal_coeffs_))
        print("Norm of crystal coefficients: {}\n".format(np.sum((np.abs(crystal_coeffs_)) ** 2)))
        print("optimized waist coefficients {}um\n".format(w0[0] * 10))
        print("optimized r_scale coefficients {}um\n".format(r_scale[0] * 10))
        print("training   objective loss:{:0.6f}".format(obj_loss_trn[epoch]))
        print("validation objective loss:{:0.6f}".format(obj_loss_vld[epoch]))

        if best_obj_loss is None or obj_loss_vld[epoch] < best_obj_loss and obj_loss_vld[epoch] == obj_loss_vld[epoch]:
            best_obj_loss = obj_loss_vld[epoch]
            epochs_without_improvement = 0
            if keep_best:
                coeffs = curr_coeffs
                print(f'\n*** best objective loss updated at epoch {epoch}')
        else:
            epochs_without_improvement += 1
            print(f'\n*** Number of epochs without improvement {epochs_without_improvement}')

        if not keep_best:
            coeffs = curr_coeffs

    print("--- training time: %s seconds ---" % (time.time() - start_time))


    curr_dir = stats_path + topic
    if save_stats:
        if os.path.isdir(curr_dir):
            for filename in os.listdir(curr_dir):
                os.remove(curr_dir + '/' + filename)
        else:
            os.makedirs(curr_dir)
        exp_details = open(curr_dir + '/' + "exp_details.txt", "w")
        exp_details.write(
            type_beam_from_pump_str(Pump.type, Pump.pump_basis_str,
                                  coeffs[0, :n_coeff_pump] + 1j * coeffs[0, n_coeff_pump:2 * n_coeff_pump], coeffs_str))

        exp_details.write(
            type_poling_from_crystal_str(
                crystal_basis,
                coeffs[0, 2 * n_coeff_pump: 2 * n_coeff_pump + n_coeff_crystal] +
                1j * coeffs[0, 2 * n_coeff_pump + n_coeff_crystal: 2 * n_coeff_pump + 2 * n_coeff_crystal], crystal_str))

        exp_details.write(
            type_waists_from_pump_str(Pump.type,
                                      Pump.pump_basis_str,
                                      coeffs[0, 2 * n_coeff_pump + 2 * n_coeff_crystal:
                                                2 * n_coeff_pump + 2 * n_coeff_crystal + n_coeff_pump] * 10))

        exp_details.write(
            type_waists_from_crystal_str(
                crystal_basis,
                coeffs[0, 2 * n_coeff_pump + 2 * n_coeff_crystal + n_coeff_pump:] * 10))
        exp_details.close()

    plt.plot(obj_loss_trn, 'r', label='training')
    plt.plot(obj_loss_vld, 'b', label='validation')
    plt.title(f'{loss_type}(G2), optimizer:{optimizer}')
    plt.ylabel('objective loss')
    plt.xlabel('#epoch')
    plt.ylim(0.2, 1)
    plt.legend()
    if save_stats:
        plt.savefig(curr_dir + '/objective_loss')
    plt.show()
    plt.close()

# show last epoch result
if save_res or save_tgt or show_res:
    print("--- inference mode ---")
    N = N_inference  # number of total-iterations (dataset size)

    batch_device, num_devices = calc_and_asserts(N)
    ###########################################
    # Set dataset
    ##########################################
    # Build a dataset of pairs Ai_vac, As_vac

    # seed vacuum samples for each gpu
    keys = random.split(random.PRNGKey(seed * 1986), num_devices)

    G2 = pmap(forward, axis_name='device')(coeffs, keys)
    G2 = G2[0]

    crystal_coeffs = coeffs[0, 2 * n_coeff_pump: 2 * n_coeff_pump + n_coeff_crystal] + 1j * \
                     coeffs[0, 2 * n_coeff_pump + n_coeff_crystal: 2 * n_coeff_pump + 2 * n_coeff_crystal]

    waist_pump     = coeffs[0, 2 * n_coeff_pump + 2 * n_coeff_crystal: 2 * n_coeff_pump + 2 * n_coeff_crystal + n_coeff_pump] * 10
    r_scale        = coeffs[0, 2 * n_coeff_pump + 2 * n_coeff_crystal + n_coeff_pump:] * 10
    coeffs         = coeffs[0, :n_coeff_pump] + 1j * coeffs[0, n_coeff_pump:2 * n_coeff_pump]

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
        np.save(curr_dir + '/' + 'PumpCoeffs_real.npy', coeffs.real)
        np.save(curr_dir + '/' + 'PumpCoeffs_imag.npy', coeffs.imag)
        np.save(curr_dir + '/' + 'CrystalCoeffs_real.npy', crystal_coeffs.real)
        np.save(curr_dir + '/' + 'CrystalCoeffs_imag.npy', crystal_coeffs.imag)
        np.save(curr_dir + '/' + 'PumpWaistCoeffs.npy', waist_pump)
        np.save(curr_dir + '/' + 'CrystalWaistCoeffs.npy', r_scale)

        exp_details = open(curr_dir + '/' + "exp_details.txt", "w")
        exp_details.write(type_beam_from_pump_str(Pump.type, Pump.pump_basis_str, coeffs, coeffs_str))
        exp_details.write(type_poling_from_crystal_str(crystal_basis, crystal_coeffs, crystal_str))
        exp_details.write(type_waists_from_pump_str(Pump.type, Pump.pump_basis_str, waist_pump))
        exp_details.write(type_waists_from_crystal_str(crystal_basis, r_scale))
        exp_details.close()

    if show_res or save_res:
        print("--- saving/plotting results ---")

        curr_dir = res_path + topic
        if save_res:
            if os.path.isdir(curr_dir):
                for filename in os.listdir(curr_dir):
                    os.remove(curr_dir + '/' + filename)
            else:
                os.makedirs(curr_dir)

            exp_details = open(curr_dir + '/' + "exp_details.txt", "w")
            exp_details.write(type_beam_from_pump_str(Pump.type, Pump.pump_basis_str, coeffs, coeffs_str))
            exp_details.write(type_poling_from_crystal_str(crystal_basis, crystal_coeffs, crystal_str))
            exp_details.write(type_waists_from_pump_str(Pump.type, Pump.pump_basis_str, waist_pump))
            exp_details.write(type_waists_from_crystal_str(crystal_basis, r_scale))
            exp_details.close()

        ################
        # Plot G2 #
        ################
        # Unwrap G2 indices
        G2_unwrap_idx_str = 'G2_unwarp_idx/G2_unwrap_max_mode{}.npy'.format(n_coeff_projections)
        savetime_flag = 0
        if savetime_flag:
            if not os.path.exists(G2_unwrap_idx_str):
                G2_unwrapped_idx_np = onp.zeros((max_mode1, max_mode2, max_mode1, max_mode2), dtype=np.float32)
                print(np.shape(onp.arange(0, n_coeff_projections * n_coeff_projections, dtype=np.float32).reshape(
                    n_coeff_projections, n_coeff_projections)))
                print(np.shape(G2_unwrapped_idx_np))
                G2_unwrapped_idx_np = \
                    unwrap_kron(G2_unwrapped_idx_np,
                                onp.arange(0, n_coeff_projections * n_coeff_projections, dtype=np.float32).reshape(
                                    max_mode1 ** 2, max_mode2 ** 2),
                                max_mode1, max_mode2).reshape(n_coeff_projections * n_coeff_projections)

                np.save(G2_unwrap_idx_str, G2_unwrapped_idx_np)

            else:
                G2_unwrapped_idx_np = np.load(G2_unwrap_idx_str)
            G2_unwrapped_idx = onp.ndarray.tolist(G2_unwrapped_idx_np)
            del G2_unwrapped_idx_np

            G2 = G2.reshape(
                            n_coeff_projections * n_coeff_projections
                            )[G2_unwrapped_idx].reshape(max_mode1, max_mode2, max_mode1, max_mode2)

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
        plt.xticks(np.arange(n_coeff_projections), np.arange(n_coeff_projections) - int(n_coeff_projections / 2))
        plt.yticks(np.arange(n_coeff_projections), np.arange(n_coeff_projections) - int(n_coeff_projections / 2))
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
        if save_res:
            np.save(curr_dir + '/' + 'PumpCoeffs_real.npy', coeffs.real)
            np.save(curr_dir + '/' + 'PumpCoeffs_imag.npy', coeffs.imag)
            np.save(curr_dir + '/' + 'CrystalCoeffs_real.npy', crystal_coeffs.real)
            np.save(curr_dir + '/' + 'CrystalCoeffs_imag.npy', crystal_coeffs.imag)
            np.save(curr_dir + '/' + 'PumpWaistCoeffs.npy', waist_pump)
            np.save(curr_dir + '/' + 'CrystalWaistCoeffs.npy', r_scale)
            np.save(curr_dir + '/' + 'G2.npy', G2)

print("\n--- Done: %s seconds ---" % (time.time() - start_time))
exit()
