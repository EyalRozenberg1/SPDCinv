from __future__ import print_function, division, absolute_import
from loss_funcs_parallel_complex import l1_loss, kl_loss, sinkhorn_loss, l2_loss
from spdc_helper import *
from spdc_funcs_parallel_complex import *
from physical_params import *

JAX_ENABLE_X64 = False
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = 'platform'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

# datetime object containing current date and time
now       = datetime.now()
print("date and time =", now.strftime("%d/%m/%Y %H:%M:%S"))
start_time_initialization = time.time()

learn_mode = True  # learn/infer
save_stats = True
show_res   = True   # display results 0/1
save_res   = True  # save results
save_tgt   = False  # save targets

res_path       = 'results/'  # path to results folder
Pt_path        = 'targets/'  # path to targets folder
stats_path     = 'stats/'

"Learning Hyperparameters"
loss_type   = 'l1'  # l1:L1 Norm, kl:Kullback Leibler Divergence, wass: Wasserstein (Sinkhorn) Distance"
step_size   = 0.05
num_epochs  = 201
batch_size  = 200   # 10, 20, 50, 100 - number of iterations #keep batch_size and N the same size!!!
N           = 200 #batch_size   # 100, 500, 1000  - number of total-iterations (dataset size)


num_batches, Ndevice, batch_device, num_devices = calc_and_asserts(N, batch_size)

"Interaction Initialization"
# Structure arrays - initialize crystal and structure arrays
PP_crystal  = Crystal(dx, dy, dz, MaxX, MaxY, MaxZ, d33)
M           = len(PP_crystal.x)  # simulation size


n_coeff, max_mode1, max_mode2 , max_mode_crystal= projection_crystal_modes()

Pump     = Beam(lam_pump, PP_crystal, Temperature, waist_pump, power_pump, projection_type, max_mode1, max_mode2)  # wavelength, crystal, tmperature,waist,power, maxmode
Signal   = Beam(lam_signal, PP_crystal, Temperature, np.sqrt(2)*Pump.waist, power_signal, projection_type, max_mode1, max_mode2, z=MaxZ/2)
Idler    = Beam(SFG_idler_wavelength(Pump.lam, Signal.lam), PP_crystal, Temperature, np.sqrt(2)*Pump.waist, power_signal, projection_type)

# phase mismatch
delta_k              = Pump.k - Signal.k - Idler.k
PP_crystal.poling_period = dk_offset * delta_k
"Interaction Parameters"
Nx = len(PP_crystal.x)
Ny = len(PP_crystal.y)

# normalization factor
g1_ss_normalization = G1_Normalization(Signal.w)
g1_ii_normalization = G1_Normalization(Idler.w)

# Initialize pump and crystal coefficients
coeffs         = HG_coeff_array(coeffs_str, max_mode1*max_mode2)
poling_parameters = poling_array_init(poling_str, max_mode1*max_mode_crystal)

#Settings for Fourier-Bessel or Hermite-Gauss crystal hologram
if projection_type == 'LG':
    Poling = Poling_profile(phi_scale, r_scale, PP_crystal.x, PP_crystal.y,  PP_crystal.MaxX, int((max_mode1-1)/2), max_mode_crystal, 'fourier_hankel')
elif projection_type == 'HG':
    Poling = Poling_profile(phi_scale, r_scale, PP_crystal.x, PP_crystal.y,  PP_crystal.MaxX, max_mode1, max_mode_crystal, 'hermite')


Poling.create_profile(poling_parameters)


# replicate parameters for gpus

params = coeffs
# params = pmap(lambda x: coeffs)(np.arange(num_devices))

print("--- the LG coefficients initiated are: {} ---\n".format(coeffs))
print("--- initialization time: %s seconds ---" % (time.time() - start_time_initialization))
start_time = time.time()

topic = now.strftime("%_Y-%m-%d") + "_Nb{}_Nx{}Ny{}_z{}_steps{}_#GPUs{}".format(
    batch_size, Nx, Ny, PP_crystal.MaxZ, len(PP_crystal.z), num_devices)
if learn_mode:
    topic += "_loss_{}".format(loss_type) + "_N{}".format(N) + "_epochs{}".format(num_epochs)


# @jit
@partial(pmap, axis_name='device')
def forward(params, vac_):  # vac_ = vac_s, vac_i
    N = vac_.shape[0]

    coeffs = params

    # initialize the vacuum and output fields:
    Siganl_field    = Field(Signal, PP_crystal, vac_[:, 0], N)
    Idler_field     = Field(Idler, PP_crystal, vac_[:, 1], N)

    # current pump structure
    Pump.create_profile(coeffs)

    # Propagate through the crystal:
    crystal_prop(Pump, Siganl_field, Idler_field, PP_crystal, Poling)
    #WAS: reshape(N, max_mode1, max_mode2)
    E_s_out = decompose(Siganl_field.E_out, Signal.hermite_arr).reshape(N, max_mode2, max_mode1)
    E_i_out = decompose(Idler_field.E_out, Signal.hermite_arr).reshape(N, max_mode2, max_mode1)
    E_i_vac = decompose(Idler_field.E_vac, Signal.hermite_arr).reshape(N, max_mode2, max_mode1)
    # say there are no higher modes by normalizing the power
    E_s_out = fix_power1(E_s_out, Siganl_field.E_out, Signal, PP_crystal)
    E_i_out = fix_power1(E_i_out, Idler_field.E_out, Signal, PP_crystal)
    E_i_vac = fix_power1(E_i_vac, Idler_field.E_vac, Signal, PP_crystal)

    #WAS: .reshape(max_mode1*max_mode1, max_mode2*max_mode2)
    # G2   = G2_calc(E_s_out, E_i_out, E_i_vac, batch_size).reshape(max_mode2*max_mode2, max_mode1*max_mode1)
    # return G2

    # G1_ss = lax.psum(kron(np.conj(E_s_out), E_s_out) / batch_device, 'device')
    # G1_ii = lax.psum(kron(np.conj(E_i_out), E_i_out) / batch_device, 'device')
    # G1_si = lax.psum(kron(np.conj(E_i_out), E_s_out) / batch_device, 'device')
    # G1_si_dagger = lax.psum(kron(np.conj(E_s_out), E_i_out) / batch_device, 'device')
    # Q_si = lax.psum(kron(E_i_vac, E_s_out) / batch_device, 'device')
    # Q_si_dagger = lax.psum(kron(np.conj(E_s_out), np.conj(E_i_vac)) / batch_device, 'device')
    # G2 = (G1_ii * G1_ss + Q_si_dagger * Q_si + G1_si_dagger * G1_si).reshape(max_mode2*max_mode2, max_mode1*max_mode1).real
    # return G2

    return E_s_out, E_i_out, E_i_vac


def loss(params, vac_, G2t):  # vac_ = vac_s, vac_i, G2t = P and G2 target correlation matrices
    params = params / np.sqrt(np.sum(np.abs(params) ** 2))

    # G2 = forward(pmap(lambda p: params)(np.arange(num_devices)), vac_)
    # G2 = G2.mean(0)
    E_s_out, E_i_out, E_i_vac = forward(params[None, :].repeat(num_devices, axis=0), vac_)
    G1_ss = kron(np.conj(E_s_out.reshape(-1, *(E_s_out.shape[2:]))),
                 E_s_out.reshape(-1, *(E_s_out.shape[2:]))) / batch_device
    G1_ii = kron(np.conj(E_i_out.reshape(-1, *(E_s_out.shape[2:]))),
                 E_i_out.reshape(-1, *(E_s_out.shape[2:]))) / batch_device
    G1_si = kron(np.conj(E_i_out.reshape(-1, *(E_s_out.shape[2:]))),
                 E_s_out.reshape(-1, *(E_s_out.shape[2:]))) / batch_device
    G1_si_dagger = kron(np.conj(E_s_out.reshape(-1, *(E_s_out.shape[2:]))),
                        E_i_out.reshape(-1, *(E_s_out.shape[2:]))) / batch_device
    Q_si = kron(E_i_vac.reshape(-1, *(E_s_out.shape[2:])), E_s_out.reshape(-1, *(E_s_out.shape[2:]))) / batch_device
    Q_si_dagger = kron(np.conj(E_s_out.reshape(-1, *(E_s_out.shape[2:]))),
                       np.conj(E_i_vac.reshape(-1, *(E_s_out.shape[2:])))) / batch_device
    G2 = ((
                  G1_ii * G1_ss + Q_si_dagger * Q_si + G1_si_dagger * G1_si
          ).real
          ).reshape(max_mode2 * max_mode2, max_mode1 * max_mode1)

    G2 = G2 / np.sum(np.abs(G2))
    if loss_type is 'l1':
        return l1_loss(G2, G2t)
    if loss_type is 'l2':
        return l2_loss(G2, G2t)
    if loss_type is 'kl':
        return kl_loss(G2, G2t, eps=1e-2)
    if loss_type is 'wass':
        return sinkhorn_loss(G2, G2t, n_coeff, eps=1e-3, max_iters=100, stop_thresh=None)
    else:
        raise Exception('Nonstandard loss choice')


def update(opt_state, i, x, G2t):
    params              = get_params(opt_state)
    batch_loss, grads   = value_and_grad(loss)(params, x, G2t)
    # grads               = np.array([lax.psum(dw, 'device') for dw in grads])
    # return lax.pmean(batch_loss, 'device'), opt_update(i, grads, opt_state)
    return batch_loss, opt_update(i, grads, opt_state)

if learn_mode:
    print("--- training mode ---")
    # load target P, G2
    # G2t        = pmap(lambda x: np.load(Pt_path + targert_folder + 'G2.npy'))(np.arange(num_devices))
    G2t = np.load(Pt_path + targert_folder + 'G2.npy')

    coeffs_gt = np.load(Pt_path + targert_folder + 'HG_coeffs.npy')

    assert len(coeffs) == len(coeffs_gt), "HG parameters and its ground truth must contain same length"

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
    obj_loss = []
    l1_HG_loss, l2_HG_loss = [], []
    for epoch in range(num_epochs):
        obj_loss_epoch = 0.0
        start_time_epoch = time.time()
        batch_set = pmap(get_train_batches)(vac_rnd, key_batch_epoch[:, epoch])
        print("Epoch {}/{} is running".format(epoch,num_epochs))
        for i, x in enumerate(batch_set):
            # idx = pmap(lambda x: np.array(epoch+i))(np.arange(num_devices))
            idx = np.array(epoch + i)
            batch_loss, opt_state = update(opt_state, idx, x, G2t)
            obj_loss_epoch += batch_loss.item()
        params = get_params(opt_state)
        obj_loss.append(obj_loss_epoch/(i+1))
        epoch_time = time.time() - start_time_epoch
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        ''' print loss value'''

        coeffs = params / np.sqrt(np.sum(np.abs(params)**2))

        l1_HG_loss.append(np.sum(np.abs(coeffs-coeffs_gt)))
        l2_HG_loss.append(np.sum((coeffs-coeffs_gt)**2))

        print("optimized LG coefficients: {}".format(coeffs))
        print("Norm of LG coefficients: {}".format(np.sum((np.abs(coeffs))**2)))
        print("objective loss:{:0.6f}".format(obj_loss[epoch]))

    print("--- training time: %s seconds ---" % (time.time() - start_time))

    curr_dir = stats_path + topic
    if os.path.isdir(curr_dir):
        for filename in os.listdir(curr_dir):
            os.remove(curr_dir + '/' + filename)
    else:
        os.makedirs(curr_dir)
    exp_details = open(curr_dir + '/' + "exp_details.txt", "w")
    exp_details.write(make_beam_from_HG_str(Pump.hermite_str, coeffs, coeffs_str, coeffs_gt))
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
    # plt.ylim(0, 5)
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

    # seed vacuum samples
    keys = random.split(random.PRNGKey(1986), num_devices)
    # generate dataset for each gpu
    vac_rnd = pmap(lambda key: random.normal(key, (batch_device, 2, 2, Nx, Ny)))(keys)  # number of devices, N iteration, 2-for vac states for signal and idler, 2 - real and imag, Nx X Ny for beam size)

    # G2 = pmap(forward)(params, vac_rnd)
    # G2       = G2.sum(0)

    E_s_out, E_i_out, E_i_vac = forward(params[None, :].repeat(num_devices, axis=0), vac_rnd)
    G1_ss = kron(np.conj(E_s_out.reshape(-1, *(E_s_out.shape[2:]))), E_s_out.reshape(-1, *(E_s_out.shape[2:]))) / batch_device
    G1_ii = kron(np.conj(E_i_out.reshape(-1, *(E_s_out.shape[2:]))), E_i_out.reshape(-1, *(E_s_out.shape[2:]))) / batch_device
    G1_si = kron(np.conj(E_i_out.reshape(-1, *(E_s_out.shape[2:]))), E_s_out.reshape(-1, *(E_s_out.shape[2:]))) / batch_device
    G1_si_dagger = kron(np.conj(E_s_out.reshape(-1, *(E_s_out.shape[2:]))), E_i_out.reshape(-1, *(E_s_out.shape[2:]))) / batch_device
    Q_si = kron(E_i_vac.reshape(-1, *(E_s_out.shape[2:])), E_s_out.reshape(-1, *(E_s_out.shape[2:]))) / batch_device
    Q_si_dagger = kron(np.conj(E_s_out.reshape(-1, *(E_s_out.shape[2:]))), np.conj(E_i_vac.reshape(-1, *(E_s_out.shape[2:])))) / batch_device
    G2 = ((
              G1_ii * G1_ss + Q_si_dagger * Q_si + G1_si_dagger * G1_si
              ).reshape(max_mode2 * max_mode2,max_mode1 * max_mode1).real
          ).reshape(max_mode2 * max_mode2, max_mode1 * max_mode1)
    # G2 = forward(pmap(lambda p: params)(np.arange(num_devices)), vac_rnd)
    # G2 = G2[0]

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

        np.save(curr_dir + '/HG_coeffs', coeffs)

        exp_details = open(curr_dir + '/' + "exp_details.txt", "w")
        if learn_mode:
            exp_details.write(make_beam_from_HG_str(Pump.hermite_str, coeffs, coeffs_str, coeffs_gt))
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
            exp_details.write(make_beam_from_HG_str(Pump.hermite_str, coeffs, coeffs_str, coeffs_gt))
            exp_details.write(make_taylor_from_phi_str(poling_parameters, poling_str))

        else:
            exp_details.write(make_beam_from_HG_str(Pump.hermite_str, coeffs, coeffs_str))
            exp_details.write(make_taylor_from_phi_str(poling_parameters, poling_str))
        exp_details.close()


        ################
        # Plot G2 #
        ################
        # Unwrap G2 indices
        G2_unwrap_idx_str = 'G2_unwarp_idx/G2_unwrap_max_mode{}.npy'.format(max_mode1*max_mode2)
        savetime_flag = 0
        if savetime_flag:
            if not os.path.exists(G2_unwrap_idx_str):
                G2_unwrapped_idx_np = onp.zeros((max_mode1, max_mode2, max_mode1, max_mode2), dtype=np.float32)
                print(np.shape(onp.arange(0, max_mode1 * max_mode2 * max_mode1 * max_mode2, dtype=np.float32).reshape(
                    max_mode1 * max_mode2, max_mode1 * max_mode2)))
                print(np.shape(G2_unwrapped_idx_np))
                G2_unwrapped_idx_np = \
                    unwrap_kron(G2_unwrapped_idx_np,
                                onp.arange(0, max_mode1 * max_mode2 * max_mode1 * max_mode2, dtype=np.float32).reshape(max_mode1 * max_mode1, max_mode2 * max_mode2),
                                max_mode1, max_mode2).reshape(max_mode1 * max_mode2 * max_mode1 * max_mode2)

                np.save(G2_unwrap_idx_str, G2_unwrapped_idx_np)

            else:
                G2_unwrapped_idx_np = np.load(G2_unwrap_idx_str)
            G2_unwrapped_idx = onp.ndarray.tolist(G2_unwrapped_idx_np)
            del G2_unwrapped_idx_np

            G2 = G2.reshape(max_mode1 * max_mode2 * max_mode1 * max_mode2)[G2_unwrapped_idx].reshape(max_mode1, max_mode2, max_mode1, max_mode2)
        else:
            G2_tensor = onp.zeros((max_mode2, max_mode1, max_mode2, max_mode1), dtype=np.float32)
            G2 = unwrap_kron(G2_tensor, G2, max_mode2, max_mode1)


        # Compute and plot reduced G2
        #G2_reduced = trace_it(G2, 0, 2)
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

        #Save arrays
        np.save(curr_dir + '/' + 'PolingCoeffs.npy', poling_parameters)
        np.save(curr_dir + '/' + 'PumpCoeffs.npy', coeffs)
        np.save(curr_dir + '/' + 'G2.npy', G2)


        # crystal's pattern
        #XX, ZZ = np.meshgrid(PP_crystal.x, PP_crystal.z)

        # Save poling
        magnitude = np.abs(Poling.crystal_profile)
        print(np.max(magnitude))
        DutyCycle = np.arcsin(magnitude)/np.pi
        phase = np.angle(Poling.crystal_profile)
        #plot and save the first Fourier coefficient of the cyrstal poling
        CrystalFourier = 0
        for m in [1]: #in range(100)
            if m==0:
                CrystalFourier = CrystalFourier + 2*DutyCycle - 1
            else:
                CrystalFourier = CrystalFourier + (2 / (m * np.pi)) * np.sin(m* pi * DutyCycle) * 2 * np.cos(m * phase)

        plt.imshow(CrystalFourier, aspect='auto')
        plt.xlabel(' x [um]')
        plt.ylabel(' y [um]')
        plt.title('Crystal\'s poling pattern: 1st Fourier coefficient')
        plt.colorbar()
        if save_res:
            plt.savefig(curr_dir + '/' + 'poling_1stFourier')
        if show_res:
            plt.show()
        #plot and save also the full cyrstal poling with all Fourier coefficients
        CrystalFourier = 0
        for m in range(100):
            if m==0:
                CrystalFourier = CrystalFourier + 2*DutyCycle - 1
            else:
                CrystalFourier = CrystalFourier + (2 / (m * np.pi)) * np.sin(m* pi * DutyCycle) * 2 * np.cos(m * phase)

        plt.imshow(CrystalFourier, aspect='auto')
        plt.xlabel(' x [um]')
        plt.ylabel(' y [um]')
        plt.title('Crystal\'s poling pattern: All Fourier coefficients')
        plt.colorbar()
        if save_res:
            plt.savefig(curr_dir + '/' + 'poling_AllFourier')
        if show_res:
            plt.show()
print("\n--- Done: %s seconds ---" % (time.time() - start_time))
exit()
