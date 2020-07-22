from __future__ import print_function, division, absolute_import
from loss_funcs import l1_loss, kl_loss, sinkhorn_loss, l2_loss
from spdc_helper import *
from spdc_funcs import *
from physical_params import *

JAX_ENABLE_X64 = False
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# datetime object containing current date and time
now       = datetime.now()
print("date and time =", now.strftime("%d/%m/%Y %H:%M:%S"))
start_time_initialization = time.time()

learn_mode = True  # learn/infer
save_stats = False
show_res   = False   # display results 0/1
save_res   = False  # save results
save_tgt   = False  # save targets

res_path       = 'results/'  # path to results folder
Pt_path        = 'targets/'  # path to targets folder
stats_path     = 'stats/'

"Learning Hyperparameters"
loss_type   = 'l1'  # l1:L1 Norm, kl:Kullback–Leibler Divergence, wass: Wasserstein (Sinkhorn) Distance"
step_size   = 0.01
num_epochs  = 30
batch_size  = 100   # 10, 20, 50, 100 - number of iterations
N           = 1000  # 100, 500, 1000  - number of total-iterations (dataset size)
alpha       = 0.5   # in [0,1]; weight for loss: (1-alpha)Pss + alpha G2


num_batches, Ndevice, batch_device, num_devices = calc_and_asserts(N, batch_size)

"Interaction Initialization"
# Structure arrays - initialize crystal and structure arrays
PP_SLT      = Crystal(dx, dy, dz, MaxX, MaxY, MaxZ, d33)
M           = len(PP_SLT.x)  # simulation size

"""
* define two pump's function (for now n_coeff must be 2) to define the pump *
* this should be later changed to the definition given by Sivan *
"""
n_coeff  = max_mode**2  # coefficients of beam-basis functions
Pump     = Beam(lam_pump, PP_SLT, Temperature, waist_pump, power_pump, max_mode)  # wavelength, crystal, tmperature,waist,power, maxmode
Signal   = Beam(lam_signal, PP_SLT, Temperature, np.sqrt(2)*Pump.waist, power_signal, max_mode)
Idler    = Beam(SFG_idler_wavelength(Pump.lam, Signal.lam), PP_SLT, Temperature, np.sqrt(2)*Pump.waist, power_signal)

# phase mismatch
delta_k              = Pump.k - Signal.k - Idler.k
PP_SLT.poling_period = dk_offset * delta_k

"Interaction Parameters"
Nx = len(PP_SLT.x)
Ny = len(PP_SLT.y)

# normalization factor
g1_ss_normalization = G1_Normalization(Signal.w)
g1_ii_normalization = G1_Normalization(Idler.w)

coeffs         = HG_coeff_array(coeffs_str, n_coeff)
phi_parameters = poling_array(poling_str)

Poling = Poling_profile(phi_scale, PP_SLT.x,  PP_SLT.MaxX, len(phi_parameters))

# replicate parameters for gpus
params = pmap(lambda x: np.concatenate((coeffs, phi_parameters)))(np.arange(num_devices))

print("--- the HG coefficients initiated are: {} ---\n".format(coeffs))
print("--- the Taylor coefficients initiated are: {} ---\n".format(phi_parameters))
print("--- initialization time: %s seconds ---" % (time.time() - start_time_initialization))
start_time = time.time()

topic = now.strftime("%_Y-%m-%d") + "_Nb{}_Nx{}Ny{}_z{}_steps{}".format(batch_size, Nx, Ny, PP_SLT.MaxZ, len(PP_SLT.z))
if learn_mode:
    topic += "_loss_{}".format(loss_type) + "_alpha{}".format(alpha) + "_N{}".format(N)


@jit
def forward(params, vac_):  # vac_ = vac_s, vac_i
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

    E_s_out = decompose(Siganl_field.E_out, Signal.hermite_arr).reshape(N, max_mode, max_mode)
    E_i_out = decompose(Idler_field.E_out, Signal.hermite_arr).reshape(N, max_mode, max_mode)
    E_i_vac = decompose(Idler_field.E_vac, Signal.hermite_arr).reshape(N, max_mode, max_mode)
    # say there are no higher modes by normalizing the power
    E_s_out = fix_power1(E_s_out, Siganl_field.E_out, Signal, PP_SLT)
    E_i_out = fix_power1(E_i_out, Idler_field.E_out, Signal, PP_SLT)
    E_i_vac = fix_power1(E_i_vac, Idler_field.E_vac, Signal, PP_SLT)

    "Coumpute k-space far field using FFT:"
    # normalization factors
    FarFieldNorm_signal = (2 * PP_SLT.MaxX) ** 2 / (np.size(Siganl_field.E_out) * Signal.lam * R)
    # FFT:
    E_s_out_k = FarFieldNorm_signal * Fourier(Siganl_field.E_out)

    P_ss = Pss_calc(E_s_out_k, Nx, Ny, M, batch_size)
    G2   = G2_calc(E_s_out, E_i_out, E_i_vac, batch_size).reshape(max_mode**2, max_mode**2)
    return P_ss, G2


def loss(params, vac_, P_ss_t, G2t):  # vac_ = vac_s, vac_i, G2t = P and G2 target correlation matrices
    coeffs, phi_parameters = params[:n_coeff], params[n_coeff:]
    coeffs = coeffs / np.sum(np.abs(coeffs))
    params = np.concatenate((coeffs, phi_parameters))

    P_ss, G2 = forward(params, vac_)
    P_ss     = P_ss / np.sum(np.abs(P_ss))
    G2       = G2 / np.sum(np.abs(G2))

    if loss_type is 'l1':
        return (1-alpha)*l1_loss(P_ss, P_ss_t) + alpha*l1_loss(G2, G2t)
    if loss_type is 'l2':
        return (1-alpha)*l2_loss(P_ss, P_ss_t) + alpha*l2_loss(G2, G2t)
    if loss_type is 'kl':
        return (1-alpha)*kl_loss(P_ss_t, P_ss, eps=1e-7)+alpha*kl_loss(G2t, G2, eps=1)
    if loss_type is 'wass':
        return (1-alpha)*sinkhorn_loss(P_ss, P_ss_t, M, eps=1e-3, max_iters=100, stop_thresh=None) + \
               alpha*sinkhorn_loss(G2, G2t, n_coeff, eps=1e-3, max_iters=100, stop_thresh=None)
    else:
        raise Exception('Nonstandard loss choice')


@partial(pmap, axis_name='device')
def update(opt_state, i, x, P_ss_t, G2t):
    params              = get_params(opt_state)
    batch_loss, grads   = value_and_grad(loss)(params, x, P_ss_t, G2t)
    grads               = np.array([lax.psum(dw, 'device') for dw in grads])
    return lax.pmean(batch_loss, 'device'), opt_update(i, grads, opt_state)

if learn_mode:
    print("--- training mode ---")
    # load target P, G2
    P_ss_t     = pmap(lambda x: np.load(Pt_path + targert_folder + 'P_ss.npy'))(np.arange(num_devices))
    G2t        = pmap(lambda x: np.load(Pt_path + targert_folder + 'G2.npy'))(np.arange(num_devices))

    coeffs_gt = np.load(Pt_path + targert_folder + 'HG_coeffs.npy')
    phi_parameters_gt = np.load(Pt_path + targert_folder + 'Taylor_coeffs.npy')

    assert len(coeffs) == len(coeffs_gt), "HG parameters and its ground truth must contain same length"
    assert len(phi_parameters) == len(phi_parameters_gt), "Taylor series phi parameters and its ground truth must contain same length"

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
    l1_tylor_loss, l2_tylor_loss = [], []
    for epoch in range(num_epochs):
        obj_loss_epoch = 0.0
        start_time_epoch = time.time()
        batch_set = pmap(get_train_batches)(vac_rnd, key_batch_epoch[:, epoch])
        print("Epoch {} is running".format(epoch))
        for i, x in enumerate(batch_set):
            idx = pmap(lambda x: np.array(epoch+i))(np.arange(num_devices))
            batch_loss, opt_state = update(opt_state, idx, x, P_ss_t, G2t)
            obj_loss_epoch += batch_loss[0].item()
        params = get_params(opt_state)
        obj_loss.append(obj_loss_epoch/(i+1))
        epoch_time = time.time() - start_time_epoch
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        ''' print loss value'''
        coeffs, phi_parameters = params[0][:n_coeff], params[0][n_coeff:]
        coeffs = coeffs / np.sum(np.abs(coeffs))
        l1_HG_loss.append(np.sum(np.abs(coeffs-coeffs_gt)))
        l2_HG_loss.append(np.sum((coeffs-coeffs_gt)**2))
        l1_tylor_loss.append(np.sum(np.abs(phi_parameters - phi_parameters_gt)))
        l2_tylor_loss.append(np.sum((phi_parameters - phi_parameters_gt) ** 2))
        print("optimized HG coefficients: {}".format(coeffs))
        print("optimized Taylor coefficients: {}".format(phi_parameters))
        print("objective loss:{:0.6f}".format(obj_loss[epoch]))
        print("l1 HG coeffs loss:{:0.6f}".format(l1_HG_loss[epoch]))
        print("l2 HG coeffs loss:{:0.6f}".format(l2_HG_loss[epoch]))
        print("l1 Taylor coeffs loss:{:0.6f}".format(l1_tylor_loss[epoch]))
        print("l2 Taylor coeffs loss:{:0.6f}".format(l2_tylor_loss[epoch]))


    print("--- training time: %s seconds ---" % (time.time() - start_time))

    curr_dir = stats_path + topic
    if os.path.isdir(curr_dir):
        for filename in os.listdir(curr_dir):
            os.remove(curr_dir + '/' + filename)
    else:
        os.makedirs(curr_dir)
    exp_details = open(curr_dir + '/' + "exp_details.txt", "w")
    exp_details.write(make_beam_from_HG_str(Pump.hermite_str, coeffs, coeffs_gt))
    exp_details.write(make_taylor_from_phi_str(phi_parameters, phi_parameters_gt))
    exp_details.close()

    plt.plot(obj_loss, 'r')
    plt.title('(1-{}) loss(Pss) + {} loss(G2), loss type:{}'.format(alpha, alpha, loss_type))
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

    plt.plot(l1_tylor_loss, 'r', label='L1')
    plt.plot(l2_tylor_loss, 'b', label='L2')
    plt.title('Taylor coefficients')
    plt.ylabel('measure loss')
    plt.xlabel('#epoch')
    plt.legend()
    if save_stats:
        plt.savefig(curr_dir + '/Taylor_coeffs_losses')
    plt.show()
    plt.close()

# show last epoch result
if save_res or save_tgt or show_res:
    print("--- inference mode ---")
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
        print("--- saving targets ---")
        curr_dir = Pt_path + topic
        if os.path.isdir(curr_dir):
            for filename in os.listdir(curr_dir):
                os.remove(curr_dir + '/' + filename)
        else:
            os.makedirs(curr_dir)

        Pss_t_name = 'P_ss'
        G2_t_name = 'G2'
        # save normalized version
        np.save(curr_dir + '/' + Pss_t_name, P_ss / np.sum(np.abs(P_ss)))
        np.save(curr_dir + '/' + G2_t_name, G2 / np.sum(np.abs(G2)))

        np.save(curr_dir + '/HG_coeffs', coeffs)
        np.save(curr_dir + '/Taylor_coeffs', phi_parameters)

        exp_details = open(curr_dir + '/' + "exp_details.txt", "w")
        if learn_mode:
            exp_details.write(make_beam_from_HG_str(Pump.hermite_str, coeffs, coeffs_gt))
            exp_details.write(make_taylor_from_phi_str(phi_parameters, phi_parameters_gt))
        else:
            exp_details.write(make_beam_from_HG_str(Pump.hermite_str, coeffs))
            exp_details.write(make_taylor_from_phi_str(phi_parameters))
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
            exp_details.write(make_beam_from_HG_str(Pump.hermite_str, coeffs, coeffs_gt))
            exp_details.write(make_taylor_from_phi_str(phi_parameters, phi_parameters_gt))

        else:
            exp_details.write(make_beam_from_HG_str(Pump.hermite_str, coeffs))
            exp_details.write(make_taylor_from_phi_str(phi_parameters))
        exp_details.close()

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
            plt.savefig(curr_dir + '/' + 'P_ss')
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
            plt.savefig(curr_dir + '/' + 'G2')
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
        if save_res:
            plt.savefig(curr_dir + '/' + 'poling')
        if show_res:
            plt.show()

print("\n--- Done: %s seconds ---" % (time.time() - start_time))
exit()
