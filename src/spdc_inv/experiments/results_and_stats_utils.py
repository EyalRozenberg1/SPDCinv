from spdc_inv.utils.defaults import COINCIDENCE_RATE, DENSITY_MATRIX, TOMOGRAPHY_MATRIX
from spdc_inv.utils.utils import G1_Normalization
from spdc_inv import RES_DIR
from jax import numpy as np

import os
import shutil
import numpy as onp
import matplotlib.pyplot as plt


def save_training_statistics(
        logs_dir,
        fit_results,
        interaction,
        model_parameters,
):
    if fit_results is not None:
        loss_trn, best_loss = fit_results

    pump_coeffs_real, \
    pump_coeffs_imag, \
    waist_pump, \
    crystal_coeffs_real, \
    crystal_coeffs_imag, \
    r_scale = model_parameters

    pump = open(os.path.join(logs_dir, 'pump.txt'), 'w')
    pump.write(
        type_coeffs_to_txt(
            interaction.pump_basis,
            interaction.pump_max_mode1,
            interaction.pump_max_mode2,
            pump_coeffs_real[0] if pump_coeffs_real is not None
            else interaction.initial_pump_coefficients()[0],
            pump_coeffs_imag[0] if pump_coeffs_imag is not None
            else interaction.initial_pump_coefficients()[1],
            waist_pump[0] if waist_pump is not None
            else interaction.initial_pump_waists(),
        )
    )

    if interaction.crystal_basis:

        crystal = open(os.path.join(logs_dir, 'crystal.txt'), 'w')
        crystal.write(
            type_coeffs_to_txt(
                interaction.crystal_basis,
                interaction.crystal_max_mode1,
                interaction.crystal_max_mode2,
                crystal_coeffs_real[0] if crystal_coeffs_real is not None
                else interaction.initial_crystal_coefficients()[0],
                crystal_coeffs_imag[0] if crystal_coeffs_imag is not None
                else interaction.initial_crystal_coefficients()[1],
                r_scale[0] if r_scale is not None
                else interaction.initial_crystal_waists(),
            )
        )

    if fit_results is not None:
        # print loss
        plt.plot(loss_trn, 'r', label='training')
        plt.ylabel('objective loss')
        plt.xlabel('#epoch')
        # plt.ylim(0.2, 1)
        plt.axhline(y=best_loss, color='gray', linestyle='--')
        plt.text(2, best_loss, f'best = {best_loss}', rotation=0, horizontalalignment='left',
                 verticalalignment='top', multialignment='center')
        plt.legend()
        plt.savefig(os.path.join(logs_dir, 'loss'))
        plt.close()

    np.save(os.path.join(logs_dir, 'parameters_pump_real.npy'),
            pump_coeffs_real[0] if pump_coeffs_real is not None
            else interaction.initial_pump_coefficients()[0])
    np.save(os.path.join(logs_dir, 'parameters_pump_imag.npy'),
            pump_coeffs_imag[0] if pump_coeffs_imag is not None
            else interaction.initial_pump_coefficients()[1])
    np.save(os.path.join(logs_dir, 'parameters_pump_waists.npy'),
            waist_pump[0] if waist_pump is not None
            else interaction.initial_pump_waists())
    if interaction.crystal_basis is not None:
        np.save(os.path.join(logs_dir, 'parameters_crystal_real.npy'),
                crystal_coeffs_real[0] if crystal_coeffs_real is not None
                else interaction.initial_crystal_coefficients()[0])
        np.save(os.path.join(logs_dir, 'parameters_crystal_imag.npy'),
                crystal_coeffs_imag[0] if crystal_coeffs_imag is not None
                else interaction.initial_crystal_coefficients()[1])
        np.save(os.path.join(logs_dir, 'parameters_crystal_effective_waists.npy'),
                r_scale[0] if r_scale is not None
                else interaction.initial_crystal_waists()
                )

    return


def save_results(
        run_name,
        observable_vec,
        observables,
        projection_coincidence_rate,
        projection_tomography_matrix,
        Signal,
        Idler,
):
    results_dir = os.path.join(RES_DIR, run_name)
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    (coincidence_rate, density_matrix, tomography_matrix) = observables

    if observable_vec[COINCIDENCE_RATE]:
        coincidence_rate = coincidence_rate[0]
        coincidence_rate = coincidence_rate / np.sum(np.abs(coincidence_rate))
        np.save(os.path.join(results_dir, 'coincidence_rate.npy'), coincidence_rate)
        coincidence_rate_plots(
            results_dir,
            coincidence_rate,
            projection_coincidence_rate,
            Signal,
            Idler,
        )

    if observable_vec[DENSITY_MATRIX]:
        density_matrix = density_matrix[0]
        density_matrix = density_matrix / np.trace(np.real(density_matrix))
        np.save(os.path.join(results_dir, 'density_matrix_real.npy'), onp.real(density_matrix))
        np.save(os.path.join(results_dir, 'density_matrix_imag.npy'), onp.imag(density_matrix))
        density_matrix_plots(
            results_dir,
            density_matrix,
        )

    if observable_vec[TOMOGRAPHY_MATRIX]:
        tomography_matrix = tomography_matrix[0]
        tomography_matrix = tomography_matrix / np.sum(np.abs(tomography_matrix))
        np.save(os.path.join(results_dir, 'tomography_matrix.npy'), tomography_matrix)
        tomography_matrix_plots(
            results_dir,
            tomography_matrix,
            projection_tomography_matrix,
            Signal,
            Idler,
        )


def coincidence_rate_plots(
        results_dir,
        coincidence_rate,
        projection_coincidence_rate,
        Signal,
        Idler,
):
    # coincidence_rate = unwrap_kron(coincidence_rate,
    #                                projection_coincidence_rate.projection_n_modes1,
    #                                projection_coincidence_rate.projection_n_modes2)
    coincidence_rate = coincidence_rate[0, :].\
        reshape(projection_coincidence_rate.projection_n_modes2, projection_coincidence_rate.projection_n_modes2)

    # Compute and plot reduced coincidence_rate
    g1_ss_normalization = G1_Normalization(Signal.w)
    g1_ii_normalization = G1_Normalization(Idler.w)
    coincidence_rate_reduced = coincidence_rate * \
                               projection_coincidence_rate.tau / (g1_ii_normalization * g1_ss_normalization)

    # plot coincidence_rate 2d
    plt.imshow(coincidence_rate_reduced)
    plt.xlabel(r'signal mode i')
    plt.ylabel(r'idle mode j')
    plt.colorbar()

    plt.savefig(os.path.join(results_dir, 'coincidence_rate'))
    plt.close()


def tomography_matrix_plots(
        results_dir,
        tomography_matrix,
        projection_tomography_matrix,
        Signal,
        Idler,
):

    # tomography_matrix = unwrap_kron(tomography_matrix,
    #                                 projection_tomography_matrix.projection_n_state1,
    #                                 projection_tomography_matrix.projection_n_state2)

    tomography_matrix = tomography_matrix[0, :].\
        reshape(projection_tomography_matrix.projection_n_state2, projection_tomography_matrix.projection_n_state2)

    # Compute and plot reduced tomography_matrix
    g1_ss_normalization = G1_Normalization(Signal.w)
    g1_ii_normalization = G1_Normalization(Idler.w)

    tomography_matrix_reduced = tomography_matrix * \
                                projection_tomography_matrix.tau / (g1_ii_normalization * g1_ss_normalization)

    # plot tomography_matrix 2d
    plt.imshow(tomography_matrix_reduced)
    plt.xlabel(r'signal mode i')
    plt.ylabel(r'idle mode j')
    plt.colorbar()

    plt.savefig(os.path.join(results_dir, 'tomography_matrix'))
    plt.close()


def density_matrix_plots(
        results_dir,
        density_matrix,
):

    density_matrix_real = onp.real(density_matrix)
    density_matrix_imag = onp.imag(density_matrix)

    plt.imshow(density_matrix_real)
    plt.xlabel(r'signal mode i')
    plt.ylabel(r'idle mode j')
    plt.colorbar()
    plt.savefig(os.path.join(results_dir, 'density_matrix_real'))
    plt.close()

    plt.imshow(density_matrix_imag)
    plt.xlabel(r'signal mode i')
    plt.ylabel(r'idle mode j')
    plt.colorbar()
    plt.savefig(os.path.join(results_dir, 'density_matrix_imag'))
    plt.close()


def type_coeffs_to_txt(
        basis,
        max_mode1,
        max_mode2,
        coeffs_real,
        coeffs_imag,
        waists):
    sign = {'1.0': '+', '-1.0': '-', '0.0': '+'}
    print_str = f'basis: {basis}({max_mode1},{max_mode2}):\n'
    for _real, _imag, _waist in zip(coeffs_real, coeffs_imag, waists):
        sign_imag = sign[str(onp.sign(_imag).item())]
        print_str += '{:.4} {} j{:.4} (waist: {:.4}[um])\n'.format(_real, sign_imag, onp.abs(_imag), _waist * 10)
    return print_str


def unwrap_kron(G, M1, M2):
    '''
    the function takes a Kronicker product of size M1^2 x M2^2 and turns is into an
    M1 x M2 x M1 x M2 tensor. It is used only for illustration and not during the learning
    Parameters
    ----------
    G: the tensor we wish to reshape
    M1: first dimension
    M2: second dimension

    Returns a reshaped tensor with shape (M1, M2, M1, M2)
    -------

    '''

    C = onp.zeros((M1, M2, M1, M2), dtype=onp.float32)

    for i in range(M1):
        for j in range(M2):
            for k in range(M1):
                for l in range(M2):
                    C[i, j, k, l] = G[k + M1 * i, l + M2 * j]
    return C
