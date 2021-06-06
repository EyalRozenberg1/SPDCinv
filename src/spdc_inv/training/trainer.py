import jax.random as random
import time
import numpy as onp

from abc import ABC
from typing import Dict, Optional, Tuple, Any
from jax import pmap
from jax import value_and_grad
from jax import lax
from functools import partial
from jax import numpy as np
from spdc_inv.utils.utils import Crystal_hologram, Beam_profile
from spdc_inv.models.spdc_model import SPDCmodel
from spdc_inv.utils.defaults import COINCIDENCE_RATE, DENSITY_MATRIX, TOMOGRAPHY_MATRIX


class BaseTrainer(ABC):
    """
    A class abstracting the various tasks of training models.
    Provides methods at multiple levels of granularity
    """
    def __init__(
            self,
            key: np.array,
            n_epochs: int,
            N_train: int,
            bs_train: int,
            N_inference: int,
            bs_inference: int,
            N_train_device: int,
            N_inference_device: int,
            learn_pump_coeffs: bool,
            learn_pump_waists: bool,
            learn_crystal_coeffs: bool,
            learn_crystal_waists: bool,
            keep_best: bool,
            n_devices: int,
            projection_coincidence_rate,
            projection_tomography_matrix,
            interaction,
            pump,
            signal,
            idler,
            observable_vec: Optional[Tuple[Dict[Any, bool]]],
    ):

        self.key          = key
        self.n_devices    = n_devices
        self.n_epochs   = n_epochs
        self.N_train      = N_train
        self.bs_train     = bs_train
        self.N_inference  = N_inference
        self.bs_inference = bs_inference
        self.N_train_device     = N_train_device
        self.N_inference_device = N_inference_device
        self.keep_best          = keep_best
        self.delta_k            = pump.k - signal.k - idler.k  # phase mismatch
        self.poling_period      = interaction.dk_offset * self.delta_k
        self.Nx = interaction.Nx
        self.Ny = interaction.Ny
        self.DeltaZ = - interaction.maxZ / 2  # DeltaZ: longitudinal middle of the crystal (with negative sign).
                                              # To propagate generated fields back to the middle of the crystal

        self.projection_coincidence_rate = projection_coincidence_rate
        self.projection_tomography_matrix = projection_tomography_matrix

        assert list(observable_vec.keys()) == [COINCIDENCE_RATE,
                                               DENSITY_MATRIX,
                                               TOMOGRAPHY_MATRIX], 'observable_vec must only contain ' \
                                                                   'the keys [coincidence_rate,' \
                                                                   'density_matrix, tomography_matrix]'

        self.coincidence_rate_observable = observable_vec[COINCIDENCE_RATE]
        self.density_matrix_observable = observable_vec[DENSITY_MATRIX]
        self.tomography_matrix_observable = observable_vec[TOMOGRAPHY_MATRIX]
        self.coincidence_rate_loss = None
        self.density_matrix_loss = None
        self.tomography_matrix_loss = None
        self.opt_init, self.opt_update, self.get_params = None, None, None
        self.target_coincidence_rate = None
        self.target_density_matrix = None
        self.target_tomography_matrix = None

        # Initialize pump and crystal coefficients
        self.pump_coeffs_real, \
        self.pump_coeffs_imag = interaction.initial_pump_coefficients()
        self.waist_pump       = interaction.initial_pump_waists()

        self.crystal_coeffs_real,\
        self.crystal_coeffs_imag = interaction.initial_crystal_coefficients()
        self.r_scale             = interaction.initial_crystal_waists()

        self.model_parameters = pmap(lambda x: (
                                                self.pump_coeffs_real,
                                                self.pump_coeffs_imag,
                                                self.waist_pump,
                                                self.crystal_coeffs_real,
                                                self.crystal_coeffs_imag,
                                                self.r_scale
        ))(np.arange(self.n_devices))

        print(f"Interaction length [m]: {interaction.maxZ} \n")
        print(f"Pump   beam  basis  coefficients: \n {self.pump_coeffs_real + 1j * self.pump_coeffs_imag}\n")
        print(f"Pump basis functions waists [um]: \n {self.waist_pump * 10}\n")

        if interaction.crystal_basis:
            print(f"3D hologram  basis  coefficients: \n {self.crystal_coeffs_real + 1j * self.crystal_coeffs_imag}\n")
            print("3D hologram basis functions-"
                  f"effective  waists (r_scale) [um]: \n {self.r_scale * 10}\n")
            self.crystal_hologram = Crystal_hologram(self.crystal_coeffs_real,
                                                     self.crystal_coeffs_imag,
                                                     self.r_scale,
                                                     interaction.x,
                                                     interaction.y,
                                                     interaction.crystal_max_mode1,
                                                     interaction.crystal_max_mode2,
                                                     interaction.crystal_basis,
                                                     signal.lam,
                                                     signal.n,
                                                     learn_crystal_coeffs or learn_crystal_waists)
        else:
            self.crystal_hologram = None

        self.pump_structure = Beam_profile(self.pump_coeffs_real,
                                           self.pump_coeffs_imag,
                                           self.waist_pump,
                                           interaction.power_pump,
                                           interaction.x,
                                           interaction.y,
                                           interaction.dx,
                                           interaction.dy,
                                           interaction.pump_max_mode1,
                                           interaction.pump_max_mode2,
                                           interaction.pump_basis,
                                           interaction.lam_pump,
                                           pump.n,
                                           learn_pump_coeffs or learn_pump_waists)

        self.model = SPDCmodel(pump,
                               signal=signal,
                               idler=idler,
                               projection_coincidence_rate=projection_coincidence_rate,
                               projection_tomography_matrix=projection_tomography_matrix,
                               interaction=interaction,
                               pump_structure=self.pump_structure,
                               crystal_hologram=self.crystal_hologram,
                               poling_period=self.poling_period,
                               DeltaZ=self.DeltaZ,
                               coincidence_rate_observable=self.coincidence_rate_observable,
                               density_matrix_observable=self.density_matrix_observable,
                               tomography_matrix_observable=self.tomography_matrix_observable,)

    def inference(self):
        self.model.learn_mode = False
        self.model.N          = self.N_inference
        self.model.N_device   = self.N_inference_device
        self.model.bs         = self.bs_inference

        # seed vacuum samples for each gpu
        self.key, subkey = random.split(self.key)
        keys = random.split(subkey, self.n_devices)
        observables = pmap(self.model.forward, axis_name='device')(self.model_parameters, keys)

        return observables

    def fit(self):
        self.model.learn_mode = True
        self.model.N          = self.N_train
        self.model.N_device   = self.N_train_device
        self.model.bs         = self.bs_train

        start_time = time.time()
        opt_state = self.opt_init(self.model_parameters)

        loss_trn, loss_vld, best_loss = [], [], None
        epochs_without_improvement = 0

        for epoch in range(self.n_epochs):
            start_time_epoch = time.time()
            print(f'running epoch {epoch}/{self.n_epochs}')

            idx  = np.array([epoch]).repeat(self.n_devices)
            self.key, subkey = random.split(self.key)
            training_subkey, validation_subkey = random.split(subkey)
            training_subkeys   = random.split(training_subkey, self.n_devices)
            validation_subkeys = random.split(validation_subkey, self.n_devices)

            training_loss, validation_loss, opt_state = self.update(opt_state,
                                                                    idx,
                                                                    training_subkeys,
                                                                    validation_subkeys)
            loss_trn.append(training_loss[0].item())
            loss_vld.append(validation_loss[0].item())

            print("in {:0.2f} sec".format(time.time() - start_time_epoch))

            print("training   objective loss:{:0.6f}".format(loss_trn[epoch]))
            print("validation objective loss:{:0.6f}".format(loss_vld[epoch]))

            if best_loss is None or loss_vld[epoch] < best_loss and not onp.isnan(loss_vld[epoch]):

                print(f'best validation objective loss is reached\n')

                model_parameters = self.get_params(opt_state)
                pump_coeffs_real, \
                pump_coeffs_imag, \
                waist_pump, \
                crystal_coeffs_real, \
                crystal_coeffs_imag, \
                r_scale = model_parameters

                normalization = np.sqrt(np.sum(np.abs(pump_coeffs_real) ** 2 +
                                               np.abs(pump_coeffs_imag) ** 2, 1, keepdims=True))
                pump_coeffs_real = pump_coeffs_real / normalization
                pump_coeffs_imag = pump_coeffs_imag / normalization

                print(f"Pump   beam  basis  coefficients: \n {pump_coeffs_real[0] + 1j * pump_coeffs_imag[0]}\n")
                print(f"Pump basis functions waists [um]: \n {waist_pump[0] * 10}\n")

                if self.crystal_hologram:
                    normalization = np.sqrt(np.sum(np.abs(crystal_coeffs_real) ** 2 +
                                                   np.abs(crystal_coeffs_imag) ** 2, 1, keepdims=True))
                    crystal_coeffs_real = crystal_coeffs_real / normalization
                    crystal_coeffs_imag = crystal_coeffs_imag / normalization

                    print(f"3D hologram  basis  coefficients: \n "
                          f"{crystal_coeffs_real[0] + 1j * crystal_coeffs_imag[0]}\n")
                    print("3D hologram basis functions-"
                          f"effective  waists (r_scale) [um]: \n {r_scale[0] * 10}\n")

                best_loss = loss_vld[epoch]
                epochs_without_improvement = 0
                if self.keep_best:
                    self.model_parameters = model_parameters
                    print(f'parameters are updated\n')
            else:
                epochs_without_improvement += 1
                print(f'number of epochs without improvement {epochs_without_improvement}, '
                      f'best objective loss {best_loss}')

            if not self.keep_best:
                model_parameters = self.get_params(opt_state)
                self.model_parameters = model_parameters

        print("training is done after: %s seconds" % (time.time() - start_time))
        return loss_trn, loss_vld, best_loss

    @partial(pmap, axis_name='device', static_broadcasted_argnums=(0,))
    def update(
            self,
            opt_state,
            i,
            training_subkeys,
            validation_subkeys
    ):

        model_parameters = self.get_params(opt_state)
        loss, grads = value_and_grad(self.loss)(model_parameters, training_subkeys)

        grads_ = []
        for g, grads_param in enumerate(grads):
            if grads_param is not None:
                grads_.append(np.array([lax.psum(dw, 'device') for dw in grads_param]))
            else:
                grads_.append(None)
        grads = tuple(grads_)

        opt_state     = self.opt_update(i, grads, opt_state)
        training_loss = lax.pmean(loss, 'device')

        model_parameters = self.get_params(opt_state)
        loss = self.loss(model_parameters, validation_subkeys)
        validation_loss = lax.pmean(loss, 'device')

        return training_loss, validation_loss, opt_state


    def loss(
            self,
            model_parameters,
            keys,
    ):
        pump_coeffs_real, \
        pump_coeffs_imag, \
        waist_pump, \
        crystal_coeffs_real, \
        crystal_coeffs_imag, \
        r_scale = model_parameters

        normalization    = np.sqrt(np.sum(np.abs(pump_coeffs_real) ** 2 + np.abs(pump_coeffs_imag) ** 2))
        pump_coeffs_real = pump_coeffs_real / normalization
        pump_coeffs_imag = pump_coeffs_imag / normalization

        if self.crystal_hologram:
            normalization = np.sqrt(np.sum(np.abs(crystal_coeffs_real) ** 2 + np.abs(crystal_coeffs_imag) ** 2))
            crystal_coeffs_real = crystal_coeffs_real / normalization
            crystal_coeffs_imag = crystal_coeffs_imag / normalization

        model_parameters = (pump_coeffs_real,
                            pump_coeffs_imag,
                            waist_pump,
                            crystal_coeffs_real,
                            crystal_coeffs_imag,
                            r_scale)

        observables = self.model.forward(model_parameters, keys)

        (coincidence_rate, density_matrix, tomography_matrix) = observables

        if self.coincidence_rate_loss.observable_as_target:
            coincidence_rate = coincidence_rate / np.sum(np.abs(coincidence_rate))
        coincidence_rate_loss = self.coincidence_rate_loss.apply(coincidence_rate,
                                                                 model_parameters, self.target_coincidence_rate)

        if self.density_matrix_loss.observable_as_target:
            density_matrix = density_matrix / np.trace(np.real(density_matrix))
        density_matrix_loss = self.density_matrix_loss.apply(density_matrix,
                                                             model_parameters, self.target_density_matrix)

        if self.tomography_matrix_loss.observable_as_target:
            tomography_matrix = tomography_matrix / np.sum(np.abs(tomography_matrix))
        tomography_matrix_loss = self.tomography_matrix_loss.apply(tomography_matrix,
                                                                   model_parameters, self.target_tomography_matrix)

        return coincidence_rate_loss + density_matrix_loss + tomography_matrix_loss

