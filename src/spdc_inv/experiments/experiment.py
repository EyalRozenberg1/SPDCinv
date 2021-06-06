import random
import os
import shutil
from jax import pmap
from jax import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List, Union
from spdc_inv import DATA_DIR
from spdc_inv import LOGS_DIR
from spdc_inv import RES_DIR
from spdc_inv.utils.utils import Beam
from spdc_inv.utils.defaults import COINCIDENCE_RATE, DENSITY_MATRIX, TOMOGRAPHY_MATRIX
from spdc_inv.utils.defaults import REAL, IMAG
from spdc_inv.loss.loss import Loss
from spdc_inv.data.interaction import Interaction
from spdc_inv.experiments.utils import Projection_coincidence_rate, Projection_tomography_matrix
from spdc_inv.experiments.results_and_stats_utils import save_results, save_training_statistics
from spdc_inv.training.trainer import BaseTrainer
from spdc_inv.optim.optimizer import Optimizer


def run_experiment(
        run_name: str,
        seed: int = 42,
        CUDA_VISIBLE_DEVICES: str = None,
        JAX_ENABLE_X64: str = 'True',
        learn_mode: bool = False,
        learn_pump_coeffs: bool = True,
        learn_pump_waists: bool = True,
        learn_crystal_coeffs: bool = True,
        learn_crystal_waists: bool = True,
        keep_best: bool = True,
        n_epochs: int = 500,
        N_train: int = 1000,
        bs_train: int = 1000,
        N_inference: int = 1000,
        bs_inference: int = 1000,
        target: str = 'qutrit',
        observable_vec: Tuple[Dict[Any, bool]] = None,
        loss_arr: Tuple[Dict[Any, Optional[Tuple[str, str]]]] = None,
        loss_weights: Tuple[Dict[Any, Optional[Tuple[float, float]]]] = None,
        reg_observable: Tuple[Dict[Any, Optional[Tuple[str, str]]]] = None,
        reg_observable_w: Tuple[Dict[Any, Optional[Tuple[float, float]]]] = None,
        reg_observable_elements: Tuple[Dict[Any, Optional[Tuple[List[int], List[int]]]]] = None,
        tau: float = 1e-9,
        l2_reg: float = 1e-5,
        optimizer: str = 'adam',
        exp_decay_lr: bool = True,
        step_size: float = 0.05,
        decay_steps: int = 50,
        decay_rate: float = 0.5,
        pump_basis: str = 'LG',
        pump_max_mode1: int = 5,
        pump_max_mode2: int = 1,
        initial_pump_coefficient: str = 'LG00',
        custom_pump_coefficient: Dict[str, Dict[int, int]] = None,
        pump_coefficient_path: str = None,
        initial_pump_waist: str = 'waist_pump0',
        pump_waists_path: str = None,
        crystal_basis: str = 'LG',
        crystal_max_mode1: int = 5,
        crystal_max_mode2: int = 2,
        initial_crystal_coefficient: str = 'LG00',
        custom_crystal_coefficient: Dict[str, Dict[int, int]] = None,
        crystal_coefficient_path: str = None,
        initial_crystal_waist: str = 'r_scale0',
        crystal_waists_path: str = None,
        lam_pump: float = 405e-9,
        crystal_str: str = 'ktp',
        power_pump: float = 1e-3,
        waist_pump0: float = 40e-6,
        r_scale0: float = 40e-6,
        dx: float = 4e-6,
        dy: float = 4e-6,
        dz: float = 10e-6,
        maxX: float = 120e-6,
        maxY: float = 120e-6,
        maxZ: float = 1e-4,
        R: float = 0.1,
        Temperature: float = 50,
        pump_polarization: str = 'y',
        signal_polarization: str = 'y',
        idler_polarization: str = 'z',
        dk_offset: float = 1.,
        power_signal: float = 1.,
        power_idler: float = 1.,
        coincidence_projection_basis: str = 'LG',
        coincidence_projection_max_mode1: int = 1,
        coincidence_projection_max_mode2: int = 4,
        coincidence_projection_waist: float = None,
        coincidence_projection_wavelength: float = None,
        coincidence_projection_polarization: str = 'y',
        coincidence_projection_z: float = 0.,
        tomography_projection_basis: str = 'LG',
        tomography_projection_max_mode1: int = 1,
        tomography_projection_max_mode2: int = 1,
        tomography_projection_waist: float = None,
        tomography_projection_wavelength: float = None,
        tomography_projection_polarization: str = 'y',
        tomography_projection_z: float = 0.,
        tomography_relative_phase: List[Union[Union[int, float], Any]] = None,
        tomography_quantum_state: str = 'qubit'

):

    now = datetime.now()
    date_and_time = now.strftime("%d/%m/%Y %H:%M:%S")
    print("date and time =", date_and_time)

    if CUDA_VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    if JAX_ENABLE_X64:
        os.environ["JAX_ENABLE_X64"] = JAX_ENABLE_X64

    import jax
    from jax.lib import xla_bridge

    if not seed:
        seed = random.randint(0, 2 ** 31)
    key = jax.random.PRNGKey(seed)

    n_devices = xla_bridge.device_count()
    print(f'Number of GPU devices: {n_devices} \n')

    assert N_train % n_devices == 0, "The number of training examples should be " \
                                     "divisible by the number of devices"
    assert N_inference % n_devices == 0, "The number of inference examples should be " \
                                         "divisible by the number of devices"

    N_train_device     = int(N_train / n_devices)
    N_inference_device = int(N_inference / n_devices)

    specs = {
        'experiment name': run_name,
        'seed': seed,
        'date and time': date_and_time,
        'number of gpu devices': n_devices,
        'JAX_ENABLE_X64': JAX_ENABLE_X64,
    }
    specs.update({'----- Learning Parameters': '----- '})
    specs.update(learning_params)
    specs.update({'----- Loss Parameters': '----- '})
    specs.update(loss_params)
    specs.update({'----- Optimizer Parameters': '----- '})
    specs.update(optimizer_params)
    specs.update({'----- Interaction Parameters': '----- '})
    specs.update(interaction_params)
    specs.update({'----- Projection Parameters': '----- '})
    specs.update(projection_params)

    logs_dir = os.path.join(LOGS_DIR, run_name)
    if os.path.exists(logs_dir):
        shutil.rmtree(logs_dir)
    os.makedirs(logs_dir, exist_ok=True)

    specs_file = os.path.join(logs_dir, 'data_specs.txt')
    with open(specs_file, 'w') as f:
        for k, v in specs.items():
            f.write(f"{k}: {str(v)}\n")

    key, interaction_key = jax.random.split(key)
    interaction = Interaction(
        pump_basis=pump_basis,
        pump_max_mode1=pump_max_mode1,
        pump_max_mode2=pump_max_mode2,
        initial_pump_coefficient=initial_pump_coefficient,
        custom_pump_coefficient=custom_pump_coefficient,
        pump_coefficient_path=pump_coefficient_path,
        initial_pump_waist=initial_pump_waist,
        pump_waists_path=pump_waists_path,
        crystal_basis=crystal_basis,
        crystal_max_mode1=crystal_max_mode1,
        crystal_max_mode2=crystal_max_mode2,
        initial_crystal_coefficient=initial_crystal_coefficient,
        custom_crystal_coefficient=custom_crystal_coefficient,
        crystal_coefficient_path=crystal_coefficient_path,
        initial_crystal_waist=initial_crystal_waist,
        crystal_waists_path=crystal_waists_path,
        lam_pump=lam_pump,
        crystal_str=crystal_str,
        power_pump=power_pump,
        waist_pump0=waist_pump0,
        r_scale0=r_scale0,
        dx=dx,
        dy=dy,
        dz=dz,
        maxX=maxX,
        maxY=maxY,
        maxZ=maxZ,
        R=R,
        Temperature=Temperature,
        pump_polarization=pump_polarization,
        signal_polarization=signal_polarization,
        idler_polarization=idler_polarization,
        dk_offset=dk_offset,
        power_signal=power_signal,
        power_idler=power_idler,
        key=interaction_key,
    )

    projection_coincidence_rate = Projection_coincidence_rate(
        calculate_observable=observable_vec[COINCIDENCE_RATE],
        waist_pump0=interaction.waist_pump0,
        signal_wavelength=interaction.lam_signal,
        crystal_x=interaction.x,
        crystal_y=interaction.y,
        temperature=interaction.Temperature,
        ctype=interaction.ctype,
        polarization=coincidence_projection_polarization,
        z=coincidence_projection_z,
        projection_basis=coincidence_projection_basis,
        max_mode1=coincidence_projection_max_mode1,
        max_mode2=coincidence_projection_max_mode2,
        waist=coincidence_projection_waist,
        wavelength=coincidence_projection_wavelength,
        tau=tau
    )

    projection_tomography_matrix = Projection_tomography_matrix(
        calculate_observable=observable_vec[DENSITY_MATRIX] or observable_vec[TOMOGRAPHY_MATRIX],
        waist_pump0=interaction.waist_pump0,
        signal_wavelength=interaction.lam_signal,
        crystal_x=interaction.x,
        crystal_y=interaction.y,
        temperature=interaction.Temperature,
        ctype=interaction.ctype,
        polarization=tomography_projection_polarization,
        z=tomography_projection_z,
        relative_phase=tomography_relative_phase,
        tomography_quantum_state=tomography_quantum_state,
        projection_basis=tomography_projection_basis,
        max_mode1=tomography_projection_max_mode1,
        max_mode2=tomography_projection_max_mode2,
        waist=tomography_projection_waist,
        wavelength=tomography_projection_wavelength,
        tau=tau,
    )
    
    Pump = Beam(lam=interaction.lam_pump,
                ctype=interaction.ctype,
                polarization=interaction.pump_polarization,
                T=interaction.Temperature,
                power=interaction.power_pump)

    Signal = Beam(lam=interaction.lam_signal,
                  ctype=interaction.ctype,
                  polarization=interaction.signal_polarization,
                  T=interaction.Temperature,
                  power=interaction.power_signal)

    Idler = Beam(lam=interaction.lam_idler,
                 ctype=interaction.ctype,
                 polarization=interaction.idler_polarization,
                 T=interaction.Temperature,
                 power=interaction.power_idler)

    trainer = BaseTrainer(
        key=key,
        n_epochs=n_epochs,
        N_train=N_train,
        bs_train=bs_train,
        N_inference=N_inference,
        bs_inference=bs_inference,
        N_train_device=N_train_device,
        N_inference_device=N_inference_device,
        learn_pump_coeffs=learn_pump_coeffs,
        learn_pump_waists=learn_pump_waists,
        learn_crystal_coeffs=learn_crystal_coeffs,
        learn_crystal_waists=learn_crystal_waists,
        keep_best=keep_best,
        n_devices=n_devices,
        projection_coincidence_rate=projection_coincidence_rate,
        projection_tomography_matrix=projection_tomography_matrix,
        interaction=interaction,
        pump=Pump,
        signal=Signal,
        idler=Idler,
        observable_vec=observable_vec,
    )

    if learn_mode:
        trainer.coincidence_rate_loss = Loss(observable_as_target=observable_vec[COINCIDENCE_RATE],
                                             target=os.path.join(target, 'coincidence_rate.npy'),
                                             loss_arr=loss_arr[COINCIDENCE_RATE],
                                             loss_weights=loss_weights[COINCIDENCE_RATE],
                                             reg_observable=reg_observable[COINCIDENCE_RATE],
                                             reg_observable_w=reg_observable_w[COINCIDENCE_RATE],
                                             reg_observable_elements=reg_observable_elements[COINCIDENCE_RATE],
                                             l2_reg=l2_reg)

        trainer.density_matrix_loss = Loss(observable_as_target=observable_vec[DENSITY_MATRIX],
                                           target=os.path.join(target, 'density_matrix.npy'),
                                           loss_arr=loss_arr[DENSITY_MATRIX],
                                           loss_weights=loss_weights[DENSITY_MATRIX],
                                           reg_observable=reg_observable[DENSITY_MATRIX],
                                           reg_observable_w=reg_observable_w[DENSITY_MATRIX],
                                           reg_observable_elements=reg_observable_elements[DENSITY_MATRIX],)

        trainer.tomography_matrix_loss = Loss(observable_as_target=observable_vec[TOMOGRAPHY_MATRIX],
                                           target=os.path.join(target, 'tomography_matrix.npy'),
                                           loss_arr=loss_arr[TOMOGRAPHY_MATRIX],
                                           loss_weights=loss_weights[TOMOGRAPHY_MATRIX],
                                           reg_observable=reg_observable[TOMOGRAPHY_MATRIX],
                                           reg_observable_w=reg_observable_w[TOMOGRAPHY_MATRIX],
                                           reg_observable_elements=reg_observable_elements[TOMOGRAPHY_MATRIX],)

        trainer.opt_init, trainer.opt_update, trainer.get_params = Optimizer(optimizer=optimizer,
                                                                             exp_decay_lr=exp_decay_lr,
                                                                             step_size=step_size,
                                                                             decay_steps=decay_steps,
                                                                             decay_rate=decay_rate)

        if trainer.coincidence_rate_loss.target_str and observable_vec[COINCIDENCE_RATE]:
            trainer.target_coincidence_rate = pmap(lambda x:
                                                   np.load(os.path.join(
                                                       DATA_DIR, 'targets', trainer.coincidence_rate_loss.target_str
                                                   )))(np.arange(n_devices))

        if trainer.density_matrix_loss.target_str and observable_vec[DENSITY_MATRIX]:
            trainer.target_density_matrix = pmap(lambda x:
                                                 np.load(os.path.join(
                                                     DATA_DIR, 'targets', trainer.density_matrix_loss.target_str
                                                 )))(np.arange(n_devices))

        if trainer.tomography_matrix_loss.target_str and observable_vec[TOMOGRAPHY_MATRIX]:
            trainer.target_tomography_matrix = pmap(lambda x:
                                                    np.load(os.path.join(
                                                        DATA_DIR, 'targets', trainer.tomography_matrix_loss.target_str
                                                    )))(np.arange(n_devices))

        fit_results = trainer.fit()
        save_training_statistics(
            logs_dir,
            fit_results,
            interaction,
            trainer.model_parameters,
        )

        observables = trainer.inference()

        results_dir = os.path.join(RES_DIR, run_name)
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        save_results(
            results_dir,
            observable_vec,
            observables,
            projection_coincidence_rate,
            projection_tomography_matrix,
            Signal,
            Idler,
        )

    else:
        observables = trainer.inference()

        results_dir = os.path.join(RES_DIR, run_name)
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        specs_file = os.path.join(results_dir, 'data_specs.txt')
        with open(specs_file, 'w') as f:
            for k, v in specs.items():
                f.write(f"{k}: {str(v)}\n")

        save_results(
            results_dir,
            observable_vec,
            observables,
            projection_coincidence_rate,
            projection_tomography_matrix,
            Signal,
            Idler,
        )


if __name__ == "__main__":

    learning_params = {
        'learn_mode': True,
        'learn_pump_coeffs': True,
        'learn_pump_waists':  True,
        'learn_crystal_coeffs':  True,
        'learn_crystal_waists':  True,
        'keep_best':  True,
        'n_epochs':  500,
        'N_train':  1000,
        'bs_train':  1000,
        'N_inference':  1000,
        'bs_inference':  1000,
        'target': 'qutrit',
        'observable_vec': {
            COINCIDENCE_RATE: True,
            DENSITY_MATRIX: False,
            TOMOGRAPHY_MATRIX: False
        }
    }

    loss_params = {
        'loss_arr': {
            COINCIDENCE_RATE: ('l1', 'l2'),
            DENSITY_MATRIX: None,
            TOMOGRAPHY_MATRIX: None
        },
        'loss_weights': {
            COINCIDENCE_RATE: (1., .5),
            DENSITY_MATRIX: None,
            TOMOGRAPHY_MATRIX: None
        },
        'reg_observable': {
            COINCIDENCE_RATE: ('sparsify', 'equalize'),
            DENSITY_MATRIX: None,
            TOMOGRAPHY_MATRIX: None
        },
        'reg_observable_w': {
            COINCIDENCE_RATE: (.5, .5),
            DENSITY_MATRIX: None,
            TOMOGRAPHY_MATRIX: None
        },
        'reg_observable_elements': {
            COINCIDENCE_RATE: ([20, 30, 40, 50, 60], [20, 30, 40, 50, 60]),
            DENSITY_MATRIX: None,
            TOMOGRAPHY_MATRIX: None
        },
        'tau': 1e-9,
    }

    optimizer_params = {
        'l2_reg': 0.,
        'optimizer': 'adam',
        'exp_decay_lr': True,
        'step_size': 0.05,
        'decay_steps': 50,
        'decay_rate': 0.5,
    }

    interaction_params = {
        'pump_basis': 'LG',
        'pump_max_mode1': 5,
        'pump_max_mode2': 1,
        'initial_pump_coefficient': 'LG00',
        'custom_pump_coefficient': {REAL: {-1: 1, 0: 1, 1: 1}, IMAG: {-1: 1, 0: 1, 1: 1}},
        'pump_coefficient_path': None,
        'initial_pump_waist': 'waist_pump0',
        'pump_waists_path': None,
        'crystal_basis': 'LG',
        'crystal_max_mode1': 5,
        'crystal_max_mode2': 2,
        'initial_crystal_coefficient': 'LG00',
        'custom_crystal_coefficient': {REAL: {-1: 1, 0: 1, 1: 1}, IMAG: {-1: 1, 0: 1, 1: 1}},
        'crystal_coefficient_path': None,
        'initial_crystal_waist': 'r_scale0',
        'crystal_waists_path': None,
        'lam_pump': 405e-9,
        'crystal_str': 'ktp',
        'power_pump': 1e-3,
        'waist_pump0': 40e-6,
        'r_scale0': 40e-6,
        'dx': 4e-6,
        'dy': 4e-6,
        'dz': 10e-6,
        'maxX': 120e-6,
        'maxY': 120e-6,
        'maxZ': 1e-4,
        'R': 0.1,
        'Temperature': 50,
        'pump_polarization': 'y',
        'signal_polarization': 'y',
        'idler_polarization': 'z',
        'dk_offset': 1.,
        'power_signal': 1.,
        'power_idler': 1.,
    }

    projection_params = {
        'coincidence_projection_basis': 'LG',
        'coincidence_projection_max_mode1': 1,
        'coincidence_projection_max_mode2': 4,
        'coincidence_projection_waist': None,
        'coincidence_projection_wavelength': None,
        'coincidence_projection_polarization': 'y',
        'coincidence_projection_z': 0.,
        'tomography_projection_basis': 'LG',
        'tomography_projection_max_mode1': 1,
        'tomography_projection_max_mode2': 1,
        'tomography_projection_waist': None,
        'tomography_projection_wavelength': None,
        'tomography_projection_polarization': 'y',
        'tomography_projection_z': 0.,
        'tomography_relative_phase': [0, np.pi, 3 * (np.pi / 2), np.pi / 2],
        'tomography_quantum_state': 'qubit',
    }


    run_experiment(
        run_name='test',
        seed=42,
        JAX_ENABLE_X64='True',
        CUDA_VISIBLE_DEVICES='0, 1',
        **learning_params,
        **loss_params,
        **optimizer_params,
        **interaction_params,
        **projection_params,
    )
