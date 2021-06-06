from abc import ABC
from jax import numpy as np
from typing import Tuple, Dict, Any, List, Union
from spdc_inv.utils.utils import HermiteBank, LaguerreBank, TomographyBank
from spdc_inv.utils.defaults import QUBIT, QUTRIT


class Projection_coincidence_rate(ABC):
    """
    A class that represents the projective basis for
    calculating the coincidence rate observable of the interaction.
    """

    def __init__(
            self,
            calculate_observable: Tuple[Dict[Any, bool], ...],
            waist_pump0: float,
            signal_wavelength: float,
            crystal_x: np.array,
            crystal_y: np.array,
            temperature: float,
            ctype,
            polarization: str,
            z: float = 0.,
            projection_basis: str = 'LG',
            max_mode1: int = 1,
            max_mode2: int = 4,
            waist: float = None,
            wavelength:  float = None,
            tau: float = 1e-9,

    ):
        """

        Parameters
        ----------
        calculate_observable: True/False, will the observable be calculated in simulation
        waist_pump0: pump waists at the center of the crystal (initial-before training)
        signal_wavelength: signal wavelength at spdc interaction
        crystal_x: x axis linspace array (transverse)
        crystal_y: y axis linspace array (transverse)
        temperature: interaction temperature
        ctype: refractive index function
        polarization: polarization for calculating effective refractive index
        z: projection longitudinal position
        projection_basis: type of projection basis
                          Can be: LG (Laguerre-Gauss) / HG (Hermite-Gauss)
        max_mode1: Maximum value of first mode of the 2D projection basis
        max_mode2: Maximum value of second mode of the 2D projection basis
        waist: waists of the projection basis functions
        wavelength: wavelength for generating projection basis
        tau: coincidence window [nano sec]
        """

        self.tau = tau

        if waist is None:
            self.waist = np.sqrt(2) * waist_pump0
        else:
            self.waist = waist

        if wavelength is None:
            wavelength = signal_wavelength

        assert projection_basis.lower() in ['lg', 'hg'], 'The projection basis is LG or HG ' \
                                                         'basis functions only'

        self.projection_basis = projection_basis
        self.max_mode1 = max_mode1
        self.max_mode2 = max_mode2

        # number of modes for projection basis
        if projection_basis.lower() == 'lg':
            self.projection_n_modes1 = max_mode1
            self.projection_n_modes2 = 2 * max_mode2 + 1
        else:
            self.projection_n_modes1 = max_mode1
            self.projection_n_modes2 = max_mode2

        # Total number of projection modes
        self.projection_n_modes = self.projection_n_modes1 * self.projection_n_modes2

        refractive_index = ctype(wavelength * 1e6, temperature, polarization)
        [x, y] = np.meshgrid(crystal_x, crystal_y)

        if calculate_observable:
            if projection_basis.lower() == 'lg':
                self.basis_arr, self.basis_str = \
                    LaguerreBank(
                        wavelength,
                        refractive_index,
                        self.waist,
                        self.max_mode1,
                        self.max_mode2,
                        x, y, z)
            else:
                self.basis_arr, self.basis_str = \
                    HermiteBank(
                        wavelength,
                        refractive_index,
                        self.waist,
                        self.max_mode1,
                        self.max_mode2,
                        x, y, z)


class Projection_tomography_matrix(ABC):
    """
    A class that represents the projective basis for
    calculating the density matrix observable of the interaction.
    """

    def __init__(
            self,
            calculate_observable: Tuple[Dict[Any, bool], ...],
            waist_pump0: float,
            signal_wavelength: float,
            crystal_x: np.array,
            crystal_y: np.array,
            temperature: float,
            ctype,
            polarization: str,
            z: float = 0.,
            projection_basis: str = 'LG',
            max_mode1: int = 1,
            max_mode2: int = 1,
            waist: float = None,
            wavelength:  float = None,
            tau: float = 1e-9,
            relative_phase: List[Union[Union[int, float], Any]] = None,
            tomography_quantum_state: str = None,

    ):
        """

        Parameters
        ----------
        calculate_observable: True/False, will the observable be calculated in simulation
        waist_pump0: pump waists at the center of the crystal (initial-before training)
        signal_wavelength: signal wavelength at spdc interaction
        crystal_x: x axis linspace array (transverse)
        crystal_y: y axis linspace array (transverse)
        temperature: interaction temperature
        ctype: refractive index function
        polarization: polarization for calculating effective refractive index
        z: projection longitudinal position
        projection_basis: type of projection basis
                          Can be: LG (Laguerre-Gauss)
        max_mode1: Maximum value of first mode of the 2D projection basis
        max_mode2: Maximum value of second mode of the 2D projection basis
        waist: waists of the projection basis functions
        wavelength: wavelength for generating projection basis
        tau: coincidence window [nano sec]
        relative_phase: The relative phase between the mutually unbiased bases (MUBs) states
        tomography_quantum_state: the current quantum state we calculate it tomography matrix.
                                  currently we support: qubit/qutrit
        """

        self.tau = tau

        if waist is None:
            self.waist = np.sqrt(2) * waist_pump0
        else:
            self.waist = waist

        if wavelength is None:
            wavelength = signal_wavelength

        assert projection_basis.lower() in ['lg'], 'The projection basis is LG ' \
                                                   'basis functions only'

        assert max_mode1 == 1, 'for Tomography projections, max_mode1 must be 1'
        assert max_mode2 == 1, 'for Tomography projections, max_mode2 must be 1'

        self.projection_basis = projection_basis
        self.max_mode1 = max_mode1
        self.max_mode2 = max_mode2

        assert tomography_quantum_state in [QUBIT, QUTRIT], f'quantum state must be {QUBIT} or {QUTRIT}, ' \
                                                            'but received {tomography_quantum_state}'
        self.tomography_quantum_state = tomography_quantum_state
        self.relative_phase = relative_phase

        self.projection_n_state1 = 1
        if self.tomography_quantum_state is QUBIT:
            self.projection_n_state2 = 6
        else:
            self.projection_n_state2 = 15

        refractive_index = ctype(wavelength * 1e6, temperature, polarization)
        [x, y] = np.meshgrid(crystal_x, crystal_y)

        if calculate_observable:
            self.basis_arr, self.basis_str = \
                TomographyBank(
                    wavelength,
                    refractive_index,
                    self.waist,
                    self.max_mode1,
                    self.max_mode2,
                    x, y, z,
                    self.relative_phase,
                    self.tomography_quantum_state
                )
