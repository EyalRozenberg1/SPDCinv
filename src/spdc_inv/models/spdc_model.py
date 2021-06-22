from abc import ABC
import jax.random as random
import jax.numpy as np
import operator
from spdc_inv.models.utils import Field
from spdc_inv.models.utils import crystal_prop, propagate
from spdc_inv.training.utils import projection_matrix_calc, projection_matrices_calc, \
    decompose, fix_power, get_density_matrix


class SPDCmodel(ABC):
    """
    A differentiable SPDC forward model
    """

    def __init__(
            self,
            pump,
            signal,
            idler,
            projection_coincidence_rate,
            projection_tomography_matrix,
            interaction,
            pump_structure,
            crystal_hologram,
            poling_period,
            DeltaZ,
            coincidence_rate_observable,
            density_matrix_observable,
            tomography_matrix_observable,
    ):

        self.pump = pump
        self.signal = signal
        self.idler = idler
        self.projection_coincidence_rate = projection_coincidence_rate
        self.projection_tomography_matrix = projection_tomography_matrix
        self.interaction = interaction
        self.pump_structure = pump_structure
        self.crystal_hologram = crystal_hologram
        self.poling_period = poling_period
        self.DeltaZ = DeltaZ
        self.coincidence_rate_observable = coincidence_rate_observable
        self.density_matrix_observable = density_matrix_observable
        self.tomography_matrix_observable = tomography_matrix_observable
        self.N = None
        self.N_device = None
        self.learn_mode = None

        self.signal_f = Field(signal, interaction.dx, interaction.dy, interaction.maxZ)
        self.idler_f  = Field(idler, interaction.dx, interaction.dy, interaction.maxZ)


    def forward(
            self,
            model_parameters,
            rand_key
    ):
        pump_coeffs_real, \
        pump_coeffs_imag, \
        waist_pump, \
        crystal_coeffs_real, \
        crystal_coeffs_imag, \
        r_scale = model_parameters

        self.pump_structure.create_profile(pump_coeffs_real, pump_coeffs_imag, waist_pump)
        if self.crystal_hologram is not None:
            self.crystal_hologram.create_profile(crystal_coeffs_real, crystal_coeffs_imag, r_scale)

        rand_key, subkey = random.split(rand_key)
        # initialize the vacuum and interaction fields
        vacuum_states = random.normal(
            subkey,
            (self.N_device, 2, 2, self.interaction.Nx, self.interaction.Ny)
        )
        signal_out,\
        idler_out,\
        idler_vac \
            = crystal_prop(self.pump_structure.E,
                           self.pump,
                           self.signal_f,
                           self.idler_f,
                           vacuum_states,
                           self.interaction,
                           self.poling_period,
                           self.N_device,
                           None if self.crystal_hologram is None else self.crystal_hologram.crystal_profile,
                           not self.learn_mode,
                           signal_init=None,
                           idler_init=None
                           )

        # Propagate generated fields back to the middle of the crystal
        signal_out = propagate(signal_out,
                               self.interaction.x,
                               self.interaction.y,
                               self.signal_f.k,
                               self.DeltaZ
                               ) * np.exp(-1j * self.signal_f.k * self.DeltaZ)

        idler_out  = propagate(idler_out,
                               self.interaction.x,
                               self.interaction.y,
                               self.idler_f.k,
                               self.DeltaZ
                               ) * np.exp(-1j * self.idler_f.k * self.DeltaZ)

        idler_vac  = propagate(idler_vac,
                               self.interaction.x,
                               self.interaction.y,
                               self.idler_f.k,
                               self.DeltaZ
                               ) * np.exp(-1j * self.idler_f.k * self.DeltaZ)

        coincidence_rate_projections, tomography_matrix_projections = \
            self.get_1st_order_projections(
                signal_out,
                idler_out,
                idler_vac,
            )

        observables = self.get_observables(coincidence_rate_projections, tomography_matrix_projections)

        return observables

    def get_1st_order_projections(
            self,
            signal_out,
            idler_out,
            idler_vac
    ):
        """
        the function calculates first order correlation functions.
            According to  https://doi.org/10.1002/lpor.201900321

        Parameters
        ----------
        signal_out: the signal at the end of interaction
        idler_out: the idler at the end of interaction
        idler_vac: the idler vacuum state at the end of interaction

        Returns: first order correlation functions according to  https://doi.org/10.1002/lpor.201900321
        -------

        """

        coincidence_rate_projections, tomography_matrix_projections = None, None
        if self.coincidence_rate_observable:
            coincidence_rate_projections = self.decompose_and_get_projections(
                signal_out,
                idler_out,
                idler_vac,
                self.projection_coincidence_rate.basis_arr,
                self.projection_coincidence_rate.projection_n_modes1,
                self.projection_coincidence_rate.projection_n_modes2
            )

        if self.tomography_matrix_observable or self.density_matrix_observable:
            tomography_matrix_projections = self.decompose_and_get_projections(
                signal_out,
                idler_out,
                idler_vac,
                self.projection_tomography_matrix.basis_arr,
                self.projection_tomography_matrix.projection_n_state1,
                self.projection_tomography_matrix.projection_n_state2
            )

        return coincidence_rate_projections, tomography_matrix_projections

    def decompose_and_get_projections(
            self,
            signal_out,
            idler_out,
            idler_vac,
            basis_arr,
            projection_n_1,
            projection_n_2
    ):
        """
        The function decompose the interacting fields onto selected basis array, and calculates first order
            correlation functions according to  https://doi.org/10.1002/lpor.201900321

        Parameters
        ----------
        signal_out
        idler_out
        idler_vac
        basis_arr
        projection_n_1
        projection_n_2

        Returns
        -------

        """

        signal_beam_decompose, idler_beam_decompose, idler_vac_decompose = \
            self.decompose(
                signal_out,
                idler_out,
                idler_vac,
                basis_arr,
                projection_n_1,
                projection_n_2
            )

        G1_ss, G1_ii, G1_si, G1_si_dagger, Q_si, Q_si_dagger = projection_matrices_calc(
            signal_beam_decompose,
            idler_beam_decompose,
            idler_vac_decompose,
            self.N
        )
        
        return G1_ss, G1_ii, G1_si, G1_si_dagger, Q_si, Q_si_dagger

    def decompose(
            self,
            signal_out,
            idler_out,
            idler_vac,
            basis_arr,
            projection_n_1,
            projection_n_2
    ):

        signal_beam_decompose = decompose(
            signal_out,
            basis_arr
        ).reshape(
            self.N_device,
            projection_n_1,
            projection_n_2)

        idler_beam_decompose = decompose(
            idler_out,
            basis_arr
        ).reshape(
            self.N_device,
            projection_n_1,
            projection_n_2)

        idler_vac_decompose = decompose(
            idler_vac,
            basis_arr
        ).reshape(
            self.N_device,
            projection_n_1,
            projection_n_2)

        # say there are no higher modes by normalizing the power
        signal_beam_decompose = fix_power(signal_beam_decompose, signal_out)
        idler_beam_decompose = fix_power(idler_beam_decompose, idler_out)
        idler_vac_decompose = fix_power(idler_vac_decompose, idler_vac)

        return signal_beam_decompose, idler_beam_decompose, idler_vac_decompose

    def get_observables(
            self,
            coincidence_rate_projections,
            tomography_matrix_projections,

    ):
        coincidence_rate, density_matrix, tomography_matrix = None, None, None

        if self.coincidence_rate_observable:
            coincidence_rate = projection_matrix_calc(
                *coincidence_rate_projections
            ).reshape(
                self.projection_coincidence_rate.projection_n_modes1 ** 2,
                self.projection_coincidence_rate.projection_n_modes2 ** 2
            )

        if self.tomography_matrix_observable or self.density_matrix_observable:
            tomography_matrix = projection_matrix_calc(
                *tomography_matrix_projections
            ).reshape(
                self.projection_tomography_matrix.projection_n_state1 ** 2,
                self.projection_tomography_matrix.projection_n_state2 ** 2)

            if self.density_matrix_observable:
                density_matrix = get_density_matrix(tomography_matrix)

        return coincidence_rate, density_matrix, tomography_matrix

