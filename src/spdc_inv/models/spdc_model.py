from abc import ABC
import jax.random as random
import jax.numpy as np
from spdc_inv.models.utils import Field
from spdc_inv.models.utils import crystal_prop, propagate
from spdc_inv.training.utils import coincidence_rate_calc, decompose, fix_power


class SPDCmodel(ABC):
    """
    A differentiable SPDC forward model
    """

    def __init__(
            self,
            pump,
            signal,
            idler,
            projection,
            interaction,
            pump_structure,
            crystal_hologram,
            poling_period,
            DeltaZ,
            coincidence_rate_observable,
            density_matrix_observable,
    ):

        self.pump = pump
        self.signal = signal
        self.idler = idler
        self.projection = projection
        self.interaction = interaction
        self.pump_structure = pump_structure
        self.crystal_hologram = crystal_hologram
        self.poling_period = poling_period
        self.DeltaZ = DeltaZ
        self.coincidence_rate_observable = coincidence_rate_observable
        self.density_matrix_observable = density_matrix_observable
        self.N = None
        self.N_device = None
        self.learn_mode = None
        self.bs = None

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

        # initialize the vacuum and interaction fields
        vacuum_states = random.normal(
            rand_key,
            (self.N_device, 2, 2, self.interaction.Nx, self.interaction.Ny)
        )

        self.pump_structure.create_profile(pump_coeffs_real, pump_coeffs_imag, waist_pump)
        if self.crystal_hologram is not None:
            self.crystal_hologram.create_profile(crystal_coeffs_real, crystal_coeffs_imag, r_scale)

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

        observables = self.get_observables(signal_out, idler_out, idler_vac)

        return observables

    def get_observables(
            self,
            signal_out,
            idler_out,
            idler_vac
    ):
        """
        Parameters
        ----------
        coincidence_rate: coincidence rate matrix
        density_matrix: Density matrix
        Returns The function returns the desired observables
        -------
        """

        signal_beam_decompose = decompose(signal_out,
                                          self.projection.basis_arr
                                          ).reshape(
            self.N_device, self.projection.projection_n_modes1, self.projection.projection_n_modes2)

        idler_beam_decompose  = decompose(idler_out,
                                          self.projection.basis_arr
                                          ).reshape(
            self.N_device, self.projection.projection_n_modes1, self.projection.projection_n_modes2)

        idler_vac_decompose   = decompose(idler_vac,
                                          self.projection.basis_arr
                                          ).reshape(
            self.N_device, self.projection.projection_n_modes1, self.projection.projection_n_modes2)

        # say there are no higher modes by normalizing the power
        signal_beam_decompose = fix_power(signal_beam_decompose, signal_out)
        idler_beam_decompose  = fix_power(idler_beam_decompose, idler_out)
        idler_vac_decompose   = fix_power(idler_vac_decompose, idler_vac)

        coincidence_rate, density_matrix = None, None
        if self.coincidence_rate_observable:

            coincidence_rate = coincidence_rate_calc(
                signal_beam_decompose,
                idler_beam_decompose,
                idler_vac_decompose,
                self.N
            ).reshape(self.projection.projection_n_modes1 ** 2, self.projection.projection_n_modes2 ** 2)

        if self.density_matrix_observable:
            density_matrix = 'should be implemented'

        return coincidence_rate, density_matrix

