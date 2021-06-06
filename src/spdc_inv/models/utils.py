from abc import ABC
from spdc_inv.utils.utils import h_bar, eps0, c

import jax.numpy as np


class Field(ABC):
    """
    A class that holds everything to do with the interaction values of a given beam
    vac   - corresponding vacuum state coefficient
    kappa - coupling constant
    k     - wave vector
    """
    def __init__(
            self,
            beam,
            dx,
            dy,
            maxZ
    ):
        """

        Parameters
        ----------
        beam: A class that holds everything to do with a beam
        dx: transverse resolution in x [m]
        dy: transverse resolution in y [m]
        maxZ: Crystal's length in z [m]
        """

        self.vac   = np.sqrt(h_bar * beam.w / (2 * eps0 * beam.n ** 2 * dx * dy * maxZ))
        self.kappa = 2 * 1j * beam.w ** 2 / (beam.k * c ** 2)
        self.k     = beam.k


def crystal_prop(
        pump_profile,
        pump,
        signal_field,
        idler_field,
        vacuum_states,
        interaction,
        poling_period,
        N,
        crystal_hologram,
        infer,
        signal_init=None,
        idler_init=None
):
    """
    Crystal propagation
    propagate through crystal using split step Fourier for 4 fields: signal, idler and two vacuum states

    Parameters
    ----------
    pump_profile: electromagnetic pump beam profile
    pump: A class that holds everything to do with the pump beam
    signal_field: A class that holds everything to do with the interaction values of the signal beam
    idler_field: A class that holds everything to do with the interaction values of the idler beam
    vacuum_states: The vacuum and interaction fields
    interaction: A class that represents the SPDC interaction process, on all of its physical parameters.
    poling_period: Poling period (dk_offset * delta_k)
    N: number of vacuum_state elements
    crystal_hologram: 3D crystal hologram
    infer: (True/False) if in inference mode, we include more coefficients in the poling
                description for better validation
    signal_init: initial signal profile. If None, initiate to zero
    idler_init: initial idler profile. If None, initiate to zero

    Returns: the interacting fields at the end of interaction medium
    -------

    """

    x  = interaction.x
    y  = interaction.y
    Nx = interaction.Nx
    Ny = interaction.Ny
    dz = interaction.dz


    if signal_init is None:
        signal_out = np.zeros([N, Nx, Ny])
    else:
        assert len(signal_init.shape) == 3
        assert signal_init.shape[0] == N
        assert signal_init.shape[1] == Nx
        assert signal_init.shape[2] == Ny
        signal_out = signal_init

    if idler_init is None:
        idler_out = np.zeros([N, Nx, Ny])
    else:
        assert len(idler_init.shape) == 3
        assert idler_init.shape[0] == N
        assert idler_init.shape[1] == Nx
        assert idler_init.shape[2] == Ny
        idler_out = idler_init


    signal_vac = signal_field.vac * (vacuum_states[:, 0, 0] + 1j * vacuum_states[:, 0, 1]) / np.sqrt(2)
    idler_vac  = idler_field.vac * (vacuum_states[:, 1, 0] + 1j * vacuum_states[:, 1, 1]) / np.sqrt(2)

    for z in interaction.z:
        # pump beam:
        E_pump = propagate(pump_profile, x, y, pump.k, z) * np.exp(-1j * pump.k * z)

        # crystal slab:
        PP     = interaction.slab(poling_period, z, crystal_hologram, infer)

        # coupled wave equations - split step
        # signal:
        dEs_out_dz = signal_field.kappa * interaction.d33 * PP * E_pump * np.conj(idler_vac)
        dEs_vac_dz = signal_field.kappa * interaction.d33 * PP * E_pump * np.conj(idler_out)
        signal_out = signal_out + dEs_out_dz * dz
        signal_vac = signal_vac + dEs_vac_dz * dz

        # idler:
        dEi_out_dz = idler_field.kappa * interaction.d33 * PP * E_pump * np.conj(signal_vac)
        dEi_vac_dz = idler_field.kappa * interaction.d33 * PP * E_pump * np.conj(signal_out)
        idler_out  = idler_out + dEi_out_dz * dz
        idler_vac  = idler_vac + dEi_vac_dz * dz

        # propagate
        signal_out = propagate(signal_out, x, y, signal_field.k, dz) * np.exp(-1j * signal_field.k * dz)
        signal_vac = propagate(signal_vac, x, y, signal_field.k, dz) * np.exp(-1j * signal_field.k * dz)
        idler_out  = propagate(idler_out, x, y, idler_field.k, dz) * np.exp(-1j * idler_field.k * dz)
        idler_vac  = propagate(idler_vac, x, y, idler_field.k, dz) * np.exp(-1j * idler_field.k * dz)
    
    return signal_out, idler_out, idler_vac


def propagate(A, x, y, k, dz):
    """
    Free Space propagation using the free space transfer function,
    (two  dimensional), according to Saleh
    Using CGS, or MKS, Boyd 2nd eddition

    Parameters
    ----------
    A: electromagnetic beam profile
    x,y: spatial vectors
    k: wave vector
    dz: The distance to propagate

    Returns the propagated field
    -------

    """
    dx      = np.abs(x[1] - x[0])
    dy      = np.abs(y[1] - y[0])

    # define the fourier vectors
    X, Y    = np.meshgrid(x, y, indexing='ij')
    KX      = 2 * np.pi * (X / dx) / (np.size(X, 1) * dx)
    KY      = 2 * np.pi * (Y / dy) / (np.size(Y, 1) * dy)

    # The Free space transfer function of propagation, using the Fresnel approximation
    # (from "Engineering optics with matlab"/ing-ChungPoon&TaegeunKim):
    H_w = np.exp(-1j * dz * (np.square(KX) + np.square(KY)) / (2 * k))
    H_w = np.fft.ifftshift(H_w)

    # Fourier Transform: move to k-space
    G = np.fft.fft2(A)  # The two-dimensional discrete Fourier transform (DFT) of A.

    # propoagte in the fourier space
    F = np.multiply(G, H_w)

    # inverse Fourier Transform: go back to real space
    Eout = np.fft.ifft2(F)  # [in real space]. E1 is the two-dimensional INVERSE discrete Fourier transform (DFT) of F1

    return Eout
