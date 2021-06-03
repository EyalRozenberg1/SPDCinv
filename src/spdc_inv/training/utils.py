import jax.numpy as np
from jax import lax
from jax import jit



def project(projection_basis, beam_profile):
    """
    The function projects some state beam_profile onto given projection_basis
    Parameters
    ----------
    projection_basis: array of basis function
    beam_profile: beam profile (2d)

    Returns
    -------

    """
    Nxx2           = beam_profile.shape[1] ** 2
    N              = beam_profile.shape[0]
    Nh             = projection_basis.shape[0]
    projection     = (np.conj(projection_basis) * beam_profile).reshape(Nh, N, Nxx2).sum(2)
    normalization1 = np.abs(beam_profile ** 2).reshape(N, Nxx2).sum(1)
    normalization2 = np.abs(projection_basis ** 2).reshape(Nh, Nxx2).sum(1)
    projection     = projection / np.sqrt(normalization1[None, :] * normalization2[:, None])
    return projection


def decompose(beam_profile, projection_basis_arr):
    """
    Decompose a given beam profile into modes defined in the dictionary
    Parameters
    ----------
    beam_profile: beam profile (2d)
    projection_basis_arr: array of basis function

    Returns: beam profile as a decomposition of basis functions
    -------

    """
    projection = project(projection_basis_arr[:, None], beam_profile)
    return np.transpose(projection)


def fix_power(decomposed_profile, beam_profile):
    """
    Normalize power and ignore higher modes
    Parameters
    ----------
    decomposed_profile: the decomposed beam profile
    beam_profile: the original beam profile

    Returns a normalized decomposed profile
    -------

    """
    scale = np.sqrt(
        np.sum(beam_profile * np.conj(beam_profile), (1, 2))) / np.sqrt(
        np.sum(decomposed_profile * np.conj(decomposed_profile), (1, 2)))

    return decomposed_profile * scale[:, None, None]


def kron(a, b, multiple_devices: bool = True):
    """
    Calculates the kronecker product between two 2d tensors
    Parameters
    ----------
    a, b: 2d tensors
    multiple_devices: (True/False) whether multiple devices are used

    Returns the kronecker product
    -------

    """
    if multiple_devices:
        return lax.psum((a[:, :, None, :, None] * b[:, None, :, None, :]).sum(0), 'device')

    else:
        return (a[:, :, None, :, None] * b[:, None, :, None, :]).sum(0)


@jit
def coincidence_rate_calc(a, b, c, batch_size):
    """

    Parameters
    ----------
    a, b, c: the interacting fields
    batch_size: number of interacting vacuum state elements

    Returns the coincidence matrix
    -------

    """
    G1_ss        = kron(np.conj(a), a) / batch_size
    G1_ii        = kron(np.conj(b), b) / batch_size
    G1_si        = kron(np.conj(b), a) / batch_size
    G1_si_dagger = kron(np.conj(a), b) / batch_size
    Q_si         = kron(c, a) / batch_size
    Q_si_dagger  = kron(np.conj(a), np.conj(c)) / batch_size

    return (G1_ii * G1_ss + Q_si_dagger * Q_si + G1_si_dagger * G1_si).real


@jit
def coincidence_rate_calc_batch(a, b, c, N, G1_ii_p, G1_ss_p, Q_si_dagger_p, Q_si_p, G1_si_dagger_p, G1_si_p):
    G1_ss        = kron(np.conj(a), a, multiple_devices=False) / N + G1_ss_p
    G1_ii        = kron(np.conj(b), b, multiple_devices=False) / N + G1_ii_p
    G1_si        = kron(np.conj(b), a, multiple_devices=False) / N + G1_si_p
    G1_si_dagger = kron(np.conj(a), b, multiple_devices=False) / N + G1_si_dagger_p
    Q_si         = kron(c, a, multiple_devices=False) / N + Q_si_p
    Q_si_dagger  = kron(np.conj(a), np.conj(c), multiple_devices=False) / N + Q_si_dagger_p
    return G1_ii, G1_ss, Q_si_dagger, Q_si, G1_si_dagger, G1_si


def Pss_calc(a, Nx, Ny, M, batch_size):
    return (kron(np.conj(a), a).real.reshape(Nx**2, Ny**2))[::M + 1, ::M + 1] / batch_size


def init_corr_mats(n_coeff):
    return np.zeros((1, 1, n_coeff, n_coeff), dtype=np.complex64), \
           np.zeros((1, 1, n_coeff, n_coeff), dtype=np.complex64), \
           np.zeros((1, 1, n_coeff, n_coeff), dtype=np.complex64), \
           np.zeros((1, 1, n_coeff, n_coeff), dtype=np.complex64), \
           np.zeros((1, 1, n_coeff, n_coeff), dtype=np.complex64), \
           np.zeros((1, 1, n_coeff, n_coeff), dtype=np.complex64)