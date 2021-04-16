import jax.numpy as np
from jax import jit
from spdc_helper_parallel_complex import kron, kron1
from jax.lib import xla_bridge


def calc_and_asserts(N):
    num_devices  = xla_bridge.device_count()
    batch_device = int(N / num_devices)

    assert N % num_devices == 0, "The number of examples should be divisible by the number of devices"

    print(f'Number of GPU devices: {num_devices} \n')

    return batch_device, num_devices

@jit
def G2_calc(a, b, c, batch_size):
    G1_ss        = kron(np.conj(a), a) / batch_size
    G1_ii        = kron(np.conj(b), b) / batch_size
    G1_si        = kron(np.conj(b), a) / batch_size
    G1_si_dagger = kron(np.conj(a), b) / batch_size
    Q_si         = kron(c, a) / batch_size
    Q_si_dagger  = kron(np.conj(a), np.conj(c)) / batch_size

    return (G1_ii * G1_ss + Q_si_dagger * Q_si + G1_si_dagger * G1_si).real


@jit
def G2_calc_batch(a, b, c, N,
                  G1_ii_p, G1_ss_p, Q_si_dagger_p, Q_si_p, G1_si_dagger_p, G1_si_p):
    G1_ss        = kron1(np.conj(a), a) / N + G1_ss_p
    G1_ii        = kron1(np.conj(b), b) / N + G1_ii_p
    G1_si        = kron1(np.conj(b), a) / N + G1_si_p
    G1_si_dagger = kron1(np.conj(a), b) / N + G1_si_dagger_p
    Q_si         = kron1(c, a) / N + Q_si_p
    Q_si_dagger  = kron1(np.conj(a), np.conj(c)) / N + Q_si_dagger_p
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
