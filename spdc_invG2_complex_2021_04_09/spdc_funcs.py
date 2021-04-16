import jax.numpy as np
from jax import jit
from spdc_helper_parallel_complex import kron
from jax.lib import xla_bridge


def calc_and_asserts(N, batch_size):
    num_devices = xla_bridge.device_count()
    num_batches = int(N / batch_size)
    Ndevice = int(N / num_devices)
    batch_device = int(batch_size / num_devices)

    assert N % batch_size == 0, "num_batches should be 'signed integer'"
    assert N % num_devices == 0, "The number of examples should be divisible by the number of devices"
    assert batch_size % num_devices == 0, "The number of examples within a batch should be divisible by the number of devices"

    print(f'Number of GPU devices: {num_devices} \n')

    return num_batches, Ndevice, batch_device, num_devices

@jit
def G2_calc(a, b, c, batch_size):
    G1_ss        = kron(np.conj(a), a) / batch_size
    G1_ii        = kron(np.conj(b), b) / batch_size
    G1_si        = kron(np.conj(b), a) / batch_size
    G1_si_dagger = kron(np.conj(a), b) / batch_size
    Q_si         = kron(c, a) / batch_size
    Q_si_dagger  = kron(np.conj(a), np.conj(c)) / batch_size

    return (G1_ii * G1_ss + Q_si_dagger * Q_si + G1_si_dagger * G1_si).real


def Pss_calc(a, Nx, Ny, M, batch_size):
    return (kron(np.conj(a), a).real.reshape(Nx**2, Ny**2))[::M + 1, ::M + 1] / batch_size