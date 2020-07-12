from __future__ import print_function, division, absolute_import
from jax import value_and_grad, pmap, lax
from jax.numpy import kron
from jax.numpy import linalg as la
from helper import *
from jax.experimental import optimizers
from jax.lib import xla_bridge
from functools import partial
import jax.random as random
import os, time
import jax.numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
num_devices = xla_bridge.device_count()

path       = 'targets/'  # path to targets folder

"Hyperparameters"
num_epochs  = 3
batch_size  = 2
N           = 4
num_batches = int(N/batch_size)
Ndevice     = int(N/num_devices)
batch_device= int(batch_size/num_devices)

assert N % batch_size == 0, "num_batches should be 'signed integer'"
assert N % num_devices == 0, "The number of examples should be divisible by the number of devices"
assert batch_size % num_devices == 0, "The number of examples within a batch should be divisible by the number of devices"

psl      = Cr(100e-6, 100e-6, 1e-5, 200e-6, 200e-6, 5e-3)
M        = len(psl.x)
Am       = Bm(532e-9, psl, 50, 100e-6, 0.03, 4)
Sm       = Bm(1064e-9, psl, 50)
Dm       = Bm(sfg_wv(Am.glr, Sm.glr), psl, 50)

mok = Am.gg - Sm.gg - Dm.gg
psl.pl_pr = 1.003 * mok

Nx = len(psl.x)
Ny = len(psl.y)

params = np.array([1.0, 1.0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 1.0, 0, 1.0, 0, 0, 1.0])
params = np.divide(params, la.norm(params))
params = pmap(lambda x: params)(np.arange(num_devices))
print("--- the parameters initiated are: {} ---".format(params[0]))

start_time = time.time()

# forward model
def forward(params, O_):
    N = O_.shape[0]

    Sf    = F_(Sm, psl, O_[:, 0], N)
    If    = F_(Dm, psl, O_[:, 1], N)

    Am.create_fig(params, N)
    arni(Am, Sf, If, psl)

    # FFT:
    E_s_out_k    = 0.094 * Fourier(Sf.Ao)
    E_i_out_k    = 0.094 * Fourier(If.Ao)
    E_i_O_k      = 0.094 * Fourier(If.E_dac)

    G1_ss        = (np.array([kron(np.conj(E_s_out_k[i]), E_s_out_k[i]) for i in range(N)]) / batch_size).sum(0)
    G1_ii        = (np.array([kron(np.conj(E_i_out_k[i]), E_i_out_k[i]) for i in range(N)]) / batch_size).sum(0)
    G1_si        = (np.array([kron(np.conj(E_i_out_k[i]), E_s_out_k[i]) for i in range(N)]) / batch_size).sum(0)
    G1_si_dag    = (np.array([kron(np.conj(E_s_out_k[i]), E_i_out_k[i]) for i in range(N)]) / batch_size).sum(0)
    Q_si         = (np.array([kron(E_i_O_k[i], E_s_out_k[i]) for i in range(N)]) / batch_size).sum(0)
    Q_si_dag     = (np.array([kron(np.conj(E_s_out_k[i]), np.conj(E_i_O_k[i])) for i in range(N)]) / batch_size).sum(0)
    return G1_ii, G1_ss, Q_si_dag, Q_si, G1_si_dag, G1_si

def loss(params, O_, Att, Btt):
    params = np.divide(params, la.norm(params))
    batched_preds   = forward(params, O_)
    G1_ii, G1_ss, Q_si_dag, Q_si, G1_si_dag, G1_si = batched_preds

    P_ss         = G1_ss[::M + 1, ::M + 1].real
    G2           = (G1_ii * G1_ss + Q_si_dag * Q_si + G1_si_dag * G1_si).real

    P_ss    = P_ss / np.sum(np.abs(P_ss))
    G2      = G2 / np.sum(np.abs(G2))

    return l1_loss(P_ss, Att) + l1_loss(G2, Btt)

@partial(pmap, axis_name='device')
def update(opt_state, i, x, Att, Btt):
    params              = get_params(opt_state)
    batch_loss, grads   = value_and_grad(loss)(params, x, Att, Btt)
    grads               = np.array([lax.psum(dw, 'device') for dw in grads])
    return lax.pmean(batch_loss, 'device'), opt_update(i, grads, opt_state)


# load targets
At  = 'At'
Bt  = 'Bt'
Att = pmap(lambda x: np.load(path + At + '.npy'))(np.arange(num_devices))
Btt = pmap(lambda x: np.load(path + Bt + '.npy'))(np.arange(num_devices))

"Build a dataset"
keys = random.split(random.PRNGKey(1986), num_devices)
# generate dataset for each gpu
O_rnd = pmap(lambda key: random.normal(key, (Ndevice, 2, 2, Nx, Ny)))(keys)

# split to batches
def get_train_batches(O_, key_):
    O_shuff = random.permutation(key_, O_)
    batch_arr = np.split(O_shuff, num_batches, axis=0)
    return batch_arr

# seed shuffle batches in epochs
key_batch_epoch = pmap(lambda i: random.split(random.PRNGKey(1989), num_epochs))(np.arange(num_devices))


# Use optimizers to set optimizer initialization and update functions
opt_init, opt_update, get_params = optimizers.adam(0.01, b1=0.9, b2=0.999, eps=1e-08)
opt_state = opt_init(params)
losses = []
for epoch in range(num_epochs):
    losses.append(0.0)
    start_time_epoch = time.time()
    batch_set = pmap(get_train_batches)(O_rnd, key_batch_epoch[:, epoch])
    print("Epoch {} is running".format(epoch))
    for i, x in enumerate(batch_set):
        idx = pmap(lambda x: np.array(epoch+i))(np.arange(num_devices))
        batch_loss, opt_state = update(opt_state, idx, x, Att, Btt)
        losses[epoch] += batch_loss[0].item()
    params_ = get_params(opt_state)[0]
    losses[epoch] /= (i+1)
    epoch_time = time.time() - start_time_epoch
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    ''' print loss value'''
    print("params:{}".format(params_))
    print("loss:{:0.6f}".format(losses[epoch]))
exit()
