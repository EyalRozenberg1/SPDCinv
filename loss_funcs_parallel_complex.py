import jax.numpy as np
from jax import jit


###########
# L1 Loss #
###########
@jit
def l1_loss(a, b: np.array = 0):
    return np.sum(np.abs(a - b))


###########
# L2 Loss #
###########
@jit
def l2_loss(a, b: np.array = 0):
    return np.sum((a - b)**2)


#########################
# kullback leibler Loss #
#########################
@jit
def kl_loss(a, b, eps=1e-7):
    # KL Divergence #
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0 (or smaller). """
    A = a + eps
    B = b + eps
    return np.sum(B * np.log(B / A))


#################
# Bhattacharyya #
#################
@jit
def bhattacharyya_loss(a, b, eps=1e-10):
    # Bhattacharyya distance #
    return np.sqrt(1. - np.sum(np.sqrt(a * b + eps)))


#################
# Sinkhorn Loss #
#################
def pairwise_distances(a, b, p=2):
    """
    Compute the pairwise distance matrix between a and b which both have size [m, n, d] or [n, d]. The result is a tensor of
    size [m, n, n] (or [n, n]) whose entry [m, i, j] contains the distance_tensor between a[m, i, :] and b[m, j, :].
    :param a: A tensor containing m batches of n points of dimension d. i.e. of size [m, n, d]
    :param b: A tensor containing m batches of n points of dimension d. i.e. of size [m, n, d]
    :param p: Norm to use for the distance_tensor
    :return: A tensor containing the pairwise distance_tensor between each pair of inputs in a batch.
    """

    squeezed = False
    if len(a.shape) == 2 and len(b.shape) == 2:
        a = a[np.newaxis, :, :]
        b = b[np.newaxis, :, :]
        squeezed = True

    if len(a.shape) != 3:
        raise ValueError("Invalid shape for a. Must be [m, n, d] or [n, d] but got", a.shape)
    if len(b.shape) != 3:
        raise ValueError("Invalid shape for a. Must be [m, n, d] or [n, d] but got", b.shape)

    ret = np.power(np.abs(a[:, :, np.newaxis, :] - b[:, np.newaxis, :, :]), p).sum(3)
    if squeezed:
        ret = np.squeeze(ret)

    return ret


def sinkhorn(a, b, D, eps, max_iters=100, stop_thresh=1e-3):
    """
    Compute the Sinkhorn divergence between two sum of dirac delta distributions, U, and V.
    This implementation is numerically stable with float32.
    :param a: A m-sized minibatch of weights for each dirac in the first distribution, U. i.e. shape = [m, n]
    :param b: A m-sized minibatch of weights for each dirac in the second distribution, V. i.e. shape = [m, n]
    :param D: A minibatch of n-by-n tensors storing the distance between each pair of diracs in U and V.
             i.e. shape = [m, n, n] and each i.e. D[k, i, j] = ||u[k,_i] - v[k, j]||
    :param eps: The reciprocal of the sinkhorn regularization parameter
    :param max_iters: The maximum number of Sinkhorn iterations
    :param stop_thresh: Stop if the change in iterates is below this value
    :return:
    """
    # a and b are tensors of size [nb, m] and [nb, n]
    # D is a tensor of size [nb, m, n]

    D = np.squeeze(D)
    a = np.squeeze(a)
    b = np.squeeze(b)
    squeezed = False

    if len(D.shape) == 2 and len(a.shape) == 1 and len(b.shape) == 1:
        D = D[np.newaxis, :, :]
        a = a[np.newaxis, :]
        b = b[np.newaxis, :]
        squeezed = True
    elif len(D.shape) == 2 and len(a.shape) != 1:
        raise ValueError("Invalid shape for a %s, expected [m,] where m is the number of samples in a and "
                         "D has shape [m, n]" % str(a.shape))
    elif len(D.shape) == 2 and len(b.shape) != 1:
        raise ValueError("Invalid shape for a %s, expected [m,] where n is the number of samples in a and "
                         "D has shape [m, n]" % str(b.shape))

    if len(D.shape) != 3:
        raise ValueError("Got unexpected shape for D %s, should be [nb, m, n] where nb is batch size, and "
                         "m and n are the number of samples in the two input measures." % str(D.shape))
    elif len(D.shape) == 3 and len(a.shape) != 2:
        raise ValueError("Invalid shape for a %s, expected [nb, m]  where nb is batch size, m is the number of samples "
                         "in a and D has shape [nb, m, n]" % str(a.shape))
    elif len(D.shape) == 3 and len(b.shape) != 2:
        raise ValueError("Invalid shape for a %s, expected [nb, m]  where nb is batch size, m is the number of samples "
                         "in a and D has shape [nb, m, n]" % str(b.shape))

    nb = D.shape[0]
    m = D.shape[1]
    n = D.shape[2]

    if a.dtype != b.dtype or a.dtype != D.dtype:
        raise ValueError("Tensors a, b, and D must have the same dtype got: dtype(a) = %s, dtype(b) = %s, dtype(D) = %s"
                         % (str(a.dtype), str(b.dtype), str(D.dtype)))
    if a.shape != (nb, m):
        raise ValueError("Got unexpected shape for tensor a (%s). Expected [nb, m] where D has shape [nb, m, n]." %
                         str(a.shape))
    if b.shape != (nb, n):
        raise ValueError("Got unexpected shape for tensor b (%s). Expected [nb, n] where D has shape [nb, m, n]." %
                         str(b.shape))

    u = np.zeros_like(a)
    v = np.zeros_like(b)

    M_t = np.transpose(D, axes=(0, 2, 1))

    def stabilized_log_sum_exp(x):
        max_x = x.max(2)
        x = x - max_x[:, :, np.newaxis]
        ret = np.log(np.sum(np.exp(x), axis=2)) + max_x
        return ret

    for current_iter in range(max_iters):
        u_prev = u
        v_prev = v

        summand_u = (-D + np.expand_dims(v, 1)) / eps
        u = eps * (np.log(a) - stabilized_log_sum_exp(summand_u))

        summand_v = (-M_t + np.expand_dims(u, 1)) / eps
        v = eps * (np.log(b) - stabilized_log_sum_exp(summand_v))

        err_u = np.sum(np.abs(u_prev - u), axis=1).max()
        err_v = np.sum(np.abs(v_prev - v), axis=1).max()

        # if err_u < stop_thresh and err_v < stop_thresh:
        #     break

    log_P = (-D + np.expand_dims(u, 2) + np.expand_dims(v, 1)) / eps

    P = np.exp(log_P)

    if squeezed:
        P = np.squeeze(P)

    return P

def sinkhorn_loss(P, Pt, M, eps, max_iters=100, stop_thresh=1e-3):

    [X, Y] = np.meshgrid(np.arange(0, M, dtype=np.float32), np.arange(0, M, dtype=np.float32))

    a = np.concatenate((P.reshape(-1, 1), X.reshape(-1, 1), Y.reshape(-1, 1)), 1)
    b = np.concatenate((Pt.reshape(-1, 1), X.reshape(-1, 1), Y.reshape(-1, 1)), 1)

    # M is a 10x100x100 array where each entry (k, i, j) is the squared distance between point a[k, i, :] and b[k, j, :]
    D = pairwise_distances(a, b)

    # w_a and w_b are masses assigned to each point. In this case each point is weighted equally.
    w_a = np.ones(a.shape[0]) / M ** 2
    w_b = np.ones(b.shape[0]) / M ** 2
    P = sinkhorn(w_a, w_b, D, eps, max_iters, stop_thresh)
    # P is the transport matrix between a and b, eps is a regularization parameter, smaller epsilons lead to
    # better approximation of the true Wasserstein distance at the expense of slower convergence

    # To get the distance as a number just compute the frobenius inner product <M, P>
    return (D * P).sum()


##################
# Trace Distance #
##################
@jit
def TraceDist_loss(rho, sigma):
    '''
    Calculate the Trace Distance between rho and sigma
    as depict in: https://en.wikipedia.org/wiki/Trace_distance#Definition
    Parameters
    ----------
    rho: density matrix rho
    sigma: density matrix sigma
    Returns: Trace distance
    -------
    '''

    T = 0.5 * np.trace(np.sqrt((rho - sigma).conj().T @ (rho - sigma)))
    return T
