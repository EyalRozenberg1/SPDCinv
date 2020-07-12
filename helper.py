import jax.numpy as np
import math

###########################################
# lambda functions:
###########################################
sfg_wv                  = lambda fg_p, fg_s: fg_p * fg_s / (fg_s - fg_p)
Fourier                 = lambda A: (np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(A))))  # Fourier


class Cr:
    def __init__(self, dx, dy, dz, MaxX, MaxY, MaxZ, pr=0):
        self.dz = dz  # resolution of z axis
        self.dx = dx  # resolution of x axis
        self.dy = dy  # resolution of y axis
        self.MaxX = MaxX
        self.MaxY = MaxY
        self.MaxZ = MaxZ
        self.x = np.arange(-MaxX, MaxX, dx)
        self.y = np.arange(-MaxY, MaxY, dy)
        self.z = np.arange(-MaxZ / 2, MaxZ / 2, dz)
        self.ct = nz
        self.slb = slb
        self.d = 20e-12
        self.pl_pr = pr

class Bm:
    def __init__(self, glr, clr, T, ais=0, fizer=0, mem=0):
        self.glr = glr
        self.ais = ais
        self.n = clr.ct(glr * 1e6, T)
        self.kk = 2 * np.pi * 2.99792458e8 / glr
        self.gg = 2 * np.pi * clr.ct(glr * 1e6, T) / glr
        self.b = ais ** 2 * self.gg
        self.fizer = fizer
        if mem:
            self.dict = Bank(glr, self.ais, self.ais, mem, clr.x, clr.y)
            self.fig  = []

    def create_fig(self, ball, Nb):
        self.fig = ball
        self.E = np.tile(make_ball(self.dict, ball),(Nb,1,1))

class F_:
    def __init__(self, ssde, clr, dac_rnd, N):
        Nx = len(clr.x)
        Ny = len(clr.y)
        self.Ao = np.zeros([N, Nx, Ny])
        dac = np.sqrt(1.054571800e-34 * ssde.kk / (2 * 8.854187817e-12 * ssde.n ** 2 * clr.dx * clr.dy * clr.MaxZ))
        self.E_dac = dac * (dac_rnd[:,0] + 1j * dac_rnd[:,1]) / np.sqrt(2)
        self.zappa = 2 * 1j * ssde.kk ** 2 / (ssde.gg * 2.99792458e8 ** 2)
        self.gg = ssde.gg

def slb(mok, z):
    return np.sign(np.cos(np.abs(mok) * z))

def nz(glr, T):
    a = np.array([5.756, 0.0983, 0.2020, 189.32, 12.52, 1.32 * 10 ** (-2)])
    b = np.array([2.860 * 10 ** (-6), 4.700 * 10 ** (-8), 6.113 * 10 ** (-8), 1.516 * 10 ** (-4)])
    f = (T - 24.5) * (T + 570.82)
    n1 = a[0]
    n2 = b[0] * f
    n3 = (a[1] + b[1] * f) / (glr ** 2 - (a[2] + b[2] * f) ** 2)
    n4 = (a[3] + b[3] * f) / (glr ** 2 - (a[4]) ** 2)
    n5 = -a[5] * glr ** 2
    nz = np.sqrt(n1 + n2 + n3 + n4 + n5)
    return nz

def arni(Swer, Sif, Iif, clr):
    for z in clr.z:
        x  = clr.x
        y  = clr.y
        dz = clr.dz

        E_Swer = Karni(Swer.E, x, y, Swer.gg, z) * np.exp(1j * Swer.gg * z)
        PP = clr.slb(clr.pl_pr, z)

        dEs_out_dz = Sif.zappa * clr.d * PP * E_Swer * np.conj(Iif.E_dac)
        dEs_dac_dz = Sif.zappa * clr.d * PP * E_Swer * np.conj(Iif.Ao)

        Sif.Ao = Sif.Ao + dEs_out_dz * dz
        Sif.E_dac = Sif.E_dac + dEs_dac_dz * dz

        dEi_out_dz = Iif.zappa * clr.d * PP * E_Swer * np.conj(Sif.E_dac)
        dEi_dac_dz = Iif.zappa * clr.d * PP * E_Swer * np.conj(Sif.Ao)

        Iif.Ao = Iif.Ao + dEi_out_dz * dz
        Iif.E_dac = Iif.E_dac + dEi_dac_dz * dz

        Sif.Ao = Karni(Sif.Ao, x, y, Sif.gg, dz) * np.exp(1j * Sif.gg * dz)
        Sif.E_dac = Karni(Sif.E_dac, x, y, Sif.gg, dz) * np.exp(1j * Sif.gg * dz)
        Iif.Ao = Karni(Iif.Ao, x, y, Iif.gg, dz) * np.exp(1j * Iif.gg * dz)
        Iif.E_dac = Karni(Iif.E_dac, x, y, Iif.gg, dz) * np.exp(1j * Iif.gg * dz)
    return

def Karni(A, x, y, k, dz):
    dx = np.abs(x[1] - x[0])
    dy = np.abs(y[1] - y[0])
    X, Y = np.meshgrid(x, y, indexing='ij')
    KX = 2 * np.pi * (X / dx) / (np.size(X, 1) * dx)
    KY = 2 * np.pi * (Y / dy) / (np.size(Y, 1) * dy)
    H_w = np.exp(-1j * dz * (np.square(KX) + np.square(KY)) / (2 * k))
    H_w = np.fft.ifftshift(H_w)
    G = np.fft.fft2(A)
    F = np.multiply(G, H_w)
    Eout = np.fft.ifft2(F)
    return Eout

def eP(n, x):
    if n == 0:
        return 1
    elif n == 1:
        return 2 * x
    else:
        return 2 * x * eP(n - 1, x) - 2 * (n - 1) * eP(n - 2, x)

def B2Dxy(glr, W0x, W0y, n, m, z, x, y):
    W0 = np.sqrt(2 * W0x ** 2)
    k  = 2 * np.pi / glr
    z0 = np.pi * W0 ** 2 / glr
    Wx = W0x * np.sqrt(1 + (z / z0) ** 2)
    invR = z / ((z ** 2) + (z0 ** 2))
    qx = 1 / (invR - 1j * glr / (np.pi * (Wx ** 2)))
    q0 = 1j * z0
    coefx = (2 / np.pi) ** 0.25 * np.sqrt(1 / (2 ** n * math.factorial(n) * W0x)) * np.sqrt(q0 / qx) * (
                q0 * np.conj(qx) / (np.conj(q0) * qx)) ** (n / 2)
    Unx = coefx * eP(n, np.sqrt(2) * x / Wx) * np.exp(-1j * k * ((x ** 2) * (1 / (2 * qx))))
    W0 = np.sqrt(2 * W0y ** 2)
    z0 = np.pi * W0 ** 2 / glr
    Wy = W0y * np.sqrt(1 + (z / z0) ** 2)
    invR = z / ((z ** 2) + (z0 ** 2))
    qy = 1 / (invR - 1j * glr / (np.pi * (Wy ** 2)))
    q0 = 1j * z0
    coefy = (2 / np.pi) ** 0.25 * np.sqrt(1 / (2 ** m * math.factorial(m) * W0y)) * np.sqrt(q0 / qy) * (
                q0 * np.conj(qy) / (np.conj(q0) * qy)) ** (m / 2)
    Uny = coefy * eP(m, np.sqrt(2) * y / Wy) * np.exp(-1j * k * ((y ** 2) * (1 / (2 * qy))))
    return Unx, Uny

def Bank(glr, W0x, W0y, mem, x, y):
    dictx = {}
    dicty = {}
    dict = {}
    for n in range(mem):
        dictx[str(n)], dicty[str(n)] = B2Dxy(glr, W0x, W0y, n, n, 0, x, y)
    for n in range(mem):
        for m in range(mem):
            Uny = dicty[str(m)]
            Unx = dictx[str(n)]
            dict[str(n) + str(m)] = np.dot(Uny.reshape(len(Unx), 1), Unx.reshape(1, len(Unx)))
    return dict

def make_ball(dict, ball):
    final = 0
    if len(ball) != len(dict):
        print('WRONG NUMBER OF PARAMETERS!!!')
        return
    for n, (_x, HG) in enumerate(dict.items()):
        final = final + ball[n] * HG
    return final

def l1_loss(a, b):
    return np.sum(np.abs(a - b))
