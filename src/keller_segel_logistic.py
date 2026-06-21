import numpy as np
import cupy as cp

class KellerSegel:
    def __init__(self, N, DU, DV, SENSITIVITY, R, CAPACITY, ALPHA, BETA, DT, seed=None):
        rng = np.random.default_rng(seed)
        self.densities = cp.asarray(1.0 + rng.uniform(-0.1, 0.1, (N, N)))
        self.chemoattractant = cp.asarray(1.0 + rng.uniform(-0.1, 0.1, (N, N)))
        self.N = N
        self.DU = DU
        self.DV = DV
        self.SENSITIVITY = SENSITIVITY
        self.R = R
        self.CAPACITY = CAPACITY
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.DT = DT
    
    def step(self):
        new_densities = implicit_diffusion(self.densities, self.DU, self.DT) - explicit_nonlinear(self.densities, self.chemoattractant, self.SENSITIVITY) * self.DT + logistic_growth(self.densities, self.R, self.CAPACITY) * self.DT
        new_chemoattractant = implicit_diffusion(self.chemoattractant, self.DV, self.DT) + (self.BETA * self.densities - self.ALPHA * self.chemoattractant) * self.DT
        self.densities = new_densities
        self.chemoattractant = new_chemoattractant
    
    def run(self, n_steps, save_every=None):
        history = []
        for i in range(n_steps + 1):
            self.step()
            if save_every and i % save_every == 0:
                history.append((i, cp.asnumpy(self.densities).copy(), cp.asnumpy(self.chemoattractant).copy()))
        return history

# Spectral Methods:

def spectral_gradient(u):
    u_hat = cp.fft.fft2(u)
    Ny, Nx = u.shape
    kx = cp.fft.fftfreq(Nx, 1.0)
    ky = cp.fft.fftfreq(Ny, 1.0)
    KX, KY = cp.meshgrid(kx, ky)
    ux_hat_grad = u_hat * KX * 1j
    uy_hat_grad = u_hat * KY * 1j
    ux_grad = cp.fft.ifft2(ux_hat_grad)
    uy_grad = cp.fft.ifft2(uy_hat_grad)
    return ux_grad, uy_grad

def spectral_divergence(ux, uy):
    ux_hat = cp.fft.fft2(ux)
    uy_hat = cp.fft.fft2(uy)
    Ny, Nx = ux.shape
    kx = cp.fft.fftfreq(Nx, 1.0)
    ky = cp.fft.fftfreq(Ny, 1.0)
    KX, KY = cp.meshgrid(kx, ky)
    ux_hat_div = ux_hat * KX * 1j
    uy_hat_div = uy_hat * KY * 1j
    div_hat = ux_hat_div + uy_hat_div
    div = cp.fft.ifft2(div_hat).real
    return div

# Finite Difference Methods:

def finite_gradient(u):
    grad_y, grad_x = cp.gradient(u)
    return grad_x, grad_y

def finite_divergence(ux, uy):
    ux_div = cp.empty((N,N))
    ux_div[:,1:-1] = (ux[:,2:] - ux[:,:-2]) / 2
    ux_div[:,0] =  ux[:,1] - ux[:,0]
    ux_div[:,-1] =  ux[:,-1] - ux[:,-2]
    uy_div = cp.empty((N,N))
    uy_div[1:-1,:] = (uy[2:,:] - uy[:-2,:]) / 2
    uy_div[0,:] =  uy[1,:] - uy[0,:]
    uy_div[-1,:] =  uy[-1,:] - uy[-2,:]
    u_div = ux_div + uy_div
    return u_div

# PDE Terms:

def implicit_diffusion(u, diff_coefficient, DT):
    u_hat = cp.fft.fft2(u)
    Ny, Nx = u.shape
    kx = cp.fft.fftfreq(Nx, 1.0)
    ky = cp.fft.fftfreq(Ny, 1.0)
    KX, KY = cp.meshgrid(kx, ky)
    K2 = (2 * cp.pi * KX)**2 + (2 * cp.pi * KY)**2
    u_hat_new = cp.exp(diff_coefficient * -K2 * DT) * u_hat
    u_new = cp.fft.ifft2(u_hat_new).real
    return u_new

def explicit_nonlinear(u, v, SENSITIVITY):
    grad_x, grad_y = spectral_gradient(v)
    flux_x = u * grad_x
    flux_y = u * grad_y
    div_uv = spectral_divergence(flux_x, flux_y)
    return div_uv * SENSITIVITY

def logistic_growth(u, R, CAPACITY):
    return R * u * ((CAPACITY - u) / CAPACITY)
