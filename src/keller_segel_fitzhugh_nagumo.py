import numpy as np
import cupy as cp

class KellerSegelFHN:
    def __init__(self, N, DU, DV, CHI, CAP, ALPHA, TAU, GAMMA, GATE, Iext, DT, seed=None):
        rng = np.random.default_rng(seed)
        self.u = cp.asarray(1.0 + rng.uniform(-0.1, 0.1, (N, N)))
        self.v = cp.zeros((N, N), dtype=np.float32)
        self.w = cp.zeros((N, N), dtype=np.float32)
        cx, cy = N//2, N//2
        self.v[cy-5:cy+5, cx-40:cx+40] = 0.9 
        self.w[cy-5:cy+5, cx-40:cx] = 0.5       
        self.N = N
        self.DU = DU
        self.DV = DV
        self.CHI = CHI
        self.CAP = CAP
        self.ALPHA = ALPHA
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.GATE = GATE
        self.DT = DT
        self.Iext = Iext
    
    def step(self):
        new_u = implicit_diffusion(self.u, self.DU, self.DT) - explicit_chemotaxis(self.u, self.v, self.CAP, self.CHI) * self.DT
        new_v = implicit_diffusion(self.v, self.DV, self.DT) + (gated_production(self.v, self.w, self.GATE, self.Iext) - self.ALPHA * self.v) * self.DT
        new_w = self.w + (self.v - self.GAMMA * self.w) / self.TAU * self.DT
        self.u = new_u
        self.v = cp.maximum(new_v, 0)
        self.w = new_w
    
    def run(self, n_steps, save_every=None, div_threshold=None):
        history = []
        for i in range(n_steps + 1):
            self.step()
            if save_every and i % save_every == 0:
                history.append((i, cp.asnumpy(self.u).copy(), cp.asnumpy(self.v).copy()))
            if div_threshold and i > 1000 and i % 100 == 0:
                density_range = np.max(self.densities) - np.min(self.densities)
                if density_range >= div_threshold:
                    return i
        if save_every and div_threshold:
            return history, None
        if div_threshold:
            return None
        else:
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

def implicit_diffusion(u, du, DT):
    u_hat = cp.fft.fft2(u)
    Ny, Nx = u.shape
    kx = cp.fft.fftfreq(Nx, 1.0)
    ky = cp.fft.fftfreq(Ny, 1.0)
    KX, KY = cp.meshgrid(kx, ky)
    K2 = (2 * cp.pi * KX)**2 + (2 * cp.pi * KY)**2
    u_hat_new = cp.exp(du * -K2 * DT) * u_hat
    u_new = cp.fft.ifft2(u_hat_new).real
    return u_new

def explicit_chemotaxis(u, v, umax, chi0):
    grad_x, grad_y = spectral_gradient(v)
    chi = chi0 * (1 - u / umax)
    flux_x = u * grad_x * chi
    flux_y = u * grad_y * chi
    div_uv = spectral_divergence(flux_x, flux_y)
    return div_uv

def gated_production(v, w, a, Iext):
    return v * (v - a) * (1 - v) - w + Iext
