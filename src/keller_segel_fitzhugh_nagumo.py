import numpy as np
import cupy as cp

class KellerSegelFHN:
    def __init__(self, N, du, dv, CHI, CAP, ALPHA, BETA, dt, seed=None):
        rng = np.random.default_rng(seed)
        self.u = cp.asarray(1.0 + rng.uniform(-0.1, 0.1, (N, N)))
        self.v = cp.zeros((N, N), dtype=np.float32)
        self.w = cp.zeros((N, N), dtype=np.float32)
        cx, cy = N//2, N//2
        self.v[cy-5:cy+5, cx-40:cx+40] = 0.5
        self.w[cy-5:cy+5, cx-40:cx] = 1.0
        self.N = N
        self.du = du
        self.dv = dv
        self.CHI = CHI
        self.CAP = CAP
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.dt = dt
    
    def step(self):
        new_u = self.u + barkley_diffusion_term(u=self.u, D=self.du, dt=self.dt, dx=1.0) - explicit_chemotaxis(self.u, self.v, self.CAP, self.CHI) * self.dt
        new_v = self.v + barkley_diffusion_term(u=self.v, D=self.dv, dt=self.dt, dx=1.0) + f(self.v, self.w, eps=0.02, a=0.8, b=0.01) * self.dt + (self.BETA * self.u - self.ALPHA * self.v) * self.dt
        new_w = self.w + barkley_diffusion_term(u=self.w, D=0, dt=self.dt, dx=1.0) + g(self.v, self.w) * self.dt
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

def implicit_diffusion(u, du, dt):
    u_hat = cp.fft.fft2(u)
    Ny, Nx = u.shape
    kx = cp.fft.fftfreq(Nx, 1.0)
    ky = cp.fft.fftfreq(Ny, 1.0)
    KX, KY = cp.meshgrid(kx, ky)
    K2 = (2 * cp.pi * KX)**2 + (2 * cp.pi * KY)**2
    u_hat_new = cp.exp(du * -K2 * dt) * u_hat
    u_new = cp.fft.ifft2(u_hat_new).real
    return u_new

def barkley_diffusion_term(u, D, dt, dx, dy=None):
    if D == 0:
        return 0
    if dy is None:
        dy = dx
    nu_x = D * dt / dx**2
    nu_y = D * dt / dy**2
    lap = (
        nu_x * (cp.roll(u, -1, axis=1) + cp.roll(u, 1, axis=1) - 2 * u) +
        nu_y * (cp.roll(u, -1, axis=0) + cp.roll(u, 1, axis=0) - 2 * u)
    )
    return lap

def explicit_chemotaxis(u, v, umax, chi0):
    grad_x, grad_y = spectral_gradient(v)
    chi = chi0 * (1 - u / umax)
    flux_x = u * grad_x * chi
    flux_y = u * grad_y * chi
    div_uv = spectral_divergence(flux_x, flux_y)
    return div_uv

def gated_production(v, w, a, Iext):
    return v * (v - a) * (1 - v) - w + Iext

def f(v, w, eps, a, b):
    return (1 / eps) * v * (1 - v) * (v - (w + b) / a)

def g(v, w):
    return v - w