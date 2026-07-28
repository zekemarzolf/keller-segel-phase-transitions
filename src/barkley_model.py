import numpy as np
import cupy as cp
from vispy import scene, app, color, plot

class BarkleySim:
    def __init__(self, N, dt, dx, dw, eps, a, b, seed=None):
        rng = np.random.default_rng(seed)
        self.v = cp.zeros((N, N), dtype=np.float32)
        self.w = cp.zeros((N, N), dtype=np.float32)
        cx, cy = N//2, N//2
        self.v[cy-5:cy+5, cx-40:cx+40] = 0.5
        self.w[cy-5:cy+5, cx-40:cx] = 1.0
        self.N = N
        self.dt = dt
        self.dx = dx
        self.dw = dw
        self.eps = eps
        self.a = a
        self.b = b

    
    def step(self):
        new_v = self.v + barkley_diffusion_term(self.v, 1.0, self.dt, self.dx) + f(self.v, self.w, self.eps, self.a, self.b) * self.dt
        new_w = self.w + g(self.v, self.w) * self.dt + barkley_diffusion_term(self.w, self.dw, self.dt, self.dx)
        self.v = new_v
        self.w = new_w
    
    def run(self, n_steps, save_every=None):
        history = []
        for i in range(n_steps + 1):
            self.step()
            if save_every and i % save_every == 0:
                history.append((i, cp.asnumpy(self.v).copy(), cp.asnumpy(self.w).copy()))
        return history

def implicit_diffusion_term(u, dt, dx):
    u_hat = cp.fft.fft2(u)
    Ny, Nx = u.shape
    kx = cp.fft.fftfreq(Nx, d=dx)
    ky = cp.fft.fftfreq(Ny, d=dx)
    KX, KY = cp.meshgrid(kx, ky)
    K2 = (2 * cp.pi * KX)**2 + (2 * cp.pi * KY)**2
    u_hat_new = cp.exp(0.1 * -K2 * dt) * u_hat
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

def f(v, w, eps, a, b):
    return (1 / eps) * v * (1 - v) * (v - (w + b) / a)

def g(v, w):
    return v - w