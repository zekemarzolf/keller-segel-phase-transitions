import sys
import numpy as np
import cupy as cp
from vispy import plot
from vispy import scene, app, color
from scipy.linalg import solve_banded

DT = 0.1
N = 512
DU = 0.2
DV = 1
SENSITIVITY = 1
SATURATION_RATE = 1
ALPHA = 0.1
BETA = 1
CAPACITY = 4
densities = np.ones((N, N)) * 2.0
densities += np.random.uniform(-0.1, 0.1, (N, N))
chemoattractant = np.ones((N, N)) * 2.0
chemoattractant += np.random.uniform(-0.1, 0.1, (N, N))

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

def implicit_diffusion(u, diff_coefficient):
    u_hat = cp.fft.fft2(u)
    Ny, Nx = u.shape
    kx = cp.fft.fftfreq(Nx, 1.0)
    ky = cp.fft.fftfreq(Ny, 1.0)
    KX, KY = cp.meshgrid(kx, ky)
    K2 = (2 * cp.pi * KX)**2 + (2 * cp.pi * KY)**2
    u_hat_new = cp.exp(diff_coefficient * -K2 * DT) * u_hat
    u_new = cp.fft.ifft2(u_hat_new).real
    return u_new

def explicit_nonlinear(u, v):
    grad_x, grad_y = spectral_gradient(v)
    sat = saturation(u, SENSITIVITY)
    flux_x = u * grad_x * sat
    flux_y = u * grad_y * sat
    div_uv = spectral_divergence(flux_x, flux_y)
    #logistic = SENSITIVITY * ((CAPACITY - u) / CAPACITY)
    return u * div_uv * DT

def saturation(u, x):
    sigmoid = 1 / (1 + np.exp(1.5 * (u - 2)))
    return sigmoid * x

canvas = scene.SceneCanvas(keys='interactive', size=(1000, 1000), show=True)
grid = canvas.central_widget.add_grid()
view = grid.add_view(row=0, col=0)
image1 = scene.visuals.Image(densities.astype(np.float32), cmap='viridis', clim=(0.5, 4), parent=view.scene)
image2 = scene.visuals.Image(chemoattractant.astype(np.float32), cmap='viridis', clim=(0, 30), parent=view.scene)
image2.transform = scene.STTransform(translate=(N+10, 0, 0))
view.camera = scene.cameras.PanZoomCamera(aspect=None)
view.camera.set_range(x=(0, densities.shape[0]), y=(0, densities.shape[1]))

def update(ev):
    global densities, chemoattractant
    densities_gpu = cp.asarray(densities)
    chemoattractant_gpu = cp.asarray(chemoattractant)
    densities_gpu = implicit_diffusion(densities_gpu, DU) - explicit_nonlinear(densities_gpu, chemoattractant_gpu)
    chemoattractant_gpu = implicit_diffusion(chemoattractant_gpu, DV) + (BETA * densities_gpu - ALPHA * chemoattractant_gpu) * DT
    densities = cp.asnumpy(cp.clip(densities_gpu, 0, None))
    chemoattractant = cp.asnumpy(cp.clip(chemoattractant_gpu, 0, None))
    image1.set_data(densities.astype(np.float32))
    image2.set_data(chemoattractant.astype(np.float32))
    canvas.update()

timer = app.Timer(interval=0.01, connect=update, start=True)

if __name__ == '__main__':
    app.run()