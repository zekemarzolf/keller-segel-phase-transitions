import sys
import numpy as np
#import cupy as cp
from vispy import plot
from vispy import scene, app, color
from scipy.linalg import solve_banded

DT = 1
N = 256
DU = 0.2
DV = 10
SENSITIVITY = 1
ALPHA = 0.1
BETA = 1
SATURATION = 0.07
CAPACITY = 4
densities = np.ones((N, N)) * 2.0
densities += np.random.uniform(-0.1, 0.1, (N, N))
chemoattractant = np.ones((N, N)) * 1.0
chemoattractant += np.random.uniform(-0.1, 0.1, (N, N))

def spectral_gradient(u):
    u_hat = np.fft.fft2(u)
    Ny, Nx = u.shape
    kx = np.fft.fftfreq(Nx, 1.0)
    ky = np.fft.fftfreq(Ny, 1.0)
    KX, KY = np.meshgrid(kx, ky)
    ux_hat_grad = u_hat * KX * 1j
    uy_hat_grad = u_hat * KY * 1j
    ux_grad = np.fft.ifft2(ux_hat_grad)
    uy_grad = np.fft.ifft2(uy_hat_grad)
    return ux_grad, uy_grad

def spectral_divergence(ux, uy):
    ux_hat = np.fft.fft2(ux)
    uy_hat = np.fft.fft2(uy)
    Ny, Nx = ux.shape
    kx = np.fft.fftfreq(Nx, 1.0)
    ky = np.fft.fftfreq(Ny, 1.0)
    KX, KY = np.meshgrid(kx, ky)
    ux_hat_div = ux_hat * KX * 1j
    uy_hat_div = uy_hat * KY * 1j
    div_hat = ux_hat_div + uy_hat_div
    div = np.fft.ifft2(div_hat).real
    return div

def implicit_diffusion(u, diff_coefficient):
    u_hat = np.fft.fft2(u)
    Ny, Nx = u.shape
    kx = np.fft.fftfreq(Nx, 1.0)
    ky = np.fft.fftfreq(Ny, 1.0)
    KX, KY = np.meshgrid(kx, ky)
    K2 = (2 * np.pi * KX)**2 + (2 * np.pi * KY)**2
    u_hat_new = np.exp(diff_coefficient * -K2 * DT) * u_hat
    u_new = np.fft.ifft2(u_hat_new).real
    return u_new

def explicit_nonlinear(u, v):
    grad_x, grad_y = spectral_gradient(v)
    flux_x = u * grad_x
    flux_y = u * grad_y
    div_uv = spectral_divergence(flux_x, flux_y)
    saturation = SENSITIVITY * ((CAPACITY - u) / CAPACITY)
    change = saturation[:,:] * div_uv[:,:]
    return u * change * DT

canvas = scene.SceneCanvas(keys='interactive', size=(1000, 1000), show=True)
grid = canvas.central_widget.add_grid()
view = grid.add_view(row=0, col=0)
image = scene.visuals.Image(np.log(densities.astype(np.float32)), cmap='viridis', clim=(0.5, 5), parent=view.scene)
view.camera = scene.cameras.PanZoomCamera(aspect=None)
view.camera.set_range(x=(0, densities.shape[0]), y=(0, densities.shape[1]))

def update(ev):
    global densities, chemoattractant
    #densities_gpu = np.asarray(densities)
    #chemoattractant_gpu = np.asarray(chemoattractant)
    densities = implicit_diffusion(densities, DU) - explicit_nonlinear(densities, chemoattractant)
    chemoattractant = implicit_diffusion(chemoattractant, DV) + (BETA * densities - ALPHA * chemoattractant) * DT
    #densities = np.asnumpy(np.clip(densities_gpu, 0, None))
    #chemoattractant = np.asnumpy(np.clip(chemoattractant_gpu, 0, None))
    chemoattractant[200:210,90:100] = 20
    image.set_data(densities.astype(np.float32))
    canvas.update()

timer = app.Timer(interval=0.01, connect=update, start=True)

if __name__ == '__main__':
    app.run()