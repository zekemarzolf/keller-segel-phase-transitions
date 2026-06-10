import sys
import numpy as np
import cupy as cp
from vispy import plot
from vispy import scene, app, color
from scipy.linalg import solve_banded

DT = 1
N = 1000
DU = 1
DV = 1
SENSITIVITY = 1
ALPHA = 1
BETA = 1
densities = np.ones((N,N))
chemoattractant_conc = np.ones((N,N))

#Spectral Gradient

#Spectral Divergence

#Implicit Diffusion

#Explicit Nonlinearity

canvas = scene.SceneCanvas(keys='interactive', size=(1000, 1000), show=True)
grid = canvas.central_widget.add_grid()
view = grid.add_view(row=0, col=0)
image = scene.visuals.Image(densities.astype(np.float32), cmap='viridis', clim=(0.0, 1000.0), parent=view.scene)
view.camera = scene.cameras.PanZoomCamera(aspect=None)
view.camera.set_range(x=(0, densities.shape[0]), y=(0, densities.shape[1]))

def update(ev):
    global densities
    #computations
    image.set_data(densities.astype(np.float32))
    canvas.update()

timer = app.Timer(interval=0.01, connect=update, start=True)

if __name__ == '__main__':
    app.run()