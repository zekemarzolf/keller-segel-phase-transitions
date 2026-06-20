import numpy as np
import cupy as cp
from vispy import plot
from vispy import scene, app, color
from keller_segel_logistic import KellerSegel

DT = 0.1
N = 512
DU = 0.2
DV = 1
SENSITIVITY = 5
R = 0.1
ALPHA = 0.1
BETA = 1
CAPACITY = 2

sim = KellerSegel(N, DU, DV, SENSITIVITY, R, CAPACITY, ALPHA, BETA, DT)

canvas = scene.SceneCanvas(keys='interactive', size=(1000, 1000), show=True)
grid = canvas.central_widget.add_grid()
view = grid.add_view(row=0, col=0)
image1 = scene.visuals.Image(cp.asnumpy(sim.densities.astype(np.float32)), cmap='viridis', clim=(0.5, 4), parent=view.scene)
image2 = scene.visuals.Image(cp.asnumpy(sim.chemoattractant.astype(np.float32)), cmap='viridis', clim=(0, 30), parent=view.scene)
image2.transform = scene.STTransform(translate=(N+10, 0, 0))
view.camera = scene.cameras.PanZoomCamera(aspect=None)
view.camera.set_range(x=(0, sim.densities.shape[0]), y=(0, sim.densities.shape[1]))

def update(ev):
    sim.step()
    densities = cp.asnumpy(cp.clip(sim.densities, 0, None))
    chemoattractant = cp.asnumpy(cp.clip(sim.chemoattractant, 0, None))
    image1.set_data(densities.astype(np.float32))
    image2.set_data(chemoattractant.astype(np.float32))
    canvas.update()

timer = app.Timer(interval=0.01, connect=update, start=True)

if __name__ == '__main__':
    app.run()