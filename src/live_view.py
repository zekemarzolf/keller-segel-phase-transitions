import numpy as np
import cupy as cp
from vispy import plot
from vispy import scene, app, color
from keller_segel_logistic import KellerSegelLogistic as KSL
from keller_segel_excitable import KellerSegelExcitable as KSE
from barkley_model import BarkleySim

N = 512

sim = KSE(N=N, du=0.05, dv=0.1, CHI=2.0, CAP=2.0, ALPHA=0.1, BETA=0.2, dt=0.05, dx=0.5)
#sim = BarkleySim(N=256, dt=0.01, dx=1.0, dw=0, eps=0.02, a=0.8, b=0.01)

def v_u_ratio(u, v):
    return v / (u + 0.02)

canvas = scene.SceneCanvas(keys='interactive', size=(1000, 1000), show=True)
grid = canvas.central_widget.add_grid()
view = grid.add_view(row=0, col=0)

image1 = scene.visuals.Image(cp.asnumpy(sim.u.astype(np.float32)), cmap='viridis', clim=(0, 0.5), parent=view.scene)
image1.transform = scene.STTransform(translate=(0, N/2+5, 0))

image2 = scene.visuals.Image(cp.asnumpy(sim.v.astype(np.float32)), cmap='viridis', clim=(0, 1), parent=view.scene)
image2.transform = scene.STTransform(translate=(N+10, N/2+5, 0))

image3 = scene.visuals.Image(cp.asnumpy(sim.w.astype(np.float32)), cmap='viridis', clim=(0, 1), parent=view.scene)
image3.transform = scene.STTransform(translate=(N+10, -N/2-5, 0))

image4 = scene.visuals.Image(cp.asnumpy(v_u_ratio(sim.u, sim.v).astype(np.float32)), clim=(0,10), cmap='viridis', parent=view.scene)
image4.transform = scene.STTransform(translate=(0, -N/2-5, 0))

view.camera = scene.cameras.PanZoomCamera(aspect=None)
view.camera.set_range(x=(0, sim.v.shape[0]), y=(0, sim.v.shape[1]))

paused = [False]
@canvas.events.key_press.connect
def on_key(event):
    if event.key == ' ':
        paused[0] = not paused[0]

i = 0
def update(ev):
    if paused[0]:
        return
    
    global i
    i += 1
    sim.step()
    
    image1.set_data(cp.asnumpy(sim.u))
    image2.set_data(cp.asnumpy(sim.v))
    image3.set_data(cp.asnumpy(sim.w))
    image4.set_data(cp.asnumpy(v_u_ratio(sim.u, sim.v)))
    
    canvas.update()

timer = app.Timer(interval=0.01, connect=update, start=True)

if __name__ == '__main__':
    app.run()