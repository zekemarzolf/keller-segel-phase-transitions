# README
## Objective:
Aiming to discover and investigate numerical phase transitions in which behaviors/structures emerge in systems like slime mold

## How I Will Do This:
Analyze transitions with persistent homology and multiple approaches such as topology, information transport, and other qualitative measures

## Model:
Using the Keller-Segel model for chemotaxis with an extra logistic growth term for stability where K represents the maximum density of cells at any point:
$$\frac{\partial u}{\partial t} = D_u\nabla^2u - \chi \nabla \cdot (u \nabla v) (\frac{K-u}{K})$$
$$\frac{\partial v}{\partial t} = D_v\nabla^2v - \alpha v + \beta u $$
## Required Libraries:
1. NumPy
2. CuPy
3. VisPy
4. SciPy
5. Matplotlib
6. PyQt6 or PySide6