# README
## Objective:
Aiming to discover and investigate numerical phase transitions in which behaviors/structures emerge in systems like slime mold

## How I Will Do This:
Analyze transitions with persistent homology and multiple approaches such as topology, information transport, and other qualitative measures

## Model:
Using the Keller-Segel model for chemotaxis with an extra saturation term for stability:
$$\frac{\partial u}{\partial t} = D_u\nabla^2u - \sigma (u) \chi \nabla \cdot (u \nabla v) $$
$$\frac{\partial v}{\partial t} = D_v\nabla^2v - \alpha v + \beta u $$
$$\sigma (u) = $$
## Required Libraries:
1. numpy
2. cupy
3. vispy
4. scipy
5. matplotlib
6. PyQt6 or PySide6