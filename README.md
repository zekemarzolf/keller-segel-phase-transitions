# README
## Objective:
Aiming to discover and investigate numerical phase transitions in which behaviors/structures emerge in systems like slime mold

## How I Will Do This:
Analyze transitions with persistent homology and multiple approaches such as topology, information transport, and other qualitative measures

## Model:
Using the Keller-Segel model with extensions and variations. The first variation is the base Keller-Segel model with an added logistic term in the density field to prevent numerical blowup.

## Current Status:
* Implemented logistic growth Keller-Segel PDE solver with GPU acceleration and spectral methods for differentiation.
* Discovered and characterized sharp phase transition near sensitivity=4.10 in Keller-Segel model with logistic growth.

## Required Libraries:
1. NumPy
2. CuPy
3. VisPy
4. SciPy
5. Matplotlib
6. PyQt6 or PySide6