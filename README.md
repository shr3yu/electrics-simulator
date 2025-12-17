# Semiconductor Physics Simulator: Charge Carrier Dynamics
A real-time, low-level C++ simulation of electron behavior in a semiconductor crystal lattice under varying electromagnetic and thermal conditions.

https://github.com/user-attachments/assets/82f124a5-2192-4ebb-8a93-31b052249b53


## Overview
This project simulates the microscopic movement of electrons within a semiconductor material. By modeling the transition between the Valence Band and the Conduction Band, the simulator visualizes how temperature and voltage influence electrical conductivity.

## Concepts Applied
1. Thermal Excitation:
    - Models the probability of electrons jumping the Band Gap using a temperature-dependent stochastic model ($P \propto T$)
  
2. Recombination & Generation:
    - Simulates the life cycle of charge carriers as they transition from free-moving conduction states back to bound valence states.
  
3. Drift-Diffusion Model:
    - Electrons exhibit "Drift Velocity" in response to an applied Electric Field (Voltage), while "Thermal Jitter" simulates Brownian-like diffusion
  
4. Hardware Acceleration:
     - Utilizes OpenGL for rendering, passing simulation data directly to the GPU via Vertex Buffer Objects (VBOs).

## Tech Stack
* C++11/17
* OpenGL 3.3
* GLSL
* GLFW/GLAD
* ImGUI

## Future Enhancements
1. P-N Junction logic to simulate diode rectification
2. Integration of the Boltzmann Factor (instead of the simplified linear relationship of P and T)

