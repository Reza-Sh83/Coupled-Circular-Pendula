# Coupled Circular Pendula Simulation

This project simulates the dynamics of **N coupled pendula arranged in a circular configuration**, combining analytical and numerical approaches. It provides a detailed exploration of coupled harmonic motion, normal modes, and energy transfer in systems with rotational symmetry.

## Overview

The system consists of pendula connected by springs in a circular loop. This simulation explores:
- **Derivation of the Equations of Motion**: Based on Newtonian mechanics and the small-angle approximation.
- **Eigenvalue Analysis**: Determines the normal modes and angular frequencies of oscillation.
- **Numerical Integration**: Uses the Runge-Kutta method (`solve_ivp`) to simulate the system's evolution over time.
- **Visualization**: Includes static plots and a dynamic circular animation to visualize oscillatory behavior and energy transfer.

## Features

- **User Input**: Customize the number of pendula, their masses, spring constants, initial displacements and velocities, and simulation time.
- **Full Analytical Pipeline**: Constructs the coupling matrix, solves the eigenvalue problem, and reconstructs motion via normal mode superposition.
- **Numerical Solution**: Simultaneously integrates the full system of coupled ODEs for verification and comparison.
- **Insightful Visualization**:
  - Time evolution plots of angular displacements.
  - Decomposition into normal modes.
  - Circular animation showing synchronized movement of the coupled pendula.

## Installation

Ensure Python 3 is installed, along with the required libraries:

```bash
pip install numpy scipy matplotlib ipython
```

## Usage

1. **Run the Script:**
   ```bash
   python Coupled Circular Pendula.py
   ```

2. **Provide Input Parameters:**
   - Number of oscillators (default: 6)
   - Masses and spring constants for each oscillator (default: 1.0)
   - Initial angular displacements and velocities (default: 0.0)
   - Simulation time range (default: 0.0 to 20.0 seconds)

3. **View the Outputs:**
   - **Reconstructed Normal Modes**:
     - <img src="Reconstructed Motion Using Normal Modes of Coupled Oscillators.png" width="450">
   - **Numerical Simulation**:
     - <img src="Numerical Solution of Coupled Oscillators.png" width="450">
   - **Individual Mode Contributions**:
     - <img src="Each Normal Modes.png" width="450">

## Contributing

Contributions and suggestions are welcome! To propose changes:
- Fork the repository
- Create a new branch
- Submit a pull request with detailed explanation

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- Inspired by the study of coupled oscillations and symmetry in physical systems.
- Uses SciPy for solving differential equations and eigenvalue problems.
- Animated with Matplotlib for dynamic visualization.

