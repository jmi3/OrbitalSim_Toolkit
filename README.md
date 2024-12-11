# Orbital Simulation Toolkit

## Overview

The **Orbital Simulation Toolkit** is a Python-based framework for simulating and with a bit of good will even analyzing the dynamics of celestial bodies using higher-order Runge-Kutta (RK) methods. This whole thing emerged as my credit project for numerical methods class. Original assignment was central motion solver. So I took it a bit further. The toolkit provides:

- Support for integrating systems governed by Hamiltonian mechanics, including planetary systems and central forces.
- Visualization tools for animations and error analysis.
- A modular structure for adding new integrators and dynamical models.

The project is designed for researchers, educators, and enthusiasts in computational physics, astronomy, and numerical analysis.

## Features

- **High-order Runge-Kutta Integration:** Implementations of 1st to 6th order RK methods, including specialized variants like 5GPT and 6GPT.
- **Hamiltonian Dynamics:** Includes prebuilt Hamiltonians for Keplerian and general Newtonian interactions.
- **Visualization:**
  - 2D and 3D animations of simulated systems using `plotting_helpers` and `plotting_helpers_3D`.
  - Energy and error plots for integration analysis.
- **Predefined Scenarios:**
  - A 3D solar system simulation based on realistic initial conditions.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.8+
- Required Python libraries:
  ```bash
  pip install numpy matplotlib
  ```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/high-precision-orbital-simulation.git
   ```
2. Navigate to the project directory:
   ```bash
   cd high-precision-orbital-simulation
   ```

## Usage

### 1. Precomputing and Visualization

#### 2D Simulation and Visualization

Run `main.py` to precompute data for a 2D simulation and visualize the results:

```bash
python main.py
```

This will compute trajectories using a chosen RK method and animate them. The script also provides options for plotting energy or angular momentum over time.

#### 3D Simulation and Visualization

Run `main3D.py` to precompute data for a 3D simulation and visualize the results:

```bash
python main3D.py
```

Similar to the 2D case, this computes trajectories and visualizes them with additional 3D-specific features, such as spatial orientation and optional motion trails.

### 2. Error Analysis

Use the `precision_testing.py` module to analyze the accuracy of various RK methods.

```python
from precision_testing import CompareSolutionErrors

# Parameters for error analysis
ICs = ...  # Initial conditions
Ns = [100, 200, 400]  # Steps to test
Ps = [1, 2, 3, 4]  # RK orders to test

CompareSolutionErrors(ICs, t0=0, tmax=10, Ns=Ns, Ps=Ps)
```

### 3. Animation and Visualization

Use `plotting_helpers` for 2D visualization and `plotting_helpers_3D` for 3D visualization.

#### 2D Animation

```python
from plotting_helpers import animate_with_energy_Newton

animate_with_energy_Newton(
    positions, momenta, masses=masses,
    central_body_index=3, xlim=(-1e9, 1e9), ylim=(-1e9, 1e9),
    dt=1000, interval=50, names=['Body1', 'Body2']
)
```

#### 3D Animation

```python
from plotting_helpers_3D import animate_with_energy_Newton_3D

animate_with_energy_Newton_3D(
    positions, momenta, masses=masses,
    central_body_index=3, xlim=(-1e9, 1e9), ylim=(-1e9, 1e9), zlim=(-1e9, 1e9),
    dt=1000, interval=50, names=['Body1', 'Body2'], show=['motion_lines']
)
```

## Project Structure

```
.
├── precision_testing.py       # Error analysis and visualization
├── rkmethods.py               # High-order Runge-Kutta implementations
├── RKTables.py                # RK coefficient tables
├── simulation_manager.py      # Solar system simulation driver
├── hamilton.py                # Hamiltonian models
├── plotting_helpers.py        # 2D visualization tools
├── plotting_helpers_3D.py     # 3D visualization tools
├── main.py                    # 2D simulation example
├── main3D.py                  # 3D simulation example
├── solar_system_3d.json       # Initial conditions for solar system simulation
```

## Configuration

Modify `main.py` or `main3D.py` to adjust settings such as timestep, initial conditions, and visualization parameters. For example:

```python
size = 2e9
RK4 = RKp(order=4)
RK4.Initialize(y0=ics, dydt=f)
RK4.Integrate(t0=0, tmax=1e7, h=1e4)
```

## To-Do List

- **Integration of CuPy for GPU acceleration** to speed up computations.
- Expand 3D visualization options (e.g., customizable views, additional energy plots).
- Add support for external celestial datasets.
- Refine loaders from .bsp and similar files from SPICE.

## Contributions

This project is currently **closed for external contributions without previous consultation**. Future updates will be managed by the project maintainer. To be added to contributors, please contact me on [jancervenan@gmail.com](mailto\:jancervenan@gmail.com).

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Authors

- **Jan Červeňan** - [GitHub Profile](https://github.com/jmi3)

## Acknowledgments

- Open-source contributors.
- Inspiration from classical mechanics and numerical analysis.
- Data from SPICE/NAIF database, from `de440.bsp`, `gm_de440.bsp` and similar files.

---

Happy simulating!

