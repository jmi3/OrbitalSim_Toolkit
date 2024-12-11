# Orbital Simulation Toolkit

## Overview

The **Orbital Simulation Toolkit** is a Python-based framework for simulating and with a bit of good will even analyzing the dynamics of celestial bodies using arbitrary-order Runge-Kutta (RK) methods. This whole thing emerged as my credit project for numerical methods class. Original assignment was to make a central motion solver, so I took it a bit further. The toolkit provides:

- Support for integrating systems from classical Hamiltonian mechanics, focus on planetary systems and central forces.
- Visualization tools for animations and error analysis.
- A modular structure for adding new integrators and dynamical models.

The project can prove useful/interesting for educators and enthusiasts in computational physics, astronomy, and numerical analysis.

## Features

- **User-defined Runge-Kutta Integration:** Implementations of 1st to 6th order RK methods, possibility to add own Buchter tableau's.
- **Hamiltonian Dynamics:** Includes prebuilt Hamiltonians for Keplerian and general Newtonian interactions. It is easy to add your own.
- **Visualization:**
  - Possibility to visualise computed data through some pre-written visualising scripts.
  - Energy and error plots for integration analysis.
- **Predefined Scenarios:**
  - A 3D and 2D solar system simulation based on realistic initial conditions from `de440.bsp` file.
  - TODO: Simple extractor of such data from SPICE database into `*.json` files.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.8+
- Python libraries from `requirements.txt` file.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jmi3/OrbitalSim_Toolkit
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

This will compute trajectories using an in-file chosen RK method and animate them. The script also provides options for plotting energy or angular momentum over time. Edit the script to obtain different results.

#### 3D Simulation and Visualization

Run `main3D.py` to precompute data for a 3D simulation and visualize the results:

```bash
python main3D.py
```

Similar to the 2D case, this computes trajectories and visualizes them with additional 3D-specific features, such as rough ecliptic position (very approximate in current version) and motion trails.

### 2. Error Analysis

Use the `precision_testing.py` module to study the accuracy of various RK methods.

```python
from precision_testing import CompareSolutionErrors

# Parameters for error analysis
ICs = ...  # Initial conditions
Ns = [100, 200, 400]  # Steps to test
Ps = [1, 2, 3, 4]  # RK orders to test

CompareSolutionErrors(ICs, t0=0, tmax=10, Ns=Ns, Ps=Ps)
```

### 3. Animation and Visualization

Use `plotting_helpers` for 2D visualization and `plotting_helpers_3D` for 3D visualization of pre-computed data. Sample usage is in `main(3D).py` files.

#### 2D Animation

```python
from plotting_helpers import animate_with_energy_Newton

# Calculate the positions and momenta here.
# Do not forget to load the masses from .json files for energy computation

animate_with_energy_Newton(
    positions, momenta, masses=masses,
    central_body_index=3, xlim=(-1e9, 1e9), ylim=(-1e9, 1e9),
    dt=1000, interval=50, names=['Body1', 'Body2']
)
```

#### 3D Animation

```python
from plotting_helpers_3D import animate_with_energy_Newton_3D

# Calculate the positions and momenta here.
# Do not forget to load the masses from .json files for energy computation

animate_with_energy_Newton_3D(
    positions, momenta, masses=masses,
    central_body_index=3, xlim=(-1e9, 1e9), ylim=(-1e9, 1e9), zlim=(-1e9, 1e9),
    dt=1000, interval=50, names=['Body1', 'Body2'], show=['motion_lines']
)
```

### Real-time simulation (no precomputing needed)



## Project Structure

```
.
├── core/
│   ├── rkmethods.py            # High-order Runge-Kutta implementations
│   ├── RKTables.py             # RK coefficient tables
│   ├── hamilton.py             # Hamiltonian models
│   ├── simulation_manager.py   # Live simulation tool
├── visualization/
│   ├── plotting_helpers.py     # 2D visualization tools
│   ├── plotting_helpers_3D.py  # 3D visualization tools    
├── examples/
│   ├── main.py                 # 2D simulation example
│   ├── main3D.py               # 3D simulation example
│   ├── precision_testing.py    
├── data/
│   ├── solar_system_3d.json
│   ├── solar_system_2d.json    
├── README.md                   
├── LICENSE
├── requirements.txt            # Python package dependencies
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

This project is currently **closed for external contributions without previous consultation**. Future updates will be managed by the project maintainer and specified contributors. To be added as contributor, please contact me on [jancervenan@gmail.com](mailto\:jancervenan@gmail.com).

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

