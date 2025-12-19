import json

import numpy as np

from core.hamilton import NewtonHamiltonian
from core.rkmethods import RKp
from core.simulation_manager import Simulation, SimulationSettings, SolverSettings


class MotionEquations:
    def __init__(self, masses, hamiltonian: NewtonHamiltonian):
        self.masses = masses
        self.hamiltonian = hamiltonian

    def __call__(self, t, qp):
        newton_constants = 6.67430e-20  # km^3 kg^-1 s^-2
        [positions, momenta] = np.split(qp, 2)
        # dq/dt = dH/dp , dp/dt = -dH/dq
        return np.append(
            self.hamiltonian.momentum_derivative(self.masses, momenta),
            -newton_constants
            * self.hamiltonian.position_derivative(self.masses, positions),
            axis=0,
        )


def obtain_ics(fp):
    eph_data = json.load(fp)
    temp_masses = []
    temp_positions = []
    temp_momenta = []
    for eph in eph_data.values():
        temp_masses.append(eph["m"])
        temp_positions.append(np.array(eph["q"]))
        temp_momenta.append(np.array(eph["p"]))
    objs = len(eph_data.values())
    qp = np.append(
        np.array(temp_positions[:objs]), np.array(temp_momenta[:objs]), axis=0
    )
    return np.array(temp_masses[:objs]), qp, list(eph_data.keys())[:objs]


with open("database/solar_system_3d.json") as file:
    masses, ics, names = obtain_ics(fp=file)


# setup integration parameters
solver_settings = SolverSettings()

solver_settings.t0 = 0
solver_settings.dt = 1050
solver_settings.tmax = 7800 * solver_settings.dt

solver_settings.initial_conditions = ics

solver_settings.order = 4


# prepare f to simulate
solver_settings.dydt = MotionEquations(masses=masses, hamiltonian=NewtonHamiltonian)

#######################
size = 5e9

sett = SimulationSettings()
##############################
# Set shown figures
sett.solar_ecliptic = False
sett.galactic_ecliptic = False

# Set size of drawn area
sett.range_x = (-size, size)
sett.range_y = (-size, size)
sett.range_z = (-size, size)

# Set FPS and stepcount
sett.fps = 10
sett.steps_per_frame = 1000

# Set names of bodies and line lengths
sett.no_of_bodies = len(names)
sett.legend = names
sett.absolute_motion_line_length = 100


mngr = Simulation(solver_class=RKp, settings=sett, solver_settings=solver_settings)
mngr.run_simulation()
