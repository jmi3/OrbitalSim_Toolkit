import json

import numpy as np

from core.hamilton import NewtonHamiltonian
from core.rkmethods import RKp
from helpers import (
    plot_rkp_solutions_3d,
)


class MotionEquations:
    def __init__(self, masses, hamiltonian: NewtonHamiltonian):
        self.masses = masses
        self.hamiltonian = hamiltonian

    def __call__(self, t, qp):
        newton_constant = 6.67430e-20  # km^3 kg^-1 s^-2
        [positions, momenta] = np.split(qp, 2)
        # dq/dt = dH/dp , dp/dt = -dH/dq
        return np.append(
            self.hamiltonian.momentum_derivative(self.masses, momenta),
            -newton_constant
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

    qp = np.append(np.array(temp_positions[:6]), np.array(temp_momenta[:6]), axis=0)
    return np.array(temp_masses[:6]), qp, list(eph_data.keys())[:6]


with open("database/solar_system_3d.json") as file:
    masses, ics, names = obtain_ics(fp=file)


rkp_solvers = {i: RKp(order=i) for i in RKp.get_all_implemented()}

# setup integration parameters
t0 = 0
h = 8400
tmax = 7800 * h


# prepare f to simulate
f = MotionEquations(masses=masses, hamiltonian=NewtonHamiltonian)

size = 2e8

RK4 = RKp(order=4)

# integrate
RK4.initialize(y0=ics, dydt=f)
RK4.integrate(t0=t0, tmax=tmax, h=h)
qp_history = RK4.get_history()
np_qp_history = np.array(qp_history)
positions, momenta = np.split(np_qp_history.transpose(1, 0, 2), 2)


# animate_Newton_3D(
#     positions=positions[:,::20],
#     momenta=momenta[:,::20],
#     dt=h,
#     masses=masses,
#     central_body_index=3,
#     mass_centre_view=False,  # or False
#     central_body_view=False,  # or True
#     xlim=(-size, size),
#     ylim=(-size, size),
#     zlim=(-size, size),
#     interval=50,
#     names=names,
#     show=["motion_lines"],
#     motion_line_length=100,
#     output_filename="Inner solar system.gif",
#     ani_title = "Inner solar system"
# )


plot_rkp_solutions_3d(
    rkp_solvers=[RKp(order=i) for i in {1, 2}],
    initial_conditions=ics,
    dydt=f,
    t0=t0,
    tmax=tmax,
    h=h,
    masses=masses,
    central_body_index=3,
    mass_centre_view=False,  # or False
    central_body_view=True,  # or True
    xlim=(-size, size),
    ylim=(-size, size),
    zlim=(-size, size),
    names=names,
)
