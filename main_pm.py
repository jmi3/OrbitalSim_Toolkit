import numpy as np
import matplotlib.pyplot as plt
import json

from pm_hamiltonian import PMHamiltonian
from core.rkmethods import RKp
from helpers.plotting_helpers import animate_rkp_motion


class dHdQP:
    def __init__(self, masses, H: PMHamiltonian):
        self.masses = masses
        self.H = H

    def __call__(self, t, QP):
        [Q, P] = np.split(QP,2)
        # dq/dt = dH/dp , dp/dt = -dH/dq
        return np.append(self.H.dHdp(self.masses, P), self.H.dHdq(self.masses, Q),axis=0)
  

def obtain_ics(fp, objs=None):
    eph_data = json.load(fp)
    temp_m = []
    temp_Q = []
    temp_P = []
    for eph in eph_data.values():
        temp_m.append(eph["m"])
        temp_Q.append(np.array(eph["q"]))
        temp_P.append(np.array(eph["p"]))
        
    objs = objs or len(temp_m)
    QP = np.append(np.array(temp_Q[:objs]), np.array(temp_P[:objs]),axis=0)
    return np.array(temp_m[:objs]), QP, list(eph_data.keys())[:objs]


with open("database/solar_system_2d.json") as file:
    masses, ics, names = obtain_ics(fp=file, objs=3)


# setup integration parameters
t0 = 0
h = 8400
tmax = 250*h


# prepare f to simulate
size = 1.5e8
cells = 256
f = dHdQP(masses = masses, H = PMHamiltonian)
f.H.SetParameters(G=6.67430e-20, workers=4, mesh_size=(cells, cells), dx=size/cells)

RK4 = RKp(order=2)

# adjust ics to be positive for PM method
[q0, p0] = np.split(ics, 2)
ics = np.append(q0 + size, p0, axis=0)

# integrate 
RK4.Initialize(y0=ics, dydt=f)
RK4.Integrate(t0=t0, tmax=tmax, h=h)
QP_history = RK4.GetHistory()
npQP_history = np.array(QP_history)
positions, momenta = np.split(npQP_history.transpose(1,0,2),2)

animate_rkp_motion(
    positions=positions[:,::10],
    masses=masses,
    central_body_index=0,
    mass_centre_view=False,  # or False
    central_body_view=True,  # or True
    xlim=(-size, size),
    ylim=(-size, size),
    interval=50,
    output_filename="innersolar_2d_pm.gif",
)

# plot_rkp_solutions(
#           rkp_solvers=[RKp(order=i) for i in {1,2,4}],
#           initial_conditions=ics,
#           dydt=f,
#           t0=t0,
#           tmax=tmax,
#           h=h,
#           masses=masses,
#           central_body_index=0,
#           mass_centre_view=False,  # or False
#           central_body_view=True,  # or True
#           xlim=(-size, size),
#           ylim=(-size, size),
#           names = names
#      )