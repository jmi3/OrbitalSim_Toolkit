import numpy as np
import json
from pm_hamiltonian import PMHamiltonian
from core.rkmethods import RKp
from helpers.plotting_helpers_3D import animate_Newton_3D

class dHdQP:
    def __init__(self, masses, H: PMHamiltonian):
        self.masses = masses
        self.H = H

    def __call__(self, t, QP):
        [Q, P] = np.split(QP, 2)
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


with open("database/solar_system_3d.json") as file:
    masses, ics, names = obtain_ics(fp=file, objs=4)


# prepare f to simulate
size = 1e9

npQP_history = np.load("numpy_checkpoint.npy")
positions, momenta = np.split(npQP_history.transpose(1,0,2),2)


animate_Newton_3D(
    positions=positions[:,::20],
    masses=masses,
    central_body_index=0,
    mass_centre_view=True,  # or False
    central_body_view=False,  # or True
    xlim=(-size, size),
    ylim=(-size, size),
    zlim=(-size, size),
    interval=50,
    names=names,
    show=["motion_lines"],
    motion_line_length=100,
    output_filename="Inner solar system.gif",
    ani_title = "Inner solar system"
)


# plot_rkp_solutions_3D(
#           rkp_solvers=[RKp(order=i) for i in {1,2}],
#           initial_conditions=ics,
#           dydt=f,
#           t0=t0,
#           tmax=tmax,
#           h=h,
#           masses=masses,
#           central_body_index=3,
#           mass_centre_view=False,  # or False
#           central_body_view=True,  # or True
#           xlim=(-size, size),
#           ylim=(-size, size),
#           zlim=(-size, size),
#           names = names
#      )