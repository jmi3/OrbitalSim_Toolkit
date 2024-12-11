import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.animation import FuncAnimation
from hamilton import NewtonHamiltonian
from rkmethods import RKp
from plotting_helpers import animate_with_energy_Newton, plot_rkp_solutions


class dHdQP:
    def __init__(self, masses, H: NewtonHamiltonian):
        self.masses = masses
        self.H = H

    def __call__(self, t, QP):
        G = 6.67430e-20  # km^3 kg^-1 s^-2
        [Q, P] = np.split(QP,2)
        # dq/dt = dH/dp , dp/dt = -dH/dq
        return np.append(self.H.dHdp(self.masses, P),-G*self.H.dHdq(self.masses, Q),axis=0)
  

def obtain_ics(fp):
    eph_data = json.load(fp)
    temp_m = []
    temp_Q = []
    temp_P = []
    for eph in eph_data.values():
        temp_m.append(eph["m"])
        temp_Q.append(np.array(eph["q"]))
        temp_P.append(np.array(eph["p"]))

    QP = np.append(np.array(temp_Q[:6]), np.array(temp_P[:6]),axis=0)
    return np.array(temp_m[:6]), QP, list(eph_data.keys())[:6]


with open("database/solar_system_2d.json") as file:
    masses, ics, names = obtain_ics(fp=file)


rkp_solvers = {i:RKp(order=i) for i in RKp.GetAllImplemented()}

# setup integration parameters
t0 = 0
h = 8400
tmax = 10000*h


# prepare f to simulate
f = dHdQP(masses = masses, H = NewtonHamiltonian)

size = 2.5e8

RK4 = RKp(order=2)
    
# integrate 
RK4.Initialize(y0=ics, dydt=f)
RK4.Integrate(t0=t0, tmax=tmax, h=h)
QP_history = RK4.GetHistory()
npQP_history = np.array(QP_history)
positions, momenta = np.split(npQP_history.transpose(1,0,2),2)


animate_with_energy_Newton(
    positions=positions[:,::50],
    momenta=momenta[:,::50],
    dt=h,
    masses=masses,
    central_body_index=3,
    mass_centre_view=False,  # or False
    central_body_view=True,  # or True
    xlim=(-size, size),
    ylim=(-size, size),
    interval=50,
    names=names
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