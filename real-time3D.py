import numpy as np
import json 

from core.hamilton import NewtonHamiltonian
from core.simulation_manager import Simulation, SimulationSettings, SolverSettings
from core.rkmethods import RKp

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
     objs = len(eph_data.values())
     QP = np.append(np.array(temp_Q[:objs]), np.array(temp_P[:objs]),axis=0)
     return np.array(temp_m[:objs]), QP, list(eph_data.keys())[:objs]


with open("database/solar_system_3d.json") as file:
     masses, ics, names = obtain_ics(fp=file)


# setup integration parameters
solSet = SolverSettings()

solSet.t0 = 0
solSet.dt = 1050
solSet.tmax =7800*solSet.dt

solSet.ICs = ics

solSet.order = 4


# prepare f to simulate
solSet.dydt = dHdQP(masses = masses, H = NewtonHamiltonian)

#######################
size = 5e9

sett = SimulationSettings()
##############################
# Set shown figures
sett.SolarEcliptic = False
sett.GalacticEcliptic = False

# Set size of drawn area
sett.RangeX = (-size, size)
sett.RangeY = (-size, size)
sett.RangeZ = (-size, size)

# Set FPS and stepcount
sett.FPS = 10
sett.StepsPerFrame = 1000

# Set names of bodies and line lengths
sett.NoOfBodies = len(names)
sett.Legend = names
sett.AbsoluteMotionLineLength = 100


mngr = Simulation(solver_class=RKp, settings=sett, solver_settings=solSet) 
mngr.Run()

