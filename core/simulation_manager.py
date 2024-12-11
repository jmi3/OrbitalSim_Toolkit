import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.animation import FuncAnimation, PillowWriter
from hamilton import NewtonHamiltonian, KeplerHamiltonian
from rkmethods import RKp

class ReturnedData:
     positions = []
     momenta = []
     

class SolverSettings:
     order = 4
     # timestep
     dt = 0.1
     # initial time
     t0 = 0
     # final time
     tmax = None
     # dydt = f(y,t) :D
     dydt = lambda t,x: - 2 * x
     # initial conditions at t0
     ICs = 1


class SimulationSettings:
     TotalEnergy = False
     KineticEnergy = False
     SolarEcliptic = False
     GalacticEcliptic = False
     MotionLines = True
     AbsoluteMotionLineLength = 20
     RangeX = (-1.2, 1.2)
     RangeY = (-1.2, 1.2)
     RangeZ = (-1.2, 1.2)
     Legend = None
     NoOfBodies = 10
     FPS = 10
     StepsPerFrame = 3
     @property
     def Interval(self):
          return 1000//self.FPS
     @property
     def MotionLineLength(self):
          return self.StepsPerFrame*self.AbsoluteMotionLineLength
import time

class Simulation:
     def __init__(self, solver_class: RKp, settings: SimulationSettings, solver_settings: SolverSettings):
          self.solver_class = solver_class
          self.settings = settings
          self.solver_settings = solver_settings
          self.last_time = time.time()
          self.frame_count = 0

     def ChangeSettings(self, settings: SimulationSettings):
          self.settings = settings

     def _InitializeSolver(self):
          self.solver : RKp = self.solver_class(order = self.solver_settings.order)
          self.solver.Initialize(y0=self.solver_settings.ICs, dydt=self.solver_settings.dydt)
          self.solver.t0 = self.solver_settings.t0
          self.solver.t = self.solver_settings.t0
          self.solver.h = self.solver_settings.dt

     def _SetupFigures(self):
          # Set up figure and axes for animation
          self.fig = plt.figure(figsize=(14, 8))
          self.grid = self.fig.add_gridspec(2, 2, 
                         width_ratios=[1, 1], height_ratios=[1, 1], 
                         wspace=0.4, hspace=0.3)
          if self.settings.TotalEnergy and self.settings.KineticEnergy:
               self.ax_motion = self.fig.add_subplot(self.grid[:, 0], projection='3d')
               self.ax_tot = self.fig.add_subplot(self.grid[0, 1])
               self.ax_kin = self.fig.add_subplot(self.grid[1, 1])
          elif (((not self.settings.TotalEnergy) and (self.settings.KineticEnergy)) 
                or ((self.settings.TotalEnergy) and (not self.settings.KineticEnergy))):
               if self.settings.TotalEnergy:
                    self.ax_motion = self.fig.add_subplot(self.grid[:, 0], projection='3d')
                    self.ax_tot = self.fig.add_subplot(self.grid[0, :])
               elif self.settings.KineticEnergy:
                    self.ax_motion = self.fig.add_subplot(self.grid[:, 0], projection='3d')
                    self.ax_kin = self.fig.add_subplot(self.grid[0, :])
          else: 
               self.ax_motion = self.fig.add_subplot(self.grid[:,:], projection='3d')

          # Set up appearance of motion animation
          self.ax_motion.set_xlim(self.settings.RangeX)
          self.ax_motion.set_ylim(self.settings.RangeY)
          self.ax_motion.set_zlim(self.settings.RangeZ)
          self.ax_motion.set_title("Animation of the system")
          self.ax_motion.set_xlabel("x")
          self.ax_motion.set_ylabel("y")
          self.ax_motion.set_zlabel("z")

          # Plot optional planes if specified
          if self.settings.SolarEcliptic:
               xx, yy = np.meshgrid(
                    np.linspace(*self.settings.RangeX, 10), 
                    np.linspace(*self.settings.RangeY, 10)
                    )
               zz = yy / 2
               self.ax_motion.plot_surface(xx, yy, zz, 
                                           alpha=0.1, color='magenta')
          if self.settings.GalacticEcliptic:
               xx, zz = np.meshgrid(
                    np.linspace(*self.settings.RangeX, 10), 
                    np.linspace(*self.settings.RangeY, 10)
                    )
               yy = np.zeros_like(zz)
               self.ax_motion.plot_surface(xx, yy, zz, 
                                           alpha=0.1, color='cyan')

          # Plot Energies if requested
          if self.settings.KineticEnergy:
               # Set up kinetic energy plot
               self.ax_kin.set_title("Kinetic energies in the system")
               self.ax_kin.set_xlabel("t")
               self.ax_kin.set_ylabel("T")
               self.ax_kin.set_xlim(0, 10)
               self.ax_kin.set_ylim(0, 1)
          if self.settings.TotalEnergy:
               # Set up total energy plot
               self.ax_tot.set_title("Total energy in the system")
               self.ax_tot.set_xlabel("t")
               self.ax_tot.set_ylabel("V")
               self.ax_tot.set_xlim(0, 10)
               self.ax_tot.set_ylim(0, 1)
          
          # Initialize plots for animation
          self.bodies = [
               self.ax_motion.plot([], [], [], 'o', label=self.settings.Legend[i], lw=2)[0] 
               for i in range(self.settings.NoOfBodies)
               ]
          self.ax_motion.legend(loc="upper right")

          if self.settings.TotalEnergy:
               self.tot_energy = self.ax_tot.plot([], [], '-', label="Total V ")[0]
          if self.settings.KineticEnergy:
               self.kin_energies = [
                    self.ax_kin.plot([], [], '-', label=self.settings.Legend[i], lw=2)[0] 
                    for i in range(self.settings.NoOfBodies)
                    ]
               self.ax_kin.legend(loc="upper right")
               
          # Initialize motion lines for each object if "motion_lines" is in show
          if self.settings.MotionLines:
               self.motion_lines = [
                    self.ax_motion.plot([], [], [], '-', alpha=0.3, color=self.bodies[i].get_color())[0] 
                    for i in range(self.settings.NoOfBodies)
                    ]

          # Text for displaying FPS
          self.fps_text = self.ax_motion.text2D(0.05, 0.95, '', transform=self.ax_motion.transAxes)

     def _InitializeSimulation(self):
     
          for body in self.bodies:
               body.set_data([],[])
               body.set_3d_properties([])
          result = [*self.bodies]
          if self.settings.KineticEnergy:
               for kin in self.kin_energies:
                    kin.set_data([], [])
                    result.append(kin)
          
          if self.settings.TotalEnergy:
               self.tot_energy.set_data([], [])
               result.append(self.tot_energy)

          if self.settings.MotionLines:
               for line in self.motion_lines:
                    line.set_data([], [])
                    line.set_3d_properties([])
                    result.append(line)
          
          # Initialize FPS text
          self.fps_text.set_text('')
          result.append(self.fps_text)
          self.last_frames = 0
          return result

     def _UpdateSimulation(self, positions):

          result = [*self.bodies]

          for i in range(len(self.bodies)):
               x_data = positions[i,:,0 ]
               y_data = positions[i,:,1]
               z_data = positions[i,:,2] 
               result[i].set_data(x_data[-1:], y_data[-1:])
               result[i].set_3d_properties(z_data[-1:])
               
               if self.settings.MotionLines:
                    self.motion_lines[i].set_data(x_data[-self.settings.MotionLineLength:], y_data[-self.settings.MotionLineLength:])
                    self.motion_lines[i].set_3d_properties(z_data[-self.settings.MotionLineLength:])
                    self.motion_lines[i].set_alpha(0.3)
                    result.append(self.motion_lines[i])

          return result


     def _UpdateEnergies(self, y):
          pass

     def _Update(self, frame):
          QP_history = self.solver.GetHistory()
          npQP_history = np.array(QP_history)
          positions, momenta = np.split(npQP_history.transpose(1,0,2),2)

          result = self._UpdateSimulation(positions=positions)

          # FPS tracking
          elapsed_time = time.time() - self.last_time
          if elapsed_time > 1.0:  # Update FPS every second
               fps = (frame - self.last_frames) / elapsed_time /self.settings.StepsPerFrame
               self.fps_text.set_text(f'FPS: {fps:.2f}')
               self.last_time = time.time()
               self.last_frames = frame

          result.append(self.fps_text)

          return result

     def Run(self):
          self._InitializeSolver()
          self._SetupFigures()
          def _data_gen():
               while True:
                    for i in range(self.settings.StepsPerFrame-1):
                         self.solver.NextStep()

                    yield self.solver.NextStep()

          self.animation = FuncAnimation(
               self.fig,
               self._Update,
               frames= _data_gen,
               init_func=self._InitializeSimulation,
               interval = self.settings.Interval,
               blit=True
               )
          plt.legend()
          plt.show()


if __name__ == "__main__":
          
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

