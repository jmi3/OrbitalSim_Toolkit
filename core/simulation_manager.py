import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from core.rkmethods import RKp


class ReturnedData:
    positions: list[np.ndarray]
    momenta: list[np.ndarray]


class SolverSettings:
    order = 4
    # timestep
    dt = 0.1
    # initial time
    t0 = 0
    # final time
    tmax = None

    # dydt = f(y,t) :D
    @staticmethod
    def dydt(t, x):
        return -2 * x

    # initial conditions at t0
    initial_conditions = 1


class SimulationSettings:
    total_energy = False
    kinetic_energy = False
    solar_ecliptic = False
    galactic_ecliptic = False
    motion_lines = True
    absolute_motion_line_length = 20
    range_x = (-1.2, 1.2)
    range_y = (-1.2, 1.2)
    range_z = (-1.2, 1.2)
    legend = None
    no_of_bodies = 10
    fps = 10
    steps_per_frame = 3

    def interval(self):
        return 1000 // self.fps

    def motion_line_length(self):
        return self.steps_per_frame * self.absolute_motion_line_length


class Simulation:
    def __init__(
        self,
        solver_class: RKp,
        settings: SimulationSettings,
        solver_settings: SolverSettings,
    ):
        self.solver_class = solver_class
        self.settings = settings
        self.solver_settings = solver_settings
        self.last_time = time.time()
        self.frame_count = 0

    def change_settings(self, settings: SimulationSettings):
        self.settings = settings

    def _initialize_solver(self):
        self.solver: RKp = self.solver_class(order=self.solver_settings.order)
        self.solver.initialize(
            y0=self.solver_settings.initial_conditions, dydt=self.solver_settings.dydt
        )
        self.solver.t0 = self.solver_settings.t0
        self.solver.t = self.solver_settings.t0
        self.solver.h = self.solver_settings.dt

    def _setup_figures(self):
        # Set up figure and axes for animation
        self.fig = plt.figure(figsize=(14, 8))
        self.grid = self.fig.add_gridspec(
            2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.4, hspace=0.3
        )
        if self.settings.total_energy and self.settings.kinetic_energy:
            self.ax_motion = self.fig.add_subplot(self.grid[:, 0], projection="3d")
            self.ax_tot = self.fig.add_subplot(self.grid[0, 1])
            self.ax_kin = self.fig.add_subplot(self.grid[1, 1])
        elif ((not self.settings.total_energy) and (self.settings.kinetic_energy)) or (
            (self.settings.total_energy) and (not self.settings.kinetic_energy)
        ):
            if self.settings.total_energy:
                self.ax_motion = self.fig.add_subplot(self.grid[:, 0], projection="3d")
                self.ax_tot = self.fig.add_subplot(self.grid[0, :])
            elif self.settings.kinetic_energy:
                self.ax_motion = self.fig.add_subplot(self.grid[:, 0], projection="3d")
                self.ax_kin = self.fig.add_subplot(self.grid[0, :])
        else:
            self.ax_motion = self.fig.add_subplot(self.grid[:, :], projection="3d")

        # Set up appearance of motion animation
        self.ax_motion.set_xlim(self.settings.range_x)
        self.ax_motion.set_ylim(self.settings.range_y)
        self.ax_motion.set_zlim(self.settings.range_z)
        self.ax_motion.set_title("Animation of the system")
        self.ax_motion.set_xlabel("x")
        self.ax_motion.set_ylabel("y")
        self.ax_motion.set_zlabel("z")

        # Plot optional planes if specified
        if self.settings.solar_ecliptic:
            xx, yy = np.meshgrid(
                np.linspace(*self.settings.range_x, 10),
                np.linspace(*self.settings.range_y, 10),
            )
            zz = yy / 2
            self.ax_motion.plot_surface(xx, yy, zz, alpha=0.1, color="magenta")
        if self.settings.galactic_ecliptic:
            xx, zz = np.meshgrid(
                np.linspace(*self.settings.range_x, 10),
                np.linspace(*self.settings.range_y, 10),
            )
            yy = np.zeros_like(zz)
            self.ax_motion.plot_surface(xx, yy, zz, alpha=0.1, color="cyan")

        # Plot Energies if requested
        if self.settings.kinetic_energy:
            # Set up kinetic energy plot
            self.ax_kin.set_title("Kinetic energies in the system")
            self.ax_kin.set_xlabel("t")
            self.ax_kin.set_ylabel("T")
            self.ax_kin.set_xlim(0, 10)
            self.ax_kin.set_ylim(0, 1)
        if self.settings.total_energy:
            # Set up total energy plot
            self.ax_tot.set_title("Total energy in the system")
            self.ax_tot.set_xlabel("t")
            self.ax_tot.set_ylabel("V")
            self.ax_tot.set_xlim(0, 10)
            self.ax_tot.set_ylim(0, 1)

        # Initialize plots for animation
        self.bodies = [
            self.ax_motion.plot([], [], [], "o", label=self.settings.legend[i], lw=2)[0]
            for i in range(self.settings.no_of_bodies)
        ]
        self.ax_motion.legend(loc="upper right")

        if self.settings.total_energy:
            self.tot_energy = self.ax_tot.plot([], [], "-", label="Total V ")[0]
        if self.settings.kinetic_energy:
            self.kin_energies = [
                self.ax_kin.plot([], [], "-", label=self.settings.legend[i], lw=2)[0]
                for i in range(self.settings.no_of_bodies)
            ]
            self.ax_kin.legend(loc="upper right")

        # Initialize motion lines for each object if "motion_lines" is in show
        if self.settings.motion_lines:
            self.motion_lines = [
                self.ax_motion.plot(
                    [], [], [], "-", alpha=0.3, color=self.bodies[i].get_color()
                )[0]
                for i in range(self.settings.no_of_bodies)
            ]

        # Text for displaying FPS
        self.fps_text = self.ax_motion.text2D(
            0.05, 0.95, "", transform=self.ax_motion.transAxes
        )

    def _initialize_simulation(self):
        for body in self.bodies:
            body.set_data([], [])
            body.set_3d_properties([])
        result = [*self.bodies]
        if self.settings.kinetic_energy:
            for kin in self.kin_energies:
                kin.set_data([], [])
                result.append(kin)

        if self.settings.total_energy:
            self.tot_energy.set_data([], [])
            result.append(self.tot_energy)

        if self.settings.motion_lines:
            for line in self.motion_lines:
                line.set_data([], [])
                line.set_3d_properties([])
                result.append(line)

        # Initialize FPS text
        self.fps_text.set_text("")
        result.append(self.fps_text)
        self.last_frames = 0
        return result

    def _update_simulation(self, positions):
        result = [*self.bodies]

        for i in range(len(self.bodies)):
            x_data = positions[i, :, 0]
            y_data = positions[i, :, 1]
            z_data = positions[i, :, 2]
            result[i].set_data(x_data[-1:], y_data[-1:])
            result[i].set_3d_properties(z_data[-1:])

            if self.settings.motion_lines:
                self.motion_lines[i].set_data(
                    x_data[-self.settings.motion_line_length :],
                    y_data[-self.settings.motion_line_length :],
                )
                self.motion_lines[i].set_3d_properties(
                    z_data[-self.settings.motion_line_length :]
                )
                self.motion_lines[i].set_alpha(0.3)
                result.append(self.motion_lines[i])

        return result

    def _update_energies(self, y):
        pass

    def _update(self, frame):
        qp_history = self.solver.get_history()
        np_qp_history = np.array(qp_history)
        positions, _ = np.split(np_qp_history.transpose(1, 0, 2), 2)

        result = self._update_simulation(positions=positions)

        # FPS tracking
        elapsed_time = time.time() - self.last_time
        if elapsed_time > 1.0:  # Update FPS every second
            fps = (
                (frame - self.last_frames)
                / elapsed_time
                / self.settings.steps_per_frame
            )
            self.fps_text.set_text(f"FPS: {fps:.2f}")
            self.last_time = time.time()
            self.last_frames = frame

        result.append(self.fps_text)

        return result

    def run_simulation(self):
        self._initialize_solver()
        self._setup_figures()

        def _data_gen():
            while True:
                for i in range(self.settings.steps_per_frame - 1):
                    self.solver.next_step()

                yield self.solver.next_step()

        self.animation = FuncAnimation(
            self.fig,
            self._update,
            frames=_data_gen,
            init_func=self._initialize_simulation,
            interval=self.settings.interval,
            blit=True,
        )
        plt.legend()
        plt.show()
