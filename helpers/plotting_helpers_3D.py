import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.animation import FuncAnimation, PillowWriter
from hamilton import NewtonHamiltonian, KeplerHamiltonian
from rkmethods import RKp



def plot_rkp_solutions_3D(rkp_solvers, initial_conditions, dydt, t0, tmax, h,
                          masses=None, central_body_index=None, mass_centre_view=False, central_body_view=False,
                          xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), zlim=(-1.2, 1.2), names=None, gridshape=None, sizeOfFig=5):
    """
    Plots RKp solutions for all given rkp_solvers in 3D.
    Returns void.
    Shows graphs containing evolution of the same system using all the rkp_solvers

    Args:
        rkp_solvers (List of RKp): Takes in array of RKps
        initial_conditions (ndarray): array of shape (2k,3) for k bodies in 3 dimensions
        dydt (function): function from RK
        t0 (float): starting time
        tmax (float): final time
        h (float): timestep
        masses (ndarray, optional): array of shape (k,) with masses. Defaults to None.
        central_body_index (int, optional): If central view is on, this represents central body. Defaults to None.
        mass_centre_view (bool, optional): Centers the view on mass center. Defaults to False.
        central_body_view (bool, optional): Centers the view on given body. Defaults to False.
        xlim (tuple, optional): Range in x to be plotted. Defaults to (-1.2, 1.2).
        ylim (tuple, optional): Range in y to be plotted. Defaults to (-1.2, 1.2).
        zlim (tuple, optional): Range in z to be plotted. Defaults to (-1.2, 1.2).
    """
    if gridshape is None:
        gridshape = (1, len(rkp_solvers))
    # Prepare the plot size and layout
    fig = plt.figure(figsize=(sizeOfFig * gridshape[1], sizeOfFig * gridshape[0]))
    axes = [fig.add_subplot(gridshape[0], gridshape[1], i + 1, projection='3d') for i in range(len(rkp_solvers))]

    if len(rkp_solvers) == 1:
        axes = [axes]

    for ax, solver in zip(axes, rkp_solvers):
        # Initialize and integrate
        solver.Initialize(y0=initial_conditions, dydt=dydt)
        solver.Integrate(t0=t0, tmax=tmax, h=h)
        QP_history = solver.GetHistory()
        npQP_history = np.array(QP_history)
        positions, momenta = np.split(npQP_history.transpose(1, 0, 2), 2)

        # Determine shifts
        if mass_centre_view and masses is not None:
            shift_x = (masses[:, np.newaxis] * positions[:, :, 0]).sum(axis=0) / masses.sum(axis=0)
            shift_y = (masses[:, np.newaxis] * positions[:, :, 1]).sum(axis=0) / masses.sum(axis=0)
            shift_z = (masses[:, np.newaxis] * positions[:, :, 2]).sum(axis=0) / masses.sum(axis=0)
        elif central_body_view and central_body_index is not None:
            shift_x = positions[central_body_index, :, 0]
            shift_y = positions[central_body_index, :, 1]
            shift_z = positions[central_body_index, :, 2]
        else:
            shift_x = np.zeros_like(positions[0, :, 0])
            shift_y = np.zeros_like(positions[0, :, 1])
            shift_z = np.zeros_like(positions[0, :, 2])

        if names is None:
            names = [f"Object {i}" for i in range(len(positions))]

        # Plot positions
        for i in range(len(positions)):
            pos = positions[i]
            x_axis = pos[:, 0] - shift_x
            y_axis = pos[:, 1] - shift_y
            z_axis = pos[:, 2] - shift_z
            ax.plot(x_axis, y_axis, z_axis, label=names[i])

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_title(f"RK{solver.order} solver")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def animate_with_energy_Newton_3D(positions, momenta, masses=None, central_body_index=None,
                                  mass_centre_view=False, central_body_view=False,
                                  xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), zlim=(-1.2, 1.2), dt=1, interval=5,
                                  names=None, show=[], motion_line_length=20):
    """
    Runs animated version of the given positions history array in 3D. Introduces energies as separate graphs.

    Args:
        positions (arr): Chronologically ordered positions of the bodies.
        masses (arr, optional): Masses of the bodies. Defaults to None.
        central_body_index (int, optional): If central view is on, this represents the central body. Defaults to None.
        mass_centre_view (bool, optional): Centers the view on the mass center. Defaults to False.
        central_body_view (bool, optional): Centers the view on the given body. Defaults to False.
        xlim (tuple, optional): Range in x to be plotted. Defaults to (-1.2, 1.2).
        ylim (tuple, optional): Range in y to be plotted. Defaults to (-1.2, 1.2).
        zlim (tuple, optional): Range in z to be plotted. Defaults to (-1.2, 1.2).
        dt (int, optional): Time step for the animation. Defaults to 1.
        interval (int, optional): How many milliseconds per frame. Defaults to 50.
        names (list, optional): Names of the bodies. Defaults to None.
        show (list, optional): Features to be visualized in the animation. Options include "solar_ecliptic", "galactic_ecliptic", and "motion_lines". Defaults to [].
        motion_line_length (int, optional): Number of steps for which motion lines (trajectories) are visible. Defaults to 20.

    Returns:
        None
    """

    # Determine shifts for centering if needed
    num_objects, num_steps, _ = positions.shape
    if names is None:
        names = [f"Object {i}" for i in range(len(positions))]
    if mass_centre_view and masses is not None:
        shift_x = (masses[:, np.newaxis] * positions[:, :, 0]).sum(axis=0) / masses.sum(axis=0)
        shift_y = (masses[:, np.newaxis] * positions[:, :, 1]).sum(axis=0) / masses.sum(axis=0)
        shift_z = (masses[:, np.newaxis] * positions[:, :, 2]).sum(axis=0) / masses.sum(axis=0)
    elif central_body_view and central_body_index is not None:
        shift_x = positions[central_body_index, :, 0]
        shift_y = positions[central_body_index, :, 1]
        shift_z = positions[central_body_index, :, 2]
    else:
        shift_x = np.zeros(num_steps)
        shift_y = np.zeros(num_steps)
        shift_z = np.zeros(num_steps)

    shifted_positions = np.array([
        [positions[obj, :, 0] - shift_x, positions[obj, :, 1] - shift_y, positions[obj, :, 2] - shift_z]
        for obj in range(num_objects)
    ])

    # Calculate energy histories
    pot_energy_history = NewtonHamiltonian.HistoryOfTotalPotentialEnergy(masses, positions=np.transpose(positions, axes=(1, 0, 2)))
    kin_energy_history: np.ndarray = np.transpose(NewtonHamiltonian.HistoryOfKineticEnergies(masses, momenta=np.transpose(momenta, axes=(1, 0, 2))), axes=(1, 0))
    tot_energy_history = (6.67430e-20 * pot_energy_history) + kin_energy_history.sum(axis=0)

    # Set up figure and axes for animation
    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.4, hspace=0.3)
    ax_motion = fig.add_subplot(grid[:, 0], projection='3d')
    ax_energy = fig.add_subplot(grid[0, 1])
    ax_kin = fig.add_subplot(grid[1, 1])

    # Set up appearance of motion animation
    ax_motion.set_xlim(xlim)
    ax_motion.set_ylim(ylim)
    ax_motion.set_zlim(zlim)
    ax_motion.set_title("Animation of the system")
    ax_motion.set_xlabel("x")
    ax_motion.set_ylabel("y")
    ax_motion.set_zlabel("z")

    # Plot optional planes if specified
    if "solar_ecliptic" in show:
        xx, yy = np.meshgrid(np.linspace(*xlim, 10), np.linspace(*ylim, 10))
        zz = yy / 2
        ax_motion.plot_surface(xx, yy, zz, alpha=0.1, color='magenta')
    if "galactic_ecliptic" in show:
        xx, yy = np.meshgrid(np.linspace(*xlim, 10), np.linspace(*ylim, 10))
        zz = np.zeros_like(xx)
        ax_motion.plot_surface(xx, yy, zz, alpha=0.1, color='cyan')

    # Set up kinetic energy plot
    ax_kin.set_title("Kinetic energies in the system")
    ax_kin.set_xlabel("t")
    ax_kin.set_ylabel("T")
    ax_kin.set_xlim(0, dt * len(positions[0]))
    ax_kin.set_ylim(kin_energy_history.min() / 1.1, kin_energy_history.max() * 1.1)

    # Set up total energy plot
    ax_energy.set_title("Total energy in the system")
    ax_energy.set_xlabel("t")
    ax_energy.set_ylabel("V")
    ax_energy.set_xlim(0, dt * len(positions[0]))
    ax_energy.set_ylim(tot_energy_history.min() * 1.1, tot_energy_history.max() / 1.1)

    # Initialize plots for animation
    bodies = [ax_motion.plot([], [], [], 'o', label=names[i], lw=2)[0] for i in range(num_objects)]
    pot_energy = ax_energy.plot([], [], '-', label="Total V ")[0]
    kin_energies = [ax_kin.plot([], [], '-', label=names[i], lw=2)[0] for i in range(num_objects)]
    t_space = np.linspace(0, dt * len(positions[0]), len(positions[0]))
    ax_motion.legend(loc="upper right")
    ax_kin.legend(loc="upper right")

    # Initialize motion lines for each object if "motion_lines" is in show
    if "motion_lines" in show:
        motion_lines = [ax_motion.plot([], [], [], '-', alpha=0.3)[0] for _ in range(num_objects)]

    # Initialize animation function
    def init():
        for body, kin in zip(bodies, kin_energies):
            body.set_data([], [])
            body.set_3d_properties([])
            kin.set_data([], [])
        pot_energy.set_data([], [])
        if "motion_lines" in show:
            for line in motion_lines:
                line.set_data([], [])
                line.set_3d_properties([])
        return [*bodies, *kin_energies, pot_energy, *(motion_lines if "motion_lines" in show else [])]

    # Updase function for each frame
    def update(frame):
        for i, (body, kin) in enumerate(zip(bodies, kin_energies)):
            x_data = shifted_positions[i, 0, :frame]
            y_data = shifted_positions[i, 1, :frame]
            z_data = shifted_positions[i, 2, :frame]
            if x_data.shape == y_data.shape == z_data.shape:
                body.set_data(x_data[-1:], y_data[-1:])
                body.set_3d_properties(z_data[-1:])
            kin.set_data(t_space[:frame + 1], kin_energy_history[i, :frame + 1])
            if "motion_lines" in show:
                motion_lines[i].set_data(x_data[-motion_line_length:], y_data[-motion_line_length:])
                motion_lines[i].set_3d_properties(z_data[-motion_line_length:])
                motion_lines[i].set_alpha(0.3)
        pot_energy.set_data(t_space[:frame + 1], tot_energy_history[:frame + 1])
        return [*bodies, *kin_energies, pot_energy, *(motion_lines if "motion_lines" in show else [])]

    ani = FuncAnimation(fig, update, frames=num_steps, init_func=init, interval=interval)
    plt.legend()
    plt.show()


def animate_Newton_3D(positions, masses=None, central_body_index=None,
                    mass_centre_view=False, central_body_view=False,
                    xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), zlim=(-1.2, 1.2), interval=5,
                    names=None, show=[], motion_line_length=20, output_filename=None, ani_title = "Animation",**kwargs):
    """
    Runs animated version of the given positions history array in 3D. Introduces energies as separate graphs.

    Args:
        positions (arr): Chronologically ordered positions of the bodies.
        masses (arr, optional): Masses of the bodies. Defaults to None.
        central_body_index (int, optional): If central view is on, this represents the central body. Defaults to None.
        mass_centre_view (bool, optional): Centers the view on the mass center. Defaults to False.
        central_body_view (bool, optional): Centers the view on the given body. Defaults to False.
        xlim (tuple, optional): Range in x to be plotted. Defaults to (-1.2, 1.2).
        ylim (tuple, optional): Range in y to be plotted. Defaults to (-1.2, 1.2).
        zlim (tuple, optional): Range in z to be plotted. Defaults to (-1.2, 1.2).
        dt (int, optional): Time step for the animation. Defaults to 1.
        interval (int, optional): How many milliseconds per frame. Defaults to 50.
        names (list, optional): Names of the bodies. Defaults to None.
        show (list, optional): Features to be visualized in the animation. Options include "solar_ecliptic", "galactic_ecliptic", and "motion_lines". Defaults to [].
        motion_line_length (int, optional): Number of steps for which motion lines (trajectories) are visible. Defaults to 20.

    Returns:
        None
    """

    # Determine shifts for centering if needed
    num_objects, num_steps, _ = positions.shape
    if names is None:
        names = [f"Object {i}" for i in range(len(positions))]
    if mass_centre_view and masses is not None:
        shift_x = (masses[:, np.newaxis] * positions[:, :, 0]).sum(axis=0) / masses.sum(axis=0)
        shift_y = (masses[:, np.newaxis] * positions[:, :, 1]).sum(axis=0) / masses.sum(axis=0)
        shift_z = (masses[:, np.newaxis] * positions[:, :, 2]).sum(axis=0) / masses.sum(axis=0)
    elif central_body_view and central_body_index is not None:
        shift_x = positions[central_body_index, :, 0]
        shift_y = positions[central_body_index, :, 1]
        shift_z = positions[central_body_index, :, 2]
    else:
        shift_x = np.zeros(num_steps)
        shift_y = np.zeros(num_steps)
        shift_z = np.zeros(num_steps)

    shifted_positions = np.array([
        [positions[obj, :, 0] - shift_x, positions[obj, :, 1] - shift_y, positions[obj, :, 2] - shift_z]
        for obj in range(num_objects)
    ])

  
    # Set up figure and axes for animation
    fig = plt.figure(figsize=(14, 8))
    ax_motion = fig.add_subplot(projection='3d')
  
    # Set up appearance of motion animation
    ax_motion.set_xlim(xlim)
    ax_motion.set_ylim(ylim)
    ax_motion.set_zlim(zlim)
    ax_motion.set_title(ani_title)
    ax_motion.set_xlabel("x")
    ax_motion.set_ylabel("y")
    ax_motion.set_zlabel("z")

    # Plot optional planes if specified
    if "solar_ecliptic" in show:
        xx, yy = np.meshgrid(np.linspace(*xlim, 10), np.linspace(*ylim, 10))
        zz = yy / 2
        ax_motion.plot_surface(xx, yy, zz, alpha=0.1, color='magenta')
    if "galactic_ecliptic" in show:
        xx, yy = np.meshgrid(np.linspace(*xlim, 10), np.linspace(*ylim, 10))
        zz = np.zeros_like(xx)
        ax_motion.plot_surface(xx, yy, zz, alpha=0.1, color='cyan')

    # Initialize plots for animation
    bodies = [ax_motion.plot([], [], [], 'o', label=names[i], lw=2)[0] for i in range(num_objects)]
    ax_motion.legend(loc="upper right")

    # Initialize motion lines for each object if "motion_lines" is in show
    if "motion_lines" in show:
        motion_lines = [ax_motion.plot([], [], [], '-', alpha=0.3)[0] for _ in range(num_objects)]

    # Initialize animation function
    def init():
        for body in bodies:
            body.set_data([], [])
            body.set_3d_properties([])
        if "motion_lines" in show:
            for line in motion_lines:
                line.set_data([], [])
                line.set_3d_properties([])
        return [*bodies, *(motion_lines if "motion_lines" in show else [])]

    # Update function for each frame
    def update(frame):
        for i, body in enumerate(bodies):
            x_data = shifted_positions[i, 0, :frame]
            y_data = shifted_positions[i, 1, :frame]
            z_data = shifted_positions[i, 2, :frame]
            if x_data.shape == y_data.shape == z_data.shape:
                body.set_data(x_data[-1:], y_data[-1:])
                body.set_3d_properties(z_data[-1:])
            if "motion_lines" in show:
                motion_lines[i].set_data(x_data[-motion_line_length:], y_data[-motion_line_length:])
                motion_lines[i].set_3d_properties(z_data[-motion_line_length:])
                motion_lines[i].set_alpha(0.3)
        
        # Rotate the view
        ax_motion.view_init(elev=10., azim=frame * 0.8)
        return [*bodies, *(motion_lines if "motion_lines" in show else [])]

    ani = FuncAnimation(fig, update, frames=num_steps, init_func=init, interval=interval)
    # Save animation as video if output filename is specified
    if output_filename is not None:
        ani.save(output_filename, writer=PillowWriter(fps=1000/interval))

    plt.legend()
    plt.show()

