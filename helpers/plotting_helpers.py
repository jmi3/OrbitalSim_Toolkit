import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.animation import FuncAnimation
from hamilton import NewtonHamiltonian, KeplerHamiltonian
from rkmethods import RKp

def plot_rkp_solutions(rkp_solvers, initial_conditions, dydt, t0, tmax, h, 
                       masses=None, central_body_index=None, mass_centre_view=False, central_body_view=False,
                       xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), names = None, gridshape=None, sizeOfFig = 5):
    """
    Plots RKp solutions for all given rkp_solvers. 
    Returns void.
    Shows graphs containing evolution of the same system using all the rkp_solvers

    Args:
        rkp_solvers (List of RKp): Takes in array of RKps
        initial_conditions (ndarray): array of shape (2k,2) for k bodies in 2 dimensions
        dydt (function): fucntion from RK
        t0 (float): starting time 
        tmax (float): final time
        h (float): timestep
        masses (ndarray, optional): array of shape (k,) with masses. Defaults to None.
        central_body_index (int, optional): If central view is on, this represents central body. Defaults to None.
        mass_centre_view (bool, optional): Centers the view on mass center. Defaults to False.
        central_body_view (bool, optional): Centers the view on given body. Defaults to False.
        xlim (tuple, optional): Range in x to be plotted. Defaults to (-1.2, 1.2).
        ylim (tuple, optional): Range in y to be plotted. Defaults to (-1.2, 1.2).
    """
    if gridshape is None:
        gridshape= (1, len(rkp_solvers))
    # Prepare the plot size and layout
    fig, axes = plt.subplots(gridshape[0], gridshape[1], figsize=(sizeOfFig*gridshape[1], sizeOfFig*gridshape[0]))
    axes = np.array(axes).flatten()
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
        elif central_body_view and central_body_index is not None:
            shift_x = positions[central_body_index, :, 0]
            shift_y = positions[central_body_index, :, 1]
        else:
            shift_x = np.zeros_like(positions[0, :, 0])
            shift_y = np.zeros_like(positions[0, :, 1])
        
        if names is None:
            names = [f"Object {i}" for i in range(len(positions))]
    
        # Plot positions
        for i in range(len(positions)):
            pos = positions[i]
            x_axis = pos[:, 0] - shift_x
            y_axis = pos[:, 1] - shift_y
            ax.plot(x_axis, y_axis, label=names[i])
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"RK{solver.order} solver")
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show(block=True)




def animate_with_energy_Newton(positions, momenta, masses=None, central_body_index=None, 
                        mass_centre_view=False, central_body_view=False,
                        xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), dt=1, interval=5, 
                        names=None):

    """
    Runs animated version of the given positions history array. Introduces energies as separate graphs.

    Args:
        positions (arr): Chronologically ordered positions of the bodies 
        masses (arr, optional): Masses. Defaults to None.
        central_body_index (int, optional): If central view is on, this represents central body. Defaults to None.
        mass_centre_view (bool, optional):  Centers the view on mass center. Defaults to False.
        central_body_view (bool, optional): Centers the view on given body. Defaults to False.
        xlim (tuple, optional):  Range in x to be plotted. Defaults to (-1.2, 1.2).
        ylim (tuple, optional):  Range in y to be plotted. Defaults to (-1.2, 1.2).
        interval (int, optional): How many ms/frame. Defaults to 50.

    Returns:
        _type_: _description_
    """
    num_objects, num_steps, _ = positions.shape
    if names is None:
            names = [f"Object {i}" for i in range(len(positions))]
    
    # Determine shifts for centering
    if mass_centre_view and masses is not None:
        shift_x = (masses[:, np.newaxis] * positions[:, :, 0]).sum(axis=0) / masses.sum(axis=0)
        shift_y = (masses[:, np.newaxis] * positions[:, :, 1]).sum(axis=0) / masses.sum(axis=0)
    elif central_body_view and central_body_index is not None:
        shift_x = positions[central_body_index, :, 0]
        shift_y = positions[central_body_index, :, 1]
    else:
        shift_x = np.zeros(num_steps)
        shift_y = np.zeros(num_steps)

    shifted_positions = np.array([
        [positions[obj, :, 0] - shift_x, positions[obj, :, 1] - shift_y]
        for obj in range(num_objects)
    ])
    pot_energy_history = NewtonHamiltonian.HistoryOfTotalPotentialEnergy(masses, positions=np.transpose(positions,axes=(1,0,2)))
    kin_energy_history:np.ndarray = np.transpose(NewtonHamiltonian.HistoryOfKineticEnergies(masses, momenta=np.transpose(momenta,axes=(1,0,2))),axes=(1,0))

    tot_energy_history = (6.67430e-20 * pot_energy_history) + kin_energy_history.sum(axis=0)
    # Set up the figure and axis
    fig = plt.figure(figsize=(14, 8))
    grid =fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.4, hspace=0.3)

    # Create the simulation subplot (big plot on the left)
    ax_motion = fig.add_subplot(grid[:, 0])  
    ax_energy = fig.add_subplot(grid[0, 1])
    ax_kin = fig.add_subplot(grid[1, 1])
    
    # Setup looks of motion animation
    ax_motion.set_xlim(xlim)
    ax_motion.set_ylim(ylim)
    ax_motion.set_aspect("equal")
    ax_motion.set_title("Animation of the system")
    ax_motion.set_xlabel("x")
    ax_motion.set_ylabel("y")

    # Setup looks of motion animation
    ax_kin.set_title("Kinetic energies in the system")
    ax_kin.set_xlabel("t")
    ax_kin.set_ylabel("T")
    ax_kin.set_xlim(0,dt*len(positions[0]))
    ax_kin.set_ylim(kin_energy_history.min()/1.1,kin_energy_history.max()*1.1)


    # Setup looks of motion animation
    ax_energy.set_title("Total energy in the system")
    ax_energy.set_xlabel("t")
    ax_energy.set_ylabel("V")
    ax_energy.set_xlim(0,dt*len(positions[0]))
    ax_energy.set_ylim(tot_energy_history.min()*1.1,tot_energy_history.max()/1.1)
    
    

    # Initialize plots for each object
    bodies = [ax_motion.plot([], [], 'o', label=names[i], lw=2)[0] for i in range(num_objects)]
    pot_energy = ax_energy.plot([], [], '-', label="Total V ")[0]
    kin_energies = [ax_kin.plot([], [], '-', label=names[i], lw=2)[0] for i in range(num_objects)]
    t_space = np.linspace(0,dt*len(positions[0]),len(positions[0]))
    # Add legends
    ax_motion.legend(loc="upper right")  # Legend for simulation
    ax_kin.legend(loc="upper right")  # Legend for kinetic energy

    def init():
        for body, kin in zip(bodies, kin_energies):
            body.set_data([], [])
            kin.set_data([], [])
        pot_energy.set_data([], [])
        return [*bodies, *kin_energies, pot_energy]

    def update(frame):
        for i, (body,  kin) in enumerate(zip(bodies, kin_energies)):
            x_data = shifted_positions[i, 0, :frame]
            y_data = shifted_positions[i, 1, :frame]
             # Update position of the body
            if x_data.shape == y_data.shape:  # Ensure shapes match
                body.set_data(x_data[-1:], y_data[-1:])  # Current position
            else:
                print(f"Shape mismatch: x_data={x_data.shape}, y_data={y_data.shape}")
            kin.set_data(t_space[:frame+1],kin_energy_history[i,:frame+1])
        pot_energy.set_data(t_space[:frame+1],tot_energy_history[:frame+1])
        return [*bodies, *kin_energies, pot_energy]

    ani = FuncAnimation(fig, update, frames=num_steps, init_func=init, interval=interval)
    plt.legend()
    plt.show()





def animate_rkp_motion(positions, masses=None, central_body_index=None, 
                        mass_centre_view=False, central_body_view=False,
                        xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), interval=50):
    """
    Runs animated version of the given positions history array

    Args:
        positions (arr): Chronologically ordered positions of the bodies 
        masses (arr, optional): Masses. Defaults to None.
        central_body_index (int, optional): If central view is on, this represents central body. Defaults to None.
        mass_centre_view (bool, optional):  Centers the view on mass center. Defaults to False.
        central_body_view (bool, optional): Centers the view on given body. Defaults to False.
        xlim (tuple, optional):  Range in x to be plotted. Defaults to (-1.2, 1.2).
        ylim (tuple, optional):  Range in y to be plotted. Defaults to (-1.2, 1.2).
        interval (int, optional): How many ms/frame. Defaults to 50.

    Returns:
        _type_: _description_
    """
    num_objects, num_steps, _ = positions.shape

    # Determine shifts for centering
    if mass_centre_view and masses is not None:
        shift_x = (masses[:, np.newaxis] * positions[:, :, 0]).sum(axis=0) / masses.sum(axis=0)
        shift_y = (masses[:, np.newaxis] * positions[:, :, 1]).sum(axis=0) / masses.sum(axis=0)
    elif central_body_view and central_body_index is not None:
        shift_x = positions[central_body_index, :, 0]
        shift_y = positions[central_body_index, :, 1]
    else:
        shift_x = np.zeros(num_steps)
        shift_y = np.zeros(num_steps)

    shifted_positions = np.array([
        [positions[obj, :, 0] - shift_x, positions[obj, :, 1] - shift_y]
        for obj in range(num_objects)
    ])

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title("Animation of the system")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)

    # Initialize plots for each object
    bodies = [ax.plot([], [], 'o', label=f"Object {i}")[0] for i in range(num_objects)]
    trails = [ax.plot([], [], '-', alpha=0.5)[0] for _ in range(num_objects)]

    def init():
        for body, trail in zip(bodies, trails):
            body.set_data([], [])
            trail.set_data([], [])
        return bodies + trails

    def update(frame):
        for i, (body, trail) in enumerate(zip(bodies, trails)):
            x_data = shifted_positions[i, 0, :frame]
            y_data = shifted_positions[i, 1, :frame]
            body.set_data(x_data[-1:], y_data[-1:])  # Update current position
            trail.set_data(x_data, y_data)  # Update trail
        return bodies + trails

    ani = FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True, interval=interval)
    plt.legend()
    plt.show()




def animate_with_energy_Kepler(positions, momenta, masses=None, central_body_index=None, 
                        mass_centre_view=False, central_body_view=False,
                        xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), dt=1, interval=5, 
                        names=None, linelength=100):

    """
    Runs animated version of the given positions history array

    Args:
        positions (arr): Chronologically ordered positions of the bodies 
        masses (arr, optional): Masses. Defaults to None.
        central_body_index (int, optional): If central view is on, this represents central body. Defaults to None.
        mass_centre_view (bool, optional):  Centers the view on mass center. Defaults to False.
        central_body_view (bool, optional): Centers the view on given body. Defaults to False.
        xlim (tuple, optional):  Range in x to be plotted. Defaults to (-1.2, 1.2).
        ylim (tuple, optional):  Range in y to be plotted. Defaults to (-1.2, 1.2).
        interval (int, optional): How many ms/frame. Defaults to 50.

    Returns:
        _type_: _description_
    """
    num_objects, num_steps, _ = positions.shape
    if names is None:
            names = [f"Object {i}" for i in range(len(positions))]
    
    # Determine shifts for centering
    if mass_centre_view and masses is not None:
        shift_x = (masses[:, np.newaxis] * positions[:, :, 0]).sum(axis=0) / masses.sum(axis=0)
        shift_y = (masses[:, np.newaxis] * positions[:, :, 1]).sum(axis=0) / masses.sum(axis=0)
    elif central_body_view and central_body_index is not None:
        shift_x = positions[central_body_index, :, 0]
        shift_y = positions[central_body_index, :, 1]
    else:
        shift_x = np.zeros(num_steps)
        shift_y = np.zeros(num_steps)

    shifted_positions = np.array([
        [positions[obj, :, 0] - shift_x, positions[obj, :, 1] - shift_y]
        for obj in range(num_objects)
    ])
    energy_history = KeplerHamiltonian.HistoryOfValues(
        momenta = np.transpose(momenta,axes=(1,0,2)),
        positions = np.transpose(positions,axes=(1,0,2)))
     
    lp_history = KeplerHamiltonian.HistoryOfLp(momenta = np.transpose(momenta,axes=(1,0,2)), positions = np.transpose(positions,axes=(1,0,2)))

    # Set up the figure and axis
    fig = plt.figure(figsize=(14, 8))
    grid =fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.4, hspace=0.3)

    # Create the simulation subplot (big plot on the left)
    ax_motion = fig.add_subplot(grid[:, 0])  
    ax_energy = fig.add_subplot(grid[0, 1])
    ax_Lp = fig.add_subplot(grid[1, 1])
    
    # Setup looks of motion animation
    ax_motion.set_xlim(xlim)
    ax_motion.set_ylim(ylim)
    ax_motion.set_aspect("equal")
    ax_motion.set_title("Animation of the system")
    ax_motion.set_xlabel("x")
    ax_motion.set_ylabel("y")

    # Setup looks of motion animation
    ax_Lp.set_title("Angular momentum in the system")
    ax_Lp.set_xlabel("t")
    ax_Lp.set_ylabel("L")
    ax_Lp.set_xlim(0,dt*len(positions[0]))
    ax_Lp.set_ylim(lp_history.min()*1.1,lp_history.max()/1.1)


    # Setup looks of motion animation
    ax_energy.set_title("Total energy in the system")
    ax_energy.set_xlabel("t")
    ax_energy.set_ylabel("V")
    ax_energy.set_xlim(0,dt*len(positions[0]))
    ax_energy.set_ylim(energy_history.min()*1.1,energy_history.max()/1.1)
    
    

    # Initialize plots for each object
    bodies = [ax_motion.plot([], [], 'o', label=names[i], lw=2)[0] for i in range(num_objects)]
    trajectories = [ax_motion.plot([], [], '-', label=names[i], alpha=0.5, color=bodies[i].get_color(), lw=2)[0] for i in range(num_objects)]
    energy = ax_energy.plot([], [], '-', label="Total V ")[0]
    Lp = ax_Lp.plot([], [], '-', label="Lp", lw=2)[0]
    t_space = np.linspace(0,dt*len(positions[0]),len(positions[0]))
    # Add legends
    ax_motion.legend(loc="upper right")  # Legend for simulation
    ax_Lp.legend(loc="upper right")  # Legend for kinetic energy

    def init():
        for (body, traj) in zip(bodies,trajectories):
            body.set_data([], [])
            traj.set_data([], [])
        Lp.set_data([], [])
        energy.set_data([], [])
        return [*bodies, *trajectories, Lp, energy]

    def update(frame):
        for i, (body,traj) in enumerate(zip(bodies,trajectories)):
            x_data = shifted_positions[i, 0, :frame]
            y_data = shifted_positions[i, 1, :frame]
             # Update position of the body
            body.set_data(x_data[-1:], y_data[-1:])  # Current position
            traj.set_data(x_data[max(frame-linelength,0):frame], y_data[max(frame-linelength,0):frame])
        Lp.set_data(t_space[:frame+1],lp_history[:frame+1])
        energy.set_data(t_space[:frame+1],energy_history[:frame+1])
        return [*bodies,*trajectories, Lp, energy]

    ani = FuncAnimation(fig, update, frames=num_steps, init_func=init, interval=interval)
    plt.legend()
    plt.show()

def animate_multiple_with_energy_Kepler(positions_dict, momenta_dict, masses_dict=None, 
                        xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), dt=1, interval=5, 
                        names_dict=None, linelength=100, sizer=1.1):

    """
    Runs animated version of the given positions history array for multiple bodies.

    Args:
        positions_dict (dict): Dictionary of chronologically ordered positions of the bodies for each system.
        momenta_dict (dict): Dictionary of momenta history for each system.
        masses_dict (dict, optional): Dictionary of masses for each system. Defaults to None.
        xlim (tuple, optional): Range in x to be plotted. Defaults to (-1.2, 1.2).
        ylim (tuple, optional): Range in y to be plotted. Defaults to (-1.2, 1.2).
        interval (int, optional): How many ms/frame. Defaults to 50.
        names_dict (dict, optional): Dictionary of names for each body in each system. Defaults to None.
        linelength (int, optional): Length of trajectory line. Defaults to 100.

    Returns:
        None
    """
    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.4, hspace=0.3)

    ax_motion = fig.add_subplot(grid[:, 0])  
    ax_energy = fig.add_subplot(grid[0, 1])
    ax_Lp = fig.add_subplot(grid[1, 1])

    ax_motion.set_xlim(xlim)
    ax_motion.set_ylim(ylim)
    ax_motion.set_aspect("equal")
    ax_motion.set_title("Animation of the system")
    ax_motion.set_xlabel("x")
    ax_motion.set_ylabel("y")

    t_space = {}
    bodies_dict = {}
    trajectories_dict = {}
    energy_lines = {}
    Lp_lines = {}
    energy_histories = {}
    lp_histories = {}

    for system_name, positions in positions_dict.items():
        momenta = momenta_dict[system_name]
        masses = masses_dict[system_name] if masses_dict and system_name in masses_dict else None
        names = names_dict.get(system_name, None) if names_dict else None

        num_objects, num_steps, _ = positions.shape
        if names is None:
            names = [f"{system_name} - Object {i}" for i in range(len(positions))]

        energy_history = KeplerHamiltonian.HistoryOfValues(
            momenta=np.transpose(momenta, axes=(1, 0, 2)),
            positions=np.transpose(positions, axes=(1, 0, 2)))
        
        lp_history = KeplerHamiltonian.HistoryOfLp(momenta=np.transpose(momenta, axes=(1, 0, 2)), positions=np.transpose(positions, axes=(1, 0, 2)))

        energy_histories[system_name] = energy_history
        lp_histories[system_name] = lp_history

        bodies = [ax_motion.plot([], [], 'o', label=names[i], lw=2)[0] for i in range(num_objects)]
        trajectories = [ax_motion.plot([], [], '-', label=names[i], alpha=0.5, color=bodies[i].get_color(), lw=2)[0] for i in range(num_objects)]
        energy_line = ax_energy.plot([], [], '-', label=f"{system_name} - Total V")[0]
        Lp_line = ax_Lp.plot([], [], '-', label=f"{system_name} - Lp", lw=2)[0]
        t_space[system_name] = np.linspace(0, dt * len(positions[0]), len(positions[0]))

        bodies_dict[system_name] = bodies
        trajectories_dict[system_name] = trajectories
        energy_lines[system_name] = energy_line
        Lp_lines[system_name] = Lp_line

    # Setting xlim and ylim for ax_Lp and ax_energy based on the computed histories
    all_lp_values = np.concatenate([lp for lp in lp_histories.values()])
    all_energy_values = np.concatenate([energy for energy in energy_histories.values()])

    max_t = max([t.max() for t in t_space.values()])
    ax_Lp.set_xlim(0, max_t)
    ax_Lp.set_ylim(all_lp_values.min()*sizer, all_lp_values.max()/sizer)
    ax_energy.set_xlim(0, max_t)
    ax_energy.set_ylim(all_energy_values.min()*sizer, all_energy_values.max()/sizer)

    ax_Lp.set_title("Angular momentum in the system")
    ax_Lp.set_xlabel("t")
    ax_Lp.set_ylabel("L")
    
    ax_energy.set_title("Total energy in the system")
    ax_energy.set_xlabel("t")
    ax_energy.set_ylabel("V")

    ax_motion.legend(loc="upper right")
    ax_energy.legend(loc="upper right")
    ax_Lp.legend(loc="upper right")

    def init():
        artists = []
        for system_name in positions_dict.keys():
            artists += [*bodies_dict[system_name], *trajectories_dict[system_name], Lp_lines[system_name], energy_lines[system_name]]
        for line in artists:
            line.set_data([], [])
        return artists

    def update(frame):
        artists = []
        for system_name, positions in positions_dict.items():
            num_objects = len(positions)

            for i, (body, traj) in enumerate(zip(bodies_dict[system_name], trajectories_dict[system_name])):
                x_data = positions[i, :, 0][:frame]
                y_data = positions[i, :, 1][:frame]
                body.set_data(x_data[-1:], y_data[-1:])  # Current position
                traj.set_data(x_data[max(frame - linelength, 0):frame], y_data[max(frame - linelength, 0):frame])
                artists += [body, traj]
            Lp_lines[system_name].set_data(t_space[system_name][:frame + 1], lp_histories[system_name][:frame + 1])
            energy_lines[system_name].set_data(t_space[system_name][:frame + 1], energy_histories[system_name][:frame + 1])
            artists += [Lp_lines[system_name], energy_lines[system_name]]
        return artists

    ani = FuncAnimation(fig, update, frames=num_steps, init_func=init, interval=interval)
    plt.legend()
    plt.show()
