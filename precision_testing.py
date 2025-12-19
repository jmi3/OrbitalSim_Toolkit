import matplotlib.pyplot as plt
import numpy as np

from core.hamilton import KeplerHamiltonian
from core.rkmethods import RKp
from helpers.plotting_helpers import (
    animate_multiple_with_energy_kepler,
    animate_with_energy_kepler,
    plot_rkp_solutions,
)


class MotionEquations:
    def __init__(self, hamiltonian: KeplerHamiltonian):
        self.hamiltonian = hamiltonian

    def __call__(self, t, qp):
        [positions, momenta] = np.split(qp, 2)
        # dq/dt = dH/dp , dp/dt = -dH/dq
        return np.append(
            self.hamiltonian.momentum_derivative(momenta),
            -self.hamiltonian.position_derivative(positions),
            axis=0,
        )


def compare_solution_errors(
    initial_conditions, t0, tmax, steps, orders, output_filename=None, **kwags
):
    order_errors = []

    for order in orders:
        errors = []

        rk_solver = RKp(order=order)
        for num_steps in steps:
            dt = 2 * np.pi / num_steps

            rk_solver.initialize(
                y0=initial_conditions, dydt=MotionEquations(KeplerHamiltonian())
            )
            rk_solver.integrate(t0=t0, tmax=tmax, h=dt)

            qp_history = rk_solver.get_history()
            np_qp_history = np.array(qp_history)
            positions, _ = np.split(np_qp_history.transpose(1, 0, 2), 2)

            error = np.linalg.norm(positions[0, 0] - positions[0, -1], ord=2)
            # Replace zeros or negatives with computer epsilon
            errors.append(max(error, 1e-15))

        order_errors.append(errors)

    # Plot the results
    fig = plt.figure(figsize=(6, 6))
    time_steps = 2 * np.pi / np.array(steps)
    for errors, order in zip(order_errors, orders):
        plt.plot(time_steps, errors, label=f"Order {order}")

    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Time step")
    plt.ylabel("Error (L2 norm)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.title("Error vs. Time step")
    if output_filename is not None:
        fig.savefig(output_filename)

    plt.show()


def integrate_and_animate(
    initial_conditions, t0, tmax, steps, orders, output_filename=None
):
    rk4 = RKp(order=orders)

    # integrate
    rk4.initialize(y0=initial_conditions, dydt=MotionEquations(KeplerHamiltonian))
    rk4.integrate(t0=t0, tmax=tmax, h=2 * np.pi / steps)
    qp_history = rk4.get_history()
    np_qp_history = np.array(qp_history)
    positions, momenta = np.split(np_qp_history.transpose(1, 0, 2), 2)

    animate_with_energy_kepler(
        positions=positions[:, ::50],
        momenta=momenta[:, ::50],
        dt=2 * np.pi / steps,
        central_body_index=0,
        mass_centre_view=True,  # or False
        central_body_view=False,  # or True
        xlim=(-2, 2),
        ylim=(-2, 2),
        interval=5,
        output_filename=output_filename,
    )


def integrate_and_animate_multiple_orders(
    initial_conditions, t0, tmax, steps, orders, sizer=1.1, output_filename=None
):
    positions_dict = {}
    momenta_dict = {}

    for order in orders:
        solver = RKp(order=order)

        # integrate
        solver.initialize(
            y0=initial_conditions, dydt=MotionEquations(KeplerHamiltonian)
        )
        solver.integrate(t0=t0, tmax=tmax, h=2 * np.pi / steps)
        qp_history = solver.get_history()
        np_qp_history = np.array(qp_history)
        positions_dict[f"RK{order}"], momenta_dict[f"RK{order}"] = np.split(
            np_qp_history.transpose(1, 0, 2)[:, ::10], 2
        )

    animate_multiple_with_energy_kepler(
        positions_dict=positions_dict,
        momenta_dict=momenta_dict,
        dt=2 * np.pi / steps,
        xlim=(-2, 2),
        ylim=(-2, 2),
        interval=5,
        sizer=sizer,
        output_filename=output_filename,
    )


def compare_solutions(
    initial_conditions,
    t0,
    tmax,
    steps,
    orders,
    gridshape=None,
    size_of_fig=5,
    output_filename=None,
):
    plot_rkp_solutions(
        rkp_solvers=[RKp(order=i) for i in orders],
        initial_conditions=initial_conditions,
        dydt=MotionEquations(KeplerHamiltonian),
        t0=t0,
        h=2 * np.pi / steps,
        tmax=tmax,
        xlim=(-2, 2),
        ylim=(-2, 2),
        gridshape=gridshape,
        size_of_fig=size_of_fig,
        output_filename=output_filename,
    )


# Setup initial conditions
e = 0.9
initial_conditions = np.array(
    [np.array([1 - e, 0]), np.array([0, np.sqrt((1 + e) / (1 - e))])]
)

# Setup integration parameters
t0 = 0
years = 5
tmax = 2 * np.pi * years

# Numbers of steps we wish to examine
steps = [100 * 2**i for i in range(9)]

# Orders we want to test
orders = [1, 2, 3, 4, 5, 6]


integrate_and_animate_multiple_orders(
    initial_conditions=initial_conditions,
    t0=t0,
    tmax=tmax,
    steps=1000,
    orders=[4, 5, 6],
    sizer=1.0001,
    output_filename="precision_testing_output/integrate_and_animate_mutiple_methods.png",
)


compare_solutions(
    initial_conditions=initial_conditions,
    t0=t0,
    tmax=tmax,
    steps=2000,
    orders=orders,
    gridshape=(2, 4),
    size_of_fig=4.5,
    output_filename="precision_testing_output/compare_solutions.png",
)


compare_solution_errors(
    initial_conditions=initial_conditions,
    t0=t0,
    tmax=2 * np.pi,
    steps=steps,
    orders=orders,
    gridshape=(2, 4),
    sizeOfFig=4.5,
    output_filename="precision_testing_output/compare_solution_errors.png",
)
