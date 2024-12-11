import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from hamilton import KeplerHamiltonian as H
from rkmethods import RKp
from plotting_helpers import animate_with_energy_Kepler,animate_multiple_with_energy_Kepler, plot_rkp_solutions



class dHdQP:
    def __init__(self, H: H):
        self.H = H

    def __call__(self, t, QP):
        [Q, P] = np.split(QP,2)
        # dq/dt = dH/dp , dp/dt = -dH/dq
        return np.append(self.H.dHdp(P),-self.H.dHdq(Q),axis=0)

def CompareSolutionErrors(ICs, t0, tmax, Ns, Ps, **kwags):
    
     P_errors = []

     for P in Ps:
          errors = []
          
          RK_solver = RKp(order=P) 
          for N in Ns:
               
               dt = 2 * np.pi / N
               
               RK_solver.Initialize(y0=ICs, dydt=dHdQP(H))
               RK_solver.Integrate(t0=t0, tmax=tmax, h=dt)
               
               QP_history = RK_solver.GetHistory()
               npQP_history = np.array(QP_history)
               positions, momenta = np.split(npQP_history.transpose(1, 0, 2), 2)

               error = np.linalg.norm(positions[0,0] - positions[0,-1], ord=2)
               # Replace zeros or negatives with computer epsilon
               errors.append(max(error, 1e-15))  

          P_errors.append(errors)

     # Plot the results
     fig = plt.figure(figsize=(6, 6))
     DTs = 2*np.pi/np.array(Ns)
     for errors, P in zip(P_errors, Ps):
          plt.plot(DTs, errors, label=f'Order {P}')

     plt.yscale('log')
     plt.xscale('log')
     plt.xlabel('Time step')
     plt.ylabel('Error (L2 norm)')
     plt.legend()
     plt.grid(True, which="both", linestyle="--", linewidth=0.5)
     plt.title('Error vs. Time step')
     plt.show()

def IntegrateAndAnimate(ICs, t0, tmax, N, P ):


     RK4 = RKp(order=P)

     # integrate 
     RK4.Initialize(y0=ICs, dydt=dHdQP(H))
     RK4.Integrate(t0=t0, tmax=tmax, h=2*np.pi/N)
     QP_history = RK4.GetHistory()
     npQP_history = np.array(QP_history)
     positions, momenta = np.split(npQP_history.transpose(1,0,2),2)



     animate_with_energy_Kepler(
          positions=positions[:,::50],
          momenta=momenta[:,::50],
          dt=2*np.pi/N,
          central_body_index=0,
          mass_centre_view=True,  # or False
          central_body_view=False,  # or True
          xlim=(-2, 2),
          ylim=(-2, 2),
          interval=5
     )

def IntegrateAndAnimateMultiplePs(ICs, t0, tmax, N, Ps, sizer = 1.1 ):
     positions_dict = {}
     momenta_dict = {}

     for P in Ps:
          solver = RKp(order=P)

          # integrate 
          solver.Initialize(y0=ICs, dydt=dHdQP(H))
          solver.Integrate(t0=t0, tmax=tmax, h=2*np.pi/N)
          QP_history = solver.GetHistory()
          npQP_history = np.array(QP_history)
          positions_dict[f"RK{P}"], momenta_dict[f"RK{P}"] = np.split(npQP_history.transpose(1,0,2)[:,::10],2)



     animate_multiple_with_energy_Kepler(
          positions_dict=positions_dict,
          momenta_dict=momenta_dict,
          dt=2*np.pi/N,
          xlim=(-2, 2),
          ylim=(-2, 2),
          interval=5,
          sizer=sizer
     )


def CompareSolutions(ICs, t0, tmax, N, Ps, gridshape = None, sizeOfFig= 5):
     
     plot_rkp_solutions(
          rkp_solvers=[RKp(order = i) for i in Ps],
          initial_conditions= ICs,
          dydt= dHdQP(H),
          t0=t0,
          h=2*np.pi/N,
          tmax=tmax,
          xlim=(-2,2),
          ylim=(-2,2),
          gridshape = gridshape,
          sizeOfFig = sizeOfFig
     )



# Setup initial conditions
e = 0.9
ICs = np.array([
     np.array([1-e,0]),
     np.array([0,np.sqrt((1+e)/(1-e))])
     ])

# Setup integration parameters
t0 = 0
years = 1
tmax = 2 * np.pi * years

# Numbers of steps we wish to examine
Ns = [100 * 2**i for i in range(9)]

# Orders we want to test
Ps = [1,2,3,4,5,"5GPT",6,"6GPT"]

# Perform desired action
# IntegrateAndAnimateMultiplePs(ICs=ICs, t0=t0, tmax=tmax, N=1000, Ps=[4,5,"5GPT",6,"6GPT"], sizer=1.00001)
# CompareSolutions(ICs=ICs, t0=t0, tmax=tmax, N=Ns[1], Ps=Ps, gridshape=(2,2), sizeOfFig=4.5)
CompareSolutionErrors(ICs=ICs, t0=t0, tmax=tmax, Ns=Ns, Ps=Ps, gridshape=(2,2), sizeOfFig=4.5)




