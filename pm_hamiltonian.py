from core.hamiltonian import Hamiltonian
from core.poisson_solver import PeriodicPoissonSolver

import numpy as np



class PMHamiltonian(Hamiltonian):
    """
        Hamiltonian with the Particle Mesh method implemented.
    """
    @classmethod
    def SetParameters(cls, G: float = 1.0, workers: int = 4, mesh_size: tuple = None, dx: float = None):
        """
        Set the parameters for the PMHamiltonian class.
        
        Parameters:
        - G: gravitational constant (default: 1.0)
        - workers: number of threads to use in FFTs (default: 4)
        - mesh_size: size of the mesh grid (default: (256, 256, 256))
        - dx: grid spacing (default: 1.0)
        """
        cls.G = G
        cls.workers = workers
        cls.solver = PeriodicPoissonSolver(G=G, workers=workers)
        cls.mesh_size = mesh_size if mesh_size is not None else (256, 256, 256)
        cls.dx = dx if dx is not None else 1.0
        
    @classmethod
    def HistoryOfKineticEnergies(cls, masses: np.ndarray, momenta: np.ndarray) -> np.ndarray:
        res = np.zeros(shape=momenta.shape[:len(momenta.shape)-1])
        for i in range(len(momenta)):
            res[i] = cls.KineticEnergies(masses=masses,momenta=momenta[i])    
        return res
    
    @classmethod
    def HistoryOfTotalPotentialEnergy(cls, masses: np.ndarray, positions: np.ndarray) -> np.ndarray:
        res = np.zeros(shape=len(positions))
        for i in range(len(positions)):
            res[i] = cls.TotalPotentialEnergy(masses=masses,positions=positions[i])    
        return res

    @classmethod
    def KineticEnergies(cls, masses: np.ndarray, momenta: np.ndarray) -> np.ndarray:
        return (momenta**2).sum(axis=1) / (2.0 * masses)

    @classmethod
    def KineticEnergy(cls, masses: np.ndarray, momenta: np.ndarray) -> float:
        return cls.KineticEnergies(masses,momenta).sum()

    @classmethod
    def TotalPotentialEnergy(cls, masses: np.ndarray, positions: np.ndarray) -> float:
        raise NotImplementedError()
    
    @classmethod
    def dHdp(cls, masses: np.ndarray, momenta: np.ndarray) -> np.ndarray:
        return momenta / np.transpose(np.array([masses]))

    
    @classmethod
    def GenerateDensity(cls, masses: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """
        Convert masses and positions to a density field (periodic NGP).
        """
        rho = np.zeros(cls.mesh_size, dtype=np.float64)
        cell_vol = cls.dx ** len(cls.mesh_size)   # volume per cell (dx^ndim)
        for mass, pos in zip(masses, positions):
            idx = tuple((pos / cls.dx).astype(int) % np.array(cls.mesh_size))
            rho[idx] += mass / cell_vol           # mass -> mass density
        return rho

    @classmethod
    def _nablaPhi(cls, phi: np.ndarray, dx: float) -> np.ndarray:
        """
        Periodic centered-difference gradient (returns +∇phi).
        """
        ndim = phi.ndim
        grad = np.empty((ndim,) + phi.shape, dtype=phi.dtype)
        for axis in range(ndim):
            forward  = np.roll(phi, -1, axis=axis)
            backward = np.roll(phi,  1, axis=axis)
            grad[axis] = (forward - backward) / (2.0 * dx)
        return grad  # +∇phi (no minus sign here)

    
    @classmethod
    def dHdq(cls, masses: np.ndarray, positions: np.ndarray) -> np.ndarray:
        rho = cls.GenerateDensity(masses, positions)
        phi = cls.solver.solve(rho, dx=cls.dx)
        grad_phi = cls._nablaPhi(phi, cls.dx)  # shape: (ndim, *grid)

        # NGP interpolation of ∇Phi at particle positions
        dHdq = np.zeros_like(positions, dtype=np.float64)
        for i, pos in enumerate(positions):
            idx = tuple((pos / cls.dx).astype(int) % np.array(cls.mesh_size))
            g = np.array([grad_phi[axis][idx] for axis in range(len(cls.mesh_size))])
            dHdq[i] = masses[i] * g  # dH/dq = m * ∇Φ
        return dHdq

        
        
        
if __name__ == "__main__":
    # Example usage
    PMHamiltonian.SetParameters(G=1.0, workers=4, mesh_size=(50, 50, 50), dx=1.0)
    
    # Define masses and positions
    masses = np.array([1.0, 2.0])
    positions = np.array([[10.0, 20.0, 0.0], [40.0, 50.0, 0.0]])
    
    # Generate density field
    density = PMHamiltonian.GenerateDensity(masses, positions)
    import matplotlib.pyplot as plt

    # Plot a slice of the density field (e.g., the middle z-plane)
    plt.imshow(density[:, :, 0], origin='lower', cmap='viridis')
    plt.colorbar(label='Density')   
    plt.title('Density Field (z-slice)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()