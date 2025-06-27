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
        return np.apply_along_axis(np.sum, axis=1, arr=(momenta ** 2))/(2 * masses)

    @classmethod
    def KineticEnergy(cls, masses: np.ndarray, momenta: np.ndarray) -> float:
        return cls.KineticEnergies(masses,momenta).sum()

    @classmethod
    def TotalPotentialEnergy(cls, masses: np.ndarray, positions: np.ndarray) -> float:
        raise NotImplementedError()
    
    @classmethod
    def dHdp(cls, masses: np.ndarray, momenta: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    @classmethod
    def _nablaPhi(cls, phi: np.ndarray, dx: float) -> np.ndarray:
        """
        Compute the gradient of the potential field phi.
        Parameters:
        - phi: ndarray representing the potential field
        - dx: grid spacing (assumed equal in all dimensions)
        Returns:
        - force: ndarray representing the gradient of phi
        """
        ndim = phi.ndim
        force = np.empty((ndim,) + phi.shape, dtype=phi.dtype)
        for axis in range(ndim):
            forward = np.roll(phi, -1, axis=axis)
            backward = np.roll(phi, 1, axis=axis)
            force[axis] = -(forward - backward) / (2 * dx)
        return force
    
    @classmethod
    def GenerateDensity(cls, masses: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """
        Convert masses and positions to a density field.
        
        Parameters:
        - masses: ndarray of masses
        - positions: ndarray of positions
        
        Returns:
        - rho: ndarray representing the density field
        """
        # Create a grid for the density field
        rho = np.zeros(cls.mesh_size)
        
        # Populate the density field based on positions and masses
        for mass, pos in zip(masses, positions):
            idx = tuple((pos / cls.dx).astype(int) % np.array(cls.mesh_size))
            rho[idx] += mass
        
        return rho
    
    @classmethod
    def dHdq(cls, masses: np.ndarray, positions: np.ndarray) -> np.ndarray:
        solver = PeriodicPoissonSolver(G=1.0, workers=4)
        rho = cls.GenerateDensity(masses, positions)
        # Solve the Poisson equation to get the potential
        phi = solver.solve(rho, dx=cls.dx)
        nabla_phi = np.gradient(phi, cls.dx, axis=(0, 1, 2))
        # Interpolate the gradient to the positions
        dHdq = np.zeros_like(positions)
        for i, pos in enumerate(positions):
            idx = tuple((pos / cls.dx).astype(int) % np.array(cls.mesh_size))
            dHdq[i] = np.array([nabla_phi[axis][idx] for axis in range(3)])
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