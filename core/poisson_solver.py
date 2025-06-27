import numpy as np
import pyfftw
from pyfftw.interfaces.scipy_fft import fftn, ifftn, fftfreq
pyfftw.interfaces.cache.enable()  # Speeds up repeated FFT planning


class PeriodicPoissonSolver:
    def __init__(self, G=1.0, workers=4):
        """
        Initialize the Poisson solver.

        Parameters:
        - G: gravitational constant (default: 1.0)
        - threads: number of threads to use in FFTs
        """
        self.G = G
        self.workers = workers
    
    def solve(self, rho, dx=1.0) -> np.ndarray:
        """
        Solve Poisson equation ∇²Φ = 4πG ρ using pyFFTW (periodic BCs).

        Parameters:
        - rho: ndarray, the density field
        - dx: float, the grid spacing

        Returns:
        - phi: ndarray, the potential field
        """
        shape = rho.shape
        ndim = rho.ndim

        # FFT of the density
        rho_k = fftn(rho, workers=self.workers)

        # Build k-space grid (angular frequencies)
        k = [fftfreq(n, d=dx) * 2 * np.pi for n in shape]
        k_grids = np.meshgrid(*k, indexing='ij')

        # Compute k²
        k_squared = sum(ki**2 for ki in k_grids)
        k_squared[tuple([0]*ndim)] = np.inf  # avoid division by 0

        # Solve in Fourier space
        phi_k = -4 * np.pi * self.G * rho_k / k_squared
        phi_k[tuple([0]*ndim)] = 0.0  # zero out mean

        # Inverse FFT to get back to real space
        phi = ifftn(phi_k, workers=self.workers).real
        return phi
    