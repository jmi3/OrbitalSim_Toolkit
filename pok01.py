import numpy as np
import pyfftw
from pyfftw.interfaces.scipy_fft import fftn, ifftn, fftfreq
pyfftw.interfaces.cache.enable()  # Speeds up repeated FFT planning

class PoissonSolver:
    def __init__(self, G=1.0, workers=4):
        """
        Initialize the Poisson solver.

        Parameters:
        - G: gravitational constant (default: 1.0)
        - threads: number of threads to use in FFTs
        """
        self.G = G
        self.workers = workers

    def solve(self, rho, dx=1.0):
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


if __name__ == "__main__":
    # ----------------------- demo ------------------------------------------------
    L   = 1.0                    # physical box size
    N   = 256                    # cells per dimension
    dx  = L / N                  # physical cell width
    mid = N // 2                 # index of z = L/2 slice

    # density field: unit mass in one cell (or use 1/dx**3 for total mass = 1)
    rho = np.zeros((N, N, N))
    rho[mid, mid, mid] = 1.0
    rho[mid//2, mid//2, mid] = 10.0
    
    rho -= rho.mean()            # remove uniform background

    phi = PoissonSolver(workers=8).solve(rho, dx)

    # ----------------------- plot in physical units ------------------------------
    x_phys = np.linspace(0, L, N, endpoint=False) + 0.5*dx  # cell centres
    extent = [x_phys.min(), x_phys.max(), x_phys.min(), x_phys.max()]
    
    from matplotlib import pyplot as plt
    plt.figure(figsize=(5, 4))
    im = plt.imshow(phi[:, :, mid],
                    extent=extent,
                    origin='lower',
                    aspect='equal')
    plt.xlabel('x (physical units)')
    plt.ylabel('y (physical units)')
    plt.title('Mid-plane potential slice (z = 0.5)')
    plt.colorbar(im, label='Potential ϕ')
    plt.tight_layout()
    plt.show()