import numpy as np

class SamplingGrid:
    def __init__(self, grid_shape, box_size):
        """
        Initialize the sampling grid.

        Parameters:
        - grid_shape: tuple of ints, shape of the grid (assumed cubic)
        - box_size: float, physical size of the domain (L)
        """
        self.grid_shape = grid_shape
        self.ndim = len(grid_shape)
        self.box_size = box_size
        self.dx = box_size / grid_shape[0]
        self.grid = np.zeros(grid_shape, dtype=np.float64)
        
    def _grid_coordinates(self, positions):
        """
        Convert physical positions to grid coordinates.

        Parameters:
        - positions: ndarray, shape (N, ndim), physical positions of particles

        Returns:
        - grid_coords: ndarray, shape (N, ndim), grid coordinates
        """
        return np.floor(positions / self.dx).astype(int) % self.grid_shape
    
    def distribute_quantity(self, positions, quantity, kernel="CIC"):
        """
        Distribute quantities (e.g., mass) to the grid based on particle positions.

        Parameters:
        - positions: ndarray, shape (N, ndim), physical positions of particles
        - quantities: ndarray, shape (N, qshape), quantities to distribute (e.g., masses)
        - kernel: str, interpolation method ('CIC' or 'NGP')

        Returns:
        - None, modifies self.grid in place
        """
        distribution = self._cic_kernel(positions) 
        
        
    def _cic_kernel(self, relative_position: np.ndarray) -> np.ndarray:
        """
        Distribute quantities using Cloud-In-Cell (CIC) method.

        Parameters:
        - relative_position: ndarray, shape (N, ndim), relative positions of particles in grid cell in grid units
        
        Returns:
        - weights: ndarray, shape (N, ndim), weights for each near grid point
        """
        return max(1 - np.absolute(relative_position),0)
        
        
        
        