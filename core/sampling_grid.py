import numpy as np

class SamplingGrid:
    def __init__(self, grid_shape, box_size):
        """
        Initialize the sampling grid (periodic domain).

        Parameters
        ----------
        grid_shape : tuple of int
            Shape of the grid, e.g. (Nz, Ny, Nx). Assumed cubic spacing.
        box_size : float
            Physical size of the domain (same for all axes).
        """
        self.grid_shape = tuple(int(n) for n in grid_shape)
        self.ndim = len(self.grid_shape)
        self.box_size = float(box_size)
        # NOTE: Keep the original cubic assumption (same dx in all axes)
        self.dx = self.box_size / self.grid_shape[0]
        # Scalar grid storage
        self.grid = np.zeros(self.grid_shape, dtype=np.float64)

    # ------------------------------------------------------------------
    # Helper methods (interface preserved)
    # ------------------------------------------------------------------
    def _grid_coordinates(self, positions):
        """
        Convert physical positions to integer grid coordinates (periodic).

        Parameters
        ----------
        positions : ndarray, shape (N, ndim)
            Physical positions of particles.

        Returns
        -------
        grid_coords : ndarray, shape (N, ndim)
            Integer indices of the containing cell for each particle.
        """
        positions = np.asarray(positions, dtype=np.float64)
        # Wrap into [0, box_size) and convert to grid units
        xg = np.mod(positions, self.box_size) / self.dx
        # Base-cell index (periodic)
        idx = np.floor(xg).astype(np.int64) % np.array(self.grid_shape, dtype=np.int64)
        return idx

    def distribute_quantity(self, positions, quantity, kernel="CIC"):
        """
        Distribute quantities (e.g., mass) to the grid based on particle positions.

        Parameters
        ----------
        positions : ndarray, shape (N, ndim)
            Physical positions of particles.
        quantity : ndarray, shape (N,)
            Scalar quantity per particle (e.g., mass).
        kernel : {"CIC", "NGP"}
            Interpolation method.

        Notes
        -----
        - Modifies self.grid in place.
        - Periodic boundary conditions.
        """
        positions = np.asarray(positions, dtype=np.float64)
        q = np.asarray(quantity, dtype=np.float64)

        if positions.ndim != 2 or positions.shape[1] != self.ndim:
            raise ValueError(f"positions must have shape (N, {self.ndim})")
        if q.shape != (positions.shape[0],):
            raise ValueError("quantity must have shape (N,)")

        if kernel.upper() == "NGP":
            self._scatter_ngp(positions, q)
        elif kernel.upper() == "CIC":
            self._scatter_cic(positions, q)
        else:
            raise ValueError("kernel must be 'CIC' or 'NGP'")

    def _cic_kernel(self, relative_position: np.ndarray) -> np.ndarray:
        """
        Cloud-In-Cell per-axis weights for fractional offsets.

        Parameters
        ----------
        relative_position : ndarray, shape (N, ndim)
            Fractional offsets f in the base cell in *grid units*, values in [0, 1).

        Returns
        -------
        weights : ndarray, shape (N, ndim, 2)
            For each particle and axis, weights to the two neighboring nodes:
            weights[..., 0] = 1 - f (left/lower), weights[..., 1] = f (right/upper).
        """
        f = np.asarray(relative_position, dtype=np.float64)
        # Ensure within [0,1) numerically
        f = f - np.floor(f)
        w0 = 1.0 - f
        w1 = f
        return np.stack((w0, w1), axis=-1)

    # ------------------------------------------------------------------
    # Internal implementations (readable but efficient)
    # ------------------------------------------------------------------
    def _wrap_positions(self, positions):
        """Wrap positions into [0, box_size) periodically (per axis)."""
        return np.mod(positions, self.box_size)

    def _indices_and_fracs(self, positions):
        """
        Map positions to base-cell index and fractional offset within that cell.

        Returns
        -------
        i : (N, ndim) int64
        f : (N, ndim) float64 in [0, 1)
        """
        pos = self._wrap_positions(positions)
        xg = pos / self.dx
        i = np.floor(xg).astype(np.int64)
        f = xg - i
        n = np.array(self.grid_shape, dtype=np.int64)
        i %= n
        return i, f

    def _scatter_ngp(self, positions, q):
        """Nearest-Grid-Point deposition (single node per particle)."""
        n = np.array(self.grid_shape, dtype=np.int64)
        idx_cont = positions / self.dx
        idx = np.rint(idx_cont).astype(np.int64) % n
        flat = np.ravel_multi_index(idx.T, self.grid_shape)
        np.add.at(self.grid.ravel(), flat, q)

    def _scatter_cic(self, positions, q):
        """Cloud-In-Cell deposition to 2^ndim neighboring nodes."""
        i, f = self._indices_and_fracs(positions)
        n = np.array(self.grid_shape, dtype=np.int64)

        # Per-axis weights and neighbor indices
        weights = self._cic_kernel(f)            # (N, ndim, 2)
        j0 = i                                   # lower/left indices
        j1 = (i + 1) % n                         # upper/right indices (periodic)

        # Enumerate all 2^ndim corners using bit masks
        corners = 1 << self.ndim
        for mask in range(corners):
            # Build indices for this corner and combined weight
            idx_parts = []
            w = q.copy()
            for d in range(self.ndim):
                use1 = (mask >> d) & 1
                jd = j1[:, d] if use1 else j0[:, d]
                wd = weights[:, d, 1] if use1 else weights[:, d, 0]
                idx_parts.append(jd)
                w *= wd

            flat = np.ravel_multi_index(np.stack(idx_parts, axis=0), self.grid_shape)
            np.add.at(self.grid.ravel(), flat, w)


if __name__ == "__main__":
    # Simple smoke test
    rng = np.random.default_rng(0)

    grid = SamplingGrid(grid_shape=(32, 32, 32), box_size=1.0)

    N = 20_000
    positions = rng.random((N, 3)) * grid.box_size
    masses = np.full(N, 1.0 / N)

    grid.distribute_quantity(positions, masses, kernel="CIC")

    print("Grid shape:", grid.grid.shape)
    print("Total mass on grid:", grid.grid.sum())
