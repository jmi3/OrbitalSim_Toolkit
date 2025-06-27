# 🛰️ Particle-Mesh Integration in OrbitalSim\_Toolkit

## ✅ Your Design Choice

You chose to:

* **Integrate Particle-Mesh (PM) methods** into the existing `Hamiltonian` class hierarchy (specifically, via a `PMHamiltonian` subclass)
* **Keep the current RK integrator (`RKp`) unchanged**, leveraging its modular `dHdq`, `dHdp` interface
* **Use Cloud-in-Cell (CIC)** interpolation for both mass assignment and force interpolation
* **Implement mass assignment and interpolation using precomputed stencils and Numba JIT compilation**, for high performance on your CPU-based laptop

---

## 🔧 Requirements

### Software

* Python 3.x
* [`numpy`](https://numpy.org/)
* [`scipy.fft`](https://docs.scipy.org/doc/scipy/tutorial/fft.html) or [`pyfftw`](https://pypi.org/project/pyFFTW/)
* [`numba`](https://numba.pydata.org/) for JIT-accelerated loops

### Hardware

* Your laptop: Acer TravelMate P215-53, 16 GB RAM, Intel i5 CPU
* Target capacity: $N \leq 10^7$ particles, $N_g \leq 256$ grid size (safe margin)

---

## 📐 Mathematical Summary

### 1. **Hamiltonian**

The system is governed by the Hamiltonian:

$$
H(\mathbf{q}, \mathbf{p}) = \sum_i \frac{|\mathbf{p}_i|^2}{2m_i} + \frac{1}{2} \sum_i m_i \, \Phi(\mathbf{q}_i)
$$

* $\mathbf{q}_i$: particle positions
* $\mathbf{p}_i$: momenta
* $\Phi(\mathbf{x})$: gravitational potential from mass distribution

### 2. **Poisson Equation**

Gravitational potential satisfies:

$$
\nabla^2 \Phi(\mathbf{x}) = 4 \pi G \rho(\mathbf{x})
$$

Solving via Fourier Transform:

$$
\hat{\Phi}(\mathbf{k}) = -\frac{4 \pi G}{|\mathbf{k}|^2} \hat{\rho}(\mathbf{k})
$$

### 3. **Cloud-in-Cell (CIC) Mass Assignment**

Mass is assigned to the mesh using a trilinear kernel:

$$
W(\mathbf{x} - \mathbf{q}) = \prod_{d=1}^{3} \left(1 - \frac{|\Delta x_d|}{\Delta}\right)
\quad \text{for } |\Delta x_d| < \Delta
$$

For a particle at position $\mathbf{q}_i$, its mass $m_i$ is distributed over the 8 surrounding grid points:

$$
\rho(\mathbf{x}_{ijk}) += m_i \cdot W(x_{ijk} - q_i)
$$

### 4. **Force Computation**

* Compute $\Phi$ via FFT
* Compute acceleration $\mathbf{a}(\mathbf{x}) = -\nabla \Phi(\mathbf{x})$ via finite differences
* Interpolate force from mesh back to particles:

$$
\mathbf{a}_i = \sum_{\text{grid}} \mathbf{a}(\mathbf{x}) \cdot W(\mathbf{x} - \mathbf{q}_i)
$$

---

## 🧮 Algorithm Outline

### Step 1: Mass Assignment (CIC)

* For each particle:

  * Compute its base grid index `i = floor(q / dx)`
  * Compute fractional offset `d = q/dx - i`
  * For each stencil offset `(ox, oy, oz)` in `{0, 1}^3`:

    * Compute weight `w = ∏ (d or 1-d)`
    * Add `m * w` to `rho[i + ox, j + oy, k + oz]`

### Step 2: FFT Poisson Solver

* FFT of density: `rho_k = FFT(rho)`
* Multiply with Green’s kernel: `phi_k = rho_k * G_k`
* Inverse FFT: `phi = IFFT(phi_k)`

### Step 3: Gradient → Acceleration

* Compute `a_x = -∂φ/∂x`, `a_y`, `a_z` using central differences

### Step 4: Force Interpolation (CIC)

* For each particle:

  * Use same stencil and weights as in assignment
  * Interpolate acceleration from grid to particle

---

## ⚙️ Future Extensions

* Add optional **softening kernel** to avoid singular forces
* Add **hybrid PM + direct** or **PM + tree** scheme
* Enable **GPU acceleration** via CuPy or Taichi

