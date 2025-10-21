import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D
import time as tm

t0 = tm.time()
## WE SHALL START

c = 3e8          # Speed of light [m/s]
mu0 = 4e-7 * np.pi  # Vacuum permeability [H/m]
dx = 0.3         # Spatial step [m]
dt = (dx / c)*1e-2     # Time step [s]

def A_update(i, Amu, Jmu, c=c, dt=dt, mu0=mu0):
    """
    Updates the four-component field Aμ at time step i
    using a finite difference scheme (wave equation with source) in MKSA units!

    Parameters
    ----------
    i : int
        Current time index (i ≥ 2).
    Amu : ndarray, shape (4, nt, nx, ny, nz)
        Field Aμ to be updated.
    Jmu : ndarray, shape (4, nt, nx, ny, nz)
        Source current.
    c : float
        Speed of light [m/s].
    dt : float
        Time step [s].
    mu0 : float
        Vacuum permeability [H/m].

    Returns
    -------
    Amu_next : ndarray, shape (4, nx-2, ny-2, nz-2)
        Updated field (internal cells only, excluding boundaries).
    """

    if i < 2:
        raise ValueError("At least two previous time states are required (i ≥ 2).")

    # Internal cells
    A_prev = Amu[:, i-1, 1:-1, 1:-1, 1:-1]
    A_prev2 = Amu[:, i-2, 1:-1, 1:-1, 1:-1]
    J_prev = Jmu[:, i-1, 1:-1, 1:-1, 1:-1]

    # Discrete Laplacian
    laplacian = (
        Amu[:, i-1, 2:, 1:-1, 1:-1] + Amu[:, i-1, :-2, 1:-1, 1:-1] +
        Amu[:, i-1, 1:-1, 2:, 1:-1] + Amu[:, i-1, 1:-1, :-2, 1:-1] +
        Amu[:, i-1, 1:-1, 1:-1, 2:] + Amu[:, i-1, 1:-1, 1:-1, :-2] -
        6 * A_prev
    )*1e-2

    # Time update in MKSA
    Amu_next = 2 * A_prev - A_prev2 + laplacian + (2 * mu0 * dx**2) * J_prev

    return Amu_next


def compute_E(Amu, i, dx, dt):
    """
    Compute electric field E at time index i from Aμ.
    
    Parameters
    ----------
    Amu : ndarray, shape (4, nt, nx, ny, nz)
        4-potential field.
    i : int
        Time index.
    dx : float
        Spatial step (assume uniform).
    dt : float
        Time step.
        
    Returns
    -------
    E : ndarray, shape (3, nx-2, ny-2, nz-2)
        Electric field components Ex, Ey, Ez at internal cells.
    """
    phi = Amu[0, i, 1:-1, 1:-1, 1:-1]
    Ax = Amu[1, i, 1:-1, 1:-1, 1:-1]
    Ay = Amu[2, i, 1:-1, 1:-1, 1:-1]
    Az = Amu[3, i, 1:-1, 1:-1, 1:-1]

    # Partial derivatives using central differences
    dphi_dx = (Amu[0, i, 2:, 1:-1, 1:-1] - Amu[0, i, :-2, 1:-1, 1:-1]) / (2*dx)
    dphi_dy = (Amu[0, i, 1:-1, 2:, 1:-1] - Amu[0, i, 1:-1, :-2, 1:-1]) / (2*dx)
    dphi_dz = (Amu[0, i, 1:-1, 1:-1, 2:] - Amu[0, i, 1:-1, 1:-1, :-2]) / (2*dx)

    # Time derivatives of vector potential
    dAx_dt = (Amu[1, i, 1:-1, 1:-1, 1:-1] - Amu[1, i-1, 1:-1, 1:-1, 1:-1]) / dt
    dAy_dt = (Amu[2, i, 1:-1, 1:-1, 1:-1] - Amu[2, i-1, 1:-1, 1:-1, 1:-1]) / dt
    dAz_dt = (Amu[3, i, 1:-1, 1:-1, 1:-1] - Amu[3, i-1, 1:-1, 1:-1, 1:-1]) / dt

    Ex = -dphi_dx - dAx_dt
    Ey = -dphi_dy - dAy_dt
    Ez = -dphi_dz - dAz_dt

    return np.array([Ex, Ey, Ez])


def compute_B(Amu, i, dx):
    """
    Compute magnetic field B at time index i from Aμ.

    Parameters
    ----------
    Amu : ndarray, shape (4, nt, nx, ny, nz)
        4-potential field.
    i : int
        Time index.
    dx : float
        Spatial step (assume uniform).
        
    Returns
    -------
    B : ndarray, shape (3, nx-2, ny-2, nz-2)
        Magnetic field components Bx, By, Bz at internal cells.
    """
    Ax = Amu[1, i, 1:-1, 1:-1, 1:-1]
    Ay = Amu[2, i, 1:-1, 1:-1, 1:-1]
    Az = Amu[3, i, 1:-1, 1:-1, 1:-1]

    # Curl using central differences
    dAz_dy = (Amu[3, i, 1:-1, 2:, 1:-1] - Amu[3, i, 1:-1, :-2, 1:-1]) / (2*dx)
    dAy_dz = (Amu[2, i, 1:-1, 1:-1, 2:] - Amu[2, i, 1:-1, 1:-1, :-2]) / (2*dx)
    dAx_dz = (Amu[1, i, 1:-1, 1:-1, 2:] - Amu[1, i, 1:-1, 1:-1, :-2]) / (2*dx)
    dAz_dx = (Amu[3, i, 2:, 1:-1, 1:-1] - Amu[3, i, :-2, 1:-1, 1:-1]) / (2*dx)
    dAy_dx = (Amu[2, i, 2:, 1:-1, 1:-1] - Amu[2, i, :-2, 1:-1, 1:-1]) / (2*dx)
    dAx_dy = (Amu[1, i, 1:-1, 2:, 1:-1] - Amu[1, i, 1:-1, :-2, 1:-1]) / (2*dx)

    Bx = dAz_dy - dAy_dz
    By = dAx_dz - dAz_dx
    Bz = dAy_dx - dAx_dy

    return np.array([Bx, By, Bz])

## Simulation parameters

ns = 100

Amus = np.zeros((4, ns, 100, 100, 100))
Jmus = np.zeros((4, ns, 100, 100, 100))

## Initial condition: lets make an example with a Gaussian pulse in A0 at the center
x0, y0, z0 = 50, 50, 50
sigma = 3.0
for i in range(2):
    x = np.arange(100)
    y = np.arange(100)
    z = np.arange(100)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    Amus[0, i] = np.exp(-((X - x0)**2 + (Y - y0)**2 + (Z - z0)**2) / (2 * sigma**2))

## Simulation loop

for i in range(2, ns):
    Amus[:, i, 1:-1, 1:-1, 1:-1] = A_update(i, Amus, Jmus)


## Plotting and animating

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_zlim(0, 100)
def update_frame(frame):
    ax.cla()
    slice_index = 50
    A0_slice = Amus[0, frame, :, :, slice_index]
    X, Y = np.meshgrid(np.arange(100), np.arange(100))
    ax.scatter(X, Y, A0_slice.T, cmap='viridis')
    ax.set_title(f'Time step: {frame}')
    return ax,
ani = anim.FuncAnimation(fig, update_frame, frames=ns, interval=100)


plt.show()

## THIS IS THE END
t1 = tm.time()
print('This took:', t1 - t0, 'seconds')
