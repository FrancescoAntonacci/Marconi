import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D
import time as tm

t0 = tm.time()
## START SIMULATION ##

# Physical constants
c = 3e8            # Speed of light [m/s]
mu0 = 4e-7 * np.pi # Vacuum permeability [H/m]
dx = 0.3           # Spatial step [m]
dt = (dx / c) * 0.5  # Time step [s]

# --------------------------------------------------------------------
# --- UPDATE FUNCTION USING 3-BUFFER ROLLING SCHEME ------------------
# --------------------------------------------------------------------
def A_update(Amu, Jmu, c=c, dt=dt, mu0=mu0):
    """
    Updates the four-component field Aμ using a finite difference scheme
    (wave equation with source) in MKSA units.

    Parameters
    ----------
    Amu : ndarray, shape (4, 3, nx, ny, nz)
        Rolling buffer for field Aμ (only 3 time steps kept).
        [0] = t-2, [1] = t-1, [2] = to be computed.
    Jmu : ndarray, shape (4, 3, nx, ny, nz)
        Source current (same shape).
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

    A_prev  = Amu[:, 1, 1:-1, 1:-1, 1:-1]
    A_prev2 = Amu[:, 0, 1:-1, 1:-1, 1:-1]
    J_prev  = Jmu[:, 1, 1:-1, 1:-1, 1:-1]

    # Discrete Laplacian
    laplacian = (
        Amu[:, 1, 2:, 1:-1, 1:-1] + Amu[:, 1, :-2, 1:-1, 1:-1] +
        Amu[:, 1, 1:-1, 2:, 1:-1] + Amu[:, 1, 1:-1, :-2, 1:-1] +
        Amu[:, 1, 1:-1, 1:-1, 2:] + Amu[:, 1, 1:-1, 1:-1, :-2] -
        6 * A_prev
    )

    # Wave equation finite difference update
    Amu_next = 2 * A_prev - A_prev2 + (c * dt / dx)**2 * laplacian + (2 * mu0 * dx**2) * J_prev

    return Amu_next

# --------------------------------------------------------------------
# --- FIELD COMPUTATION FUNCTIONS ------------------------------------
# --------------------------------------------------------------------
def compute_E(Amu, dx, dt):
    phi = Amu[0, 1, 1:-1, 1:-1, 1:-1]
    dphi_dx = (Amu[0, 1, 2:, 1:-1, 1:-1] - Amu[0, 1, :-2, 1:-1, 1:-1]) / (2 * dx)
    dphi_dy = (Amu[0, 1, 1:-1, 2:, 1:-1] - Amu[0, 1, 1:-1, :-2, 1:-1]) / (2 * dx)
    dphi_dz = (Amu[0, 1, 1:-1, 1:-1, 2:] - Amu[0, 1, 1:-1, 1:-1, :-2]) / (2 * dx)
    dA_dt = (Amu[1:4, 1, 1:-1, 1:-1, 1:-1] - Amu[1:4, 0, 1:-1, 1:-1, 1:-1]) / dt
    Ex = -dphi_dx - dA_dt[0]
    Ey = -dphi_dy - dA_dt[1]
    Ez = -dphi_dz - dA_dt[2]
    return np.array([Ex, Ey, Ez])

def compute_B(Amu, dx):
    Ax = Amu[1, 1]; Ay = Amu[2, 1]; Az = Amu[3, 1]
    dAz_dy = (Az[:, 2:, :] - Az[:, :-2, :]) / (2 * dx)
    dAy_dz = (Ay[:, :, 2:] - Ay[:, :, :-2]) / (2 * dx)
    dAx_dz = (Ax[:, :, 2:] - Ax[:, :, :-2]) / (2 * dx)
    dAz_dx = (Az[2:, :, :] - Az[:-2, :, :]) / (2 * dx)
    dAy_dx = (Ay[2:, :, :] - Ay[:-2, :, :]) / (2 * dx)
    dAx_dy = (Ax[:, 2:, :] - Ax[:, :-2, :]) / (2 * dx)
    Bx = dAz_dy - dAy_dz
    By = dAx_dz - dAz_dx
    Bz = dAy_dx - dAx_dy
    return np.array([Bx, By, Bz])

# --------------------------------------------------------------------
# --- INITIAL CONDITIONS ---------------------------------------------
# --------------------------------------------------------------------
Amus = np.zeros((4, 3, 100, 100, 100))
Jmus = np.zeros((4, 3, 100, 100, 100))

x0, y0, z0 = 50, 50, 50
sigma = 3.0
x = np.arange(100)
y = np.arange(100)
z = np.arange(100)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Gaussian pulse in scalar potential A0
for i in range(2):
    Amus[0, i] = np.exp(-((X - x0)**2 + (Y - y0)**2 + (Z - z0)**2) / (2 * sigma**2))

# --------------------------------------------------------------------
# --- SIMULATION + VISUALIZATION LOOP --------------------------------
# --------------------------------------------------------------------
ns = 300                 # number of time steps
fps = 50               # frames per second

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_zlim(0, 100)

slice_index = 50
writer = anim.FFMpegWriter(fps=fps, bitrate=1800)
X, Y = np.meshgrid(np.arange(100), np.arange(100))

with writer.saving(fig, "4D_wave_equation04.mp4", dpi=100):
    for frame in range(2, ns):
        # --- Compute next time step ---
        Amus[:, 2, 1:-1, 1:-1, 1:-1] = A_update(Amus, Jmus)

        # --- Plot current state ---
        ax.cla()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_zlim(0, 3)
        ax.set_title(f"Time step: {frame}", fontsize=12)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("A₀ (scalar potential)")

        A0_slice = Amus[0, 2, :, :, slice_index]
        surf = ax.plot_surface(X, Y, A0_slice.T, cmap='viridis', edgecolor='none')

        if frame == 2:
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="A₀ magnitude")

        writer.grab_frame()
        # Tracker
        print(f"Rendered frame: {frame - 1}/{ns - 2}")

        # --- Rotate buffers (advance time) ---
        Amus[:, 0] = Amus[:, 1]
        Amus[:, 1] = Amus[:, 2]

plt.close(fig)

t1 = tm.time()
print("Simulation completed and rendered in:", round(t1 - t0, 2), "[s]")
