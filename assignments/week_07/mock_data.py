"""
mock_galaxy.py

Generates 3D particle coordinates for a mock barred spiral galaxy.

Students don't need to use this notebook directly unless doing the advanced timing tests, 
but it may be interesting to see how the mock data was constructed.
It can also be used to check your answers.

The galaxy disk lies in the x-y plane by construction, so the galaxy is
face-on when viewed along the z-axis. This makes it easy to verify that
the inertia tensor method recovers the correct orientation.

Components
----------
- Bulge       : spherical, Plummer density profile
- Bar         : elongated ellipsoid along x-axis
- Disk        : exponential surface density, thin vertical profile
- Spiral arms : two logarithmic spiral arms with gaussian scatter

Output
------
Saves particle positions to 'galaxy_particles.npy' as an (N, 3) array
of (x, y, z) coordinates in arbitrary units (kpc-like).
Produces a quick diagnostic plot.

Author: PHY 225 course materials
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=42)


# ── Component sizes (number of particles) ────────────────────────────────────
# ADVANCED TIMING TESTS:
# edit num_scale to change total number of particles
# currently, the code will save the file with the same name,
# so you will want to come up with a way to keep track of which file corresponds to which num_scale 
# e.g., by renaming the output files manually or (better) automatically

num_scale = 1000

N_BULGE  = 2*num_scale
N_BAR    = 3*num_scale
N_DISK   = 8*num_scale
N_ARM    = 5*num_scale    # per arm, so 10_000 total


# ── Galaxy shape parameters (all distances in kpc-like units) ────────────────

# Bulge: Plummer sphere with scale radius a
BULGE_SCALE   = 0.8       # kpc — controls how concentrated the bulge is

# Bar: semi-axes of the ellipsoid (elongated along x)
BAR_A = 4.0               # half-length along x
BAR_B = 1.2               # half-width along y
BAR_C = 0.6               # half-height along z

# Disk: exponential scale radius and thin vertical scale height
DISK_R_SCALE  = 5.0       # kpc
DISK_Z_SCALE  = 0.3       # kpc — small relative to R to make disk flat

# Spiral arms: logarithmic spiral r = R0 * exp(b * theta)
ARM_R0        = 4.0       # kpc — radius where arms leave the bar ends
ARM_B         = 0.2       # controls how tightly wound the arms are
ARM_THETA_MAX = 2 * np.pi # how far the arms wrap (radians)
ARM_WIDTH     = 0.6       # kpc — gaussian scatter perpendicular to arm
ARM_Z_SCALE   = 0.25      # kpc — arm vertical thickness


# ── Helper: sample Plummer sphere ────────────────────────────────────────────

def sample_plummer(n, a):
    """
    Returns (n, 3) positions drawn from a Plummer density profile.

    The Plummer profile has density  rho(r) ∝ (1 + r²/a²)^(-5/2).
    We sample it by inverse-CDF on the mass enclosed:
        M(r)/M_total = [r² / (r² + a²)]^(3/2)
    """
    u = rng.uniform(0, 1, n)
    r = a / np.sqrt(u ** (-2/3) - 1)

    # Uniform points on the unit sphere
    cos_theta = rng.uniform(-1, 1, n)
    phi       = rng.uniform(0, 2 * np.pi, n)
    sin_theta = np.sqrt(1 - cos_theta**2)

    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * cos_theta
    return np.column_stack([x, y, z])


# ── Helper: sample uniform ellipsoid ─────────────────────────────────────────

def sample_ellipsoid(n, a, b, c):
    """
    Returns (n, 3) positions uniform inside an ellipsoid with semi-axes a, b, c.

    Strategy: sample uniformly inside a sphere of radius 1 using rejection,
    then scale axes.
    """
    pts = []
    while len(pts) < n:
        batch = rng.uniform(-1, 1, size=(2 * n, 3))
        r2    = np.sum(batch**2, axis=1)
        batch = batch[r2 <= 1]
        pts.append(batch)
    pts = np.vstack(pts)[:n]
    pts[:, 0] *= a
    pts[:, 1] *= b
    pts[:, 2] *= c
    return pts


# ── Helper: sample exponential disk ──────────────────────────────────────────

def sample_disk(n, r_scale, z_scale):
    """
    Returns (n, 3) positions from an exponential disk.

    Radial profile: Σ(R) ∝ exp(-R / r_scale), sampled by inverse CDF.
    Vertical profile: p(z) ∝ exp(-|z| / z_scale), sampled by inverse CDF.
    """
    # Radial CDF: M(R)/M_total = 1 - (1 + R/h)*exp(-R/h)
    # Sample by numerical inversion via uniform draw
    u = rng.uniform(0, 1, n)
    # Iterative solution (Newton's method) for R from CDF
    R = r_scale * np.ones(n)
    for _ in range(50):
        f    = 1 - (1 + R / r_scale) * np.exp(-R / r_scale) - u
        dfdR = (R / r_scale) * np.exp(-R / r_scale) / r_scale
        R   -= f / dfdR
    R = np.abs(R)

    phi = rng.uniform(0, 2 * np.pi, n)
    x   = R * np.cos(phi)
    y   = R * np.sin(phi)

    # Vertical: double-exponential (Laplace distribution)
    z = rng.laplace(loc=0, scale=z_scale, size=n)
    return np.column_stack([x, y, z])


# ── Helper: sample logarithmic spiral arm ────────────────────────────────────

def sample_spiral_arm(n, r0, b, theta_max, width, z_scale, phi_offset=0):
    """
    Returns (n, 3) positions scattered around a logarithmic spiral arm.

    The arm follows  r(theta) = r0 * exp(b * theta),
    with particles scattered transversely with gaussian width and
    a thin vertical distribution.

    phi_offset rotates the arm (use 0 and pi for a two-armed galaxy).
    """
    theta = rng.uniform(0, theta_max, n)
    r_arm = r0 * np.exp(b * theta)

    # Scatter perpendicular to the arm (approximated as radial scatter here)
    r = r_arm + rng.normal(0, width, n)
    r = np.abs(r)

    phi = theta + phi_offset

    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = rng.laplace(loc=0, scale=z_scale, size=n)
    return np.column_stack([x, y, z])


# ── Generate all components ───────────────────────────────────────────────────

print("Generating galaxy components...")

bulge  = sample_plummer(N_BULGE, BULGE_SCALE)
bar    = sample_ellipsoid(N_BAR, BAR_A, BAR_B, BAR_C)
disk   = sample_disk(N_DISK, DISK_R_SCALE, DISK_Z_SCALE)
arm1   = sample_spiral_arm(N_ARM, ARM_R0, ARM_B, ARM_THETA_MAX,
                            ARM_WIDTH, ARM_Z_SCALE, phi_offset=0)
arm2   = sample_spiral_arm(N_ARM, ARM_R0, ARM_B, ARM_THETA_MAX,
                            ARM_WIDTH, ARM_Z_SCALE, phi_offset=np.pi)

all_particles = np.vstack([bulge, bar, disk, arm1, arm2])

N_TOTAL = len(all_particles)
print(f"Total particles: {N_TOTAL:,}")
print(f"  Bulge:       {N_BULGE:,}")
print(f"  Bar:         {N_BAR:,}")
print(f"  Disk:        {N_DISK:,}")
print(f"  Spiral arms: {N_ARM * 2:,}")


# ── Assign particle masses ────────────────────────────────────────────────────
#
# Masses are assigned using the pre-rotation positions, where the physical
# structure (distance from center) is well-defined.
#
# Physical motivation: galaxy mass surface density falls off roughly
# exponentially with radius. We reflect this by mapping each particle's
# 3D distance from the center to a mass in [10^6, 10^8] solar masses using:
#
#   log10(m) = log10(m_min) + (log10(m_max) - log10(m_min)) * exp(-r / r_mass)
#
# This gives central particles masses near 10^8 M_sun, tapering smoothly
# toward 10^6 M_sun in the outer disk.  A small amount of lognormal scatter
# is added so the distribution isn't perfectly smooth.

M_MIN      = 1e6    # solar masses
M_MAX      = 1e8    # solar masses
R_MASS     = 3.0    # kpc — sets how quickly mass falls off with radius
MASS_SIGMA = 0.3    # dex — lognormal scatter around the smooth profile

r = np.linalg.norm(all_particles, axis=1)    # distance from center, pre-rotation

log_m_min  = np.log10(M_MIN)
log_m_max  = np.log10(M_MAX)

log_mass   = log_m_min + (log_m_max - log_m_min) * np.exp(-r / R_MASS)
log_mass  += rng.normal(0, MASS_SIGMA, size=len(r))    # add scatter
log_mass   = np.clip(log_mass, log_m_min, log_m_max)   # enforce hard bounds

masses = 10**log_mass     # shape (N,), units of solar masses

print(f"\nParticle masses (solar masses):")
print(f"  Min:    {masses.min():.3e}")
print(f"  Max:    {masses.max():.3e}")
print(f"  Median: {np.median(masses):.3e}")
print(f"  Total:  {masses.sum():.3e}")


# ── Apply arbitrary rotation ──────────────────────────────────────────────────
#
# Goal: tilt the galaxy so the disk is NOT face-on in any of the three
# coordinate projections (x-y, x-z, y-z).
#
# Strategy: rather than guess Euler angles, we explicitly choose where we want
# the disk normal to point after rotation, then build R using Rodrigues'
# rotation formula.
#
# The disk normal starts as n_0 = (0, 0, 1).
# We want it to end up at n_target = (1, 1, 1)/sqrt(3), which has equal
# components in all three axes — guaranteeing no projection looks face-on.
#
# Rodrigues' formula gives the rotation matrix that takes n_0 to n_target:
#   R = I + sin(θ) K + (1 - cos(θ)) K²
# where K is the skew-symmetric matrix of the rotation axis k = n_0 × n_target,
# and θ is the angle between n_0 and n_target.

n0     = np.array([0.0, 0.0, 1.0])
ntarget = np.array([1.0, 1.0, 1.0])
ntarget = ntarget / np.linalg.norm(ntarget)     # normalize

cos_theta = np.dot(n0, ntarget)                 # = 1/sqrt(3) ≈ 0.577
sin_theta = np.sqrt(1 - cos_theta**2)
theta     = np.arccos(cos_theta)

k = np.cross(n0, ntarget)                       # rotation axis (unnormalized)
k = k / np.linalg.norm(k)                       # normalize

# Skew-symmetric cross-product matrix K such that K @ v = k × v
K = np.array([
    [ 0,    -k[2],  k[1]],
    [ k[2],  0,    -k[0]],
    [-k[1],  k[0],  0   ],
])

R_tilt = np.eye(3) + sin_theta * K + (1 - cos_theta) * (K @ K)

# Add an in-plane twist (rotation about n_target) so the bar doesn't
# happen to lie along a convenient axis after tilting.
twist_angle = np.radians(47)
cos_t = np.cos(twist_angle)
sin_t = np.sin(twist_angle)
K2 = np.array([
    [ 0,        -ntarget[2],  ntarget[1]],
    [ ntarget[2],  0,        -ntarget[0]],
    [-ntarget[1],  ntarget[0],  0        ],
])
R_twist = np.eye(3) + sin_t * K2 + (1 - cos_t) * (K2 @ K2)

R = R_twist @ R_tilt

# Verify: the disk normal should now point along ntarget
rotated_normal = R @ n0
print("\nRotated disk normal (should be close to [0.577, 0.577, 0.577]):")
print(np.round(rotated_normal, 4))
print("Rotation matrix R:")
print(np.round(R, 4))

rotated_particles = all_particles @ R.T


# ── Save to file ──────────────────────────────────────────────────────────────
#
# Positions and masses are saved as separate .npy files so students can
# load them independently and inspect their shapes before combining them.

np.save("galaxy_particles.npy", rotated_particles)
np.save("galaxy_masses.npy", masses)
print("\nSaved: galaxy_particles.npy  —  shape:", rotated_particles.shape)
print("Saved: galaxy_masses.npy     —  shape:", masses.shape)


# ── Diagnostic plot ───────────────────────────────────────────────────────────
#
# Row 1: face-on and edge-on views BEFORE rotation (uniform color)
# Row 2: x-y and x-z projections AFTER rotation, colored by log10(mass)
#         to verify that the mass distribution follows the galaxy structure.

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Mock Barred Spiral Galaxy — Positions and Masses", fontsize=13)

x0, y0, z0 = all_particles[:, 0], all_particles[:, 1], all_particles[:, 2]
x1, y1, z1 = rotated_particles[:, 0], rotated_particles[:, 1], rotated_particles[:, 2]
log_m_plot  = np.log10(masses)

# -- Before rotation --
axes[0, 0].scatter(x0, y0, s=0.3, alpha=0.3, color="steelblue")
axes[0, 0].set_xlabel("x (kpc)")
axes[0, 0].set_ylabel("y (kpc)")
axes[0, 0].set_title("Before rotation — face-on (x-y)")
axes[0, 0].set_aspect("equal")
axes[0, 0].set_xlim(-20, 20)
axes[0, 0].set_ylim(-20, 20)

axes[0, 1].scatter(x0, z0, s=0.3, alpha=0.3, color="steelblue")
axes[0, 1].set_xlabel("x (kpc)")
axes[0, 1].set_ylabel("z (kpc)")
axes[0, 1].set_title("Before rotation — edge-on (x-z)")
axes[0, 1].set_aspect("equal")
axes[0, 1].set_xlim(-20, 20)
axes[0, 1].set_ylim(-20, 20)

# -- After rotation, colored by log10(mass) --
sc1 = axes[1, 0].scatter(x1, y1, s=0.3, alpha=0.4, c=log_m_plot,
                          cmap="inferno", vmin=6, vmax=8)
axes[1, 0].set_xlabel("x (kpc)")
axes[1, 0].set_ylabel("y (kpc)")
axes[1, 0].set_title("After rotation — x-y (colored by log₁₀ mass)")
axes[1, 0].set_aspect("equal")
axes[1, 0].set_xlim(-20, 20)
axes[1, 0].set_ylim(-20, 20)
plt.colorbar(sc1, ax=axes[1, 0], label="log₁₀(M / M☉)")

sc2 = axes[1, 1].scatter(x1, z1, s=0.3, alpha=0.4, c=log_m_plot,
                          cmap="inferno", vmin=6, vmax=8)
axes[1, 1].set_xlabel("x (kpc)")
axes[1, 1].set_ylabel("z (kpc)")
axes[1, 1].set_title("After rotation — x-z (colored by log₁₀ mass)")
axes[1, 1].set_aspect("equal")
axes[1, 1].set_xlim(-20, 20)
axes[1, 1].set_ylim(-20, 20)
plt.colorbar(sc2, ax=axes[1, 1], label="log₁₀(M / M☉)")

plt.tight_layout()
plt.savefig("galaxy_diagnostic.png", dpi=150)
plt.show()
print("Saved: galaxy_diagnostic.png")