# %% [markdown]
# # Coupled Oscillators in a Circular Configuration
# 
# ## Problem Overview
# We consider a system of $ n $ coupled oscillators arranged in a circular configuration. The oscillators are connected by springs, and the system exhibits coupled harmonic motion. 
# 
# We aim to:
# 1. Derive the equations of motion for the system.
# 2. Solve the equations analytically and numerically.
# 3. Analyze the normal modes and angular frequencies.
# 4. Visualize the motion of oscillators using plots and animations.

# %%
import numpy as np
from scipy.linalg import eigh
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from IPython.display import HTML

# Input parameters
n = int(input("Enter the number of oscillators (e.g., 6 for a hexagonal circular system)[default 6]: ") or 6)
masses = []
for i in range(n):
    m = float(input(f"Enter the mass of oscillator {i + 1} (kg) [default 1.0]: ") or 1.0)
    masses.append(m)

spring_constants = []
for i in range(n):
    k = float(input(f"Enter spring constant K{i + 1} (N/m) [default 1.0]: ") or 1.0)
    spring_constants.append(k)

# Initial angular displacements and angular velocities
initial_angles = []
initial_angular_velocities = []
for i in range(n):
    theta0 = float(input(f"Enter initial angular displacement of oscillator {i + 1} (radians) [default 0.0]: ") or 0.0)
    omega0 = float(input(f"Enter initial angular velocity of oscillator {i + 1} (rad/s) [default 0.0]: ") or 0.0)
    initial_angles.append(theta0)
    initial_angular_velocities.append(omega0)

# Time parameters
t_start = float(input("Enter start time for simulation (s) [default 0.0]: ") or 0.0)
t_end = float(input("Enter end time for simulation (s) [default 20.0]: ") or 20.0)
num_points = int((t_end - t_start) * 50)
t_eval = np.linspace(t_start, t_end, num_points)

# %% [markdown]
# ## Derivation of Equations of Motion
# 
# ### General Setup
# - Let each oscillator have mass $ m_i $ and connect to its neighbors via springs with spring constants $ k_i $.
# - Denote the angular displacement of the $ i $-th oscillator by $ \theta_i(t) $.
# 
# ### Equations of Motion
# For each oscillator, Newton's second law gives:
# $$
# m_i \frac{d^2 \theta_i}{dt^2} = -k_i (\theta_i - \theta_{i+1}) - k_{i-1} (\theta_i - \theta_{i-1}),
# $$
# where indices are taken modulo $ n $ to account for the circular arrangement.
# 
# Rewriting in matrix form:
# $$
# \mathbf{M} \ddot{\boldsymbol{\theta}} = -\mathbf{K} \boldsymbol{\theta},
# $$
# where:
# - $ \mathbf{M} = \text{diag}(m_1, m_2, \dots, m_n) $ is the mass matrix,
# - $ \mathbf{K} $ is the stiffness (coupling) matrix.
# 
# The coupling matrix $ \mathbf{K} $ is given by:
# $$
# K_{ii} = k_i + k_{i-1}, \quad
# K_{i,i+1} = -k_i, \quad
# K_{i,i-1} = -k_{i-1}.
# $$
# 
# ### Assumptions for Simplification
# 1. Uniform masses: $ m_i = m \; \forall i $.
# 2. Uniform spring constants: $ k_i = k \; \forall i $.
# 
# This leads to a symmetric coupling matrix $ \mathbf{C} $:
# $$
# C_{ij} =
# \begin{cases}
# \frac{2k}{m} & \text{if } i = j, \\
# -\frac{k}{m} & \text{if } j = i \pm 1 \text{ (mod $ n $)}, \\
# 0 & \text{otherwise}.
# \end{cases}
# $$
# 
# ### Analytical Solution
# To solve for normal modes, we find the eigenvalues $ \lambda_i $ and eigenvectors $ \mathbf{v}_i $ of $ \mathbf{C} $. The eigenvalues are related to the squared angular frequencies:
# $$
# \omega_i^2 = \lambda_i.
# $$
# 
# For $ n = 6 $, the eigenvalues can be explicitly calculated using the symmetry of the system.
# 
# ---
# 
# ## Analytical Solution for $ n = 6 $
# 
# ### Eigenvalue Calculation
# Using the symmetry of the circular configuration, the eigenvalues of $ \mathbf{C} $ are:
# $$
# \lambda_k = \frac{2k}{m} \left[ 1 - \cos\left(\frac{2\pi k}{n}\right) \right], \quad k = 0, 1, \dots, n-1.
# $$
# 
# ### Eigenvectors
# The eigenvectors $ \mathbf{v}_k $ correspond to sinusoidal variations across the oscillators:
# $$
# v_k(j) = \cos\left(\frac{2\pi k j}{n}\right), \quad j = 0, 1, \dots, n-1.
# $$

# %% [markdown]
# ## Coupling Matrix and Normal Modes Analysis
# 
# ### Coupling Matrix for Angular Displacements in a Circular Configuration
# The coupling matrix \( C \) governs the interactions between oscillators, accounting for their neighbors' influence. It is derived based on the spring constants and masses of the oscillators.

# %%
# Define the coupling matrix
def get_coupling_matrix(K, m):
    """
    Constructs the coupling matrix for angular displacements in a circular configuration.

    Parameters:
    - K: array of spring constants, size n.
    - m: array of masses, size n.

    Returns:
    - C: symmetric coupling matrix of size (n x n).
    """
    C = np.zeros((n, n))  # Initialize an n x n zero matrix
    for i in range(n):
        C[i, i] = (K[i] + K[i - 1]) / m[i]  # Diagonal: self-coupling
        C[i, (i - 1) % n] = -K[i - 1] / m[i]  # Left neighbor coupling
        C[i, (i + 1) % n] = -K[i] / m[i]  # Right neighbor coupling
    return (C + C.T) / 2  # Ensure matrix symmetry due to numerical errors

# Compute the coupling matrix
C = get_coupling_matrix(spring_constants, masses)

# Eigenvalue and eigenvector computation
w2, A = eigh(C)  # w2: Eigenvalues, A: Eigenvectors

# Check for numerical issues: negative eigenvalues
negative_eigenvalues = w2[w2 < 0]
if len(negative_eigenvalues) > 0:
    print("Warning: Negative eigenvalues detected!")
    print("Negative eigenvalues:", negative_eigenvalues)
    print("This may indicate instability or numerical issues in the model.")

# Correct small negative eigenvalues due to numerical error
tolerance = 1e-10
w2[w2 < tolerance] = 0  # Replace small negatives with zero

# Compute angular frequencies
frequencies = np.sqrt(w2)  # Angular frequencies

# Compute mode amplitudes
amp = np.linalg.inv(A) @ initial_angles  # Amplitudes for each normal mode

# Construct the normal modes
modes = np.zeros((n, len(t_eval)))
for i in range(n):
    modes[i, :] = amp[i] * np.cos(frequencies[i] * t_eval)  # Normal mode evolution

# Reconstruct the total motion
theta_sum = A @ modes  # Superpose normal modes

# %% [markdown]
# ## Explanation of the Code
# 
# ### Coupling Matrix Construction:
# - The diagonal elements $ C[i, i] $ represent the combined coupling from the neighboring springs to the $ i $-th oscillator.
# - Off-diagonal elements $ C[i, (i \pm 1) \% n] $ handle the coupling to the left and right neighbors in the circular configuration.
# 
# ### Eigenvalue Analysis:
# - Eigenvalues ($ w^2 $) represent the squared angular frequencies of the system's normal modes.
# - A warning is issued if negative eigenvalues are detected, as they indicate potential model instability or numerical errors.
# 
# ### Frequency Computation:
# - Non-negative eigenvalues are used to compute angular frequencies $ \omega = \sqrt{w^2} $.
# 
# ### Mode Amplitudes:
# - Using the initial conditions, the amplitudes of the system's normal modes are calculated as:
#   $$
#   \text{amp} = A^{-1} \cdot \text{initial-angles}.
#   $$
# 
# ### Normal Mode Evolution:
# - The time evolution of each normal mode is computed using:
#   $$
#   \text{mode}_i(t) = \text{amp}_i \cdot \cos(\omega_i \cdot t).
#   $$
# 
# ### Total Motion Reconstruction:
# - The total motion of the oscillators is reconstructed as a superposition of normal modes:
#   $$
#   \boldsymbol{\theta}(t) = \mathbf{A} \cdot \text{modes}(t),
#   $$
#   where $ \mathbf{A} $ is the eigenvector matrix.
# 
# This implementation allows for a robust numerical solution and analysis of the system's dynamics.
# 

# %%
# Define the system of ODEs
def coupled_angular_odes(t, y):
    theta = y[:n]  # Angular displacements
    omega = y[n:]  # Angular velocities
    dtheta_dt = omega
    domega_dt = -C @ theta
    return np.concatenate([dtheta_dt, domega_dt])

# Initial conditions
initial_conditions = np.concatenate([initial_angles, initial_angular_velocities])

# Solve the system numerically
solution = solve_ivp(coupled_angular_odes, [t_start, t_end], initial_conditions, t_eval=t_eval)

# Extract angular displacements over time
t = solution.t
theta = solution.y[:n]

# %% [markdown]
# ## Explanation of the Code
# 
# ### System of ODEs:
# - The coupled system of ordinary differential equations (ODEs) is defined as:
#   - $ \theta $: Angular displacements of the oscillators ($ y[:n] $).
#   - $ \omega $: Angular velocities of the oscillators ($ y[n:] $).
#   - The equations of motion are split into:
#     - $ \frac{d\theta}{dt} = \omega $: The rate of change of angular displacement.
#     - $ \frac{d\omega}{dt} = -\mathbf{C} \cdot \theta $: The rate of change of angular velocity, derived from the coupling matrix $ \mathbf{C} $.
# 
# ### Initial Conditions:
# - The initial conditions combine:
#   - **Initial angular displacements**: $ \theta_0 $.
#   - **Initial angular velocities**: $ \omega_0 $.
# - These conditions are concatenated into a single array: 
#   $
#   \text{initial-conditions} = \begin{bmatrix}
#   \theta_0 \\
#   \omega_0
#   \end{bmatrix}.
#   $
# 
# ### Numerical Solution:
# - The `solve_ivp` function is used to numerically solve the coupled ODEs:
#   - **Time range**: From $ t_{\text{start}} $ to $ t_{\text{end}} $.
#   - **Evaluation points**: Specified by $ t_{\text{eval}} $.
#   - The solution provides the angular displacements ($ \theta $) and velocities ($ \omega $) at each time step.
# 
# ### Extraction of Results:
# - The time points ($ t $) and angular displacements ($ \theta $) are extracted from the solution for further analysis and visualization.
# 
# This section numerically computes the time evolution of the oscillators in the coupled system using the defined ODEs and initial conditions.
# 

# %% [markdown]
# ## Visualization of Results
# 
# ### 1. Reconstructed Motion Using Normal Modes
# - This plot shows the reconstructed motion of each oscillator using the superposition of normal modes.
# - **Key Details**:
#   - Each curve corresponds to the angular displacement of an oscillator reconstructed from the normal modes.
#   - The reconstruction uses the eigenvector matrix $ \mathbf{A} $ and time-evolved modes.
# 
# ---
# 
# ### 2. Numerical Solution of Coupled Oscillators
# - This plot represents the true motion of each oscillator computed numerically by solving the system of ODEs.
# - **Key Details**:
#   - $ \theta_i(t) $: Angular displacement of the $ i $-th oscillator over time.
#   - Each curve shows the independent evolution of an oscillator's angular displacement.
# 
# ---
# 
# ### 3. Visualization of Each Normal Mode
# - This set of plots shows the contribution of each normal mode to the motion of the system.
# - **Key Details**:
#   - Each row represents a separate mode, labeled with its angular frequency $ \omega $.
#   - The $ i $-th oscillator's contribution to the $ j $-th mode is scaled by the eigenvector matrix $ A[i, j] $.
# 

# %%
# 1. Reconstructed Motion Using Normal Modes
plt.figure(figsize=(20, 6))
for i in range(n):
    plt.plot(t, theta_sum[i, :], label=f"Reconstructed x_{i+1}(t)")
plt.title("Reconstructed Motion Using Normal Modes of Coupled Oscillators", pad=20)
plt.xlabel("Time (s)")
plt.ylabel("Angular Displacement (radians)")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
plt.grid(True)
plt.annotate("Each curve represents an oscillator's motion reconstructed from normal modes.",
             xy=(0.5, 0.9), xycoords='axes fraction', ha='center', fontsize=10)
plt.tight_layout()
plt.show()

# 2. Numerical Solution of Coupled Oscillators
plt.figure(figsize=(20, 6))
for i in range(n):
    plt.plot(t, theta[i], label=f"theta_{i+1}(t)")
plt.title("Numerical Solution of Coupled Oscillators", pad=20)
plt.xlabel("Time (s)")
plt.ylabel("Angular Displacement (radians)")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
plt.grid(True)
plt.annotate("This plot shows the true motion of each oscillator obtained numerically.",
             xy=(0.5, 0.9), xycoords='axes fraction', ha='center', fontsize=10)
plt.tight_layout()
plt.show()

# 3. Plot Each Normal Mode Separately (Vertical layout)
fig, axes = plt.subplots(n, 1, figsize=(20, 5 * n))  # n rows, 1 column
for mode_index in range(n):
    for i in range(n):
        x_i = A[i, mode_index] * modes[mode_index, :]
        axes[mode_index].plot(t, x_i, label=f"x_{i+1}")
    axes[mode_index].set_title(f"Mode {mode_index + 1}\nω = {frequencies[mode_index]:.3f} rad/s", pad=20)
    axes[mode_index].legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    axes[mode_index].grid(True)
    axes[mode_index].annotate(f"Mode {mode_index + 1} with frequency ω = {frequencies[mode_index]:.3f}",
                             xy=(0.5, 0.9), xycoords='axes fraction', ha='center', fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Output Description:
# 
# #### **Combined Animation:**
# - The total motion, as a sum of all normal modes, is animated to show the system's behavior comprehensively.
# - The combined motion is labeled as **"Combined Motion of All Modes"** in the title.

# %%
# Animation setup: Initialize figure and axes for circular configuration
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1.5, 1.5)  # Set x-axis limits
ax.set_ylim(-1.5, 1.5)  # Set y-axis limits

# Add a gray dashed circle to represent the equilibrium positions
circle = plt.Circle((0, 0), 1, color="gray", fill=False, linestyle="--")
ax.add_artist(circle)

# Initialize points for the oscillators
points, = ax.plot([], [], "bo-", lw=2)  # Blue dots connected by lines
ax.grid()  # Add grid for reference

# Title and axis labels for clarity
ax.set_title("Oscillator Motion in Circular Configuration", fontsize=14, pad=20)
ax.set_xlabel("X Position (m)", fontsize=12)
ax.set_ylabel("Y Position (m)", fontsize=12)

# Initialize the animation by clearing the data
def init():
    points.set_data([], [])  # No data initially
    return points,

# Update function for each frame of the animation
def update(frame):
    # Compute angular positions in equilibrium (evenly spaced)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Get angular displacements at the current time step
    displacements = theta[:, frame]
    # Update total angles by adding displacements to equilibrium positions
    total_angles = angles + displacements

    # Convert updated angles to Cartesian coordinates
    x_coords = np.cos(total_angles)  # X-coordinates on the circle
    y_coords = np.sin(total_angles)  # Y-coordinates on the circle

    # Update the points data and close the loop by reconnecting the first point
    points.set_data(np.append(x_coords, x_coords[0]), np.append(y_coords, y_coords[0]))
    return points,

# Create and save the animation
ani = anim.FuncAnimation(
    fig, update, frames=len(t_eval), init_func=init, blit=True, interval=50
)
plt.close()  # Prevents static plot display

# Display the animation in Jupyter Notebook
HTML(ani.to_jshtml())

# %% [markdown]
# ## Visualization of Coupled Oscillators in a Circular Configuration
# 
# ### 1. Animation Overview:
# This animation showcases the dynamic motion of oscillators arranged in a **circular configuration**. The system evolves over time based on the angular displacements calculated for each oscillator.
# 
# ---
# 
# ### Key Features of the Animation:
# 
# #### **Reconstructed Motion:**
# - The oscillators (points) are connected to visually represent their interactions, forming a closed loop.
# - As the animation progresses, each oscillator moves in the circle based on the combined effect of the normal modes.
# - The angular displacements evolve over time, showing the system's harmonic oscillations.
# 
# #### **Equilibrium and Perturbations:**
# - **Equilibrium Positions**:
#   - Initially, the oscillators are evenly spaced along the circumference of the circle.
# - **Perturbations**:
#   - The displacements caused by external or initial conditions result in oscillatory motion.
# 
# ---
# 
# ### 2. Motion Patterns in the Animation:
# The animation captures how the system's dynamics are influenced by the **natural frequencies** and **coupling effects**:
# - **Individual Oscillator Displacements**:
#   - Each oscillator's motion is dictated by its interaction with neighbors through coupling springs.
#   - The displacements are computed from the normal modes and visualized as oscillations around equilibrium positions.
# 
# - **Coupled Dynamics**:
#   - The coupling between oscillators results in synchronized yet distinct motion patterns.
#   - At times, oscillators move in-phase, while at others, they oscillate out-of-phase, depending on the system's mode contributions.
# 
# ---
# 
# ### 3. Key Details in the Animation:
# 
# #### **Spring Configuration**:
# - The animation dynamically updates the positions of oscillators based on their angular displacements.
# - The circular layout helps highlight the symmetry and interactions in the coupled system.
# 
# #### **Phase Relationships**:
# - Different oscillators can oscillate with varying amplitudes and phase shifts, demonstrating the complexity of coupled motion.
# 
# #### **Superposition of Modes**:
# - The total motion is the result of the **superposition** of individual modes, where each mode contributes to the system's behavior based on its frequency and amplitude.
# 
# ---
# 
# ### 4. Observations During the Animation:
# 
# - **Smooth Transitions**:
#   - The oscillators smoothly transition between positions, highlighting harmonic motion.
#   
# - **Coupling Effects**:
#   - Oscillators influence each other’s motion through their neighbors, maintaining the circular configuration.
# 
# - **System Stability**:
#   - The animation provides visual feedback on the system's stability. Unstable systems would exhibit divergent motion patterns or non-periodic oscillations.
# 
# ---
# 
# ### Conclusion:
# This animation visually represents the motion of coupled oscillators in a circular configuration. It highlights the complex interplay between individual angular displacements, coupling effects, and normal modes of vibration. The dynamic visualization provides a deeper understanding of how physical properties like mass, spring constants, and coupling influence the system's behavior.
# 


