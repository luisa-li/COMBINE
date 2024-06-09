"""
This code runs a little simulation of argon gas, considering wall collisions and particle to particle collisions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import cdist

def normalize_velocities(velocities: np.array, N: int) -> np.array:
    """
    Normalizes the velocities of the N particles.
    Args:
        velocities (np.array): Velocities of the particles, of shape (N, 3)
        N (int): Number of particles 

    Returns:
        np.array: Normalized velocities 
    """
    average = np.sum(velocities, axis=0) / N
    return velocities - average

def proj(u: np.array, v: np.array) -> np.array:
    """
    Calculates the projection of u onto v.
    """
    dot = np.dot(u, v)
    mag_squared = np.dot(v, v)
    proj = (dot / mag_squared) * v 
    return proj

def calculate_collision(u: np.array, v: np.array, up: np.array, vp: np.array) -> tuple[np.array, np.array]: 
    """
    Calculates the velocity vector of two particles after their collision.
    u, v are the velocity vectors, and up and vp are the position vectors
    """

    dot = np.dot((u - v), (up - vp))
    if dot > 0:
        # particles are moving away from each other, do not make them collide again to avoid shaking
        return u, v
    else:
        vel_u = u - (2 * proj(u, u - v))
        vel_v = v - (2 * proj(v, u - v))
        return vel_u, vel_v

def filter_equivalent_pairs(pairs: np.array) -> np.array:
    """
    Filters out equivalent pairs from the np.array, which represents a list of pairs. 
    An example of an equivalent pair would be (3, 4) and (4, 3).

    Args:
        pairs (np.array): List of pairs.

    Returns:
        np.array: Deduplicated list of pairs.
    """
    tups = [tuple(sorted(pair)) for pair in pairs] # terrible runtime that I would not like to fix
    return list(set(tups))

# initialize simulation constants 
num_particles = 100
mass = 1.0
dt = 0.01
t_max = 10
box_min = -3
box_max = 3
min_initial_velocity = -3
max_initial_velocity = 3
normalization_interval = 50
collision_radius = 0.5

# initialize positions, velocities and time steps
positions = np.random.uniform(box_min, box_max, size=(num_particles, 3))
initial_velocities = np.random.uniform(min_initial_velocity, max_initial_velocity, size=(num_particles, 3))
velocities = normalize_velocities(initial_velocities, num_particles)
time_steps = np.arange(0, t_max, dt)

# initialize the scatter plot to visualize
fig = plt.figure(figsize=(8, 8))  # Adjust the figsize as needed
ax = fig.add_subplot(111, projection='3d')
np.random.seed(0)
colors = np.arange(num_particles)
scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors)
ax.set_xlim(box_min, box_max)
ax.set_ylim(box_min, box_max)
ax.set_zlim(box_min, box_max)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Molecular Dynamics Simulation of Argon')

def update(t):
    
    global positions, velocities

    if t % 50 == 0: # every 50 time steps, normalize velocity
        velocities = normalize_velocities(velocities, num_particles)
    
    positions += (velocities * dt)

    # calculating particle to particle collision
    distances = cdist(positions, positions, 'euclidean')
    particle_collisions = distances < collision_radius
    np.fill_diagonal(particle_collisions, False)

    if particle_collisions.any():
        # collision has happened between two particles, recalculate velocity for the collided particles
        collided_pairs = np.transpose(np.where(particle_collisions))
        filtered_pairs = filter_equivalent_pairs(collided_pairs)
        for p1, p2 in filtered_pairs:
            velocities[p1], velocities[p2] = calculate_collision(velocities[p1], velocities[p2], positions[p1], positions[p2])
    
    # calculating particle to wall collision, in this case, flip the dimension that it hits
    wall_collisions = abs(positions) > box_max # this is a (N, 3) of T and F
    if wall_collisions.any():
        multiplier = np.ones(wall_collisions.shape)
        multiplier[wall_collisions] = -1
        velocities = np.multiply(velocities, multiplier)

    # updating scatterplot 
    scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])

ani = animation.FuncAnimation(fig, update, frames=int(t_max/dt), interval=50)

plt.show()