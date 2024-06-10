"""
This code runs a little simulation of argon gas, considering wall collisions and particle to particle collisions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import cdist

def normalize_quantities(quantities: np.array, N: int) -> np.array:
    """
    Normalizes the velocities or acceleration of the N particles.
    Args:
        velocities (np.array): Some quantity of the particles, of shape (N, 3)
        N (int): Number of particles 

    Returns:
        np.array: Normalized quantities 
    """
    average = np.sum(quantities, axis=0) / N
    return quantities - average

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


def leonnard_jones_force(r: float, epsilon: float, sigma: float):
    """
    Calculates the Leonnard Jones force between two particles of distance r apart. 
    """
    return (3 * epsilon) * (2 * (sigma / r)**12 - (sigma / r)**6)


def extend_to_magnitude(mag: float, dir: np.array): 
    """
    Calculates the vector of the magnitude mag in the direction dir. 
    """
    magnitude = np.linalg.norm(dir)
    unit_vector = dir / magnitude
    return unit_vector * mag 


# initialize simulation constants 
NUM_PARTICLES = 3
MASS = 1
DT = 0.01 # time steps
T_MAX = 10 # max time 
BOX_SIZE = 5 # size of the cube these little particles live in
MIN_INITIAL_VELOCITY = -3
MAX_INITIAL_VELOCITY = 3
MIN_INITIAL_ACCELERATION = -2
MAX_INITIAL_ACCELERATION = 2

NORMALIZATION_INTERVAL = 50 # every x number of steps, normalize velocities and acceleration to conserve momentum and force
COLLISION_RADIUS = 0.5 # particles that come this close to each other in distance will colllide

# initialize positions, velocities and time steps
positions = np.random.uniform(-BOX_SIZE / 2, BOX_SIZE / 2, size=(NUM_PARTICLES, 3))
initial_velocities = np.random.uniform(MIN_INITIAL_VELOCITY, MAX_INITIAL_VELOCITY, size=(NUM_PARTICLES, 3))
velocities = normalize_quantities(initial_velocities, NUM_PARTICLES)
initial_acceleration = np.random.uniform(MIN_INITIAL_ACCELERATION, MAX_INITIAL_ACCELERATION, size=(NUM_PARTICLES, 3))
acceleration = normalize_quantities(initial_acceleration, NUM_PARTICLES)

# leonnard jones potential constant to use
EPSILON = 0.238
SIGMA = 3.405

# initialize the scatter plot to visualize
fig = plt.figure(figsize=(8, 8))  
ax = fig.add_subplot(111, projection='3d')
np.random.seed(0)
colors = np.arange(NUM_PARTICLES)
scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors)
ax.set_xlim(-BOX_SIZE / 2, BOX_SIZE / 2)
ax.set_ylim(-BOX_SIZE / 2, BOX_SIZE / 2)
ax.set_zlim(-BOX_SIZE / 2, BOX_SIZE / 2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Molecular Dynamics Simulation of Argon')

def update(t):
    
    global positions, velocities, acceleration

    if t % 50 == 0: # every 50 time steps, normalize velocity
        velocities = normalize_quantities(velocities, NUM_PARTICLES)
        acceleration = normalize_quantities(acceleration, NUM_PARTICLES)
    
    positions = positions + (velocities * DT) + (1/2) * acceleration * DT * DT

    # calculating particle to particle collision
    distances = cdist(positions, positions, 'euclidean')
    particle_collisions = distances < COLLISION_RADIUS
    np.fill_diagonal(particle_collisions, False)

    if particle_collisions.any():
        # collision has happened between two particles, recalculate velocity for the collided particles
        collided_pairs = np.transpose(np.where(particle_collisions))
        filtered_pairs = filter_equivalent_pairs(collided_pairs)
        for p1, p2 in filtered_pairs:
            velocities[p1], velocities[p2] = calculate_collision(velocities[p1], velocities[p2], positions[p1], positions[p2])
    
    # calculating particle to wall collision, in this case, flip the dimension that it hits
    wall_collisions = abs(positions) > BOX_SIZE / 2 # this is a (N, 3) of T and F
    if wall_collisions.any():
        multiplier = np.ones(wall_collisions.shape)
        multiplier[wall_collisions] = -1
        velocities = np.multiply(velocities, multiplier)

    velocities = velocities + (acceleration * DT)

    np.fill_diagonal(distances, 1e20) # fill the diagonal with dummy values since particles should not interact with each other 
    LJ_function = np.vectorize(leonnard_jones_force)
    result = LJ_function(distances, EPSILON, SIGMA)
    np.fill_diagonal(result, 0) # fill the forces with 0 since particles have a zero force on each other
    accel_changes = np.zeros(positions.shape)
    for p1 in range(NUM_PARTICLES):
        for p2 in range(NUM_PARTICLES):
            if p1 == p2:
                continue
            else:
                force = result[p1, p2]
                vector = p2 - p1
                accel_changes[p1] += extend_to_magnitude(force, vector)

    acceleration = acceleration + accel_changes

    # updating scatterplot 
    scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])

ani = animation.FuncAnimation(fig, update, frames=int(T_MAX/DT), interval=50)

plt.show()