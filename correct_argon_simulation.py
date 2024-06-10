import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import cdist

# other physical constants
ARGON_MASS = 6.63e-26 # kg, mass of argon
KB = 1.380e-23 # J/K, Boltzmanns' constant 

# defining the Leonnard Jones potential calculation
EPSILON = 1.658e-21 # kJ/mol, the well depth and how strongly the particles attract each other
SIGMA = 3.4e-10 # meters, the distance at which the intermolecular potential for the two particles is zero

# defining the constants used for the simulation
NUM_PARTICLES = 100
DT = 1e-13 # seconds
NSTEPS = 200 
BOX_SIZE = 100 * SIGMA
TEMP = 90 # kelvins
NORMALIZATION_INTERVAL = 25 # normalize every x steps to correct for numerical errors

# defining the scaling factor for the Maxwell-Boltzmann velocity distribution
SCALING_FACTOR = np.sqrt(KB * TEMP / ARGON_MASS) 

def normalize_velocities(quantities: np.array) -> np.array:
    """
    Normalizes the velocity over the N particles. 
    This means subtracting the velocity by its average in all three dimensions.
    Args:
        velocities (np.array): Velocity of the particles, of shape (N, 3)
        N (int): Number of particles 

    Returns:
        np.array: Normalized quantities 
    """
    average = np.sum(quantities, axis=0) / NUM_PARTICLES
    return quantities - average

def lennard_jones_force(r: float) -> float: 
    """
    Calculate the Lennard Jones force between argon atoms, this formula is obtained by taking the derivative of the Lennard Jones potential with respect to r. 
    The first term is repulsion, and the second term is atttraction. 

    Args:
        r (float): The distance between the two molecules. 

    Returns:
        float: The force between the two argon atoms. 
    """
    t1 = SIGMA**12 / r**13
    t2 = (1/2) * (SIGMA**6 / r**7)
    force = 48 * EPSILON * (t1 - t2)
    return force

def extend_to_magnitude(mag: float, dir: np.array): 
    """
    Calculates the vector of the magnitude mag in the direction dir. 
    """
    magnitude = np.linalg.norm(dir)
    unit_vector = dir / magnitude
    return unit_vector * mag 

def calculate_acceleration(positions: np.array) -> np.array:
    """
    Calculates the acceleration of the argon atoms according to their positions by calculating the Lennard Jones force on each of the atoms. 

    Args:
        positions (np.array): Positions of the argon atoms, shape (NUM_PARTICLES, 3)

    Returns:
        np.array: The acceleleration of the particles, shape (NUM_PARTICLES, N)
    """

    distances = cdist(positions, positions, 'euclidean')
    lj_function = np.vectorize(lennard_jones_force)
    np.fill_diagonal(distances, 1e-10) # filling the diagonal with a tiny value to avoid division by zero errors
    forces = lj_function(distances)
    np.fill_diagonal(forces, 0) # molecules have no force on themselves
    vectorized_forces = np.zeros(positions.shape)
    for p1 in range(NUM_PARTICLES):
        for p2 in range(NUM_PARTICLES):
            if p1 == p2:
                continue
            else:
                force = forces[p1, p2]
                vector = p1 - p2
                vectorized_forces[p1] += extend_to_magnitude(force, vector)
    
    acceleration = vectorized_forces / ARGON_MASS 
    return acceleration

# assigning random starting positions in the box 
positions = np.random.uniform(-BOX_SIZE / 2, BOX_SIZE / 2, size=(NUM_PARTICLES, 3))
# assigning random velocities according to the Maxwell-Boltzmann velocity distribution
velocities = np.random.normal(0, 1, (NUM_PARTICLES, 3)) * SCALING_FACTOR
velocities = normalize_velocities(velocities)
# updating accelerations according to the positions 
accelerations = calculate_acceleration(positions)

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
    
    global positions, velocities, accelerations
    
    if t % NORMALIZATION_INTERVAL == 0: # every x time steps, normalize velocity
        velocities = normalize_velocities(velocities)
    
    positions = positions + (velocities * DT) + (1/2) * accelerations * DT * DT

    wall_collisions = abs(positions) > BOX_SIZE / 2 # this is a (N, 3) of T and F
    if wall_collisions.any():
        multiplier = np.ones(wall_collisions.shape)
        multiplier[wall_collisions] = -1
        velocities = np.multiply(velocities, multiplier)

    velocities = velocities + (accelerations * DT)

    accelerations = calculate_acceleration(positions)

    scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])

ani = animation.FuncAnimation(fig, update, frames=int(NSTEPS), interval=50)

plt.show()