import numpy as np
from numba import jit

class NotCubicNumber(Exception):
    pass



def init_pos(box_size, num_particles, scale=1.):
    """
    Initialisation of the position of N particles in a box with a given side length in 3 dimensions. The particles are
    positioned on a 3D-lattice with equidistant points.
    :param box_size:  <int> box size length
    :param num_particles: <int> number of particles (must be a perfect cube)
    :param scale: <float> scale of how to shrink the distance between the partciles. This always remain centred. The
    default is set to 1, which means the particles spread from one end of the box to the other.
    :return: r: <array> of 3D positions of each particle.
    """
    n = round(num_particles ** (1 / 3))
    if n ** 3 != num_particles:
        raise NotCubicNumber(
            "Number of particles N is not a perfect cube and cannot be used in this implementation.")
    one_direction = np.linspace(box_size / 2 * -1, box_size / 2, n)
    one_direction *= scale
    positions = []
    for i in one_direction:
        for j in one_direction:
            for k in one_direction:
                positions.append(np.array([i, j, k]))
    return np.array(positions)


def init_vel(num_of_particles, positions, temp, dt, m=1):
    """
    Initialise the velocities taking temperature into account. Also relative to the velocity of the center of mass.
    :param num_of_particles: <int> number of particles to initialise.
    :param positions: <array> of 3D position of the particles
    :param temp: <float> initial temperature of the system
    :param dt: <float> size of the time step
    :param m: <float> or <array> mass of the particles
    :return:
    """
    velocities = np.random.random(size=(num_of_particles, 3))
    total_p = m * velocities.sum(axis=0)
    total_kin = 0.5 * m * np.sum(velocities * velocities, axis=1).sum()
    vel_center_of_mass = total_p / num_of_particles / m
    mean_kin = total_kin / num_of_particles
    scale_factor = np.sqrt(3 / 2 * temp / mean_kin)
    velocities = scale_factor * (velocities - vel_center_of_mass)
    prev_positions = positions - velocities * dt
    return velocities, prev_positions, total_kin

@jit
def distance(positions):
    """
    Calculate the distances between all particles.
    :param positions: <array> with all 3D positions of the particles
    :return: <array> of shape (3, 1, num_of_particles)
    """
    p2 = positions[:, np.newaxis, :]
    return positions - p2

@jit
def lennard_jones_potential(r, sigma=1, epsilon=1):
    """
    Return the value of the potential between to given particles
    :param r: <float> distance between the particles
    :param sigma: <float> constant
    :param epsilon: <float> constant
    :return: <float> with potential value at this distance
    """
    r6i = 1 / r ** 6
    s6 = sigma ** 6
    return 4 * epsilon * r6i * s6 * (r6i * s6 - 1)

@jit
def kin_energy(velocities, m=1):
    """
    Calculates the mean kinetic energy of the system
    :param velocities: <arraya> of 3D particle velocities
    :param m: <float> or N-D <array> with mass of the particles
    :return: <float> mean kinetic energy
    """
    total_kin = 0.5 * m * np.sum(velocities * velocities, axis=1).sum()
    mean_kin = total_kin / len(velocities)
    return mean_kin


def force(positions, box_size, rc=3.5):
    e_cut = lennard_jones_potential(3.5)
    
    xr = distance(positions)
    
    xr = xr - box_size * np.round(xr / box_size)
    r2 = np.sum(xr * xr, axis=2)
    mask = r2 > rc ** 2
    r2[mask] = 0
    r2i = 1 / r2
    mask_inf = np.isinf(r2i)
    r2i[mask_inf] = 0
    r6i = r2i ** 3
    ff = 48 * r2i * r6i * (r6i - 0.5)
    f = np.inner(ff, xr.transpose((0, 2, 1))).diagonal().transpose()
    
    energy = np.sum(4 * r6i * (r6i - 1) - e_cut)/len(positions)#energy calculation
    
    
    
    return f,energy

def Momentum(velocities,Mass):
    momentum =(np.sum(Mass * velocities))
    return momentum
    
def main():
    pass


if __name__ == '__main__':
    pass
positions = np.random.random(size=(5, 3))
velocities = np.random.random(size=(5,3))
#%%
f,energy=force(positions,30,3.5)
Momentum = Momentum(velocities,1)
