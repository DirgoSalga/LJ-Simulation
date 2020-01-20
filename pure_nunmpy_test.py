import numpy as np


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


def init_vel(positions, temp, dt, m=1):
    """
    Initialise the velocities taking temperature into account. Also relative to the velocity of the center of mass.
    :param positions: <array> of 3D position of the particles
    :param temp: <float> initial temperature of the system
    :param dt: <float> size of the time step
    :param m: <float> or <array> mass of the particles
    :return: <array> velocities, <array> previous positions and kinetic_energy <float>
    """
    num_of_particles = len(positions)
    velocities = np.random.random(size=(num_of_particles, 3))
    total_p = m * velocities.sum(axis=0)
    total_kin = 0.5 * m * np.sum(velocities * velocities, axis=1).sum()
    vel_center_of_mass = total_p / num_of_particles / m
    mean_kin = total_kin / num_of_particles
    scale_factor = np.sqrt(3 / 2 * temp / mean_kin)
    velocities = scale_factor * (velocities - vel_center_of_mass)
    prev_positions = positions - velocities * dt
    return velocities, prev_positions, total_kin


def distance(positions):
    """
    Calculate the distances between all particles.
    :param positions: <array> with all 3D positions of the particles
    :return: <array> of shape (3, 1, num_of_particles)
    """
    p2 = positions[:, np.newaxis, :]
    return positions - p2


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
    energy = 4 * r6i * (r6i - 1) - e_cut
    energy[mask_inf] = 0
    energy = energy.sum() / len(positions)
    return f, energy


def momentum(velocities, mass=1):
    p = np.sum(mass * velocities, axis=1)
    return p


def update_position(r, v, f_prev, dt, bs, method="velocity_verlet", m=1):
    """

    :param r: Nx3 <array> containing all posistions
    :param v: Nx3 <array> containing all velocities
    :param dt: <float> size of a time step
    :param bs: <float> box size in any given direction
    :param method: <string> selects which algorithm to use to update position. options: 'velocity_verlet', 'leap_frog'
    :param m: <float> or <array> of particle masses
    :return: Current values for positions, velocities and forces correspondingly
    """
    a_prev = f_prev / m
    if method == "velocity_verlet":
        r += v * dt + 0.5 * a_prev * dt * dt
        f, e_pot = force(r, bs)
        a = f / m
        v += 0.5 * (a_prev + a) * dt
        return r, v, f, e_pot
    elif method == "leap_frog":
        pass


def xyz_line(coordinates):
    """
    Function produces a string with an index to be filled and the three spatial coordinates, like XYZ format.
    :return: <str> to be formated with line break.
    """
    return f"{{atom}} {coordinates[0]:.3f} {coordinates[1]:.3f} {coordinates[2]:.3f}\n"


def main():
    """
    This is the main routine. Change parameters within this function to change the parameters of the simulation.
    :return: None
    """
    particles_num = 125
    box_size = 30
    temp = 3
    time_step = 0.0001
    integration_steps = 10000
    r = init_pos(box_size, particles_num, scale=(np.cbrt(particles_num) - 1) / box_size)
    v, r_prev, e_kin = init_vel(r, temp, time_step)
    f, e_pot = force(r, box_size)
    i = 0
    while i < integration_steps:
        r, v, f, e_pot = update_position(r, v, f, time_step, box_size)
        e_kin = kin_energy(v)
        p = momentum(v)
        i += 1


if __name__ == '__main__':
    import time

    start_time = time.time()
    main()
    end_time = time.time()
    print(end_time - start_time)
