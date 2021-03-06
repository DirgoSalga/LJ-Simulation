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
    one_direction = np.linspace(box_size[0] / 2 * -1, box_size[0] / 2, n)
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
    :param velocities: <array> of 3D particle velocities
    :param m: <float> or N-D <array> with mass of the particles
    :return: <float> mean kinetic energy
    """
    total_kin = 0.5 * m * np.sum(velocities * velocities, axis=1).sum()
    mean_kin = total_kin / len(velocities)
    return mean_kin


def force(positions, box_size, rc=3.5):
    """
    Calculates the forces acting on the particles and the mean potential energy.
    :param positions: <array> position of the particles
    :param box_size: <float> size of the box in any direction
    :param rc: <float> value of the distance where a cut-off should be applied
    :return: Nx3 force array, <float> value of the potential energy.
    """
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
    f = np.matmul(ff, np.transpose(xr, (1, 0, 2))).diagonal().transpose()
    energy = 4 * r6i * (r6i - 1) - e_cut
    energy[mask_inf] = 0
    energy = energy.sum() / 2 / len(positions)
    return f, energy


def momentum(velocities, mass=1):
    """
    Calculates the total momentum of the system.
    :param velocities: <array> velocities of all particles
    :param mass: <float or array> mass of the particles
    :return: <array> 3D momenta of all particles.
    """
    p = np.sum(mass * velocities, axis=0)
    return p


def update_position(r, v, f_prev, dt, bs, m=1):
    """
    Updates the positions of all the particles according to the Velocity Verlet algorithm.
    :param r: Nx3 <array> containing all posistions
    :param v: Nx3 <array> containing all velocities
    :param f_prev: <array> previous forces
    :param dt: <float> size of a time step
    :param bs: <float> box size in any given direction
    :param m: <float or array> particle masses
    :return: Current positions <array>, velocities <array> and forces <array>.
    """
    a_prev = f_prev / m
    r += v * dt + 0.5 * a_prev * dt * dt
    f, e_pot = force(r, bs)
    a = f / m
    v += 0.5 * (a_prev + a) * dt
    return r, v, f, e_pot


def plot_check(kin, pot, p):
    fig = plt.figure()
    x = np.arange(0, len(kin), 1)
    plt.plot(x, kin, label="Kinetic energy")
    plt.plot(x, pot, label="Potential energy")
    plt.plot(x, kin + pot, label="Total energy")
    plt.plot(x, p, label="Momentum")
    plt.xlabel("Simulation step")
    plt.ylabel("Energy/Momentum (A.U.)")
    plt.legend(loc=0, fancybox=True)
    fig.savefig("check_plot.pdf")


def unify_xyz(directory, header, num_of_files, cleanup=True):
    """
    Unifies all the xyz files
    :param directory: <str> path to the directory containing the single xyz files
    :param header: <str> header of the xyz files
    :param num_of_files: <int> number of files to be unified
    :return: None
    """
    import os
    written_files = os.listdir(directory)
    written_files.pop(written_files.index(f"{directory}.xyz"))
    with open(f"{directory}/{directory}.xyz", "w") as f:
        for file in written_files:
            with open(f"{directory}/{file}", "r") as g:
                f.write(header)
                f.write(g.read().format(atom="H"))
            if cleanup:
                os.remove(f"{directory}/{file}")


def main():
    """
    This is the main routine. Change parameters within this function to change the parameters of the simulation.
    :return: Momentary mean kinetic energy, potential energy and momentum for every step.
    """
    particles_num = 125
    box_size = np.array([30, 30, 120])
    temp = 0.3
    time_step = 0.001
    integration_steps = 10000
    collision_frequency = 10
    write_out_step = 1000
    p_list = np.zeros(integration_steps)
    e_kin_list = np.zeros(integration_steps)
    e_pot_list = np.zeros(integration_steps)
    r = init_pos(box_size, particles_num, scale=(np.cbrt(particles_num) - 1) / box_size[0])  # Distance 1
    v, r_prev, e_kin = init_vel(r, temp, time_step)
    f, e_pot = force(r, box_size)
    i = 0
    np.savetxt("positions/positions_0.xyz", r, fmt="{atom} %.3f %.3f %.3f")
    np.savetxt("velocities/velocities_0.xyz", v, fmt="{atom} %.3f %.3f %.3f")
    while i < integration_steps:
        e_kin = kin_energy(v)
        r, v, f, e_pot = update_position(r, v, f, time_step, box_size)
        sigma = np.sqrt(temp)
        collision_mask = np.random.random_sample(particles_num) < collision_frequency * time_step
        v[collision_mask] = np.random.normal(0, sigma, (len(v[collision_mask]), 3))
        p = np.linalg.norm(momentum(v))
        p_list[i] = p
        e_kin_list[i] = e_kin
        e_pot_list[i] = e_pot
        i += 1
        if i % write_out_step == 0:
            np.savetxt("positions/positions_{0}.xyz".format(i), r, fmt="{atom} %.3f %.3f %.3f")
            np.savetxt("velocities/velocities_{0}.xyz".format(i), v, fmt="{atom} %.3f %.3f %.3f")
    unify_xyz("positions", f"{particles_num}\n\n", integration_steps)
    unify_xyz("velocities", f"{particles_num}\n\n", integration_steps)
    return e_kin_list, e_pot_list, p_list


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt

    start_time = time.time()
    e_kin, e_pot, p_total = main()
    end_time = time.time()
    print(end_time - start_time)
    plot_check(e_kin, e_pot, p_total)
