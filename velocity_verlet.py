import numpy as np
from random import random
import matplotlib.pyplot as plt


class NotCubicNumber(Exception):
    pass


class Particle:
    number_of_particles = 0
    potential_energy = 0
    kinetic_energy = 0

    def __init__(self, pos, mass):
        self.pos = pos
        self.m = mass
        self.prev_pos = None
        self.v = None
        self.f = None
        self.a = None
        Particle.number_of_particles += 1

    def half_step_velocity(self, dt):
        """
        Updates the velocity vector of a particle after half a time interval dt according to the velocity Verlet algorithm.
        :param dt: <float> time interval
        :return: None
        """
        v_half_step = self.v + 0.5 * self.a * dt
        self.v = v_half_step

    def acceleration(self):
        """
        Updates the acceleration attribute of the particle after the force has been calculated
        """
        self.a = self.f / self.m

    def update_position(self, dt):
        """
        Updates the stored position using the Verlet method. Here the boundary condition is considered.
        :param dt: <float> time interval to consider when applying Verlet method
        """
        self.pos += self.v * dt

    def xyz_line(self, velocities=False):
        """
        Function produces a string with an index to be filled and the three spatial coordinates, like XYZ format.
        :return: <str> to be formated with line break.
        """
        if velocities:
            return f"{{atom}} {self.v[0]:.3f} {self.v[1]:.3f} {self.v[2]:.3f}\n"
        else:
            return f"{{atom}} {self.pos[0]:.3f} {self.pos[1]:.3f} {self.pos[2]:.3f}\n"

    def __str__(self):
        return f"Previous position: {self.prev_pos}\nCurrent position: {self.pos}\nCurrent acceleration: {self.a}\nCurrent velocity: {self.v}"


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


def init_vel(p_list, temp, dt):
    """
    Iinitialisation of the starting velocity of N particles.
    :param p_list: <array> of particle instances
    :param temp: <float> temperature
    :param dt: <float> time element
    :return: v: <array> of 3D initial velocities of each particle.
    """
    N = len(p_list)
    total_p = 0
    total_kin = 0
    total_m = np.array([i.m for i in p_list]).sum()
    for particle in p_list:
        vel = np.array([random() - 0.5 for _ in range(3)])
        total_p += vel * particle.m
        total_kin += 0.5 * particle.m * np.dot(vel, vel)
        particle.v = vel
    vel_center_of_mass = total_p / total_m
    mean_kin = total_kin / N
    scale_factor = np.sqrt(3 / 2 * temp / mean_kin)
    for particle in p_list:
        particle.v = scale_factor * (particle.v - vel_center_of_mass)
        particle.prev_pos = particle.pos - particle.v * dt
    Particle.kinetic_energy = mean_kin


def kinetic_energy(p_list):
    """
    Updates the value of the kinetic energy of the system
    :param p_list: <array> of particle instances
    """
    sumv = 0
    sumv2 = 0
    for particle in p_list:
        sumv += particle.v
        sumv2 += np.dot(particle.v, particle.v)
    # temp = sumv2 / (3 * len(p_list))
    energy = 0.5 * sumv2 / len(p_list)
    Particle.kinetic_energy = energy


def force(p_list, box_size, rc=3.5):
    """
    Updates the forces for all particles, considering all interactions using Leonard-Jones potential
    :param p_list: <array> of particle instances
    :param box_size: <float> length of the box
    :param rc: <float> cut-off radius at which the potential is negligible
    """
    energy = 0
    f = np.zeros((len(p_list), 3))
    n = len(p_list)
    e_cut = lennard_jones_potential(rc)
    for i in range(n - 1):
        for j in range(i + 1, n):
            xr = p_list[i].pos - p_list[j].pos
            xr = xr - box_size * np.round(xr / box_size)
            r2 = np.dot(xr, xr)
            if r2 < rc ** 2:
                r2i = 1 / r2
                r6i = r2i ** 3
                ff = 48 * r2i * r6i * (r6i - 0.5)
                f[i] += ff * xr
                f[j] -= ff * xr
                energy += 4 * r6i * (r6i - 1) - e_cut
    for i in range(len(p_list)):
        p_list[i].f = f[i]
    Particle.potential_energy = energy / n


if __name__ == "__main__":
    m = 1
    time_step = 0.001
    positions = init_pos(30, 125, scale=2 / 15)
    particles = [Particle(i, 1) for i in positions]
    init_vel(particles, 0.3, time_step)
    force(particles, 30)
    for p in particles:
        p.acceleration()
    pot = []
    kin = []
    momentum = []
    steps = range(10000)

    with open("position.xyz", "w") as position_file:
        with open("velocities.xyz", "w") as velocity_file:
            for l in steps:
                position_file.write("{0}\n\n".format(
                    Particle.number_of_particles))
                velocity_file.write("{0}\n\n".format(
                    Particle.number_of_particles))
                for p in particles:
                    position_file.write(p.xyz_line().format(atom="H"))
                    velocity_file.write(p.xyz_line(
                        velocities=True).format(atom="H"))
                    # This order is important! 1. v, 2. r
                    p.half_step_velocity(time_step)
                    p.update_position(time_step)
                force(particles, 30)
                pot.append(Particle.potential_energy)
                resulting_momentum_vector = np.array([0., 0., 0.])
                for p in particles:
                    p.acceleration()
                    p.half_step_velocity(time_step)
                    resulting_momentum_vector += p.v
                kinetic_energy(particles)
                kin.append(Particle.kinetic_energy)
                momentum.append(np.linalg.norm(resulting_momentum_vector))
    fig = plt.figure()
    plt.plot(steps, pot, label="Potential energy")
    plt.plot(steps, kin, label="Kinetic energy")
    plt.plot(steps, np.array(pot) + np.array(kin), label="Total energy")
    plt.plot(steps, momentum, label="Total momentum")
    plt.xlabel("Steps $\Delta t$")
    plt.ylabel("Energy / Momentum (a.u.)")
    plt.legend(loc=0, fancybox=True)
    fig.savefig("velocity_verlet_results.pdf")
