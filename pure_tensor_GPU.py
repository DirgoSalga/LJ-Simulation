import numpy as np
import torch
import matplotlib.pyplot as plt


class NotCubicNumber(Exception):
    pass


def init_pos(box_size, num_particles, scale=1.0):
    """
    Initialisation of the position of N particles in a box with a given side length in 3 dimensions. The particles are
    positioned on a 3D-lattice with equidistant points.
    :param box_size:  <int> box size length
    :param num_particles: <int> number of particles (must be a perfect cube)
    :param scale: <float> scale of how to shrink the distance between the partciles. This always remain centred. The
    default is set to 1, which means the particles spread from one end of the box to the other.
    :return: r: <array> of 3D positions of each particle.
    """
    with torch.cuda.device(device):
        n = round(num_particles ** (1 / 3))
        if n ** 3 != num_particles:
            raise NotCubicNumber(
                "Number of particles N is not a perfect cube and cannot be used in this implementation."
            )
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
    velocities = torch.randn(num_of_particles, 3).cuda()
    total_p = m * velocities.sum(axis=0)
    total_kin = 0.5 * m * torch.tensordot(velocities, velocities)
    vel_center_of_mass = total_p / num_of_particles / m
    mean_kin = total_kin / num_of_particles
    scale_factor = (3 / 2 * temp / mean_kin) ** (1 / 2)
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
    total_kin = 0.5 * m * torch.tensordot(velocities, velocities)
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
    positions = positions.cuda()
    xr = distance(positions)
    xr = xr - box_size * torch.round(xr / box_size)
    r2 = torch.sum(xr * xr, axis=2)
    mask = r2 > rc ** 2
    r2[mask] = 0
    r2i = 1 / r2
    mask_inf = torch.isinf(r2i)
    r2i[mask_inf] = 0
    r6i = r2i ** 3
    ff = 48 * r2i * r6i * (r6i - 0.5)
    f = (torch.matmul(ff, xr.permute(1, 0, 2)).diagonal()).permute(1, 0)
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

    velocities = velocities.cuda()
    p = torch.sum(mass * velocities, axis=0)
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
    with torch.cuda.device(device):
        a_prev = f_prev / m
        r += v * dt + 0.5 * a_prev * dt * dt
        f, e_pot = force(r, bs)
        f = f.cuda()
        e_pot = e_pot.cuda()
        a = f / m
        v += 0.5 * (a_prev + a) * dt
        return r, v, f, e_pot


def plot_check(kin, pot, p):
    fig = plt.figure()
    x = np.arange(0, len(kin), 1)
    plt.plot(x, kin, label="Kinetic energy")
    plt.plot(x, pot, label="Potential energy")
    plt.plot(x, kin + pot, label="Total energy")
    # plt.plot(x, p, label="Momentum")
    plt.xlabel("Simulation step")
    plt.ylabel("Energy/Momentum (A.U.)")
    plt.legend(loc=0, fancybox=True)
    fig.savefig("check_plot.pdf")


def unify_xyz(directory, header, integration_steps, write_out_step, cleanup=True):
    """
    Writes multiple xyz files into a single file
    :param directory: <str> path to the directory
    :param header: <str> Header of the xyz file
    :param cleanup: <boolean> set True if you would like to delete files after unification
    :return: None
    """
    import os

    written_files = os.listdir(directory)
    written_files.sort()
    with open(f"{directory}/{directory}.xyz", "w") as f:
        for i in range(0, integration_steps + 1, write_out_step):
            with open(f"{directory}/{directory}_{i}.xyz", "r") as g:
                f.write(header)
                f.write(g.read().format(atom="H"))
            if cleanup:
                os.remove(f"{directory}/{directory}_{i}.xyz")


def unify_pt(directory, cleanup=True):
    """
    Unify binary torch files for later numpy unification
    :param directory: <str> path to directory
    :param cleanup: <boolean> set True if you would like to delete pt files after unification
    :return: None
    """
    import os

    written_files = os.listdir(directory)
    if f"{directory}.xyz" in written_files:
        written_files.pop(written_files.index(f"{directory}.xyz"))
    for file in written_files:
        load_str = f"{directory}/{file}"
        x = torch.load(load_str)
        base = file.split(".")[0]
        np.savetxt(f"{directory}/{base}.xyz", x.cpu(), fmt="{atom} %.3f %.3f %.3f")
        if cleanup:
            os.remove(load_str)


def main(particles_num, temp, time_step, integration_steps):
    """
    Main routine of the LJ simulation
    :param particles_num: <int> number of particles in the simulation
    :param temp: <float> starting temperature of the simulation
    :param time_step: <float> size of the time step
    :param integration_steps: <int> number of simulation steps
    :return: <array> Momentary mean kinetic energy, <array> potential energy and <array> momentum for every step.
    """
    box_size = np.array([10, 10, 20])
    collision_frequency = 10
    write_out_step = 1000
    p_list = np.zeros(integration_steps)
    e_kin_list = np.zeros(integration_steps)
    e_pot_list = np.zeros(integration_steps)
    r = init_pos(box_size, particles_num, scale=(np.cbrt(particles_num) - 1) / box_size[0])
    r = torch.tensor(r, dtype=torch.float).cuda()
    v, r_prev, e_kin = init_vel(r, temp, time_step)
    box_size = torch.tensor(box_size, dtype=torch.float).cuda()
    f, e_pot = force(r, box_size)
    i = 0
    np.savetxt("positions/positions_0.xyz", r.cpu(), fmt="{atom} %.3f %.3f %.3f")
    # np.savetxt("positions/velocities_0.xyz", v.cpu(), fmt="{atom} %.3f %.3f %.3f")
    # torch.save(r, "positions/positions_0.pt")
    # torch.save(v, "velocities/velocities_0.pt")
    with torch.cuda.device(device):
        while i < integration_steps:
            e_kin = kin_energy(v)
            r, v, f, e_pot = update_position(r, v, f, time_step, box_size)
            if i < (integration_steps / 2):
                sigma = np.sqrt(temp)
                collision_mask = (
                        np.random.random_sample(particles_num)
                        < collision_frequency * time_step
                )
                v[collision_mask] = torch.normal(
                    0, sigma, (len(v[collision_mask]), 3)
                ).cuda()
            p = torch.norm(momentum(v))
            p_list[i] = p
            e_kin_list[i] = e_kin
            e_pot_list[i] = e_pot
            i += 1
            if i % write_out_step == 0:
                np.savetxt(
                    "positions/positions_{0}.xyz".format(i), r.cpu(), fmt="{atom} %.3f %.3f %.3f")
                # np.savetxt("velocities/velocities_{0}.xyz".format(i),v.cpu(),fmt="{atom} %.3f %.3f %.3f")
                # torch.save(r, "positions/positions_{0}.pt".format(i))
                # torch.save(v, "velocities/positions_{0}.pt".format(i))
        # unify_pt("positions")
        # unify_pt("velocities")
        unify_xyz("positions", f"{particles_num}\n\n", integration_steps, write_out_step)
        # unify_xyz("velocities", f"{particles_num}\n\n")

    return e_kin_list, e_pot_list, p_list


if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser(description="Runs an LJ-simulation in a box.")
    parser.add_argument('temp', metavar='Temperature', type=float, help='Initial temperature of the simulation.')
    parser.add_argument('-p', '--particles_num', metavar="Number of particles", type=int, default=1000,
                        help='Number of particles in the simulation. Defaul 1000')
    parser.add_argument('-t', '--time_step', metavar="Time step", type=float, default=0.005,
                        help='Size of the simulation time step.')
    parser.add_argument('-s', '--integration_steps', metavar='Integration steps', type=int, default=1000000,
                        help='Number of steps in the simulation.')
    parser.add_argument('-d', '--device', metavar='Device ID', default=0, type=int,
                        help="CUDA device number. Default 0.")
    args = parser.parse_args()  # gets the arguments from terminal now!!!
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    print(args)
    start_time = time.time()
    e_kin, e_pot, p_total = main(args.particles_num, args.temp, args.time_step, args.integration_steps)
    end_time = time.time()
    np.savetxt("E_kin.txt", e_kin)
    np.savetxt("E_pot.txt", e_pot)
    np.savetxt("Momentum.txt", p_total)

    print(end_time - start_time)
    plot_check(e_kin, e_pot, p_total)
