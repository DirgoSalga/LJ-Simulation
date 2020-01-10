from velocity_verlet import *

if __name__ == '__main__':
    m = 1
    nu = 10
    # temperature_init = 0.1
    temperature_init = 0.6
    time_step = 0.001
    box_length = 30
    positions = init_pos(30, 125, scale=2 / 15)
    particles = [Particle(i, 1) for i in positions]
    init_vel(particles, temperature_init, time_step)
    force(particles, box_length)
    with open("thermostat_positions.xyz", "w") as file:
        for p in particles:
            p.acceleration()
        pot = []
        temperature_list = []
        time_steps = range(10000)
        for _ in time_steps:
            file.write(f"{Particle.number_of_particles}\n\n")
            for p in particles:
                file.write(p.xyz_line().format(atom="H"))
                p.half_step_velocity(time_step)
                p.update_position(time_step)
            force(particles, box_length)
            pot.append(Particle.potential_energy)
            temp = 0
            for p in particles:
                p.acceleration()
                p.half_step_velocity(time_step)
                temp += np.dot(p.v, p.v)
            temp /= (3 * len(particles))
            sigma = np.sqrt(temperature_init)  # temperature of the heat bath
            temperature_list.append(temp)
            for p in particles:
                if np.random.random_sample() < nu * time_step:
                    p.v = np.random.normal(0, sigma, 3)
    fig = plt.figure()
    plt.plot(time_steps, pot, label='Potential energy')
    plt.plot(time_steps, temperature_list, label='Temperature')
    plt.xlabel("Steps $\Delta t$")
    plt.ylabel("Energy / Temperature (a.u.)")
    plt.legend(loc=0, fancybox=True)
    # fig.savefig("andersen_thermostat_results_b.pdf")
    fig.savefig("andersen_thermostat_results_c.pdf")
