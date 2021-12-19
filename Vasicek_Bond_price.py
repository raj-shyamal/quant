import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


NUM_OF_SIMULATIONS = 1000
NUM_OF_POINTS = 200  # number of points in a single r(t) process


def monte_carlo_simulation(x, r0, kappa, theta, sigma, T=1):

    dt = T/float(NUM_OF_POINTS)
    result = []

    for _ in range(NUM_OF_SIMULATIONS):
        rates = [r0]

        for _ in range(NUM_OF_POINTS):
            dr = kappa*(theta-rates[-1])*dt + sigma * \
                np.sqrt(dt)*np.random.normal()
            rates.append(rates[-1]+dr)

        result.append(rates)

    simulation = pd.DataFrame(result)
    simulation = simulation.T

    # calculate the integral of the r(t) based on the simulated paths
    integral = simulation.sum()*dt

    # present value integral
    present_integral = np.exp(-integral)

    # mean because we are simulating many paths
    bond_price = x*np.mean(present_integral)

    print('Bond price based on Monte-Carlo simulation: $%.2f' % bond_price)


if __name__ == '__main__':

    monte_carlo_simulation(1000, 0.1, 0.3, 0.3, 0.03)
