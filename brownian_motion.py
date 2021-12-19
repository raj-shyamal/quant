import numpy as np
import matplotlib.pyplot as plt


def simulate_geometric_random_walk(S0, T=2, N=1000, mu=0.1, sigma=0.05):

    dt = T/N
    t = np.linspace(0, T, N)

    # standard normal distribution N(0,1)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)
    X = (mu - 0.5*sigma**2)*t + sigma*W
    S = S0 * np.exp(X)

    return t, S


def plot_simulation(t, S):
    plt.plot(t, S)
    plt.xlabel('time (t)')
    plt.ylabel('Stock price S(t)')
    plt.title('Geometric Brownian Motion')
    plt.show()


if __name__ == '__main__':

    time, data = simulate_geometric_random_walk(100)
    plot_simulation(time, data)
