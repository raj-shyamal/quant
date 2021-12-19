import numpy as np
import matplotlib.pyplot as plt


def vasicek(r0, kappa, theta, sigma, T=1, N=1000):

    dt = T/float(N)
    t = np.linspace(0, T, N+1)
    rates = [r0]

    for _ in range(N):
        dr = kappa*(theta-rates[-1])*dt + sigma*np.sqrt(dt)*np.random.normal()
        rates.append(rates[-1] + dr)

    return t, rates


def plot(t, r):
    plt.plot(t, r)
    plt.xlabel('time (t)')
    plt.ylabel('interest rates r(t)')
    plt.title('Vasicek model')
    plt.show()


if __name__ == '__main__':

    time, data = vasicek(1.3, 0.9, 1.4, 0.05)
    plot(time, data)
