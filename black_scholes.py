from scipy import stats
from numpy import log, exp, sqrt


def call_option_price(S, E, T, rf, sigma):

    d1 = (log(S/E) + (rf + sigma*sigma/2.0)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    print("d1 = ", d1)
    print("d2 = ", d2)

    # use N(x) to calculate the price of the call option
    return S*stats.norm.cdf(d1) - E*exp(-rf*T)*stats.norm.cdf(d2)


def put_option_price(S, E, T, rf, sigma):

    d1 = (log(S/E) + (rf + sigma*sigma/2.0)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    print("d1 = ", d1)
    print("d2 = ", d2)

    # use N(x) to calculate the price of the call option
    return -S*stats.norm.cdf(-d1) + E*exp(-rf*T)*stats.norm.cdf(-d2)


if __name__ == "__main__":

    S0 = 100.0      # initial stock price
    E = 100.0       # strike price
    T = 1.0         # time to maturity
    rf = 0.05       # risk-free rate
    sigma = 0.2     # volatility

    print("Call option price = ", call_option_price(S0, E, T, rf, sigma))
    print("Put option price = ", put_option_price(S0, E, T, rf, sigma))
