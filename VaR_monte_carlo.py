import numpy as np
import yfinance as yf
import pandas as pd
import datetime


def download_data(stock, start, end):
    data = {}
    ticker = yf.download(stock, start, end)
    data['Adj Close'] = ticker['Adj Close']
    return pd.DataFrame(data)


class ValueAtRiskMonteCarlo:

    def __init__(self, S, mu, sigma, c, n, iterations):
        self.S = S
        self.mu = mu
        self.sigma = sigma
        self.c = c
        self.n = n
        self.iterations = iterations

    def simulation(self):

        random = np.random.normal(0, 1, [1, self.iterations])

        # Calculate the value of the portfolio at risk
        stock_price = self.S * \
            np.exp(self.n * (self.mu - 0.5 * self.sigma ** 2) +
                   self.sigma * np.sqrt(self.n) * random)

        stock_price = np.sort(stock_price)

        percentile = np.percentile(stock_price,  (1-self.c)*100)

        return self.S - percentile


if __name__ == '__main__':

    S = 1e6
    c = 0.95
    n = 1
    iterations = 100000

    start_date = datetime.datetime(2020, 1, 1)
    end_date = datetime.datetime(2021, 1, 1)

    apple = download_data('AAPL', start_date, end_date)
    apple['returns'] = apple['Adj Close'].pct_change()

    mu = np.mean(apple['returns'])
    sigma = np.std(apple['returns'])

    model = ValueAtRiskMonteCarlo(S, mu, sigma, c, n, iterations)

    print('value at risk (Monte Carlo): $', model.simulation())
