import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import datetime


def download_data(stock, start_date, end_date):
    data = {}
    ticker = yf.download(stock, start_date, end_date)
    data[stock] = ticker['Adj Close']
    return pd.DataFrame(data)


def calculate_VaR(position, c, mu, sigma):
    var = position * (mu - sigma * norm.ppf(1-c))  # VaR tomorrow (n=1)
    return var


def calculate_VaR_n(position, c, mu, sigma, n):
    var = position * (mu * n - sigma * np.sqrt(n) *
                      norm.ppf(1-c))  # VaR n days from now
    return var


if __name__ == '__main__':

    start = datetime.datetime(2020, 1, 1)
    end = datetime.datetime(2021, 1, 1)

    stock_data = download_data('AAPL', start, end)
    stock_data['returns'] = np.log(stock_data / stock_data.shift(1))
    stock_data = stock_data[1:]

    S = 1e6                                            # Initial position
    c = 0.95                                           # Confidence level 95%

    mu = np.mean(stock_data['returns'])
    sigma = np.std(stock_data['returns'])

    print('Value at risk : $', calculate_VaR_n(S, c, mu, sigma, 1))
