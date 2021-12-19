import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

RISK_FREE_RATE = 0.05
MONTHS = 12


class CAPM:

    def __init__(self, stocks, start_date, end_date):

        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

    def download_data(self):

        data = {}

        for stock in self.stocks:
            ticker = yf.download(stock, self.start_date, self.end_date)
            data[stock] = ticker['Adj Close']

        return pd.DataFrame(data)

    def initialize(self):

        stock_data = self.download_data()

        # use monthly returns instead of daily returns
        stock_data = stock_data.resample('M').last()

        self.data = pd.DataFrame({'s_adjclose': stock_data[self.stocks[0]],
                                  'm_adjclose': stock_data[self.stocks[1]]})

        # logarithmic monthly returns
        self.data[['s_returns', 'm_returns']] = np.log(
            self.data[['s_adjclose', 'm_adjclose']] / self.data[['s_adjclose', 'm_adjclose']].shift(1))

        self.data = self.data[1:]

    def calculate_beta(self):

        # covariance matrix: the diagonal elements are the variances
        # off diagonals are the covariances
        #matrix is symmetric

        covariance_matrix = np.cov(
            self.data['s_returns'], self.data['m_returns'])
        beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
        print('Beta from formula: ', beta)

    def regression(self):
        # using linear regression to fit a line to the data
        # [stock_returns, market_returns] - slope is the beta

        beta, alpha = np.polyfit(
            self.data['m_returns'], self.data['s_returns'], deg=1)
        print('Beta from regression: ', beta)

        # calculate the expected return

        expected_return = RISK_FREE_RATE + beta * \
            (self.data['m_returns'].mean()*MONTHS - RISK_FREE_RATE)
        print('Expected return: ', expected_return)

        self.plot_regression(alpha, beta)

    def plot_regression(self, alpha, beta):

        fig, axis = plt.subplots(1, figsize=(10, 6))
        axis.scatter(self.data['m_returns'],
                     self.data['s_returns'], label='Data points')
        axis.plot(self.data['m_returns'], alpha + beta *
                  self.data['m_returns'], label='Regression line')
        axis.set_xlabel('Market returns')
        axis.set_ylabel('Stock returns')
        axis.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':

    stocks = ['GOOG', '^GSPC']
    start_date = '2010-01-01'
    end_date = '2021-01-01'

    capm = CAPM(stocks, start_date, end_date)
    capm.initialize()
    capm.calculate_beta()
    capm.regression()
