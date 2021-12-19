import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as sco


NUM_TRADING_DAYS = 252

NUM_PORFOLIOS = 10000

# stocks
stocks = ['AAPL', 'GOOG', 'TSLA', 'GE', 'MSFT', 'GS']

# historical data - start date and end date
start_date = '2010-01-01'
end_date = '2021-01-01'


def download_data():
    # name of the stock (key) - stock value (values)
    data = {}

    for stock in stocks:
        # closing price
        ticker = yf.Ticker(stock)
        data[stock] = ticker.history(start=start_date, end=end_date)['Close']

    return pd.DataFrame(data)


def show_data(data):
    # show data
    data.plot(figsize=(10, 6))
    plt.show()


def calculate_return(data):
    # Normalization
    log_return = np.log(data / data.shift(1))
    return log_return[1:]


def show_statistics(returns):
    # instead of daily metrics, we will use annual metrics
    print(returns.mean() * NUM_TRADING_DAYS)
    print(returns.cov() * NUM_TRADING_DAYS)


def show_mean_variance(returns, weights):

    portfolio_return = np.sum(
        returns.mean() * weights) * NUM_TRADING_DAYS  # mean
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(
        returns.cov() * NUM_TRADING_DAYS, weights)))  # standard deviation

    print('Expected Portfolio return:', portfolio_return)
    print('Expected Portfolio volatility:', portfolio_volatility)


def generate_portfolios(returns):

    portfolio_mean = []
    portfolio_risk = []
    portfolio_weights = []

    for i in range(NUM_PORFOLIOS):
        w = np.random.random(len(stocks))
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_mean.append(np.sum(returns.mean() * w) * NUM_TRADING_DAYS)
        portfolio_risk.append(
            np.sqrt(np.dot(w.T, np.dot(returns.cov() * NUM_TRADING_DAYS, w))))

    return np.array(portfolio_weights), np.array(portfolio_mean), np.array(portfolio_risk)


def show_portfolios(mean, risk):
    plt.figure(figsize=(10, 6))
    plt.scatter(risk, mean, c=mean / risk, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()


def statistics(weights, returns):
    portfolio_return = np.sum(
        returns.mean() * weights) * NUM_TRADING_DAYS  # mean
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(
        returns.cov() * NUM_TRADING_DAYS, weights)))  # standard deviation

    return np.array([portfolio_return, portfolio_volatility, portfolio_return / portfolio_volatility])


# scipy optimize module can find the minimum of a function
# maximum of f(x) is the minimum of -f(x)
def min_function_sharpe(weights, returns):
    return -statistics(weights, returns)[2]


# sum of weights = 1
# f(x) = 0 is the function to be minimized
def optimize_portfolio(weights, returns):
    # sum of weights = 1
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x)-1}

    # weights can be 1 at most: 1 when 100% of the portfolio is invested in a single stock
    bounds = tuple((0, 1) for x in range(len(stocks)))

    return sco.minimize(fun=min_function_sharpe, x0=weights[0], args=returns, method='SLSQP', bounds=bounds, constraints=constraints)


def print_optimized_portfolio(optimum, returns):
    print('Optimal Portfolio: ', optimum['x'].round(3))
    print('Expected return, volatility, Sharpe ratio:',
          statistics(optimum['x'].round(3), returns))


def show_optimal_portfolio(opt, rets, portfolio_return, portfolio_volatility):
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_volatility, portfolio_return,
                c=portfolio_return / portfolio_volatility, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.plot(statistics(opt['x'], rets)[1], statistics(
        opt['x'], rets)[0], 'g*', markersize=15.0)
    plt.show()


if __name__ == '__main__':

    data = download_data()
    # print(data)
    # show_data(data)
    # print(calculate_return(data))
    # show_statistics(calculate_return(data))

    log_return = calculate_return(data)
    weights, mean, risk = generate_portfolios(log_return)

    #show_portfolios(mean, risk)

    opt = optimize_portfolio(weights, log_return)
    print_optimized_portfolio(opt, log_return)
    show_optimal_portfolio(opt, log_return, mean, risk)
