import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=252*3, freq='D')

stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
n_stocks = len(stocks)

price_data = {}
for stock in stocks:
    returns = np.random.normal(0.0008, 0.02, len(dates))
    prices = [100]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    price_data[stock] = prices

df = pd.DataFrame(price_data, index=dates)
print("Sample of stock price data:")
print(df.head())

returns = df.pct_change().dropna()
print("\nSample of daily returns:")
print(returns.head())

annual_returns = returns.mean() * 252
print("\nExpected Annual Returns:")
for stock, ret in annual_returns.items():
    print(f"{stock}: {ret:.2%}")

cov_matrix = returns.cov() * 252
print("\nAnnualized Covariance Matrix:")
print(cov_matrix)

def portfolio_stats(weights, mean_returns, cov_matrix):
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    p_return, p_volatility = portfolio_stats(weights, mean_returns, cov_matrix)
    return -(p_return - risk_free_rate) / p_volatility

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_stats(weights, mean_returns, cov_matrix)[1]

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(n_stocks))
initial_guess = np.array(n_stocks * [1. / n_stocks])

max_sharpe_result = minimize(negative_sharpe_ratio, 
                           initial_guess,
                           args=(annual_returns, cov_matrix),
                           method='SLSQP',
                           bounds=bounds,
                           constraints=constraints)

max_sharpe_weights = max_sharpe_result.x
max_sharpe_return, max_sharpe_volatility = portfolio_stats(max_sharpe_weights, annual_returns, cov_matrix)

print("\n" + "="*50)
print("MAXIMUM SHARPE RATIO PORTFOLIO")
print("="*50)
print(f"Expected Return: {max_sharpe_return:.2%}")
print(f"Volatility: {max_sharpe_volatility:.2%}")
print(f"Sharpe Ratio: {(max_sharpe_return - 0.02) / max_sharpe_volatility:.3f}")
print("\nOptimal Weights:")
for stock, weight in zip(stocks, max_sharpe_weights):
    print(f"{stock}: {weight:.2%}")

min_vol_result = minimize(portfolio_volatility,
                         initial_guess,
                         args=(annual_returns, cov_matrix),
                         method='SLSQP',
                         bounds=bounds,
                         constraints=constraints)

min_vol_weights = min_vol_result.x
min_vol_return, min_vol_volatility = portfolio_stats(min_vol_weights, annual_returns, cov_matrix)

print("\n" + "="*50)
print("MINIMUM VARIANCE PORTFOLIO")
print("="*50)
print(f"Expected Return: {min_vol_return:.2%}")
print(f"Volatility: {min_vol_volatility:.2%}")
print("\nOptimal Weights:")
for stock, weight in zip(stocks, min_vol_weights):
    print(f"{stock}: {weight:.2%}")

def efficient_frontier(mean_returns, cov_matrix, num_portfolios=100):
    min_ret = min_vol_return
    max_ret = annual_returns.max()
    target_returns = np.linspace(min_ret, max_ret, num_portfolios)
    
    efficient_portfolios = []
    
    for target in target_returns:
        target_constraint = {'type': 'eq', 'fun': lambda x, target=target: 
                           np.sum(annual_returns * x) - target}
        
        all_constraints = [constraints, target_constraint]
        
        result = minimize(portfolio_volatility,
                         initial_guess,
                         args=(annual_returns, cov_matrix),
                         method='SLSQP',
                         bounds=bounds,
                         constraints=all_constraints)
        
        if result.success:
            ret, vol = portfolio_stats(result.x, annual_returns, cov_matrix)
            efficient_portfolios.append([ret, vol])
    
    return np.array(efficient_portfolios)

print("\nGenerating Efficient Frontier...")
efficient_portfolios = efficient_frontier(annual_returns, cov_matrix, 50)

plt.figure(figsize=(12, 8))
plt.scatter(efficient_portfolios[:, 1] * 100, 
           efficient_portfolios[:, 0] * 100, 
           c='blue', alpha=0.6, label='Efficient Frontier')

individual_volatility = np.sqrt(np.diag(cov_matrix)) * 100
individual_returns = annual_returns * 100

plt.scatter(individual_volatility, individual_returns, 
           c='red', s=100, alpha=0.8, label='Individual Stocks')

for i, stock in enumerate(stocks):
    plt.annotate(stock, 
                (individual_volatility[i], individual_returns[i]),
                xytext=(5, 5), textcoords='offset points')

plt.scatter(max_sharpe_volatility * 100, max_sharpe_return * 100, 
           c='green', s=200, marker='*', label='Max Sharpe Ratio')

plt.scatter(min_vol_volatility * 100, min_vol_return * 100, 
           c='orange', s=200, marker='*', label='Min Variance')

plt.xlabel('Volatility (%)', fontsize=12)
plt.ylabel('Expected Return (%)', fontsize=12)
plt.title('Modern Portfolio Theory - Efficient Frontier', fontsize=16, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("RISK-RETURN ANALYSIS")
print("="*50)

equal_weights = np.array([1/n_stocks] * n_stocks)
equal_return, equal_volatility = portfolio_stats(equal_weights, annual_returns, cov_matrix)

print(f"\nEqual-Weighted Portfolio:")
print(f"Expected Return: {equal_return:.2%}")
print(f"Volatility: {equal_volatility:.2%}")
print(f"Sharpe Ratio: {(equal_return - 0.02) / equal_volatility:.3f}")

print(f"\nPORTFOLIO COMPARISON:")
print(f"{'Portfolio':<20} {'Return':<10} {'Risk':<10} {'Sharpe':<10}")
print("-" * 50)
print(f"{'Equal Weight':<20} {equal_return:<10.2%} {equal_volatility:<10.2%} {(equal_return - 0.02) / equal_volatility:<10.3f}")
print(f"{'Min Variance':<20} {min_vol_return:<10.2%} {min_vol_volatility:<10.2%} {(min_vol_return - 0.02) / min_vol_volatility:<10.3f}")
print(f"{'Max Sharpe':<20} {max_sharpe_return:<10.2%} {max_sharpe_volatility:<10.2%} {(max_sharpe_return - 0.02) / max_sharpe_volatility:<10.3f}")

print("\n" + "="*50)
print("ANALYSIS COMPLETE!")
print("="*50)
