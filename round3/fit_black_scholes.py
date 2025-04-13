import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

# -------------------------------
# Load and Clean Data
# -------------------------------
training_data = pd.read_csv('sample_data/round-3-island-data-bottle/prices_round_3_day_0.csv', delimiter=';')
training_data = training_data.fillna(0)

# -------------------------------
# Identify Underlying Price
# -------------------------------
underlying_row = training_data[training_data['product'] == 'VOLCANIC_ROCK']
S = underlying_row['mid_price'].values[0]

# -------------------------------
# Filter Option Rows
# -------------------------------
option_rows = training_data[training_data['product'].str.startswith('VOLCANIC_ROCK_VOUCHER')]
option_rows = option_rows.copy()

# Extract strike price from product name
option_rows['strike'] = option_rows['product'].str.extract(r'(\d+)').astype(float)
option_rows['market_price'] = option_rows['mid_price']
option_rows = option_rows[['strike', 'market_price']].dropna()

# -------------------------------
# Time to Maturity
# -------------------------------
T = 1.0  # Assuming 1 year, adjust if needed

# -------------------------------
# Black-Scholes Function
# -------------------------------
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

# -------------------------------
# Sigma Curve: Quadratic
# -------------------------------
def sigma_model(K, a, b, c):
    return np.maximum(0.01, a*K**2 + b*K + c)

# -------------------------------
# Loss Function
# -------------------------------
def loss(params):
    r, a, b, c = params
    loss = 0.0
    for _, row in option_rows.iterrows():
        K = row['strike']
        market_price = row['market_price']
        sigma = sigma_model(K, a, b, c)
        model_price = black_scholes_call(S, K, T, r, sigma)
        loss += (model_price - market_price) ** 2
    return loss

# -------------------------------
# Optimize Parameters
# -------------------------------
initial_guess = [0.01, 1e-9, -1e-5, 0.3]  # r, a, b, c
bounds = [(0, 0.1), (-1e-6, 1e-6), (-1e-3, 1e-3), (0.01, 1.0)]

res = minimize(loss, initial_guess, bounds=bounds)

r_opt, a_opt, b_opt, c_opt = res.x
print(f" Calibrated r: {r_opt:.5f}")
print(f" Sigma(K) = {a_opt:.2e} * K^2 + {b_opt:.2e} * K + {c_opt:.4f}")
