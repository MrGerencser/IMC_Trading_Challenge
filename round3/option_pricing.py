import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


# --------------------------
# 1. Plotting Function
# --------------------------
def plot_iv_vs_price_differences(iv_df, voucher_type):
    iv_df['intrinsic'] = np.maximum(iv_df['underlying_price'] - iv_df['strike'], 0)
    iv_df['diff_market_strike'] = iv_df['underlying_price'] - iv_df['strike']

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Voucher Type: {voucher_type}', fontsize=16)

    axs[0].scatter(iv_df['intrinsic'], iv_df['implied_vol'], color='blue', alpha=0.5)
    axs[0].set_title('Intrinsic Value vs. Implied Volatility')
    axs[0].set_xlabel('Intrinsic Value (max(S - K, 0))')
    axs[0].set_ylabel('Implied Volatility')
    axs[0].grid(True)

    axs[1].scatter(iv_df['diff_market_strike'], iv_df['implied_vol'], color='green', alpha=0.5)
    axs[1].set_title('Market Price - Strike vs. Implied Volatility')
    axs[1].set_xlabel('Market Price - Strike')
    axs[1].set_ylabel('Implied Volatility')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
# --------------------------
def cdf_standard_normal(x):
     """Approximate CDF of the standard normal distribution."""
     return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    
def vega(S, K, r, T, sigma):
    """Partial derivative of the call price wrt sigma (aka Vega)."""
    d1 = (math.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return S * math.sqrt(T) * (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * d1**2)


def black_scholes_call_price(S, K, r, T, sigma):
    """
    Computes the Black–Scholes price for a call option.
    S: Underlying price
    K: Strike price
    r: Risk-free rate
    T: Time to expiration (in years)
    sigma: volatility (annualized)
    """
    if T <= 0:
        # If time to expiry is 0, the option's value is max(S-K, 0).
        return max(S - K, 0)
    # d1 and d2
    d1 = (math.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    # Call price
    call_val = (S * cdf_standard_normal(d1)) - (K * math.exp(-r * T) * cdf_standard_normal(d2))
    return call_val

def implied_vol_call_price(S, K, r, T, market_price, initial_guess=0.2, tol=1e-6, max_iterations=1000):
    """
    Numerically find implied volatility via Newton–Raphson.
    S: Underlying price
    K: Strike price
    r: Risk-free rate
    T: Time to expiration (in years)
    market_price: the current market premium of the option
    initial_guess: starting volatility guess
    tol: tolerance for the final result
    max_iterations: maximum number of iterations
    """
    # Edge case: Deep in-the-money with very low time value
    intrinsic_value = max(S - K, 0)
    if T < 1e-5 and abs(market_price - intrinsic_value) < tol:
        return 1e-5  # minimal volatility, since price is nearly intrinsic

    sigma = initial_guess
    for i in range(max_iterations):
        price = black_scholes_call_price(S, K, r, T, sigma)
        diff = price - market_price  # how far off we are
        if abs(diff) < tol:
            return sigma  # found a good enough solution
        v = vega(S, K, r, T, sigma)
        if v < 1e-8:
            # If vega is extremely small, we risk dividing by zero or huge jumps.
            break
        # Newton step
        sigma = sigma - diff / v
        # keep sigma positive
        if sigma < 0:
            sigma = 1e-5
    # If we exit the loop without returning, we can either raise an error or return the last sigma
    return sigma
# --------------------------
# 4. Load and Preprocess CSV
# --------------------------
def load_data(filepath):
    df = pd.read_csv(filepath, delimiter=';')
    df = df.fillna(0)
    return df

# --------------------------
# 5. Extract Data
# --------------------------

def get_sorted_volcanic_dfs_from_file(filepath: str) -> dict:
    # Read the CSV file from disk
    df = pd.read_csv(filepath, sep=";")
    df.fillna(0, inplace=True)  # Fill NaN values with 0
    
    # List of the specific volcanic products to filter and sort
    volcanic_products = [
        "VOLCANIC_ROCK_VOUCHER_9500",
        "VOLCANIC_ROCK_VOUCHER_9750",
        "VOLCANIC_ROCK_VOUCHER_10000",
        "VOLCANIC_ROCK_VOUCHER_10250",
        "VOLCANIC_ROCK_VOUCHER_10500",
        "VOLCANIC_ROCK"
    ]
    
    # Build a dictionary of DataFrames, filtered by product and sorted by mid_price
    volcanic_dfs = {
        product: df[df['product'] == product].sort_values(by='mid_price').reset_index(drop=True)
        for product in volcanic_products
    }
    
    return volcanic_dfs

# --------------------------
# 6. Compute Implied Vols
# --------------------------
def compute_implied_vols(S, df, r=0.0):
    # Filter underlying and voucher data
    voucher_df = df
    # Extract strike from product name
    voucher_df['strike'] = voucher_df['product'].str.extract(r'_(\d+)$').astype(float)
    voucher_df['underlying_price'] = S['mid_price']

    results = []
    T_max = 70000 
    i = 0
    for _, row in voucher_df.iterrows():
        S = row['underlying_price']
        K = row['strike']
        market_price = row['mid_price']
        T = T_max - i
        sigma = implied_vol_call_price(S, K, r, T, market_price)
        print(sigma)
        print(S, K, r, T, market_price)
        results.append({
            'timestamp': row['timestamp'],
            'product': row['product'],
            'strike': K,
            'market_price': market_price,
            'underlying_price': S,
            'T': T,
            'implied_vol': sigma
        })

        if i % 1000 == 0:
            print(f"Processed {i} rows...")
        i += 1

    return pd.DataFrame(results)
# --------------------------
# 7. Run Everything Per Voucher Type
# --------------------------
if __name__ == '__main__':
    filepath = 'sample_data/round-3-island-data-bottle/prices_round_3_day_0.csv'
    df_dict = get_sorted_volcanic_dfs_from_file(filepath)
    S = df_dict['VOLCANIC_ROCK']
    for product in df_dict.keys():
        if product != 'VOLCANIC_ROCK':
            print(f"Processing {product}...")
            voucher_df = df_dict[product]
            print(voucher_df.shape)
            iv_df = compute_implied_vols(S, voucher_df)
            plot_iv_vs_price_differences(iv_df, S)