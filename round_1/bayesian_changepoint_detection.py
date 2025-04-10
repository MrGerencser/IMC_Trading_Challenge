import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def log_likelihood_normal(x, mean, std):
    # Use the log of the PDF of normal distribution
    n = len(x)
    if std == 0 or n == 0:
        return -np.inf
    ll = -0.5 * n * math.log(2 * math.pi) - n * math.log(std) - np.sum((x - mean)**2) / (2 * std**2)
    return ll

def detect_changepoints(series: pd.Series, min_size=100):
    series = series.dropna().values
    N = len(series)
    changepoints = []

    scores = []
    for t in range(min_size, N - min_size):
        segment1 = series[:t]
        segment2 = series[t:]

        mu1, std1 = np.mean(segment1), np.std(segment1)
        mu2, std2 = np.mean(segment2), np.std(segment2)
        mu_all, std_all = np.mean(series), np.std(series)

        ll1 = log_likelihood_normal(segment1, mu1, std1)
        ll2 = log_likelihood_normal(segment2, mu2, std2)
        ll_full = log_likelihood_normal(series, mu_all, std_all)

        # The higher the delta, the more likely it's a changepoint
        score = (ll1 + ll2) - ll_full
        scores.append(score)

    return scores


def get_strong_changepoints(scores, threshold_percentile=95, min_gap=50):
    threshold = np.percentile(scores, threshold_percentile)
    strong_cps = []
    for i, score in enumerate(scores):
        if score > threshold:
            if not strong_cps or (i - strong_cps[-1] > min_gap):
                strong_cps.append(i)
    return strong_cps

def plot_price_with_changepoints(price_series, changepoints, min_size):
    plt.figure(figsize=(12, 5))
    plt.plot(price_series.reset_index(drop=True), label="Mid Price")
    for cp in changepoints:
        plt.axvline(x=cp + min_size, color='red', linestyle='--', alpha=0.7, label='Changepoint' if cp == changepoints[0] else None)
    plt.title("Mid Price with Strong Changepoints")
    plt.xlabel("Timestep")
    plt.ylabel("Mid Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    df = pd.read_csv("sample_data/round-1-island-data-bottle/prices_round_1_day_0.csv", sep=";")
    df["product"] = df["product"].str.strip().str.upper()
    product_name = "SQUID_INK"

    if product_name in df["product"].unique():
        mid_prices = df[df["product"] == product_name]["mid_price"].dropna().reset_index(drop=True)
        returns = mid_prices.pct_change().dropna().reset_index(drop=True)

        min_size = 30
        scores = detect_changepoints(returns, min_size=min_size)
        changepoints = get_strong_changepoints(scores, threshold_percentile=98, min_gap=100)

        print("📌 Strong changepoints found at indices:", [cp + min_size for cp in changepoints])
        plot_price_with_changepoints(mid_prices, changepoints, min_size)
    else:
        print(f"Product '{product_name}' not found.")

