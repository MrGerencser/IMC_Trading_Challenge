import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rbf_kernel(x1, x2, length_scale=1.0, sigma_f=1.0):
    sqdist = np.subtract.outer(x1, x2)**2
    return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)

def gp_predict(X_train, Y_train, X_test, length_scale=1.0, sigma_f=1.0, sigma_y=1e-4):
    K = rbf_kernel(X_train, X_train, length_scale, sigma_f) + sigma_y**2 * np.eye(len(X_train))
    K_s = rbf_kernel(X_train, X_test, length_scale, sigma_f)
    K_ss = rbf_kernel(X_test, X_test, length_scale, sigma_f) + 1e-8 * np.eye(len(X_test))

    K_inv = np.linalg.inv(K)

    # Predictive mean
    mu_s = K_s.T @ K_inv @ Y_train

    # Predictive variance
    cov_s = K_ss - K_s.T @ K_inv @ K_s
    std_s = np.sqrt(np.diag(cov_s))

    return mu_s, std_s

if __name__ == "__main__":
    # Load your price series
    df = pd.read_csv("sample_data/round-1-island-data-bottle/prices_round_1_day_0.csv", sep=";")
    df["product"] = df["product"].str.strip().str.upper()
    series = df[df["product"] == "SQUID_INK"]["mid_price"].dropna().reset_index(drop=True)

    # Use recent 200 points to train
    N_train = 10000
    X_train = np.arange(N_train)
    Y_train = series[:N_train].values

    # Predict next 50 steps
    X_test = np.arange(N_train, N_train + 100)

    mu, std = gp_predict(X_train, Y_train, X_test)

    # Plot results
    plt.figure(figsize=(12, 5))
    plt.plot(X_train, Y_train, 'b', label="Training Data")
    plt.plot(X_test, mu, 'r', label="GP Mean Forecast")
    plt.fill_between(X_test, mu - 1.96 * std, mu + 1.96 * std, color='pink', alpha=0.4, label="Confidence Interval")
    plt.title("Gaussian Process Forecast for Mid Price")
    plt.legend()
    plt.grid(True)
    plt.show()
